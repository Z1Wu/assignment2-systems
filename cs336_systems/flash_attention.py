import torch
import einops
import math
import triton.language as tl
import triton


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        device = Q.device
        # Q: [b, n_seq, n_d]
        # K: [b, n_seq, n_d]
        # V: [b, n_seq, n_d]

        dim_d = Q.shape[-1]
        sqrt_dim_d = math.sqrt(dim_d)
        
        # Q = einops.rearrange(Q, '..., seq, d -> (...), seq, d')
        # K = einops.rearrange(Q, '..., seq, d -> (...), seq, d')
        # V = einops.rearrange(Q, '..., seq, d -> (...), seq, d')
        dim_batch = Q.shape[0]
        dim_seq = Q.shape[-2]
        O = torch.empty_like(Q, device=device)
        L = torch.empty((dim_batch, dim_seq, 1), device=device)
        
        b_q = 4
        b_k = 4
        t_q = math.ceil(dim_seq / b_q)
        t_k = math.ceil(dim_seq / b_k)

        for b in range(dim_batch):
            for i in range(1, t_q + 1):
                Q_i = Q[b, (i - 1) * b_q: i * b_q, :]
                O_i = torch.zeros((b_q, dim_d), device=device)
                l_i = torch.zeros((b_q, 1), device=device)
                m_i = torch.ones((b_q, 1), device=device) * float('-inf')
                for j in range(1, t_k + 1):
                    K_j = K[b, (j - 1) * b_k: j * b_k, :] # [b_k, d]
                    V_j = V[b, (j - 1) * b_k: j * b_k, :] # [b_k, d]
                    S_i = einops.einsum(Q_i, K_j, 'b_q d, b_k d -> b_q b_k') / sqrt_dim_d
                    # 计算此处的 mask 
                    if is_causal:
                        ul_i = (i - 1) * b_q
                        ul_j = (j - 1) * b_k
                        col_idx = torch.arange(ul_j, ul_j + b_k).view(1, b_k)
                        row_idx = torch.arange(ul_i, ul_i + b_q).view(b_q, 1)
                        mask = row_idx <= col_idx
                        S_i[mask] = float('-inf')
                    
                    m_i_1 = torch.max(torch.cat([m_i, S_i], dim = 1), dim=1, keepdim=True).values
                    P_i = torch.exp(S_i - m_i_1) # broacast here
                    l_i = torch.exp(m_i - m_i_1) * l_i + torch.sum(P_i, dim=1, keepdim=True)
                    # shape
                    O_i = torch.exp(m_i - m_i_1) * O_i + einops.einsum(P_i, V_j, 'b_q b_k, b_k d -> b_q d')
                    m_i = m_i_1
                O_i = O_i / l_i
                L_i = m_i + torch.log(l_i)
                O[b, (i - 1) * b_q: i * b_q, :] = O_i
                L[b, (i - 1) * b_q: i * b_q, :] = L_i
        ctx.save_for_backward(L.reshape(dim_batch, dim_seq))
        return O

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplemented

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(0),
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1),padding_option="zero") # [Q_TILE_SIZE, D]
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype = tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE, 1), dtype = tl.float32)
    m_i = tl.zeros((Q_TILE_SIZE, 1), dtype = tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # [K_TILE_SIZE, D]
        V_j = tl.load(V_block_ptr, boundary_check=(0,),padding_option="zero") # [K_TILE_SIZE, D]
        S_i = tl.dot(Q_i, tl.dot(K_j)) / scale
        m_i_1 = tl.max(tl.cat(m_i, S_i), axis = 1, keep_dims=True)
        P_i = tl.exp(S_i - m_i_1)
        l_i = tl.exp(m_i - m_i_1) * l_i + tl.sum(P_i, dim=1, keep_dims=True)
        O_i = tl.exp(m_i - m_i_1) * O_i + tl.dot(P_i.to(V_j.dtype), V_j)
        m_i = m_i_1
        K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l_i
    L_i = m_i + tl.log(l_i)
    tl.store(
        O_block_ptr,
        O_i.to(O_block_ptr.type.element_ty),
        boundary_check=(0,1),
    )

    tl.store(
        L_block_ptr,
        L_i.to(O_block_ptr.type.element_ty),
        boundary_check=(0,1),
    )


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        dim_d = Q.shape[-1]
        scale = math.sqrt(dim_d)
        dim_batch = Q.shape[0]
        dim_seq_q = Q.shape[-2]
        dim_seq_k = Q.shape[-2]
        device = Q.device

        # [b, seq, d]
        Q = einops.rearrange(Q, '... d -> (...) d')
        K = einops.rearrange(K, '... d -> (...) d')
        V = einops.rearrange(V, '... d -> (...) d')
        
        n_query = Q.shape[0]
        n_keys = K.shape[0]
        O = torch.empty_like(Q, device=device)
        L = torch.empty((n_query, 1), device=device)

        K_TILE_SIZE = 4
        Q_TILE_SIZE = 4

        flash_fwd_kernel[(triton.cdiv(Q_TILE_SIZE, dim_batch),)](
            Q, K, V,
            O, L,
            dim_seq_q * dim_d, dim_d, 1, # Q
            dim_seq_k * dim_d, dim_d, 1, # K
            dim_seq_k * dim_d, dim_d, 1, # V
            dim_seq_q * dim_d, dim_d, 1, # O
            dim_seq_q, 1, # L
            N_QUERIES = n_query, N_KEYS = n_keys,
            scale = scale,
            D = dim_d, # type: ignore
            Q_TILE_SIZE = Q_TILE_SIZE, # type: ignore
            K_TILE_SIZE = K_TILE_SIZE, # type: ignore
        )
        return einops.rearrange(O, '(...) d -> b seq d', b = dim_batch, seq = dim_seq_q)



if __name__ == "__main__":
    def _make_attn_inputs(device=None):
        torch.random.manual_seed(0)
        batch_size = 1
        n_queries = 8
        n_keys = 8
        D = 4
        q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
        k = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
        v = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
        do = torch.randn(batch_size, n_queries, D, device=device)

        return q, k, v, do
    
    q, k, v, do = _make_attn_inputs('cuda')
    f = FlashAttentionPytorch.apply
    out = f(q, k, v, False)
    print(out)