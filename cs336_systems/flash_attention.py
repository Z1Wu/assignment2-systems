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
        
        b_q = 16
        b_k = 16
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
                        mask = row_idx < col_idx
                        S_i[mask] -= 1e6
                    
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
        # Q,K,V => [b, seq, d]
        # L => [b, seq]
        ctx.save_for_backward(Q, K, V, O, L.reshape(dim_batch, dim_seq))
        ctx.sqrt_d = sqrt_dim_d
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        Q, K, V, O, L = ctx.saved_tensors
        # b, seq, dim = Q.shape
        sqrt_dim_d = ctx.sqrt_d
        D = torch.sum(O * dO, dim = -1, keepdim = True)
        S = einops.einsum(Q, K, 'b seq_q dim, b seq_k dim -> b seq_q seq_k') / sqrt_dim_d
        L = einops.rearrange(L, '... -> ... 1')
        P = torch.exp(S - L)
        dV = einops.einsum(P, dO, 'b seq_q seq_k, b seq_q dim_d -> b seq_k dim_d')
        dP = einops.einsum(dO, V, 'b seq_q dim_d, b seq_k dim_d -> b seq_q seq_k')
        dS = P * (dP - D)
        dQ = einops.einsum(dS, K, 'b seq_q seq_k, b seq_k dim_d -> b seq_q dim_d') / sqrt_dim_d 
        dK = einops.einsum(dS, Q, 'b seq_q seq_k, b seq_q dim_d -> b seq_k dim_d') / sqrt_dim_d
        return dQ, dK, dV, None

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
    is_causal: tl.constexpr
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
        order=(1, 0)
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1),padding_option="zero") # [Q_TILE_SIZE, D]
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype = tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE, 1), dtype = tl.float32)
    m_i = tl.full((Q_TILE_SIZE, 1), float('-inf'), dtype = tl.float32)
    for j in range(tl.cdiv(stride_kb // D, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # [K_TILE_SIZE, D]
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1),padding_option="zero") # [K_TILE_SIZE, D]
        S_i = tl.dot(Q_i, tl.trans(K_j)) / scale
            
        m_i_1 = tl.maximum(
            m_i, # [Q_TILE_SIZE, 1]
            tl.max(S_i, axis=1, keep_dims=True) # [Q_TILE_SIZE, 1]
        )
        if is_causal:
            col_offs = j * K_TILE_SIZE
            row_offs = query_tile_index * Q_TILE_SIZE
            col_idx = (tl.arange(0, K_TILE_SIZE) + col_offs).view(1, K_TILE_SIZE)
            row_idx = (tl.arange(0, Q_TILE_SIZE) + row_offs).view(Q_TILE_SIZE, 1)
            mask = row_idx < col_idx
            S_i -= mask * (1e6)
        P_i = tl.exp(S_i - m_i_1)
        l_i = tl.exp(m_i - m_i_1) * l_i + tl.sum(P_i, axis=1, keep_dims=True)
        O_i = tl.exp(m_i - m_i_1) * O_i + tl.dot(P_i.to(V_j.dtype), V_j)
        m_i = m_i_1
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # tl.device_print('b_i: ', batch_index)
    # tl.device_print(f'l_i: shape[1]', int(l_i.shape[0]))
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

        K_TILE_SIZE = 16
        Q_TILE_SIZE = 16
        lauch_grid = (triton.cdiv(dim_seq_q, Q_TILE_SIZE), dim_batch)
        print('triton launch grid: ', lauch_grid)
        flash_fwd_kernel[lauch_grid](
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
            is_causal = is_causal # type: ignore
        )
        ctx.save_for_backward(L.reshape(dim_batch, dim_seq_q))
        ctx.is_causal = is_causal
        return einops.rearrange(O, '(b seq) d -> b seq d', b = dim_batch, seq = dim_seq_q)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplemented


# @triton.jit
# def kernel():
#     # t = tl.zeros((2, 1), dtype=tl.float32)
#     # tl.device_print('test', t)
#     # import pdb
#     # pdb.set_trace()
#     tl.device_print('test', tl.program_id(1))

if __name__ == "__main__":
    # kernel[(4,)]()
    # exit()

    def _make_attn_inputs(device=None):
        torch.random.manual_seed(0)
        batch_size = 2
        n_queries = 32
        n_keys = 32
        D = 16
        q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
        k = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
        v = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
        do = torch.randn(batch_size, n_queries, D, device=device)
        return q, k, v, do
    
    q, k, v, do = _make_attn_inputs('cuda')
    f1 = FlashAttentionPytorch.apply
    out1 = f1(q, k, v, True)
    out1.backward(torch.ones(out1.shape).cuda()) # type: ignore

    # f2 = FlashAttentionTriton.apply
    # out2 = f2(q, k, v, True)

    # print(out1[0][0]) # type: ignore
    # print(out2[0][0]) # type: ignore