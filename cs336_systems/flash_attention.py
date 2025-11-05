import torch
import einops
import math
import triton.language as tl
import triton
import torch.nn.functional as F


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
        dtype = Q.dtype
        
        b_q = 16
        b_k = 16
        t_q = math.ceil(dim_seq / b_q)
        t_k = math.ceil(dim_seq / b_k)

        for b in range(dim_batch):
            for i in range(1, t_q + 1):
                Q_i = Q[b, (i - 1) * b_q: i * b_q, :]
                O_i = torch.zeros((b_q, dim_d), device=device, dtype=dtype)
                l_i = torch.zeros((b_q, 1), device=device, dtype=dtype)
                m_i = torch.ones((b_q, 1), device=device, dtype=dtype) * float('-inf')
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
        dim_seq_k = K.shape[-2]
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
        # print('triton launch grid: ', lauch_grid)
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
        ctx.save_for_backward(
            Q.reshape(dim_batch, dim_seq_q, dim_d), 
            K.reshape(dim_batch, dim_seq_k, dim_d),
            V.reshape(dim_batch, dim_seq_k, dim_d),
            O.reshape(dim_batch, dim_seq_q, dim_d), 
            L.reshape(dim_batch, dim_seq_q)
        )
        ctx.is_causal = is_causal
        ctx.scale = scale
        return einops.rearrange(O, '(b seq) d -> b seq d', b = dim_batch, seq = dim_seq_q)

    @staticmethod
    @torch.compile
    def backward(ctx, dO: torch.Tensor):
        Q, K, V, O, L = ctx.saved_tensors
        dim_seq_q = Q.shape[-2]
        dim_seq_k = K.shape[-2]
        device = Q.device
        # b, seq, dim = Q.shape
        scale = ctx.scale
        is_causal = ctx.is_causal
        D = torch.sum(O * dO, dim = -1, keepdim = True)
        S = einops.einsum(Q, K, 'b seq_q dim, b seq_k dim -> b seq_q seq_k') / scale
        if is_causal:
            # dim_seq_q, dim_seq_k
            ctx.mask = einops.rearrange(
                torch.arange(dim_seq_q).reshape((-1, 1)) >= torch.arange(dim_seq_k).reshape((1, -1)),
                '... -> 1 ...'
            ).to(device)
            # dim_batch, dim_seq_q, dim_seq_k
            S = torch.where(ctx.mask, S, float('-inf'))
        L = einops.rearrange(L, '... -> ... 1')
        P = torch.exp(S - L)
        dV = einops.einsum(P, dO, 'b seq_q seq_k, b seq_q dim_d -> b seq_k dim_d')
        dP = einops.einsum(dO, V, 'b seq_q dim_d, b seq_k dim_d -> b seq_q seq_k')
        dS = P * (dP - D)
        dQ = einops.einsum(dS, K, 'b seq_q seq_k, b seq_k dim_d -> b seq_q dim_d') / scale 
        dK = einops.einsum(dS, Q, 'b seq_q seq_k, b seq_q dim_d -> b seq_k dim_d') / scale
        return dQ, dK, dV, None

def _get_torch_dtype_from_str(dtype_str: str):
    if dtype_str == 'float32':
        return torch.float32
    elif dtype_str == 'float16':
        return torch.float16
    elif dtype_str == 'bfloat16':
        return torch.bfloat16
    else:
        raise ValueError('invalid dtype str:', dtype_str)

def _make_attn_inputs(batch_size, n_queries, n_keys, D, device=None, dtype = None):
    torch.random.manual_seed(0)
    q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True, dtype=dtype)
    k = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True,dtype=dtype)
    v = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True,dtype=dtype)
    do = torch.randn(batch_size, n_queries, D, device=device, dtype=dtype)
    return q, k, v, do

def benchmark():

    # D_list = (16, 32, 64, 128)
    D_list = [64, 128]
    # seq_list = (128, 256, 1024, 2048)
    seq_list = [1024, 2048]
    precision_list = ['float32']
    compile_torch_attn = torch.compile(F.scaled_dot_product_attention)
    func_list = {
        'fa_triton': FlashAttentionTriton.apply,
        'fa_torch': lambda q, k, v, is_causal: compile_torch_attn(q, k, v, is_causal=is_causal)
    }
    
    batch_size = 16
    warmup_sec = 5
    running_sec = 10
    res_list = []
    for D in D_list:
        for seq_len in seq_list:
            for precision in precision_list:
                for func_name, func in func_list.items(): 
                    q, k, v, do = _make_attn_inputs(batch_size, seq_len, seq_len, D, 'cuda', _get_torch_dtype_from_str(precision))
                    
                    forward_mean_ms = triton.testing.do_bench(
                        lambda : func(q, k, v, True),
                        warmup= warmup_sec * 1000,
                        rep = running_sec * 1000
                    )
                    
                    @torch.compile
                    def _backward_iter():
                        func(q, k, v, True).backward(do)
                    backward_and_forward_mean_ms = triton.testing.do_bench(
                        lambda : _backward_iter(),
                        warmup= warmup_sec * 1000,
                        rep = running_sec * 1000
                    )
                    res_line = {
                        'D': D,
                        'seq_len': seq_len,
                        'dtype': precision,
                        'func_name': func_name,
                        'forward_mean_ms': forward_mean_ms,
                        'backward_and_forward_mean_ms': backward_and_forward_mean_ms,
                    }
                    print(res_line)
                    res_list.append(res_line)
    
    import pandas as pd
    import datetime
    print(res_list)
    pd.DataFrame(res_list).to_excel(f'out_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx')

if __name__ == "__main__":
    benchmark()
    exit()
    batch_size, n_queries, n_keys, D = 4, 1024, 1024, 128
    q, k, v, do = _make_attn_inputs(batch_size, n_queries, n_keys, D, 'cuda')
    # f1 = FlashAttentionPytorch.apply
    # out1 = f1(q, k, v, True)
    # out1.backward(torch.ones(out1.shape).cuda()) # type: ignore
    
    f2 = FlashAttentionTriton.apply
    # out2 = f2(q, k, v, True)
    # out2.backward(torch.ones(out2.shape).cuda()) # type: ignore
    # forward
    out = triton.testing.do_bench(
        lambda : f2(q, k, v, True),
        warmup= 5 * 1000,
        rep = 15 * 1000
    )
    print(out)

    # f =  F.scaled_dot_product_attention
    # # out2 = f2(q, k, v, True)
    # # out2.backward(torch.ones(out2.shape).cuda()) # type: ignore
    # # forward
    # out = triton.testing.do_bench(
    #     lambda : f(q, k, v, is_causal=True),
    #     warmup= 5 * 1000,
    #     rep = 15 * 1000
    # )
    # print(out)

    # print(out1[0][0]) # type: ignore
    # print(out2[0][0]) # type: ignore

    # F.scaled_dot_product_attention()