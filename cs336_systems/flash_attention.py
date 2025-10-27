import torch
import einops
import math

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