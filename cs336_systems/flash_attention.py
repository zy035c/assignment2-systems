import math
import torch
import triton
import triton.language as tl


class FlashAttentionWoTrion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):

        dtype = Q.dtype
        device = Q.device
        N_QUERIES, D = Q.shape[-2], Q.shape[-1]
        N_KEYS, _ = K.shape[-2], K.shape[-1]

        Bq, Bk = 64, 64  # Tile sizes

        Tq = math.ceil(N_QUERIES / Bq)
        Tk = math.ceil(N_KEYS / Bk)

        other_dims = Q.shape[:-2]
        O = torch.zeros_like((*other_dims, N_QUERIES, D), device=device, dtype=dtype)
        L = torch.zeros_like((*other_dims, N_QUERIES), device=device, dtype=dtype)

        for i in range(Tq):  # loop for program instances
            Qi_upper_bound = (i + 1) * Bq if (i + 1) * Bq < N_QUERIES else N_QUERIES  # padding
            q_tile = Q[..., i * Bq: Qi_upper_bound, :]
            Oi = O[..., i * Bq: Qi_upper_bound, :]

            actual_Bq = q_tile.shape[-2]  # because of padding

            li = torch.zeros((*other_dims, actual_Bq,), device=device, dtype=dtype)
            mi = torch.full((*other_dims, actual_Bq,), fill_value=-torch.inf, device=device, dtype=dtype)

            for j in range(Tk):  # loop of a single program
                Kj_upper_bound = (j + 1) * Bk if (j + 1) * Bk < N_KEYS else N_KEYS  # padding
                k_tile = K[..., j * Bk: Kj_upper_bound, :]
                Vj = V[..., j * Bk: Kj_upper_bound, :]

                k_tile = k_tile.transpose(-1, -2)  # transpose for matmul
                Sij = q_tile @ k_tile / math.sqrt(D)  # [*, Bq, Bk]

                Sij_rowmax = torch.max(Sij, dim=-1, keepdim=False).values




@triton.jit
def flash_attention_forward_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,  #  = 1/sqrt(d)
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,  # Bq
    K_TILE_SIZE: tl.constexpr,  # Bk
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
    )

    Tq = triton.cdiv(N_QUERIES, Q_TILE_SIZE)
    Tk = triton.cdiv(N_KEYS, K_TILE_SIZE)

    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    for j in range(Tk):
        k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Compute attention scores
        

class FlashAttention2:

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        flash_attention_forward_kernel[()]()

    @staticmethod
    def backward(ctx, grad_o: torch.Tensor):
        pass
