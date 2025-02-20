import torch
from typing import List, Optional, Tuple
import numpy as np

def block_dequant_int4(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    x_z: torch.Tensor,
    block_size: List[int],
):
    """This function converts block-wise quantization to tensor-wise quantization.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The outputs are tensor-wise quantization tensor and tensor-wise quantization scale.
    Note only float8 is supported for now.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    x_dq_block_tiles = [
        [
            x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block_tiles[j][i][:, :] = (x_dq_block_tiles[j][i] - x_z[j*block_n:min((j + 1) * block_n, n), i].reshape(-1, 1)) * x_s[j*block_n:min((j + 1) * block_n, n), i].reshape(-1, 1)

    return x_dq_block.to(torch.float8_e4m3fn)

def input_to_int4(w_fp8, w_bit=4,
                   zero_point=True,
                   q_group_size=128,
                   inplace=False,
                   get_scale_zp=False
                           ):
    w = w_fp8.to(torch.float16)
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** w_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        pass

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        w_int4 = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int).reshape(org_w_shape)
        w_fp8 = ((w_int4 - zeros) * scales).to(torch.float8_e4m3fn)
        return w, scales.view(w.shape[0],), zeros.view(w.shape[0],), torch.clamp(torch.round(w / scales) + zeros, min_int, max_int).reshape(org_w_shape)
    else:
        return w

def block_quantization(x:torch.Tensor,
                       block_size: List[int],
                       kernel_block_k: int,
                       ):

    block_n, block_k = block_size[0], block_size[1]
    if len(x.shape) == 3:
        e, n, k = x.shape
        x_q_blocks = []
        x_dq_blocks = []
        x_s_blocks = []
        x_z_blocks = []
        for e_idx in range(e):
            n_tiles = (n + block_n - 1) // block_n
            k_tiles = (k + block_k - 1) // block_k
            x_dq_block = x[e_idx]

            x_s = torch.empty((n, k_tiles), dtype=torch.float16).to(x.device)
            x_s[:] = torch.finfo(torch.float16).min

            x_z = torch.zeros((n, k_tiles), dtype=torch.int32).to(x.device)

            x_dq_block_tiles = [
                [
                    x_dq_block[j*block_n:min((j+1) *block_n, n),
                            i * block_k : min((i + 1) * block_k, k),]
                    for i in range(k_tiles)
                ]
                for j in range(n_tiles)
            ]

            x_q_block = x[e_idx].clone().to(torch.int)
            x_q_block_tiles = [
                [
                    x_q_block[j * block_n:min((j + 1) * block_n, n),
                    i * block_k: min((i + 1) * block_k, k), ]
                    for i in range(k_tiles)
                ]
                for j in range(n_tiles)
            ]

            for i in range(k_tiles):
                for j in range(n_tiles):
                    x_dq_block_tiles[j][i][:, :], x_s_part, x_z_part, x_q_block_tiles[j][i][:, :] = input_to_int4(x_dq_block_tiles[j][i], q_group_size=block_k, get_scale_zp=True)
                    x_s[j*block_n:j*block_n + x_s_part.shape[0], i] = x_s_part
                    x_z[j*block_n:j*block_n + x_z_part.shape[0], i] = x_z_part

            intweight = x_q_block.t().contiguous().cpu()
            intweight = intweight.numpy().astype(np.uint8)

            row = 0
            qweight = np.zeros(((intweight.shape[0] + 1) // 2, intweight.shape[1]), dtype=np.uint8)
            while row < qweight.shape[0]:
                for j in range(0, 2):
                    idx = row + kernel_block_k * (row // kernel_block_k) + kernel_block_k * j
                    if idx < intweight.shape[0]:
                        qweight[row] |= intweight[idx] << (4 * j)
                row += 1
            qweight = torch.from_numpy(qweight).to(x.device).contiguous().t()

            zeros = x_z.t().cpu().numpy().astype(np.uint8)
            qzeros = np.zeros(((zeros.shape[0] + 1) // 2, zeros.shape[1]), dtype=np.uint8)
            row = 0
            idx_offset = 0
            while row < qzeros.shape[0]:
                for j in range(0, 2):
                    idx = row * 2 + j
                    if idx < zeros.shape[0]:
                        qzeros[row] |= zeros[idx] << (4 * j)
                row += 1

            qzeros = torch.from_numpy(qzeros).to(x.device).contiguous().t()
            x_q_blocks.append(qweight)
            x_dq_blocks.append(x_dq_block)
            x_s_blocks.append(x_s)
            x_z_blocks.append(qzeros)

        x_q = torch.stack(x_q_blocks, dim=0)
        x_dq = torch.stack(x_dq_blocks, dim=0).to(torch.float8_e4m3fn)
        x_s = torch.stack(x_s_blocks, dim=0)
        x_z = torch.stack(x_z_blocks, dim=0)
        return x_dq, x_s, x_z, x_q

