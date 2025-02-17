import unittest

import torch
from typing import List, Optional, Tuple

from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
)

from sglang.srt.layers.quantization.fp4_kernel import (
    w4a8_block_fp8_matmul,
)
import logging
import numpy as np

class TestFP4Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = 64
        cls.N = 1536
        cls.K = 7168
        cls.group_size = 128
        cls.quant_type = torch.float8_e4m3fn
        cls.output_type = torch.float16

    @staticmethod
    def _make_A(M, K, group_size, out_dtype):
        quant_A = torch.rand(
            M, K // group_size, group_size, dtype=torch.float32, device="cuda"
        )
        # -1 ~ 1
        quant_A = quant_A * 2 - 1
        # scaling abs max to fmax
        finfo = torch.finfo(out_dtype)
        fmax = finfo.max
        scaling = fmax / quant_A.abs().amax(-1, keepdim=True)
        quant_A *= scaling
        quant_A = quant_A.to(out_dtype).to(torch.float32)

        # create scale and A
        scale = torch.rand(M, K // group_size, dtype=torch.float32, device="cuda")
        scale /= fmax
        A = quant_A * scale[..., None]

        A = A.reshape(M, K)
        quant_A = quant_A.reshape(M, K).to(out_dtype)
        return A, quant_A, scale

    @staticmethod
    def _make_B(K, N, group_size, out_dtype):
        def _aligned_size(a, b):
            return (a + b - 1) // b * b

        K_aligned = _aligned_size(K, group_size)
        N_aligned = _aligned_size(N, group_size)

        quant_B = torch.rand(
            K_aligned // group_size,
            group_size,
            N_aligned // group_size,
            group_size,
            dtype=torch.float32,
            device="cuda",
        )
        quant_B = quant_B * 2 - 1

        # scaling abs max to fmax
        finfo = torch.finfo(out_dtype)
        fmax = finfo.max
        scaling = fmax / quant_B.abs().amax((1, 3), keepdim=True)
        quant_B *= scaling
        quant_B = quant_B.to(out_dtype).to(torch.float32)

        scale = torch.rand(
            K_aligned // group_size,
            1,
            N_aligned // group_size,
            1,
            dtype=torch.float32,
            device="cuda",
        )
        scale /= fmax

        B = quant_B * scale

        B = B.reshape(K_aligned, N_aligned)[:K, :N]
        quant_B = quant_B.reshape(K_aligned, N_aligned).to(out_dtype)[:K, :N]
        scale = scale.reshape(K_aligned // group_size, N_aligned // group_size)
        return B, quant_B, scale

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

def input_to_int4(w, w_bit=4,
                   zero_point=True,
                   q_group_size=128,
                   inplace=False,
                   get_scale_zp=False
                           ):
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
        return w, scales.view(w.shape[0],), zeros.view(w.shape[0],), torch.clamp(torch.round(w / scales) + zeros, min_int, max_int).reshape(org_w_shape)
    else:
        return w

def block_quantization(x:torch.Tensor,
                       block_size: List[int],
                       ):

    block_n, block_k = block_size[0], block_size[1]
    n, k = x.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    x_dq_block = x.clone()

    x_s = torch.empty((n, k_tiles), dtype=torch.float32).to(x.device)
    x_s[:] = torch.finfo(torch.float32).min

    x_z = torch.zeros((n, k_tiles), dtype=torch.int32).to(x.device)

    x_dq_block_tiles = [
        [
            x_dq_block[j*block_n:min((j+1) *block_n, n),
                      i * block_k : min((i + 1) * block_k, k),]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]

    x_q_block = x.clone()
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

    return x_dq_block, x_s, x_z, x_q_block

class TestW4A8BlockFP8Matmul(TestFP4Base):
    def test_w4a8_block_fp8_matmul(self):
        if torch.cuda.get_device_capability()[0] < 9:
            return
        A, A_quant_gt, A_scale_gt = self._make_A(
            M=self.M, K=self.K, group_size=self.group_size, out_dtype=self.quant_type
        )
        B, B_quant_gt, B_scale_gt = self._make_B(
            K=self.K, N=self.N, group_size=self.group_size, out_dtype=self.quant_type
        )
        C_gt = A.to(self.output_type) @ B.to(self.output_type)
        x_dq_block, x_s, x_z, x_q_block = block_quantization(B_quant_gt.t().to(torch.float32), block_size=[128, 128])
        x_s = x_s.to(torch.float16)

        intweight = x_q_block.t().contiguous().cpu()
        intweight = intweight.numpy().astype(np.uint32)

        row = 0
        qweight = np.zeros((intweight.shape[0] // 2, intweight.shape[1]), dtype=np.uint8)
        while row < qweight.shape[0]:
            for j in range(0, 2):
                idx = row + 128 * (row // 128) + 128 * j
                if idx < intweight.shape[0]:
                    qweight[row] |= intweight[idx] << (4 * j)
            row += 1

        qweight = torch.from_numpy(qweight).contiguous().t()

        zeros = x_z.t().cpu().numpy().astype(np.uint8)
        qzeros = np.zeros((zeros.shape[0] * 4 // 8, zeros.shape[1]), dtype=np.uint8)
        row = 0
        idx_offset = 0
        while row < qzeros.shape[0]:
            for j in range(0, 2):
                idx = row * 2 + j
                if idx < zeros.shape[0]:
                    qzeros[row] |= zeros[idx] << (4 * j)
            row += 1

        qzeros = torch.from_numpy(qzeros).contiguous().t()
        C = w4a8_block_fp8_matmul(
            A=A_quant_gt,
            B=qweight.cuda().contiguous(),
            As=A_scale_gt,
            Bs=B_scale_gt.cuda().T.contiguous(),
            Bqs=x_s,
            Bqz=qzeros.cuda().contiguous(),
            block_size=[128, 128],
            output_dtype=self.output_type,
        )

        test_dq = block_dequant_int4(x_q_block, x_s, x_z, block_size=[128, 128])
        C_fake_fp8_gt = w8a8_block_fp8_matmul(
            A=A_quant_gt,
            B=test_dq.cuda().contiguous(),
            As=A_scale_gt,
            Bs=B_scale_gt.cuda().T.contiguous(),
            block_size=[128, 128],
            output_dtype=self.output_type,
        )
        torch.testing.assert_close(C, C_fake_fp8_gt, atol=0.5, rtol=1e-4)

        # benchmark
        test_loops = 1000
        import time
        start = time.time()
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(test_loops):
            C_fake_fp8_gt = w8a8_block_fp8_matmul(
                A=A_quant_gt,
                B=test_dq.cuda().contiguous(),
                As=A_scale_gt,
                Bs=B_scale_gt.cuda().T.contiguous(),
                block_size=[128, 128],
                output_dtype=self.output_type,
            )
        for _ in range(test_loops):
            C = w4a8_block_fp8_matmul(
                A=A_quant_gt,
                B=qweight.cuda().contiguous(),
                As=A_scale_gt,
                Bs=B_scale_gt.cuda().T.contiguous(),
                Bqs=x_s,
                Bqz=qzeros.cuda().contiguous(),
                block_size=[128, 128],
                output_dtype=self.output_type,
            )
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print(f'w4a8_block_fp8_matmul kernel time: {(time.time() - start) * 1000 / test_loops} ms')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # setup_seed(0)
    unittest.main()

