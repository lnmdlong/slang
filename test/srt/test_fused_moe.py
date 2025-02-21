import unittest

import torch
from vllm.model_executor.layers.fused_moe import fused_moe as fused_moe_vllm

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe

from quant_utils import (
    block_dequant_int4,
    input_to_int4,
    block_quantization
)
import numpy as np

class TestFusedMOE(unittest.TestCase):
    NUM_EXPERTS = [256]
    TOP_KS = [8]

    def torch_naive_moe(self, a, w1, w2, score, topk):
        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = SiluAndMul()(a[mask] @ w1[i].transpose(0, 1)) @ w2[
                    i
                ].transpose(0, 1)
        return (
            out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
        ).sum(dim=1)

    def _test_case(self, m, n, k, e, topk, dtype, quant_mode='default', block_scale=False):
        if quant_mode == 'w4a8':
            if not block_scale:
                # AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
                capability = torch.cuda.get_device_capability()
                if not (capability[0] >= 9 or capability == (8, 9)):
                    return

                a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
                w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
                w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
                w1 = w1.to(torch.float8_e4m3fn)
                w2 = w2.to(torch.float8_e4m3fn)
                score = torch.randn((m, e), device="cuda", dtype=dtype)

                w1_scale = torch.randn(e, dtype=torch.float32, device="cuda")
                w2_scale = torch.randn(e, dtype=torch.float32, device="cuda")
                a1_scale = torch.randn(1, dtype=torch.float32, device="cuda")
                a2_scale = torch.randn(1, dtype=torch.float32, device="cuda")

                w1_dq_block, w1_q4_scale, w1_q4_zero, w1_q4_block = block_quantization(w1.to(torch.float16), block_size=[2 * n, k], kernel_block_k=128)
                w2_dq_block, w2_q4_scale, w2_q4_zero, w2_q4_block = block_quantization(w2.to(torch.float16), block_size=[k, n], kernel_block_k=128)

                sglang_output = fused_moe(
                    a,
                    w1_q4_block,
                    w2_q4_block,
                    score,
                    topk,
                    renormalize=False,
                    use_int4_w4a8=True,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    w1_q4_scale=w1_q4_scale,
                    w1_q4_zero=w1_q4_zero,
                    w2_q4_scale=w2_q4_scale,
                    w2_q4_zero=w2_q4_zero
                )

                sglang_w8a8_output = fused_moe(
                    a,
                    w1_dq_block,
                    w2_dq_block,
                    score,
                    topk,
                    renormalize=False,
                    use_fp8_w8a8=True,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                )

                # torch.testing.assert_close(sglang_output, sglang_w8a8_output, atol=2e-2, rtol=0)
            else:
                # AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
                capability = torch.cuda.get_device_capability()
                if not (capability[0] >= 9 or capability == (8, 9)):
                    return

                block_shape = (128, 128)
                a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
                w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype)
                w2 = torch.randn((e, k, n), device="cuda", dtype=dtype)
                w1 = w1.to(torch.float8_e4m3fn)
                w2 = w2.to(torch.float8_e4m3fn)
                score = torch.randn((m, e), device="cuda", dtype=dtype)

                w1_scale = torch.randn((e, (2 * n + block_shape[0] - 1) // block_shape[0], (k + block_shape[0] - 1) // block_shape[1]), dtype=torch.float32, device="cuda")
                w2_scale = torch.randn((e, (k + block_shape[0] - 1) // block_shape[0], (n + block_shape[1] - 1) // block_shape[1]), dtype=torch.float32, device="cuda")
                a1_scale = torch.randn(1, dtype=torch.float32, device="cuda")
                a2_scale = torch.randn(1, dtype=torch.float32, device="cuda")

                w1_dq_block, w1_q4_scale, w1_q4_zero, w1_q4_block = block_quantization(w1.to(torch.float16), block_size=[block_shape[0], block_shape[1]], kernel_block_k=128)
                w2_dq_block, w2_q4_scale, w2_q4_zero, w2_q4_block = block_quantization(w2.to(torch.float16), block_size=[block_shape[0], block_shape[1]], kernel_block_k=128)

                sglang_output = fused_moe(
                    a,
                    w1_q4_block,
                    w2_q4_block,
                    score,
                    topk,
                    renormalize=False,
                    use_int4_w4a8=True,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    w1_q4_scale=w1_q4_scale,
                    w1_q4_zero=w1_q4_zero,
                    w2_q4_scale=w2_q4_scale,
                    w2_q4_zero=w2_q4_zero,
                    block_shape=block_shape,
                )

                sglang_w8a8_output = fused_moe(
                    a,
                    w1_dq_block,
                    w2_dq_block,
                    score,
                    topk,
                    renormalize=False,
                    use_fp8_w8a8=True,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    block_shape=block_shape,
                )

                # torch.testing.assert_close(sglang_output, sglang_w8a8_output, atol=2e-2, rtol=0)
                test_loops = 1000
                torch.cuda.cudart().cudaProfilerStart()
                for _ in range(test_loops):
                    sglang_output = fused_moe(
                        a,
                        w1_q4_block,
                        w2_q4_block,
                        score,
                        topk,
                        renormalize=False,
                        use_int4_w4a8=True,
                        w1_scale=w1_scale,
                        w2_scale=w2_scale,
                        a1_scale=a1_scale,
                        a2_scale=a2_scale,
                        w1_q4_scale=w1_q4_scale,
                        w1_q4_zero=w1_q4_zero,
                        w2_q4_scale=w2_q4_scale,
                        w2_q4_zero=w2_q4_zero
                    )
                for _ in range(test_loops):
                    sglang_w8a8_output = fused_moe(
                        a,
                        w1_dq_block,
                        w2_dq_block,
                        score,
                        topk,
                        renormalize=False,
                        use_fp8_w8a8=True,
                        w1_scale=w1_scale,
                        w2_scale=w2_scale,
                        a1_scale=a1_scale,
                        a2_scale=a2_scale,
                    )
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
        elif quant_mode == 'w8a8':
            # AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
            capability = torch.cuda.get_device_capability()
            if not (capability[0] >= 9 or capability == (8, 9)):
                return

            a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
            w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype)
            w2 = torch.randn((e, k, n), device="cuda", dtype=dtype)
            w1 = w1.to(torch.float8_e4m3fn)
            w2 = w2.to(torch.float8_e4m3fn)
            score = torch.randn((m, e), device="cuda", dtype=dtype)

            w1_scale = torch.randn(e, dtype=torch.float32, device="cuda")
            w2_scale = torch.randn(e, dtype=torch.float32, device="cuda")
            a1_scale = torch.randn(1, dtype=torch.float32, device="cuda")
            a2_scale = torch.randn(1, dtype=torch.float32, device="cuda")

            sglang_output = fused_moe(
                a,
                w1,
                w2,
                score,
                topk,
                renormalize=False,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
            )

            vllm_output = fused_moe_vllm(
                a,
                w1,
                w2,
                score,
                topk,
                renormalize=False,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
            )

            torch.testing.assert_close(sglang_output, vllm_output, atol=2e-2, rtol=0)

        else:
            a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
            w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
            w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
            score = torch.randn((m, e), device="cuda", dtype=dtype)

            triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)
            torch_output = self.torch_naive_moe(a, w1, w2, score, topk)
            torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)

    def test_various_configurations(self):
        # m_values = [1, 33, 64, 222, 1024 * 128]
        # n_values = [128, 1024, 2048]
        # k_values = [128, 511, 1024]
        # dtypes = [torch.float16, torch.bfloat16]
        # fp8_modes = [False, True]
        # quant_modes = ["w8a8"]
        # block_scale = [False]

        # for m in m_values:
        #     for n in n_values:
        #         for k in k_values:
        #             for e in self.NUM_EXPERTS:
        #                 for topk in self.TOP_KS:
        #                     for dtype in dtypes:
        #                         for quant_mode in quant_modes:
        #                             with self.subTest(
        #                                 m=m,
        #                                 n=n,
        #                                 k=k,
        #                                 e=e,
        #                                 topk=topk,
        #                                 dtype=dtype,
        #                                 quant_mode=quant_mode,
        #                                 block_scale=block_scale,
        #                             ):
        #                                 self._test_case(
        #                                     m,
        #                                     n,
        #                                     k,
        #                                     e,
        #                                     topk,
        #                                     dtype,
        #                                     quant_mode=quant_mode,
        #                                     block_scale=block_scale,
        #                                 )

        # w4a8 only supports k and n divisible by (block_k * 2) for now
        m_values = [32]
        n_values = [1024]
        k_values = [7168]
        dtypes = [torch.float16]
        quant_modes = ["w4a8"]
        block_scale = [True]

        for m in m_values:
            for n in n_values:
                for k in k_values:
                    for e in self.NUM_EXPERTS:
                        for topk in self.TOP_KS:
                            for dtype in dtypes:
                                for quant_mode in quant_modes:
                                    with self.subTest(
                                        m=m,
                                        n=n,
                                        k=k,
                                        e=e,
                                        topk=topk,
                                        dtype=dtype,
                                        quant_mode=quant_mode,
                                        block_scale=block_scale,
                                    ):
                                        self._test_case(
                                            m,
                                            n,
                                            k,
                                            e,
                                            topk,
                                            dtype,
                                            quant_mode=quant_mode,
                                            block_scale=block_scale,
                                        )


if __name__ == "__main__":
    unittest.main()
