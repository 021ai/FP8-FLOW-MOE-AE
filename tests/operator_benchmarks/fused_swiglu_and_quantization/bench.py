# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import pytest
from typing import List

import transformer_engine
import transformer_engine_torch as tex
from megatron.core.fusions.fused_weighted_swiglu_quant import (
    fused_weighted_swiglu_quant,
    fused_weighted_swiglu_quant_back,
)
from megatron.core.fp8_utils import fp8_quantize
from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu, weighted_swiglu_back
from megatron.core.enums import Fp8Recipe

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def perf_test_cuda_kernel(cuda_kernel_fn):
    if torch.cuda.is_available():
        # create CUDA event
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # warmup
        for _ in range(50):
            cuda_kernel_fn()

        start_event.record()
        for _ in range(100):
            cuda_kernel_fn()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        return elapsed_time_ms / 100
    else:
        pytest.skip("CUDA is not available.")


# TE tensor dtypes
_te_dtypes: List[tex.DType] = [tex.DType.kBFloat16]


def _test_swiglu_quant(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    MOE_INTERMEDIATE_SIZE,
    BENCHMARK=False,
):
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    print(
        "mask map:"
        f" token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} {te_dtype}, MOE_INTERMEDIATE_SIZE:{MOE_INTERMEDIATE_SIZE}"
    )

    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == tex.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == tex.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == tex.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    input = torch.rand((num_out_tokens, 2*MOE_INTERMEDIATE_SIZE), dtype=dtype).cuda()
    weights = torch.rand((num_out_tokens, 1), dtype=dtype).cuda()
    grad_output = torch.rand((num_out_tokens, MOE_INTERMEDIATE_SIZE), dtype=dtype).cuda()

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    fp8_recipe = Fp8Recipe.blockwise
    if BENCHMARK:
        def weight_swiglu_quant():
            weighted_swiglu_result = weighted_swiglu(input, weights)
            fp8_quantize(fp8_recipe, weighted_swiglu_result)

        def fusion_weight_swiglu_quant():
            fused_weighted_swiglu_quant(input, weights)

        weight_swiglu_quant_time = perf_test_cuda_kernel(lambda: weight_swiglu_quant())
        fusion_weight_swiglu_quant_time = perf_test_cuda_kernel(lambda: fusion_weight_swiglu_quant())

        def weight_swiglu_quant_back():
            tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
            fp8_quantize(fp8_recipe, tmp)

        def fusion_weight_swiglu_quant_back():
            fused_weighted_swiglu_quant_back(grad_output, input, weights)

        weight_swiglu_quant_back_time = perf_test_cuda_kernel(lambda: weight_swiglu_quant_back())
        fusion_weight_swiglu_quant_back_time = perf_test_cuda_kernel(lambda: fusion_weight_swiglu_quant_back())

        fwd_naive_time = weight_swiglu_quant_time
        fwd_fused_time = fusion_weight_swiglu_quant_time
        bwd_naive_time = weight_swiglu_quant_back_time
        bwd_fused_time = fusion_weight_swiglu_quant_back_time

        fwd_speedup = fwd_naive_time / fwd_fused_time
        bwd_speedup = bwd_naive_time / bwd_fused_time

        print(f"fwd: naive: {fwd_naive_time:.3f} ms,  fused: {fwd_fused_time:.3f} ms, speedup: {fwd_speedup:.2f}")
        print(f"bwd: naive: {bwd_naive_time:.3f} ms,  fused: {bwd_fused_time:.3f} ms, speedup: {bwd_speedup:.2f}")

        return {
            "fwd_naive_ms": round(fwd_naive_time, 3),
            "fwd_fused_ms": round(fwd_fused_time, 3),
            "fwd_speedup": round(fwd_speedup, 2),
            "bwd_naive_ms": round(bwd_naive_time, 3),
            "bwd_fused_ms": round(bwd_fused_time, 3),
            "bwd_speedup": round(bwd_speedup, 2),
        }

@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_out_tokens", [None])
@pytest.mark.parametrize(
    "num_tokens, hidden_size, num_expert, topK, MOE_INTERMEDIATE_SIZE",
    [
        (4096, 1280, 64,  7, 1024),  #2b
        (4096, 2048, 64,  6, 1408), #16B
        (4096, 5120, 160, 6, 1536), # 200B
        (4096, 7168, 256, 8, 2048), # 600B
    ],
)
def test_swiglu_quant(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    MOE_INTERMEDIATE_SIZE,
):
    BENCHMARK = True

    _test_swiglu_quant(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        MOE_INTERMEDIATE_SIZE=MOE_INTERMEDIATE_SIZE,
        BENCHMARK=BENCHMARK,
    )


###################################################################################################################################
#
# Standalone benchmark: python bench.py  ->  perf_data.csv
#
###################################################################################################################################
if __name__ == "__main__":
    import csv
    import os

    te_dtype = tex.DType.kBFloat16

    configs = [
        # (num_tokens, hidden_size, num_expert, topK, MOE_INTERMEDIATE_SIZE)
        (4096, 1280, 64,  7, 1024),   # 2B
        (4096, 2048, 64,  6, 1408),   # 16B
        (4096, 5120, 160, 6, 1536),   # 200B
        (4096, 7168, 256, 8, 2048),   # 600B
    ]

    header = [
        "config",
        "fwd_naive_ms", "fwd_fused_ms", "fwd_speedup",
        "bwd_naive_ms", "bwd_fused_ms", "bwd_speedup",
    ]

    rows = []
    for num_tokens, hidden_size, num_expert, topK, moe_intermediate_size in configs:
        config_str = f"({num_tokens}, {hidden_size}, {num_expert}, {topK}, {moe_intermediate_size})"
        row = [config_str]

        result = _test_swiglu_quant(
            te_dtype=te_dtype,
            num_tokens=num_tokens,
            num_expert=num_expert,
            hidden_size=hidden_size,
            topK=topK,
            num_out_tokens=None,
            MOE_INTERMEDIATE_SIZE=moe_intermediate_size,
            BENCHMARK=True,
        )
        row.extend([
            result["fwd_naive_ms"],
            result["fwd_fused_ms"],
            result["fwd_speedup"],
            result["bwd_naive_ms"],
            result["bwd_fused_ms"],
            result["bwd_speedup"],
        ])

        rows.append(row)

    output_csv = os.path.join(os.path.dirname(__file__), "perf_data.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nResults saved to {output_csv}")
