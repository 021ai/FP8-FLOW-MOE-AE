
import random

import torch
import pytest
from typing import List

import transformer_engine
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockQuantizer,
    Float8BlockwiseQTensor,
)

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


def make_m_splits(total_rows, group_m, base, jitter_max=512, seed=42):
    """Generate m_splits with random jitter around a base value, summing to total_rows."""
    random.seed(seed)
    splits = []
    for i in range(group_m):
        jitter = random.randint(-jitter_max // 2, jitter_max // 2)
        val = max(1, base + jitter)
        splits.append(val)

    # Distribute remainder to match exact sum
    cur_sum = sum(splits)
    diff = total_rows - cur_sum
    i = 0
    while diff != 0:
        idx = i % group_m
        if diff > 0:
            splits[idx] += 1
            diff -= 1
        else:
            if splits[idx] > 1:
                splits[idx] -= 1
                diff += 1
        i += 1
    assert sum(splits) == total_rows
    return splits


def get_quantizer(all_gather_usage):
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        amax_epsilon=0.0,
        force_pow_2_scales=True,
        block_scaling_dim=1,
        all_gather_usage=all_gather_usage,
    )


# TE tensor dtypes
_te_dtypes: List[tex.DType] = [tex.DType.kBFloat16]


def _test_scaling_aware_fp8_transpose(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    ep,
    dp,
    BENCHMARK=False,
):
    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == tex.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == tex.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == tex.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    # Compute local dimensions under EP/DP parallelism
    local_num_expert = num_expert // ep
    total_tokens = num_tokens * topK
    local_total_tokens = (total_tokens // ep) * dp
    tokens_per_expert_base = local_total_tokens // local_num_expert

    inp = torch.rand((local_total_tokens, hidden_size), dtype=dtype).cuda()
    m_splits = make_m_splits(
        local_total_tokens, local_num_expert,
        base=tokens_per_expert_base, jitter_max=512,
        seed=100 + local_num_expert,
    )

    # print(
    #     f"token:{num_tokens} hidden_size:{hidden_size} expert:{local_num_expert} "
    #     f"topK:{topK} ep:{ep} dp:{dp} inp:{inp.shape} group_m:{len(m_splits)}"
    # )

    inp_fp8 = get_quantizer(all_gather_usage=True)(inp)
    inp_fp8._data_format = tex.Float8BlockScaleTensorFormat.GEMM_READY
    quantizer = get_quantizer(all_gather_usage=False)
    input_quantizers = [quantizer] * local_num_expert

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    if BENCHMARK:
        naive_inp_fp8 = inp_fp8.clone()
        scaling_aware_transpose_inp_fp8 = inp_fp8.clone()

        def double_quant():
            naive_inp = naive_inp_fp8.dequantize()
            tex.split_quantize(naive_inp, m_splits, input_quantizers)

        def scaling_aware_transpose():
            scaling_aware_transpose_inp_fp8.split_scaling_aware_fp8_transpose(m_splits, input_quantizers)

        naive_time = perf_test_cuda_kernel(lambda: double_quant())
        scaling_aware_transpose_time = perf_test_cuda_kernel(lambda: scaling_aware_transpose())

        speedup = naive_time / scaling_aware_transpose_time

        print(
            f"shape: ({num_tokens * topK}, {hidden_size}),  "
            f"naive: {naive_time:.3f} ms,  "
            f"scaling_aware_transpose: {scaling_aware_transpose_time:.3f} ms, "
            f"speedup: {speedup:.2f}"
        )

        return {
            "naive_ms": round(naive_time, 3),
            "scaling_aware_transpose_ms": round(scaling_aware_transpose_time, 3),
            "speedup": round(speedup, 2),
        }


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize(
    "num_tokens, num_expert, hidden_size, topK, ep, dp",
    [
        (4096, 64,  1280, 7, 4,  4),   # 2B
        (4096, 64,  2048, 6, 8,  4),   # 16B
        (4096, 160, 5120, 6, 8,  4),   # 200B
        (4096, 256, 7168, 8, 32, 6),   # 600B
        (4096, 384, 8192, 8, 64, 6),
        (4096, 512, 9216, 8, 64, 6),
    ],
)
def test_scaling_aware_fp8_transpose(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    ep,
    dp,
):
    BENCHMARK = True

    _test_scaling_aware_fp8_transpose(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        ep=ep,
        dp=dp,
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
        # (num_tokens, num_expert, hidden_size, topK, ep, dp)
        (4096, 64,  1280, 7, 4,  4),   # 2B
        (4096, 64,  2048, 6, 8,  4),   # 16B
        (4096, 160, 5120, 6, 8,  4),   # 200B
        (4096, 256, 7168, 8, 32, 6),   # 600B
        (4096, 384, 8192, 8, 64, 6),
        (4096, 512, 9216, 8, 64, 6),
    ]

    header = [
        "config",
        "naive_ms", "scaling_aware_transpose_ms", "speedup",
    ]

    rows = []
    for num_tokens, num_expert, hidden_size, topK, ep, dp in configs:
        result = _test_scaling_aware_fp8_transpose(
            te_dtype=te_dtype,
            num_tokens=num_tokens,
            num_expert=num_expert,
            hidden_size=hidden_size,
            topK=topK,
            ep=ep,
            dp=dp,
            BENCHMARK=True,
        )
        M = num_tokens * topK
        N = hidden_size
        config_str = f"({M},{N})"
        rows.append([
            config_str,
            result["naive_ms"],
            result["scaling_aware_transpose_ms"],
            result["speedup"],
        ])

    output_csv = os.path.join(os.path.dirname(__file__), "perf_data.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nResults saved to {output_csv}")
