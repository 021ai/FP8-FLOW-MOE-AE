# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import random

import torch
import pytest
from typing import Dict, List

from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    # moe_permute as te_permute,
    moe_permute_with_probs as te_permute_with_probs,
    moe_permute_and_pad_with_probs as te_permute_and_pad_with_probs,
    moe_unpermute as te_unpermute,
)
from transformer_engine.pytorch.utils import is_bf16_compatible
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch import Fp8Padding, Fp8Unpadding
import copy

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
# mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
    FP8GlobalStateManager.is_fp8_block_scaling_available()
)


def dtype_tols(te_dtype: tex.DType) -> Dict[str, float]:
    """Estimated tolerances for a datatype

    Based on tolerances for torch.testing.assert_close.

    """
    if te_dtype == tex.DType.kFloat32:
        return dict(rtol=1.0e-6, atol=1.0e-6)
    if te_dtype == tex.DType.kFloat16:
        return dict(rtol=3.0e-3, atol=1.0e-5)
    if te_dtype == tex.DType.kBFloat16:
        return dict(rtol=2.0e-2, atol=1.0e-5)
    if te_dtype == tex.DType.kFloat8E5M2 or te_dtype == tex.DType.kFloat8E4M3:
        return dict(rtol=2.0e-1, atol=1.0e-1)
    raise ValueError(f"Unsuppored dtype ({te_dtype})")


def backward_wrapper(
    act, backward_input, forward_input=[], retain_graph=True, accumulate_grad=False
):
    # Set forward_input.grad to None to avoid grad accumulation.
    if accumulate_grad == False:
        for i in forward_input:
            i.grad = None
    return act.backward(backward_input, retain_graph=retain_graph)


def _test_permutation_and_padding_mask_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    align_size=16,
    BENCHMARK=False,
    FP8=False,
):
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    print(
        "mask map:"
        f"te_dtype:{te_dtype} token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} FP8:{FP8}"
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

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = (
        torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()
    )

    probs = torch.rand(num_tokens, num_expert).cuda() * routing_map
    row_sums = probs.sum(dim=1, keepdim=True)
    probs = probs / row_sums
    probs = probs.to(dtype)
    probs.requires_grad_(True)

    tokens_per_expert = routing_map.sum(dim=0).cpu()
    target_tokens_per_expert = (
        torch.ceil(tokens_per_expert / align_size) * align_size
    ).long()
    num_permute_pad_out_tokens = target_tokens_per_expert.sum().item()

    permute_pad_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    permute_pad_bwd_input = torch.rand(
        (num_permute_pad_out_tokens, hidden_size), dtype=dtype
    ).cuda()
    unpermute_unpad_bwd_input = torch.rand(
        (num_tokens, hidden_size), dtype=dtype
    ).cuda()
    permute_pad_fwd_input.requires_grad_(True)

    restore_shape = permute_pad_fwd_input.shape
    ###################################################################################################################################
    #
    # moe_permute_with_probs and Fp8Padding, moe_unpermute and Fp8Unpadding
    #
    ###################################################################################################################################
    # permute + padding
    permuted_output, permuted_probs, row_id_map = te_permute_with_probs(
        permute_pad_fwd_input,
        probs,
        routing_map,
        num_out_tokens=num_out_tokens,
    )
    tokens_per_expert_list = tokens_per_expert.tolist()
    fp8_padding = Fp8Padding(num_expert, align_size)
    permuted_paded_output, _ = fp8_padding(permuted_output, tokens_per_expert_list)
    permuted_paded_probs, _ = fp8_padding(
        permuted_probs.unsqueeze(-1), tokens_per_expert_list
    )

    permuted_paded_output.backward(permute_pad_bwd_input, retain_graph=True)

    # unpadding + unpermute

    unpermute_unpad_fwd_input = permuted_paded_output.detach()
    unpermute_unpad_fwd_input.requires_grad_(True)

    fp8_unpadding = Fp8Unpadding(num_expert, align_size)
    unpaded_output = fp8_unpadding(unpermute_unpad_fwd_input, tokens_per_expert_list)
    unpermuted_unpaded_output = te_unpermute(
        unpaded_output, row_id_map, restore_shape=restore_shape
    )

    unpermuted_unpaded_output.backward(unpermute_unpad_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # fusion moe_permute_with_probs and Fp8Padding, fusion fusion moe_unpermute and Fp8Unpadding
    #
    ###################################################################################################################################
    # fusion permute_and_pad
    fusion_permute_and_pad_fwd_input = permute_pad_fwd_input.detach()
    fusion_permute_and_pad_fwd_input.requires_grad_(True)
    probs = probs.detach()
    probs.requires_grad_(True)

    (
        fusion_permuted_padded_output,
        fusion_permuted_padded_probs,
        row_id_map,
        pad_offsets,
        target_tokens_per_expert,
    ) = te_permute_and_pad_with_probs(
        fusion_permute_and_pad_fwd_input,
        probs,
        routing_map,
        tokens_per_expert,
        align_size,
    )
    fusion_permuted_padded_probs = fusion_permuted_padded_probs.unsqueeze(-1)

    fusion_permute_pad_bwd_input = permute_pad_bwd_input.detach()
    fusion_permuted_padded_output.backward(
        fusion_permute_pad_bwd_input, retain_graph=True
    )

    # fusion unpad and unpermute
    fusion_unpermute_unpad_fwd_input = fusion_permuted_padded_output.detach()
    fusion_unpermute_unpad_fwd_input.requires_grad_(True)

    fusion_unpermuted_unpaded_output = te_unpermute(
        fusion_unpermute_unpad_fwd_input,
        row_id_map,
        restore_shape=restore_shape,
        pad_offsets=pad_offsets,
    )

    fusion_unpermute_bwd_input = unpermute_unpad_bwd_input.detach()
    fusion_unpermuted_unpaded_output.backward(
        fusion_unpermute_bwd_input, retain_graph=True
    )

    ###################################################################################################################################
    #
    # Results Check
    #
    ###################################################################################################################################
    tols = dtype_tols(te_dtype)

    permuted_paded_output_ = permuted_paded_output.float()
    fusion_permuted_padded_output_ = fusion_permuted_padded_output.float()
    permute_pad_fwd_input_grad = permute_pad_fwd_input.grad.float()
    fusion_permute_and_pad_fwd_input_grad = (
        fusion_permute_and_pad_fwd_input.grad.float()
    )

    unpermuted_unpaded_output_ = unpermuted_unpaded_output.float()
    fusion_unpermuted_unpaded_output_ = fusion_unpermuted_unpaded_output.float()
    unpermute_unpad_fwd_input_grad = unpermute_unpad_fwd_input.grad.float()
    fusion_unpermute_unpad_fwd_input_grad = (
        fusion_unpermute_unpad_fwd_input.grad.float()
    )

    torch.testing.assert_close(
        permuted_paded_output_,
        fusion_permuted_padded_output_,
        msg=f"Mismatch in te_permute_and_pad fwd",
        **tols,
    )
    torch.testing.assert_close(
        permute_pad_fwd_input_grad,
        fusion_permute_and_pad_fwd_input_grad,
        msg=f"Mismatch in te_permute_and_pad bwd",
        **tols,
    )
    torch.testing.assert_close(
        unpermuted_unpaded_output_,
        fusion_unpermuted_unpaded_output_,
        msg=f"Mismatch in te_unpermute fwd",
        **tols,
    )
    torch.testing.assert_close(
        unpermute_unpad_fwd_input_grad,
        fusion_unpermute_unpad_fwd_input_grad,
        msg=f"Mismatch in te_unpermute bwd",
        **tols,
    )
    torch.testing.assert_close(
        permuted_paded_probs.float(),
        fusion_permuted_padded_probs.float(),
        msg=f"Mismatch in te_permute_and_pad bwd",
        **tols,
    )

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    if BENCHMARK:
        if FP8:
            quantizer = Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
                amax_epsilon=0.0,
                force_pow_2_scales=True,
                block_scaling_dim=1,
                all_gather_usage=True,
            )
            permute_pad_fwd_input = quantizer(permute_pad_fwd_input)
            fusion_permute_and_pad_fwd_input = quantizer(fusion_permute_and_pad_fwd_input)
                                                                                                                                                
        def permute_and_pad():
            permuted_output, permuted_probs, row_id_map = te_permute_with_probs(
                permute_pad_fwd_input,
                probs,
                routing_map,
                num_out_tokens=num_out_tokens,
            )

            fp8_padding(permuted_output, tokens_per_expert_list)
            fp8_padding(permuted_probs.unsqueeze(-1), tokens_per_expert_list)

        def fusion_permute_and_pad():
            (
                fusion_permuted_padded_output,
                fusion_permuted_padded_probs,
                row_id_map,
                pad_offsets,
                target_tokens_per_expert,
            ) = te_permute_and_pad_with_probs(
                fusion_permute_and_pad_fwd_input,
                probs,
                routing_map,
                tokens_per_expert,
                align_size,
            )
            fusion_permuted_padded_probs = fusion_permuted_padded_probs.unsqueeze(-1)

        permute_and_pad_naive_time = 0
        permute_and_pad_fused_time = 0
        unpermute_and_unpad_naive_time = 0
        unpermute_and_unpad_fused_time = 0

        t1 = perf_test_cuda_kernel(lambda: permute_and_pad())
        t2 = perf_test_cuda_kernel(lambda: fusion_permute_and_pad())
        permute_and_pad_naive_time += t1
        permute_and_pad_fused_time += t2
        # print(f"permute_and_pad\t\tfwd: naive: {t1:.3f} ms,  fusion: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                permuted_paded_output,
                permute_pad_bwd_input,
                forward_input=[permute_pad_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                fusion_permuted_padded_output,
                fusion_permute_pad_bwd_input,
                forward_input=[fusion_permute_and_pad_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        unpermute_and_unpad_naive_time += t1
        unpermute_and_unpad_fused_time += t2
        # print(f"permute_and_pad\t\tbwd: naive: {t1:.3f} ms,  fusion: {t2:.3f} ms")

        def unpad_unpermute():
            unpaded_output = fp8_unpadding(
                unpermute_unpad_fwd_input, tokens_per_expert_list
            )
            unpermuted_unpaded_output = te_unpermute(
                unpaded_output, row_id_map, restore_shape=restore_shape
            )

            unpermuted_unpaded_output.backward(
                unpermute_unpad_bwd_input, retain_graph=True
            )

        t1 = perf_test_cuda_kernel(lambda: unpad_unpermute())
        t2 = perf_test_cuda_kernel(
            lambda: te_unpermute(
                fusion_unpermute_unpad_fwd_input,
                row_id_map,
                restore_shape=restore_shape,
                pad_offsets=pad_offsets,
            )
        )
        unpermute_and_unpad_naive_time += t1
        unpermute_and_unpad_fused_time += t2
        # print(f"unpermute_and_unpad\tfwd: naive: {t1:.3f} ms,  fusion: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                unpermuted_unpaded_output,
                unpermute_unpad_bwd_input,
                forward_input=([unpermute_unpad_fwd_input, probs]),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                fusion_unpermuted_unpaded_output,
                fusion_unpermute_bwd_input,
                forward_input=([fusion_unpermute_unpad_fwd_input, probs]),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        permute_and_pad_naive_time += t1
        permute_and_pad_fused_time += t2
        # print(f"unpermute_and_unpad\tbwd: naive: {t1:.3f} ms,  fusion: {t2:.3f} ms")

        perm_pad_speedup = permute_and_pad_naive_time / permute_and_pad_fused_time
        unperm_unpad_speedup = unpermute_and_unpad_naive_time / unpermute_and_unpad_fused_time

        print(f"permute_and_pad\t\t: naive: {permute_and_pad_naive_time:.3f} ms,  fusion: {permute_and_pad_fused_time:.3f} ms, speedup: {perm_pad_speedup:.2f}")
        print(f"unpermute_and_unpad\t: naive: {unpermute_and_unpad_naive_time:.3f} ms,  fusion: {unpermute_and_unpad_fused_time:.3f} ms, speedup: {unperm_unpad_speedup:.2f}")

        return {
            "perm_pad_naive_ms": round(permute_and_pad_naive_time, 3),
            "perm_pad_fused_ms": round(permute_and_pad_fused_time, 3),
            "perm_pad_speedup": round(perm_pad_speedup, 2),
            "unperm_unpad_naive_ms": round(unpermute_and_unpad_naive_time, 3),
            "unperm_unpad_fused_ms": round(unpermute_and_unpad_fused_time, 3),
            "unperm_unpad_speedup": round(unperm_unpad_speedup, 2),
        }


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
_te_dtypes: List[tex.DType] = []
if is_bf16_compatible():
    _te_dtypes.append(tex.DType.kBFloat16)


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_out_tokens", [None])
@pytest.mark.parametrize(
    "num_tokens, num_expert, hidden_size, topK",
    [
        (4096, 64, 1280, 7),
        (4096, 64, 2048, 6),
        (4096, 160, 5120, 6),
        (4096, 256, 7168, 8),
        (4096, 384, 8192, 8),
        (4096, 512, 9216, 8),
    ],
)
@pytest.mark.parametrize("FP8", [False, True])
def test_permutation_and_padding_mask_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    FP8,
):
    BENCHMARK = True

    _test_permutation_and_padding_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        BENCHMARK=BENCHMARK,
        FP8=FP8,
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
        (4096, 64, 1280, 7),
        (4096, 64, 2048, 6),
        (4096, 160, 5120, 6),
        (4096, 256, 7168, 8),
        (4096, 384, 8192, 8),
        (4096, 512, 9216, 8),
    ]

    header = [
        "config",
        "bf16_perm_pad_naive_ms", "bf16_perm_pad_fused_ms", "bf16_perm_pad_speedup",
        "bf16_unperm_unpad_naive_ms", "bf16_unperm_unpad_fused_ms", "bf16_unperm_unpad_speedup",
        "fp8_perm_pad_naive_ms", "fp8_perm_pad_fused_ms", "fp8_perm_pad_speedup",
        "fp8_unperm_unpad_naive_ms", "fp8_unperm_unpad_fused_ms", "fp8_unperm_unpad_speedup",
    ]

    rows = []
    for num_tokens, num_expert, hidden_size, topK in configs:
        config_str = f"({num_tokens}, {num_expert}, {hidden_size}, {topK})"
        row = [config_str]

        for fp8 in [False, True]:
            result = _test_permutation_and_padding_mask_map(
                te_dtype=te_dtype,
                num_tokens=num_tokens,
                num_expert=num_expert,
                hidden_size=hidden_size,
                topK=topK,
                num_out_tokens=None,
                BENCHMARK=True,
                FP8=fp8,
            )
            row.extend([
                result["perm_pad_naive_ms"],
                result["perm_pad_fused_ms"],
                result["perm_pad_speedup"],
                result["unperm_unpad_naive_ms"],
                result["unperm_unpad_fused_ms"],
                result["unperm_unpad_speedup"],
            ])

        rows.append(row)

    output_csv = os.path.join(os.path.dirname(__file__), "perf_data.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nResults saved to {output_csv}")
