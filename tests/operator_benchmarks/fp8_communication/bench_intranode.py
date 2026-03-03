import os
import time
import csv

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import (
    init_dist,
    bench,
    calc_diff,
    inplace_unique,
    per_token_cast_to_fp8,
    per_token_cast_back,
)

import transformer_engine
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockQuantizer,
    Float8BlockwiseQTensor,
)

HAVE_OPENMIXOPL = True

if HAVE_OPENMIXOPL:
    try:
        from OpenMixOpl.triton import (
            act_quant_B_ptr as blockwise_quant,
            act_dequant_B_ptr as blockwise_dequant
        )
    except ImportError:
        print('use te blockwise quant')

else:
    def blockwise_quant(x):
        print(f'www')
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            amax_epsilon=0.0,
            force_pow_2_scales=True,
            block_scaling_dim=1,
            all_gather_usage=True,
        )
        quantized_tensor = quantizer(x)
        return (
            quantized_tensor._rowwise_data.view(torch.float8_e4m3fn),
            quantized_tensor._rowwise_scale_inv,
        )


    def blockwise_dequant(x, x_scale):
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            amax_epsilon=0.0,
            force_pow_2_scales=True,
            block_scaling_dim=1,
            all_gather_usage=True,
        )
        fp8_tensor = Float8BlockwiseQTensor(
            shape=x.shape,
            dtype=torch.bfloat16,
            rowwise_data=x.view(torch.uint8),
            rowwise_scale_inv=x_scale,
            columnwise_data=None,
            columnwise_scale_inv=None,
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=quantizer,
            is_2D_scaled=False,
            requires_grad=x.requires_grad,
            data_format=tex.Float8BlockScaleTensorFormat.COMPACT,
        )
        return fp8_tensor.dequantize()


def fused(x, buffer, dispatch_args):
    if isinstance(dispatch_args["x"], tuple):
        x_e4m3 = blockwise_quant(x)
        dispatch_args["x"] = x_e4m3
    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = (
        buffer.dispatch(**dispatch_args)
    )
    if isinstance(recv_x, tuple):
        blockwise_dequant(*recv_x)


def test_main(
    num_sms: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    output = []

    configs = [
        ("16B",  4096, 2048, 6, 64),
        ("200B", 4096, 5120, 6, 160),
        ("600B", 4096, 7168, 8, 256),
    ]

    for config in configs:
        model_size, num_tokens, hidden, num_topk, num_experts = config
        num_experts = (num_experts // num_ranks) * num_ranks
        bench_output = [(num_tokens * num_topk, hidden, num_ranks)]
        assert num_experts % num_ranks == 0

        # Random data
        x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
        x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

        x_e4m3 = blockwise_quant(x)
        quant_t = bench(lambda: blockwise_quant(x))[0]
        dequant_t = bench(lambda: blockwise_dequant(*x_e4m3))[0]

        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
        topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda") * rank
        topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda")
        rank_idx = topk_idx // (num_experts // num_ranks)
        rank_idx.masked_fill_(topk_idx == -1, -1)
        inplace_unique(rank_idx, num_ranks)

        # Expert meta
        num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
        for i in range(num_experts):
            num_tokens_per_expert[i] = (topk_idx == i).sum()
        gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
        dist.all_reduce(gbl_num_tokens_per_expert, group=group)

        # Rank layout meta
        num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
        token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device="cuda")
        for i in range(num_ranks):
            num_tokens_per_rank[i] = (rank_idx == i).sum()
            token_sel = (rank_idx == i).max(dim=-1)[0]
            count = token_sel.sum().item()
            tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
            tokens[:count] = torch.sort(tokens[:count])[0]
            token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device="cuda")
        token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
        is_token_in_rank = token_idx_in_rank >= 0
        gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
        dist.all_reduce(gbl_num_tokens_per_rank, group=group)

        ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = (
            buffer.get_dispatch_layout(topk_idx, num_experts)
        )
        assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
        assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
        assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
        t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
        group.barrier()
        time.sleep(1)

        # Config
        nvl_buffer_size = 256
        config = deep_ep.Config(num_sms, 8, nvl_buffer_size)

        # Test dispatch
        # noinspection PyShadowingNames
        def check_data(check_x, rank_prefix_matrix):
            assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
            check_start = 0
            for i in range(num_ranks):
                check_end = rank_prefix_matrix[i][rank].item()
                assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
                check_start = check_end

        for previous_mode in (False,):
            for async_mode in (False,):
                for current_x in (x, x_e4m3):
                    for with_topk in (True,):
                        dispatch_args = {
                            "x": current_x,
                            "num_tokens_per_rank": num_tokens_per_rank,
                            "is_token_in_rank": is_token_in_rank,
                            "num_tokens_per_expert": num_tokens_per_expert,
                            "config": config,
                            "async_finish": async_mode,
                        }
                        if with_topk:
                            dispatch_args.update({
                                "topk_idx": topk_idx,
                                "topk_weights": topk_weights_pure_rand if current_x is x_pure_rand else topk_weights,
                            })
                        if previous_mode:
                            dispatch_args.update({"previous_event": buffer.capture()})

                        t = bench(lambda: fused(x, buffer, dispatch_args))[0]
                        t_dis = bench(lambda: buffer.dispatch(**dispatch_args))[0]
                        if local_rank == 0:
                            if isinstance(current_x, tuple):
                                bench_output.append(float(f"{quant_t:.6f}"))
                                bench_output.append(float(f"{dequant_t:.6f}"))
                                bench_output.append(float(f"{t:.6f}"))
                            else:
                                bench_output.append(float(f"{t:.6f}"))

                        recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = (
                            buffer.dispatch(**dispatch_args)
                        )
                        event.current_stream_wait() if async_mode else ()
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                        # Checks
                        rank_prefix_matrix = handle[0]
                        assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), (
                            f"{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}"
                        )
                        assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                        if current_x is not x_pure_rand:
                            check_data(recv_x, rank_prefix_matrix)
                        if with_topk:
                            # Check topk_idx
                            assert (
                                recv_topk_idx.eq(-1)
                                | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))
                            ).sum().item() == recv_topk_idx.numel()
                            for i, count in enumerate(recv_num_tokens_per_expert_list):
                                assert recv_topk_idx.eq(i).sum().item() == count

                            # Check topk_weights
                            if current_x is not x_pure_rand:
                                recv_topk_weights[recv_topk_idx.eq(-1)] = (
                                    recv_topk_weights.amax(dim=1, keepdim=True)
                                    .expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                                )
                                check_data(recv_topk_weights, rank_prefix_matrix)

                        # Test cached dispatch (must without top-k staffs)
                        if not with_topk:
                            dispatch_args = {
                                "x": current_x,
                                "handle": handle,
                                "config": config,
                                "async_finish": async_mode,
                            }
                            if previous_mode:
                                dispatch_args.update({"previous_event": buffer.capture()})
                            recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                            event.current_stream_wait() if async_mode else ()
                            recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                            if current_x is not x_pure_rand:
                                check_data(recv_x, rank_prefix_matrix)

                        # Test combine
                        combine_args = {
                            "x": recv_x,
                            "handle": handle,
                            "config": config,
                            "async_finish": async_mode,
                        }
                        if with_topk:
                            combine_args.update({"topk_weights": recv_topk_weights})
                        if previous_mode:
                            combine_args.update({"previous_event": buffer.capture()})
                        combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                        event.current_stream_wait() if async_mode else ()
                        check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)
                        ref_x = x_pure_rand if current_x is x_pure_rand else x
                        assert calc_diff(check_x, ref_x) < 5e-6
                        if with_topk:
                            check_topk_weights = (
                                combined_topk_weights
                                if (current_x is x_pure_rand)
                                else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                            )
                            ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                            assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                        if local_rank == 0:
                            print(" passed", flush=True)

        output.append(bench_output)

    if local_rank == 0:
        header = [
            "config",
            "bf16_ms", "fp8_ms",
            "quant_ms", "dequant_ms", "fp8_comm_ms",
            "speedup_comm", "speedup_all",
        ]
        rows = []
        for (M, N, EP), BF16_ms, Quant_ms, Dequant_ms, FP8_ms in output:
            FP8_comm_ms = round(FP8_ms - (Quant_ms + Dequant_ms), 6)
            speedup_comm = round(BF16_ms / FP8_comm_ms, 2)
            speedup_all = round(BF16_ms / FP8_ms, 2)
            config_str = f"({M},{N},{EP})"
            rows.append([
                config_str,
                BF16_ms, FP8_ms,
                Quant_ms, Dequant_ms, FP8_comm_ms,
                speedup_comm, speedup_all,
            ])
            print(
                f"{config_str}  BF16: {BF16_ms}, FP8: {FP8_ms} "
                f"(quant: {Quant_ms}, dequant: {Dequant_ms}, comm: {FP8_comm_ms}), "
                f"speedup(comm): {speedup_comm}, speedup(all): {speedup_all}"
            )

        output_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"perf_data_intranode_ep{num_ranks}.csv",
        )
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nResults saved to {output_csv}")

# noinspection PyUnboundLocalVariable
def test_loop(local_rank: int, num_local_ranks: int):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    test_ll_compatibility, num_rdma_bytes = False, 0
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            ll_num_tokens, ll_hidden, num_ranks, ll_num_experts
        )

    buffer = deep_ep.Buffer(
        group, int(1e9), num_rdma_bytes,
        low_latency_mode=test_ll_compatibility,
        num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1),
    )
    torch.manual_seed(rank)

    for i in (12,):
        test_main(i, local_rank, num_ranks, rank, buffer, group)
        if local_rank == 0:
            print("", flush=True)


if __name__ == "__main__":
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)
