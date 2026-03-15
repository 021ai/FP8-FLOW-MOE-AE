#!/usr/bin/env python3
"""
FP8-FLOW-MoE Operator Benchmarks: One-click runner and report generator.

Usage:
    # Run single-GPU benchmarks and generate report
    python run_all_benchmarks.py

    # Also run intra-node FP8 communication benchmark (requires 8 GPUs)
    python run_all_benchmarks.py --with-comm

    # Only generate report from existing results without re-running benchmarks
    python run_all_benchmarks.py --report-only

Output:
    - Per-benchmark perf_data.csv files in each sub-directory
    - Summary report printed to console and saved to validation_report.txt

Report strategy:
    The script collects all speedup values from each benchmark's output CSV,
    groups them by operation type, and compares against the expected
    speedup range.  The report presents the comparison clearly so that AE
    reviewers can evaluate the results.  No hard PASS/FAIL judgement is made;
    the final assessment is left to the reviewers.
"""

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------
# Each benchmark defines:
#   speedup_groups: dict mapping group_name -> {
#       "columns": list of CSV column names belonging to this group,
#       "expect_range": (min_speedup, max_speedup) expected range,
#       "description": short description for the report
#   }
#   expected_csv: path to expected output CSV (relative to subdir)

SINGLE_GPU_BENCHMARKS = [
    {
        "name": "Fused Permute and Padding",
        "section": "Section 4.4.1",
        "subdir": "fused_permute_and_padding",
        "run_cmd": [sys.executable, "bench.py"],
        "output_csv": "perf_data.csv",
        "expected_csv": "expected_output/perf_data_expected.csv",
        "speedup_groups": {
            "perm_pad": {
                "columns": ["bf16_perm_pad_speedup", "fp8_perm_pad_speedup"],
                "expect_range": (0.76, 1.70),
                "description": "permute+padding speedup (BF16 & FP8)",
            },
            "unperm_unpad": {
                "columns": ["bf16_unperm_unpad_speedup", "fp8_unperm_unpad_speedup"],
                "expect_range": (4.24, 7.33),
                "description": "unpermute+unpadding speedup (BF16 & FP8)",
            },
        },
    },
    {
        "name": "Fused SwiGLU and Quantization",
        "section": "Section 4.4.3",
        "subdir": "fused_swiglu_and_quantization",
        "run_cmd": [sys.executable, "bench.py"],
        "output_csv": "perf_data.csv",
        "expected_csv": "expected_output/perf_data_expected.csv",
        "speedup_groups": {
            "fwd": {
                "columns": ["fwd_speedup"],
                "expect_range": (1.69, 2.18),
                "description": "Forward fused SwiGLU+Quant speedup",
            },
            "bwd": {
                "columns": ["bwd_speedup"],
                "expect_range": (2.56, 4.01),
                "description": "Backward fused SwiGLU+Quant speedup",
            },
        },
    },
    {
        "name": "Scaling-aware FP8 Transpose",
        "section": "Section 4.4.4",
        "subdir": "scaling_aware_fp8_transpose",
        "run_cmd": [sys.executable, "bench.py"],
        "output_csv": "perf_data.csv",
        "expected_csv": "expected_output/perf_data_expected.csv",
        "speedup_groups": {
            "transpose": {
                "columns": ["speedup"],
                "expect_range": (2.3, 3.06),
                "description": "Scaling-aware FP8 transpose speedup",
            },
        },
    },
]

COMM_BENCHMARK_INTRANODE = {
    "name": "FP8 Communication (intra-node, EP=8)",
    "section": "Section 4.4.2",
    "subdir": "fp8_communication",
    "run_cmd": [sys.executable, "bench_intranode.py"],
    "output_csv": "perf_data_intranode_ep8.csv",
    "expected_csv": "expected_output/perf_data_intranode_ep8_expected.csv",
    "speedup_groups": {
        "comm": {
            "columns": ["speedup_comm"],
            "expect_range": (1.4, 1.65),
            "description": "FP8 communication-only speedup",
        },
        "overall": {
            "columns": ["speedup_all"],
            "expect_range": (1.0, 1.18),
            "description": "FP8 overall (comm+quant+dequant) speedup",
        },
    },
}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def read_csv_rows(csv_path: str) -> Tuple[List[str], List[dict]]:
    """Read CSV and return (header, list-of-row-dicts)."""
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def build_expected_lookup(csv_path: str, speedup_columns: List[str]) -> Dict[Tuple[str, str], float]:
    """Build a lookup dict: (config, column) -> expected speedup value."""
    lookup = {}
    if not os.path.exists(csv_path):
        return lookup
    _, rows = read_csv_rows(csv_path)
    for row_idx, row in enumerate(rows):
        config_str = row.get("config", f"row_{row_idx}")
        for col in speedup_columns:
            if col in row:
                try:
                    lookup[(config_str, col)] = float(row[col])
                except (ValueError, TypeError):
                    pass
    return lookup


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------
@dataclass
class ValueEntry:
    """A single speedup measurement."""
    config: str
    column: str
    actual: float
    expect: Optional[float]  # from expected CSV, None if not available


@dataclass
class GroupResult:
    """Collected results for one speedup group across all configs."""
    group_name: str
    description: str
    expect_range: Tuple[float, float]  # (min, max) expected range
    entries: List[ValueEntry]


@dataclass
class BenchmarkReport:
    name: str
    section: str
    run_success: bool
    run_error: str = ""
    group_results: List[GroupResult] = field(default_factory=list)


def collect_results(bench: dict) -> BenchmarkReport:
    """Collect speedup values from benchmark output CSV and expected CSV."""
    report = BenchmarkReport(
        name=bench["name"], section=bench["section"], run_success=True
    )

    subdir = os.path.join(SCRIPT_DIR, bench["subdir"])
    actual_path = os.path.join(subdir, bench["output_csv"])

    if not os.path.exists(actual_path):
        report.run_success = False
        report.run_error = f"Output CSV not found: {actual_path}"
        return report

    _, actual_rows = read_csv_rows(actual_path)

    if len(actual_rows) == 0:
        report.run_success = False
        report.run_error = "Output CSV has no data rows"
        return report

    # Collect all speedup columns across groups for expected lookup
    all_speedup_cols = []
    for group_def in bench["speedup_groups"].values():
        all_speedup_cols.extend(group_def["columns"])

    # Build expected lookup from expected CSV
    expected_path = os.path.join(subdir, bench.get("expected_csv", ""))
    expected_lookup = build_expected_lookup(expected_path, all_speedup_cols)

    for group_name, group_def in bench["speedup_groups"].items():
        columns = group_def["columns"]
        expect_range = group_def["expect_range"]
        description = group_def["description"]

        entries = []
        for row_idx, row in enumerate(actual_rows):
            config_str = row.get("config", f"row_{row_idx}")
            for col in columns:
                if col not in row:
                    continue
                actual_val = float(row[col])
                expect_val = expected_lookup.get((config_str, col))
                entries.append(ValueEntry(
                    config=config_str,
                    column=col,
                    actual=actual_val,
                    expect=expect_val,
                ))

        report.group_results.append(
            GroupResult(
                group_name=group_name,
                description=description,
                expect_range=expect_range,
                entries=entries,
            )
        )

    return report


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_benchmark(bench: dict) -> Tuple[bool, str]:
    """Run a single benchmark with real-time output. Returns (success, error_message)."""
    subdir = os.path.join(SCRIPT_DIR, bench["subdir"])
    cmd = bench["run_cmd"]
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working dir: {subdir}")
    print(flush=True)

    try:
        result = subprocess.run(
            cmd,
            cwd=subdir,
            timeout=1800,  # 30 min timeout per benchmark
        )
        if result.returncode != 0:
            return False, f"Exit code {result.returncode}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Benchmark timed out (30 min)"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def format_report(reports: List[BenchmarkReport]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("FP8-FLOW-MoE Operator Benchmarks — Results Report")
    lines.append("=" * 80)

    for report in reports:
        lines.append("")
        lines.append(f"  {report.name} ({report.section})")
        lines.append("-" * 80)

        if not report.run_success:
            lines.append(f"  ERROR: {report.run_error}")
            continue

        for g in report.group_results:
            actual_vals = [e.actual for e in g.entries]
            if actual_vals:
                actual_range_str = f"[{min(actual_vals):.2f}x, {max(actual_vals):.2f}x]"
            else:
                actual_range_str = "no data"

            expect_range_str = f"[{g.expect_range[0]:.2f}x, {g.expect_range[1]:.2f}x]"

            lines.append(
                f"  {g.description}"
                f"  |  Actual: {actual_range_str}"
                f"  Expect: {expect_range_str}"
            )
            lines.append("")

            # Table header
            lines.append(
                f"      {'Config':<35s}  {'Item':<30s}  {'Actual':>10s}  {'Expect':>10s}"
            )
            lines.append(
                f"      {'-'*35}  {'-'*30}  {'-'*10}  {'-'*10}"
            )

            # Table rows
            for e in g.entries:
                expect_str = f"{e.expect:.2f}x" if e.expect is not None else "N/A"
                lines.append(
                    f"      {e.config:<35s}  {e.column:<30s}  {e.actual:>9.2f}x  {expect_str:>10s}"
                )

            lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run all FP8-FLOW-MoE operator benchmarks and generate comparison report."
    )
    parser.add_argument(
        "--with-comm",
        action="store_true",
        help="Also run FP8 communication intra-node benchmark (requires 8 GPUs)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip running benchmarks; only generate report from existing perf_data.csv files",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=os.path.join(SCRIPT_DIR, "validation_report.txt"),
        help="Path to save the report (default: validation_report.txt)",
    )
    args = parser.parse_args()

    benchmarks = list(SINGLE_GPU_BENCHMARKS)
    if args.with_comm:
        benchmarks.append(COMM_BENCHMARK_INTRANODE)

    # --- Clean up previous output files ---
    # Always remove old report file
    if os.path.exists(args.report_file):
        os.remove(args.report_file)
        print(f"Removed old report: {args.report_file}")

    # Remove old perf_data.csv files (skip when --report-only)
    if not args.report_only:
        for bench in benchmarks:
            csv_path = os.path.join(SCRIPT_DIR, bench["subdir"], bench["output_csv"])
            if os.path.exists(csv_path):
                os.remove(csv_path)
                print(f"Removed old output: {csv_path}")

    reports: List[BenchmarkReport] = []

    print("")
    print("=" * 80)
    print("FP8-FLOW-MoE Operator Benchmarks")
    print(f"Benchmarks to run: {len(benchmarks)}")
    print("=" * 80)

    for i, bench in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] {bench['name']} ({bench['section']})")
        print("-" * 60)

        # --- Run ---
        if not args.report_only:
            print("  Running benchmark ...")
            success, error = run_benchmark(bench)
            if success:
                print("  Benchmark completed successfully.")
            else:
                print(f"  Benchmark FAILED: {error[:200]}")
                reports.append(
                    BenchmarkReport(
                        name=bench["name"],
                        section=bench["section"],
                        run_success=False,
                        run_error=error,
                    )
                )
                continue

        # --- Collect results ---
        print("  Collecting results ...")
        report = collect_results(bench)
        reports.append(report)

        if report.run_success:
            n_values = sum(len(g.entries) for g in report.group_results)
            print(f"  Collected {n_values} speedup values across {len(report.group_results)} groups")
        else:
            print(f"  ERROR: {report.run_error}")

    # --- Final report ---
    report_text = format_report(reports)
    print("\n")
    print(report_text)

    with open(args.report_file, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {args.report_file}")

    sys.exit(0)


if __name__ == "__main__":
    main()
