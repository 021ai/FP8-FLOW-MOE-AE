[ [Back to index](https://cTuning.org/ae) ]

# Artifact Checklist for FP8-FLOW-MoE

## Abstract

This artifact accompanies the paper **FP8-FLOW-MOE: A CASTING-FREE FP8 RECIPE WITHOUT DOUBLE QUANTIZATION ERROR**. 

The artifact includes:

1. **Modified TransformerEngine** — with fused FP8-aware operators (permute-and-padding, scaling-aware FP8 transpose, and FP8 handling logic inside `grouped_linear`).
2. **Modified Megatron-LM** — integrating the FP8-FLOW-MoE training pipeline and fused FP8-aware operators (permute-and-padding, SwiGLU+quantization, FP8 communication), plus fine-grained recomputation for `moe_expert`.
3. **FP8-FLOW-MoE test suite** — benchmark scripts and expected-result checks for reproducing the paper’s experiments.

**Minimum hardware:** NVIDIA Hopper GPUs (FP8-capable, High HBM recommended). Operator micro-benchmarks require 1 GPU; FP8 communication micro-benchmarks require 1–4 nodes (8 GPUs/node); large-scale end-to-end efficiency/ablation/convergence experiments are designed for up to 32 nodes (256 GPUs), depending on the specific configuration.

**Key results to reproduce:**
- Convergence validation showing no loss degradation under FP8 training (Section 4.1).
- End-to-end training efficiency improvements on DeepSeek-V3/V2 and Qwen3-235B models (Section 4.2).
- End-to-end component ablations on DeepSeek-V3 (671B) under the same setup as Section 4.2 with full activation checkpointing (Section 4.3).
- Operator-level speedups for fused permute-and-padding, fused SwiGLU+quantization, scaling-aware FP8 transpose, and FP8 communication (Section 4.4).

---

## Checklist

* **Algorithm:** Yes. FP8-FLOW-MoE is an end-to-end FP8 mixed-precision training framework for MoE models, including fused operators and FP8 communication strategies.
* **Program:** Modified TransformerEngine, modified Megatron-LM, and custom benchmark scripts (Python + shell). All benchmark scripts are included in this artifact.
* **Compilation:** TransformerEngine must be compiled from source (requires CUDA toolkit).
* **Model:** DeepSeek-V3/V2 and Qwen3-235B model configurations are used for end-to-end evaluation. Model weights are initialized/continued from publicly released checkpoints.
* **Data set:** End-to-end experiments and convergence validation use publicly available datasets. Operator micro-benchmarks use synthetic data (no external datasets required).
* **Run-time environment:** Linux (tested on Ubuntu 24.04.2 LTS). Requires CUDA 12.9+, PyTorch, Triton 3.2.0, DeepEP 1.0.0.5. No root access required for running experiments.
* **Hardware:** NVIDIA Hopper GPUs. Intra-node: NVLink; inter-node: RoCE.
  - Operator micro-benchmarks: 1 GPU.
  - FP8 communication micro-benchmarks: 1–4 nodes (8 GPUs/node).
  - End-to-end efficiency/ablation experiments: designed for multi-node (up to 32 nodes/256 GPUs depending on EP/PP settings).
* **Run-time state:** Micro-benchmarks include warm-up iterations before timing to reduce cold-cache effects. Results may vary due to GPU thermal state and system load.
* **Execution:** Sole-user access during benchmarking is recommended. No special process pinning required. Operator benchmarks complete in minutes; end-to-end experiments can take hours per configuration at scale.
* **Metrics:** Kernel execution time (ms), speedup ratio (fused vs baseline), training throughput (TGS: tokens/GPU/s), peak GPU memory (GB/GPU), convergence loss curves.
* **Output:**  
  - Operator micro-benchmarks output CSV files (`perf_data*.csv`), console logs, and optional PDF plots; expected-result checks are included (e.g., aggregated reports).  
  - End-to-end experiments output Megatron training logs; throughput is read from logs; peak memory is measured externally.
* **Experiments:** We provide container support and step-by-step reproduction instructions in the repository `git@github.com:021ai/FP8-FLOW-MOE-AE.git` (see `README.md` and `tests/` subdirectories).
* **How much disk space required (approximately)?:** ~10 GB for source code + build artifacts (Megatron-LM + TransformerEngine + tests). End-to-end experiments require additional disk space for datasets and checkpoints (varies by model/dataset).
* **How much time is needed to prepare workflow (approximately)?:** ~2 hour for cloning repos and compiling/installing TransformerEngine (excluding large dataset/checkpoint downloads).
* **How much time is needed to complete experiments (approximately)?:** Operator benchmarks ~1 hour total; end-to-end experiments are several hours per configuration (and depend on cluster scale).
* **Publicly available?:** Yes. Source code is publicly available on GitHub.
* **Code licenses (if publicly available)?:** 
* **Workflow frameworks used?** No special workflow framework; standard Python scripts and shell scripts.
* **Archived?:** 

---

## Description

### How to access

The artifact consists of 3 repositories.

**Option 1 (Recommended): Clone our pre-integrated repositories**

> **Strongly recommended:** please clone our *pre-integrated* repositories directly. They include our integrated end-to-end test launchers and the component-ablation implementations validated in our environment, which is the most reliable way to reproduce the paper’s results.

```bash
# Megatron-LM with FP8-FLOW-MoE patches
git@github.com:021ai/Megatron-LM-FP8FlowMoe.git

# TransformerEngine with FP8-FLOW-MoE patches
git@github.com:021ai/TransformerEngine-FP8FlowMoe.git

# Test scripts and benchmarks
git@github.com:021ai/FP8-FLOW-MOE-AE.git
```

**Option 2: Apply upstream PRs to official repositories**

- **Megatron-LM (official repo):** https://github.com/NVIDIA/Megatron-LM  
  - Base commit: `4af25fe`
  - Our PRs to apply:
    - https://github.com/NVIDIA/Megatron-LM/pull/2763
    - https://github.com/NVIDIA/Megatron-LM/pull/2764

- **TransformerEngine (official repo):** https://github.com/NVIDIA/TransformerEngine  
  - Base commit: release_v2.8 `40c69e7`
  - Our PRs to apply:
    - https://github.com/NVIDIA/TransformerEngine/pull/1921
    - https://github.com/NVIDIA/TransformerEngine/pull/2547
    - https://github.com/NVIDIA/TransformerEngine/pull/2144

- **FP8-FLOW-MoE test scripts (our repo):**
  ```bash
  git clone git@github.com:021ai/FP8-FLOW-MOE-AE.git
  ```

Approximate disk space after cloning: ~5 GB (excluding build artifacts and checkpoints/datasets).

---

### Hardware dependencies

- **GPU:** NVIDIA Hopper (FP8-capable, High HBM recommended)
- **Intra-node interconnect:** NVLink
- **Inter-node interconnect:** RoCE, Mellanox ConnectX (mlx5)

**Minimum requirements by experiment:**

| Experiment | GPUs Required |
|---|---|
| Convergence validation (Section 4.1) | **Multi-GPU** (exact scale depends on your launcher; designed to be runnable at smaller scale than full 32-node runs) |
| End-to-end efficiency evaluation (Section 4.2) | Multi-node at scale (up to 32 nodes / 256 GPUs depending on EP/PP settings) |
| End-to-end component ablations (Section 4.3) | Same setup as Section 4.2 |
| Operator micro-benchmarks: fused kernels (Section 4.4.1/4.4.3/4.4.4) | 1 GPU |
| FP8 communication micro-benchmarks (Section 4.4.2) | EP8: 1 node (8 GPUs); EP16: 2 nodes (16 GPUs); EP32: 4 nodes (32 GPUs) |

---

### Software dependencies

| Software | Version (tested) | Notes |
|---|---:|---|
| OS | Ubuntu 24.04.2 LTS | Other modern Linux distros should also work |
| CUDA Toolkit | 12.9.41 | Tested version |
| Python | 3.12 | Python >= 3.10 should also work |
| PyTorch | 2.7.0a0+79aa17489c.nv25.04 | Tested with NVIDIA NGC nightly build |
| Triton | 3.2.0 | Typically installed alongside PyTorch |
| DeepEP | 1.0.0.5+6519814 | https://github.com/deepseek-ai/DeepEP.git (1.2.1+b57e5e2 also tested) |
| NCCL | CUDA 12.x compatible | For multi-GPU / multi-node comm |
| GCC/G++ | 13.3.0 | Tested version |

---

### Data sets

- **Operator micro-benchmarks:** No external datasets required (synthetic inputs).
- **Convergence validation (Section 4.1), end-to-end efficiency (Section 4.2), ablations (Section 4.3):** Datasets are **not bundled** with this artifact.

**How to download / prepare datasets**
- Please follow the **latest dataset download & preprocessing instructions** in the public README of this artifact repository:
  ```text
  https://github.com/021ai/FP8-FLOW-MOE-AE
---

### Models

- **End-to-end efficiency (Section 4.2) and ablations (Section 4.3):** Model configurations follow DeepSeek-V2/V3 and Qwen3-235B architectures.
- Model weights are **not bundled** with this artifact. We continue from **publicly released checkpoints** to ensure comparable initialization across methods.

**How to download / prepare model weights**
- Please follow the **latest checkpoint download instructions** in the public README of this artifact repository:
  ```text
  https://github.com/021ai/FP8-FLOW-MOE-AE
---

## Installation

Assuming the working directory is `/mnt/data/FP8-FLOW-MOE-AE`:

```bash
# 1. Create and enter working directory
mkdir -p /mnt/data/FP8-FLOW-MOE-AE
cd /mnt/data/FP8-FLOW-MOE-AE

# 2. Clone repositories
git clone  -b release_v1.0 --recursive git@github.com:021ai/Megatron-LM-FP8FlowMoe.git
git clone  -b release_v1.0 --recursive git@github.com:021ai/TransformerEngine-FP8FlowMoe.git
git clone git@github.com:021ai/FP8-FLOW-MOE-AE.git

# 3. Install TransformerEngine from source
pip uninstall -y transformer_engine  # remove any existing installation
cd TransformerEngine-FP8FlowMoe
export NVTE_FRAMEWORK=pytorch
pip3 install --no-build-isolation .
cd ..

# 4. Install Megatron-LM
cd Megatron-LM-FP8FlowMoe
pip install -e .
cd ..

```

---

## Experiment workflow

This artifact provides four groups of experiments under `FP8-FLOW-MOE-AE/tests/`, each corresponding to one experimental section in the paper:

```text
tests/
├── convergence_validation/       # Section 4.1: Convergence validation (loss via TensorBoard)
├── e2e_efficiency_evaluation/    # Section 4.2: End-to-end efficiency (TGS & peak memory)
├── e2e_component_ablations/      # Section 4.3: End-to-end ablations (TGS & peak memory)
└── operator_benchmarks/          # Section 4.4: Operator-level micro-benchmarks
    ├── fused_permute_and_padding/
    ├── fp8_communication/
    ├── fused_swiglu_and_quantization/
    └── scaling_aware_fp8_transpose/
```

### Common workflow stages (applies to all experiments)

1. **Environment preparation**
   - Install/compile dependencies (TransformerEngine-FP8FlowMoe, Megatron-LM-FP8FlowMoe, DeepEP).
   - Ensure CUDA/NCCL/NVSHMEM are correctly configured for your cluster.
2. **Experiment execution**
   - Run the corresponding launcher script (`.sh`) or benchmark script (`bench.py`, `bench_intranode.py`, ...).
3. **Result collection**
   - Operator micro-benchmarks generate `perf_data*.csv` and an aggregated report (e.g., `validation_report.txt`).
   - End-to-end runs produce Megatron training logs under `${OUTPUT_DIR}` (configured in each launcher).
   - Convergence validation writes TensorBoard event files under the run output directory (see Group A below).
4. **(Optional) Plotting for paper-ready figures**
   - Operator benchmarks may include plotting scripts (e.g., `plot_perf.py`) to generate PDF figures.

> **Important (path customization):** End-to-end launchers contain **cluster-specific absolute paths** for datasets/tokenizers/checkpoints.  
> For AE evaluation, reviewers must replace these paths with their local equivalents (or parameterize them via environment variables).  
> The required inputs are listed explicitly in **Evaluation and expected result**.

### Experiment group A: Convergence validation (Section 4.1)

- **Goal:** validate that FP8-FLOW-MoE does not degrade convergence compared to BF16.
- **Entry script:** `tests/convergence_validation/run_convergence.sh <bf16|fp8flowmoe>`
- **Interface:**
  ```bash
  cd FP8-FLOW-MOE-AE/tests/convergence_validation
  bash run_convergence.sh <bf16|fp8flowmoe>
  ```
- **Outputs:** logs are written under `${OUTPUT_DIR}` (default: `FP8-FLOW-MOE-AE/output/`).
  Loss curves should be inspected via TensorBoard event files under the run’s log directory, e.g.:
  - `${OUTPUT_DIR}/<run_name>/tensorboard/events.out.tfevents.*`

### Experiment group B: End-to-end efficiency evaluation (Section 4.2)

- **Goal:** reproduce throughput (TGS: tokens/GPU/s) and peak memory across precision modes and EP/AC settings.
- **Entry script:** `tests/e2e_efficiency_evaluation/run_dlc_e2e.sh`
- **Interface:**
  ```bash
  cd FP8-FLOW-MOE-AE/tests/e2e_efficiency_evaluation
  bash run_dlc_e2e.sh <MODEL> <PRECISION> <EP> <AC>
  ```
- **Outputs:** training logs under `${OUTPUT_DIR}` or console; expected reference values in `tests/e2e_efficiency_evaluation/expected_output/expected_results.csv`.
- **Metrics:** TGS from logs (use stable iters 2–6); peak memory measured externally.

### Experiment group C: End-to-end component ablations (Section 4.3)

- **Goal:** reproduce component ablations on **DeepSeek-V3 (671B)** under the **same setup as Group B** with **AC=full**.
- **Entry script:** `tests/e2e_component_ablations/run_dlc_e2e.sh`
- **Interface:**
  ```bash
  cd FP8-FLOW-MOE-AE/tests/e2e_component_ablations
  bash run_dlc_e2e.sh <perm_pad|double_quant> <ep8|ep16|ep32>
  ```
- **Outputs:** training logs under `${OUTPUT_DIR}`; expected reference values in `tests/e2e_component_ablations/expected_output/expected_results.csv`.
- **Metrics:** **same as Group B**.

### Experiment group D: Operator-level micro-benchmarks (Section 4.4)

- **Entry points:**
  - Single-GPU benchmarks: `operator_benchmarks/*/bench.py`
  - Intra-node communication: `operator_benchmarks/fp8_communication/bench_intranode.py`
  - Inter-node communication: `operator_benchmarks/fp8_communication/bench_internode.py`
  - One-click runner & reporter: `operator_benchmarks/run_all_benchmarks.py`
- **Outputs:**
  - Per-benchmark `perf_data*.csv`
  - Aggregated report `operator_benchmarks/validation_report.txt`

---

## Evaluation and expected result

This section is a **reproduction/validation checklist**: for each key claim, we provide (1) concrete steps to run, (2) where to find outputs, and (3) success criteria (including acceptable variation).

> **Feasibility note:** Large-scale end-to-end experiments (Sections 4.2–4.3) require substantial Hopper GPU cluster resources.
> If reviewers do not have such a cluster, they can still fully evaluate Section 4.4 (operator micro-benchmarks) and partially evaluate convergence depending on their available scale.

### Claim 1: Convergence validation (Section 4.1)

#### A. Run
```bash
cd FP8-FLOW-MOE-AE/tests/convergence_validation
bash run_convergence.sh <bf16|fp8flowmoe>
```

#### B. Inputs that must be provided
The script contains hard-coded paths that must be edited:
- `TOKENIZER_PATH`
- `DATASET_FILE`

#### C. Where to check loss curves
Loss is logged via TensorBoard under the run output directory:
- `${OUTPUT_DIR}/<run_name>/tensorboard/events.out.tfevents.*`

To visualize:
```bash
tensorboard --logdir ${OUTPUT_DIR}/<run_name>/tensorboard --bind_all --port 6006
```

#### D. Pass criteria / acceptable variation
A successful reproduction should show:
- Stable convergence (no divergence / NaNs), and
- The **relative loss error remains consistently below 0.25%** (FP8-FLOW-MoE vs BF16) over the compared steps.

---

### Claim 2: End-to-end training efficiency improvements (Section 4.2)

#### A. Inputs that must be provided (before running)
The launcher `tests/e2e_efficiency_evaluation/run_dlc_e2e.sh` contains cluster-specific absolute paths that must be edited:
- `CPT_PRETRAIN_CHECKPOINT_PATH`
- `TOKENIZER_PATH`
- `DATASET_FILE`

#### B. Run
```bash
cd FP8-FLOW-MOE-AE/tests/e2e_efficiency_evaluation
bash run_dlc_e2e.sh <MODEL> <PRECISION> <EP> <AC>
```

#### C. How to read the metrics (TGS & peak memory)

**TGS (tokens/GPU/s):** Each run trains for 6 iterations. Use the **best TGS (= minimum elapsed time)** among iterations 2–6.

Example log line:
```
iteration 4/6 | elapsed time per iteration (ms): 113993.1 | token throughput per GPU (tokens/s/GPU): 1149.8 | ...
```

**Peak memory (GB/GPU):** Not reported in training logs. Measure peak `memory.used` per GPU via your cluster's GPU monitoring system (e.g., `nvidia-smi`, DCGM, or equivalent) during training.

#### D. Expected results & acceptable variation

The full set of expected values (Tables 1–6 from the paper) is provided in paper or reference file:

```
tests/e2e_efficiency_evaluation/expected_output/expected_results.csv
```

-- Expected qualitative trends match the paper (FP8-FLOW-MoE ≥ BF16 throughput; larger gains at higher EP; selective recomputation enables higher EP without OOM).
-- Acceptable variation: **±15%** deviation in absolute TGS is within normal range due to hardware and network differences across clusters.

---

### Claim 3: Component ablations (Section 4.3)

#### A. Run
```bash
cd FP8-FLOW-MOE-AE/tests/e2e_component_ablations
bash run_dlc_e2e.sh <perm_pad|double_quant> <ep8|ep16|ep32>
```

#### B. Metrics and judgement
Use **the same method as Claim 2** to read TGS and peak memory.  
Acceptable variation is **the same as Claim 2**.

#### C. Expected results

All ablations run on **DeepSeek-V3 (671B) with AC=full**. The full expected values (Table 7 from the paper) are provided as a reference file:

```
tests/e2e_component_ablations/expected_output/expected_results.csv
```

**Key qualitative claims to verify:**
- **`perm_pad`**: Disabling fused permute-and-padding reduces throughput by up to **~5%** and increases peak memory by approximately **1 GB** compared to the non-ablated FP8-FLOW-MoE.
- **`double_quant`**: Replacing scaling-aware FP8 transpose with DoubleQuant reduces throughput by up to **~18%** and increases peak memory by approximately **1 GB** depending on EP.

Acceptable variation: same **±15%** as Claim 2.

---

### Claim 4: Operator-level speedups (Section 4.4)

#### A. Run

**Option 1: one-click (recommended)**
```bash
cd FP8-FLOW-MOE-AE/tests/operator_benchmarks

python run_all_benchmarks.py                # single-GPU benchmarks only
python run_all_benchmarks.py --with-comm   # also run FP8 comm benchmark (requires 8 GPUs)
```

**Option 2: run individual sub-experiments**

Each sub-experiment can also be run independently. After running, use `--report-only` to generate the aggregated report.

```bash
cd FP8-FLOW-MOE-AE/tests/operator_benchmarks

# Section 4.4.1 — Fused Permute and Padding (1 GPU)
cd fused_permute_and_padding && python bench.py && cd ..

# Section 4.4.2 — FP8 Communication, intra-node (8 GPUs)
cd fp8_communication && python bench_intranode.py && cd ..

# Section 4.4.3 — Fused SwiGLU and Quantization (1 GPU)
cd fused_swiglu_and_quantization && python bench.py && cd ..

# Section 4.4.4 — Scaling-aware FP8 Transpose (1 GPU)
cd scaling_aware_fp8_transpose && python bench.py && cd ..

# Generate aggregated report from existing results(without comm results)
python run_all_benchmarks.py --report-only
```

#### B. Expected outputs
- `tests/operator_benchmarks/*/perf_data*.csv`
- `tests/operator_benchmarks/validation_report.txt` (also printed to console)

#### C. Pass criteria / acceptable variation
- Fused operators and FP8 communication should be faster than baselines (speedup > 1) across most tested configurations.
- Acceptable variation: **±15%** deviation in speedup is acceptable as long as the qualitative conclusion holds.

---

## Experiment customization

> **Skip this section if the main experiments in Sections 4.1–4.3 can be reproduced successfully on your cluster.**
> The customization options below are provided for reviewers who lack sufficient cluster resources to run the full paper configurations.

### Single-node functional smoke-test (DeepSeek-V2-Lite, 16B)

If your evaluation environment cannot provision the multi-node cluster required for the paper's main experiments (DeepSeek-V3 671B / Qwen3-235B / DeepSeek-V2 236B with EP8–EP32), the script `tests/e2e_efficiency_evaluation/run_dsw_DeepSeek-V2-Lite.sh` provides a lightweight alternative that runs on **a single node (8 GPUs)**.

| Property | Value |
|---|---|
| Model | DeepSeek-V2-Lite (16B) |
| Hardware required | 1 node, 8 × GPUs |
| Parallelism | PP=2, EP=4 |
| Training iterations | 6 (configurable via `TRAIN_ITERS`) |
| Purpose | Functional correctness — verifying that all four precision modes launch and complete without errors |

**Usage:**
```bash
cd FP8-FLOW-MOE-AE
bash tests/e2e_efficiency_evaluation/run_dsw_DeepSeek-V2-Lite.sh <CASE>
# CASE: bf16 | blockwise | tensorwise | fp8flowmoe
```

**Important caveats:**
- The throughput (TGS) numbers produced by this script **cannot be compared to any table in the paper**. The model, EP degree, batch size, and node topology all differ from the paper's configurations.
- This script is intended **only** to confirm end-to-end functional correctness (model compiles, training steps run, no NaN/OOM errors) across all four precision modes.
- Before running, update `BASE_DIR`, `TOKENIZER_PATH`, and `DATASET_FILE` inside the script to match your local paths.

## Notes

- The end-to-end scripts contain **cluster-specific** environment settings and absolute paths. AE reviewers should adapt them to their own environment.
- For best reproducibility, run performance benchmarks on an idle system and keep GPU clocks stable if possible.
