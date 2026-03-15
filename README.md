[ [Back to index](https://cTuning.org/ae) ]

# Artifact Checklist for FP8-FLOW-MOE

## Abstract

This artifact accompanies the paper **FP8-FLOW-MOE: A CASTING-FREE FP8 RECIPE WITHOUT DOUBLE QUANTIZATION ERROR**. The artifact includes:

**The artifact is organized into three repositories**:
- **Modified TransformerEngine** — with fused FP8-aware operators and FP8-FLOW-MOE pipeline support.
- **Modified Megatron-LM** — integrating the FP8-FLOW-MOE training pipeline, fused FP8-aware operators, and fine-grained recomputation for `moe_expert`.
- **FP8-FLOW-MOE test suite** — benchmark scripts and expected-result checks for reproducing the paper’s experiments.

**Minimum hardware:** 
- NVIDIA Hopper GPUs (FP8-capable; larger HBM capacity recommended). 
- Operator micro-benchmarks require 1 GPU or 32 GPUS; 
- End-to-end efficiency/ablation/convergence experiments are designed for up to 32 nodes (256 GPUs).

**Key results to reproduce:**
- Convergence validation showing no loss degradation under FP8 training (Section 4.1).
- End-to-end training efficiency improvements on DeepSeek-V3/V2 and Qwen3-235B models (Section 4.2).
- End-to-end component ablations, using the same setup as Section 4.2 with full activation checkpointing (Section 4.3).
- Operator-level speedups for fused permute-and-padding, fused SwiGLU+quantization, scaling-aware FP8 transpose, and FP8 communication (Section 4.4).

---

## Checklist

* **Algorithm:** Yes. FP8-FLOW-MOE is an end-to-end FP8 mixed-precision training framework for MoE models, including fused operators and FP8 communication strategies.
* **Program:** Modified TransformerEngine, modified Megatron-LM, and custom benchmark/launcher scripts (Python and shell). All scripts needed for the reported benchmarks are included in this artifact.
* **Compilation:** TransformerEngine must be compiled from source (requires CUDA toolkit).
* **Model:** DeepSeek-V3/V2 and Qwen3-235B model configurations are used for end-to-end evaluation. Model weights are publicly available on HuggingFace and must be converted to Megatron-Core format (see [Models](#models) section for download and conversion instructions).
* **Data set:** The paper used curated internal training datasets; due to their large size, they cannot be publicly hosted.
* **Run-time environment:** Linux (tested on Ubuntu 24.04.2 LTS). Requires CUDA 12.9+, PyTorch, Triton 3.2.0, DeepEP 1.0.0.5. No root access is required to run the experiments after dependencies are installed.
* **Hardware:** NVIDIA Hopper GPUs. Intra-node: NVLink; inter-node: RoCE.
  - Operator micro-benchmarks: 1 GPU.
  - FP8 communication micro-benchmarks: 1–4 nodes (8 GPUs/node).
  - End-to-end convergence/efficiency/ablation experiments: designed for multi-node (up to 32 nodes/256 GPUs depending on EP/PP settings).
* **Run-time state:** Micro-benchmarks include warm-up iterations before timing to reduce cold-cache effects. Results may vary due to GPU thermal state and system load.
* **Execution:** Sole-user access during benchmarking is recommended. No special process pinning required. Operator benchmarks complete in minutes; end-to-end experiments can take hours per configuration at scale.
* **Metrics:** Kernel execution time (ms), speedup ratio (fused vs baseline), training throughput (TGS: tokens/GPU/s), peak GPU memory (GB/GPU), convergence loss curves.
* **Output:**  
  - Operator micro-benchmarks output CSV files (`perf_data*.csv`), console logs, and optional PDF plots; expected-result checks are included (e.g., aggregated reports).  
  - End-to-end experiments output Megatron training logs; throughput is read from logs; peak memory is measured externally.
* **Experiments:** We provide step-by-step reproduction instructions in the repository `https://github.com/021ai/FP8-FLOW-MOE-AE.git` (see `README.md`).
* **How much disk space required (approximately)?:** ~5-10 GB for source code + build artifacts (Megatron-LM + TransformerEngine + tests). End-to-end experiments require additional disk space for datasets and checkpoints (varies by model/dataset).
* **How much time is needed to prepare workflow (approximately)?:** ~2 hours for cloning repos and compiling/installing TransformerEngine (excluding large dataset/checkpoint downloads).
* **How much time is needed to complete experiments (approximately)?:** Operator benchmarks ~1 hour total; end-to-end experiments are several hours per configuration (and depend on cluster scale).
<!-- * **Workflow frameworks used?** No special workflow framework; standard Python scripts and shell scripts. -->

---

## Description

### How to access

**Option 1 (Recommended): Clone our pre-integrated repositories**

> **Strongly recommended:** please clone our *pre-integrated* repositories directly. They include our integrated end-to-end test launchers and the component-ablation implementations validated in our environment.

```bash
# Megatron-LM with FP8-FLOW-MOE patches
git clone --recursive https://github.com/021ai/Megatron-LM-FP8FlowMoe.git

# TransformerEngine with FP8-FLOW-MOE patches
git clone --recursive https://github.com/021ai/TransformerEngine-FP8FlowMoe.git

# Test scripts and benchmarks
git clone https://github.com/021ai/FP8-FLOW-MOE-AE.git
```

**Option 2: Apply upstream PRs to official repositories**

- **Megatron-LM (official repo):** https://github.com/NVIDIA/Megatron-LM  
  - Megatron-LM base commit: `4af25fe`
  - Our PRs to apply:
    - https://github.com/NVIDIA/Megatron-LM/pull/2763
    - https://github.com/NVIDIA/Megatron-LM/pull/2764

- **TransformerEngine (official repo):** https://github.com/NVIDIA/TransformerEngine  
  - TransformerEngine base commit: `40c69e7` (from release_v2.8)
  - Our PRs to apply:
    - https://github.com/NVIDIA/TransformerEngine/pull/1921
    - https://github.com/NVIDIA/TransformerEngine/pull/2547
    - https://github.com/NVIDIA/TransformerEngine/pull/2144

- **FP8-FLOW-MOE test scripts (our repo):**
  ```bash
  git clone https://github.com/021ai/FP8-FLOW-MOE-AE.git
  ```
---

### Hardware dependencies

- **GPU:** NVIDIA Hopper (FP8-capable; larger HBM capacity recommended)
- **Intra-node interconnect:** NVLink
- **Inter-node interconnect:** RoCE, Mellanox ConnectX (mlx5)

**Minimum requirements by experiment:**

| Experiment | GPUs Required |
|---|---|
| Convergence validation (Section 4.1) | Multi-node at scale (up to 30 nodes / 240 GPUs) |
| End-to-end efficiency evaluation (Section 4.2) | Multi-node at scale (up to 32 nodes / 256 GPUs depending on EP/PP settings) |
| End-to-end component ablations (Section 4.3) | Same setup as Section 4.2 |
| Operator micro-benchmarks: fused kernels (Section 4.4.1/4.4.3/4.4.4) | 1 GPU |
| Operator micro-benchmarks: FP8 communication  (Section 4.4.2) | EP8: 1 node (8 GPUs); EP16: 2 nodes (16 GPUs); EP32: 4 nodes (32 GPUs) |

---

### Software dependencies

| Software | Version (tested) | Notes |
|---|---:|---|
| OS | Ubuntu 24.04.2 LTS | Other modern Linux distros should also work |
| CUDA Toolkit | 12.9.41 | Tested version |
| PyTorch | 2.7.0a0+79aa17489c.nv25.04 | Tested with NVIDIA NGC nightly build |
| Triton | 3.2.0 | Typically installed alongside PyTorch |
| DeepEP | 1.0.0.5+6519814 | https://github.com/deepseek-ai/DeepEP.git (1.2.1+b57e5e2 also tested), need NVSHMEM support |
| NCCL | CUDA 12.x compatible | For multi-GPU / multi-node comm |

---

### Data sets

- **Operator micro-benchmarks:** No external datasets required.
- **End-to-end experiments (Sections 4.1–4.3):** If real data is not available, **mock data** can be used as a drop-in replacement for functional AE evaluation, but not for reproducing the paper’s convergence or efficiency numbers.

---

### Models

Reviewers can obtain checkpoints by downloading from HuggingFace and converting to Megatron-Core `torch_dist` format as described below. Which can be very large for the large-scale models above. Reviewers may use only the model(s) needed for the specific experiment they plan to reproduce.

| Model | HuggingFace source | Used in |
|---|---|---|
| DeepSeek-V3-0324 (671B) | [deepseek-ai/DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) | Section 4.2 (e2e efficiency), Section 4.3 (ablations) |
| DeepSeek-V2 (236B) | [deepseek-ai/DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) | Section 4.2 (e2e efficiency) |
| Qwen3-235B-A22B | [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) | Section 4.2 (e2e efficiency) |
| DeepSeek-V2-Lite (16B) | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) |  Section 4.1  (convergence_validation) and Single-node smoke tests |


## Installation

```bash
# 1. Create and enter working directory
export FP8FLOWMOE_PATH=/your/path/to
mkdir -p $FP8FLOWMOE_PATH/
cd $FP8FLOWMOE_PATH/

# 2. Clone repositories
git clone --recursive https://github.com/021ai/Megatron-LM-FP8FlowMoe.git
git clone --recursive https://github.com/021ai/TransformerEngine-FP8FlowMoe.git
git clone https://github.com/021ai/FP8-FLOW-MOE-AE.git

# 3. Install TransformerEngine from source
pip uninstall -y transformer_engine  # remove any existing installation
cd $FP8FLOWMOE_PATH/TransformerEngine-FP8FlowMoe
export NVTE_FRAMEWORK=pytorch
pip3 install --no-build-isolation .

# 4. Install Megatron-LM
cd $FP8FLOWMOE_PATH/Megatron-LM-FP8FlowMoe
pip3 install --no-build-isolation .
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
   - Install/compile dependencies (TransformerEngine-FP8FlowMoe, Megatron-LM-FP8FlowMoe).
   - Ensure DeepEP/CUDA/NCCL/NVSHMEM are correctly configured for your cluster.
2. **Experiment execution**
   - Run the corresponding launcher script (`.sh`) or benchmark script (`bench.py`, `bench_intranode.py`, ...).
3. **Result collection**
   - Operator micro-benchmarks generate `perf_data*.csv` and an aggregated report (e.g., `validation_report.txt`).
   - End-to-end runs produce Megatron training logs under `${OUTPUT_DIR}` (configured in each launcher).
   - Convergence validation writes TensorBoard event files under the run output directory (see Group A below).


### Experiment group A: Convergence validation (Section 4.1)

- **Goal:** validate that FP8-FLOW-MOE does not degrade convergence compared to BF16.
- **Entry script:** `tests/convergence_validation/run_convergence.sh <bf16|fp8flowmoe>`
- **Outputs:** logs are written under `${OUTPUT_DIR}` .
  Loss curves should be inspected via TensorBoard event files under the run’s log directory, e.g.: `${OUTPUT_DIR}/<run_name>/tensorboard/events.out.tfevents.*`

### Experiment group B: End-to-end efficiency evaluation (Section 4.2)

- **Goal:** reproduce throughput (TGS: tokens/GPU/s) and peak memory across precision modes and EP/AC settings.
- **Entry script:** `tests/e2e_efficiency_evaluation/run_dlc_e2e.sh`
- **Outputs:** training logs under `${OUTPUT_DIR}` or console; expected reference values in `tests/e2e_efficiency_evaluation/expected_output/expected_results.csv`.
- **Metrics:** TGS from logs (use stable iters 2–6); peak memory measured externally.

### Experiment group C: End-to-end component ablations (Section 4.3)

- **Goal:** reproduce component ablations on **DeepSeek-V3 (671B)** under the **same setup as Group B** with **AC=full**.
- **Entry script:** `tests/e2e_component_ablations/run_dlc_e2e.sh`
- **Outputs:** expected reference values in `tests/e2e_component_ablations/expected_output/expected_results.csv`.
- **Metrics:** **same as Group B**.

### Experiment group D: Operator-level micro-benchmarks (Section 4.4)

- **Entry points:**
  - Single-GPU benchmarks: `tests/operator_benchmarks/*/bench.py`
  - Intra-node communication: `tests/operator_benchmarks/fp8_communication/bench_intranode.py`
  - Inter-node communication: `tests/operator_benchmarks/fp8_communication/bench_internode.py`
  - One-click runner & reporter: `tests/operator_benchmarks/run_all_benchmarks.py`
- **Outputs:**
  - Per-benchmark `perf_data*.csv`
  - Aggregated report `operator_benchmarks/validation_report.txt`

---

## Evaluation and expected result

This section is a **reproduction/validation checklist**.

> **Feasibility note:** Large-scale end-to-end experiments (Sections 4.2–4.3) require substantial Hopper GPU cluster resources.
> If reviewers do not have such a cluster, they can still fully evaluate Section 4.4 (operator micro-benchmarks).

### Claim 1: Convergence validation (Section 4.1)

#### A. Inputs
The script contains cluster-specific paths that must be edited:
- `TOKENIZER_PATH` — Can be replaced with a public HuggingFace model directory (see [Models](#models))
- `DATASET_FILE` — The script retains the original internal datasets path. To use mock data instead, set `MOCK_DATASET=true` and `DATASET_FILE=/dev/null` (see [Data sets](#data-sets))

#### B. Run
```bash
cd $FP8FLOWMOE_PATH/FP8-FLOW-MOE-AE/tests/convergence_validation
bash run_convergence.sh <bf16|fp8flowmoe>
```

#### C. Expected outputs
Loss is logged via TensorBoard under the run output directory: `${OUTPUT_DIR}/<run_name>/tensorboard/events.out.tfevents.*`

#### D. Expected results & acceptable variation
A successful reproduction should show stable convergence (no divergence / NaNs), and the **relative loss error remains consistently below 0.25%** (FP8-FLOW-MOE vs BF16) over the compared steps.

---

### Claim 2: End-to-end efficiency evaluation (Section 4.2)

#### A. Inputs
- `CPT_PRETRAIN_CHECKPOINT_PATH` — Megatron-Core format checkpoint (see [Models](#models) for conversion instructions). Can be obtained by converting public HuggingFace weights.
- `TOKENIZER_PATH` — HuggingFace model directory or converted checkpoint directory containing the tokenizer. Can be replaced with the corresponding public HuggingFace model directory.
- `DATASET_FILE` — The scripts retain the original internal dataset paths used in the paper. To use mock data instead, set `MOCK_DATASET=true` and `DATASET_FILE=/dev/null` (see [Data sets](#data-sets)).

#### B. Run
```bash
cd $FP8FLOWMOE_PATH/FP8-FLOW-MOE-AE/tests/e2e_efficiency_evaluation
bash run_dlc_e2e.sh <MODEL> <PRECISION> <EP> <AC>
```

#### C. Expected outputs

**TGS (tokens/GPU/s):** Each run trains for 6 iterations. Use the **best TGS (= minimum elapsed time)** among iterations 2–6.

TGS is calculated from `elapsed time per iteration (ms)` in the log:

```
TGS = (GBS * SEQ_LEN) / (elapsed_time_ms / 1000) / NUM_GPUS
```

**Peak memory (GB/GPU):** Not reported in training logs. Measure peak per GPU via your cluster's GPU monitoring system.

#### D. Expected results & acceptable variation

The full set of expected values (Tables 1–6 in the paper) is provided in paper or reference file: 
`tests/e2e_efficiency_evaluation/expected_output/expected_results.csv`

- Expected qualitative trends match the paper (FP8-FLOW-MOE ≥ BF16 throughput; larger gains at higher EP; selective recomputation enables higher EP without OOM).
- Acceptable variation: **±15%** deviation in absolute TGS is within normal range due to hardware and network differences across clusters.

---

### Claim 3: End-to-end component ablations (Section 4.3)

#### A. Run
```bash
cd $FP8FLOWMOE_PATH/FP8-FLOW-MOE-AE/tests/e2e_component_ablations
bash run_dlc_e2e.sh <perm_pad|double_quant> <ep8|ep16|ep32>
```

#### B. Expected outputs
Use **the same method as Claim 2** method.

#### C. Expected results & acceptable variation
Acceptable variation: same **±15%** as Claim 2.

---

### Claim 4: Operator-level micro-benchmarks (Section 4.4)

#### A. Run

**Option 1: one-click (recommended)**
```bash
cd $FP8FLOWMOE_PATH/FP8-FLOW-MOE-AE/tests/operator_benchmarks
python run_all_benchmarks.py                # single-GPU benchmarks only
python run_all_benchmarks.py --with-comm   # also run FP8 comm benchmark (requires 8 GPUs)
```

**Option 2: run individual sub-experiments**

Each sub-experiment can also be run independently. After running, use `--report-only` to generate the aggregated report.

```bash
cd $FP8FLOWMOE_PATH/FP8-FLOW-MOE-AE/tests/operator_benchmarks

# Section 4.4.1 — Fused Permute and Padding (1 GPU)
cd fused_permute_and_padding && python bench.py && cd ..

# Section 4.4.2 — FP8 Communication, intra-node (8 GPUs)
cd fp8_communication && python bench_intranode.py && cd ..

# Section 4.4.3 — Fused SwiGLU and Quantization (1 GPU)
cd fused_swiglu_and_quantization && python bench.py && cd ..

# Section 4.4.4 — Scaling-aware FP8 Transpose (1 GPU)
cd scaling_aware_fp8_transpose && python bench.py && cd ..

# Generate aggregated report from existing results (without comm results)
python run_all_benchmarks.py --report-only
```

#### B. Expected outputs
- `tests/operator_benchmarks/*/perf_data*.csv`
- `tests/operator_benchmarks/validation_report.txt` (also printed to console)

#### C. Expected results & acceptable variation
- Fused operators and FP8 communication should be faster than baselines (speedup > 1) across most tested configurations.
- Acceptable variation: **±15%** deviation in speedup is acceptable as long as the qualitative conclusion holds.

---

## Experiment customization

> **Skip this section if the main experiments in Sections 4.1–4.3 can be reproduced successfully on your cluster.**

**Important caveats:**
If a multi-node cluster is unavailable for end-to-end efficiency and component ablation experiments (Sections 4.2–4.3), the following single-node smoke tests provide a lightweight alternative using **DeepSeek-V2-Lite (16B)** on **1 node (8 GPUs)**.

| Property | Value |
|---|---|
| Model | DeepSeek-V2-Lite (16B) |
| Hardware required | 1 node, 8 × GPUs |
| Parallelism | PP=2, EP=4 |
| Data | Mock data (`MOCK_DATASET=true`); update `TOKENIZER_PATH` to your downloaded DeepSeek-V2-Lite HuggingFace directory (see [Models](#models)) |
| Purpose | Functional correctness only — verifying that training launches and completes without errors (no NaN/OOM). TGS numbers **cannot** be compared to the paper (different model, EP, batch size, topology) |

### Smoke-test A: End-to-end efficiency (Section 4.2)

```bash
cd $FP8FLOWMOE_PATH/FP8-FLOW-MOE-AE/tests/e2e_efficiency_evaluation/
bash run_dsw_DeepSeek-V2-Lite.sh <CASE>
# CASE: bf16 | blockwise | tensorwise | fp8flowmoe
```

### Smoke-test B: Component ablations (Section 4.3)

```bash
cd $FP8FLOWMOE_PATH/FP8-FLOW-MOE-AE/tests/e2e_component_ablations/
bash run_dsw_DeepSeek-V2-Lite.sh <CASE>
# CASE: fp8flowmoe | perm_pad | double_quant
```

## Notes

- The end-to-end scripts contain **cluster-specific** environment settings and absolute paths. AE reviewers should adapt them to their own environment.
- For best reproducibility, run performance benchmarks on an idle system and keep GPU clocks stable if possible.
