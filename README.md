# Hybrid MoE: Deterministic Computation Experts for Sparse MoE Transformers

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/main.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)

> Embed exact domain computation inside MoE routing. No tool calls, no orchestration, one forward pass.

**[Paper](paper/main.pdf)** | **[Architecture](#architecture)** | **[Results](#key-results)** | **[Quick Start](#quick-start)** | **[Citation](#citation)**

## TL;DR

LLMs are bad at arithmetic. Tool-calling fixes this but adds latency (separate inference passes, API calls). We take a different approach: **add deterministic Python functions as experts in the MoE routing pool**. The model's own router decides when to invoke computation. The math is exact. Everything happens in a single forward pass, **173x faster** than generation-based alternatives.

## Architecture

```
Input: "Calculate DSCR for a property with $875K NOI and $620K annual debt service"
                              │
                    ┌─────────▼──────────┐
                    │   Frozen gpt-oss-20b   │
                    │   (21B params, 3.6B active) │
                    └─────────┬──────────┘
                              │ hidden state (2880-dim)
                    ┌─────────▼──────────┐
                    │  Extended Router     │
                    │  36 experts:         │
                    │  32 neural (frozen)  │
                    │   4 deterministic    │
                    └──┬────────────┬─────┘
                       │            │
              Neural Expert    DSCR Expert (#32)
              (frozen FFN)         │
                       │    ┌──────▼───────┐
                       │    │ 1. Extract:   │
                       │    │    MLP → NOI,  │
                       │    │    ADS         │
                       │    │ 2. Compute:   │
                       │    │    NOI / ADS   │
                       │    │    = 1.411x    │ ← exact
                       │    │ 3. Project:   │
                       │    │    → 2880-dim  │
                       │    └──────┬───────┘
                       │           │
                    ┌──▼───────────▼──┐
                    │  Weighted sum    │
                    │  → residual      │
                    └─────────────────┘
```

Each deterministic expert has three stages:
1. **Extraction MLP** (learned): recovers function arguments from the hidden state
2. **Deterministic function** (exact): computes the financial formula in PyTorch
3. **Projection MLP** (learned): maps the result back to hidden dimension

Each extraction probe has ~1.5M parameters (~12M total across 8 input probes + router). The entire 21B base model stays frozen.

## CRE Underwriting Experts

| ID | Expert | Formula | Inputs |
|----|--------|---------|--------|
| 32 | DSCR | NOI / Annual Debt Service | NOI, ADS |
| 33 | LTV | (Loan / Value) x 100 | Loan amount, Property value |
| 34 | Cap Rate | (NOI / Price) x 100 | NOI, Purchase price |
| 35 | Debt Yield | (NOI / Loan) x 100 | NOI, Loan amount |

## Key Results

Evaluated on 200 held-out CRE underwriting scenarios using gpt-oss-20b on NVIDIA B200.

### Extraction Quality (5-fold CV, layer 12)

| Probe | R² (mean +/- std) |
|-------|-------------------|
| DSCR / Debt Service | 0.980 +/- 0.002 |
| LTV / Loan Amount | 0.982 +/- 0.001 |
| LTV / Property Value | 0.983 +/- 0.001 |
| Cap Rate / Purchase Price | 0.982 +/- 0.002 |
| Debt Yield / Loan Amount | 0.983 +/- 0.001 |
| Debt Yield / NOI | 0.977 +/- 0.002 |
| Cap Rate / NOI | 0.975 +/- 0.002 |
| DSCR / NOI | 0.885 +/- 0.183 |

### End-to-End Accuracy (% within 20% error)

| Expert | Hybrid MoE (N=200) | Zero-Shot | CoT | Structured |
|--------|-------------------|-----------|-----|------------|
| DSCR | 60.0% | 85.7% (N=35) | 92.9% (N=14) | 36.4% (N=187) |
| LTV | 86.5% | 100% (N=7) | 50.0% (N=2) | 95.8% (N=48) |
| Cap Rate | 65.0% | 100% (N=10) | 50.0% (N=6) | 100% (N=38) |
| Debt Yield | 65.0% | N/A (N=0) | 0% (N=1) | 100% (N=51) |

**Key insight**: Hybrid MoE always produces predictions (N=200). Baselines can be more accurate *when they parse*, but fail on 6-100% of scenarios.

### Latency

| Metric | Hybrid MoE | Generation Baseline |
|--------|-----------|-------------------|
| Single-sample | **40 ms** | 6,993 ms |
| Speedup | **173x** | 1x |

## Project Structure

```
hybrid-moe/
├── src/
│   ├── model/
│   │   ├── cre_experts.py           # Deterministic PyTorch computation functions
│   │   ├── deterministic_expert.py  # DeterministicExpert + ExpertBank modules
│   │   ├── extended_router.py       # Router extension: 32 → 36 experts
│   │   └── hybrid_moe_layer.py     # MoE layer integration for gpt-oss-20b
│   ├── data/
│   │   └── generate_synthetic.py    # Synthetic CRE scenario generator
│   ├── training/
│   │   ├── precompute.py            # Cache hidden states from frozen model
│   │   └── train.py                 # Train extraction MLPs + router classifier
│   ├── evaluation/
│   │   ├── metrics.py               # Accuracy metrics and value extraction
│   │   └── baselines/
│   │       └── prompts.py           # Zero-shot, CoT, structured prompt templates
│   └── inference/
│       └── __init__.py
├── scripts/
│   ├── setup.sh                     # Environment setup
│   ├── explore_model.py             # Inspect gpt-oss-20b architecture
│   ├── run_pipeline.py              # Full pipeline: precompute → train → evaluate
│   ├── evaluate.py                  # Standalone evaluation
│   └── generate_results.py          # Generate paper-ready tables
├── paper/                           # LaTeX source and compiled PDF
├── data/                            # Synthetic CRE underwriting scenarios
├── configs/
│   └── config.yaml                  # Hyperparameters
├── results/                         # Evaluation outputs (generated)
└── checkpoints/                     # Trained probes (generated, gitignored)
```

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (MXFP4 quantization)
- Access to gpt-oss-20b weights

### Installation

```bash
git clone https://github.com/pavan-kotha/hybrid-moe.git
cd hybrid-moe
pip install -r requirements.txt
```

### Unit Tests (no GPU needed)

```bash
python -m src.model.cre_experts
python -m src.evaluation.metrics
```

### Regenerate Synthetic Data

```bash
python -m src.data.generate_synthetic
```

### Full Pipeline (requires GPU + model)

```bash
bash scripts/setup.sh
python scripts/run_pipeline.py --model_path ./models/gpt-oss-20b
```

### Generate Paper Tables

```bash
python scripts/generate_results.py
```

## Base Model

- **Model**: [OpenAI gpt-oss-20b](https://openai.com/index/introducing-gpt-oss) (Apache 2.0)
- **Total parameters**: 21B (3.6B active per token)
- **Architecture**: 24 layers, 32 experts/layer, top-4 routing, hidden dim 2880
- **Features**: SwiGLU activations, RoPE, sigmoid-normalized routing, MXFP4 quantization

## How It Differs From Related Work

| Approach | Mechanism | Limitations |
|----------|-----------|-------------|
| **Toolformer / ReAct** | External tool calls during generation | Extra inference passes, API latency |
| **OccamLLM** (NeurIPS 2024) | Hidden states control symbolic OccamNet | Separate module, not integrated into routing |
| **IGC** (2025) | Gated calculator in forward pass | Requires fine-tuning, basic arithmetic only |
| **MoE++** (ICLR 2025) | Zero/copy/constant experts | Trivial operations, no computation |
| **Hybrid MoE (ours)** | Deterministic experts in MoE routing | Extraction quality is the bottleneck |

Our key differentiator: we use the **existing MoE routing mechanism** to dispatch to deterministic experts. No new architectural components, no external tools, no additional inference passes.

## Known Limitations

- **Extraction is the bottleneck**: R² = 0.98 sounds high, but 2% variance in inputs compounds through division
- **Synthetic data only**: Not validated on real-world CRE underwriting documents
- **Base model baselines**: gpt-oss-20b is not instruction-tuned, which handicaps generation baselines
- **4 experts, 1 domain**: Generality beyond CRE finance is unproven
- **Probe training, not end-to-end**: Extraction MLPs trained separately on cached hidden states

## Citation

```bibtex
@article{kotha2025hybridmoe,
  title={Hybrid MoE: Integrating Deterministic Computation Experts into Sparse Mixture-of-Experts Transformers},
  author={Kotha, Pavan Kumar},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
