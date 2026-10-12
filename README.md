# CIRCUITRA:Circuit Intelligence through Hierarchical Reinforcement and Temporal Abstraction
> **Hierarchical Reinforcement Learning for Mechanistic Circuit Localization in Transformer Language Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**HRL Circuit Localization** is a framework that frames the mechanistic circuit-discovery problem in transformer language models as a sequential decision-making task solved with **Hierarchical Reinforcement Learning (HRL)**.

A *circuit* is a minimal subgraph of a model's computational graph (attention heads, MLP layers, and their connections) that is causally sufficient for a target capability. Finding circuits is expensive because the search space is exponential in the number of edges. This project replaces exhaustive search or greedy ablation with a learned policy that:

1. **Macro-level (option/manager policy):** Selects a *region* of the graph to prune next (e.g., a layer range, a component type).
2. **Micro-level (primitive policy):** Decides exactly which edges within that region to keep or remove.

Reward combines **faithfulness** (how well the pruned circuit retains task performance), **sparsity** (fraction of edges removed), and **computational cost** (number of causal evaluations consumed).

---

## Architecture

```
src/hrl_circuits/
├── __init__.py          # Package entry-point & public API
├── types.py             # Core frozen dataclasses (EdgeRecord, CandidateGraph, …)
├── config.py            # Pydantic config schema + YAML loader
├── seed.py              # Deterministic seeding utility
├── logging_utils.py     # Structured logging setup
├── provenance.py        # ReproducibilityManifest – full run provenance
├── cli.py               # Click CLI entry-point
├── graph/               # Graph construction & manipulation
├── env/                 # RL environments (flat & hierarchical)
├── agents/              # PPO agent implementations
├── evaluation/          # Faithfulness & causal evaluation
├── reward/              # Reward computation
└── benchmarks/          # Task benchmarks (IOI, GTQ, …)
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU recommended (CPU execution is supported but slow)

### From source

```bash
git clone https://github.com/your-org/hrl-circuits.git
cd hrl-circuits

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install with development extras
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Prepare a config file

```yaml
# config/my_run.yaml
project:
  name: my_ioi_run
  experiment_id: ioi_gpt2_001

benchmark:
  task: ioi
  model_name: gpt2-small
  split_ratios: [0.7, 0.15, 0.15]
  seed: 42

reward:
  alpha_faithfulness: 1.0
  beta_sparsity: 0.3
  gamma_cost: 0.01
  guard_threshold: 0.5

ppo:
  lr_actor: 3e-4
  lr_critic: 1e-3
  clip_epsilon: 0.2
  epochs_per_update: 4
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95
```

### 2. Run training

```bash
hrl-circuits train --config config/my_run.yaml --seed 42
```

### 3. Evaluate a checkpoint

```bash
hrl-circuits eval --config config/my_run.yaml --checkpoint checkpoints/best.pt
```

### 4. Export a circuit mask

```bash
hrl-circuits export --checkpoint checkpoints/best.pt --output circuits/ioi_circuit.json
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `hrl-circuits train` | Run the full HRL training loop |
| `hrl-circuits eval` | Evaluate faithfulness of a saved circuit |
| `hrl-circuits export` | Export edge mask as JSON |
| `hrl-circuits provenance` | Print/save a reproducibility manifest |
| `hrl-circuits config validate` | Validate a YAML config file |

---

## Key Concepts

### EdgeRecord
Represents a directed edge in the model's computational graph between two components (attention head, MLP, residual stream node). Edges are the atomic unit of pruning.

### CandidateGraph
The full set of edges under consideration for a given task instance. The agent acts on this graph to produce a *circuit mask*.

### RegionRecord
A named cluster of edges (e.g., "layers 4–6, attention heads") that the manager policy selects as the next pruning target.

### EvalResult / EvalCounter
Track faithfulness metrics and computational budget consumption across episodes.

### ReproducibilityManifest
A snapshot of every factor needed to reproduce a run: git commit, config hash, seed, hardware, package versions, benchmark revision, and final circuit path.

---

## Reward Function

$$
r_t = \alpha \cdot \Delta\text{faith} + \beta \cdot \Delta\text{sparsity} - \gamma \cdot \Delta\text{cost} - \lambda \cdot \mathbf{1}[\text{faith} < \tau] + r_\text{terminal}
$$

| Term | Symbol | Role |
|------|--------|------|
| Faithfulness delta | $\alpha \cdot \Delta\text{faith}$ | Reward for maintaining task capability |
| Sparsity delta | $\beta \cdot \Delta\text{sparsity}$ | Reward for removing more edges |
| Cost delta | $-\gamma \cdot \Delta\text{cost}$ | Penalty for expensive evaluations |
| Guard penalty | $-\lambda \cdot \mathbf{1}[…]$ | Hard penalty when faithfulness drops below threshold |
| Terminal bonus | $r_\text{terminal}$ | Shaped bonus at episode end |

---

## Benchmarks

| Task | Description | Metric |
|------|-------------|--------|
| **IOI** | Indirect Object Identification | Logit difference |
| **GTQ** | Greater-Than Queries | Probability ratio |
| **Docstring** | Code docstring completion | Exact match |

---

## Reproducibility

Every training run automatically saves a `ReproducibilityManifest` JSON alongside the checkpoint:

```json
{
  "git_commit": "a1b2c3d",
  "config_hash": "sha256:...",
  "seed": 42,
  "python_version": "3.11.4",
  "torch_version": "2.2.0",
  "hardware": {"gpu": "NVIDIA A100", "cpu_count": 32},
  "task": "ioi",
  "candidate_graph_hash": "sha256:...",
  "final_faithfulness": 0.94
}
```

---

## Testing

```bash
# Run all tests
pytest

# Run only fast tests
pytest -m "not slow"

# Run with coverage
pytest --cov=hrl_circuits --cov-report=html
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit your changes: `git commit -m 'feat: add my feature'`
4. Push and open a Pull Request

Please run `ruff check .` and `mypy src/` before submitting.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
