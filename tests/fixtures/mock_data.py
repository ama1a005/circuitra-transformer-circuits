"""
tests.fixtures.mock_data
========================

Shared test fixtures and mock data generators used across the test suite.

All factories are deterministic — given the same arguments they always
produce the same output, enabling reproducible tests without external state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from hrl_circuits.graph.schema import CandidateGraph, EdgeRecord, RegionRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_edge(edge_id: int, src_layer: int, dst_layer: int,
               edge_type: str = "attention") -> EdgeRecord:
    """Build a single EdgeRecord for testing."""
    return EdgeRecord(
        edge_id=edge_id,
        src_layer=src_layer,
        dst_layer=dst_layer,
        src_node_id=f"L{src_layer}.h{edge_id % 4}",
        dst_node_id=f"L{dst_layer}.h{(edge_id + 1) % 4}",
        edge_type=edge_type,
        native_src_id=f"blocks.{src_layer}.attn.hook_q_{edge_id % 4}",
        native_dst_id=f"blocks.{dst_layer}.attn.hook_k_{(edge_id + 1) % 4}",
        metadata={},
    )


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

def make_tiny_graph(n_edges: int = 16) -> CandidateGraph:
    """Return a small :class:`CandidateGraph` for fast unit tests.

    Parameters
    ----------
    n_edges:
        Number of edges to include.  Defaults to 16.  The graph spans
        layers 0-3 with alternating attention/mlp edge types.

    Returns
    -------
    CandidateGraph
        A frozen graph with monotonic ``edge_id`` values (0 … n_edges-1)
        and a precomputed ``graph_hash``.
    """
    import hashlib, json

    n_layers = max(2, math.ceil(math.sqrt(n_edges)))
    edges = []
    for i in range(n_edges):
        src_layer = i % n_layers
        dst_layer = (i % n_layers) + 1
        etype = "attention" if i % 2 == 0 else "mlp"
        edges.append(_make_edge(i, src_layer, dst_layer, etype))

    # Compute hash exactly as the production code would
    canonical_dicts = sorted(
        [e.to_hashable_dict() for e in edges],
        key=lambda d: (d["src_layer"], d["dst_layer"], d["src_node_id"],
                       d["dst_node_id"], d["edge_type"]),
    )
    graph_hash = hashlib.sha256(
        json.dumps(canonical_dicts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return CandidateGraph(
        edges=tuple(edges),
        graph_hash=graph_hash,
        benchmark_metadata={
            "benchmark_revision": "v1.0.0-test",
            "model_id": "gpt2-tiny-test",
            "task": "ioi_mock",
        },
    )


def make_tiny_regions(
    graph: CandidateGraph,
    mode: str = "component",
) -> list[RegionRecord]:
    """Partition the edges of *graph* into non-overlapping regions.

    Parameters
    ----------
    graph:
        The candidate graph whose edges will be partitioned.
    mode:
        ``"component"`` – group by edge_type (attention / mlp).
        ``"layer"``     – group by src_layer value.

    Returns
    -------
    list[RegionRecord]
        A list of regions that together cover every edge exactly once.
    """
    if mode == "component":
        groups: dict[str, frozenset[int]] = {}
        for edge in graph.edges:
            groups.setdefault(edge.edge_type, set()).add(edge.edge_id)
        regions = [
            RegionRecord(
                region_id=f"region_{etype}",
                edge_ids=frozenset(ids),
                metadata={"edge_type": etype},
            )
            for etype, ids in sorted(groups.items())
        ]
    elif mode == "layer":
        groups = {}
        for edge in graph.edges:
            key = f"layer_{edge.src_layer}"
            groups.setdefault(key, set()).add(edge.edge_id)
        regions = [
            RegionRecord(
                region_id=rid,
                edge_ids=frozenset(ids),
                metadata={"src_layer": int(rid.split("_")[1])},
            )
            for rid, ids in sorted(groups.items())
        ]
    else:
        raise ValueError(f"Unknown mode {mode!r}. Choose 'component' or 'layer'.")
    return regions


# ---------------------------------------------------------------------------
# Mock environment
# ---------------------------------------------------------------------------

class _MockEvaluator:
    """Fake evaluator that returns deterministic faithfulness values.

    Faithfulness = fraction of live (unmasked) edges, clipped to [0, 1].
    This is completely artificial but is sufficient for testing reward
    computation and episode dynamics without loading any real model.
    """

    def __init__(self, n_edges: int) -> None:
        self._n_edges = n_edges
        self.call_count = 0

    def evaluate(self, mask: list[int]) -> dict[str, float]:
        """Return faithfulness proportional to the fraction of live edges."""
        self.call_count += 1
        live = sum(mask)
        faithfulness = live / max(self._n_edges, 1)
        return {
            "faithfulness": faithfulness,
            "task_metric": faithfulness * 4.2,
            "causal_eval_invocations_delta": 1,
            "model_forward_calls_delta": 2,
            "examples_processed_delta": 8,
            "wall_clock_seconds_delta": 0.001,
        }


@dataclass
class MockCircuitEnv:
    """Minimal mock environment that supports the same step/reset interface
    expected by tests without requiring a real model or dataset.

    Attributes
    ----------
    graph : CandidateGraph
        The candidate graph for this environment.
    regions : list[RegionRecord]
        Region partition of the graph.
    mask : list[int]
        Current edge mask (1 = live, 0 = pruned).  Mutated by :meth:`step`.
    evaluator : _MockEvaluator
        Fake evaluator used to compute faithfulness.
    budget : int
        Remaining evaluation budget (decrements on each evaluate call).
    step_count : int
        Number of steps taken so far.
    max_steps : int
        Maximum steps per episode.
    done : bool
        Whether the current episode has terminated.
    last_reward : float
        Reward from the most recent step.
    last_faithfulness : float
        Faithfulness after the most recent evaluation.
    alpha : float
        Faithfulness reward weight.
    beta : float
        Sparsity reward weight.
    gamma : float
        Cost penalty weight.
    """

    graph: CandidateGraph
    regions: list[RegionRecord]
    mask: list[int] = field(default_factory=list)
    evaluator: _MockEvaluator = field(default_factory=lambda: _MockEvaluator(0))
    budget: int = 50
    step_count: int = 0
    max_steps: int = 100
    done: bool = False
    last_reward: float = 0.0
    last_faithfulness: float = 1.0
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 0.01

    def __post_init__(self) -> None:
        if not self.mask:
            self.mask = [1] * self.graph.num_edges
        if self.evaluator._n_edges == 0:
            self.evaluator = _MockEvaluator(self.graph.num_edges)

    def reset(self) -> dict[str, Any]:
        """Reset the environment and return the initial observation."""
        self.mask = [1] * self.graph.num_edges
        self.step_count = 0
        self.done = False
        self.last_reward = 0.0
        self.last_faithfulness = 1.0
        return self._observation()

    def step(self, action: dict[str, Any]) -> tuple[dict, float, bool, dict]:
        """Execute one environment step.

        Parameters
        ----------
        action:
            Dict with keys:
            * ``"type"`` – ``"prune"`` | ``"stop"`` | ``"evaluate"``
            * ``"edge_id"`` – (for prune actions) the edge index to prune.

        Returns
        -------
        obs, reward, done, info
        """
        assert not self.done, "Cannot step a terminated environment."
        atype = action.get("type", "stop")

        prev_mask = list(self.mask)
        if atype == "prune":
            eid = action["edge_id"]
            if 0 <= eid < len(self.mask):
                self.mask[eid] = 0
        elif atype == "stop":
            self.done = True

        # Evaluate if budget allows
        if not self.done and self.budget > 0:
            result = self.evaluator.evaluate(self.mask)
            self.budget -= 1
            new_faith = result["faithfulness"]
        else:
            new_faith = self.last_faithfulness

        # Compute reward
        delta_faith = new_faith - self.last_faithfulness
        prev_live = sum(prev_mask)
        new_live = sum(self.mask)
        delta_sparse = (prev_live - new_live) / max(self.graph.num_edges, 1)
        cost_penalty = -self.gamma * (1 if self.budget < 50 else 0)
        reward = self.alpha * delta_faith + self.beta * delta_sparse + cost_penalty

        self.last_faithfulness = new_faith
        self.last_reward = reward
        self.step_count += 1

        if self.budget <= 0 or self.step_count >= self.max_steps:
            self.done = True

        return self._observation(), reward, self.done, {
            "faithfulness": new_faith,
            "step": self.step_count,
        }

    def action_mask(self) -> list[int]:
        """Return a binary mask over actions: 1 = valid, 0 = invalid.

        For simplicity, an edge action is valid iff the edge is still live.
        Action n_edges is the STOP action (always valid).
        """
        n = self.graph.num_edges
        mask = list(self.mask)  # edges: valid if live
        mask.append(1)  # STOP is always valid
        return mask

    def _observation(self) -> dict[str, Any]:
        return {
            "mask": list(self.mask),
            "step": self.step_count,
            "budget_remaining": self.budget,
            "faithfulness": self.last_faithfulness,
        }


# ---------------------------------------------------------------------------
# Tiny config
# ---------------------------------------------------------------------------

@dataclass
class _RewardCfg:
    alpha_faithfulness: float = 1.0
    beta_sparsity: float = 0.3
    gamma_cost: float = 0.01
    guard_threshold: float = 0.5
    terminal_success_bonus: float = 1.0
    terminal_failure_penalty: float = -1.0


@dataclass
class _PPOCfg:
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    clip_epsilon: float = 0.2
    epochs_per_update: int = 4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95


@dataclass
class _BenchmarkCfg:
    task: str = "ioi_mock"
    model_name: str = "gpt2-tiny-test"
    split_ratios: list[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    seed: int = 42


@dataclass
class _ProjectCfg:
    name: str = "test_run"
    experiment_id: str = "test_exp_001"


@dataclass
class TinyConfig:
    """Minimal configuration object for use in tests.

    Mirrors the structure of the production ``HRLConfig`` without
    requiring Pydantic or loading YAML files.
    """
    project: _ProjectCfg = field(default_factory=_ProjectCfg)
    benchmark: _BenchmarkCfg = field(default_factory=_BenchmarkCfg)
    reward: _RewardCfg = field(default_factory=_RewardCfg)
    ppo: _PPOCfg = field(default_factory=_PPOCfg)


def make_tiny_config() -> TinyConfig:
    """Return a fully-populated :class:`TinyConfig` with test defaults.

    Returns
    -------
    TinyConfig
        A config object with sensible defaults for fast unit tests.
    """
    return TinyConfig()


def make_mock_env(n_edges: int = 32, n_regions: int = 4) -> MockCircuitEnv:
    """Construct a :class:`MockCircuitEnv` for integration and learning tests.

    Parameters
    ----------
    n_edges:
        Total number of edges in the candidate graph.
    n_regions:
        Number of regions to partition the graph into.

    Returns
    -------
    MockCircuitEnv
        A fully-initialised mock environment ready to be reset.
    """
    graph = make_tiny_graph(n_edges=n_edges)

    # Partition edges evenly into n_regions regions
    all_ids = [e.edge_id for e in graph.edges]
    chunk = max(1, len(all_ids) // n_regions)
    regions = []
    for i in range(n_regions):
        region_ids = all_ids[i * chunk: (i + 1) * chunk]
        if not region_ids:
            break
        regions.append(RegionRecord(
            region_id=f"region_{i}",
            edge_ids=frozenset(region_ids),
            metadata={"region_index": i},
        ))

    # Remaining edges (if any) go into last region
    covered = set().union(*[r.edge_ids for r in regions])
    remaining = frozenset(all_ids) - covered
    if remaining and regions:
        last = regions[-1]
        regions[-1] = RegionRecord(
            region_id=last.region_id,
            edge_ids=last.edge_ids | remaining,
            metadata=last.metadata,
        )

    return MockCircuitEnv(graph=graph, regions=regions)
