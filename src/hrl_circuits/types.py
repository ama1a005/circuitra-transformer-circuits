"""
hrl_circuits.types
==================

Core frozen and mutable dataclasses representing all entities in the HRL
circuit-localization pipeline.

Design principles
-----------------
* **Frozen dataclasses** (``@dataclass(frozen=True)``) for value objects that
  must be hashable and immutable once created (edges, regions).
* **Regular dataclasses** for mutable accumulators and result containers.
* All fields are annotated with precise types; ``field(default_factory=…)``
  is used instead of mutable defaults.
* Each class carries a docstring describing its role and every field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class ComponentType(str, Enum):
    """Identifies the type of a computational component in the model graph.

    Values correspond to the standard transformer component taxonomy used
    by TransformerLens.
    """

    ATTENTION_HEAD = "attention_head"
    """A single attention head within a multi-head attention layer."""

    MLP = "mlp"
    """The feed-forward / MLP sub-layer of a transformer block."""

    RESIDUAL = "residual"
    """A node on the residual stream (pre/post layer-norm hook points)."""

    EMBED = "embed"
    """Token or positional embedding."""

    UNEMBED = "unembed"
    """The unembedding (logit-lens) projection."""

    LAYER_NORM = "layer_norm"
    """A layer-normalisation component."""

    BIAS = "bias"
    """A standalone bias component (rare but present in some models)."""

    UNKNOWN = "unknown"
    """Fallback for components that could not be classified."""


class EdgeType(str, Enum):
    """Characterises the *type* of connection represented by an edge.

    Edge types encode the semantic relationship between two components and
    are used by the manager policy to group edges into pruning regions.
    """

    QK = "qk"
    """Query-Key interaction edge (attends-from / attends-to)."""

    OV = "ov"
    """Output-Value projection edge."""

    RESIDUAL_FORWARD = "residual_forward"
    """An edge that flows forward through the residual stream."""

    MLP_INPUT = "mlp_input"
    """Edge from the residual stream into an MLP sub-layer."""

    MLP_OUTPUT = "mlp_output"
    """Edge from an MLP sub-layer back into the residual stream."""

    HEAD_OUTPUT = "head_output"
    """Edge from an attention-head output into the residual stream."""

    DIRECT_EFFECT = "direct_effect"
    """Edge representing a component's direct logit contribution."""

    UNKNOWN = "unknown"
    """Fallback for edges whose type could not be determined."""


class TerminatedReason(str, Enum):
    """Enumerates the reasons why an RL episode may terminate.

    Used in :class:`EpisodeResult` to record how the agent stopped.
    """

    BUDGET_EXHAUSTED = "budget_exhausted"
    """The evaluation budget (number of causal interventions) was depleted."""

    FAITHFULNESS_THRESHOLD = "faithfulness_threshold"
    """Faithfulness dropped below the guard threshold and the agent aborted."""

    ALL_EDGES_PRUNED = "all_edges_pruned"
    """Every edge was removed; no edges remain in the candidate graph."""

    MAX_STEPS_REACHED = "max_steps_reached"
    """The maximum number of episode steps was reached."""

    AGENT_TERMINATED = "agent_terminated"
    """The agent produced an explicit ``terminate`` action."""

    MANUAL = "manual"
    """Terminated manually (e.g., via keyboard interrupt or signal)."""


# ─────────────────────────────────────────────────────────────────────────────
# EdgeRecord
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EdgeRecord:
    """Immutable record representing a directed edge in the model's circuit graph.

    An edge connects two computational components (nodes) within a transformer
    model.  Edges are the atomic unit of pruning — each step of the RL agent
    either retains or removes a subset of edges.

    Frozen so that :class:`EdgeRecord` instances are hashable and can be used
    as dictionary keys or stored in sets.

    Attributes
    ----------
    edge_id : str
        Globally unique identifier for this edge, typically constructed as
        ``"<src_node_id>→<dst_node_id>"``.
    src_node_id : str
        Unique identifier of the source (upstream) node. Convention:
        ``"L{layer}.{component_type}.{index}"``, e.g. ``"L3.attn_head.5"``.
    dst_node_id : str
        Unique identifier of the destination (downstream) node.
    src_layer : int
        Zero-indexed transformer layer index of the source component.
        Use ``-1`` for embedding components that precede layer 0.
    dst_layer : int
        Zero-indexed transformer layer index of the destination component.
        Use ``len(model.blocks)`` for the unembedding layer.
    src_component_type : ComponentType
        The type of the source component (attention head, MLP, residual …).
    dst_component_type : ComponentType
        The type of the destination component.
    edge_type : EdgeType
        The semantic type of this edge (QK, OV, residual-forward, …).
    layer_distance : int
        ``dst_layer - src_layer``.  Always ≥ 0 for forward edges.
        Negative values indicate a skip connection in an unusual architecture.

    Examples
    --------
    >>> e = EdgeRecord(
    ...     edge_id="L2.attn_head.0→L3.mlp",
    ...     src_node_id="L2.attn_head.0",
    ...     dst_node_id="L3.mlp",
    ...     src_layer=2,
    ...     dst_layer=3,
    ...     src_component_type=ComponentType.ATTENTION_HEAD,
    ...     dst_component_type=ComponentType.MLP,
    ...     edge_type=EdgeType.RESIDUAL_FORWARD,
    ...     layer_distance=1,
    ... )
    >>> e.edge_id
    'L2.attn_head.0→L3.mlp'
    """

    edge_id: str
    """Globally unique edge identifier."""

    src_node_id: str
    """Unique node identifier of the source component."""

    dst_node_id: str
    """Unique node identifier of the destination component."""

    src_layer: int
    """Layer index of the source component (−1 for embeddings)."""

    dst_layer: int
    """Layer index of the destination component."""

    src_component_type: ComponentType
    """Type of the source component."""

    dst_component_type: ComponentType
    """Type of the destination component."""

    edge_type: EdgeType
    """Semantic type of this edge."""

    layer_distance: int
    """dst_layer − src_layer; non-negative for standard forward edges."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise this record to a JSON-serialisable dictionary."""
        return {
            "edge_id": self.edge_id,
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "src_layer": self.src_layer,
            "dst_layer": self.dst_layer,
            "src_component_type": self.src_component_type.value,
            "dst_component_type": self.dst_component_type.value,
            "edge_type": self.edge_type.value,
            "layer_distance": self.layer_distance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeRecord":
        """Deserialise an :class:`EdgeRecord` from a dictionary."""
        return cls(
            edge_id=data["edge_id"],
            src_node_id=data["src_node_id"],
            dst_node_id=data["dst_node_id"],
            src_layer=int(data["src_layer"]),
            dst_layer=int(data["dst_layer"]),
            src_component_type=ComponentType(data["src_component_type"]),
            dst_component_type=ComponentType(data["dst_component_type"]),
            edge_type=EdgeType(data["edge_type"]),
            layer_distance=int(data["layer_distance"]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CandidateGraph
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CandidateGraph:
    """The full set of edges considered by the agent for a given task instance.

    A :class:`CandidateGraph` is constructed once per benchmark task and
    model, then shared across all episodes. The agent progressively masks
    out edges to discover the minimal faithful circuit.

    Attributes
    ----------
    graph_id : str
        Human-readable identifier, e.g. ``"gpt2-small_ioi_v1"``.
    edges : list[EdgeRecord]
        Ordered list of all edges in this candidate graph.
    node_ids : list[str]
        Ordered list of all unique node identifiers that appear in ``edges``.
    benchmark_metadata : dict[str, Any]
        Arbitrary key-value metadata describing the benchmark or dataset slice
        that was used to construct this graph (e.g. dataset version, split,
        number of examples, attribution method, threshold used).
    graph_hash : str
        SHA-256 hex digest of the canonical JSON serialisation of ``edges``.
        Populated automatically by :meth:`compute_hash` if left empty.
    """

    graph_id: str
    """Human-readable unique identifier for this graph."""

    edges: list[EdgeRecord] = field(default_factory=list)
    """All edges in the candidate graph, ordered deterministically."""

    node_ids: list[str] = field(default_factory=list)
    """All unique node IDs referenced by at least one edge."""

    benchmark_metadata: dict[str, Any] = field(default_factory=dict)
    """Benchmark-level provenance metadata."""

    graph_hash: str = field(default="")
    """SHA-256 hash of the canonical edge list. Set via :meth:`compute_hash`."""

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def num_edges(self) -> int:
        """Total number of edges in the candidate graph."""
        return len(self.edges)

    @property
    def num_nodes(self) -> int:
        """Total number of unique nodes in the candidate graph."""
        return len(self.node_ids)

    # ── methods ───────────────────────────────────────────────────────────────

    def compute_hash(self) -> str:
        """Compute and store the SHA-256 hash of the canonical edge list.

        The hash is derived from the JSON representation of all edges, sorted
        deterministically by ``edge_id``.  Calling this method also sets
        ``self.graph_hash`` in-place and returns the hex digest.

        Returns
        -------
        str
            The SHA-256 hex digest (64 hex characters).
        """
        canonical = json.dumps(
            [e.to_dict() for e in sorted(self.edges, key=lambda e: e.edge_id)],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.graph_hash = hashlib.sha256(canonical).hexdigest()
        return self.graph_hash

    def edge_index(self) -> dict[str, EdgeRecord]:
        """Return a dictionary mapping ``edge_id → EdgeRecord`` for O(1) lookup."""
        return {e.edge_id: e for e in self.edges}

    def to_dict(self) -> dict[str, Any]:
        """Serialise this graph to a JSON-serialisable dictionary."""
        return {
            "graph_id": self.graph_id,
            "edges": [e.to_dict() for e in self.edges],
            "node_ids": self.node_ids,
            "benchmark_metadata": self.benchmark_metadata,
            "graph_hash": self.graph_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateGraph":
        """Deserialise a :class:`CandidateGraph` from a dictionary."""
        return cls(
            graph_id=data["graph_id"],
            edges=[EdgeRecord.from_dict(e) for e in data.get("edges", [])],
            node_ids=list(data.get("node_ids", [])),
            benchmark_metadata=dict(data.get("benchmark_metadata", {})),
            graph_hash=str(data.get("graph_hash", "")),
        )


# ─────────────────────────────────────────────────────────────────────────────
# RegionRecord
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RegionRecord:
    """Immutable record representing a named region (cluster of edges) in the graph.

    The manager policy selects *regions* at each macro-step. A region groups
    related edges (e.g., all attention heads in layers 4–6) so the primitive
    policy can focus on a tractable sub-problem.

    Frozen so that :class:`RegionRecord` instances are hashable and suitable
    as dictionary keys.

    Attributes
    ----------
    region_id : str
        Globally unique identifier for this region, e.g.
        ``"attn_L4-L6_heads"``.
    region_key : str
        A short, human-readable key used for logging and reward shaping,
        e.g. ``"attention_block_4_6"``.
    edge_ids : frozenset[str]
        The set of ``edge_id`` values belonging to this region.
    layer : int
        Representative layer index for this region.  For multi-layer regions,
        this is typically the *lowest* (earliest) layer index.
    component_type : ComponentType
        The predominant component type in this region.  Used by the manager
        to build its action-space description.

    Examples
    --------
    >>> r = RegionRecord(
    ...     region_id="attn_L3",
    ...     region_key="attention_layer_3",
    ...     edge_ids=frozenset({"L3.attn_head.0→residual", "L3.attn_head.1→residual"}),
    ...     layer=3,
    ...     component_type=ComponentType.ATTENTION_HEAD,
    ... )
    >>> len(r.edge_ids)
    2
    """

    region_id: str
    """Globally unique region identifier."""

    region_key: str
    """Short human-readable key for logging."""

    edge_ids: frozenset[str]
    """Frozenset of edge IDs belonging to this region."""

    layer: int
    """Representative (lowest) layer index for this region."""

    component_type: ComponentType
    """Predominant component type within this region."""

    @property
    def size(self) -> int:
        """Number of edges in this region."""
        return len(self.edge_ids)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-serialisable dictionary."""
        return {
            "region_id": self.region_id,
            "region_key": self.region_key,
            "edge_ids": sorted(self.edge_ids),
            "layer": self.layer,
            "component_type": self.component_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionRecord":
        """Deserialise a :class:`RegionRecord` from a dictionary."""
        return cls(
            region_id=data["region_id"],
            region_key=data["region_key"],
            edge_ids=frozenset(data.get("edge_ids", [])),
            layer=int(data["layer"]),
            component_type=ComponentType(data["component_type"]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# EvalResult
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """The result of a single faithfulness / causal evaluation call.

    Every time the evaluator runs a forward pass (or a causal patching
    experiment) it produces an :class:`EvalResult` that is used both to
    compute the reward signal and to update the :class:`EvalCounter`.

    Attributes
    ----------
    faithfulness : float
        Primary faithfulness score in ``[0, 1]``. Typically the normalised
        logit difference or probability ratio relative to the full model.
    task_metric : float
        Raw task-specific metric value (e.g. logit difference, accuracy).
    benchmark_metrics : dict[str, float]
        Additional benchmark-specific metrics keyed by name (e.g.
        ``{"ioi_logit_diff": 4.2, "ioi_prob_correct": 0.91}``).
    causal_eval_invocations_delta : int
        Number of causal evaluation invocations consumed by this call.
        Contributes to the budget tracked in :class:`EvalCounter`.
    model_forward_calls_delta : int
        Number of model forward passes executed during this evaluation.
    examples_processed_delta : int
        Number of dataset examples processed during this evaluation.
    wall_clock_seconds_delta : float
        Wall-clock time (seconds) consumed by this evaluation.
    """

    faithfulness: float
    """Normalised faithfulness score in [0, 1]."""

    task_metric: float
    """Raw task-specific metric value."""

    benchmark_metrics: dict[str, float] = field(default_factory=dict)
    """Additional benchmark-specific metric values."""

    causal_eval_invocations_delta: int = 0
    """Causal evaluation invocations consumed."""

    model_forward_calls_delta: int = 0
    """Model forward passes executed."""

    examples_processed_delta: int = 0
    """Dataset examples processed."""

    wall_clock_seconds_delta: float = 0.0
    """Wall-clock time consumed (seconds)."""

    def __post_init__(self) -> None:
        if not (0.0 <= self.faithfulness <= 1.0):
            raise ValueError(
                f"faithfulness must be in [0, 1], got {self.faithfulness}"
            )
        if self.causal_eval_invocations_delta < 0:
            raise ValueError("causal_eval_invocations_delta must be ≥ 0")
        if self.model_forward_calls_delta < 0:
            raise ValueError("model_forward_calls_delta must be ≥ 0")
        if self.examples_processed_delta < 0:
            raise ValueError("examples_processed_delta must be ≥ 0")
        if self.wall_clock_seconds_delta < 0.0:
            raise ValueError("wall_clock_seconds_delta must be ≥ 0")


# ─────────────────────────────────────────────────────────────────────────────
# EvalCounter
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvalCounter:
    """Mutable counter accumulating evaluation costs across an episode.

    Updated by calling :meth:`update` with each :class:`EvalResult` returned
    by the evaluator. Used to enforce the evaluation budget and compute the
    cost component of the reward.

    Attributes
    ----------
    causal_eval_invocations : int
        Running total of causal evaluation invocations (complete eval calls).
    model_forward_calls_causal : int
        Running total of model forward passes in causal eval mode.
    examples_processed_causal : int
        Running total of dataset examples processed in causal eval mode.
    model_backward_calls_attribution : int
        Running total of model backward passes used for attribution
        (e.g. gradient-based edge scoring).
    """

    causal_eval_invocations: int = 0
    """Total causal evaluation invocations so far."""

    model_forward_calls_causal: int = 0
    """Total model forward passes (causal eval) so far."""

    examples_processed_causal: int = 0
    """Total examples processed in causal eval so far."""

    model_backward_calls_attribution: int = 0
    """Total model backward passes for attribution so far."""

    def update(self, result: EvalResult) -> None:
        """Accumulate costs from a single :class:`EvalResult`.

        Parameters
        ----------
        result : EvalResult
            The evaluation result whose delta fields will be accumulated.
        """
        self.causal_eval_invocations += result.causal_eval_invocations_delta
        self.model_forward_calls_causal += result.model_forward_calls_delta
        self.examples_processed_causal += result.examples_processed_delta

    def reset(self) -> None:
        """Reset all counters to zero (e.g., at the start of a new episode)."""
        self.causal_eval_invocations = 0
        self.model_forward_calls_causal = 0
        self.examples_processed_causal = 0
        self.model_backward_calls_attribution = 0

    def to_dict(self) -> dict[str, int]:
        """Serialise to a JSON-serialisable dictionary."""
        return {
            "causal_eval_invocations": self.causal_eval_invocations,
            "model_forward_calls_causal": self.model_forward_calls_causal,
            "examples_processed_causal": self.examples_processed_causal,
            "model_backward_calls_attribution": self.model_backward_calls_attribution,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EpisodeResult
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodeResult:
    """Complete summary of a single RL episode.

    Produced at episode termination and written to the trajectory log and
    the results database. Contains both outcome metrics and provenance
    information needed to reproduce the episode.

    Attributes
    ----------
    run_id : str
        Unique identifier for the training run, e.g. a UUID or timestamp.
    task : str
        Name of the benchmark task, e.g. ``"ioi"`` or ``"gtq"``.
    split : str
        Dataset split used: ``"train"``, ``"val"``, or ``"test"``.
    seed : int
        Random seed active during this episode.
    graph_hash : str
        SHA-256 hash of the :class:`CandidateGraph` used in this episode.
    final_mask_path : str
        Filesystem path to the saved edge-mask JSON file for the final circuit.
    final_edge_count : int
        Number of edges retained in the final circuit mask.
    retained_edge_fraction : float
        Fraction of the original candidate-graph edges retained (in ``[0, 1]``).
    final_faithfulness : float
        Faithfulness score of the final circuit mask (in ``[0, 1]``).
    causal_eval_invocations : int
        Total causal evaluation invocations consumed during this episode.
    model_forward_calls_causal : int
        Total causal model forward passes during this episode.
    examples_processed_causal : int
        Total examples processed in causal mode during this episode.
    wall_clock_seconds : float
        Total wall-clock time (seconds) for the episode.
    terminated_reason : TerminatedReason
        The reason the episode terminated.
    trajectory_path : str
        Filesystem path to the saved trajectory file (step-by-step log).
    """

    run_id: str
    """Unique training-run identifier."""

    task: str
    """Benchmark task name (e.g. 'ioi', 'gtq')."""

    split: str
    """Dataset split used ('train', 'val', 'test')."""

    seed: int
    """Random seed for this episode."""

    graph_hash: str
    """SHA-256 hash of the candidate graph used."""

    final_mask_path: str
    """Path to the final edge-mask JSON."""

    final_edge_count: int
    """Number of edges retained in the final circuit."""

    retained_edge_fraction: float
    """Fraction of edges retained (in [0, 1])."""

    final_faithfulness: float
    """Faithfulness of the final circuit (in [0, 1])."""

    causal_eval_invocations: int
    """Total causal evaluation invocations for this episode."""

    model_forward_calls_causal: int
    """Total causal model forward passes for this episode."""

    examples_processed_causal: int
    """Total examples processed in causal mode."""

    wall_clock_seconds: float
    """Total wall-clock time for the episode (seconds)."""

    terminated_reason: TerminatedReason
    """Reason the episode ended."""

    trajectory_path: str
    """Path to the trajectory file (step-by-step log)."""

    def __post_init__(self) -> None:
        if not (0.0 <= self.retained_edge_fraction <= 1.0):
            raise ValueError(
                f"retained_edge_fraction must be in [0, 1], "
                f"got {self.retained_edge_fraction}"
            )
        if not (0.0 <= self.final_faithfulness <= 1.0):
            raise ValueError(
                f"final_faithfulness must be in [0, 1], "
                f"got {self.final_faithfulness}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-serialisable dictionary."""
        return {
            "run_id": self.run_id,
            "task": self.task,
            "split": self.split,
            "seed": self.seed,
            "graph_hash": self.graph_hash,
            "final_mask_path": self.final_mask_path,
            "final_edge_count": self.final_edge_count,
            "retained_edge_fraction": self.retained_edge_fraction,
            "final_faithfulness": self.final_faithfulness,
            "causal_eval_invocations": self.causal_eval_invocations,
            "model_forward_calls_causal": self.model_forward_calls_causal,
            "examples_processed_causal": self.examples_processed_causal,
            "wall_clock_seconds": self.wall_clock_seconds,
            "terminated_reason": self.terminated_reason.value,
            "trajectory_path": self.trajectory_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeResult":
        """Deserialise an :class:`EpisodeResult` from a dictionary."""
        return cls(
            run_id=data["run_id"],
            task=data["task"],
            split=data["split"],
            seed=int(data["seed"]),
            graph_hash=data["graph_hash"],
            final_mask_path=data["final_mask_path"],
            final_edge_count=int(data["final_edge_count"]),
            retained_edge_fraction=float(data["retained_edge_fraction"]),
            final_faithfulness=float(data["final_faithfulness"]),
            causal_eval_invocations=int(data["causal_eval_invocations"]),
            model_forward_calls_causal=int(data["model_forward_calls_causal"]),
            examples_processed_causal=int(data["examples_processed_causal"]),
            wall_clock_seconds=float(data["wall_clock_seconds"]),
            terminated_reason=TerminatedReason(data["terminated_reason"]),
            trajectory_path=data["trajectory_path"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# RewardBreakdown
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RewardBreakdown:
    """Decomposition of the scalar reward at a single timestep.

    Provides a detailed, component-wise breakdown of the reward signal
    for interpretability, debugging, and curriculum design.

    The total reward is defined as::

        total = delta_faithfulness + delta_sparsity + delta_cost
                + guard_penalty + terminal_component

    All components (except ``guard_penalty`` which is ≤ 0) may be positive
    or negative depending on the transition outcome.

    Attributes
    ----------
    delta_faithfulness : float
        Shaped reward for the change in faithfulness relative to the previous
        step, scaled by ``alpha`` from the reward config.
    delta_sparsity : float
        Shaped reward for the change in sparsity (fraction of edges removed),
        scaled by ``beta``.  Positive when more edges are pruned.
    delta_cost : float
        Penalty for evaluation budget consumed at this step, scaled by
        ``-gamma``.  Always ≤ 0.
    guard_penalty : float
        Penalty applied when faithfulness falls below the guard threshold
        ``tau``.  Always ≤ 0 (zero when faithfulness ≥ threshold).
    terminal_component : float
        Bonus or penalty applied only at episode termination (zero for
        non-terminal steps).  Encodes the terminal shaping signal (e.g.,
        a bonus for achieving a compact, faithful circuit).
    total : float
        Sum of all components. This is the scalar reward fed to the agent.

    Examples
    --------
    >>> rb = RewardBreakdown(
    ...     delta_faithfulness=0.05,
    ...     delta_sparsity=0.10,
    ...     delta_cost=-0.02,
    ...     guard_penalty=0.0,
    ...     terminal_component=0.0,
    ...     total=0.13,
    ... )
    >>> rb.total
    0.13
    """

    delta_faithfulness: float
    """Reward component for faithfulness improvement (scaled by alpha)."""

    delta_sparsity: float
    """Reward component for sparsity gain (scaled by beta)."""

    delta_cost: float
    """Penalty for evaluation budget consumed (scaled by −gamma; ≤ 0)."""

    guard_penalty: float
    """Hard penalty when faithfulness < guard threshold (≤ 0)."""

    terminal_component: float
    """Terminal shaping bonus/penalty (non-zero only at episode end)."""

    total: float
    """Sum of all reward components; the scalar reward fed to the agent."""

    def to_dict(self) -> dict[str, float]:
        """Serialise to a JSON-serialisable dictionary."""
        return {
            "delta_faithfulness": self.delta_faithfulness,
            "delta_sparsity": self.delta_sparsity,
            "delta_cost": self.delta_cost,
            "guard_penalty": self.guard_penalty,
            "terminal_component": self.terminal_component,
            "total": self.total,
        }

    @classmethod
    def zero(cls) -> "RewardBreakdown":
        """Return a zero :class:`RewardBreakdown` (all components = 0)."""
        return cls(
            delta_faithfulness=0.0,
            delta_sparsity=0.0,
            delta_cost=0.0,
            guard_penalty=0.0,
            terminal_component=0.0,
            total=0.0,
        )

    def __add__(self, other: "RewardBreakdown") -> "RewardBreakdown":
        """Add two breakdowns component-wise (useful for episode aggregation)."""
        return RewardBreakdown(
            delta_faithfulness=self.delta_faithfulness + other.delta_faithfulness,
            delta_sparsity=self.delta_sparsity + other.delta_sparsity,
            delta_cost=self.delta_cost + other.delta_cost,
            guard_penalty=self.guard_penalty + other.guard_penalty,
            terminal_component=self.terminal_component + other.terminal_component,
            total=self.total + other.total,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__: list[str] = [
    # Enums
    "ComponentType",
    "EdgeType",
    "TerminatedReason",
    # Dataclasses
    "EdgeRecord",
    "CandidateGraph",
    "RegionRecord",
    "EvalResult",
    "EvalCounter",
    "EpisodeResult",
    "RewardBreakdown",
]
