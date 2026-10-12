"""
tests.unit.test_types
=====================

Tests for the core dataclasses in ``hrl_circuits.types``.

Verifies that every public dataclass:
* Can be instantiated with valid arguments.
* Serialises to a dictionary via ``to_dict()``.
* Round-trips correctly through ``from_dict(to_dict())``.
* Raises on invalid field values.
"""

from __future__ import annotations

import pytest

from hrl_circuits.types import (
    CandidateGraph,
    ComponentType,
    EdgeRecord,
    EdgeType,
    EpisodeResult,
    EvalCounter,
    EvalResult,
    RegionRecord,
    RewardBreakdown,
    TerminatedReason,
)


# ---------------------------------------------------------------------------
# EdgeRecord
# ---------------------------------------------------------------------------

def _make_edge_record() -> EdgeRecord:
    return EdgeRecord(
        edge_id="L2.attn.0→L3.mlp",
        src_node_id="L2.attn.0",
        dst_node_id="L3.mlp",
        src_layer=2,
        dst_layer=3,
        src_component_type=ComponentType.ATTENTION_HEAD,
        dst_component_type=ComponentType.MLP,
        edge_type=EdgeType.OV,
        layer_distance=1,
    )


class TestEdgeRecord:
    def test_instantiation(self) -> None:
        e = _make_edge_record()
        assert e.edge_id == "L2.attn.0→L3.mlp"
        assert e.src_layer == 2
        assert e.dst_layer == 3
        assert e.layer_distance == 1

    def test_frozen(self) -> None:
        e = _make_edge_record()
        with pytest.raises((AttributeError, TypeError)):
            e.edge_id = "changed"  # type: ignore[misc]

    def test_hashable(self) -> None:
        e = _make_edge_record()
        d = {e: "value"}
        assert d[e] == "value"

    def test_to_dict_keys(self) -> None:
        e = _make_edge_record()
        d = e.to_dict()
        required = {
            "edge_id", "src_node_id", "dst_node_id",
            "src_layer", "dst_layer", "src_component_type",
            "dst_component_type", "edge_type", "layer_distance",
        }
        assert required <= d.keys()

    def test_roundtrip(self) -> None:
        e = _make_edge_record()
        restored = EdgeRecord.from_dict(e.to_dict())
        assert restored == e

    def test_component_type_values(self) -> None:
        d = _make_edge_record().to_dict()
        assert d["src_component_type"] == ComponentType.ATTENTION_HEAD.value
        assert d["dst_component_type"] == ComponentType.MLP.value

    def test_edge_type_value(self) -> None:
        d = _make_edge_record().to_dict()
        assert d["edge_type"] == EdgeType.OV.value


# ---------------------------------------------------------------------------
# CandidateGraph (types.py version)
# ---------------------------------------------------------------------------

class TestCandidateGraphTypes:
    def _make_graph(self) -> CandidateGraph:
        edges = [_make_edge_record()]
        g = CandidateGraph(
            graph_id="test_graph",
            edges=edges,
            node_ids=["L2.attn.0", "L3.mlp"],
            benchmark_metadata={"task": "ioi"},
        )
        g.compute_hash()
        return g

    def test_num_edges(self) -> None:
        g = self._make_graph()
        assert g.num_edges == 1

    def test_num_nodes(self) -> None:
        g = self._make_graph()
        assert g.num_nodes == 2

    def test_compute_hash_nonempty(self) -> None:
        g = self._make_graph()
        assert len(g.graph_hash) == 64  # SHA-256 hex

    def test_compute_hash_deterministic(self) -> None:
        g1 = self._make_graph()
        g2 = self._make_graph()
        assert g1.graph_hash == g2.graph_hash

    def test_to_dict_roundtrip(self) -> None:
        g = self._make_graph()
        restored = CandidateGraph.from_dict(g.to_dict())
        assert restored.graph_id == g.graph_id
        assert len(restored.edges) == len(g.edges)
        assert restored.graph_hash == g.graph_hash

    def test_edge_index(self) -> None:
        g = self._make_graph()
        idx = g.edge_index()
        assert "L2.attn.0→L3.mlp" in idx


# ---------------------------------------------------------------------------
# RegionRecord (types.py version)
# ---------------------------------------------------------------------------

class TestRegionRecordTypes:
    def _make_region(self) -> RegionRecord:
        return RegionRecord(
            region_id="attn_L3",
            region_key="attention_layer_3",
            edge_ids=frozenset({"L3.attn_head.0→residual", "L3.attn_head.1→residual"}),
            layer=3,
            component_type=ComponentType.ATTENTION_HEAD,
        )

    def test_size(self) -> None:
        r = self._make_region()
        assert r.size == 2

    def test_frozen(self) -> None:
        r = self._make_region()
        with pytest.raises((AttributeError, TypeError)):
            r.region_id = "changed"  # type: ignore[misc]

    def test_to_dict_roundtrip(self) -> None:
        r = self._make_region()
        d = r.to_dict()
        restored = RegionRecord.from_dict(d)
        assert restored.region_id == r.region_id
        assert restored.edge_ids == r.edge_ids
        assert restored.layer == r.layer


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

class TestEvalResult:
    def test_valid_construction(self) -> None:
        er = EvalResult(faithfulness=0.8, task_metric=3.5)
        assert er.faithfulness == 0.8

    def test_rejects_faithfulness_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            EvalResult(faithfulness=1.1, task_metric=0.0)

        with pytest.raises(ValueError):
            EvalResult(faithfulness=-0.1, task_metric=0.0)

    def test_rejects_negative_invocations(self) -> None:
        with pytest.raises(ValueError):
            EvalResult(faithfulness=0.5, task_metric=0.0,
                       causal_eval_invocations_delta=-1)

    def test_benchmark_metrics_default(self) -> None:
        er = EvalResult(faithfulness=0.5, task_metric=2.0)
        assert er.benchmark_metrics == {}


# ---------------------------------------------------------------------------
# EvalCounter
# ---------------------------------------------------------------------------

class TestEvalCounter:
    def test_update(self) -> None:
        counter = EvalCounter()
        result = EvalResult(
            faithfulness=0.9,
            task_metric=1.0,
            causal_eval_invocations_delta=2,
            model_forward_calls_delta=4,
            examples_processed_delta=16,
        )
        counter.update(result)
        assert counter.causal_eval_invocations == 2
        assert counter.model_forward_calls_causal == 4
        assert counter.examples_processed_causal == 16

    def test_reset(self) -> None:
        counter = EvalCounter()
        counter.causal_eval_invocations = 10
        counter.reset()
        assert counter.causal_eval_invocations == 0

    def test_to_dict_keys(self) -> None:
        counter = EvalCounter()
        d = counter.to_dict()
        assert "causal_eval_invocations" in d
        assert "model_forward_calls_causal" in d


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

class TestEpisodeResult:
    def _make(self) -> EpisodeResult:
        return EpisodeResult(
            run_id="run-001",
            task="ioi",
            split="train",
            seed=42,
            graph_hash="abc" * 21 + "d",
            final_mask_path="/tmp/mask.json",
            final_edge_count=8,
            retained_edge_fraction=0.5,
            final_faithfulness=0.9,
            causal_eval_invocations=10,
            model_forward_calls_causal=20,
            examples_processed_causal=80,
            wall_clock_seconds=1.23,
            terminated_reason=TerminatedReason.AGENT_TERMINATED,
            trajectory_path="/tmp/traj.json",
        )

    def test_roundtrip(self) -> None:
        ep = self._make()
        restored = EpisodeResult.from_dict(ep.to_dict())
        assert restored.run_id == ep.run_id
        assert restored.terminated_reason == ep.terminated_reason
        assert restored.final_faithfulness == ep.final_faithfulness

    def test_rejects_bad_fraction(self) -> None:
        with pytest.raises(ValueError):
            EpisodeResult(
                run_id="r", task="t", split="train", seed=0,
                graph_hash="x", final_mask_path="p",
                final_edge_count=1, retained_edge_fraction=1.5,
                final_faithfulness=0.5, causal_eval_invocations=0,
                model_forward_calls_causal=0, examples_processed_causal=0,
                wall_clock_seconds=0.0,
                terminated_reason=TerminatedReason.AGENT_TERMINATED,
                trajectory_path="t",
            )


# ---------------------------------------------------------------------------
# RewardBreakdown
# ---------------------------------------------------------------------------

class TestRewardBreakdown:
    def test_sum_consistency(self) -> None:
        rb = RewardBreakdown(
            delta_faithfulness=0.1,
            delta_sparsity=0.2,
            delta_cost=-0.05,
            guard_penalty=0.0,
            terminal_component=0.0,
            total=0.25,
        )
        assert rb.total == pytest.approx(0.25)

    def test_zero(self) -> None:
        rb = RewardBreakdown.zero()
        assert rb.total == 0.0
        assert rb.delta_faithfulness == 0.0

    def test_add(self) -> None:
        a = RewardBreakdown(0.1, 0.2, -0.05, 0.0, 0.0, 0.25)
        b = RewardBreakdown(0.05, 0.1, -0.02, 0.0, 0.0, 0.13)
        c = a + b
        assert c.delta_faithfulness == pytest.approx(0.15)
        assert c.total == pytest.approx(0.38)

    def test_to_dict_roundtrip(self) -> None:
        rb = RewardBreakdown(0.1, 0.2, -0.05, 0.0, 0.5, 0.75)
        d = rb.to_dict()
        assert d["total"] == pytest.approx(0.75)
        assert "delta_faithfulness" in d
        assert "terminal_component" in d
