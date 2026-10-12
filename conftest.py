"""
conftest.py
===========

Top-level pytest configuration file.  Provides shared fixtures that are
available in all test modules without explicit imports.

Scope hierarchy:
  session  – one instance for the entire test run
  module   – one instance per test file
  function – one fresh instance per test function (default)
"""

from __future__ import annotations

import pytest

from tests.fixtures.mock_data import (
    TinyConfig,
    make_mock_env,
    make_tiny_config,
    make_tiny_graph,
    make_tiny_regions,
    MockCircuitEnv,
)
from hrl_circuits.graph.schema import CandidateGraph, RegionRecord


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_graph_16() -> CandidateGraph:
    """A small 16-edge CandidateGraph, shared for the whole session."""
    return make_tiny_graph(n_edges=16)


@pytest.fixture(scope="session")
def tiny_graph_32() -> CandidateGraph:
    """A 32-edge CandidateGraph, shared for the whole session."""
    return make_tiny_graph(n_edges=32)


@pytest.fixture(scope="function")
def tiny_graph() -> CandidateGraph:
    """A fresh 16-edge CandidateGraph for each test function."""
    return make_tiny_graph(n_edges=16)


# ---------------------------------------------------------------------------
# Region fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_regions_16(tiny_graph_16: CandidateGraph) -> list[RegionRecord]:
    """Regions partitioned from the 16-edge graph (session scope)."""
    return make_tiny_regions(tiny_graph_16, mode="component")


@pytest.fixture(scope="function")
def tiny_regions(tiny_graph: CandidateGraph) -> list[RegionRecord]:
    """Fresh regions for each test function."""
    return make_tiny_regions(tiny_graph, mode="component")


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_config() -> TinyConfig:
    """Shared TinyConfig object (session scope)."""
    return make_tiny_config()


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def mock_env() -> MockCircuitEnv:
    """A fresh MockCircuitEnv(n_edges=32, n_regions=4) per test."""
    env = make_mock_env(n_edges=32, n_regions=4)
    env.reset()
    return env


@pytest.fixture(scope="function")
def mock_env_small() -> MockCircuitEnv:
    """A very small MockCircuitEnv(n_edges=8, n_regions=2) per test."""
    env = make_mock_env(n_edges=8, n_regions=2)
    env.reset()
    return env
