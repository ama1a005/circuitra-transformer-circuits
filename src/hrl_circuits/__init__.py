"""
hrl_circuits
============

Hierarchical Reinforcement Learning for Mechanistic Circuit Localization
in Transformer Language Models.

Public API surface:
    - Core data types: EdgeRecord, CandidateGraph, RegionRecord,
      EvalResult, EvalCounter, EpisodeResult, RewardBreakdown
    - Config loading: load_config, HRLConfig
    - Seeding: seed_everything
    - Logging: setup_logging, get_logger
    - Provenance: ReproducibilityManifest
"""

from importlib.metadata import PackageNotFoundError, version

# ── version ──────────────────────────────────────────────────────────────────
try:
    __version__: str = version("hrl-circuits")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0.dev"

# ── public re-exports ─────────────────────────────────────────────────────────
from hrl_circuits.types import (
    CandidateGraph,
    EdgeRecord,
    EpisodeResult,
    EvalCounter,
    EvalResult,
    RegionRecord,
    RewardBreakdown,
)
from hrl_circuits.config import HRLConfig, load_config
from hrl_circuits.seed import seed_everything
from hrl_circuits.logging_utils import get_logger, setup_logging
from hrl_circuits.provenance import ReproducibilityManifest

__all__: list[str] = [
    # types
    "EdgeRecord",
    "CandidateGraph",
    "RegionRecord",
    "EvalResult",
    "EvalCounter",
    "EpisodeResult",
    "RewardBreakdown",
    # config
    "HRLConfig",
    "load_config",
    # utilities
    "seed_everything",
    "setup_logging",
    "get_logger",
    # provenance
    "ReproducibilityManifest",
    # meta
    "__version__",
]
