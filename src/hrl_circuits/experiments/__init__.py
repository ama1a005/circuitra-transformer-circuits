"""
Experiments package for HRL Circuit Localization.

Provides:
- ExperimentRunner    : runs flat and HRL experiments, registers results
- ExperimentRegistry  : CSV-backed registry; never overwrites completed runs
- PairedRunManager    : creates and verifies matched flat/HRL run pairs
- ConfigFreezer       : marks configs as frozen before final experiments
"""

from .runner import ExperimentRunner
from .registry import ExperimentRegistry
from .paired_runs import PairedRunManager
from .freeze import ConfigFreezer, FrozenConfigError

__all__ = [
    "ExperimentRunner",
    "ExperimentRegistry",
    "PairedRunManager",
    "ConfigFreezer",
    "FrozenConfigError",
]
