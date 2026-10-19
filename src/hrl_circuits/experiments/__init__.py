"""experiments/__init__.py"""
from hrl_circuits.experiments.runner import ExperimentRunner, RunResult
from hrl_circuits.experiments.registry import ExperimentRegistry
from hrl_circuits.experiments.paired_runs import PairedRunManager
from hrl_circuits.experiments.freeze import (
    ConfigFreezer,
    FrozenConfigError,
    freeze_config,
    is_frozen,
    require_frozen_config,
)

__all__ = [
    "ExperimentRunner",
    "RunResult",
    "ExperimentRegistry",
    "PairedRunManager",
    "ConfigFreezer",
    "FrozenConfigError",
    "freeze_config",
    "is_frozen",
    "require_frozen_config",
]
