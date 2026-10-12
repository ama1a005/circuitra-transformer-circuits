"""
hrl_circuits.benchmark
======================
Benchmark protocol, split safety guards, circuit I/O,
and adapter stubs for MIB and InterpBench.
"""

from .base import CircuitBenchmark
from .split_guard import SplitGuard, TuningSplitViolationError
from .circuit_io import (
    save_mask,
    load_mask,
    export_circuit_manifest,
    CircuitManifest,
)

__all__ = [
    "CircuitBenchmark",
    "SplitGuard",
    "TuningSplitViolationError",
    "save_mask",
    "load_mask",
    "export_circuit_manifest",
    "CircuitManifest",
]
