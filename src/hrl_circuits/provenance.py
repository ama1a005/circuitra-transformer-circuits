"""
hrl_circuits.provenance
=======================

Reproducibility manifest for the HRL Circuit Localization project.

The :class:`ReproducibilityManifest` captures every factor needed to
reproduce a training run in a single, serialisable snapshot:

* Git commit hash and dirty-state flag
* Config file path and SHA-256 hash of its contents
* Random seed and determinism flag
* Hardware information (GPUs, CPUs, RAM)
* Python and key package versions
* Benchmark and model revision tags
* Prefilter and reward hyper-parameters
* Option horizon and evaluation budget
* File paths to the checkpoint, final circuit, and result files

Usage
-----
::

    from hrl_circuits.provenance import ReproducibilityManifest

    manifest = ReproducibilityManifest.capture(
        config_path="config/my_run.yaml",
        seed=42,
        task="ioi",
        candidate_graph_hash="sha256:abc123…",
        option_horizon=10,
        evaluation_budget=200,
    )
    manifest.save("outputs/manifests/run_001_manifest.json")

    # Later, load and inspect:
    loaded = ReproducibilityManifest.load("outputs/manifests/run_001_manifest.json")
    print(loaded.git_commit)
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Packages whose versions are captured in every manifest
_TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pyyaml",
    "pydantic",
    "transformer-lens",
    "einops",
    "tqdm",
    "matplotlib",
    "hrl-circuits",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_git_commit() -> tuple[str, bool]:
    """Return ``(commit_hash, is_dirty)`` for the current git repository.

    Returns ``("unknown", False)`` if git is not available or the working
    directory is not inside a git repository.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown", False


def _hash_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's contents.

    Returns ``"<file not found>"`` if the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        return "<file not found>"
    sha256 = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_package_versions(packages: tuple[str, ...]) -> dict[str, str]:
    """Return a dict of ``{package_name: version}`` for the given packages."""
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "<not installed>"
    return versions


def _get_hardware_info() -> dict[str, Any]:
    """Collect hardware information: GPU, CPU count, and total RAM."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_model": platform.processor() or "unknown",
        "python_implementation": platform.python_implementation(),
        "gpus": [],
    }

    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "unknown"
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": round(
                            props.total_memory / (1024**3), 2
                        ),
                        "major": props.major,
                        "minor": props.minor,
                        "multi_processor_count": props.multi_processor_count,
                    }
                )
        else:
            info["cuda_version"] = None
    except ImportError:
        info["cuda_version"] = None

    try:
        import psutil  # type: ignore[import]  # optional dependency
        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024**3), 2)
    except ImportError:
        info["total_ram_gb"] = None

    return info


# ─────────────────────────────────────────────────────────────────────────────
# ReproducibilityManifest
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ReproducibilityManifest:
    """A complete provenance record for a single HRL training run.

    Attributes
    ----------
    git_commit : str
        The full git commit hash of the codebase at run time.
        ``"unknown"`` if not in a git repository.
    git_dirty : bool
        ``True`` if there were uncommitted changes at the time of capture.
    config_path : str
        Absolute path to the YAML configuration file.
    config_hash : str
        SHA-256 hex digest of the config file contents at capture time.
    seed : int
        The global random seed used for the run.
    deterministic : bool
        Whether deterministic CUDA algorithms were enabled.
    hardware : dict[str, Any]
        Hardware information: GPU model(s), CPU count, total RAM.
    python_version : str
        Full Python version string (``sys.version``).
    package_versions : dict[str, str]
        Versions of all tracked packages (torch, numpy, …).
    benchmark_revision : str
        Dataset / benchmark version/revision tag.
    model_revision : str
        Model checkpoint revision tag.
    task : str
        The benchmark task (e.g. ``"ioi"``).
    candidate_graph_hash : str
        SHA-256 hash of the :class:`~hrl_circuits.types.CandidateGraph` used.
    prefilter_settings : dict[str, Any]
        Settings used to pre-filter the candidate graph (attribution method,
        edge threshold, etc.).
    reward_settings : dict[str, Any]
        Reward function hyper-parameters (alpha, beta, gamma, guard, …).
    option_horizon : int
        Maximum number of worker steps per manager option (HRL-specific).
    evaluation_budget : int
        Maximum causal evaluation invocations per episode.
    checkpoint_path : str
        Path to the final (or best) model checkpoint.
    final_circuit_path : str
        Path to the saved final circuit edge-mask JSON.
    result_path : str
        Path to the episode result JSON file.
    captured_at : str
        ISO-8601 UTC timestamp of when the manifest was created.
    """

    # ── provenance fields ─────────────────────────────────────────────────────

    git_commit: str = "unknown"
    """Full git commit SHA-1 hash."""

    git_dirty: bool = False
    """True if there were uncommitted changes at capture time."""

    config_path: str = ""
    """Absolute path to the YAML config file."""

    config_hash: str = ""
    """SHA-256 hash of the config file contents."""

    seed: int = 0
    """Global random seed."""

    deterministic: bool = True
    """Whether deterministic CUDA algorithms were active."""

    hardware: dict[str, Any] = field(default_factory=dict)
    """Hardware inventory (GPUs, CPUs, RAM)."""

    python_version: str = ""
    """Full Python version string."""

    package_versions: dict[str, str] = field(default_factory=dict)
    """Versions of key packages."""

    benchmark_revision: str = ""
    """Benchmark / dataset version tag."""

    model_revision: str = "main"
    """Model checkpoint revision tag."""

    task: str = ""
    """Benchmark task name."""

    candidate_graph_hash: str = ""
    """SHA-256 hash of the CandidateGraph."""

    prefilter_settings: dict[str, Any] = field(default_factory=dict)
    """Pre-filter / graph-construction hyper-parameters."""

    reward_settings: dict[str, Any] = field(default_factory=dict)
    """Reward function hyper-parameters."""

    option_horizon: int = 0
    """Max worker steps per manager option."""

    evaluation_budget: int = 0
    """Max causal evaluation invocations per episode."""

    checkpoint_path: str = ""
    """Path to the final/best model checkpoint."""

    final_circuit_path: str = ""
    """Path to the final circuit edge-mask JSON."""

    result_path: str = ""
    """Path to the episode result JSON."""

    captured_at: str = field(default_factory=lambda: _utc_now_iso())
    """ISO-8601 UTC timestamp of manifest creation."""

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def capture(
        cls,
        *,
        config_path: str = "",
        seed: int = 0,
        deterministic: bool = True,
        task: str = "",
        candidate_graph_hash: str = "",
        prefilter_settings: dict[str, Any] | None = None,
        reward_settings: dict[str, Any] | None = None,
        option_horizon: int = 0,
        evaluation_budget: int = 0,
        benchmark_revision: str = "",
        model_revision: str = "main",
        checkpoint_path: str = "",
        final_circuit_path: str = "",
        result_path: str = "",
    ) -> "ReproducibilityManifest":
        """Capture a full provenance snapshot and return a new manifest.

        This is the primary factory method.  It automatically determines the
        git commit, config hash, hardware info, and package versions.

        Parameters
        ----------
        config_path : str
            Path to the YAML config file.  The file hash is computed
            automatically.
        seed : int
            The global random seed used for this run.
        deterministic : bool
            Whether deterministic CUDA operations were enabled.
        task : str
            Benchmark task name.
        candidate_graph_hash : str
            SHA-256 hash of the candidate graph.
        prefilter_settings : dict | None
            Graph construction / pre-filter settings.
        reward_settings : dict | None
            Reward hyper-parameters.
        option_horizon : int
            Max worker steps per option.
        evaluation_budget : int
            Max causal evaluation invocations per episode.
        benchmark_revision : str
            Benchmark version tag.
        model_revision : str
            Model checkpoint revision.
        checkpoint_path : str
            Path to checkpoint file (may be empty at capture time).
        final_circuit_path : str
            Path to final circuit mask (may be empty at capture time).
        result_path : str
            Path to result JSON (may be empty at capture time).

        Returns
        -------
        ReproducibilityManifest
            A fully populated manifest.

        Examples
        --------
        >>> m = ReproducibilityManifest.capture(
        ...     config_path="config/ioi.yaml",
        ...     seed=42,
        ...     task="ioi",
        ... )
        >>> isinstance(m.git_commit, str)
        True
        """
        git_commit, git_dirty = _get_git_commit()
        config_abs = str(Path(config_path).resolve()) if config_path else ""
        config_hash = _hash_file(config_abs) if config_abs else ""

        logger.debug(
            "Capturing reproducibility manifest: commit=%s dirty=%s",
            git_commit,
            git_dirty,
        )

        return cls(
            git_commit=git_commit,
            git_dirty=git_dirty,
            config_path=config_abs,
            config_hash=config_hash,
            seed=seed,
            deterministic=deterministic,
            hardware=_get_hardware_info(),
            python_version=sys.version,
            package_versions=_get_package_versions(_TRACKED_PACKAGES),
            benchmark_revision=benchmark_revision,
            model_revision=model_revision,
            task=task,
            candidate_graph_hash=candidate_graph_hash,
            prefilter_settings=prefilter_settings or {},
            reward_settings=reward_settings or {},
            option_horizon=option_horizon,
            evaluation_budget=evaluation_budget,
            checkpoint_path=checkpoint_path,
            final_circuit_path=final_circuit_path,
            result_path=result_path,
            captured_at=_utc_now_iso(),
        )

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Serialise and write the manifest to a JSON file.

        The parent directory is created automatically if it does not exist.

        Parameters
        ----------
        path : str | Path
            Destination file path.  Should end in ``.json``.

        Returns
        -------
        Path
            The resolved path of the written file.

        Examples
        --------
        >>> manifest.save("outputs/manifests/run_001.json")
        PosixPath('outputs/manifests/run_001.json')
        """
        dest = Path(path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("ReproducibilityManifest saved to %s", dest)
        return dest

    @classmethod
    def load(cls, path: str | Path) -> "ReproducibilityManifest":
        """Load a manifest from a JSON file.

        Parameters
        ----------
        path : str | Path
            Path to the JSON manifest file.

        Returns
        -------
        ReproducibilityManifest
            The deserialised manifest.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        """
        src = Path(path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Manifest file not found: {src}")
        with src.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        logger.info("ReproducibilityManifest loaded from %s", src)
        return cls(**data)

    # ── helpers ───────────────────────────────────────────────────────────────

    def update(self, **kwargs: Any) -> "ReproducibilityManifest":
        """Return a new manifest with selected fields overridden.

        Useful for updating ``checkpoint_path`` or ``final_circuit_path``
        after they become available at the end of training.

        Parameters
        ----------
        **kwargs
            Field names and new values.

        Returns
        -------
        ReproducibilityManifest
            A new instance with the specified overrides applied.
        """
        current = asdict(self)
        current.update(kwargs)
        return ReproducibilityManifest(**current)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"ReproducibilityManifest("
            f"task={self.task!r}, "
            f"seed={self.seed}, "
            f"git_commit={self.git_commit[:8]!r}{'*' if self.git_dirty else ''}, "
            f"captured_at={self.captured_at!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__: list[str] = ["ReproducibilityManifest"]
