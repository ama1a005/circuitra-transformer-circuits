"""
Checkpoint management for flat PPO and HRL (manager + worker) agents.

Layout
------
Flat checkpoint directory::

    <output_dir>/
        model.pt            – EdgeEncoder + FlatActor + FlatCritic state dicts
        optimizer.pt        – optimizer state dict
        config.yaml         – serialised training / environment config
        manifest.json       – metadata (timestamp, module names, param counts)
        graph_hash.txt      – hex SHA-256 of the circuit graph
        prefilter_hash.txt  – hex SHA-256 of the prefilter output
        training_state.json – epoch, step, best_reward, etc.

HRL checkpoint directory::

    <output_dir>/
        manager_model.pt    – RegionEncoder + ManagerActor + ManagerCritic
        worker_model.pt     – EdgeEncoder + WorkerActor + WorkerCritic
        manager_optimizer.pt
        worker_optimizer.pt
        config.yaml
        manifest.json
        graph_hash.txt
        prefilter_hash.txt
        training_state.json

Validation
----------
On load, the manager checks:
1. Graph hash matches (raises CheckpointIncompatibleError on mismatch).
2. Prefilter hash matches (same error).
3. Feature dimensions stored in the manifest are compatible with the
   current model architecture.
4. Action schema (include_stop flag, n_regions / n_edges) is compatible.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import Optimizer

from hrl_circuits.rl.flat_agent import FlatAgent
from hrl_circuits.rl.manager import ManagerPolicy
from hrl_circuits.rl.worker import WorkerPolicy


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class CheckpointIncompatibleError(RuntimeError):
    """Raised when a checkpoint cannot be loaded into the current model."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_flat_dims(agent: FlatAgent) -> dict[str, int]:
    """Extract architectural dimensions from a FlatAgent for manifest storage."""
    return {
        "edge_encoder_input_dim": agent.edge_encoder.input_dim,
        "edge_encoder_output_dim": agent.edge_encoder.output_dim,
        "include_stop": int(agent.flat_actor.include_stop),
    }


def _extract_manager_dims(manager: ManagerPolicy) -> dict[str, int]:
    return {
        "region_encoder_edge_embedding_dim": manager.region_encoder.edge_embedding_dim,
        "region_encoder_region_stat_dim": manager.region_encoder.region_stat_dim,
        "region_encoder_output_dim": manager.region_encoder.output_dim,
        "manager_actor_include_stop": int(manager.manager_actor.include_stop),
    }


def _extract_worker_dims(worker: WorkerPolicy) -> dict[str, int]:
    return {
        "edge_encoder_input_dim": worker.edge_encoder.input_dim,
        "edge_encoder_output_dim": worker.edge_encoder.output_dim,
    }


def _build_manifest(
    agent_type: str,
    dim_info: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "agent_type": agent_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **dim_info,
    }
    if extra:
        manifest.update(extra)
    return manifest


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _validate_hash(checkpoint_dir: Path, filename: str, expected_hash: str) -> None:
    actual = _read_text(checkpoint_dir / filename)
    if actual != expected_hash:
        raise CheckpointIncompatibleError(
            f"Hash mismatch in '{filename}':\n"
            f"  checkpoint : {actual}\n"
            f"  current    : {expected_hash}"
        )


def _validate_dims(manifest: dict[str, Any], required: dict[str, int]) -> None:
    for key, value in required.items():
        if key in manifest and manifest[key] != value:
            raise CheckpointIncompatibleError(
                f"Dimension mismatch for '{key}': "
                f"checkpoint={manifest[key]}, model={value}"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Static-method collection for saving and loading flat / HRL checkpoints.
    """

    # ------------------------------------------------------------------ flat

    @staticmethod
    def save_flat_checkpoint(
        agent: FlatAgent,
        optimizer: Optimizer,
        config: dict[str, Any],
        graph_hash: str,
        prefilter_hash: str,
        training_state: dict[str, Any],
        output_dir: str | os.PathLike,
    ) -> Path:
        """
        Persist a flat PPO checkpoint.

        Parameters
        ----------
        agent : FlatAgent
        optimizer : Optimizer
        config : dict
            Full training / environment configuration.
        graph_hash : str
            Hex SHA-256 of the circuit graph used for this run.
        prefilter_hash : str
            Hex SHA-256 of the prefilter artefact.
        training_state : dict
            Arbitrary training metadata (epoch, step, best_reward, …).
        output_dir : path-like
            Directory to write checkpoint files into (created if absent).

        Returns
        -------
        Path
            Resolved path to *output_dir*.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Model weights
        torch.save(
            {
                "edge_encoder": agent.edge_encoder.state_dict(),
                "flat_actor": agent.flat_actor.state_dict(),
                "flat_critic": agent.flat_critic.state_dict(),
            },
            out / "model.pt",
        )

        # Optimizer
        torch.save(optimizer.state_dict(), out / "optimizer.pt")

        # Config
        with open(out / "config.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False)

        # Manifest
        dim_info = _extract_flat_dims(agent)
        manifest = _build_manifest(
            "flat",
            dim_info,
            extra={
                "parameter_count": agent.get_parameter_count(),
                "optimizer_class": type(optimizer).__name__,
            },
        )
        with open(out / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        # Hashes
        _write_text(out / "graph_hash.txt", graph_hash)
        _write_text(out / "prefilter_hash.txt", prefilter_hash)

        # Training state
        with open(out / "training_state.json", "w", encoding="utf-8") as fh:
            json.dump(training_state, fh, indent=2)

        return out

    @staticmethod
    def load_flat_checkpoint(
        checkpoint_dir: str | os.PathLike,
        agent: FlatAgent,
        optimizer: Optimizer,
        graph_hash: str | None = None,
        prefilter_hash: str | None = None,
    ) -> dict[str, Any]:
        """
        Load a flat PPO checkpoint into *agent* and *optimizer* in-place.

        Parameters
        ----------
        checkpoint_dir : path-like
        agent : FlatAgent
            Must be constructed with compatible architecture.
        optimizer : Optimizer
        graph_hash : str or None
            If provided, validates against the stored hash.
        prefilter_hash : str or None
            If provided, validates against the stored hash.

        Returns
        -------
        training_state : dict

        Raises
        ------
        CheckpointIncompatibleError
            On hash or dimension mismatch.
        FileNotFoundError
            If required checkpoint files are missing.
        """
        ckpt = Path(checkpoint_dir)

        # Hash validation
        if graph_hash is not None:
            _validate_hash(ckpt, "graph_hash.txt", graph_hash)
        if prefilter_hash is not None:
            _validate_hash(ckpt, "prefilter_hash.txt", prefilter_hash)

        # Dimension validation
        with open(ckpt / "manifest.json", encoding="utf-8") as fh:
            manifest = json.load(fh)

        if manifest.get("agent_type") not in ("flat", None):
            raise CheckpointIncompatibleError(
                f"Expected agent_type='flat', got '{manifest.get('agent_type')}'."
            )

        required_dims = _extract_flat_dims(agent)
        _validate_dims(manifest, required_dims)

        # Load weights
        model_state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
        agent.edge_encoder.load_state_dict(model_state["edge_encoder"])
        agent.flat_actor.load_state_dict(model_state["flat_actor"])
        agent.flat_critic.load_state_dict(model_state["flat_critic"])

        # Load optimizer
        opt_state = torch.load(ckpt / "optimizer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(opt_state)

        # Return training state
        with open(ckpt / "training_state.json", encoding="utf-8") as fh:
            training_state: dict[str, Any] = json.load(fh)

        return training_state

    # ------------------------------------------------------------------ HRL

    @staticmethod
    def save_hrl_checkpoint(
        manager: ManagerPolicy,
        worker: WorkerPolicy,
        manager_optimizer: Optimizer,
        worker_optimizer: Optimizer,
        config: dict[str, Any],
        graph_hash: str,
        prefilter_hash: str,
        training_state: dict[str, Any],
        output_dir: str | os.PathLike,
    ) -> Path:
        """
        Persist an HRL (manager + worker) checkpoint.

        Parameters
        ----------
        manager : ManagerPolicy
        worker : WorkerPolicy
        manager_optimizer : Optimizer
        worker_optimizer : Optimizer
        config : dict
        graph_hash : str
        prefilter_hash : str
        training_state : dict
        output_dir : path-like

        Returns
        -------
        Path
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Manager weights
        torch.save(
            {
                "region_encoder": manager.region_encoder.state_dict(),
                "manager_actor": manager.manager_actor.state_dict(),
                "manager_critic": manager.manager_critic.state_dict(),
            },
            out / "manager_model.pt",
        )

        # Worker weights
        torch.save(
            {
                "edge_encoder": worker.edge_encoder.state_dict(),
                "worker_actor": worker.worker_actor.state_dict(),
                "worker_critic": worker.worker_critic.state_dict(),
            },
            out / "worker_model.pt",
        )

        # Optimizers
        torch.save(manager_optimizer.state_dict(), out / "manager_optimizer.pt")
        torch.save(worker_optimizer.state_dict(), out / "worker_optimizer.pt")

        # Config
        with open(out / "config.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False)

        # Manifest
        dim_info: dict[str, Any] = {
            "manager": _extract_manager_dims(manager),
            "worker": _extract_worker_dims(worker),
        }
        manifest = _build_manifest("hrl", dim_info)
        with open(out / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        # Hashes
        _write_text(out / "graph_hash.txt", graph_hash)
        _write_text(out / "prefilter_hash.txt", prefilter_hash)

        # Training state
        with open(out / "training_state.json", "w", encoding="utf-8") as fh:
            json.dump(training_state, fh, indent=2)

        return out

    @staticmethod
    def load_hrl_checkpoint(
        checkpoint_dir: str | os.PathLike,
        manager: ManagerPolicy,
        worker: WorkerPolicy,
        manager_optimizer: Optimizer,
        worker_optimizer: Optimizer,
        graph_hash: str | None = None,
        prefilter_hash: str | None = None,
    ) -> dict[str, Any]:
        """
        Load an HRL checkpoint into *manager* and *worker* in-place.

        Parameters
        ----------
        checkpoint_dir : path-like
        manager : ManagerPolicy
        worker : WorkerPolicy
        manager_optimizer : Optimizer
        worker_optimizer : Optimizer
        graph_hash : str or None
        prefilter_hash : str or None

        Returns
        -------
        training_state : dict

        Raises
        ------
        CheckpointIncompatibleError
        FileNotFoundError
        """
        ckpt = Path(checkpoint_dir)

        # Hash validation
        if graph_hash is not None:
            _validate_hash(ckpt, "graph_hash.txt", graph_hash)
        if prefilter_hash is not None:
            _validate_hash(ckpt, "prefilter_hash.txt", prefilter_hash)

        # Manifest validation
        with open(ckpt / "manifest.json", encoding="utf-8") as fh:
            manifest = json.load(fh)

        if manifest.get("agent_type") not in ("hrl", None):
            raise CheckpointIncompatibleError(
                f"Expected agent_type='hrl', got '{manifest.get('agent_type')}'."
            )

        # Dimension checks
        if "manager" in manifest:
            _validate_dims(manifest["manager"], _extract_manager_dims(manager))
        if "worker" in manifest:
            _validate_dims(manifest["worker"], _extract_worker_dims(worker))

        # Load manager
        mgr_state = torch.load(ckpt / "manager_model.pt", map_location="cpu", weights_only=True)
        manager.region_encoder.load_state_dict(mgr_state["region_encoder"])
        manager.manager_actor.load_state_dict(mgr_state["manager_actor"])
        manager.manager_critic.load_state_dict(mgr_state["manager_critic"])
        manager_optimizer.load_state_dict(
            torch.load(ckpt / "manager_optimizer.pt", map_location="cpu", weights_only=True)
        )

        # Load worker
        wkr_state = torch.load(ckpt / "worker_model.pt", map_location="cpu", weights_only=True)
        worker.edge_encoder.load_state_dict(wkr_state["edge_encoder"])
        worker.worker_actor.load_state_dict(wkr_state["worker_actor"])
        worker.worker_critic.load_state_dict(wkr_state["worker_critic"])
        worker_optimizer.load_state_dict(
            torch.load(ckpt / "worker_optimizer.pt", map_location="cpu", weights_only=True)
        )

        # Return training state
        with open(ckpt / "training_state.json", encoding="utf-8") as fh:
            training_state: dict[str, Any] = json.load(fh)

        return training_state
