"""
HRL Circuit Localization – RL subsystem.

Exposes the main classes and utilities that compose the PPO-based flat
and hierarchical (manager/worker) agents.
"""

from hrl_circuits.rl.buffers import (
    RolloutBuffer,
    WorkerRolloutBuffer,
    ManagerOptionBuffer,
)
from hrl_circuits.rl.advantages import (
    compute_gae,
    compute_manager_gae,
    normalize_advantages,
)
from hrl_circuits.rl.masks import (
    masked_categorical,
    apply_action_mask,
    validate_action_mask,
)
from hrl_circuits.rl.networks import (
    EdgeEncoder,
    RegionEncoder,
    GlobalEncoder,
    FlatActor,
    FlatCritic,
    ManagerActor,
    ManagerCritic,
    WorkerActor,
    WorkerCritic,
)
from hrl_circuits.rl.ppo_core import (
    ppo_clip_loss,
    value_loss,
    entropy_bonus,
    total_ppo_loss,
)
from hrl_circuits.rl.flat_agent import FlatAgent
from hrl_circuits.rl.manager import ManagerPolicy
from hrl_circuits.rl.worker import WorkerPolicy
from hrl_circuits.rl.checkpoint import (
    CheckpointManager,
    CheckpointIncompatibleError,
)

__all__ = [
    # buffers
    "RolloutBuffer",
    "WorkerRolloutBuffer",
    "ManagerOptionBuffer",
    # advantages
    "compute_gae",
    "compute_manager_gae",
    "normalize_advantages",
    # masks
    "masked_categorical",
    "apply_action_mask",
    "validate_action_mask",
    # networks
    "EdgeEncoder",
    "RegionEncoder",
    "GlobalEncoder",
    "FlatActor",
    "FlatCritic",
    "ManagerActor",
    "ManagerCritic",
    "WorkerActor",
    "WorkerCritic",
    # ppo core
    "ppo_clip_loss",
    "value_loss",
    "entropy_bonus",
    "total_ppo_loss",
    # agents
    "FlatAgent",
    "ManagerPolicy",
    "WorkerPolicy",
    # checkpoint
    "CheckpointManager",
    "CheckpointIncompatibleError",
]
