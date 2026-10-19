"""hrl_circuits.agents.ppo
==========================
Re-exports for the PPO agent API, bridging the test-suite import path
``hrl_circuits.agents.ppo`` to the actual implementations in
``hrl_circuits.rl.*``.
"""
from hrl_circuits.rl.advantages import compute_gae
from hrl_circuits.rl.masks import MaskedCategorical, masked_categorical
from hrl_circuits.rl.networks import FlatActor, FlatCritic
from hrl_circuits.rl.ppo_core import ppo_clip_loss

__all__ = [
    "FlatActor",
    "FlatCritic",
    "MaskedCategorical",
    "compute_gae",
    "ppo_clip_loss",
]
