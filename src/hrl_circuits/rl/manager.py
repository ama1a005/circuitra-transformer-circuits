"""
HRL Manager policy.

Wraps RegionEncoder + ManagerActor + ManagerCritic into a unified
interface for acting over regions (semi-MDP decisions).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from hrl_circuits.rl.masks import masked_categorical
from hrl_circuits.rl.networks import RegionEncoder, ManagerActor, ManagerCritic


class ManagerPolicy(nn.Module):
    """
    HRL Manager that selects which circuit region to explore next.

    Parameters
    ----------
    region_encoder : RegionEncoder
        Aggregates per-edge embeddings and region stats → region embedding.
    manager_actor : ManagerActor
        Scores regions (+ STOP) to produce a categorical policy.
    manager_critic : ManagerCritic
        Estimates V_manager(s).
    optimizer : Optimizer
        Joint optimiser for all three sub-modules.
    """

    def __init__(
        self,
        region_encoder: RegionEncoder,
        manager_actor: ManagerActor,
        manager_critic: ManagerCritic,
        optimizer: Optimizer,
    ) -> None:
        super().__init__()
        self.region_encoder = region_encoder
        self.manager_actor = manager_actor
        self.manager_critic = manager_critic
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # Acting (single observation, no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        region_features: Tensor,
        global_features: Tensor,
        valid_manager_mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Choose a region to assign to the worker.

        Parameters
        ----------
        region_features : Tensor, shape (R, region_feat_dim)
            Pre-built region embeddings (output of RegionEncoder or raw).
        global_features : Tensor, shape (global_dim,)
        valid_manager_mask : Tensor, shape (R+1,) or (R,)  – bool
        deterministic : bool

        Returns
        -------
        region_action : Tensor  scalar int64
        log_prob      : Tensor  scalar float32
        value         : Tensor  scalar float32
        """
        rf = region_features.unsqueeze(0)          # (1, R, feat)
        gf = global_features.unsqueeze(0)          # (1, global)
        vm = valid_manager_mask.unsqueeze(0)        # (1, A)

        logits = self.manager_actor(rf, gf)         # (1, A)

        dist = masked_categorical(logits.squeeze(0), vm.squeeze(0))

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        value = self.manager_critic(rf, gf).squeeze(0)

        return action, log_prob, value

    # ------------------------------------------------------------------
    # Evaluation (batch, with grad)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        region_features: Tensor,
        global_features: Tensor,
        actions: Tensor,
        valid_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Re-compute log-probs, values, entropy for a minibatch.

        Parameters
        ----------
        region_features : Tensor, shape (batch, R, region_feat_dim)
        global_features : Tensor, shape (batch, global_dim)
        actions         : Tensor, shape (batch,)  – int64
        valid_masks     : Tensor, shape (batch, A)  – bool

        Returns
        -------
        log_probs : Tensor, shape (batch,)
        values    : Tensor, shape (batch,)
        entropy   : Tensor, shape (batch,)
        """
        logits = self.manager_actor(region_features, global_features)  # (B, A)
        masked_logits = logits.masked_fill(~valid_masks, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.manager_critic(region_features, global_features)

        return log_probs, values, entropy
