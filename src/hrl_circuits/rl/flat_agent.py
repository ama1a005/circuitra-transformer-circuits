"""
Flat PPO agent.

Wraps an EdgeEncoder + FlatActor + FlatCritic into a unified interface
for acting and evaluating actions.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from hrl_circuits.rl.masks import masked_categorical, validate_action_mask
from hrl_circuits.rl.networks import EdgeEncoder, FlatActor, FlatCritic


class FlatAgent(nn.Module):
    """
    PPO agent for the *flat* (non-hierarchical) circuit-localization policy.

    Parameters
    ----------
    edge_encoder : EdgeEncoder
        Maps raw per-edge features to dense embeddings.
    flat_actor : FlatActor
        Scores edges (+ optional STOP) given embeddings and global context.
    flat_critic : FlatCritic
        Estimates V(s) by pooling edge embeddings.
    optimizer : Optimizer
        Joint optimiser for all three sub-modules.
    """

    def __init__(
        self,
        edge_encoder: EdgeEncoder,
        flat_actor: FlatActor,
        flat_critic: FlatCritic,
        optimizer: Optimizer,
    ) -> None:
        super().__init__()
        self.edge_encoder = edge_encoder
        self.flat_actor = flat_actor
        self.flat_critic = flat_critic
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # Acting (single observation, no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        edge_features: Tensor,
        global_features: Tensor,
        valid_mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample (or greedily select) an action.

        Parameters
        ----------
        edge_features : Tensor, shape (E, feat_dim)
        global_features : Tensor, shape (global_dim,)
        valid_mask : Tensor, shape (E+1,) or (E,)  – bool
        deterministic : bool
            If True, take argmax instead of sampling.

        Returns
        -------
        action_idx : Tensor  scalar int64
        log_prob   : Tensor  scalar float32
        value      : Tensor  scalar float32
        entropy    : Tensor  scalar float32
        """
        # Add batch dimension
        ef = edge_features.unsqueeze(0)          # (1, E, feat)
        gf = global_features.unsqueeze(0)        # (1, global)
        vm = valid_mask.unsqueeze(0)             # (1, A)

        edge_emb = self.edge_encoder(ef)         # (1, E, emb)
        logits = self.flat_actor(edge_emb, gf)   # (1, A)

        dist = masked_categorical(logits.squeeze(0), vm.squeeze(0))

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.flat_critic(edge_emb, gf).squeeze(0)

        return action, log_prob, value, entropy

    # ------------------------------------------------------------------
    # Evaluation (batch of observations, with grad)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        edge_features: Tensor,
        global_features: Tensor,
        actions: Tensor,
        valid_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Re-compute log-probs, values, and entropy for a minibatch.

        Parameters
        ----------
        edge_features : Tensor, shape (batch, E, feat_dim)
        global_features : Tensor, shape (batch, global_dim)
        actions : Tensor, shape (batch,)  – int64
        valid_masks : Tensor, shape (batch, A)  – bool

        Returns
        -------
        log_probs : Tensor, shape (batch,)
        values    : Tensor, shape (batch,)
        entropy   : Tensor, shape (batch,)
        """
        edge_emb = self.edge_encoder(edge_features)       # (B, E, emb)
        logits = self.flat_actor(edge_emb, global_features)  # (B, A)

        # Apply mask batchwise
        masked_logits = logits.masked_fill(~valid_masks, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.flat_critic(edge_emb, global_features)

        return log_probs, values, entropy

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_parameter_count(self) -> int:
        """Total number of trainable parameters across all sub-modules."""
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
