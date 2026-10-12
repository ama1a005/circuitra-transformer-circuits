"""
HRL Worker policy.

Wraps EdgeEncoder + WorkerActor + WorkerCritic into a unified interface
for selecting primitive actions within an assigned region.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from hrl_circuits.rl.masks import masked_categorical
from hrl_circuits.rl.networks import EdgeEncoder, WorkerActor, WorkerCritic


class WorkerPolicy(nn.Module):
    """
    HRL Worker that executes primitive test/probe actions inside a region.

    Parameters
    ----------
    edge_encoder : EdgeEncoder
        Maps raw per-edge features → dense edge embeddings.
    worker_actor : WorkerActor
        Scores in-region edges + END_OPTION action.
    worker_critic : WorkerCritic
        Estimates V_worker(s, option).
    optimizer : Optimizer
        Joint optimiser for all three sub-modules.
    """

    def __init__(
        self,
        edge_encoder: EdgeEncoder,
        worker_actor: WorkerActor,
        worker_critic: WorkerCritic,
        optimizer: Optimizer,
    ) -> None:
        super().__init__()
        self.edge_encoder = edge_encoder
        self.worker_actor = worker_actor
        self.worker_critic = worker_critic
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # Acting (single observation, no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        edge_features: Tensor,
        region_embedding: Tensor,
        global_features: Tensor,
        option_step: int,
        valid_worker_mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Choose a primitive action (edge or END_OPTION) within the region.

        Parameters
        ----------
        edge_features : Tensor, shape (E_r, feat_dim)
            Raw features for edges in the active region.
        region_embedding : Tensor, shape (region_emb_dim,)
            Pre-computed region embedding from the RegionEncoder.
        global_features : Tensor, shape (global_dim,)
        option_step : int
            Number of primitive steps already taken in this option.
        valid_worker_mask : Tensor, shape (E_r+1,)  – bool
        deterministic : bool

        Returns
        -------
        action_idx : Tensor  scalar int64
        log_prob   : Tensor  scalar float32
        value      : Tensor  scalar float32
        """
        ef = edge_features.unsqueeze(0)            # (1, E_r, feat)
        re = region_embedding.unsqueeze(0)         # (1, reg_dim)
        gf = global_features.unsqueeze(0)          # (1, global)
        vm = valid_worker_mask.unsqueeze(0)        # (1, A)
        os_t = torch.tensor([option_step], dtype=torch.long)

        edge_emb = self.edge_encoder(ef)           # (1, E_r, emb)
        logits = self.worker_actor(edge_emb, re, gf)  # (1, A)

        dist = masked_categorical(logits.squeeze(0), vm.squeeze(0))

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        value = self.worker_critic(edge_emb, re, gf, os_t).squeeze(0)

        return action, log_prob, value

    # ------------------------------------------------------------------
    # Evaluation (batch, with grad)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        edge_features: Tensor,
        region_embeddings: Tensor,
        global_features: Tensor,
        option_steps: Tensor,
        actions: Tensor,
        valid_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Re-compute log-probs, values, entropy for a minibatch.

        Parameters
        ----------
        edge_features     : Tensor, shape (batch, E_r, feat_dim)
        region_embeddings : Tensor, shape (batch, region_emb_dim)
        global_features   : Tensor, shape (batch, global_dim)
        option_steps      : Tensor, shape (batch,)  – int64
        actions           : Tensor, shape (batch,)  – int64
        valid_masks       : Tensor, shape (batch, A)  – bool

        Returns
        -------
        log_probs : Tensor, shape (batch,)
        values    : Tensor, shape (batch,)
        entropy   : Tensor, shape (batch,)
        """
        edge_emb = self.edge_encoder(edge_features)  # (B, E_r, emb)
        logits = self.worker_actor(
            edge_emb, region_embeddings, global_features
        )  # (B, A)

        masked_logits = logits.masked_fill(~valid_masks, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.worker_critic(
            edge_emb, region_embeddings, global_features, option_steps
        )

        return log_probs, values, entropy
