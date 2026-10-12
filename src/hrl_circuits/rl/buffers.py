"""
Rollout buffers for flat PPO, HRL worker, and HRL manager policies.

All buffers are CPU-resident (tensors stored as float32 / int64) and
are moved to the appropriate device by the consumer at training time.
"""

from __future__ import annotations

import math
from typing import Generator, Iterator, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Flat PPO rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Circular-style rollout buffer for a flat PPO agent.

    Stores per-step transitions:
        observations  : (T, *obs_shape)   float32
        actions       : (T,)              int64
        log_probs     : (T,)              float32
        values        : (T,)              float32
        rewards       : (T,)              float32
        dones         : (T,)              float32  (1.0 = terminal)
        action_masks  : (T, n_actions)    bool

    Call ``add()`` for each environment step, ``clear()`` between rollouts,
    and ``iter_minibatches()`` during the PPO update phase.
    """

    def __init__(self) -> None:
        self._observations: list[Tensor] = []
        self._actions: list[Tensor] = []
        self._log_probs: list[Tensor] = []
        self._values: list[Tensor] = []
        self._rewards: list[Tensor] = []
        self._dones: list[Tensor] = []
        self._action_masks: list[Tensor] = []

        # Cached tensors (built lazily by _build_tensors)
        self._built: bool = False
        self.observations: Optional[Tensor] = None
        self.actions: Optional[Tensor] = None
        self.log_probs: Optional[Tensor] = None
        self.values: Optional[Tensor] = None
        self.rewards: Optional[Tensor] = None
        self.dones: Optional[Tensor] = None
        self.action_masks: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(
        self,
        observation: Tensor,
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: float,
        done: bool,
        action_mask: Tensor,
    ) -> None:
        """Append one transition to the buffer."""
        self._observations.append(observation.detach().cpu())
        self._actions.append(action.detach().cpu().long().flatten())
        self._log_probs.append(log_prob.detach().cpu().flatten())
        self._values.append(value.detach().cpu().flatten())
        self._rewards.append(torch.tensor([reward], dtype=torch.float32))
        self._dones.append(torch.tensor([float(done)], dtype=torch.float32))
        self._action_masks.append(action_mask.detach().cpu().bool().flatten())
        self._built = False

    def clear(self) -> None:
        """Reset the buffer for the next rollout."""
        self._observations.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()
        self._action_masks.clear()
        self._built = False
        self.observations = None
        self.actions = None
        self.log_probs = None
        self.values = None
        self.rewards = None
        self.dones = None
        self.action_masks = None

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return len(self._actions)

    def iter_minibatches(
        self, batch_size: int
    ) -> Generator[dict[str, Tensor], None, None]:
        """
        Yield shuffled minibatches as dicts with keys matching field names.

        Calls ``_build_tensors()`` on first use.
        """
        if not self._built:
            self._build_tensors()

        n = self.size
        indices = torch.randperm(n)
        num_batches = math.ceil(n / batch_size)

        for i in range(num_batches):
            idx = indices[i * batch_size : (i + 1) * batch_size]
            yield {
                "observations": self.observations[idx],
                "actions": self.actions[idx],
                "log_probs": self.log_probs[idx],
                "values": self.values[idx],
                "rewards": self.rewards[idx],
                "dones": self.dones[idx],
                "action_masks": self.action_masks[idx],
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tensors(self) -> None:
        """Materialise the stored lists into stacked tensors."""
        self.observations = torch.stack(self._observations, dim=0)
        self.actions = torch.cat(self._actions, dim=0)
        self.log_probs = torch.cat(self._log_probs, dim=0)
        self.values = torch.cat(self._values, dim=0)
        self.rewards = torch.cat(self._rewards, dim=0)
        self.dones = torch.cat(self._dones, dim=0)
        self.action_masks = torch.stack(self._action_masks, dim=0)
        self._built = True


# ---------------------------------------------------------------------------
# HRL worker rollout buffer
# ---------------------------------------------------------------------------

class WorkerRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer for the HRL *worker* policy.

    Extends :class:`RolloutBuffer` with two extra fields:
        option_id    : (T,)  int64  – which option/region the worker is executing
        option_step  : (T,)  int64  – how many steps the current option has run
    """

    def __init__(self) -> None:
        super().__init__()
        self._option_ids: list[Tensor] = []
        self._option_steps: list[Tensor] = []

        self.option_ids: Optional[Tensor] = None
        self.option_steps: Optional[Tensor] = None

    # ------------------------------------------------------------------
    def add(  # type: ignore[override]
        self,
        observation: Tensor,
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: float,
        done: bool,
        action_mask: Tensor,
        option_id: int,
        option_step: int,
    ) -> None:
        """Append one worker transition (includes option context)."""
        super().add(observation, action, log_prob, value, reward, done, action_mask)
        self._option_ids.append(torch.tensor([option_id], dtype=torch.long))
        self._option_steps.append(torch.tensor([option_step], dtype=torch.long))

    def clear(self) -> None:
        super().clear()
        self._option_ids.clear()
        self._option_steps.clear()
        self.option_ids = None
        self.option_steps = None

    def iter_minibatches(  # type: ignore[override]
        self, batch_size: int
    ) -> Generator[dict[str, Tensor], None, None]:
        if not self._built:
            self._build_tensors()

        n = self.size
        indices = torch.randperm(n)
        num_batches = math.ceil(n / batch_size)

        for i in range(num_batches):
            idx = indices[i * batch_size : (i + 1) * batch_size]
            yield {
                "observations": self.observations[idx],
                "actions": self.actions[idx],
                "log_probs": self.log_probs[idx],
                "values": self.values[idx],
                "rewards": self.rewards[idx],
                "dones": self.dones[idx],
                "action_masks": self.action_masks[idx],
                "option_ids": self.option_ids[idx],
                "option_steps": self.option_steps[idx],
            }

    def _build_tensors(self) -> None:
        super()._build_tensors()
        self.option_ids = torch.cat(self._option_ids, dim=0)
        self.option_steps = torch.cat(self._option_steps, dim=0)


# ---------------------------------------------------------------------------
# HRL manager option buffer
# ---------------------------------------------------------------------------

class ManagerOptionBuffer:
    """
    Buffer for the HRL *manager* policy.

    Each entry represents one completed option (semi-MDP transition):

        manager_state          : (T, state_dim)   float32  – state when option started
        region_action          : (T,)             int64    – region chosen by manager
        old_log_prob           : (T,)             float32  – log π_manager(region | state)
        manager_value          : (T,)             float32  – V_manager(state)
        option_discounted_reward : (T,)           float32  – discounted cumulative reward
        option_duration        : (T,)             int64    – number of primitive steps d_k
        next_manager_state     : (T, state_dim)   float32  – state after option finished
        done                   : (T,)             float32  – 1.0 if episode ended
        valid_manager_mask     : (T, n_regions)   bool     – feasible region actions
    """

    def __init__(self) -> None:
        self._manager_states: list[Tensor] = []
        self._region_actions: list[Tensor] = []
        self._old_log_probs: list[Tensor] = []
        self._manager_values: list[Tensor] = []
        self._option_discounted_rewards: list[Tensor] = []
        self._option_durations: list[Tensor] = []
        self._next_manager_states: list[Tensor] = []
        self._dones: list[Tensor] = []
        self._valid_manager_masks: list[Tensor] = []

        self._built: bool = False

        # Cached tensors
        self.manager_states: Optional[Tensor] = None
        self.region_actions: Optional[Tensor] = None
        self.old_log_probs: Optional[Tensor] = None
        self.manager_values: Optional[Tensor] = None
        self.option_discounted_rewards: Optional[Tensor] = None
        self.option_durations: Optional[Tensor] = None
        self.next_manager_states: Optional[Tensor] = None
        self.dones: Optional[Tensor] = None
        self.valid_manager_masks: Optional[Tensor] = None

    # ------------------------------------------------------------------
    def add(
        self,
        manager_state: Tensor,
        region_action: int,
        old_log_prob: float,
        manager_value: float,
        option_discounted_reward: float,
        option_duration: int,
        next_manager_state: Tensor,
        done: bool,
        valid_manager_mask: Tensor,
    ) -> None:
        """Append one completed-option transition."""
        self._manager_states.append(manager_state.detach().cpu().float())
        self._region_actions.append(torch.tensor([region_action], dtype=torch.long))
        self._old_log_probs.append(
            torch.tensor([old_log_prob], dtype=torch.float32)
        )
        self._manager_values.append(
            torch.tensor([manager_value], dtype=torch.float32)
        )
        self._option_discounted_rewards.append(
            torch.tensor([option_discounted_reward], dtype=torch.float32)
        )
        self._option_durations.append(
            torch.tensor([option_duration], dtype=torch.long)
        )
        self._next_manager_states.append(
            next_manager_state.detach().cpu().float()
        )
        self._dones.append(torch.tensor([float(done)], dtype=torch.float32))
        self._valid_manager_masks.append(
            valid_manager_mask.detach().cpu().bool().flatten()
        )
        self._built = False

    def clear(self) -> None:
        """Reset the buffer."""
        self._manager_states.clear()
        self._region_actions.clear()
        self._old_log_probs.clear()
        self._manager_values.clear()
        self._option_discounted_rewards.clear()
        self._option_durations.clear()
        self._next_manager_states.clear()
        self._dones.clear()
        self._valid_manager_masks.clear()
        self._built = False
        self.manager_states = None
        self.region_actions = None
        self.old_log_probs = None
        self.manager_values = None
        self.option_discounted_rewards = None
        self.option_durations = None
        self.next_manager_states = None
        self.dones = None
        self.valid_manager_masks = None

    @property
    def size(self) -> int:
        """Number of completed-option transitions stored."""
        return len(self._region_actions)

    def iter_minibatches(
        self, batch_size: int = 64
    ) -> Generator[dict[str, Tensor], None, None]:
        """Yield shuffled minibatches."""
        if not self._built:
            self._build_tensors()

        n = self.size
        indices = torch.randperm(n)
        num_batches = math.ceil(n / batch_size)

        for i in range(num_batches):
            idx = indices[i * batch_size : (i + 1) * batch_size]
            yield {
                "manager_states": self.manager_states[idx],
                "region_actions": self.region_actions[idx],
                "old_log_probs": self.old_log_probs[idx],
                "manager_values": self.manager_values[idx],
                "option_discounted_rewards": self.option_discounted_rewards[idx],
                "option_durations": self.option_durations[idx],
                "next_manager_states": self.next_manager_states[idx],
                "dones": self.dones[idx],
                "valid_manager_masks": self.valid_manager_masks[idx],
            }

    # ------------------------------------------------------------------
    def _build_tensors(self) -> None:
        self.manager_states = torch.stack(self._manager_states, dim=0)
        self.region_actions = torch.cat(self._region_actions, dim=0)
        self.old_log_probs = torch.cat(self._old_log_probs, dim=0)
        self.manager_values = torch.cat(self._manager_values, dim=0)
        self.option_discounted_rewards = torch.cat(
            self._option_discounted_rewards, dim=0
        )
        self.option_durations = torch.cat(self._option_durations, dim=0)
        self.next_manager_states = torch.stack(self._next_manager_states, dim=0)
        self.dones = torch.cat(self._dones, dim=0)
        self.valid_manager_masks = torch.stack(self._valid_manager_masks, dim=0)
        self._built = True
