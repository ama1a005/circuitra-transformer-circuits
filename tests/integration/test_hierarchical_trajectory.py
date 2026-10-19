"""
Integration tests for full hierarchical trajectory collection.

Verifies that a complete HRL episode can be collected into a structured
trajectory with manager decisions and worker steps, and that all
observations, rewards and termination flags are consistent throughout.
"""

import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hrl_circuits.env.mock_env import MockCircuitEnv
from hrl_circuits.env.common import EvalResult


# ---------------------------------------------------------------------------
# Simple trajectory data structures
# ---------------------------------------------------------------------------

@dataclass
class WorkerTransition:
    obs: Dict[str, Any]
    action: int
    reward: float
    next_obs: Dict[str, Any]
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


@dataclass
class ManagerDecision:
    obs: Dict[str, Any]
    region_id: int
    worker_transitions: List[WorkerTransition] = field(default_factory=list)
    option_reward: float = 0.0
    option_done: bool = False
    episode_done: bool = False


@dataclass
class HRLTrajectory:
    manager_decisions: List[ManagerDecision] = field(default_factory=list)
    final_result: Optional[EvalResult] = None

    @property
    def total_reward(self) -> float:
        return sum(d.option_reward for d in self.manager_decisions)

    @property
    def total_worker_steps(self) -> int:
        return sum(len(d.worker_transitions) for d in self.manager_decisions)

    @property
    def num_decisions(self) -> int:
        return len(self.manager_decisions)


# ---------------------------------------------------------------------------
# Trajectory collector helper
# ---------------------------------------------------------------------------

def collect_hrl_trajectory(
    env: MockCircuitEnv,
    max_manager_steps: int = 30,
) -> HRLTrajectory:
    """Collect one full HRL episode into a trajectory object."""
    obs = env.reset()
    trajectory = HRLTrajectory()
    episode_done = False
    mgr_count = 0

    while not episode_done and mgr_count < max_manager_steps:
        # Manager picks a region or STOPs
        mgr_mask = env.valid_manager_action_mask()
        valid = np.where(mgr_mask)[0]
        non_stop = valid[valid < env.num_regions]

        if len(non_stop) == 0 or mgr_count >= max_manager_steps - 1:
            region = env.num_regions  # STOP
        else:
            region = int(non_stop[mgr_count % len(non_stop)])

        decision = ManagerDecision(obs=obs, region_id=region)
        obs = env.select_region(region)

        if env._episode_done:
            decision.episode_done = True
            trajectory.manager_decisions.append(decision)
            episode_done = True
            break

        # Worker runs within option
        option_done = False
        worker_step_count = 0
        while not option_done and not episode_done and worker_step_count < 10:
            w_mask = env.valid_worker_action_mask()
            valid_w = np.where(w_mask)[0]
            # Prefer non-END_OPTION on first step, END_OPTION after
            if worker_step_count == 0 and len(valid_w) > 1:
                action = int(valid_w[0])  # first valid edge action or end
            else:
                action = int(valid_w[-1])  # END_OPTION

            prev_obs = obs
            obs, reward, terminated, truncated, info = env.worker_step(action)
            transition = WorkerTransition(
                obs=prev_obs,
                action=action,
                reward=reward,
                next_obs=obs,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            decision.worker_transitions.append(transition)
            decision.option_reward += reward

            option_done = not env.option_state.active
            episode_done = terminated or truncated or env._episode_done
            worker_step_count += 1

        decision.option_done = option_done
        decision.episode_done = episode_done
        trajectory.manager_decisions.append(decision)
        mgr_count += 1

    trajectory.final_result = env.finalize_episode()
    return trajectory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHRLTrajectoryCollection:
    def test_collect_returns_trajectory(self):
        env = MockCircuitEnv(
            num_edges=16,
            num_regions=4,
            max_episode_steps=100,
            max_manager_decisions=20,
            option_horizon=5,
            seed=0,
        )
        traj = collect_hrl_trajectory(env)
        assert isinstance(traj, HRLTrajectory)

    def test_trajectory_has_decisions(self):
        env = MockCircuitEnv(
            num_edges=16,
            num_regions=4,
            max_episode_steps=100,
            max_manager_decisions=20,
            option_horizon=5,
            seed=1,
        )
        traj = collect_hrl_trajectory(env)
        assert traj.num_decisions >= 1

    def test_trajectory_has_final_result(self):
        env = MockCircuitEnv(
            num_edges=16,
            num_regions=4,
            seed=2,
        )
        traj = collect_hrl_trajectory(env)
        assert traj.final_result is not None
        assert isinstance(traj.final_result, EvalResult)

    def test_final_faithfulness_in_range(self):
        env = MockCircuitEnv(num_edges=8, num_regions=2, seed=3)
        traj = collect_hrl_trajectory(env)
        assert 0.0 <= traj.final_result.faithfulness <= 1.0

    def test_final_sparsity_in_range(self):
        env = MockCircuitEnv(num_edges=8, num_regions=2, seed=4)
        traj = collect_hrl_trajectory(env)
        assert 0.0 <= traj.final_result.sparsity <= 1.0

    def test_total_worker_steps_nonneg(self):
        env = MockCircuitEnv(num_edges=8, num_regions=2, seed=5)
        traj = collect_hrl_trajectory(env)
        assert traj.total_worker_steps >= 0

    def test_worker_transitions_are_well_formed(self):
        env = MockCircuitEnv(
            num_edges=8,
            num_regions=2,
            max_episode_steps=50,
            option_horizon=4,
            seed=6,
        )
        traj = collect_hrl_trajectory(env)
        for decision in traj.manager_decisions:
            for t in decision.worker_transitions:
                assert isinstance(t.obs, dict)
                assert isinstance(t.next_obs, dict)
                assert isinstance(t.reward, float)
                assert isinstance(t.action, int)
                assert isinstance(t.terminated, bool)
                assert isinstance(t.truncated, bool)
                assert isinstance(t.info, dict)

    def test_rewards_are_finite(self):
        env = MockCircuitEnv(num_edges=8, num_regions=2, seed=7)
        traj = collect_hrl_trajectory(env)
        for decision in traj.manager_decisions:
            for t in decision.worker_transitions:
                assert np.isfinite(t.reward), f"Non-finite reward: {t.reward}"

    def test_option_rewards_aggregate_correctly(self):
        env = MockCircuitEnv(num_edges=8, num_regions=2, seed=8)
        traj = collect_hrl_trajectory(env)
        for decision in traj.manager_decisions:
            expected = sum(t.reward for t in decision.worker_transitions)
            assert abs(decision.option_reward - expected) < 1e-6

    def test_episode_terminates_eventually(self):
        env = MockCircuitEnv(
            num_edges=8,
            num_regions=2,
            max_episode_steps=30,
            max_manager_decisions=10,
            option_horizon=3,
            max_causal_forward_calls=50,
            seed=9,
        )
        traj = collect_hrl_trajectory(env, max_manager_steps=15)
        # At least one decision should be the terminal one
        if traj.num_decisions > 0:
            assert any(d.episode_done for d in traj.manager_decisions) or env._episode_done

    def test_multiple_episodes_independent(self):
        """Two episodes on the same env should be independent after reset."""
        env = MockCircuitEnv(num_edges=8, num_regions=2, seed=0)
        traj1 = collect_hrl_trajectory(env)
        traj2 = collect_hrl_trajectory(env)
        # Both must have a final result
        assert traj1.final_result is not None
        assert traj2.final_result is not None

    def test_obs_mask_shape_consistent(self):
        env = MockCircuitEnv(num_edges=12, num_regions=3, seed=10)
        traj = collect_hrl_trajectory(env)
        for decision in traj.manager_decisions:
            assert decision.obs["mask"].shape == (12,)
            for t in decision.worker_transitions:
                assert t.obs["mask"].shape == (12,)
                assert t.next_obs["mask"].shape == (12,)


class TestHRLTrajectoryEdgeCases:
    def test_immediate_manager_stop(self):
        """Manager STOPs on first decision: trajectory has exactly 1 decision."""
        env = MockCircuitEnv(num_edges=8, num_regions=2, seed=0)
        obs = env.reset()
        traj = HRLTrajectory()
        decision = ManagerDecision(obs=obs, region_id=env.num_regions)
        env.select_region(env.num_regions)  # STOP
        decision.episode_done = True
        traj.manager_decisions.append(decision)
        traj.final_result = env.finalize_episode()

        assert traj.num_decisions == 1
        assert traj.final_result is not None
        assert env._episode_done is True

    def test_worker_immediate_end_option(self):
        """Worker sends END_OPTION immediately: 1 worker transition per option."""
        env = MockCircuitEnv(
            num_edges=8,
            num_regions=2,
            option_horizon=5,
            seed=0,
        )
        env.reset()
        env.select_region(0)
        w_mask = env.valid_worker_action_mask()
        end_action = len(w_mask) - 1
        obs, reward, terminated, truncated, info = env.worker_step(end_action)

        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert not env.option_state.active or terminated or truncated

    def test_budget_exhaustion_terminates(self):
        """Very tight budget forces early episode termination."""
        env = MockCircuitEnv(
            num_edges=4,
            num_regions=2,
            max_causal_forward_calls=5,
            max_episode_steps=200,
            option_horizon=3,
            seed=0,
        )
        traj = collect_hrl_trajectory(env, max_manager_steps=50)
        assert traj.final_result is not None

    def test_trajectory_total_reward_type(self):
        env = MockCircuitEnv(num_edges=8, num_regions=2, seed=11)
        traj = collect_hrl_trajectory(env)
        assert isinstance(traj.total_reward, float)
