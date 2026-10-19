"""
Integration tests for MockCircuitEnv HRL (manager + worker) interface.

Tests verify:
- select_region() returns obs dict (NOT a tuple)
- worker_step() returns 5-tuple (obs, reward, terminated, truncated, info)
- option_state.active / option_state.region_id
- Manager STOP (select_region(num_regions)) terminates episode
- Worker END_OPTION (worker_step(len(mask)-1)) ends option
- valid_manager_action_mask / valid_worker_action_mask shapes
- Full HRL episode loop runs without error
"""

import numpy as np
import pytest

from hrl_circuits.env.mock_env import MockCircuitEnv
from hrl_circuits.env.common import EvalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return MockCircuitEnv(
        num_edges=16,
        num_regions=4,
        ground_truth_edges={1, 3, 9},
        severity=1.0,
        max_causal_forward_calls=200,
        max_episode_steps=100,
        max_manager_decisions=20,
        option_horizon=5,
        seed=0,
    )


# ---------------------------------------------------------------------------
# Manager interface tests
# ---------------------------------------------------------------------------

class TestManagerInterface:
    def test_valid_manager_mask_shape(self, env):
        env.reset()
        mask = env.valid_manager_action_mask()
        assert mask.shape == (env.num_regions + 1,)

    def test_valid_manager_mask_dtype(self, env):
        env.reset()
        mask = env.valid_manager_action_mask()
        assert mask.dtype == bool

    def test_stop_action_always_valid(self, env):
        env.reset()
        mask = env.valid_manager_action_mask()
        assert mask[-1], "Manager STOP must always be valid"

    def test_select_region_returns_dict(self, env):
        env.reset()
        obs = env.select_region(0)
        assert isinstance(obs, dict), "select_region() must return a dict, not a tuple"

    def test_select_region_activates_option(self, env):
        env.reset()
        env.select_region(0)
        assert env.option_state.active is True

    def test_select_region_sets_region_id(self, env):
        env.reset()
        env.select_region(2)
        assert env.option_state.region_id == 2

    def test_manager_stop_terminates_episode(self, env):
        env.reset()
        env.select_region(env.num_regions)  # STOP
        assert env._episode_done is True

    def test_manager_stop_returns_dict(self, env):
        env.reset()
        obs = env.select_region(env.num_regions)  # STOP
        assert isinstance(obs, dict)

    def test_invalid_region_id_raises(self, env):
        env.reset()
        with pytest.raises(ValueError):
            env.select_region(env.num_regions + 5)


# ---------------------------------------------------------------------------
# Worker interface tests
# ---------------------------------------------------------------------------

class TestWorkerInterface:
    def test_valid_worker_mask_shape(self, env):
        env.reset()
        env.select_region(0)
        mask = env.valid_worker_action_mask()
        # Shape: (edges_in_region + 1,)
        assert mask.shape[0] >= 2  # at least 1 edge + END_OPTION

    def test_valid_worker_mask_dtype(self, env):
        env.reset()
        env.select_region(0)
        mask = env.valid_worker_action_mask()
        assert mask.dtype == bool

    def test_end_option_always_valid(self, env):
        env.reset()
        env.select_region(0)
        mask = env.valid_worker_action_mask()
        assert mask[-1], "END_OPTION (last) must always be valid"

    def test_worker_step_without_option_raises(self, env):
        env.reset()
        # No active option
        with pytest.raises(RuntimeError):
            env.worker_step(0)

    def test_worker_step_returns_five_values(self, env):
        env.reset()
        env.select_region(0)
        result = env.worker_step(0)
        assert len(result) == 5

    def test_worker_step_obs_is_dict(self, env):
        env.reset()
        env.select_region(0)
        obs, reward, terminated, truncated, info = env.worker_step(0)
        assert isinstance(obs, dict)

    def test_worker_step_reward_float(self, env):
        env.reset()
        env.select_region(0)
        obs, reward, terminated, truncated, info = env.worker_step(0)
        assert isinstance(reward, float)

    def test_worker_step_info_is_dict(self, env):
        env.reset()
        env.select_region(0)
        obs, reward, terminated, truncated, info = env.worker_step(0)
        assert isinstance(info, dict)

    def test_end_option_via_worker_step(self, env):
        env.reset()
        env.select_region(0)
        mask = env.valid_worker_action_mask()
        end_action = len(mask) - 1
        obs, reward, terminated, truncated, info = env.worker_step(end_action)
        # After END_OPTION, option should be inactive (unless episode also ended)
        assert not env.option_state.active or terminated or truncated

    def test_option_state_active_after_select(self, env):
        env.reset()
        env.select_region(1)
        assert env.option_state.active is True

    def test_option_state_region_id_correct(self, env):
        env.reset()
        env.select_region(3)
        assert env.option_state.region_id == 3


# ---------------------------------------------------------------------------
# Option state property tests
# ---------------------------------------------------------------------------

class TestOptionState:
    def test_option_not_active_at_start(self, env):
        env.reset()
        assert env.option_state.active is False

    def test_option_region_none_at_start(self, env):
        env.reset()
        assert env.option_state.region_id is None

    def test_option_active_after_select_region(self, env):
        env.reset()
        env.select_region(0)
        assert env.option_state.active is True

    def test_option_region_id_after_select(self, env):
        env.reset()
        env.select_region(2)
        assert env.option_state.region_id == 2

    def test_option_inactive_after_end(self, env):
        env.reset()
        env.select_region(0)
        mask = env.valid_worker_action_mask()
        end_action = len(mask) - 1
        env.worker_step(end_action)
        # After END_OPTION action, option_state.active should be False
        # (unless the episode itself terminated for another reason)
        assert not env.option_state.active or env._episode_done


# ---------------------------------------------------------------------------
# Full HRL episode loop
# ---------------------------------------------------------------------------

class TestHRLFullEpisode:
    def test_full_hrl_episode_no_crash(self):
        env = MockCircuitEnv(
            num_edges=16,
            num_regions=4,
            max_episode_steps=100,
            max_causal_forward_calls=200,
            max_manager_decisions=20,
            option_horizon=5,
            seed=99,
        )
        obs = env.reset()
        episode_done = False
        manager_steps = 0

        while not episode_done and manager_steps < 25:
            # Manager picks a region
            mgr_mask = env.valid_manager_action_mask()
            valid_regions = np.where(mgr_mask)[0]
            # Prefer non-STOP regions while we still have budget
            non_stop = valid_regions[valid_regions < env.num_regions]
            if len(non_stop) == 0 or manager_steps >= 15:
                region = env.num_regions  # STOP
            else:
                region = int(non_stop[0])

            obs = env.select_region(region)
            if env._episode_done:
                episode_done = True
                break

            # Worker runs until option done or episode done
            option_done = False
            worker_steps = 0
            while not option_done and not episode_done and worker_steps < 10:
                w_mask = env.valid_worker_action_mask()
                valid_w = np.where(w_mask)[0]
                action = int(valid_w[0])
                obs, reward, terminated, truncated, info = env.worker_step(action)
                option_done = not env.option_state.active
                episode_done = terminated or truncated or env._episode_done
                worker_steps += 1

            manager_steps += 1

        assert episode_done or manager_steps >= 25

    def test_hrl_episode_finalize(self):
        env = MockCircuitEnv(
            num_edges=8,
            num_regions=2,
            max_episode_steps=30,
            max_causal_forward_calls=50,
            max_manager_decisions=10,
            option_horizon=3,
            seed=5,
        )
        env.reset()
        # Run one option then stop
        env.select_region(0)
        w_mask = env.valid_worker_action_mask()
        env.worker_step(len(w_mask) - 1)  # END_OPTION immediately
        result = env.finalize_episode()
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.faithfulness <= 1.0

    def test_manager_decisions_increment(self):
        env = MockCircuitEnv(
            num_edges=8,
            num_regions=2,
            max_manager_decisions=10,
            option_horizon=3,
            seed=0,
        )
        env.reset()
        assert env.manager_decisions == 0
        env.select_region(0)
        assert env.manager_decisions == 1
        # End option and select again
        w_mask = env.valid_worker_action_mask()
        env.worker_step(len(w_mask) - 1)
        if not env._episode_done:
            env.select_region(1)
            assert env.manager_decisions == 2
