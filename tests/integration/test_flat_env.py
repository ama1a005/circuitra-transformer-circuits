"""
Integration tests for the MockCircuitEnv flat (CircuitEnv) interface.

Tests verify:
- reset() returns a valid obs dict
- step() returns 5-tuple (obs, reward, terminated, truncated, info)
- valid_action_mask() shape and dtype
- STOP action terminates episode
- eval_counter tracks calls
- finalize_episode() returns EvalResult with expected fields
- _episode_done flag is set correctly (no .done property)
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
        max_causal_forward_calls=100,
        max_episode_steps=50,
        seed=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFlatEnvReset:
    def test_reset_returns_dict(self, env):
        obs = env.reset()
        assert isinstance(obs, dict), "reset() must return a dict"

    def test_obs_contains_mask(self, env):
        obs = env.reset()
        assert "mask" in obs
        assert len(obs["mask"]) == 16

    def test_mask_all_ones_on_reset(self, env):
        obs = env.reset()
        np.testing.assert_array_equal(obs["mask"], np.ones(16, dtype=np.float32))

    def test_episode_done_false_after_reset(self, env):
        env.reset()
        assert env._episode_done is False

    def test_reset_with_seed(self, env):
        obs1 = env.reset(seed=42)
        obs2 = env.reset(seed=42)
        np.testing.assert_array_equal(obs1["mask"], obs2["mask"])


class TestFlatEnvStep:
    def test_step_returns_five_values(self, env):
        env.reset()
        result = env.step(0)
        assert len(result) == 5, "step() must return (obs, reward, terminated, truncated, info)"

    def test_step_obs_is_dict(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, dict)

    def test_step_reward_is_float(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, float)

    def test_step_terminated_and_truncated_are_bool(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_info_is_dict(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(info, dict)

    def test_step_toggles_edge(self, env):
        env.reset()
        # Toggle edge 0 (initially included)
        obs, _, _, _, _ = env.step(0)
        assert obs["mask"][0] == 0.0, "Edge 0 should be pruned after toggling"

    def test_stop_action_terminates(self, env):
        env.reset()
        stop_action = env.num_edges  # STOP
        obs, reward, terminated, truncated, info = env.step(stop_action)
        assert terminated or truncated, "STOP action should end episode"

    def test_episode_done_set_after_stop(self, env):
        env.reset()
        env.step(env.num_edges)  # STOP
        assert env._episode_done is True

    def test_eval_counter_increments(self, env):
        env.reset()
        calls_before = env.eval_counter.used
        env.step(0)
        assert env.eval_counter.used > calls_before

    def test_multiple_steps_without_error(self, env):
        env.reset()
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(i)
            if terminated or truncated:
                break

    def test_step_after_done_raises(self, env):
        env.reset()
        env.step(env.num_edges)  # STOP
        with pytest.raises(RuntimeError):
            env.step(0)


class TestFlatEnvActionMask:
    def test_valid_action_mask_shape(self, env):
        env.reset()
        mask = env.valid_action_mask()
        assert mask.shape == (env.num_edges + 1,)

    def test_valid_action_mask_dtype(self, env):
        env.reset()
        mask = env.valid_action_mask()
        assert mask.dtype == bool

    def test_stop_always_valid(self, env):
        env.reset()
        mask = env.valid_action_mask()
        assert mask[-1], "STOP action (last) must always be valid"

    def test_all_edges_valid_at_start(self, env):
        env.reset()
        mask = env.valid_action_mask()
        assert mask[:env.num_edges].all()


class TestFlatEnvFinalize:
    def test_finalize_returns_eval_result(self, env):
        env.reset()
        result = env.finalize_episode()
        assert isinstance(result, EvalResult)

    def test_finalize_faithfulness_in_range(self, env):
        env.reset()
        result = env.finalize_episode()
        assert 0.0 <= result.faithfulness <= 1.0

    def test_finalize_sparsity_in_range(self, env):
        env.reset()
        result = env.finalize_episode()
        assert 0.0 <= result.sparsity <= 1.0

    def test_finalize_num_edges_included(self, env):
        env.reset()
        result = env.finalize_episode()
        assert 0 <= result.num_edges_included <= env.num_edges

    def test_finalize_num_edges_total(self, env):
        env.reset()
        result = env.finalize_episode()
        assert result.num_edges_total == env.num_edges

    def test_finalize_eval_calls_used_positive(self, env):
        env.reset()
        result = env.finalize_episode()
        # eval_calls_used is from the last internal evaluation call, >= 1
        assert result.eval_calls_used >= 1

    def test_finalize_sets_episode_done(self, env):
        env.reset()
        env.finalize_episode()
        assert env._episode_done is True


class TestFlatEnvFullEpisode:
    def test_full_episode_no_crash(self):
        env = MockCircuitEnv(
            num_edges=8,
            num_regions=2,
            max_episode_steps=20,
            max_causal_forward_calls=50,
            seed=7,
        )
        obs = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            action_mask = env.valid_action_mask()
            valid = np.where(action_mask)[0]
            action = int(valid[0])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
            if steps > 200:
                break
        assert done, "Episode must terminate"
        assert isinstance(total_reward, float)

    def test_faithfulness_drops_when_gt_pruned(self):
        gt = {0, 1}
        env = MockCircuitEnv(
            num_edges=8,
            num_regions=2,
            ground_truth_edges=gt,
            severity=1.0,
            max_episode_steps=50,
            seed=0,
        )
        env.reset()
        # Prune both GT edges
        env.step(0)
        env.step(1)
        result = env.finalize_episode()
        assert result.faithfulness == 0.0, "All GT pruned => faithfulness=0"
