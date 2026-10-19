#!/usr/bin/env python
"""scripts/smoke_test.py
========================
Quick end-to-end smoke test that verifies the core pipeline works on
the MockCircuitEnv without requiring a GPU or real benchmark data.

Usage
-----
    python scripts/smoke_test.py [--mode flat|hrl|full]

Exit 0 on success, 1 on any failure.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------
# Make sure src/ is importable even without editable install
# ------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np


def _make_mock_env(num_edges: int = 32, num_regions: int = 4):
    from hrl_circuits.env.mock_env import MockCircuitEnv
    return MockCircuitEnv(
        num_edges=num_edges,
        num_regions=num_regions,
        ground_truth_edges={1, 3, 9, 17, 18},
        max_causal_forward_calls=200,
        option_horizon=4,
        max_episode_steps=64,
        seed=0,
    )


def _step_unpack(result):
    """Unpack step() which returns (obs, reward, terminated, truncated, info) or (obs, reward, done, info)."""
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    elif len(result) == 4:
        obs, reward, done, info = result
    else:
        raise ValueError(f"Unexpected step return length: {len(result)}")
    return obs, reward, done, info


def run_flat_episode(verbose: bool = True) -> dict:
    """Run one random-policy flat episode, using STOP action to terminate."""
    env = _make_mock_env()
    obs = env.reset(seed=42)
    done = False
    steps = 0
    total_reward = 0.0

    while not done:
        valid = env.valid_action_mask()
        valid_indices = np.where(valid)[0]
        action = int(np.random.choice(valid_indices))
        obs, reward, done, info = _step_unpack(env.step(action))
        total_reward += reward
        steps += 1
        # Safety: if only STOP is valid, take it
        if steps > 60:
            stop_action = len(valid) - 1
            obs, reward, done, info = _step_unpack(env.step(stop_action))
            total_reward += reward
            steps += 1
            break

    result = env.finalize_episode()
    faithfulness = getattr(result, "final_faithfulness", None) or getattr(result, "faithfulness", 0.0)
    if verbose:
        print(f"  [Flat] steps={steps} reward={total_reward:.4f} "
              f"faithfulness={faithfulness:.4f}")
    return {"steps": steps, "reward": total_reward, "faithfulness": faithfulness}


def run_hrl_episode(verbose: bool = True) -> dict:
    """Run one scripted HRL episode:
    Manager->region0, Worker prune x2, Worker END_OPTION -> (episode done, or Manager->STOP).
    """
    env = _make_mock_env()
    obs = env.reset(seed=42)

    manager_steps = 0
    total_reward = 0.0

    # --- Option 1: Manager selects region 0 ---
    valid_mgr = env.valid_manager_action_mask()
    available_regions = [i for i, v in enumerate(valid_mgr[:-1]) if v]
    if not available_regions:
        print("  [HRL] No available regions -- skipping.")
        return {}

    region_action = available_regions[0]
    env.select_region(region_action)
    manager_steps += 1
    if verbose:
        print(f"  [HRL] Manager selected region {region_action}")

    # --- Worker prunes 2 edges within the selected region ---
    worker_step_count = 0
    episode_done = env._episode_done
    for _ in range(2):
        if episode_done:
            break
        valid_wkr = env.valid_worker_action_mask()
        live_in_region = [i for i, v in enumerate(valid_wkr[:-1]) if v]
        if not live_in_region:
            break
        action = live_in_region[0]
        obs, reward, done, info = _step_unpack(env.worker_step(action))
        total_reward += reward
        worker_step_count += 1
        episode_done = done
        if verbose:
            print(f"  [HRL] Worker prune step={worker_step_count}, action={action}, "
                  f"reward={reward:.4f}, done={done}")

    # --- Worker END_OPTION (if episode not already done) ---
    if not episode_done:
        valid_wkr = env.valid_worker_action_mask()
        end_opt_idx = len(valid_wkr) - 1  # last index = END_OPTION
        obs, reward, done, info = _step_unpack(env.worker_step(end_opt_idx))
        total_reward += reward
        episode_done = done
        if verbose:
            print(f"  [HRL] Worker END_OPTION: reward={reward:.4f}, done={done}")

    # --- Manager STOP (if episode not already done after END_OPTION) ---
    if not episode_done:
        valid_mgr2 = env.valid_manager_action_mask()
        stop_mgr_idx = len(valid_mgr2) - 1  # last = STOP
        env.select_region(stop_mgr_idx)  # manager STOP
        manager_steps += 1
        episode_done = env._episode_done
        if verbose:
            print(f"  [HRL] Manager STOP, episode_done={episode_done}")

    result = env.finalize_episode()
    faithfulness = getattr(result, "final_faithfulness", None) or getattr(result, "faithfulness", 0.0)
    if verbose:
        print(f"  [HRL] manager_steps={manager_steps} total_reward={total_reward:.4f} "
              f"faithfulness={faithfulness:.4f}")
    return {
        "manager_steps": manager_steps,
        "reward": total_reward,
        "faithfulness": faithfulness,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="HRL Circuit Localization Smoke Test")
    parser.add_argument(
        "--mode",
        choices=["flat", "hrl", "full"],
        default="full",
        help="Which parts to test (default: full)",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs/dev") / f"smoke_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  HRL Circuits Smoke Test  [{args.mode}]")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}\n")

    results = {}
    errors = []

    if args.mode in ("flat", "full"):
        print(">> Flat random-policy episode...")
        try:
            np.random.seed(0)
            results["flat"] = run_flat_episode(verbose=args.verbose)
            print("  [OK] Flat episode OK\n")
        except Exception as e:
            print(f"  [FAIL] Flat episode FAILED: {e}")
            traceback.print_exc()
            errors.append(f"flat: {e}")

    if args.mode in ("hrl", "full"):
        print(">> HRL scripted episode...")
        try:
            results["hrl"] = run_hrl_episode(verbose=args.verbose)
            print("  [OK] HRL episode OK\n")
        except Exception as e:
            print(f"  [FAIL] HRL episode FAILED: {e}")
            traceback.print_exc()
            errors.append(f"hrl: {e}")

    # Save summary
    import json
    summary = {
        "timestamp": timestamp,
        "mode": args.mode,
        "results": results,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    if errors:
        print(f"  RESULT: FAIL  ({len(errors)} error(s))")
        for err in errors:
            print(f"    - {err}")
        print(f"{'='*60}\n")
        return 1
    else:
        print(f"  RESULT: PASS  (all checks succeeded)")
        print(f"  Summary saved to: {output_dir}/summary.json")
        print(f"{'='*60}\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
