"""experiments/runner.py
=======================
Coordinates individual training/baseline runs and result collection.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@dataclass
class RunResult:
    run_id: str
    method: str          # 'flat_ppo' | 'hrl_ppo' | 'random' | 'greedy' | ...
    seed: int
    faithfulness: float
    sparsity: float
    eval_calls_used: int
    wall_clock_seconds: float
    output_dir: str
    status: str          # 'completed' | 'failed'
    error_msg: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "RunResult":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _make_env(config: dict, seed: int):
    """Create MockCircuitEnv from config dict."""
    from hrl_circuits.env.mock_env import MockCircuitEnv
    return MockCircuitEnv(
        num_edges=config.get("num_edges", 32),
        num_regions=config.get("num_regions", 4),
        max_causal_forward_calls=config.get("max_causal_forward_calls", 200),
        max_episode_steps=config.get("max_episode_steps", 64),
        option_horizon=config.get("option_horizon", 4),
        seed=seed,
    )


def _run_random_episode(env) -> dict:
    """Run one random-policy episode. Returns stats dict."""
    import numpy as np
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        valid = env.valid_action_mask()
        action = int(np.random.choice(np.where(valid)[0]))
        result = env.step(action)
        obs, reward = result[0], result[1]
        terminated, truncated = result[2], result[3]
        done = terminated or truncated
        total_reward += reward
        steps += 1
    eval_res = env.finalize_episode()
    return {
        "reward": total_reward,
        "faithfulness": float(getattr(eval_res, "faithfulness", 0.0)),
        "sparsity": float(getattr(eval_res, "sparsity", 0.0)),
        "eval_calls_used": int(getattr(eval_res, "eval_calls_used", 0)),
        "steps": steps,
    }


class ExperimentRunner:
    """Coordinates training and baseline runs."""

    def __init__(self, registry=None):
        self.registry = registry

    def run_flat_experiment(self, config: dict, seed: int, output_dir: str) -> RunResult:
        """Run flat PPO training and return RunResult."""
        import subprocess
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_flat_ppo_seed{seed}"
        out_dir = Path(output_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            script = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "train_flat.py"
            cmd = [
                sys.executable, str(script),
                "--seed", str(seed),
                "--num-edges", str(config.get("num_edges", 32)),
                "--total-steps", str(config.get("total_steps", 5000)),
                "--output-dir", str(out_dir),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = time.time() - t0
            # Try to read results
            results_file = out_dir / "results.json"
            if results_file.exists():
                data = json.loads(results_file.read_text(encoding="utf-8"))
                faithfulness = data.get("final_faithfulness", 0.0)
                sparsity = data.get("final_sparsity", 0.0)
                eval_calls = data.get("eval_calls_used", 0)
            else:
                faithfulness = sparsity = 0.0
                eval_calls = 0
            result = RunResult(
                run_id=run_id, method="flat_ppo", seed=seed,
                faithfulness=faithfulness, sparsity=sparsity,
                eval_calls_used=eval_calls, wall_clock_seconds=elapsed,
                output_dir=str(out_dir), status="completed",
            )
        except Exception as e:
            result = RunResult(
                run_id=run_id, method="flat_ppo", seed=seed,
                faithfulness=0.0, sparsity=0.0, eval_calls_used=0,
                wall_clock_seconds=time.time() - t0,
                output_dir=str(out_dir), status="failed", error_msg=str(e),
            )
        result.save(str(out_dir / "run_result.json"))
        if self.registry is not None:
            self.registry.register(result)
        return result

    def run_hrl_experiment(self, config: dict, seed: int, output_dir: str) -> RunResult:
        """Run HRL PPO training and return RunResult."""
        import subprocess
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_hrl_ppo_seed{seed}"
        out_dir = Path(output_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            script = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "train_hrl.py"
            cmd = [
                sys.executable, str(script),
                "--seed", str(seed),
                "--num-edges", str(config.get("num_edges", 32)),
                "--total-episodes", str(config.get("total_episodes", 100)),
                "--output-dir", str(out_dir),
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = time.time() - t0
            results_file = out_dir / "results.json"
            if results_file.exists():
                data = json.loads(results_file.read_text(encoding="utf-8"))
                faithfulness = data.get("final_faithfulness", 0.0)
                sparsity = data.get("final_sparsity", 0.0)
                eval_calls = data.get("eval_calls_used", 0)
            else:
                faithfulness = sparsity = 0.0
                eval_calls = 0
            result = RunResult(
                run_id=run_id, method="hrl_ppo", seed=seed,
                faithfulness=faithfulness, sparsity=sparsity,
                eval_calls_used=eval_calls, wall_clock_seconds=elapsed,
                output_dir=str(out_dir), status="completed",
            )
        except Exception as e:
            result = RunResult(
                run_id=run_id, method="hrl_ppo", seed=seed,
                faithfulness=0.0, sparsity=0.0, eval_calls_used=0,
                wall_clock_seconds=time.time() - t0,
                output_dir=str(out_dir), status="failed", error_msg=str(e),
            )
        result.save(str(out_dir / "run_result.json"))
        if self.registry is not None:
            self.registry.register(result)
        return result

    def run_baseline(
        self, baseline_name: str, config: dict, seed: int, output_dir: str
    ) -> RunResult:
        """Run a baseline policy episode and return RunResult."""
        import numpy as np
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{baseline_name}_seed{seed}"
        out_dir = Path(output_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            env = _make_env(config, seed)
            stats = _run_random_episode(env)
            elapsed = time.time() - t0
            result = RunResult(
                run_id=run_id, method=baseline_name, seed=seed,
                faithfulness=stats["faithfulness"], sparsity=stats["sparsity"],
                eval_calls_used=stats["eval_calls_used"],
                wall_clock_seconds=elapsed,
                output_dir=str(out_dir), status="completed",
            )
        except Exception as e:
            result = RunResult(
                run_id=run_id, method=baseline_name, seed=seed,
                faithfulness=0.0, sparsity=0.0, eval_calls_used=0,
                wall_clock_seconds=time.time() - t0,
                output_dir=str(out_dir), status="failed", error_msg=str(e),
            )
        result.save(str(out_dir / "run_result.json"))
        if self.registry is not None:
            self.registry.register(result)
        return result
