"""
hrl_circuits.cli
================

Click-based command-line interface for the HRL Circuit Localization project.

Commands
--------
* ``train``    – run the full HRL training loop
* ``eval``     – evaluate faithfulness of a saved circuit
* ``export``   – export an edge mask as a JSON file
* ``provenance`` – print or save a reproducibility manifest
* ``config``   – config file utilities (``validate`` sub-command)

Usage
-----
::

    hrl-circuits --help
    hrl-circuits train --config config/ioi.yaml --seed 42
    hrl-circuits eval  --config config/ioi.yaml --checkpoint checkpoints/best.pt
    hrl-circuits export --checkpoint checkpoints/best.pt --output circuits/mask.json
    hrl-circuits provenance --config config/ioi.yaml --output manifests/run.json
    hrl-circuits config validate --config config/ioi.yaml
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from hrl_circuits import __version__
from hrl_circuits.config import load_config
from hrl_circuits.logging_utils import setup_logging
from hrl_circuits.provenance import ReproducibilityManifest
from hrl_circuits.seed import seed_everything


# ─────────────────────────────────────────────────────────────────────────────
# Shared options
# ─────────────────────────────────────────────────────────────────────────────


def _config_option(required: bool = True) -> click.Option:
    return click.option(
        "--config",
        "-c",
        type=click.Path(exists=True, dir_okay=False, readable=True),
        required=required,
        help="Path to the YAML configuration file.",
    )


def _seed_option() -> click.Option:
    return click.option(
        "--seed",
        type=int,
        default=None,
        show_default=False,
        help="Override the random seed from config.",
    )


def _log_level_option() -> click.Option:
    return click.option(
        "--log-level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
        default="INFO",
        show_default=True,
        help="Logging verbosity level.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Root group
# ─────────────────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(version=__version__, prog_name="hrl-circuits")
@click.pass_context
def main(ctx: click.Context) -> None:
    """HRL Circuit Localization – discover minimal faithful circuits in LLMs
    using Hierarchical Reinforcement Learning.

    Run any sub-command with ``--help`` for detailed options.
    """
    ctx.ensure_object(dict)


# ─────────────────────────────────────────────────────────────────────────────
# train
# ─────────────────────────────────────────────────────────────────────────────


@main.command("train")
@_config_option()
@_seed_option()
@_log_level_option()
@click.option(
    "--checkpoint",
    type=click.Path(),
    default=None,
    help="Resume training from this checkpoint.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Override the output directory from config.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate config and environment without running training.",
)
@click.pass_context
def train_cmd(
    ctx: click.Context,
    config: str,
    seed: int | None,
    log_level: str,
    checkpoint: str | None,
    output_dir: str | None,
    dry_run: bool,
) -> None:
    """Run the HRL training loop.

    Loads the configuration from CONFIG, seeds all random number generators,
    initialises the environment and agent, and starts the training loop.

    \b
    Examples:
        hrl-circuits train --config config/ioi.yaml
        hrl-circuits train --config config/ioi.yaml --seed 7 --log-level DEBUG
        hrl-circuits train --config config/ioi.yaml --dry-run
    """
    # ── load config ───────────────────────────────────────────────────────────
    overrides: dict[str, Any] = {}
    if seed is not None:
        overrides["training.seed"] = seed
    if output_dir is not None:
        overrides["output.output_dir"] = output_dir
    if checkpoint is not None:
        overrides["training.resume_checkpoint"] = checkpoint

    cfg = load_config(config, overrides=overrides or None)

    # ── logging ───────────────────────────────────────────────────────────────
    out_root = Path(cfg.output.output_dir)
    log = setup_logging(
        level=log_level,
        log_dir=str(out_root / cfg.logging.log_dir),
        json_file=True,
    )

    log.info("hrl-circuits v%s  |  train", __version__)
    log.info("Config: %s", config)
    log.info("Task:   %s  |  Model: %s", cfg.benchmark.task, cfg.benchmark.model_name)
    log.info("Agent:  %s", cfg.training.agent_type)

    # ── seed ──────────────────────────────────────────────────────────────────
    effective_seed = cfg.training.seed
    seed_everything(effective_seed, deterministic=cfg.training.deterministic)
    log.info("Seed: %d  (deterministic=%s)", effective_seed, cfg.training.deterministic)

    if dry_run:
        log.info("--dry-run: configuration valid. Exiting without training.")
        click.echo("✓ Config validated successfully. Dry-run complete.")
        return

    # ── training ──────────────────────────────────────────────────────────────
    log.info("Starting training for %d timesteps…", cfg.training.total_timesteps)
    click.echo(
        f"[hrl-circuits] Training {cfg.training.agent_type} agent on "
        f"{cfg.benchmark.task} ({cfg.benchmark.model_name})…"
    )

    # Training loop is implemented in hrl_circuits.training (future module).
    # This stub raises a clear error directing developers to implement it.
    try:
        from hrl_circuits.training import run_training  # type: ignore[import]
        run_training(cfg)
    except ImportError:
        log.error(
            "hrl_circuits.training module not yet implemented. "
            "Please implement src/hrl_circuits/training.py."
        )
        click.echo(
            "ERROR: Training module not yet implemented.\n"
            "Implement src/hrl_circuits/training.py and expose run_training(cfg).",
            err=True,
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# eval
# ─────────────────────────────────────────────────────────────────────────────


@main.command("eval")
@_config_option()
@_seed_option()
@_log_level_option()
@click.option(
    "--checkpoint",
    "-k",
    type=click.Path(exists=True),
    required=True,
    help="Path to the model checkpoint to evaluate.",
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"], case_sensitive=False),
    default="test",
    show_default=True,
    help="Dataset split to evaluate on.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Write evaluation results to this JSON file.",
)
@click.pass_context
def eval_cmd(
    ctx: click.Context,
    config: str,
    seed: int | None,
    log_level: str,
    checkpoint: str,
    split: str,
    output: str | None,
) -> None:
    """Evaluate the faithfulness of a saved circuit checkpoint.

    Loads the circuit mask from CHECKPOINT, runs causal evaluation on SPLIT,
    and reports faithfulness and task metrics.

    \b
    Examples:
        hrl-circuits eval --config config/ioi.yaml --checkpoint checkpoints/best.pt
        hrl-circuits eval --config config/ioi.yaml --checkpoint checkpoints/best.pt \\
            --split val --output results/eval_val.json
    """
    overrides: dict[str, Any] = {}
    if seed is not None:
        overrides["training.seed"] = seed

    cfg = load_config(config, overrides=overrides or None)
    log = setup_logging(level=log_level)
    log.info("hrl-circuits v%s  |  eval", __version__)
    log.info("Checkpoint: %s", checkpoint)
    log.info("Split:      %s", split)

    effective_seed = seed if seed is not None else cfg.training.seed
    seed_everything(effective_seed, deterministic=cfg.training.deterministic)

    try:
        from hrl_circuits.evaluation import run_evaluation  # type: ignore[import]
        result = run_evaluation(cfg, checkpoint_path=checkpoint, split=split)
        click.echo(json.dumps(result, indent=2))
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
            log.info("Evaluation results written to %s", output)
    except ImportError:
        log.error("hrl_circuits.evaluation module not yet implemented.")
        click.echo("ERROR: Evaluation module not yet implemented.", err=True)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# export
# ─────────────────────────────────────────────────────────────────────────────


@main.command("export")
@click.option(
    "--checkpoint",
    "-k",
    type=click.Path(exists=True),
    required=True,
    help="Checkpoint file containing the circuit mask.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Destination JSON file for the exported edge mask.",
)
@_log_level_option()
@click.pass_context
def export_cmd(
    ctx: click.Context,
    checkpoint: str,
    output: str,
    log_level: str,
) -> None:
    """Export a circuit edge mask from a checkpoint to a JSON file.

    The exported JSON lists retained edge IDs alongside their source/destination
    node IDs, component types, and edge types.

    \b
    Examples:
        hrl-circuits export --checkpoint checkpoints/best.pt --output circuits/ioi.json
    """
    log = setup_logging(level=log_level)
    log.info("hrl-circuits v%s  |  export", __version__)
    log.info("Checkpoint: %s  →  %s", checkpoint, output)

    try:
        from hrl_circuits.export import export_mask  # type: ignore[import]
        export_mask(checkpoint_path=checkpoint, output_path=output)
        click.echo(f"✓ Circuit mask exported to {output}")
    except ImportError:
        log.error("hrl_circuits.export module not yet implemented.")
        click.echo("ERROR: Export module not yet implemented.", err=True)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# provenance
# ─────────────────────────────────────────────────────────────────────────────


@main.command("provenance")
@_config_option(required=False)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Seed to record in the manifest.",
)
@click.option(
    "--task",
    type=str,
    default="",
    help="Task name to record.",
)
@click.option(
    "--graph-hash",
    type=str,
    default="",
    help="Candidate graph hash to record.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Save manifest to this JSON file (prints to stdout if omitted).",
)
@_log_level_option()
def provenance_cmd(
    config: str | None,
    seed: int,
    task: str,
    graph_hash: str,
    output: str | None,
    log_level: str,
) -> None:
    """Capture and display a reproducibility manifest.

    The manifest encodes the git commit, config hash, hardware info, package
    versions, and all hyper-parameters needed to reproduce the run.

    \b
    Examples:
        hrl-circuits provenance --config config/ioi.yaml
        hrl-circuits provenance --config config/ioi.yaml --output manifests/run.json
    """
    log = setup_logging(level=log_level)

    prefilter: dict[str, Any] = {}
    reward: dict[str, Any] = {}
    option_horizon = 0
    eval_budget = 0
    benchmark_revision = ""
    model_revision = "main"

    if config:
        try:
            cfg = load_config(config)
            prefilter = {
                "attribution_method": cfg.graph.attribution_method,
                "edge_threshold": cfg.graph.edge_threshold,
                "max_edges": cfg.graph.max_edges,
            }
            reward = {
                "alpha_faithfulness": cfg.reward.alpha_faithfulness,
                "beta_sparsity": cfg.reward.beta_sparsity,
                "gamma_cost": cfg.reward.gamma_cost,
                "guard_threshold": cfg.reward.guard_threshold,
                "lambda_guard": cfg.reward.lambda_guard,
            }
            option_horizon = cfg.hierarchical.option_horizon
            eval_budget = cfg.evaluator.evaluation_budget
            benchmark_revision = cfg.benchmark.benchmark_revision
            model_revision = cfg.benchmark.model_revision
            if not task:
                task = cfg.benchmark.task
            if not seed:
                seed = cfg.training.seed
        except Exception as exc:
            log.warning("Could not load config for manifest: %s", exc)

    manifest = ReproducibilityManifest.capture(
        config_path=config or "",
        seed=seed,
        task=task,
        candidate_graph_hash=graph_hash,
        prefilter_settings=prefilter,
        reward_settings=reward,
        option_horizon=option_horizon,
        evaluation_budget=eval_budget,
        benchmark_revision=benchmark_revision,
        model_revision=model_revision,
    )

    payload = json.dumps(manifest.to_dict(), indent=2, default=str)

    if output:
        saved = manifest.save(output)
        click.echo(f"✓ Manifest saved to {saved}")
    else:
        click.echo(payload)


# ─────────────────────────────────────────────────────────────────────────────
# config group
# ─────────────────────────────────────────────────────────────────────────────


@main.group("config")
def config_group() -> None:
    """Config file utilities."""


@config_group.command("validate")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the YAML configuration file.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print the parsed config as JSON after validation.",
)
def config_validate(config: str, verbose: bool) -> None:
    """Validate a YAML configuration file against the HRLConfig schema.

    Exits with code 0 on success, 1 on validation failure.

    \b
    Examples:
        hrl-circuits config validate --config config/ioi.yaml
        hrl-circuits config validate --config config/ioi.yaml --verbose
    """
    try:
        cfg = load_config(config)
        click.echo(f"✓ Config is valid: {config}")
        if verbose:
            click.echo(json.dumps(cfg.to_dict(), indent=2, default=str))
    except Exception as exc:
        click.echo(f"✗ Config validation failed:\n{exc}", err=True)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
