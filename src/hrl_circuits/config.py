"""
hrl_circuits.config
===================

Pydantic-based configuration schema for the HRL Circuit Localization
project.  The full configuration is defined as a hierarchy of nested
``BaseModel`` sub-configs, each corresponding to a logical section of
the YAML config file (spec section 7).

Usage
-----
::

    from hrl_circuits.config import load_config, HRLConfig

    cfg: HRLConfig = load_config("config/my_run.yaml")
    print(cfg.project.name)
    print(cfg.ppo.lr_actor)

The top-level :func:`load_config` function handles YAML parsing,
environment-variable substitution (``${ENV_VAR}`` syntax), and
full Pydantic validation with informative error messages.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Helper: environment variable substitution
# ─────────────────────────────────────────────────────────────────────────────

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute ``${VAR}`` references with environment values.

    Parameters
    ----------
    obj : Any
        A parsed YAML object (dict, list, str, or scalar).

    Returns
    -------
    Any
        The same structure with all ``${VAR}`` strings replaced.

    Raises
    ------
    KeyError
        If a referenced environment variable is not set.
    """
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    if isinstance(obj, str):
        def _replace(match: re.Match[str]) -> str:
            var = match.group(1)
            if var not in os.environ:
                raise KeyError(
                    f"Config references environment variable ${{{var}}} "
                    f"but it is not set."
                )
            return os.environ[var]
        return _ENV_VAR_RE.sub(_replace, obj)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────


class ProjectConfig(BaseModel):
    """Top-level project / experiment metadata.

    Attributes
    ----------
    name : str
        Short project name, used in log prefixes and output directory names.
    experiment_id : str
        Unique experiment identifier (e.g. ``"ioi_gpt2_v1"``).  Drives the
        output directory structure.
    notes : str
        Optional free-text notes about the experiment.
    tags : list[str]
        Optional tags for filtering experiments in dashboards.
    """

    name: str = Field(..., description="Short project name.")
    experiment_id: str = Field(..., description="Unique experiment identifier.")
    notes: str = Field("", description="Free-text experiment notes.")
    tags: list[str] = Field(default_factory=list, description="Experiment tags.")


class BenchmarkConfig(BaseModel):
    """Benchmark / task configuration.

    Attributes
    ----------
    task : str
        Task name, e.g. ``"ioi"``, ``"gtq"``, ``"docstring"``.
    model_name : str
        TransformerLens model name, e.g. ``"gpt2-small"``.
    dataset_path : str | None
        Optional path to a local dataset file; uses HuggingFace Hub if None.
    split_ratios : list[float]
        Train / validation / test split ratios.  Must sum to 1.0.
    num_examples : int
        Total number of examples to use from the dataset (−1 = all).
    seed : int
        Dataset-level random seed for splitting and shuffling.
    benchmark_revision : str
        Dataset/benchmark version/revision tag for provenance.
    model_revision : str
        Model checkpoint revision tag for provenance.
    cache_dir : str | None
        Optional path for HuggingFace cache.
    """

    task: str = Field(..., description="Benchmark task name.")
    model_name: str = Field(..., description="TransformerLens model name.")
    dataset_path: str | None = Field(None, description="Local dataset path.")
    split_ratios: list[float] = Field(
        default=[0.7, 0.15, 0.15],
        description="Train/val/test split ratios (must sum to 1.0).",
    )
    num_examples: int = Field(-1, description="Number of examples to use; -1 = all.")
    seed: int = Field(42, description="Dataset random seed.")
    benchmark_revision: str = Field("", description="Benchmark version tag.")
    model_revision: str = Field("main", description="Model checkpoint revision.")
    cache_dir: str | None = Field(None, description="HuggingFace cache directory.")

    @field_validator("split_ratios")
    @classmethod
    def _validate_split_ratios(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("split_ratios must have exactly 3 values (train/val/test).")
        total = sum(v)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split_ratios must sum to 1.0, got {total:.6f}."
            )
        if any(r < 0 for r in v):
            raise ValueError("split_ratios values must be non-negative.")
        return v


class GraphConfig(BaseModel):
    """Candidate-graph construction settings.

    Attributes
    ----------
    attribution_method : str
        Edge-scoring method for graph construction.
        One of: ``"activation_patching"``, ``"integrated_gradients"``,
        ``"attention_attribution"``, ``"eap_ig"``.
    edge_threshold : float
        Attribution score threshold below which edges are excluded from the
        candidate graph (absolute value).
    max_edges : int
        Maximum number of edges to include; the top-k by attribution score
        are kept if the candidate set exceeds this limit (−1 = no limit).
    include_mlp : bool
        Whether to include MLP edges in the candidate graph.
    include_attention : bool
        Whether to include attention head edges.
    include_residual : bool
        Whether to include residual-stream edges.
    graph_cache_dir : str | None
        Directory to cache pre-computed candidate graphs.
    """

    attribution_method: Literal[
        "activation_patching",
        "integrated_gradients",
        "attention_attribution",
        "eap_ig",
    ] = Field("activation_patching", description="Edge-scoring method.")
    edge_threshold: float = Field(
        0.01, ge=0.0, description="Attribution score threshold."
    )
    max_edges: int = Field(-1, description="Max edges; -1 = no limit.")
    include_mlp: bool = Field(True, description="Include MLP edges.")
    include_attention: bool = Field(True, description="Include attention edges.")
    include_residual: bool = Field(True, description="Include residual edges.")
    graph_cache_dir: str | None = Field(None, description="Graph cache directory.")


class EvaluatorConfig(BaseModel):
    """Faithfulness evaluator settings.

    Attributes
    ----------
    eval_method : str
        Evaluation method; one of ``"activation_patching"``,
        ``"zero_ablation"``, ``"mean_ablation"``, ``"resample_ablation"``.
    eval_batch_size : int
        Number of examples per evaluation batch.
    num_eval_examples : int
        Number of examples to evaluate faithfulness on (−1 = all).
    evaluation_budget : int
        Maximum total causal evaluation invocations per episode.
    faithfulness_metric : str
        Which metric to report as the primary faithfulness score.
        E.g. ``"logit_diff_normalised"``, ``"probability_ratio"``.
    device : str
        Torch device string, e.g. ``"cuda"``, ``"cpu"``, ``"cuda:1"``.
    use_cache : bool
        Whether to cache clean/corrupted activations between eval calls.
    """

    eval_method: Literal[
        "activation_patching", "zero_ablation", "mean_ablation", "resample_ablation"
    ] = Field("activation_patching", description="Evaluation method.")
    eval_batch_size: int = Field(32, gt=0, description="Evaluation batch size.")
    num_eval_examples: int = Field(-1, description="Examples per eval; -1 = all.")
    evaluation_budget: int = Field(
        200, gt=0, description="Max causal eval invocations per episode."
    )
    faithfulness_metric: str = Field(
        "logit_diff_normalised", description="Primary faithfulness metric."
    )
    device: str = Field("cuda", description="Torch device.")
    use_cache: bool = Field(True, description="Cache activations between eval calls.")


class RewardConfig(BaseModel):
    """Reward function hyper-parameters.

    The total reward at each step is::

        r = alpha * Δfaith + beta * Δsparsity - gamma * Δcost
            - lambda_guard * I[faith < guard_threshold]
            + terminal_bonus * I[terminal]

    Attributes
    ----------
    alpha_faithfulness : float
        Scaling coefficient for the faithfulness-change component (≥ 0).
    beta_sparsity : float
        Scaling coefficient for the sparsity-change component (≥ 0).
    gamma_cost : float
        Scaling coefficient for the cost-penalty component (≥ 0).
    guard_threshold : float
        Faithfulness threshold below which the guard penalty fires.
        Must be in (0, 1].
    lambda_guard : float
        Magnitude of the guard penalty (≥ 0).
    terminal_bonus : float
        Reward bonus applied at episode termination when faithfulness
        exceeds ``guard_threshold``.
    terminal_penalty : float
        Penalty applied at episode termination when faithfulness falls
        below ``guard_threshold``.
    normalise_sparsity : bool
        If True, Δsparsity is normalised by the total number of edges.
    """

    alpha_faithfulness: float = Field(
        1.0, ge=0.0, description="Faithfulness reward coefficient."
    )
    beta_sparsity: float = Field(
        0.3, ge=0.0, description="Sparsity reward coefficient."
    )
    gamma_cost: float = Field(
        0.01, ge=0.0, description="Cost penalty coefficient."
    )
    guard_threshold: float = Field(
        0.5, gt=0.0, le=1.0, description="Faithfulness guard threshold."
    )
    lambda_guard: float = Field(
        1.0, ge=0.0, description="Guard penalty magnitude."
    )
    terminal_bonus: float = Field(
        2.0, description="Terminal reward when faithfulness ≥ guard threshold."
    )
    terminal_penalty: float = Field(
        -2.0, description="Terminal penalty when faithfulness < guard threshold."
    )
    normalise_sparsity: bool = Field(
        True, description="Normalise sparsity delta by total edge count."
    )


class PPOConfig(BaseModel):
    """Proximal Policy Optimisation hyper-parameters (shared by both levels).

    Attributes
    ----------
    lr_actor : float
        Learning rate for the actor (policy) network.
    lr_critic : float
        Learning rate for the critic (value) network.
    clip_epsilon : float
        PPO clipping parameter ε for the policy ratio.
    epochs_per_update : int
        Number of optimisation epochs per collected rollout.
    batch_size : int
        Mini-batch size for each optimisation step.
    gamma : float
        Discount factor γ for future rewards.
    gae_lambda : float
        GAE λ for advantage estimation.
    value_loss_coef : float
        Coefficient for the value loss term.
    entropy_coef : float
        Coefficient for the entropy bonus (encourages exploration).
    max_grad_norm : float
        Maximum gradient norm for clipping.
    rollout_steps : int
        Number of environment steps to collect per rollout.
    normalise_advantages : bool
        Whether to normalise advantages across the mini-batch.
    """

    lr_actor: float = Field(3e-4, gt=0.0, description="Actor learning rate.")
    lr_critic: float = Field(1e-3, gt=0.0, description="Critic learning rate.")
    clip_epsilon: float = Field(0.2, gt=0.0, description="PPO clip epsilon.")
    epochs_per_update: int = Field(4, gt=0, description="Epochs per rollout update.")
    batch_size: int = Field(64, gt=0, description="Mini-batch size.")
    gamma: float = Field(0.99, gt=0.0, le=1.0, description="Discount factor.")
    gae_lambda: float = Field(
        0.95, gt=0.0, le=1.0, description="GAE lambda."
    )
    value_loss_coef: float = Field(0.5, gt=0.0, description="Value loss coefficient.")
    entropy_coef: float = Field(0.01, ge=0.0, description="Entropy bonus coefficient.")
    max_grad_norm: float = Field(0.5, gt=0.0, description="Gradient clip norm.")
    rollout_steps: int = Field(2048, gt=0, description="Steps per rollout.")
    normalise_advantages: bool = Field(True, description="Normalise advantages.")


class FlatAgentConfig(BaseModel):
    """Configuration specific to the flat (single-level) PPO agent.

    Attributes
    ----------
    hidden_dim : int
        Hidden dimension of the actor/critic MLP.
    num_layers : int
        Number of hidden layers in the actor/critic MLP.
    activation : str
        Activation function: ``"relu"``, ``"tanh"``, or ``"gelu"``.
    edge_embedding_dim : int
        Embedding dimension for edge features.
    use_graph_encoder : bool
        Whether to use a GNN encoder for edge-feature aggregation.
    """

    hidden_dim: int = Field(256, gt=0, description="MLP hidden dimension.")
    num_layers: int = Field(2, gt=0, description="Number of hidden layers.")
    activation: Literal["relu", "tanh", "gelu"] = Field(
        "relu", description="Activation function."
    )
    edge_embedding_dim: int = Field(64, gt=0, description="Edge embedding dimension.")
    use_graph_encoder: bool = Field(
        False, description="Use GNN for edge-feature aggregation."
    )


class HierarchicalAgentConfig(BaseModel):
    """Configuration specific to the hierarchical (HRL) agent.

    Attributes
    ----------
    manager_hidden_dim : int
        Hidden dimension of the manager (macro-level) policy network.
    worker_hidden_dim : int
        Hidden dimension of the worker (micro-level) policy network.
    option_horizon : int
        Maximum number of worker steps per manager option.
    num_options : int | None
        Number of discrete options available to the manager.
        If None, derived from the number of graph regions.
    option_embedding_dim : int
        Embedding dimension for option (region) identifiers.
    manager_update_freq : int
        Number of worker steps between manager updates.
    intrinsic_reward_coef : float
        Coefficient for intrinsic (worker-level) reward from the manager.
    manager_entropy_coef : float
        Entropy bonus coefficient for the manager policy.
    worker_entropy_coef : float
        Entropy bonus coefficient for the worker policy.
    """

    manager_hidden_dim: int = Field(512, gt=0, description="Manager hidden dim.")
    worker_hidden_dim: int = Field(256, gt=0, description="Worker hidden dim.")
    option_horizon: int = Field(10, gt=0, description="Max worker steps per option.")
    num_options: int | None = Field(
        None, description="Number of discrete manager options."
    )
    option_embedding_dim: int = Field(32, gt=0, description="Option embedding dim.")
    manager_update_freq: int = Field(
        10, gt=0, description="Worker steps between manager updates."
    )
    intrinsic_reward_coef: float = Field(
        0.5, ge=0.0, description="Intrinsic reward coefficient."
    )
    manager_entropy_coef: float = Field(
        0.05, ge=0.0, description="Manager entropy bonus."
    )
    worker_entropy_coef: float = Field(
        0.01, ge=0.0, description="Worker entropy bonus."
    )


class TrainingConfig(BaseModel):
    """Overall training loop settings.

    Attributes
    ----------
    agent_type : str
        Which agent to train: ``"flat"`` or ``"hierarchical"``.
    total_timesteps : int
        Total number of environment steps to train for.
    seed : int
        Global random seed.
    eval_interval : int
        Evaluate on the validation split every this many timesteps.
    eval_episodes : int
        Number of episodes to run per validation evaluation.
    save_interval : int
        Save a checkpoint every this many timesteps.
    resume_checkpoint : str | None
        Path to a checkpoint to resume training from.
    deterministic : bool
        Whether to enforce deterministic CUDA operations.
    num_envs : int
        Number of parallel environments for rollout collection.
    max_episode_steps : int
        Maximum steps per episode before forced termination.
    """

    agent_type: Literal["flat", "hierarchical"] = Field(
        "hierarchical", description="Agent type."
    )
    total_timesteps: int = Field(
        1_000_000, gt=0, description="Total training timesteps."
    )
    seed: int = Field(42, description="Global random seed.")
    eval_interval: int = Field(
        10_000, gt=0, description="Eval frequency (timesteps)."
    )
    eval_episodes: int = Field(10, gt=0, description="Episodes per validation eval.")
    save_interval: int = Field(
        20_000, gt=0, description="Checkpoint frequency (timesteps)."
    )
    resume_checkpoint: str | None = Field(
        None, description="Path to resume checkpoint."
    )
    deterministic: bool = Field(True, description="Deterministic CUDA ops.")
    num_envs: int = Field(1, gt=0, description="Number of parallel environments.")
    max_episode_steps: int = Field(
        100, gt=0, description="Max steps per episode."
    )


class LoggingConfig(BaseModel):
    """Logging and monitoring settings.

    Attributes
    ----------
    level : str
        Python logging level: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``.
    log_dir : str
        Directory for log files.
    use_tensorboard : bool
        Write TensorBoard summaries.
    use_wandb : bool
        Use Weights & Biases for experiment tracking.
    wandb_project : str
        W&B project name.
    wandb_entity : str | None
        W&B entity (team/user); if None uses the default.
    log_interval : int
        Log training metrics every this many timesteps.
    trajectory_log_interval : int
        Save full trajectory logs every this many episodes.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging level."
    )
    log_dir: str = Field("logs", description="Log output directory.")
    use_tensorboard: bool = Field(False, description="Enable TensorBoard.")
    use_wandb: bool = Field(False, description="Enable W&B tracking.")
    wandb_project: str = Field("hrl-circuits", description="W&B project name.")
    wandb_entity: str | None = Field(None, description="W&B entity.")
    log_interval: int = Field(1000, gt=0, description="Logging frequency (steps).")
    trajectory_log_interval: int = Field(
        10, gt=0, description="Trajectory log frequency (episodes)."
    )


class OutputConfig(BaseModel):
    """Output / artefact directory settings.

    Attributes
    ----------
    output_dir : str
        Root output directory.  Checkpoints, masks, trajectories, and
        manifests are written under this directory.
    checkpoint_dir : str
        Sub-directory for model checkpoints (relative to ``output_dir``).
    mask_dir : str
        Sub-directory for circuit mask JSON files.
    trajectory_dir : str
        Sub-directory for episode trajectory files.
    manifest_dir : str
        Sub-directory for reproducibility manifests.
    result_dir : str
        Sub-directory for episode result JSON files.
    overwrite : bool
        If False, raise an error when ``output_dir`` already exists.
    """

    output_dir: str = Field("outputs", description="Root output directory.")
    checkpoint_dir: str = Field(
        "checkpoints", description="Checkpoint sub-directory."
    )
    mask_dir: str = Field("masks", description="Circuit mask sub-directory.")
    trajectory_dir: str = Field(
        "trajectories", description="Trajectory sub-directory."
    )
    manifest_dir: str = Field("manifests", description="Manifest sub-directory.")
    result_dir: str = Field("results", description="Result JSON sub-directory.")
    overwrite: bool = Field(
        False, description="Allow overwriting existing output directories."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level HRLConfig
# ─────────────────────────────────────────────────────────────────────────────


class HRLConfig(BaseModel):
    """Root configuration model for the HRL Circuit Localization project.

    Contains all sub-configuration sections corresponding to spec section 7.
    Instantiate via :func:`load_config` rather than directly.

    Attributes
    ----------
    project : ProjectConfig
        Project and experiment metadata.
    benchmark : BenchmarkConfig
        Benchmark / task configuration.
    graph : GraphConfig
        Candidate-graph construction settings.
    evaluator : EvaluatorConfig
        Faithfulness evaluator settings.
    reward : RewardConfig
        Reward function hyper-parameters.
    ppo : PPOConfig
        PPO optimiser settings shared by all agent levels.
    flat : FlatAgentConfig
        Settings specific to the flat PPO agent.
    hierarchical : HierarchicalAgentConfig
        Settings specific to the hierarchical HRL agent.
    training : TrainingConfig
        Training loop configuration.
    logging : LoggingConfig
        Logging and monitoring settings.
    output : OutputConfig
        Output directory configuration.
    """

    project: ProjectConfig
    benchmark: BenchmarkConfig
    graph: GraphConfig = Field(default_factory=GraphConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    flat: FlatAgentConfig = Field(default_factory=FlatAgentConfig)
    hierarchical: HierarchicalAgentConfig = Field(
        default_factory=HierarchicalAgentConfig
    )
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _cross_validate(self) -> "HRLConfig":
        """Perform cross-field validation after all fields are set."""
        # Ensure benchmark seed and training seed are consistent when both set
        # (training.seed is authoritative; benchmark.seed may differ for data)
        if self.training.agent_type == "flat" and self.hierarchical is not None:
            # Flat agent — hierarchical-specific params are present but ignored.
            pass
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full config to a plain nested dictionary."""
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# load_config
# ─────────────────────────────────────────────────────────────────────────────


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> HRLConfig:
    """Load, validate, and return a :class:`HRLConfig` from a YAML file.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file.
    overrides : dict[str, Any] | None
        Optional flat or nested dict of values that override the file content.
        Supports dot-notation keys, e.g. ``{"ppo.lr_actor": 1e-4}``.

    Returns
    -------
    HRLConfig
        A fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the YAML is invalid or fails Pydantic validation.
    KeyError
        If an ``${ENV_VAR}`` reference in the YAML is unset.

    Examples
    --------
    >>> cfg = load_config("config/ioi.yaml")
    >>> cfg.project.name
    'ioi_run'
    >>> cfg = load_config("config/ioi.yaml", overrides={"ppo.lr_actor": 1e-3})
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # Substitute environment variables
    raw = _substitute_env_vars(raw)

    # Apply overrides (dot-notation keys expand into nested dicts)
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            target = raw
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value

    try:
        return HRLConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(
            f"Configuration validation failed for '{config_path}':\n{exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__: list[str] = [
    "HRLConfig",
    "ProjectConfig",
    "BenchmarkConfig",
    "GraphConfig",
    "EvaluatorConfig",
    "RewardConfig",
    "PPOConfig",
    "FlatAgentConfig",
    "HierarchicalAgentConfig",
    "TrainingConfig",
    "LoggingConfig",
    "OutputConfig",
    "load_config",
]
