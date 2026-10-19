"""experiments/freeze.py
========================
Config freezing utilities to prevent accidental changes after final runs.
"""
from __future__ import annotations

from pathlib import Path


class FrozenConfigError(Exception):
    """Raised when an operation requires a frozen config but config is not frozen."""


class ConfigFreezer:
    """Freeze experiment configs before final runs to ensure reproducibility."""

    def freeze_config(self, config: dict, output_path: str) -> None:
        """Add frozen=True marker and save config as YAML."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")
        frozen_config = dict(config)
        frozen_config["frozen"] = True
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(frozen_config, f, default_flow_style=False, sort_keys=True)
        print(f"[ConfigFreezer] Frozen config saved: {output_path}")

    def is_frozen(self, config_path: str) -> bool:
        """Return True if the config file has frozen=True."""
        p = Path(config_path)
        if not p.exists():
            return False
        try:
            import yaml
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return bool(data.get("frozen", False))
        except Exception:
            return False

    def require_frozen_config(self, config_path: str) -> None:
        """Raise FrozenConfigError if config is not frozen."""
        if not self.is_frozen(config_path):
            raise FrozenConfigError(
                f"Config at '{config_path}' is not frozen. "
                "Run ConfigFreezer.freeze_config() before final experiments."
            )


def require_frozen_config(config_path: str) -> None:
    """Module-level convenience wrapper."""
    ConfigFreezer().require_frozen_config(config_path)


def freeze_config(config: dict, output_path: str) -> None:
    """Module-level convenience wrapper."""
    ConfigFreezer().freeze_config(config, output_path)


def is_frozen(config_path: str) -> bool:
    """Module-level convenience wrapper."""
    return ConfigFreezer().is_frozen(config_path)
