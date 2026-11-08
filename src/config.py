"""YAML config loader + small CLI helper."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "ps6.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the YAML config (default = repo configs/ps6.yaml)."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (override wins)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_common_args() -> argparse.Namespace:
    """Common CLI flags shared by stage scripts."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", default=None, help="path to YAML config (default: configs/ps6.yaml)")
    ap.add_argument("--project", default=None, help="restrict to a single project slug")
    ap.add_argument("--max-epochs", type=int, default=None, help="override train.max_epochs (debug)")
    return ap.parse_args()


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply light CLI overrides (e.g. --max-epochs) onto a loaded config."""
    if getattr(args, "max_epochs", None) is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    return cfg
