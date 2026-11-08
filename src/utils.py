"""Shared utilities: seeding, slugify, logging, experiment log."""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

LOG_FMT = "%(asctime)s | %(levelname)7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(LOG_FMT))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def set_global_seed(seed: int) -> None:
    """Seed python / numpy / torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except ImportError:
        pass


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Canonical filename-safe project slug, e.g. 'Mule C++ driver (Appcelerator)' -> 'mule_cpp_driver_appcelerator'."""
    s = name.strip().lower()
    s = _SLUG_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def package_versions() -> Dict[str, str]:
    """Record versions of the libraries that affect reproducibility."""
    out: Dict[str, str] = {}
    for pkg in ["numpy", "pandas", "torch", "sentence_transformers",
                "transformers", "sklearn", "matplotlib", "yaml"]:
        try:
            if pkg == "yaml":
                import yaml as _m
            elif pkg == "sklearn":
                import sklearn as _m
            elif pkg == "sentence_transformers":
                import sentence_transformers as _m
            elif pkg == "torch":
                import torch as _m
            else:
                _m = __import__(pkg)
            out[pkg] = getattr(_m, "__version__", "n/a")
        except Exception:
            out[pkg] = "n/a"
    return out


def git_hash() -> str:
    """Best-effort git HEAD hash for reproducibility stamping."""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "n/a"


def append_experiment_log(
    log_path: str | Path,
    *,
    config: Dict[str, Any],
    stage: str,
    project: str = "ALL",
    notes: str = "",
) -> None:
    """Append one row to results/experiment_log.csv (creates header on first use)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "project": project,
        "seed": config.get("seed"),
        "device": resolve_device(config.get("device", "auto")),
        "sbert_model": config.get("sbert", {}).get("model_name"),
        "cnn_filters": json.dumps(config.get("model", {}).get("cnn_filters")),
        "fold_scheme": config.get("split", {}).get("fold_scheme"),
        "n_folds": config.get("split", {}).get("n_folds"),
        "git_hash": git_hash(),
        "versions": json.dumps(package_versions()),
        "notes": notes,
    }
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def config_signature(config: Dict[str, Any]) -> str:
    """Stable short hash of the parts of the config that affect outputs."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()[:10]
