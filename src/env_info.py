"""Record the software environment (guidelines 3.7: record model versions and configuration).

    python -m src.env_info                         # writes results/environment.json
    python -m src.env_info --write-requirements    # also writes requirements.lock.txt with exact versions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .config import load_config

PACKAGES = ["torch", "sentence-transformers", "transformers", "numpy", "pandas", "scikit-learn",
            "matplotlib", "PyYAML", "SQLAlchemy", "PyMySQL"]


def _version(pkg: str):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def collect(cfg: dict) -> dict:
    info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": {p: _version(p) for p in PACKAGES},
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
            "seed": cfg.get("seed"), "sbert_model": cfg.get("sbert", {}).get("model_name"),
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()}
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["cuda_available"] = None
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/environment.json")
    ap.add_argument("--write-requirements", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    info = collect(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    if args.write_requirements:
        lines = [f"{p}=={v}" for p, v in info["packages"].items() if v]
        Path("requirements.lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\nWrote requirements.lock.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())