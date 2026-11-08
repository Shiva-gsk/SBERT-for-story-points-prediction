"""Stage 3: temporal 80/20 split + chronological blocks per project.

For each eligible project (data/processed/<slug>.csv):
  * STABLE sort by created_at (ties keep file order)
  * earliest dev_ratio  -> development, latest (1 - dev_ratio) -> held-out test
  * development is cut into contiguous chronological blocks (fold_id):
      rolling : n_folds + 1 blocks by default (ids 0..n_folds). Block 0 is train-only; validation fold k = block k
                (k = 1..n_folds) is predicted from blocks < k  -> exactly n_folds causal validation runs.
                (config split.n_blocks can override, e.g. 5 blocks -> only 4 validation runs)
      blocked : n_folds blocks (ids 0..n_folds-1). Every block is validated once, trained on ALL others
                (includes FUTURE blocks -> NOT temporally valid; use only as a labelled comparison).

Output: data/splits/<slug>_splits.csv  (all processed columns + fold_id, role, scheme, n_blocks)
        data/splits/split_summary.csv  (one row per project)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .utils import append_experiment_log, get_logger

log = get_logger("make_splits")

SCHEMES = ("rolling", "blocked")


def n_blocks_for(scheme: str, n_folds: int, n_blocks_cfg=None) -> int:
    """Number of chronological blocks the development set is cut into.
    default: rolling -> n_folds + 1 (block 0 is train-only, so n_folds validation runs); blocked -> n_folds."""
    if n_blocks_cfg:
        return int(n_blocks_cfg)
    return n_folds + 1 if scheme == "rolling" else n_folds


def n_validation_folds(scheme: str, n_blocks: int) -> int:
    return n_blocks - 1 if scheme == "rolling" else n_blocks


def make_temporal_splits(df: pd.DataFrame, dev_ratio: float, n_folds: int, scheme: str,
                         n_blocks: int | None = None) -> pd.DataFrame:
    if scheme not in SCHEMES:
        raise ValueError(f"fold_scheme must be one of {SCHEMES}, got {scheme!r}")
    df = df.sort_values("created_at", kind="stable").reset_index(drop=True)
    n = len(df)
    n_dev = int(round(n * dev_ratio))
    dev = df.iloc[:n_dev].copy()
    test = df.iloc[n_dev:].copy()

    n_blocks = n_blocks_for(scheme, n_folds, n_blocks)
    boundaries = [int(round(i * n_dev / n_blocks)) for i in range(n_blocks + 1)]
    dev["fold_id"] = np.repeat(np.arange(n_blocks), np.diff(boundaries))
    dev["role"] = "dev"
    test["fold_id"] = "test"
    test["role"] = "test"

    out = pd.concat([dev, test], ignore_index=True)
    out["scheme"] = scheme
    out["n_blocks"] = n_blocks
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--project", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    proc_dir = Path(cfg["data"]["processed_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    dev_ratio = float(cfg["split"]["dev_ratio"])
    test_ratio = cfg["split"].get("test_ratio")
    if test_ratio is not None and abs(dev_ratio + float(test_ratio) - 1.0) > 1e-9:
        raise ValueError(f"split.dev_ratio ({dev_ratio}) + split.test_ratio ({test_ratio}) must equal 1")
    n_folds = int(cfg["split"]["n_folds"])
    scheme = str(cfg["split"]["fold_scheme"])
    n_blocks_cfg = cfg["split"].get("n_blocks")
    if scheme not in SCHEMES:
        raise ValueError(f"split.fold_scheme must be one of {SCHEMES}")
    n_blocks = n_blocks_for(scheme, n_folds, n_blocks_cfg)
    n_val = n_validation_folds(scheme, n_blocks)
    log.info("scheme=%s -> dev cut into %d chronological blocks -> %d validation folds", scheme, n_blocks, n_val)
    if n_val != n_folds:
        log.warning("split.n_folds=%d but this configuration yields %d validation folds", n_folds, n_val)
    if scheme == "blocked":
        log.warning("fold_scheme=blocked trains on FUTURE blocks to predict past ones; "
                    "this violates temporal validation. Use 'rolling' for the main results.")

    csvs = sorted(p for p in proc_dir.glob("*.csv") if p.name != "dataset_stats.csv")
    if args.project:
        csvs = [p for p in csvs if p.stem == args.project]
    if not csvs:
        log.error("No processed CSVs in %s. Run `python -m src.preprocess` first.", proc_dir)
        return 1

    summary, n_written = [], 0
    for p in csvs:
        df = pd.read_csv(p, parse_dates=["created_at"])
        if len(df) < cfg["data"]["min_issues"]:
            log.info("  %s skipped (n=%d < min_issues)", p.name, len(df))
            continue
        splits = make_temporal_splits(df, dev_ratio, n_folds, scheme, n_blocks)
        out_path = splits_dir / f"{p.stem}_splits.csv"
        splits.to_csv(out_path, index=False)

        dev = splits[splits["role"] == "dev"]
        test = splits[splits["role"] == "test"]
        block_sizes = dev.groupby("fold_id").size()
        tie_at_cut = bool(len(test) and len(dev) and test["created_at"].iloc[0] == dev["created_at"].iloc[-1])
        if block_sizes.min() < 10:
            log.warning("  %s: smallest block has %d issues - fold metrics will be very noisy",
                        p.stem, block_sizes.min())
        if tie_at_cut:
            log.info("  %s: dev/test boundary falls inside a group of identical timestamps "
                     "(deterministic thanks to the stable sort)", p.stem)
        log.info("  %s: dev=%d test=%d blocks=%d scheme=%s -> %s",
                 p.stem, len(dev), len(test), splits["n_blocks"].iloc[0], scheme, out_path.name)
        summary.append({"slug": p.stem, "n": len(splits), "n_dev": len(dev), "n_test": len(test),
                        "min_block": int(block_sizes.min()), "scheme": scheme,
                        "dev_start": str(dev["created_at"].iloc[0]), "dev_end": str(dev["created_at"].iloc[-1]),
                        "test_start": str(test["created_at"].iloc[0]) if len(test) else "",
                        "test_end": str(test["created_at"].iloc[-1]) if len(test) else "",
                        "tie_at_cut": tie_at_cut})
        n_written += 1

    if summary:
        pd.DataFrame(summary).to_csv(splits_dir / "split_summary.csv", index=False)
    append_experiment_log(cfg["results"]["experiment_log"], config=cfg, stage="make_splits",
                          notes=f"{n_written} project split files written; scheme={scheme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())