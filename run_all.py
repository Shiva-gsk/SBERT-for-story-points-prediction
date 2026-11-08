"""Run the whole PS-6 pipeline in order (guidelines 3.6: instructions to reproduce the principal results).

    python run_all.py                              # all stages
    python run_all.py --project alloy_framework    # one project (stages that support it)
    python run_all.py --from-stage train_cv        # resume
    python run_all.py --dry-run                    # print the commands only

Prerequisite (needs the TAWOS MySQL dump): python extract_data_from_sql_v3.py  -> data/raw/sbert_project6_issues.csv
After changing the config, filters, model or scheme, delete data/processed, data/splits, data/embeddings and results first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

# (module, accepts --project)
STAGES = [("env_info", False), ("preprocess", True), ("make_splits", True), ("extract_embeddings", True),
          ("train_cv", True), ("evaluate_test", True), ("baselines", True), ("summarize", False),
          ("coverage_report", False), ("make_plots", False)]


def main() -> int:
    names = [s for s, _ in STAGES]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--from-stage", choices=names, default=names[0])
    ap.add_argument("--to-stage", choices=names, default=names[-1])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lo, hi = names.index(args.from_stage), names.index(args.to_stage)
    for mod, takes_project in STAGES[lo:hi + 1]:
        cmd = [sys.executable, "-m", f"src.{mod}"]
        if args.config:
            cmd += ["--config", args.config]
        if args.project and takes_project:
            cmd += ["--project", args.project]
        print(f"\n=== {mod} ===\n$ {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"\nStage {mod} failed (exit code {rc}); stopping. Fix it, then resume with --from-stage {mod}")
            return rc
        print(f"[{mod} done in {time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())