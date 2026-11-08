"""Which of the 39 projects named in the PS guidelines (section 3.3) made it into this run, and why not?

Reads data/processed/dataset_stats.csv (written by preprocess) and writes results/summary/project_coverage.csv.
Matching is by slug (exact, else the slug ends with the project name), so check any 'NOT FOUND' by hand:
the project may exist under a different name in your TAWOS export, or may have been removed by the
extraction filters (see data/reports/filter_funnel.csv and project_summary.csv).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .utils import get_logger

log = get_logger("coverage_report")

PDF_PROJECTS = [
    "XD", "Usergrid", "Titanium Mobile Platform", "The Titanium SDK", "Software Cloud", "Sawtooth", "Nexus",
    "MXNet", "Mule APIkit", "Mule", "Moodle", "MongoDB Core Server", "MESOS", "Lyrasis Dura Cloud",
    "Lsstcorp Data management", "Jira Server", "Jira Cloud", "Java driver", "Indy SDK", "Indy Node", "FishEye",
    "Fabric", "Evergreen", "DotNetNuke Platform", "DataCass", "Crowd", "Confluence Server", "Confluence Cloud",
    "Compass", "Command-Line Interface", "Clover", "C++ driver (Appcelerator)", "C++ driver (MongoDB)",
    "Blockchain Explorer", "Bamboo", "Atlassian Software Server", "Appcelerator Studio", "Appcelerator Daemon",
    "Alloy Framework",
]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    stats_path = Path(cfg["data"]["processed_dir"]) / "dataset_stats.csv"
    if not stats_path.exists():
        log.error("%s not found - run preprocess first.", stats_path)
        return 1
    st = pd.read_csv(stats_path)
    if "slug" not in st.columns:
        log.error("dataset_stats.csv has no slug column.")
        return 1
    st = st.dropna(subset=["slug"])
    slugs = st["slug"].astype(str).tolist()

    rows, used = [], set()
    for name in PDF_PROJECTS:
        ns = _slug(name)
        hits = [s for s in slugs if s == ns] or [s for s in slugs if s.endswith("_" + ns)]
        used.update(hits)
        if hits:
            for h in hits:
                r = st[st["slug"] == h].iloc[0]
                rows.append({"pdf_name": name, "slug": h, "n_issues": r.get("n_issues"),
                             "eligible": r.get("eligible"), "status": "found" if r.get("eligible") else "below min_issues"})
        else:
            rows.append({"pdf_name": name, "slug": "", "n_issues": 0, "eligible": False, "status": "NOT FOUND"})
    for s in sorted(set(slugs) - used):
        r = st[st["slug"] == s].iloc[0]
        rows.append({"pdf_name": "", "slug": s, "n_issues": r.get("n_issues"), "eligible": r.get("eligible"),
                     "status": "not in guideline list"})
    out = pd.DataFrame(rows)
    p = Path(cfg["results"]["summary_dir"]) / "project_coverage.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)

    vc = out["status"].value_counts().to_dict()
    log.info("coverage of the %d guideline projects: %s", len(PDF_PROJECTS), vc)
    for _, r in out[out["status"].isin(["NOT FOUND", "below min_issues"])].iterrows():
        log.warning("  %-28s %s (n=%s)", r["pdf_name"], r["status"], r["n_issues"])
    log.info("wrote %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())