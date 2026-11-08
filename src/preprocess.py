"""Stage 2: clean + unify schema per project.

Reads the raw CSV (data.raw_file if set, else every CSV in data/raw/), unifies columns using the
candidate-name lists in the config, and writes one cleaned CSV per eligible project to
data/processed/<slug>.csv plus data/processed/dataset_stats.csv.

Order within each project file is chronological (STABLE sort on created_at, so ties keep the order of
the raw export, which the extractor sorted by Creation_Date, Issue_ID).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .utils import append_experiment_log, get_logger, slugify
from .download_data import discover_raw_csvs

log = get_logger("preprocess")

_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# Whole {code}...{code} / {noformat}...{noformat} blocks are removed (code is noise for SBERT) ...
_CODE_BLOCK_RE = re.compile(r"\{(code|noformat)(?:[:|][^}]*)?\}.*?\{\1\}", re.S | re.I)
# ... and only known Jira formatting tags are stripped, not arbitrary {...} text.
_JIRA_TAG_RE = re.compile(r"\{(?:code|noformat|quote|panel|color)(?:[:|][^}]*)?\}", re.I)
_ID_COL_RE = re.compile(r"(^|[_\s])id$")


def _resolve_column(df_cols: list[str], candidates: list[str], used: set[str] | None = None) -> str | None:
    """Resolve a logical column from candidate names.

    1. exact (case-insensitive) match, in candidate order;
    2. substring match as a last resort, but ONLY for candidates >= 5 chars, never on *_id columns,
       and never on a column already assigned to another logical field.
    """
    used = used or set()
    lower = {c.lower().strip(): c for c in df_cols if c not in used}
    for cand in candidates:
        c = cand.lower().strip()
        if c in lower:
            return lower[c]
    for cand in candidates:
        cl = cand.lower().strip()
        if len(cl) < 5:
            continue
        for lc, orig in lower.items():
            if cl in lc and not _ID_COL_RE.search(lc):
                return orig
    return None


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = _CODE_BLOCK_RE.sub(" ", text)
    t = _JIRA_TAG_RE.sub(" ", t)
    t = _HTML_RE.sub(" ", t)
    t = (t.replace("&nbsp;", " ")
           .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
           .replace("&quot;", '"').replace("&#39;", "'"))
    return _WS_RE.sub(" ", t).strip()


def coerce_story_point(s) -> float | None:
    """Parse a story point value; return None if not a non-negative number."""
    if pd.isna(s):
        return None
    try:
        v = float(str(s).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None
    if v != v or v < 0:
        return None
    return v


def process_one(raw_path: Path, mapping: dict, out_dir: Path, min_issues: int,
                global_project_col: str | None = None,
                sp_min: float | None = 1, sp_max: float | None = 100,
                only_slug: str | None = None) -> tuple[list[str], list[dict]]:
    """Returns (slugs_seen, list_of_stat_dicts). Always returns a LIST of dicts (also when skipped)."""
    df = pd.read_csv(raw_path, low_memory=False, encoding="utf-8-sig")
    cols = list(df.columns)

    used: set[str] = set()

    def pick(key):
        col = _resolve_column(cols, mapping.get(key, []), used)
        if col is not None:
            used.add(col)
        return col

    issue_key_col = pick("issue_key")
    title_col = pick("title")
    desc_col = pick("description")
    text_col = pick("text")
    sp_col = pick("story_point")
    created_col = pick("created_at")
    proj_col = global_project_col or pick("project")
    repo_col = pick("repository")
    log.info("  column resolution: key=%s title=%s desc=%s text=%s sp=%s created=%s project=%s repo=%s",
             issue_key_col, title_col, desc_col, text_col, sp_col, created_col, proj_col, repo_col)

    have_parts = (title_col is not None) and (desc_col is not None)
    have_combined = text_col is not None
    missing = [k for k, v in [("issue_key", issue_key_col), ("story_point", sp_col),
                              ("created_at", created_col)] if v is None]
    if not have_parts and not have_combined:
        missing.append("title+description OR text")
    if missing:
        log.warning("  %s: missing candidate columns for %s - skipped.", raw_path.name, missing)
        return [], [{"path": str(raw_path), "skipped": True, "missing_columns": ";".join(missing)}]

    rename_map = {issue_key_col: "issue_key", sp_col: "story_point", created_col: "created_at"}
    if title_col is not None:
        rename_map[title_col] = "title"
    if desc_col is not None:
        rename_map[desc_col] = "description"
    if text_col is not None:
        rename_map[text_col] = "combined_text"
    if repo_col is not None:
        rename_map[repo_col] = "repository"
    if proj_col is not None:
        rename_map[proj_col] = "project"
    df = df.rename(columns=rename_map)
    if "project" not in df.columns:
        df["project"] = raw_path.stem
    df["project"] = df["project"].fillna("unknown").astype(str).str.strip()

    df["issue_key"] = df["issue_key"].astype(str).str.strip()
    if have_parts:
        df["title"] = df["title"].fillna("").astype(str)
        df["description"] = df["description"].fillna("").astype(str)
        df["text"] = (df["title"].str.strip() + "\n\n" + df["description"].str.strip()).map(clean_text)
    else:
        df["text"] = df["combined_text"].fillna("").astype(str).map(clean_text)
    df["story_point"] = df["story_point"].map(coerce_story_point)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if "repository" in df.columns:
        df["repository"] = df["repository"].fillna("unknown").astype(str).str.strip()

    before_counts = df.groupby("project").size().to_dict()

    df = df.dropna(subset=["story_point", "created_at"])
    if sp_min is not None:
        df = df[df["story_point"] >= sp_min]
    if sp_max is not None:
        df = df[df["story_point"] <= sp_max]
    df = df[df["text"].str.len() >= 5]
    df = df.drop_duplicates(subset=["issue_key"])
    keep_cols = ["issue_key", "project", "text", "story_point", "created_at"]
    if "repository" in df.columns:
        keep_cols.append("repository")
    df = df[keep_cols].sort_values("created_at", kind="stable")   # STABLE: ties keep export order

    out_records: list[dict] = []
    slugs: list[str] = []
    for project_name, sub in df.groupby("project", sort=False):
        slug = slugify(str(project_name))
        if only_slug and slug != only_slug:
            continue
        slugs.append(slug)
        out_path = out_dir / f"{slug}.csv"
        rec = {"project": str(project_name), "slug": slug, "n_issues": int(len(sub)),
               "n_raw": int(before_counts.get(project_name, len(sub))),
               "rows_dropped": int(before_counts.get(project_name, len(sub)) - len(sub)),
               "mean_sp": float(sub["story_point"].mean()),
               "median_sp": float(sub["story_point"].median()),
               "first_issue": str(sub["created_at"].iloc[0]), "last_issue": str(sub["created_at"].iloc[-1]),
               "raw_file": raw_path.name}
        if len(sub) < min_issues:
            log.info("  project=%s slug=%s skipped (n=%d < min_issues=%d)", project_name, slug, len(sub), min_issues)
            if out_path.exists():            # remove stale output from an earlier, lower threshold
                out_path.unlink()
            rec["eligible"] = False
        else:
            sub.to_csv(out_path, index=False)
            log.info("  project=%-30s slug=%-25s n=%5d  -> %s", project_name, slug, len(sub), out_path.name)
            rec["eligible"] = True
        out_records.append(rec)
    return slugs, out_records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--project", default=None, help="restrict to a single project slug")
    args = ap.parse_args()
    cfg = load_config(args.config)

    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = cfg["data"]["column_mapping"]
    min_issues = int(cfg["data"]["min_issues"])

    raw_file = cfg["data"].get("raw_file")
    if raw_file:
        p = raw_dir / raw_file
        csvs = [p] if p.exists() else []
    else:
        csvs = discover_raw_csvs(raw_dir)
    if not csvs:
        log.error("No raw CSV found in %s (raw_file=%r). Run the extraction step first.", raw_dir, raw_file)
        return 1

    sp_min = cfg["data"].get("sp_min", 1)
    sp_max = cfg["data"].get("sp_max", 100)
    all_stats: list[dict] = []
    for raw_path in csvs:
        log.info("Processing %s ...", raw_path.name)
        _, stats = process_one(raw_path, mapping, out_dir, min_issues,
                               sp_min=sp_min, sp_max=sp_max, only_slug=args.project)
        all_stats.extend(stats)

    stats_path = out_dir / "dataset_stats.csv"
    new = pd.DataFrame(all_stats)
    if args.project and stats_path.exists() and not new.empty and "slug" in new.columns:
        old = pd.read_csv(stats_path)                       # keep other projects' rows
        if "slug" in old.columns:
            new = pd.concat([old[old["slug"] != args.project], new], ignore_index=True)
    new.to_csv(stats_path, index=False)
    log.info("Wrote %s (%d rows)", stats_path, len(new))
    append_experiment_log(cfg["results"]["experiment_log"], config=cfg, stage="preprocess",
                          notes=f"{len(all_stats)} project(s) processed; min_issues={min_issues}")
    return 0


if __name__ == "__main__":
    sys.exit(main())