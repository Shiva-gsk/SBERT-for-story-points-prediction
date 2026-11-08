"""Stage 7: aggregate everything into report tables (results/summary/).

MODEL COMPARISON (CNN vs baselines, same temporal splits; the baselines come from src.baselines)
  comparative_table.csv             ONE ROW PER MODEL: validation + test MAE/MdAE (macro and pooled), wins, mean rank,
                                    reduction vs the median baseline, reliable-project metrics. Common projects only.
  model_results_long.csv            every project x model: chosen setting, validation mean/SD, test MAE/MdAE
  model_comparison_by_project.csv   one row per project: test MAE/MdAE of every model, best model, CNN rank
  paired_comparison.csv             CNN vs each baseline: wins/losses, median difference, Wilcoxon p-value
  guideline_table.csv               the guidelines' 3-column table, SBERT + CNN row (the PS-6 assigned model)
CNN DETAIL
  five_fold_by_lr.csv               every (project, lr): mean/SD of fold MAE and MdAE   [model selection]
  five_fold_summary.csv             per project, SELECTED lr only: mean/SD MAE, MdAE
  five_fold_validation_results.csv  per-fold rows for the SELECTED lr only
  test_results.csv                  per-project held-out test metrics (+ median/mean baselines)
  project_results.csv               CNN validation (selected lr) and test side by side, one row per project
  aggregate_results.csv             CNN macro/pooled aggregates, fold counts by early-stop source, "reliable" subset
SD = sample standard deviation (ddof=1) across folds. "Common projects" = projects that have a result for EVERY model
present, so all models are compared on identical projects. "Reliable" = n_test >= eval.min_test_issues_reliable (30).
Run src.baselines BEFORE this stage to get the baseline rows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import DEFAULT_ALPHAS, DEFAULT_KS
from .config import load_config
from .utils import append_experiment_log, get_logger

log = get_logger("summarize")

MODEL_ORDER = ["cnn", "ridge_sbert", "knn_sbert", "tfidf_ridge", "median"]
LABEL = {"cnn": "CNN (SBERT)", "ridge_sbert": "Ridge (SBERT)", "knn_sbert": "kNN (SBERT)",
         "tfidf_ridge": "TF-IDF + Ridge", "median": "Median (constant)"}
LONG_COLS = ["project", "model", "best_param", "n_test", "val_MAE_mean", "val_MAE_sd", "val_MdAE_mean",
             "val_MdAE_sd", "test_MAE", "test_MdAE"]


def _read_all(d: Path, exclude_suffix: str = "_val_predictions") -> pd.DataFrame:
    files = sorted(p for p in d.glob("*.csv") if not p.stem.endswith(exclude_suffix))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p).assign(project=p.stem) for p in files], ignore_index=True)


def _representation(m: str, cfg: dict) -> str:
    if m in ("cnn", "ridge_sbert", "knn_sbert"):
        return f"SBERT sentence embeddings ({str(cfg['sbert']['model_name']).split('/')[-1]})"
    if m == "tfidf_ridge":
        return "TF-IDF, 1-2 grams (non-SBERT reference)"
    return "none (constant prediction)"


def _configuration(m: str, cfg: dict) -> str:
    b = cfg.get("baselines", {}) or {}
    a, ks = b.get("ridge_alphas", DEFAULT_ALPHAS), b.get("knn_ks", DEFAULT_KS)
    return {"cnn": f"3-layer CNN, filters {cfg['model']['cnn_filters']}, lr per project from {cfg['train']['lr_grid']}, "
                   f"L1 loss, seed {cfg['seed']}",
            "ridge_sbert": f"Ridge regression, alpha per project from {a}",
            "knn_sbert": f"kNN (cosine), median story point of k neighbours, k per project from {ks}",
            "tfidf_ridge": f"TF-IDF + Ridge, alpha per project from {a}; vocabulary fit on training rows only",
            "median": "median story point of the development rows"}[m]


# ------------------------------------------------------------------ assemble per-project x model results
def build_long(cfg: dict, sel: pd.DataFrame, test_df: pd.DataFrame, bdir: Path) -> pd.DataFrame:
    parts = []
    if len(test_df) and "test_MAE" in test_df:
        t = test_df.copy()
        if "best_lr" not in t:
            t["best_lr"] = np.nan
        c = t[["project", "n_test", "test_MAE", "test_MdAE", "best_lr"]].rename(columns={"best_lr": "best_param"})
        if len(sel):
            s = sel.rename(columns={"mean_MAE": "val_MAE_mean", "sd_MAE": "val_MAE_sd",
                                    "mean_MdAE": "val_MdAE_mean", "sd_MdAE": "val_MdAE_sd"})
            c = c.merge(s[["project", "val_MAE_mean", "val_MAE_sd", "val_MdAE_mean", "val_MdAE_sd"]], on="project", how="left")
        c["model"] = "cnn"
        parts.append(c)
    bp = bdir / "baseline_results.csv"
    have_median = False
    if bp.exists():
        b = pd.read_csv(bp)
        parts.append(b)
        have_median = (b["model"] == "median").any()
    if len(test_df) and not have_median and "baseline_median_MAE" in test_df:    # fallback from evaluate_test
        m = test_df[["project", "n_test", "baseline_median_MAE", "baseline_median_MdAE"]].rename(
            columns={"baseline_median_MAE": "test_MAE", "baseline_median_MdAE": "test_MdAE"})
        m["model"], m["best_param"] = "median", "-"
        parts.append(m)
    if not parts:
        return pd.DataFrame(columns=LONG_COLS)
    L = pd.concat(parts, ignore_index=True)
    for col in LONG_COLS:
        if col not in L:
            L[col] = np.nan
    L = L[L["model"].isin(MODEL_ORDER)]
    return L[LONG_COLS].sort_values(["project", "model"]).reset_index(drop=True)


def _pred_path(m: str, p: str, tdir: Path, bdir: Path) -> Path:
    return (tdir / "predictions" / f"{p}_test_predictions.csv" if m == "cnn"
            else bdir / "predictions" / f"{p}_{m}_test_predictions.csv")


def _pooled(m: str, projects, tdir: Path, bdir: Path):
    errs = []
    for p in projects:
        f = _pred_path(m, p, tdir, bdir)
        if not f.exists():
            return np.nan, np.nan
        d = pd.read_csv(f)
        errs.append((d["y_true"] - d["y_pred"]).abs().to_numpy())
    if not errs:
        return np.nan, np.nan
    e = np.concatenate(errs)
    return float(e.mean()), float(np.median(e))


def _wilcoxon(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) < 6:
        return np.nan
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        return np.nan


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    nd = int(cfg.get("eval", {}).get("round_decimals", 4))
    rel_min = int(cfg.get("eval", {}).get("min_test_issues_reliable", 30))

    fold_dir = Path(cfg["results"]["five_fold_dir"])
    test_dir = Path(cfg["results"]["test_dir"])
    bdir = Path(cfg["results"].get("baselines_dir", "results/baselines"))
    out = Path(cfg["results"]["summary_dir"])
    out.mkdir(parents=True, exist_ok=True)
    R = lambda df: df.round(nd)

    # ---------- CNN validation: by lr, then selected lr ----------
    fold_df = _read_all(fold_dir)
    if fold_df.empty:
        log.warning("No fold CSVs in %s.", fold_dir)
        by_lr = sel = sel_rows = pd.DataFrame()
    else:
        by_lr = (fold_df.groupby(["project", "lr"])
                 .agg(n_folds=("fold", "count"), mean_MAE=("MAE", "mean"), sd_MAE=("MAE", "std"),
                      mean_MdAE=("MdAE", "mean"), sd_MdAE=("MdAE", "std")).reset_index())
        sel = (by_lr.loc[by_lr.groupby("project")["mean_MAE"].idxmin()]
               .rename(columns={"lr": "best_lr"}).reset_index(drop=True))
        sel_rows = fold_df.merge(sel[["project", "best_lr"]], left_on=["project", "lr"],
                                 right_on=["project", "best_lr"]).drop(columns="best_lr")
    by_lr.to_csv(out / "five_fold_by_lr.csv", index=False)
    sel.to_csv(out / "five_fold_summary.csv", index=False)
    sel_rows.to_csv(out / "five_fold_validation_results.csv", index=False)
    log.info("CNN validation: %d projects, %d fold rows (selected lr)", len(sel), len(sel_rows))

    # ---------- CNN test ----------
    test_df = _read_all(test_dir)
    test_df.to_csv(out / "test_results.csv", index=False)
    log.info("CNN test: %d projects", len(test_df))

    proj = pd.DataFrame()
    if len(sel) and len(test_df):
        proj = sel.merge(test_df.drop(columns=[c for c in ("best_lr",) if c in test_df.columns]),
                         on="project", how="outer", suffixes=("", "_test"))
        proj = proj.rename(columns={"mean_MAE": "val_MAE_mean", "sd_MAE": "val_MAE_sd",
                                    "mean_MdAE": "val_MdAE_mean", "sd_MdAE": "val_MdAE_sd"})
        proj["reliable"] = proj["n_test"].ge(rel_min)
        n_missing = int(proj["test_MAE"].isna().sum())
        if n_missing:
            log.warning("%d project(s) have validation but no test result (run evaluate_test).", n_missing)
    proj.to_csv(out / "project_results.csv", index=False)

    # ---------- CNN aggregates (as before) ----------
    agg = {}
    if len(sel_rows) and "early_stop_source" in sel_rows.columns:
        for k, v in sel_rows["early_stop_source"].value_counts().items():
            agg[f"folds_{k}"] = int(v)
        agg["folds_total"] = int(len(sel_rows))
    if len(sel):
        agg.update(n_projects_validated=len(sel), val_MAE_macro=sel["mean_MAE"].mean(), val_MdAE_macro=sel["mean_MdAE"].mean())
    if len(test_df):
        agg.update(n_projects_tested=len(test_df), test_MAE_macro=test_df["test_MAE"].mean(),
                   test_MdAE_macro=test_df["test_MdAE"].mean(), test_MAE_macro_sd_over_projects=test_df["test_MAE"].std())
        rel = test_df[test_df["n_test"] >= rel_min]
        agg.update(min_test_issues_reliable=rel_min, n_projects_reliable=len(rel))
        if len(rel):
            agg.update(test_MAE_macro_reliable=rel["test_MAE"].mean(), test_MdAE_macro_reliable=rel["test_MdAE"].mean())
            if "baseline_median_MAE" in rel:
                agg["reliable_projects_beating_median_baseline"] = int((rel["test_MAE"] < rel["baseline_median_MAE"]).sum())
        for c in ("baseline_median_MAE", "baseline_median_MdAE", "baseline_mean_MAE"):
            if c in test_df:
                agg[c + "_macro"] = test_df[c].mean()
        if "baseline_median_MAE" in test_df:
            agg["projects_beating_median_baseline_MAE"] = int((test_df["test_MAE"] < test_df["baseline_median_MAE"]).sum())
        pm, pd_ = _pooled("cnn", test_df["project"], test_dir, bdir)
        agg.update(test_MAE_pooled=pm, test_MdAE_pooled=pd_, n_test_issues_pooled=int(test_df["n_test"].sum()))
    pd.DataFrame([{k: (round(v, nd) if isinstance(v, (float, np.floating)) else v) for k, v in agg.items()}]
                 ).to_csv(out / "aggregate_results.csv", index=False)

    # ---------- guideline-format table (SBERT + CNN, the assigned model) ----------
    t_mae, t_mdae = agg.get("test_MAE_macro", np.nan), agg.get("test_MdAE_macro", np.nan)
    pd.DataFrame([{"representation": "SBERT sentence embeddings (PS-6)",
                   "model_configuration": (f"{_configuration('cnn', cfg)}; within-project, temporal 80/20, "
                                           f"{cfg['split']['n_folds']}-fold {cfg['split']['fold_scheme']} CV; "
                                           f"MAE/MdAE = test, macro-average over {int(agg.get('n_projects_tested', 0))} projects"),
                   "MAE": round(float(t_mae), nd) if not np.isnan(t_mae) else "[TO FILL]",
                   "MdAE": round(float(t_mdae), nd) if not np.isnan(t_mdae) else "[TO FILL]"}]
                 ).to_csv(out / "guideline_table.csv", index=False)

    # ================= MODEL COMPARISON =================
    L = build_long(cfg, sel, test_df, bdir)
    R(L).to_csv(out / "model_results_long.csv", index=False)
    present = [m for m in MODEL_ORDER if (L["model"] == m).any()]
    if not (bdir / "baseline_results.csv").exists():
        log.warning("No %s - the comparison only has the CNN and the median predictor. Run src.baselines first.",
                    bdir / "baseline_results.csv")
    if "cnn" not in present or len(present) < 2:
        log.warning("Not enough models for a comparison (have %s).", present)
        R(pd.DataFrame()).to_csv(out / "comparative_table.csv", index=False)
        return 0

    wideM = L.pivot(index="project", columns="model", values="test_MAE")[present]
    wideD = L.pivot(index="project", columns="model", values="test_MdAE")[present]
    n_test = L.groupby("project")["n_test"].first()
    common = wideM.dropna().index.tolist()
    dropped = sorted(set(wideM.index) - set(common))
    if dropped:
        log.warning("%d project(s) lack at least one model and are excluded from the comparison: %s",
                    len(dropped), dropped[:8])
    if not common:
        log.warning("No project has results for all models.")
        return 0
    rel_common = [p for p in common if n_test.get(p, 0) >= rel_min]
    ranks = wideM.loc[common].rank(axis=1, method="average")
    best = wideM.loc[common].min(axis=1)
    med = wideM.loc[common, "median"].mean() if "median" in present else np.nan

    rows = []
    for m in present:
        sub = L[(L["model"] == m) & L["project"].isin(common)]
        pm, pdn = _pooled(m, common, test_dir, bdir)
        r = {"model_key": m, "Model": LABEL[m], "Representation": _representation(m, cfg),
             "Configuration": _configuration(m, cfg), "n_projects": len(common),
             "Val_MAE_mean": sub["val_MAE_mean"].mean(), "Val_MAE_fold_SD_avg": sub["val_MAE_sd"].mean(),
             "Val_MdAE_mean": sub["val_MdAE_mean"].mean(),
             "Test_MAE_macro": wideM.loc[common, m].mean(), "Test_MdAE_macro": wideD.loc[common, m].mean(),
             "Test_MAE_pooled": pm, "Test_MdAE_pooled": pdn,
             "n_reliable_projects": len(rel_common),
             "Test_MAE_macro_reliable": wideM.loc[rel_common, m].mean() if rel_common else np.nan,
             "Test_MdAE_macro_reliable": wideD.loc[rel_common, m].mean() if rel_common else np.nan,
             "Wins_MAE": int((wideM.loc[common, m] <= best).sum()),
             "Mean_rank_MAE": ranks[m].mean(),
             "MAE_reduction_vs_median_pct": (100 * (med - wideM.loc[common, m].mean()) / med) if m != "median" and med == med else np.nan,
             "Projects_better_than_median": int((wideM.loc[common, m] < wideM.loc[common, "median"]).sum()) if m != "median" and "median" in present else np.nan}
        rows.append(r)
    CT = pd.DataFrame(rows).sort_values("Test_MAE_macro").reset_index(drop=True)
    CT.insert(0, "Rank_by_Test_MAE", np.arange(1, len(CT) + 1))
    R(CT).to_csv(out / "comparative_table.csv", index=False)

    # ---------- per-project wide table ----------
    PB = pd.DataFrame({"project": wideM.index, "n_test": n_test.reindex(wideM.index).values})
    for m in present:
        PB[f"{m}_test_MAE"] = wideM[m].values
        PB[f"{m}_test_MdAE"] = wideD[m].values
    PB["best_model"] = wideM.idxmin(axis=1).map(LABEL).values
    PB["all_models"] = wideM.notna().all(axis=1).values
    PB["reliable"] = PB["n_test"] >= rel_min
    PB["cnn_rank"] = wideM.rank(axis=1, method="average")["cnn"].values
    R(PB).to_csv(out / "model_comparison_by_project.csv", index=False)

    # ---------- paired comparison: CNN vs each baseline ----------
    pr = []
    for m in present:
        if m == "cnn":
            continue
        for scope, projs in (("all", common), ("reliable", rel_common)):
            if not projs:
                continue
            d = (wideM.loc[projs, m] - wideM.loc[projs, "cnn"]).to_numpy()      # > 0  => CNN better
            pr.append({"baseline": LABEL[m], "scope": scope, "n_projects": len(projs),
                       "cnn_better": int((d > 0).sum()), "baseline_better": int((d < 0).sum()), "ties": int((d == 0).sum()),
                       "median_MAE_diff_(baseline-cnn)": float(np.median(d)), "mean_MAE_diff_(baseline-cnn)": float(d.mean()),
                       "wilcoxon_p": _wilcoxon(d)})
    R(pd.DataFrame(pr)).to_csv(out / "paired_comparison.csv", index=False)

    log.info("model comparison on %d common projects:\n%s", len(common),
             R(CT[["Rank_by_Test_MAE", "Model", "Val_MAE_mean", "Test_MAE_macro", "Test_MdAE_macro",
                   "Wins_MAE", "Mean_rank_MAE"]]).to_string(index=False))
    log.info("Wrote summary tables to %s", out)
    append_experiment_log(cfg["results"]["experiment_log"], config=cfg, stage="summarize",
                          notes=f"{len(sel)} CNN projects; {len(common)} common projects; models={present}")
    return 0


if __name__ == "__main__":
    sys.exit(main())