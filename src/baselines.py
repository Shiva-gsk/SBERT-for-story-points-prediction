"""Stage 6b: classical baselines on the SAME splits as the CNN (answers: does the CNN add anything?).

Models, per project (nothing here is random, so no seeds are needed):
  median        constant = median story point of the training rows
  ridge_sbert   Ridge regression on the SBERT vectors            grid: baselines.ridge_alphas
  knn_sbert     median story point of the k most cosine-similar training issues   grid: baselines.knn_ks
  tfidf_ridge   TF-IDF (1-2 grams) + Ridge  - a NON-SBERT reference representation   grid: ridge_alphas

Protocol identical to train_cv.py / evaluate_test.py:
  * the same split manifest (data/splits/<slug>_splits.csv): same dev/test rows, same rolling validation folds
  * hyper-parameters chosen by mean validation MAE over the folds (development data only)
  * the chosen setting is refit on ALL development rows and scores the held-out test set ONCE
  * the TF-IDF vocabulary / IDF weights are fit on the TRAINING rows of each fold only (never on val/test text)
  * predictions are clipped at data.sp_min, exactly like the CNN

Outputs (results/baselines/):
  baseline_results.csv        one row per project x model (validation mean/SD, test MAE/MdAE, chosen setting)
  baseline_fold_results.csv   every project x model x setting x fold
  comparison.csv              one row per project: CNN vs every baseline (test and validation MAE)
  baseline_aggregate.csv      macro averages per model + how many projects the CNN beats each baseline on
  predictions/<slug>_<model>_test_predictions.csv

Optional config (defaults shown):
  baselines:
    ridge_alphas: [0.01, 0.1, 1, 10, 100]
    knn_ks: [5, 10, 20, 50]
    tfidf: {max_features: 20000, ngram_range: [1, 2], min_df: 2, sublinear_tf: true, stop_words: english}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .make_splits import n_blocks_for, n_validation_folds
from .utils import append_experiment_log, get_logger

log = get_logger("baselines")

MODELS = ["median", "ridge_sbert", "knn_sbert", "tfidf_ridge"]
DEFAULT_ALPHAS = [0.01, 0.1, 1, 10, 100]
DEFAULT_KS = [5, 10, 20, 50]


# --------------------------------------------------------------------------- helpers
def _metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    err = np.abs(y - p)
    return float(err.mean()), float(np.median(err))


def build_fold_indices(splits: pd.DataFrame, scheme: str):
    """Identical to train_cv.build_fold_indices (copied so this stage needs no torch)."""
    fid = pd.to_numeric(splits["fold_id"], errors="coerce")
    is_dev = (splits["role"] == "dev").to_numpy()
    dev_ids = sorted(int(x) for x in fid[is_dev].dropna().unique())
    folds = dev_ids[1:] if scheme == "rolling" else dev_ids
    out = []
    for f in folds:
        if scheme == "rolling":
            tr = np.where(is_dev & (fid < f).to_numpy())[0]
        else:
            tr = np.where(is_dev & (fid != f).to_numpy() & fid.notna().to_numpy())[0]
        va = np.where(is_dev & (fid == f).to_numpy())[0]
        out.append((f, tr, va))
    return out


def _unit(A: np.ndarray) -> np.ndarray:
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def make_predictors(X, y, texts, bcfg: dict, lo: float) -> dict:
    """name -> (grid, fn(train_idx, query_idx) -> {param: predictions}). X may be None (no embeddings)."""
    alphas = [float(a) for a in bcfg.get("ridge_alphas", DEFAULT_ALPHAS)]
    ks = [int(k) for k in bcfg.get("knn_ks", DEFAULT_KS)]
    tf = dict(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
    tf.update(bcfg.get("tfidf", {}) or {})
    tf["ngram_range"] = tuple(tf["ngram_range"])
    clip = lambda d: {k: np.clip(np.asarray(v, dtype=float), lo, None) for k, v in d.items()}

    P = {"median": (["-"], lambda tr, q: clip({"-": np.full(len(q), np.median(y[tr]))}))}

    if X is not None:
        def ridge_sbert(tr, q):
            from sklearn.linear_model import Ridge
            return clip({a: Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[q]) for a in alphas})

        def knn_sbert(tr, q):
            sims = _unit(X[q]) @ _unit(X[tr]).T
            kmax = min(max(ks), len(tr))
            order = np.argsort(-sims, axis=1, kind="stable")[:, :kmax]
            nb = y[tr][order]
            return clip({k: np.median(nb[:, :min(k, kmax)], axis=1) for k in ks})

        P["ridge_sbert"] = (alphas, ridge_sbert)
        P["knn_sbert"] = (ks, knn_sbert)

    def tfidf_ridge(tr, q):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        try:
            vec = TfidfVectorizer(**tf)
            Xt = vec.fit_transform(texts[tr])                    # vocabulary + IDF from TRAINING rows only
        except ValueError:                                       # empty vocabulary (tiny fold): relax filters
            vec = TfidfVectorizer(**{**tf, "min_df": 1, "stop_words": None})
            Xt = vec.fit_transform(texts[tr])
        Xq = vec.transform(texts[q])
        return clip({a: Ridge(alpha=a).fit(Xt, y[tr]).predict(Xq) for a in alphas})

    P["tfidf_ridge"] = (alphas, tfidf_ridge)
    return P


def _merge_csv(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Write `new`; rows of other (project, model) pairs from an earlier run are kept, so --project / --models
    reruns update only what they recomputed."""
    if path.exists() and len(new):
        old = pd.read_csv(path)
        fresh = set(zip(new["project"], new["model"]))
        keep = [(p, m) not in fresh for p, m in zip(old["project"], old["model"])]
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


# --------------------------------------------------------------------------- one project
def run_project(slug: str, cfg: dict, only: list[str]) -> tuple[list[dict], list[dict]]:
    splits_path = Path(cfg["data"]["splits_dir"]) / f"{slug}_splits.csv"
    emb_path = Path(cfg["data"]["embeddings_dir"]) / f"{slug}.npz"
    scheme = str(cfg["split"]["fold_scheme"])
    splits = pd.read_csv(splits_path)

    exp_blocks = n_blocks_for(scheme, int(cfg["split"]["n_folds"]), cfg["split"].get("n_blocks"))
    if "n_blocks" not in splits.columns or str(splits["scheme"].iloc[0]) != scheme \
            or int(splits["n_blocks"].iloc[0]) != exp_blocks:
        log.error("  %s: manifest does not match the config - re-run make_splits.", slug)
        return [], []
    if "text" not in splits.columns:
        log.error("  %s: manifest has no text column.", slug)
        return [], []

    y = splits["story_point"].to_numpy(dtype=float)
    texts = splits["text"].fillna("").astype(str).to_numpy(dtype=object)
    keys = splits["issue_key"].astype(str).to_numpy()

    X = None
    if emb_path.exists():
        with np.load(emb_path, allow_pickle=False) as z:
            pos = {k: i for i, k in enumerate(z["issue_key"].astype(str))}
            if all(k in pos for k in keys):
                X = np.asarray(z["vector"], dtype=np.float64)[[pos[k] for k in keys]]
            else:
                log.warning("  %s: embeddings do not cover the manifest - SBERT baselines skipped.", slug)
    else:
        log.warning("  %s: no embeddings - SBERT baselines skipped.", slug)

    dev_idx = np.where(splits["role"] == "dev")[0]
    test_idx = np.where(splits["role"] == "test")[0]
    ca = pd.to_datetime(splits["created_at"], utc=True)
    if len(dev_idx) == 0 or len(test_idx) == 0 or ca.iloc[dev_idx].max() > ca.iloc[test_idx].min():
        log.error("  %s: empty dev/test or test is not newer than dev.", slug)
        return [], []
    folds = [(f, tr, va) for f, tr, va in build_fold_indices(splits, scheme) if len(tr) and len(va)]
    if len(folds) != n_validation_folds(scheme, exp_blocks):
        log.warning("  %s: only %d usable validation folds.", slug, len(folds))
    if not folds:
        return [], []

    lo = float(cfg["data"].get("sp_min", 0) or 0)
    predictors = make_predictors(X, y, texts, cfg.get("baselines", {}) or {}, lo)
    pred_dir = Path(cfg["results"].get("baselines_dir", "results/baselines")) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    res_rows, fold_rows = [], []
    for name, (grid, fn) in predictors.items():
        if only and name not in only:
            continue
        scores = {p: [] for p in grid}
        for f, tr, va in folds:
            out = fn(tr, va)
            for p in grid:
                mae, mdae = _metrics(y[va], out[p])
                scores[p].append((mae, mdae))
                fold_rows.append({"project": slug, "model": name, "param": p, "fold": f,
                                  "n_train": len(tr), "n_val": len(va), "MAE": mae, "MdAE": mdae})
        mean_mae = {p: float(np.mean([s[0] for s in scores[p]])) for p in grid}
        best = min(grid, key=lambda p: mean_mae[p])                # ties -> first (the first grid value)
        sc = np.array(scores[best])
        test_pred = fn(dev_idx, test_idx)[best]                     # refit on ALL dev rows, score test once
        t_mae, t_mdae = _metrics(y[test_idx], test_pred)
        res_rows.append({"project": slug, "model": name, "best_param": best, "n_dev": len(dev_idx),
                         "n_test": len(test_idx), "n_val_folds": len(folds),
                         "val_MAE_mean": sc[:, 0].mean(), "val_MAE_sd": sc[:, 0].std(ddof=1) if len(sc) > 1 else np.nan,
                         "val_MdAE_mean": sc[:, 1].mean(), "val_MdAE_sd": sc[:, 1].std(ddof=1) if len(sc) > 1 else np.nan,
                         "test_MAE": t_mae, "test_MdAE": t_mdae, "fold_scheme": scheme})
        pd.DataFrame({"project": slug, "model": name, "issue_key": keys[test_idx], "created_at": ca.iloc[test_idx].astype(str).values,
                      "y_true": y[test_idx], "y_pred": test_pred}).to_csv(pred_dir / f"{slug}_{name}_test_predictions.csv", index=False)
        log.info("  %-12s best=%-6s val MAE=%.3f | TEST MAE=%.3f MdAE=%.3f", name, best, sc[:, 0].mean(), t_mae, t_mdae)
    return res_rows, fold_rows


# --------------------------------------------------------------------------- comparison with the CNN
def build_comparison(cfg: dict, res: pd.DataFrame, out_dir: Path) -> None:
    rel_min = int(cfg.get("eval", {}).get("min_test_issues_reliable", 30))
    wide = res.pivot(index="project", columns="model", values="test_MAE").add_suffix("_test_MAE")
    wide = wide.join(res.pivot(index="project", columns="model", values="test_MdAE").add_suffix("_test_MdAE"))
    wide = wide.join(res.pivot(index="project", columns="model", values="val_MAE_mean").add_suffix("_val_MAE"))
    wide = wide.join(res.groupby("project")["n_test"].first()).reset_index()

    tdir = Path(cfg["results"]["test_dir"])
    cnn = [pd.read_csv(p).assign(project=p.stem) for p in sorted(tdir.glob("*.csv"))]
    if cnn:
        cnn = pd.concat(cnn, ignore_index=True)[["project", "test_MAE", "test_MdAE"]].rename(
            columns={"test_MAE": "cnn_test_MAE", "test_MdAE": "cnn_test_MdAE"})
        wide = wide.merge(cnn, on="project", how="left")
    vals = []                                               # CNN validation MAE straight from the fold files
    for p in sorted(Path(cfg["results"]["five_fold_dir"]).glob("*.csv")):
        if p.stem.endswith("_val_predictions"):
            continue
        d = pd.read_csv(p)
        if {"lr", "MAE"} <= set(d.columns):
            vals.append({"project": p.stem, "cnn_val_MAE": float(d.groupby("lr")["MAE"].mean().min())})
    if vals:
        wide = wide.merge(pd.DataFrame(vals), on="project", how="left")
    wide["reliable"] = wide["n_test"] >= rel_min
    wide.to_csv(out_dir / "comparison.csv", index=False)

    rows = []
    have_cnn = "cnn_test_MAE" in wide
    full = wide.dropna(subset=["cnn_test_MAE"]) if have_cnn else wide.iloc[0:0]
    for m in (["cnn"] if have_cnn else []) + MODELS:
        col = f"{m}_test_MAE"
        if col not in wide:
            continue
        sub = full if have_cnn else wide
        sub = sub.dropna(subset=[col])
        r = {"model": m, "n_projects": len(sub), "test_MAE_macro": sub[col].mean(),
             "test_MdAE_macro": sub[f"{m}_test_MdAE"].mean() if f"{m}_test_MdAE" in sub else np.nan,
             "val_MAE_macro": sub[f"{m}_val_MAE"].mean() if f"{m}_val_MAE" in sub else np.nan}
        if have_cnn and m != "cnn" and len(sub):
            r["cnn_beats_this_(test_MAE)"] = int((sub["cnn_test_MAE"] < sub[col]).sum())
            rel = sub[sub["reliable"]]
            r["cnn_beats_this_reliable_only"] = int((rel["cnn_test_MAE"] < rel[col]).sum())
            r["n_reliable"] = len(rel)
        rows.append(r)
    agg = pd.DataFrame(rows)
    agg.to_csv(out_dir / "baseline_aggregate.csv", index=False)
    log.info("aggregate (macro over projects that have a CNN result):\n%s",
             agg.round(3).to_string(index=False))


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--models", nargs="*", choices=MODELS, default=None, help="subset of models")
    args = ap.parse_args()
    cfg = load_config(args.config)

    splits_dir = Path(cfg["data"]["splits_dir"])
    slugs = sorted(p.name[:-len("_splits.csv")] for p in splits_dir.glob("*_splits.csv"))
    if args.project:
        slugs = [s for s in slugs if s == args.project]
    if not slugs:
        log.error("No split manifests in %s - run make_splits first.", splits_dir)
        return 1

    out_dir = Path(cfg["results"].get("baselines_dir", "results/baselines"))
    out_dir.mkdir(parents=True, exist_ok=True)
    all_res, all_fold, done = [], [], set()
    for s in slugs:
        log.info("=== Project: %s ===", s)
        r, f = run_project(s, cfg, args.models or [])
        if r:
            all_res += r; all_fold += f; done.add(s)

    if not all_res:
        log.error("No baseline results produced.")
        return 1
    res = _merge_csv(out_dir / "baseline_results.csv", pd.DataFrame(all_res))
    _merge_csv(out_dir / "baseline_fold_results.csv", pd.DataFrame(all_fold))
    build_comparison(cfg, res, out_dir)
    append_experiment_log(cfg["results"]["experiment_log"], config=cfg, stage="baselines",
                          notes=f"{len(done)} project(s); models={args.models or MODELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())