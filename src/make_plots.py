"""Stage 8: render plots to results/plots/ from the summary tables (run after src.summarize).

CNN detail
  per_project_test_mae_mdae.png     CNN test MAE vs MdAE per project
  val_vs_test_mae.png               CNN validation MAE (mean +- SD over folds) vs test MAE vs median baseline
  fold_mae_boxplot.png              distribution of CNN fold MAE per project (selected lr)
  pooled_test_scatter.png           CNN actual vs predicted story points, all test issues
Model comparison (CNN vs ridge / kNN / TF-IDF / median, same splits)
  model_comparison_macro.png        macro test MAE & MdAE per model  +  validation vs test MAE per model
  models_test_mae_by_project.png    test MAE of every model, per project
  relative_to_median_heatmap.png    model MAE / median-predictor MAE per project (< 1 = better than the median)
  paired_difference.png             baseline MAE - CNN MAE per project (> 0 = CNN better)
  pooled_scatter_by_model.png       actual vs predicted per model (shows predictions collapsing to a constant)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .summarize import LABEL, MODEL_ORDER
from .utils import append_experiment_log, get_logger

log = get_logger("make_plots")
plt.rcParams["axes.unicode_minus"] = False
COLOR = dict(zip(MODEL_ORDER, plt.rcParams["axes.prop_cycle"].by_key()["color"]))


def _w(n: int) -> float:
    return max(8, 0.45 * n + 4)


def _save(fig, path: Path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", path.name)


def _read(p: Path):
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 2 else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    sdir = Path(cfg["results"]["summary_dir"])
    pdir = Path(cfg["results"]["plots_dir"])
    pdir.mkdir(parents=True, exist_ok=True)
    tdir = Path(cfg["results"]["test_dir"])
    bdir = Path(cfg["results"].get("baselines_dir", "results/baselines"))

    # ================= CNN detail =================
    P = _read(sdir / "project_results.csv")
    T = P.sort_values("project").dropna(subset=["test_MAE"]) if len(P) and "test_MAE" in P else pd.DataFrame()
    if len(T):
        x, w = np.arange(len(T)), 0.38
        fig, ax = plt.subplots(figsize=(_w(len(T)), 5), constrained_layout=True)
        ax.bar(x - w / 2, T["test_MAE"], w, label="MAE")
        ax.bar(x + w / 2, T["test_MdAE"], w, label="MdAE")
        ax.set_xticks(x); ax.set_xticklabels(T["project"], rotation=45, ha="right")
        ax.set_ylabel("Test error (story points)")
        ax.set_title("PS-6: SBERT + CNN, within-project temporal split - test MAE vs MdAE")
        ax.legend()
        _save(fig, pdir / "per_project_test_mae_mdae.png")

        fig, ax = plt.subplots(figsize=(_w(len(T)), 5), constrained_layout=True)
        ax.errorbar(x, T["val_MAE_mean"], yerr=T["val_MAE_sd"].fillna(0), fmt="o", capsize=3, label="Validation MAE (mean ± SD)")
        ax.scatter(x, T["test_MAE"], marker="s", label="Test MAE", zorder=3)
        if "baseline_median_MAE" in T:
            ax.scatter(x, T["baseline_median_MAE"], marker="x", color="gray", label="Median-predictor baseline (test)")
        ax.set_xticks(x); ax.set_xticklabels(T["project"], rotation=45, ha="right")
        ax.set_ylabel("MAE (story points)")
        ax.set_title("CNN: validation vs held-out test MAE (a large gap suggests temporal drift)")
        ax.legend()
        _save(fig, pdir / "val_vs_test_mae.png")

    ff = _read(sdir / "five_fold_validation_results.csv")
    if len(ff):
        projects = sorted(ff["project"].unique())
        data = [ff.loc[ff["project"] == p, "MAE"].values for p in projects]
        fig, ax = plt.subplots(figsize=(_w(len(projects)), 5), constrained_layout=True)
        try:
            ax.boxplot(data, tick_labels=projects, showmeans=True)      # matplotlib >= 3.9
        except TypeError:
            ax.boxplot(data, labels=projects, showmeans=True)
        ax.set_xticklabels(projects, rotation=45, ha="right")
        ax.set_ylabel("Validation MAE (story points)")
        ax.set_title("CNN: chronological validation MAE per project (selected lr)")
        _save(fig, pdir / "fold_mae_boxplot.png")

    pf = sorted((tdir / "predictions").glob("*_test_predictions.csv"))
    if pf:
        D = pd.concat([pd.read_csv(p) for p in pf], ignore_index=True)
        fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
        ax.scatter(D["y_true"], D["y_pred"], s=8, alpha=0.3)
        lim = float(max(D["y_true"].max(), D["y_pred"].max()))
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlabel("Actual story points"); ax.set_ylabel("Predicted story points")
        ax.set_title(f"CNN held-out test, all projects (n={len(D)})")
        _save(fig, pdir / "pooled_test_scatter.png")

    # ================= model comparison =================
    CT = _read(sdir / "comparative_table.csv")
    PB = _read(sdir / "model_comparison_by_project.csv")
    if len(CT) < 2:
        log.warning("No multi-model comparison available (run src.baselines, then src.summarize).")
    else:
        keys = CT["model_key"].tolist()
        n_common = int(CT["n_projects"].iloc[0])

        # --- macro comparison + validation vs test ---
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
        x, w = np.arange(len(CT)), 0.38
        b1 = a1.bar(x - w / 2, CT["Test_MAE_macro"], w, label="Test MAE", color=[COLOR[k] for k in keys])
        b2 = a1.bar(x + w / 2, CT["Test_MdAE_macro"], w, label="Test MdAE", color=[COLOR[k] for k in keys], alpha=0.5, hatch="//")
        for bars in (b1, b2):
            for r in bars:
                a1.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                            ha="center", va="bottom", fontsize=8)
        a1.set_xticks(x); a1.set_xticklabels(CT["Model"], rotation=20, ha="right")
        a1.set_ylabel("Error (story points), macro-average over projects")
        a1.set_title(f"Held-out test error per model ({n_common} common projects)")
        a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="gray"), plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.5, hatch="//")],
                  labels=["MAE", "MdAE"])
        if CT["Val_MAE_mean"].notna().any():
            a2.bar(x - w / 2, CT["Val_MAE_mean"], w, label="Validation MAE (rolling folds)", color="lightgray", edgecolor="black")
            a2.bar(x + w / 2, CT["Test_MAE_macro"], w, label="Test MAE", color=[COLOR[k] for k in keys])
            a2.set_xticks(x); a2.set_xticklabels(CT["Model"], rotation=20, ha="right")
            a2.set_ylabel("MAE (story points)")
            a2.set_title("Validation vs test MAE per model (gap = drift or overfit)")
            a2.set_ylim(0, float(max(CT["Val_MAE_mean"].max(), CT["Test_MAE_macro"].max())) * 1.3)   # room for the legend
            a2.legend(loc="upper center", ncol=2)
        else:
            a2.axis("off")
        _save(fig, pdir / "model_comparison_macro.png")

    if len(PB) and len(CT) >= 2:
        keys = [k for k in CT["model_key"] if f"{k}_test_MAE" in PB]
        order = [k for k in MODEL_ORDER if k in keys]
        B = PB.sort_values("project").reset_index(drop=True)

        # --- per-project grouped bars ---
        x, w = np.arange(len(B)), 0.8 / len(order)
        fig, ax = plt.subplots(figsize=(_w(len(B)) * 1.2, 5), constrained_layout=True)
        for i, k in enumerate(order):
            ax.bar(x + (i - (len(order) - 1) / 2) * w, B[f"{k}_test_MAE"], w, label=LABEL[k], color=COLOR[k])
        ax.set_xticks(x); ax.set_xticklabels(B["project"], rotation=45, ha="right")
        ax.set_ylabel("Test MAE (story points)")
        ax.set_title("Held-out test MAE per project, all models (same temporal splits)")
        ax.legend(ncol=len(order))
        _save(fig, pdir / "models_test_mae_by_project.png")

        # --- heatmap relative to the median predictor ---
        if "median" in order and len(order) > 1:
            cols = [k for k in order if k != "median"]
            M = np.column_stack([B[f"{k}_test_MAE"] / B["median_test_MAE"] for k in cols])
            fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 3, max(4, 0.32 * len(B) + 1.5)), constrained_layout=True)
            im = ax.imshow(M, cmap="RdYlGn_r", vmin=0.5, vmax=1.5, aspect="auto")
            ax.set_xticks(range(len(cols))); ax.set_xticklabels([LABEL[k] for k in cols], rotation=20, ha="right")
            ax.set_yticks(range(len(B))); ax.set_yticklabels(B["project"], fontsize=8)
            if len(B) <= 45:
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        if np.isfinite(M[i, j]):
                            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, label="test MAE / median-predictor MAE  (< 1 = beats the median)")
            ax.set_title("Is each model better than just predicting the median?")
            _save(fig, pdir / "relative_to_median_heatmap.png")

        # --- paired differences vs CNN ---
        others = [k for k in order if k != "cnn"]
        if "cnn" in order and others:
            fig, ax = plt.subplots(figsize=(1.8 * len(others) + 3, 5), constrained_layout=True)
            rng = np.random.default_rng(0)
            for i, k in enumerate(others):
                d = (B[f"{k}_test_MAE"] - B["cnn_test_MAE"]).dropna().to_numpy()
                ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=22, alpha=0.7, color=COLOR[k])
                if len(d):
                    ax.hlines(np.median(d), i - 0.3, i + 0.3, color="black", lw=2)
            ax.axhline(0, color="gray", ls="--", lw=1)
            ax.set_xticks(range(len(others))); ax.set_xticklabels([LABEL[k] for k in others], rotation=15, ha="right")
            ax.set_ylabel("baseline MAE − CNN MAE   (> 0 = CNN better)")
            ax.set_title("Per-project paired difference vs the CNN (black bar = median)")
            _save(fig, pdir / "paired_difference.png")

        # --- pooled actual vs predicted, per model ---
        panels = []
        for k in order:
            files = (sorted((tdir / "predictions").glob("*_test_predictions.csv")) if k == "cnn"
                     else sorted((bdir / "predictions").glob(f"*_{k}_test_predictions.csv")))
            if files:
                panels.append((k, pd.concat([pd.read_csv(f) for f in files], ignore_index=True)))
        if panels:
            lim = max(max(d["y_true"].max(), d["y_pred"].max()) for _, d in panels)
            fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.9), constrained_layout=True, squeeze=False)
            for ax, (k, d) in zip(axes[0], panels):
                ax.scatter(d["y_true"], d["y_pred"], s=6, alpha=0.25, color=COLOR[k])
                ax.plot([0, lim], [0, lim], "r--", lw=1)
                ax.set_title(LABEL[k], fontsize=10); ax.set_xlabel("Actual")
            axes[0][0].set_ylabel("Predicted")
            _save(fig, pdir / "pooled_scatter_by_model.png")

    append_experiment_log(cfg["results"]["experiment_log"], config=cfg, stage="make_plots", notes="plots rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())