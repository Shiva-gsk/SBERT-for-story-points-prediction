"""Stage 5: chronological 5-fold validation with the small HP grid.

For every eligible project:
  * load SBERT .npz + the temporal split manifest (made by make_splits.py with the SAME fold_scheme)
  * for each lr in train.lr_grid and each validation fold:
        train the CNN on the fold's training rows
        early-stop on an INNER chronological hold-out (last train.inner_val_frac of the training rows),
        restore the weights of the best inner epoch, then score the validation fold ONCE.
        If there is too little data for an inner hold-out, train a FIXED number of epochs
        (train.fallback_epochs, default train.min_epochs) and score the validation fold once at the end.
        The validation fold therefore never influences epoch selection in any case, so every fold's
        MAE is unbiased and safe to average when picking the best lr.
  * write results/five_fold/<slug>.csv (one row per lr x fold) and
    results/five_fold/predictions/<slug>_val_predictions.csv (subfolder, so *.csv globs never mix them)

Model selection (best lr) and the final held-out test run live in evaluate_test.py.
For the final model there, retrain on all dev rows for the MEDIAN `best_epoch` of the winning lr
(rows with early_stop_source == "inner_holdout"), rather than training until patience runs out.
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .dataset import EmbeddingDataset, load_npz
from .make_splits import n_blocks_for, n_validation_folds
from .model import build_model
from .utils import append_experiment_log, get_logger, resolve_device, set_global_seed

log = get_logger("train_cv")

MIN_INNER = 30          # minimum rows for the inner early-stopping hold-out (and what must remain to fit)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.abs(y_true - y_pred)
    return {"MAE": float(np.mean(err)), "MdAE": float(np.median(err))}


def target_shift(sp: np.ndarray, idx: np.ndarray, log1p: bool) -> float:
    """Targets are trained as (story_point - median of the TRAINING rows), so the untrained network already
    predicts the training median (the optimal constant for MAE) instead of ~0, which clipping at sp_min
    would turn into a constant 1. Not used with log1p."""
    return 0.0 if log1p else float(np.median(sp[idx]))


def eff_batch_size(n_rows: int, bs: int) -> int:
    """Cap the batch size so tiny training sets still get ~4 optimiser steps per epoch."""
    return int(min(bs, max(8, int(np.ceil(n_rows / 4)))))


def _postprocess(pred: np.ndarray, y: np.ndarray, log1p: bool, lo: float, shift: float = 0.0):
    """Back-transform (log1p) or undo the centring shift, then clip predictions at the smallest valid story point."""
    pred, y = pred.reshape(-1), y.reshape(-1)
    if log1p:
        pred, y = np.expm1(pred), np.expm1(y)
    else:
        pred, y = pred + shift, y + shift
    return np.clip(pred, lo, None), y


def _predict(model, dl, device, log1p: bool, lo: float, shift: float = 0.0):
    import torch
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yb in dl:
            ps.append(model(xb.to(device)).cpu().numpy())
            ys.append(yb.numpy())
    return _postprocess(np.concatenate(ps), np.concatenate(ys), log1p, lo, shift)


def build_fold_indices(splits: pd.DataFrame, scheme: str):
    """Return [(fold, train_idx, val_idx), ...] with indices into the manifest's row order.

    fold_id is 'test' for held-out rows, so it is converted with to_numeric (NaN for test); the original
    `.astype(int)` on the whole column raised ValueError.
    """
    fid = pd.to_numeric(splits["fold_id"], errors="coerce")
    is_dev = (splits["role"] == "dev").to_numpy()
    dev_ids = sorted(int(x) for x in fid[is_dev].dropna().unique())
    if scheme == "rolling":
        folds = dev_ids[1:]                  # block 0 is train-only
    elif scheme == "blocked":
        folds = dev_ids
    else:
        raise ValueError(f"unknown fold_scheme {scheme!r}")
    out = []
    for f in folds:
        if scheme == "rolling":
            tr = np.where(is_dev & (fid < f).to_numpy())[0]
        else:
            tr = np.where(is_dev & (fid != f).to_numpy() & fid.notna().to_numpy())[0]
        va = np.where(is_dev & (fid == f).to_numpy())[0]
        out.append((f, tr, va))
    return out


def _train_one_fold(vectors, sp, train_idx, val_idx, cfg, device, lr):
    """Train one fold. Returns dict(metrics, best_epoch, es_mae, source, val_pred, val_true, n_fit, bs, shift).

    The validation fold is scored exactly once, after training has finished, so it cannot influence
    epoch selection.
    """
    import torch
    from torch.utils.data import DataLoader

    set_global_seed(cfg["seed"])
    log1p = bool(cfg["model"].get("log1p_target", False))
    lo = float(cfg["data"].get("sp_min", 0) or 0)
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    min_epochs = int(cfg["train"].get("min_epochs", 30))
    max_epochs = int(cfg["train"]["max_epochs"])

    # --- inner chronological hold-out for early stopping (train_idx is ascending = chronological) ---
    inner_frac = float(cfg["train"].get("inner_val_frac", 0.15))
    n_tr = len(train_idx)
    n_inner = int(round(n_tr * inner_frac))
    use_inner = inner_frac > 0 and n_inner >= MIN_INNER and (n_tr - n_inner) >= MIN_INNER
    if use_inner:
        fit_idx, es_idx = train_idx[:-n_inner], train_idx[-n_inner:]
        source = "inner_holdout"
        n_epochs = max_epochs
    else:
        # Too little data to early-stop reliably: fixed epoch budget, no peeking at the validation fold.
        fit_idx, es_idx = train_idx, None
        source = "fixed_epochs"
        n_epochs = min(int(cfg["train"].get("fallback_epochs", min_epochs)), max_epochs)

    shift = target_shift(sp, fit_idx, log1p)
    bs = eff_batch_size(len(fit_idx), bs)
    sp_c = sp - np.float32(shift)
    mk = lambda idx: EmbeddingDataset(vectors[idx], sp_c[idx], log1p=log1p)
    fit_ds = mk(fit_idx)
    drop_last = len(fit_ds) > bs and len(fit_ds) % bs == 1          # avoid a size-1 batch (BatchNorm)
    fit_dl = DataLoader(fit_ds, batch_size=bs, shuffle=True, drop_last=drop_last, num_workers=nw)
    val_dl = DataLoader(mk(val_idx), batch_size=bs * 2, shuffle=False, num_workers=nw)
    es_dl = DataLoader(mk(es_idx), batch_size=bs * 2, shuffle=False, num_workers=nw) if use_inner else None

    model = build_model(cfg, vectors.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(cfg["train"]["weight_decay"]))
    loss_fn = torch.nn.L1Loss() if cfg["train"]["loss"] == "l1" else torch.nn.MSELoss()

    patience = int(cfg["train"]["patience"])
    best_es, best_epoch, no_improve = float("inf"), -1, 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in fit_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        if not use_inner:
            continue

        e_pred, e_true = _predict(model, es_dl, device, log1p, lo, shift)           # drives early stopping
        es_mae = _metrics(e_true, e_pred)["MAE"]
        if es_mae < best_es - 1e-9:
            best_es, best_epoch, no_improve = es_mae, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience and epoch >= min_epochs:
                break

    if use_inner:
        if best_state is None:
            raise RuntimeError("training never produced a finite inner-holdout MAE (NaN loss?)")
        model.load_state_dict(best_state)
    else:
        best_epoch, best_es = n_epochs, float("nan")

    # Single, final look at the validation fold.
    v_pred, v_true = _predict(model, val_dl, device, log1p, lo, shift)
    if not np.all(np.isfinite(v_pred)):
        raise RuntimeError("non-finite validation predictions (NaN loss?)")
    return {"metrics": _metrics(v_true, v_pred), "best_epoch": best_epoch, "es_mae": best_es,
            "source": source, "val_pred": v_pred, "val_true": v_true, "n_fit": int(len(fit_idx)),
            "bs": bs, "shift": shift}


def run_project(slug: str, cfg: dict, device: str) -> int:
    emb_path = Path(cfg["data"]["embeddings_dir"]) / f"{slug}.npz"
    split_path = Path(cfg["data"]["splits_dir"]) / f"{slug}_splits.csv"
    if not emb_path.exists() or not split_path.exists():
        log.error("  %s: missing embeddings or splits - run earlier stages first.", slug)
        return 1

    data = load_npz(emb_path)
    splits = pd.read_csv(split_path)                      # already chronological; do NOT re-sort
    scheme = str(cfg["split"]["fold_scheme"])
    n_folds = int(cfg["split"]["n_folds"])
    exp_blocks = n_blocks_for(scheme, n_folds, cfg["split"].get("n_blocks"))
    if "n_blocks" not in splits.columns or "scheme" not in splits.columns:
        log.error("  %s: split manifest is from an old make_splits version (no n_blocks/scheme column) - "
                  "delete data/splits and re-run make_splits.", slug)
        return 1
    if str(splits["scheme"].iloc[0]) != scheme or int(splits["n_blocks"].iloc[0]) != exp_blocks:
        log.error("  %s: manifest has scheme=%s, n_blocks=%s but config implies scheme=%s, n_blocks=%d - re-run make_splits.",
                  slug, splits["scheme"].iloc[0], splits["n_blocks"].iloc[0], scheme, exp_blocks)
        return 1
    if scheme == "blocked":
        log.warning("  %s: fold_scheme='blocked' trains on blocks that come AFTER the validation fold; "
                    "these scores are not strictly temporal (future-to-past leakage). Use 'rolling' for "
                    "a true temporal estimate.", slug)

    # Align embeddings to the manifest by issue_key (never by position).
    keys_split = splits["issue_key"].astype(str).to_numpy()
    keys_emb = np.asarray(data["issue_key"]).astype(str)
    pos = {k: i for i, k in enumerate(keys_emb)}
    missing = [k for k in keys_split if k not in pos]
    if missing:
        log.error("  %s: %d manifest issues have no embedding (e.g. %s) - re-run extract_embeddings.",
                  slug, len(missing), missing[:3])
        return 1
    idx = np.array([pos[k] for k in keys_split])
    vectors = np.asarray(data["vector"])[idx]
    sp = np.asarray(data["story_point"]).astype(np.float32)[idx]

    fold_defs = build_fold_indices(splits, scheme)
    exp_folds = n_validation_folds(scheme, exp_blocks)
    if len(fold_defs) != exp_folds:
        log.error("  %s: manifest yields %d validation folds but the config implies %d - re-run make_splits.",
                  slug, len(fold_defs), exp_folds)
        return 1
    if exp_folds != n_folds:
        log.warning("  %s: running %d validation folds (split.n_folds=%d)", slug, exp_folds, n_folds)

    out_dir = Path(cfg["results"]["five_fold_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, pred_rows = [], []
    for lr in cfg["train"]["lr_grid"]:
        for f, train_idx, val_idx in fold_defs:
            if len(train_idx) == 0 or len(val_idx) == 0:
                log.warning("  %s: lr=%s fold=%d empty split - skipping", slug, lr, f)
                continue
            log.info("  %s: lr=%s fold=%d (train=%d, val=%d) training ...", slug, lr, f, len(train_idx), len(val_idx))
            r = _train_one_fold(vectors, sp, train_idx, val_idx, cfg, device, lr)
            m = r["metrics"]
            rows.append({"project": slug, "fold": f, "lr": lr,
                         "n_train": int(len(train_idx)), "n_fit": r["n_fit"], "n_val": int(len(val_idx)),
                         "MAE": m["MAE"], "MdAE": m["MdAE"], "best_epoch": r["best_epoch"],
                         "best_es_MAE": r["es_mae"],
                         "early_stop_source": r["source"], "batch_size_eff": r["bs"], "target_shift": r["shift"],
                         "seed": cfg["seed"],
                         "fold_scheme": scheme, "sbert_model": cfg["sbert"]["model_name"]})
            for k, yt, yp in zip(keys_split[val_idx], r["val_true"], r["val_pred"]):
                pred_rows.append({"project": slug, "lr": lr, "fold": f, "issue_key": k,
                                  "y_true": float(yt), "y_pred": float(yp)})
            log.info("    -> MAE=%.4f MdAE=%.4f epoch=%d (%s)",
                     m["MAE"], m["MdAE"], r["best_epoch"], r["source"])

    if not rows:
        log.warning("  %s: no folds were trained - nothing written.", slug)
        return 1

    out_path = out_dir / f"{slug}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(pred_dir / f"{slug}_val_predictions.csv", index=False)
    log.info("  wrote %s (%d rows) + predictions", out_path.name, len(rows))
    append_experiment_log(cfg["results"]["experiment_log"], config=cfg, stage="train_cv", project=slug,
                          notes=f"{len(rows)} fold rows; scheme={scheme}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    device = resolve_device(cfg["device"])
    set_global_seed(cfg["seed"])

    emb_dir = Path(cfg["data"]["embeddings_dir"])
    npzs = sorted(emb_dir.glob("*.npz"))
    if args.project:
        npzs = [p for p in npzs if p.stem == args.project]
    if not npzs:
        log.error("No embeddings in %s.", emb_dir)
        return 1

    n_ok = 0
    for p in npzs:
        log.info("=== Project: %s ===", p.stem)
        if run_project(p.stem, cfg, device) == 0:
            n_ok += 1
    log.info("Trained %d project(s).", n_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())