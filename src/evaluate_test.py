"""Stage 6: held-out test evaluation.

For every project:
  * select the lr with the lowest mean validation MAE (results/five_fold/<slug>.csv) and the epoch count =
    median best_epoch of that lr's folds  (development data only; the test set is never touched before this)
  * retrain a fresh model on the FULL development set for exactly that many epochs (no early stopping,
    no test-set feedback)
  * predict the newest-20% test set once, with the same back-transform + clipping used in train_cv
  * write results/test/<slug>.csv (+ results/test/predictions/<slug>_test_predictions.csv)
  * save the final model to results/models/<slug>.pt (used by src/predict.py)
  * report two trivial baselines (dev-mean, dev-median predictor) so the model can be judged against them.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .dataset import EmbeddingDataset, load_npz
from .model import build_model
from .train_cv import _metrics, _predict, eff_batch_size, target_shift
from .utils import append_experiment_log, get_logger, resolve_device, set_global_seed

log = get_logger("evaluate_test")


def select_config(fold_csv: Path) -> tuple[float, int, float, str]:
    """(best_lr, retrain_epochs, mean_val_MAE_of_best_lr, epochs_basis), from validation folds only.

    Epoch count = median best_epoch over the winning lr's folds that really early-stopped
    (early_stop_source == "inner_holdout"). Folds that trained a FIXED budget just echo that budget, so they
    are only used when no fold early-stopped (then the fixed budget itself is reused).
    """
    df = pd.read_csv(fold_csv)
    g = df.groupby("lr")["MAE"].mean()
    best_lr = float(g.idxmin())
    sel = df[np.isclose(df["lr"].astype(float), best_lr)]
    if "early_stop_source" in sel.columns and (sel["early_stop_source"] == "inner_holdout").any():
        ep, basis = sel.loc[sel["early_stop_source"] == "inner_holdout", "best_epoch"], "inner_holdout"
    else:
        ep, basis = sel["best_epoch"], "fixed_epochs"
    return best_lr, max(1, int(round(float(ep.median())))), float(g.min()), basis


def _fit_fixed_epochs(vectors, sp, idx, cfg, device, lr, epochs, shift):
    import torch
    from torch.utils.data import DataLoader
    set_global_seed(cfg["seed"])
    log1p = bool(cfg["model"].get("log1p_target", False))
    bs = eff_batch_size(len(idx), int(cfg["train"]["batch_size"]))
    ds = EmbeddingDataset(vectors[idx], sp[idx] - np.float32(shift), log1p=log1p)
    drop_last = len(ds) > bs and len(ds) % bs == 1
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=drop_last,
                    num_workers=int(cfg["train"]["num_workers"]))
    model = build_model(cfg, vectors.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(cfg["train"]["weight_decay"]))
    loss_fn = torch.nn.L1Loss() if cfg["train"]["loss"] == "l1" else torch.nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


def run_project(slug: str, cfg: dict, device: str) -> int:
    from torch.utils.data import DataLoader
    emb_path = Path(cfg["data"]["embeddings_dir"]) / f"{slug}.npz"
    split_path = Path(cfg["data"]["splits_dir"]) / f"{slug}_splits.csv"
    fold_csv = Path(cfg["results"]["five_fold_dir"]) / f"{slug}.csv"
    if not (emb_path.exists() and split_path.exists() and fold_csv.exists()):
        log.error("  %s: missing inputs (embeddings/splits/folds).", slug)
        return 1

    data = load_npz(emb_path)
    splits = pd.read_csv(split_path)                       # already chronological; do not re-sort
    scheme = str(cfg["split"]["fold_scheme"])
    if "scheme" in splits.columns and str(splits["scheme"].iloc[0]) != scheme:
        log.error("  %s: manifest scheme=%s != config %s - re-run make_splits/train_cv.", slug,
                  splits["scheme"].iloc[0], scheme)
        return 1

    keys_split = splits["issue_key"].astype(str).to_numpy()
    pos = {k: i for i, k in enumerate(np.asarray(data["issue_key"]).astype(str))}
    if any(k not in pos for k in keys_split):
        log.error("  %s: manifest issues missing from embeddings - re-run extract_embeddings.", slug)
        return 1
    idx = np.array([pos[k] for k in keys_split])
    vectors = np.asarray(data["vector"])[idx]
    sp = np.asarray(data["story_point"]).astype(np.float32)[idx]

    dev_idx = np.where(splits["role"] == "dev")[0]
    test_idx = np.where(splits["role"] == "test")[0]
    if len(dev_idx) == 0 or len(test_idx) == 0:
        log.error("  %s: empty dev or test set.", slug)
        return 1
    ca = pd.to_datetime(splits["created_at"], utc=True)
    if ca.iloc[dev_idx].max() > ca.iloc[test_idx].min():
        log.error("  %s: test issues are not all newer than dev issues - split is not temporal.", slug)
        return 1

    best_lr, epochs, val_mae, basis = select_config(fold_csv)
    log.info("  %s: selected lr=%.0e, retrain_epochs=%d [%s] (mean val MAE=%.4f)", slug, best_lr, epochs, basis, val_mae)

    log1p = bool(cfg["model"].get("log1p_target", False))
    lo = float(cfg["data"].get("sp_min", 0) or 0)
    shift = target_shift(sp, dev_idx, log1p)            # median of the DEVELOPMENT targets only
    model = _fit_fixed_epochs(vectors, sp, dev_idx, cfg, device, best_lr, epochs, shift)

    # ---- save the final model + everything predict.py needs (demo / reuse) ----
    import torch
    models_dir = Path(cfg["results"].get("models_dir", "results/models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"project": slug,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "input_dim": int(vectors.shape[1]), "model_cfg": cfg["model"], "sbert": cfg["sbert"],
                "target_shift": float(shift), "log1p": log1p, "sp_min": lo,
                "best_lr": float(best_lr), "epochs": int(epochs), "seed": int(cfg["seed"]),
                "n_dev": int(len(dev_idx)), "dev_median": float(np.median(sp[dev_idx])),
                "dev_end": str(ca.iloc[dev_idx].max())}, models_dir / f"{slug}.pt")

    test_dl = DataLoader(EmbeddingDataset(vectors[test_idx], sp[test_idx] - np.float32(shift), log1p=log1p),
                         batch_size=int(cfg["train"]["batch_size"]) * 2, shuffle=False,
                         num_workers=int(cfg["train"]["num_workers"]))
    preds, ys = _predict(model, test_dl, device, log1p, lo, shift)
    test_m = _metrics(ys, preds)

    dev_sp = sp[dev_idx]
    b_med = _metrics(ys, np.full_like(ys, np.median(dev_sp)))
    b_mean = _metrics(ys, np.full_like(ys, dev_sp.mean()))

    out_dir = Path(cfg["results"]["test_dir"])
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"project": slug, "issue_key": keys_split[test_idx], "created_at": ca.iloc[test_idx].astype(str).values,
                  "y_true": ys, "y_pred": preds}).to_csv(out_dir / "predictions" / f"{slug}_test_predictions.csv", index=False)

    row = {"project": slug, "n_dev": int(len(dev_idx)), "n_test": int(len(test_idx)),
           "test_MAE": test_m["MAE"], "test_MdAE": test_m["MdAE"],
           "baseline_median_MAE": b_med["MAE"], "baseline_median_MdAE": b_med["MdAE"],
           "baseline_mean_MAE": b_mean["MAE"], "baseline_mean_MdAE": b_mean["MdAE"],
           "val_MAE_selected": val_mae, "best_lr": best_lr, "retrain_epochs": int(epochs), "epochs_basis": basis,
           "test_start": str(ca.iloc[test_idx].min()), "test_end": str(ca.iloc[test_idx].max()),
           "seed": cfg["seed"], "fold_scheme": scheme, "sbert_model": cfg["sbert"]["model_name"],
           "cnn_filters": str(cfg["model"]["cnn_filters"])}
    with (out_dir / f"{slug}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    log.info("  %s: TEST MAE=%.4f MdAE=%.4f | median-baseline MAE=%.4f | n_test=%d",
             slug, test_m["MAE"], test_m["MdAE"], b_med["MAE"], len(test_idx))
    append_experiment_log(cfg["results"]["experiment_log"], config=cfg, stage="evaluate_test", project=slug,
                          notes=f"test_MAE={test_m['MAE']:.4f} test_MdAE={test_m['MdAE']:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--project", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    set_global_seed(cfg["seed"])

    fold_dir = Path(cfg["results"]["five_fold_dir"])
    csvs = sorted(p for p in fold_dir.glob("*.csv") if not p.stem.endswith("_val_predictions"))
    if args.project:
        csvs = [p for p in csvs if p.stem == args.project]
    if not csvs:
        log.error("No fold CSVs in %s.", fold_dir)
        return 1
    n_ok = 0
    for p in csvs:
        log.info("=== Project: %s ===", p.stem)
        if run_project(p.stem, cfg, device) == 0:
            n_ok += 1
    log.info("Evaluated %d project(s).", n_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())