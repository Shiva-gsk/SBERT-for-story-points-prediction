"""Demo: predict the story points of a NEW issue with a project's final model.

    python -m src.predict --list
    python -m src.predict --project alloy_framework --title "Fix login failure" \
        --description "User enters a wrong password and the page crashes"
    python -m src.predict --project alloy_framework          # prompts for title/description

Uses results/models/<slug>.pt written by evaluate_test.py (trained on that project's full development set).
The text goes through the same cleaning and SBERT settings as training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import load_config
from .model import build_model
from .preprocess import clean_text
from .utils import get_logger, resolve_device

log = get_logger("predict")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--project", help="project slug, e.g. alloy_framework")
    ap.add_argument("--title", default=None)
    ap.add_argument("--description", default=None)
    ap.add_argument("--text", default=None, help="title and description as one string")
    ap.add_argument("--list", action="store_true", help="list projects that have a saved model")
    args = ap.parse_args()
    cfg = load_config(args.config)
    models_dir = Path(cfg["results"].get("models_dir", "results/models"))

    available = sorted(p.stem for p in models_dir.glob("*.pt"))
    if args.list or not args.project:
        print("Projects with a saved model:" if available else f"No saved models in {models_dir} - run evaluate_test first.")
        for s in available:
            print("  ", s)
        return 0 if args.list else 1
    path = models_dir / f"{args.project}.pt"
    if not path.exists():
        log.error("No model for %r in %s. Use --list.", args.project, models_dir)
        return 1

    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    if args.text is not None:
        raw = args.text
    else:
        title = args.title if args.title is not None else input("Issue title: ")
        desc = args.description if args.description is not None else input("Issue description: ")
        raw = title.strip() + "\n\n" + desc.strip()
    text = clean_text(raw)
    if len(text) < 5:
        log.error("Text is empty after cleaning.")
        return 1

    from sentence_transformers import SentenceTransformer
    device = resolve_device(cfg.get("device", "auto"))
    enc = SentenceTransformer(ckpt["sbert"]["model_name"], device=device)
    vec = enc.encode([text], convert_to_numpy=True,
                     normalize_embeddings=bool(ckpt["sbert"]["normalize"])).astype(np.float32)

    model = build_model({"model": ckpt["model_cfg"]}, int(ckpt["input_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        out = float(model(torch.from_numpy(vec)).numpy().reshape(-1)[0])
    pred = float(np.expm1(out)) if ckpt["log1p"] else out + float(ckpt["target_shift"])
    pred = max(pred, float(ckpt["sp_min"]))

    print(f"\nProject                : {ckpt['project']}  (trained on {ckpt['n_dev']} issues up to {ckpt['dev_end']})")
    print(f"Predicted story points : {pred:.2f}")
    print(f"Project median (baseline): {ckpt['dev_median']:.2f}")
    print(f"SBERT model / lr / epochs: {ckpt['sbert']['model_name']} / {ckpt['best_lr']} / {ckpt['epochs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())