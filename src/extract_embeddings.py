"""Stage 4: SBERT sentence-embedding extraction (cached as .npz).

For each eligible project, encode the unified `text` column and save data/embeddings/<slug>.npz with keys:
  issue_key, vector, story_point, created_at, fingerprint
`fingerprint` = hash(model name, normalize flag, every text). A cache hit requires the SAME fingerprint, so
changing the model, normalisation, cleaning, filters or min_issues re-encodes automatically.
Row order = processed CSV order; downstream stages must join on issue_key, not on position.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .utils import append_experiment_log, get_logger, resolve_device

log = get_logger("extract_embeddings")


def load_encoder(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def encode_one(model, texts: list[str], batch_size: int, normalize: bool, show_progress: bool) -> np.ndarray:
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress,
                        convert_to_numpy=True, normalize_embeddings=normalize)
    return np.asarray(vecs, dtype=np.float32)


def fingerprint(model_name: str, normalize: bool, keys: list[str], texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(f"{model_name}|{normalize}|{len(texts)}".encode())
    for k, t in zip(keys, texts):
        h.update(k.encode()); h.update(b"\x00"); h.update(t.encode()); h.update(b"\x01")
    return h.hexdigest()


def cache_is_valid(path: Path, fp: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return "fingerprint" in z.files and str(z["fingerprint"][0]) == fp
    except Exception as exc:                                   # corrupt / old-format cache
        log.warning("  cache %s unreadable (%s) - will re-encode", path.name, exc)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--no-cache", action="store_true", help="re-encode even if a valid .npz exists")
    args = ap.parse_args()
    cfg = load_config(args.config)

    proc_dir = Path(cfg["data"]["processed_dir"])
    emb_dir = Path(cfg["data"]["embeddings_dir"])
    emb_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg["device"])
    model_name = cfg["sbert"]["model_name"]
    batch_size = int(cfg["sbert"]["batch_size"])
    normalize = bool(cfg["sbert"]["normalize"])

    csvs = sorted(p for p in proc_dir.glob("*.csv") if p.name != "dataset_stats.csv")
    if args.project:
        csvs = [p for p in csvs if p.stem == args.project]
    if not csvs:
        log.error("No processed CSVs in %s.", proc_dir)
        return 1

    model = None                                               # lazy: only load if something needs encoding
    n_encoded = n_cached = 0
    for p in csvs:
        out_path = emb_dir / f"{p.stem}.npz"
        df = pd.read_csv(p)
        if df["issue_key"].duplicated().any():
            raise ValueError(f"{p.name}: duplicate issue_key values")
        keys = df["issue_key"].astype(str).tolist()
        texts = df["text"].astype(str).tolist()
        fp = fingerprint(model_name, normalize, keys, texts)

        if not args.no_cache and cache_is_valid(out_path, fp):
            log.info("  %s: valid cache hit -> %s", p.stem, out_path.name)
            n_cached += 1
            continue

        if model is None:
            import sentence_transformers
            log.info("Loading SBERT model: %s (device=%s, normalize=%s, sentence-transformers %s)",
                     model_name, device, normalize, sentence_transformers.__version__)
            model = load_encoder(model_name, device)
            log.info("  embedding dim=%d, max_seq_length=%s (longer texts are TRUNCATED)",
                     model.get_sentence_embedding_dimension(), getattr(model, "max_seq_length", "?"))

        log.info("  %s: encoding %d issues ...", p.stem, len(texts))
        vecs = encode_one(model, texts, batch_size, normalize, show_progress=len(texts) > 200)
        np.savez_compressed(
            out_path,
            issue_key=np.asarray(keys, dtype="U"),
            vector=vecs,
            story_point=df["story_point"].to_numpy(dtype=np.float32),
            created_at=np.asarray(pd.to_datetime(df["created_at"], utc=True).astype(str), dtype="U"),
            fingerprint=np.asarray([fp], dtype="U"),
        )
        log.info("  %s: wrote %s (n=%d, dim=%d)", p.stem, out_path.name, *vecs.shape)
        n_encoded += 1

    append_experiment_log(cfg["results"]["experiment_log"], config=cfg, stage="extract_embeddings",
                          notes=f"{n_encoded} encoded, {n_cached} cached; model={model_name}; normalize={normalize}")
    return 0


if __name__ == "__main__":
    sys.exit(main())