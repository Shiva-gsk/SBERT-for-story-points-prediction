"""torch Dataset wrapper around cached SBERT .npz embeddings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Wrap (vector, story_point) tensors; optionally log1p-transform the target."""

    def __init__(self, vectors: np.ndarray, story_points: np.ndarray,
                 log1p: bool = False):
        self.x = torch.from_numpy(np.asarray(vectors, dtype=np.float32))
        y = np.asarray(story_points, dtype=np.float32)
        if log1p:
            y = np.log1p(y)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def load_npz(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "issue_key": z["issue_key"],
        "vector": z["vector"],
        "story_point": z["story_point"],
        "created_at": z["created_at"],
    }
