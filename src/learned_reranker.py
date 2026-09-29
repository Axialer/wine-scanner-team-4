"""Small logistic reranker over visual and OCR-lexicon features.

Weights live in index/reranker.npy as 19 floats:
weights (6), bias, feature mean (6), feature std (6).
The head is separate from index/finetune_proj.npy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ocr_shortlist import is_generic_token, text_tokens

FEATURE_COUNT = 6


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in text_tokens(text or "")
        if token and not is_generic_token(token)
    }


def candidate_features(
    *,
    visual: float,
    grape_tokens: list[str],
    name_tokens: list[str],
    winery_tokens: list[str],
    grape: str,
    name: str,
    winery: str,
    text_cosine: float,
) -> np.ndarray:
    grapes = _tokens(grape)
    names = _tokens(name)
    wineries = _tokens(winery)
    grape_hits = sum(1 for token in grape_tokens if token in grapes)
    name_hits = sum(1 for token in name_tokens if token in names or token in wineries)
    winery_match = 1.0 if any(
        token in wineries or token in names for token in winery_tokens
    ) else 0.0
    grape_missing = sum(1 for token in grape_tokens if token not in grapes)
    return np.array(
        [
            float(visual),
            float(grape_hits),
            float(name_hits),
            winery_match,
            float(grape_missing),
            float(text_cosine),
        ],
        dtype=np.float32,
    )


class LearnedReranker:
    def __init__(self, packed: np.ndarray):
        vec = np.asarray(packed, dtype=np.float32).reshape(-1)
        if vec.shape[0] != 1 + FEATURE_COUNT * 3:
            raise ValueError(f"reranker vector {vec.shape[0]}")
        n = FEATURE_COUNT
        self.w = vec[:n]
        self.b = float(vec[n])
        self.mean = vec[n + 1:n + 1 + n]
        self.std = vec[n + 1 + n:]
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

    def score(self, features: np.ndarray) -> float:
        z = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
        logit = float(self.w @ z + self.b)
        if logit >= 0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exp_logit = float(np.exp(logit))
        return exp_logit / (1.0 + exp_logit)


def pack_reranker(weights, bias, mean, std) -> np.ndarray:
    return np.concatenate([
        np.asarray(weights, dtype=np.float32).reshape(-1),
        np.asarray([bias], dtype=np.float32),
        np.asarray(mean, dtype=np.float32).reshape(-1),
        np.asarray(std, dtype=np.float32).reshape(-1),
    ]).astype(np.float32)


def load_reranker(path: Path) -> LearnedReranker | None:
    if not path.is_file():
        return None
    packed = np.load(path)
    try:
        return LearnedReranker(packed)
    except ValueError:
        return None
