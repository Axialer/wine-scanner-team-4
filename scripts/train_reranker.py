"""Logistic reranker on synthetic catalog pairs.

Positive: a wine against its own text. Hard negative: another wine of
the same winery with a different grape. Catalog text stands in for OCR.
Real phone photos are not read. Writes index/reranker.npy and leaves
index/reranker_enabled.txt untouched.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learned_reranker import candidate_features, pack_reranker  # noqa: E402
from ocr_shortlist import is_generic_token, text_tokens  # noqa: E402

EMB_PATH = ROOT / "index" / "catalog_embeddings.npy"
META_PATH = ROOT / "index" / "catalog_meta.json"
CSV_PATH = ROOT / "data" / "catalog" / "strapi_output0709.csv"
TEXT_EMB_PATH = ROOT / "index" / "catalog_text_emb.npy"
TEXT_SLUGS_PATH = ROOT / "index" / "catalog_text_slugs.json"
OUT_PATH = ROOT / "index" / "reranker.npy"


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _kept(text: str) -> list[str]:
    seen: list[str] = []
    for token in text_tokens(text):
        if len(token) < 4 or is_generic_token(token) or token in seen:
            continue
        seen.append(token)
    return seen


def _encode_label_queries(queries: dict[str, str]) -> dict[str, np.ndarray]:
    """SigLIP text vectors of the catalog words we pretend were OCR."""

    import torch
    from transformers import AutoModel, AutoProcessor

    name = "google/siglip2-base-patch16-224"
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    slugs = list(queries)
    texts = [queries[slug] if queries[slug].strip() else "вино" for slug in slugs]
    out: dict[str, np.ndarray] = {}
    batch = 32
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        inputs = processor(
            text=chunk,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inputs = {key: value for key, value in inputs.items() if torch.is_tensor(value)}
        with torch.inference_mode():
            output = model.get_text_features(**inputs)
        if hasattr(output, "pooler_output"):
            features = output.pooler_output
        elif hasattr(output, "text_embeds"):
            features = output.text_embeds
        else:
            features = output
        features = features.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features /= norms
        for offset, slug in enumerate(slugs[start:start + batch]):
            out[slug] = features[offset]
        print(f"encoded {min(start + batch, len(texts))}/{len(texts)}", flush=True)
    return out


def main() -> None:
    import torch

    print("cuda", torch.cuda.is_available(), flush=True)
    embeddings = np.load(EMB_PATH).astype(np.float32)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    items = meta["items"]
    if len(items) != len(embeddings):
        raise SystemExit(f"row mismatch {len(items)} vs {len(embeddings)}")

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for index, item in enumerate(items):
        slug = item["slug"]
        sums[slug] = embeddings[index] + sums.get(slug, 0.0)
        counts[slug] = counts.get(slug, 0) + 1
    image: dict[str, np.ndarray] = {}
    for slug, total in sums.items():
        vec = total / counts[slug]
        norm = np.linalg.norm(vec)
        image[slug] = vec / norm if norm else vec

    text_slugs = json.loads(TEXT_SLUGS_PATH.read_text(encoding="utf-8"))
    text_matrix = np.load(TEXT_EMB_PATH).astype(np.float32)
    text = {slug: text_matrix[i] for i, slug in enumerate(text_slugs)}

    frame = pd.read_csv(CSV_PATH)
    wines: dict[str, dict] = {}
    for _, row in frame.iterrows():
        slug = _cell(row.get("Slug"))
        if not slug or slug in wines:
            continue
        if slug not in image or slug not in text:
            continue
        grapes = _kept(_cell(row.get("Сорт винограда")))
        names = [token for token in _kept(_cell(row.get("Название вина"))) if token not in grapes]
        wineries = _kept(_cell(row.get("Винодельня")))
        if not grapes and not names:
            continue
        wines[slug] = {
            "grape": _cell(row.get("Сорт винограда")),
            "name": _cell(row.get("Название вина")),
            "winery": _cell(row.get("Винодельня")),
            "grapes": grapes,
            "names": names,
            "wineries": wineries,
            "grape_key": " ".join(grapes),
        }

    query_vec = _encode_label_queries({
        slug: " ".join(wine["names"] + wine["grapes"] + wine["wineries"])
        for slug, wine in wines.items()
    })

    by_winery: dict[str, list[str]] = defaultdict(list)
    for slug, wine in wines.items():
        key = " ".join(wine["wineries"])
        if key:
            by_winery[key].append(slug)

    rng = np.random.default_rng(0)
    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    slugs = list(wines)
    hard_pairs = 0

    def emit(slug: str, other: str, visual: float, label: int) -> None:
        wine = wines[slug]
        cand = wines[other]
        cos = float(np.dot(query_vec[slug], text[other]))
        rows_x.append(candidate_features(
            visual=visual,
            grape_tokens=wine["grapes"],
            name_tokens=wine["names"],
            winery_tokens=wine["wineries"],
            grape=cand["grape"],
            name=cand["name"],
            winery=cand["winery"],
            text_cosine=cos,
        ))
        rows_y.append(label)

    for slug, wine in wines.items():
        key = " ".join(wine["wineries"])
        siblings = [
            other
            for other in by_winery.get(key, [])
            if other != slug and wines[other]["grape_key"] != wine["grape_key"]
        ]
        if not siblings:
            continue
        hard = siblings[int(rng.integers(0, len(siblings)))]
        visual = float(np.dot(image[slug], image[hard]))
        visual = max(0.0, min(1.0, visual))
        emit(slug, slug, visual, 1)
        emit(slug, hard, visual, 0)
        hard_pairs += 1
        easy = slugs[int(rng.integers(0, len(slugs)))]
        if easy == slug or easy == hard:
            continue
        easy_visual = float(np.dot(image[slug], image[easy]))
        emit(slug, slug, min(1.0, max(visual, easy_visual) + 0.08), 1)
        emit(slug, easy, max(0.0, easy_visual), 0)

    x = np.stack(rows_x).astype(np.float32)
    y = np.asarray(rows_y, dtype=np.int32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0, max_iter=400)
    clf.fit(z, y)
    packed = pack_reranker(clf.coef_.reshape(-1), float(clf.intercept_[0]), mean, std)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, packed)
    pred = clf.predict(z)
    print(
        f"pairs {hard_pairs} rows {len(y)} train_acc {(pred == y).mean():.3f} "
        f"wrote {OUT_PATH} {packed.shape}",
        flush=True,
    )
    print("weights", np.round(clf.coef_.reshape(-1), 3).tolist(), flush=True)


if __name__ == "__main__":
    main()
