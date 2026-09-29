"""Linear head on frozen SigLIP catalog embeddings.

Same-grape bottles pull together. A different grape pushes apart.
Real-photo files are not read. Slugs named in data/eval/real_photo_labels.json
are left out of the pairs. Writes index/finetune_proj.npy and does not
turn the head on; the server flag is a separate file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EMB_PATH = ROOT / "index" / "catalog_embeddings.npy"
META_PATH = ROOT / "index" / "catalog_meta.json"
CSV_PATH = ROOT / "data" / "catalog" / "strapi_output0709.csv"
LABELS_PATH = ROOT / "data" / "eval" / "real_photo_labels.json"
OUT_PATH = ROOT / "index" / "finetune_proj.npy"

WIDTH = 128
EPOCHS = 6
BATCH = 256
MARGIN = 0.20
LR = 0.05


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def grape_key(text: str) -> str:
    token = _cell(text).casefold().replace("ё", "е")
    for piece in token.replace(",", " ").replace(";", " ").split():
        if len(piece) >= 4:
            return piece
    return token


def main() -> None:
    embeddings = np.load(EMB_PATH).astype(np.float32)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    items = meta["items"]
    if len(items) != len(embeddings):
        raise SystemExit(f"row mismatch {len(items)} vs {len(embeddings)}")

    label_rows = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["labels"].values()
    held_slugs = {row["slug"] for row in label_rows if isinstance(row, dict) and row.get("slug")}

    frame = pd.read_csv(CSV_PATH)
    grape_by_slug: dict[str, str] = {}
    for _, row in frame.iterrows():
        slug = _cell(row.get("Slug"))
        if slug and slug not in grape_by_slug:
            grape_by_slug[slug] = grape_key(_cell(row.get("Сорт винограда")))

    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        slug = item["slug"]
        if slug in held_slugs:
            continue
        key = grape_by_slug.get(slug) or ""
        if not key:
            continue
        groups.setdefault(key, []).append(index)
    groups = {key: rows for key, rows in groups.items() if len(rows) >= 2}
    keys = list(groups)
    if len(keys) < 2:
        raise SystemExit("not enough grape groups")

    rng = np.random.default_rng(0)
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    for key, rows in groups.items():
        others = [other for other in keys if other != key]
        for row in rows:
            mate = int(rng.choice([item for item in rows if item != row]))
            negative_key = others[int(rng.integers(0, len(others)))]
            negative = int(rng.choice(groups[negative_key]))
            anchors.append(row)
            positives.append(mate)
            negatives.append(negative)

    bank = torch.from_numpy(embeddings)
    weight = torch.nn.Parameter(torch.eye(embeddings.shape[1], WIDTH))
    opt = torch.optim.Adam([weight], lr=LR)
    count = len(anchors)
    print(f"pairs {count} groups {len(groups)} held_out_slugs {len(held_slugs)}", flush=True)

    for epoch in range(EPOCHS):
        order = rng.permutation(count)
        total = 0.0
        seen = 0
        for start in range(0, count, BATCH):
            choose = order[start:start + BATCH]
            a = bank[[anchors[int(i)] for i in choose]]
            p = bank[[positives[int(i)] for i in choose]]
            n = bank[[negatives[int(i)] for i in choose]]
            za = torch.nn.functional.normalize(a @ weight, dim=1)
            zp = torch.nn.functional.normalize(p @ weight, dim=1)
            zn = torch.nn.functional.normalize(n @ weight, dim=1)
            loss = torch.relu(MARGIN - (za * zp).sum(dim=1) + (za * zn).sum(dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(choose)
            seen += len(choose)
        print(f"epoch {epoch + 1} loss {total / max(seen, 1):.4f}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, weight.detach().cpu().numpy().astype(np.float32))
    print(f"wrote {OUT_PATH} {tuple(weight.shape)}", flush=True)


if __name__ == "__main__":
    main()
