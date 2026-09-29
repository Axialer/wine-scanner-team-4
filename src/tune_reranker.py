import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

EVAL_FILE = ROOT / "index" / "evaluation_all.json"
CSV_FILE = ROOT / "data" / "catalog" / "strapi_output0709.csv"
OUT_FILE = ROOT / "index" / "reranker_tuning.json"


# ============================================================
# TEXT
# ============================================================

def norm(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("ё", "е")

    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def similarity(a, b):
    a = norm(a)
    b = norm(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.90

    # token overlap
    at = set(a.split())
    bt = set(b.split())

    common = at & bt

    token_score = (
        len(common) / max(1, len(bt))
    )

    seq_score = SequenceMatcher(
        None,
        a,
        b
    ).ratio()

    return max(
        token_score,
        seq_score
    )


# ============================================================
# OCR EXTRACTION
# ============================================================

def extract_ocr_text(value):
    """
    Пытаемся вытащить OCR независимо от того,
    в каком формате его сохранил evaluate_all.py.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):

        for key in (
            "text",
            "ocr_text",
            "raw_text",
            "result"
        ):
            if key in value:
                result = extract_ocr_text(
                    value[key]
                )

                if result:
                    return result

        parts = []

        for v in value.values():
            text = extract_ocr_text(v)

            if text:
                parts.append(text)

        return " ".join(parts)

    if isinstance(value, list):

        parts = []

        for item in value:
            text = extract_ocr_text(item)

            if text:
                parts.append(text)

        return " ".join(parts)

    return ""


# ============================================================
# OCR SCORE
# ============================================================

def ocr_score(ocr_text, metadata):
    if not ocr_text:
        return 0.0

    fields = [
        metadata.get("Название вина", ""),
        metadata.get("Винодельня", ""),
        metadata.get("Сорт винограда", ""),
        metadata.get("Регион", ""),
        metadata.get("Цвет", ""),
        metadata.get("Категория", ""),
    ]

    fields = [
        str(x)
        for x in fields
        if x and str(x).strip()
    ]

    if not fields:
        return 0.0

    scores = [
        similarity(ocr_text, field)
        for field in fields
    ]

    scores.sort(reverse=True)

    if len(scores) >= 3:
        return (
            scores[0] * 0.60
            + scores[1] * 0.25
            + scores[2] * 0.15
        )

    if len(scores) == 2:
        return (
            scores[0] * 0.70
            + scores[1] * 0.30
        )

    return scores[0]


# ============================================================
# VISUAL SCORE
# ============================================================

def normalize_candidates(candidates):
    if not candidates:
        return {}

    scores = [
        float(x.get("score", 0))
        for x in candidates
    ]

    mn = min(scores)
    mx = max(scores)

    result = {}

    for rank, item in enumerate(candidates):

        slug = item.get("slug")

        if not slug:
            continue

        score = float(
            item.get("score", 0)
        )

        if mx > mn:
            normalized = (
                score - mn
            ) / (
                mx - mn
            )
        else:
            normalized = 1.0

        result[slug] = {
            "score": score,
            "normalized": normalized,
            "rank": rank + 1
        }

    return result


# ============================================================
# CANDIDATES
# ============================================================

def get_candidates(item, mode):
    if mode == "raw":
        return item.get(
            "visual_raw_candidates",
            []
        )

    if mode == "normalized":
        return item.get(
            "visual_normalized_candidates",
            []
        )

    if mode == "hybrid":

        raw = item.get(
            "visual_raw_candidates",
            []
        )

        normalized = item.get(
            "visual_normalized_candidates",
            []
        )

        return raw, normalized

    return []


# ============================================================
# RANKING
# ============================================================

def rank_item(
    item,
    metadata,
    visual_mode,
    visual_weight,
    ocr_weight
):
    ocr_text = extract_ocr_text(
        item.get("ocr")
    )

    if visual_mode == "hybrid":

        raw = normalize_candidates(
            item.get(
                "visual_raw_candidates",
                []
            )
        )

        normalized = normalize_candidates(
            item.get(
                "visual_normalized_candidates",
                []
            )
        )

        slugs = set(raw) | set(normalized)

        ranked = []

        for slug in slugs:

            raw_score = raw.get(
                slug,
                {}
            ).get(
                "normalized",
                0.0
            )

            norm_score = normalized.get(
                slug,
                {}
            ).get(
                "normalized",
                0.0
            )

            visual = (
                raw_score * visual_weight
                + norm_score * (1 - visual_weight)
            )

            meta = metadata.get(
                slug,
                {}
            )

            ocr = ocr_score(
                ocr_text,
                meta
            )

            final = (
                visual * (1 - ocr_weight)
                + ocr * ocr_weight
            )

            ranked.append(
                (
                    slug,
                    final
                )
            )

    else:

        candidates = normalize_candidates(
            get_candidates(
                item,
                visual_mode
            )
        )

        ranked = []

        for slug, info in candidates.items():

            visual = info["normalized"]

            meta = metadata.get(
                slug,
                {}
            )

            ocr = ocr_score(
                ocr_text,
                meta
            )

            final = (
                visual * (1 - ocr_weight)
                + ocr * ocr_weight
            )

            ranked.append(
                (
                    slug,
                    final
                )
            )

    ranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        slug
        for slug, score in ranked
    ]


# ============================================================
# METRICS
# ============================================================

def evaluate(
    data,
    metadata,
    visual_mode,
    visual_weight,
    ocr_weight
):
    top1 = 0
    top5 = 0
    rr = 0.0

    for item in data:

        expected = item.get(
            "expected_slug"
        )

        if not expected:
            continue

        ranked = rank_item(
            item,
            metadata,
            visual_mode,
            visual_weight,
            ocr_weight
        )

        if not ranked:
            continue

        if ranked[0] == expected:
            top1 += 1

        if expected in ranked[:5]:
            top5 += 1

        if expected in ranked:
            rank = ranked.index(
                expected
            ) + 1

            rr += 1.0 / rank

    n = len(data)

    return {
        "top1": top1 / n * 100,
        "top5": top5 / n * 100,
        "mrr": rr / n,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(" WINE RERANKER TUNING")
    print("=" * 70)
    print()

    if not EVAL_FILE.exists():
        print(
            f"[ERROR] Нет файла:\n{EVAL_FILE}"
        )
        print()
        print(
            "Сначала нужен evaluation_all.json"
        )
        return

    print("[1/4] Загружаю evaluation_all.json...")

    with open(
        EVAL_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        data = (
            payload.get("results")
            or payload.get("items")
            or payload.get("data")
            or []
        )
    else:
        data = payload

    print(
        f"      Тестов: {len(data)}"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    print(
        "[2/4] Загружаю метаданные каталога..."
    )

    df = pd.read_csv(
        CSV_FILE,
        sep=",",
        encoding="utf-8-sig"
    )

    metadata = {}

    for _, row in df.iterrows():

        slug = str(
            row.get("Slug", "")
        ).strip()

        if not slug:
            continue

        if slug not in metadata:
            metadata[slug] = {
                column: (
                    ""
                    if pd.isna(row.get(column))
                    else str(row.get(column))
                )
                for column in df.columns
            }

    print(
        f"      Slug metadata: {len(metadata)}"
    )

    # --------------------------------------------------------
    # BASELINES
    # --------------------------------------------------------

    print()
    print("[3/4] Считаю baseline...")
    print()

    baselines = {}

    for mode in (
        "raw",
        "normalized"
    ):

        result = evaluate(
            data,
            metadata,
            mode,
            1.0,
            0.0
        )

        baselines[mode] = result

        print(
            f"{mode:12} "
            f"Top-1 {result['top1']:6.2f}%   "
            f"Top-5 {result['top5']:6.2f}%   "
            f"MRR {result['mrr']:.4f}"
        )

    # --------------------------------------------------------
    # GRID SEARCH
    # --------------------------------------------------------

    print()
    print("[4/4] Перебираю веса reranker...")
    print()

    experiments = []

    # --------------------------------------------------------
    # RAW + OCR
    # --------------------------------------------------------

    for ocr_weight in [
        0.02,
        0.05,
        0.08,
        0.10,
        0.12,
        0.15,
        0.18,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
    ]:

        result = evaluate(
            data,
            metadata,
            "raw",
            1.0,
            ocr_weight
        )

        experiments.append({
            "mode": "raw+ocr",
            "visual_weight": 1.0,
            "ocr_weight": ocr_weight,
            **result
        })

    # --------------------------------------------------------
    # NORMALIZED + OCR
    # --------------------------------------------------------

    for ocr_weight in [
        0.02,
        0.05,
        0.08,
        0.10,
        0.12,
        0.15,
        0.18,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
    ]:

        result = evaluate(
            data,
            metadata,
            "normalized",
            1.0,
            ocr_weight
        )

        experiments.append({
            "mode": "normalized+ocr",
            "visual_weight": 1.0,
            "ocr_weight": ocr_weight,
            **result
        })

    # --------------------------------------------------------
    # RAW + NORMALIZED + OCR
    # --------------------------------------------------------

    for raw_weight in [
        0.25,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]:

        for ocr_weight in [
            0.02,
            0.05,
            0.08,
            0.10,
            0.12,
            0.15,
            0.18,
            0.20,
            0.25,
            0.30,
        ]:

            result = evaluate(
                data,
                metadata,
                "hybrid",
                raw_weight,
                ocr_weight
            )

            experiments.append({
                "mode": "raw+normalized+ocr",
                "visual_weight": raw_weight,
                "ocr_weight": ocr_weight,
                **result
            })

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    experiments.sort(
        key=lambda x: (
            x["top1"],
            x["top5"],
            x["mrr"]
        ),
        reverse=True
    )

    print(
        "-" * 70
    )

    print(
        "TOP-10 КОНФИГУРАЦИЙ:"
    )

    for i, result in enumerate(
        experiments[:10],
        1
    ):

        print(
            f"{i:2}. "
            f"{result['mode']:23} "
            f"raw={result['visual_weight']:.2f} "
            f"ocr={result['ocr_weight']:.2f} | "
            f"Top-1={result['top1']:6.2f}% | "
            f"Top-5={result['top5']:6.2f}% | "
            f"MRR={result['mrr']:.4f}"
        )

    best = experiments[0]

    print()
    print("=" * 70)
    print(" BEST CONFIGURATION")
    print("=" * 70)

    print(
        f"Mode:          {best['mode']}"
    )

    print(
        f"Visual weight: {best['visual_weight']:.2f}"
    )

    print(
        f"OCR weight:    {best['ocr_weight']:.2f}"
    )

    print(
        f"Top-1:         {best['top1']:.2f}%"
    )

    print(
        f"Top-5:         {best['top5']:.2f}%"
    )

    print(
        f"MRR:           {best['mrr']:.4f}"
    )

    print()

    output = {
        "baselines": baselines,
        "experiments": experiments,
        "best": best
    }

    with open(
        OUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Результаты сохранены:\n{OUT_FILE}"
    )

    print()


if __name__ == "__main__":
    main()