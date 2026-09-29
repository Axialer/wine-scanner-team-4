"""Free-text catalog ranker. Vectors are injected, so SigLIP stays unloaded.

Run:
  .\\venv\\Scripts\\python.exe tests\\test_text_recommend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_similarity import (  # noqa: E402
    SHORT_QUERY_MESSAGE,
    query_too_short_message,
    rank_catalog_for_sentence,
)


def _wine(**kwargs):
    base = {
        "slug": "wine",
        "name": "Вино",
        "winery": "Дом",
        "grape": "",
        "color": "",
        "description": "",
        "image": "",
    }
    base.update(kwargs)
    return base


def test_closer_embedding_ranks_sauvignon_above_merlot():
    """The query vector sits next to the white sauvignon row, not the red."""

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    red = np.array([0.05, 1.0, 0.0], dtype=np.float32)
    white = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    catalog = [
        _wine(
            slug="merlo-krasnoe",
            name="Мерло",
            grape="Мерло",
            color="рубиновый",
            description="Плотное красное, вишня, к мясу.",
            image="/catalog-image/merlo-krasnoe",
        ),
        _wine(
            slug="sovinon-blan",
            name="Совиньон блан",
            grape="Совиньон Блан",
            color="соломенный",
            description="Свежее белое с цитрусом, хорошо к рыбе и морепродуктам.",
            image="/catalog-image/sovinon-blan",
            winery="Берег",
        ),
    ]
    ranked = rank_catalog_for_sentence(
        catalog,
        query,
        np.stack([red, white]),
        "хочу белое совиньон к рыбе",
    )
    assert ranked[0]["slug"] == "sovinon-blan"
    assert ranked[0]["grape"] == "Совиньон Блан"
    assert ranked[0]["winery"] == "Берег"
    assert "рыб" in ranked[0]["description"].lower()
    assert ranked[0]["image"].endswith("sovinon-blan")
    assert "color" in ranked[0]


def test_close_scores_prefer_grape_and_food_overlap():
    """A slightly weaker cosine still wins when the words are actually there."""

    query = np.array([1.0, 0.0], dtype=np.float32)
    red = np.array([0.99, 0.14], dtype=np.float32)
    white = np.array([0.97, 0.24], dtype=np.float32)
    catalog = [
        _wine(
            slug="merlo-krasnoe",
            name="Мерло",
            grape="Мерло",
            description="Плотное красное к мясу.",
        ),
        _wine(
            slug="sovinon-blan",
            name="Совиньон блан",
            grape="Совиньон Блан",
            description="Свежее белое, хорошо к рыбе.",
        ),
    ]
    matrix = np.stack([red, white])
    ranked = rank_catalog_for_sentence(
        catalog,
        query,
        matrix,
        "совиньон к рыбе",
    )
    assert ranked[0]["slug"] == "sovinon-blan"
    assert all(item["slug"] != "merlo-krasnoe" for item in ranked)


def test_named_grape_beats_a_closer_embedding():
    """A named grape stays ahead even when another wine's vector is closer."""

    query = np.array([1.0, 0.0], dtype=np.float32)
    closer = np.array([1.0, 0.0], dtype=np.float32)
    farther = np.array([0.15, 0.99], dtype=np.float32)
    catalog = [
        _wine(slug="merlo-krasnoe", name="Мерло", grape="Мерло", description="к мясу"),
        _wine(
            slug="sovinon-blan",
            name="Совиньон блан",
            grape="Совиньон Блан",
            description="к рыбе",
        ),
    ]
    ranked = rank_catalog_for_sentence(
        catalog,
        query,
        np.stack([closer, farther]),
        "совиньон к рыбе",
    )
    assert ranked[0]["slug"] == "sovinon-blan"
    assert "совиньон" in ranked[0]["grape"].lower()


def test_identical_embeddings_drop_the_later_slug():
    query = np.array([1.0, 0.0], dtype=np.float32)
    same = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0], dtype=np.float32)
    catalog = [
        _wine(slug="pervyj", name="Первый", description="совиньон"),
        _wine(slug="kopiya", name="Копия", description="совиньон блан"),
        _wine(slug="merlo-krasnoe", name="Мерло", grape="Мерло"),
    ]
    ranked = rank_catalog_for_sentence(
        catalog,
        query,
        np.stack([same, same.copy(), other]),
        "совиньон",
    )
    slugs = [item["slug"] for item in ranked]
    assert slugs[0] == "pervyj"
    assert "kopiya" not in slugs
    assert "merlo-krasnoe" in slugs


def test_returns_top_and_three_more():
    query = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rows = [np.eye(6, dtype=np.float32)[index] for index in range(6)]
    catalog = [
        _wine(slug=f"wine-{index}", name=f"Вино {index}", description=f"описание {index}")
        for index in range(6)
    ]
    ranked = rank_catalog_for_sentence(catalog, query, np.stack(rows), "цитрусовое белое")
    assert len(ranked) == 4
    assert ranked[0]["slug"] == "wine-0"


def test_red_cabernet_beats_closer_white_chardonnay():
    """«красное каберне к мясу» must not come back as white chardonnay."""

    query = np.array([1.0, 0.0], dtype=np.float32)
    catalog = [
        _wine(
            slug="shardone-premium",
            name="Шардоне. Премиум",
            grape="Шардоне",
            category="Белое",
            color="соломенный",
            description="Свежее белое.",
        ),
        _wine(
            slug="saperavi-premium",
            name="Саперави. Премиум",
            grape="Саперави",
            category="Красное",
            color="рубиновый",
            description="Плотное красное к мясу.",
        ),
        _wine(
            slug="kaberne-sovinon",
            name="Каберне Совиньон",
            grape="Каберне Совиньон",
            category="Красное",
            color="рубиновый",
            description="Красное, хорошо к мясу.",
        ),
    ]
    # Chardonnay is the closest vector. Cosine alone would return it.
    matrix = np.stack([
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.2, 0.98], dtype=np.float32),
        np.array([0.05, 1.0], dtype=np.float32),
    ])
    red = rank_catalog_for_sentence(catalog, query, matrix, "красное каберне к мясу")
    assert red[0]["slug"] == "kaberne-sovinon"
    assert "каберне" in red[0]["grape"].lower()
    assert red[0]["category"] == "Красное"
    assert all(item["slug"] != "shardone-premium" for item in red)


def test_white_sauvignon_preferred_for_fish():
    query = np.array([1.0, 0.0], dtype=np.float32)
    catalog = [
        _wine(
            slug="shardone-premium",
            name="Шардоне. Премиум",
            grape="Шардоне",
            category="Белое",
            color="соломенный",
            description="Белое к рыбе.",
        ),
        _wine(
            slug="sovinon-blan",
            name="Совиньон блан",
            grape="Совиньон Блан",
            category="Белое",
            color="соломенный",
            description="Свежее белое, хорошо к рыбе.",
        ),
        _wine(
            slug="kaberne-sovinon",
            name="Каберне Совиньон",
            grape="Каберне Совиньон",
            category="Красное",
            color="рубиновый",
            description="Красное к мясу.",
        ),
    ]
    matrix = np.stack([
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.1, 0.99], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ])
    ranked = rank_catalog_for_sentence(
        catalog,
        query,
        matrix,
        "белое совиньон к рыбе",
    )
    assert ranked[0]["slug"] == "sovinon-blan"
    assert "совиньон" in ranked[0]["grape"].lower()
    assert ranked[0]["category"] == "Белое"


def test_real_catalog_red_and_white_asks_differ():
    """Catalog metadata, zero query vector: attributes still separate the asks."""

    import json

    import pandas as pd

    csv_path = ROOT / "data" / "catalog" / "strapi_output0709.csv"
    if not csv_path.is_file():
        print("skip real catalog")
        return
    frame = pd.read_csv(csv_path)
    by_slug: dict[str, dict] = {}
    for _, row in frame.iterrows():
        slug = _cell(row.get("Slug"))
        if not slug or slug in by_slug:
            continue
        by_slug[slug] = _wine(
            slug=slug,
            name=_cell(row.get("Название вина")),
            grape=_cell(row.get("Сорт винограда")),
            category=_cell(row.get("Категория")),
            color=_cell(row.get("Цвет")),
            description=_cell(row.get("Описание")),
            food=_cell(row.get("Гастрономия")) or _cell(row.get("Сочетание с едой")),
            sugar=_cell(row.get("Сахар")) or _cell(row.get("Содержание сахара")),
        )
    slugs_path = ROOT / "index" / "catalog_text_slugs.json"
    if slugs_path.is_file():
        order = json.loads(slugs_path.read_text(encoding="utf-8"))
        catalog = [by_slug[slug] for slug in order if slug in by_slug]
    else:
        catalog = list(by_slug.values())
    assert catalog
    zeros = np.zeros((len(catalog), 4), dtype=np.float32)
    query = np.zeros(4, dtype=np.float32)
    red = rank_catalog_for_sentence(catalog, query, zeros, "красное каберне к мясу")
    white = rank_catalog_for_sentence(catalog, query, zeros, "белое совиньон к рыбе")
    assert red and white
    assert red[0]["slug"] != white[0]["slug"]
    assert "каберне" in red[0]["grape"].lower()
    assert red[0]["category"].casefold().startswith("крас")
    assert "совиньон" in white[0]["grape"].lower()
    assert white[0]["category"].casefold().startswith("бел")
    assert "шардон" not in white[0]["grape"].lower()
    print("real", red[0]["slug"], "vs", white[0]["slug"])


def _cell(value) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def test_short_text_message_is_russian():
    assert query_too_short_message("") == SHORT_QUERY_MESSAGE
    assert query_too_short_message("  а ") == SHORT_QUERY_MESSAGE
    assert "сорт" in SHORT_QUERY_MESSAGE
    assert query_too_short_message("к рыбе") is None


def main() -> None:
    tests = [
        test_closer_embedding_ranks_sauvignon_above_merlot,
        test_close_scores_prefer_grape_and_food_overlap,
        test_named_grape_beats_a_closer_embedding,
        test_identical_embeddings_drop_the_later_slug,
        test_returns_top_and_three_more,
        test_red_cabernet_beats_closer_white_chardonnay,
        test_white_sauvignon_preferred_for_fish,
        test_real_catalog_red_and_white_asks_differ,
        test_short_text_message_is_russian,
    ]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
