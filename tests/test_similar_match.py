"""Found vs not-found without loading SigLIP.

Run:
  .\\venv\\Scripts\\python.exe tests\\test_similar_match.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from similar_match import (  # noqa: E402
    ExternalProfile,
    build_similar,
    decide_match,
    public_view,
    rank_catalog_by_attributes,
)
from text_similarity import rank_catalog_by_text  # noqa: E402


KNOWN = {"АЛИГОТЕ", "ЦИТРОННЫЙ", "КАБЕРНЕ", "СОВИНЬОН", "МЕРЛО", "ШАРДОНЕ"}

ZHEMCHUZHNAYA_OCR = "жемчжная арг AлиготЕ цитрон белое сухоё"


def _wine(**kwargs):
    base = {
        "slug": "wine",
        "name": "Вино",
        "winery": "Дом",
        "grape": "",
        "color": "рубиновый",
        "category": "Красное",
        "region": "Крым",
        "description": "",
        "image": "/catalog-image/wine",
        "visual_score": 0.5,
        "final_score": 0.5,
    }
    base.update(kwargs)
    return base


def test_zhemchuzhnaya_fixture_stays_found():
    winner = _wine(
        slug="zhemchuzhnaya-9-aligote-czitron",
        name="Жемчужная 9 Алиготе, Цитрон",
        winery="АРАТТИ",
        grape="Алиготе, Цитронный Магарача",
        color="светло-соломенный",
        category="Белое",
        final_score=0.9018,
        visual_score=0.607,
    )
    assert decide_match(
        winner,
        None,
        ZHEMCHUZHNAYA_OCR,
        method="ocr_visual",
        exclusive=True,
        shortlist_size=1,
        known_grape_tokens=KNOWN,
    )


def test_low_confidence_weak_ocr_is_not_found_with_similar():
    leader = _wine(
        slug="lookalike",
        name="Похожая этикетка",
        visual_score=0.40,
        final_score=0.42,
        category="Белое",
        grape="Совиньон Блан",
    )
    second = _wine(
        slug="other",
        name="Другое",
        visual_score=0.38,
        final_score=0.40,
        category="Красное",
        grape="Мерло",
    )
    ocr = "к8 фааговия хнко"
    assert not decide_match(
        leader,
        second,
        ocr,
        method="visual_ocr",
        exclusive=False,
        shortlist_size=0,
        known_grape_tokens=KNOWN,
        dictionary_tokens=0,
    )
    similar = build_similar([leader, second], ocr, KNOWN)
    assert similar
    assert similar[0]["slug"] == "lookalike"
    view = public_view(
        status="not_found",
        ocr=ocr,
        method="visual_ocr",
        visual_compared=2103,
        timing={"total": 1.2},
        wine=leader,
        slug="lookalike",
        confidence=0.42,
        similar=similar,
    )
    assert view["status"] == "not_found"
    assert view["slug"] is None
    assert view["name"] == ""
    assert view["ocr"] == ocr
    assert view["method"] == "visual_ocr"
    assert view["similar"][0]["name"] == "Похожая этикетка"
    assert "визуально близкая этикетка" in view["similar"][0]["reason"]


def test_close_sibling_scores_are_not_found():
    # Tied visuals inside one shortlist are not a clear SigLIP winner.
    winner = _wine(
        slug="zhemchuzhnaya-9-czitron-shardone-1",
        grape="Цитронный Магарача, Шардоне",
        category="Белое",
        final_score=0.9324,
        visual_score=0.70,
    )
    runner = _wine(
        slug="czitronnyj-magaracha-sovinon-blan",
        grape="Совиньон Блан, Цитронный Магарача",
        category="Белое",
        final_score=0.9222,
        visual_score=0.70,
    )
    assert not decide_match(
        winner,
        runner,
        "Жемчужная АРАТТИ цитрон белое",
        method="ocr_visual",
        exclusive=True,
        shortlist_size=10,
        known_grape_tokens=KNOWN,
    )


def test_visual_tie_keeps_dictionary_winner():
    # Visual scores inside VISUAL_TIE are noise once a catalog word hit.
    winner = _wine(
        slug="estate-blanc",
        grape="Совиньон Блан",
        category="Белое",
        final_score=0.84,
        visual_score=0.760,
    )
    runner = _wine(
        slug="estate-red",
        grape="Каберне Совиньон",
        category="Красное",
        final_score=0.50,
        visual_score=0.768,
    )
    assert decide_match(
        winner,
        runner,
        "совиньон",
        method="visual_ocr",
        exclusive=False,
        shortlist_size=1,
        known_grape_tokens=KNOWN,
        dictionary_tokens=1,
    )
    assert not decide_match(
        winner,
        _wine(
            slug="estate-red",
            grape="Каберне Совиньон",
            final_score=0.70,
            visual_score=0.82,
        ),
        "совиньон",
        method="visual_ocr",
        exclusive=False,
        shortlist_size=1,
        known_grape_tokens=KNOWN,
        dictionary_tokens=1,
    )


def test_open_catalog_needs_a_large_gap():
    clear = _wine(slug="oleg", grape="Олег", final_score=0.8705, visual_score=0.81)
    far = _wine(slug="other", final_score=0.4978, visual_score=0.76)
    assert decide_match(
        clear,
        far,
        "олег",
        method="visual_ocr",
        exclusive=False,
        shortlist_size=1,
        known_grape_tokens=KNOWN | {"ОЛЕГ"},
    )

    relikta = _wine(
        slug="wrong-relikta",
        grape="Красностоп",
        category="Красное",
        final_score=0.8152,
        visual_score=0.67,
    )
    nxt = _wine(slug="next", final_score=0.7112, visual_score=0.69)
    assert not decide_match(
        relikta,
        nxt,
        "реликта сухое",
        method="visual_ocr",
        exclusive=False,
        shortlist_size=17,
        known_grape_tokens=KNOWN,
    )


def test_producer_only_flat_visual_is_not_a_match():
    # Scores that used to pass on visual >= 0.50. The gap is under 0.03.
    winner = _wine(
        slug="aratti-sovinon",
        name="Совиньон Блан",
        winery="АРАТТИ",
        grape="Совиньон Блан",
        category="Белое",
        final_score=0.91,
        visual_score=0.742,
    )
    runner = _wine(
        slug="aratti-risling",
        name="Рислинг",
        winery="АРАТТИ",
        grape="Рислинг",
        category="Белое",
        final_score=0.88,
        visual_score=0.721,
    )
    assert not decide_match(
        winner,
        runner,
        "АРАТТИ",
        method="ocr_visual",
        exclusive=True,
        shortlist_size=12,
        known_grape_tokens=KNOWN,
        dictionary_tokens=1,
        producer_only=True,
    )


def test_unique_title_hit_stays_found():
    winner = _wine(
        slug="belaya-lvicza",
        name="Белая Львица",
        winery="АРАТТИ",
        grape="Рислинг",
        category="Белое",
        final_score=0.98,
        visual_score=0.753,
    )
    runner = _wine(
        slug="other-white",
        name="Другое белое",
        visual_score=0.741,
        final_score=0.70,
    )
    assert decide_match(
        winner,
        runner,
        "АРАТТИ БЕЛАЯ",
        method="ocr_visual",
        exclusive=True,
        shortlist_size=1,
        known_grape_tokens=KNOWN,
        dictionary_tokens=2,
        producer_only=False,
    )


def test_generic_grape_open_catalog_lead_can_match():
    winner = _wine(
        slug="pobeda",
        grape="Каберне Совиньон",
        category="Красное",
        final_score=0.014,
        visual_score=0.784,
    )
    runner = _wine(
        slug="other-red",
        grape="Каберне Совиньон",
        category="Красное",
        final_score=0.01,
        visual_score=0.561,
    )
    assert decide_match(
        winner,
        runner,
        "каберне",
        method="visual_ocr",
        exclusive=False,
        shortlist_size=13,
        known_grape_tokens=KNOWN,
        dictionary_tokens=1,
        generic_only=True,
    )


def test_generic_grape_only_is_not_a_match():
    winner = _wine(
        slug="some-pinot",
        grape="Пино Нуар",
        final_score=0.92,
        visual_score=0.84,
    )
    runner = _wine(
        slug="other-pinot",
        grape="Пино Нуар",
        final_score=0.60,
        visual_score=0.70,
    )
    assert not decide_match(
        winner,
        runner,
        "ПИНО",
        method="visual_ocr",
        exclusive=False,
        shortlist_size=40,
        known_grape_tokens=KNOWN,
        dictionary_tokens=1,
        generic_only=True,
    )


def test_full_catalog_without_tokens_needs_a_clear_lead():
    close = _wine(slug="near", visual_score=0.80, final_score=0.80)
    nxt = _wine(slug="next", visual_score=0.77, final_score=0.77)
    assert not decide_match(
        close,
        nxt,
        "",
        method="visual",
        exclusive=False,
        shortlist_size=0,
        known_grape_tokens=KNOWN,
        dictionary_tokens=0,
    )
    low = _wine(slug="modest", visual_score=0.62, final_score=0.62)
    lower = _wine(slug="behind", visual_score=0.50, final_score=0.50)
    assert not decide_match(
        low,
        lower,
        "",
        method="visual",
        exclusive=False,
        shortlist_size=0,
        known_grape_tokens=KNOWN,
        dictionary_tokens=0,
    )
    ahead = _wine(slug="clear", visual_score=0.82, final_score=0.82)
    behind = _wine(slug="far", visual_score=0.76, final_score=0.76)
    assert decide_match(
        ahead,
        behind,
        "",
        method="visual",
        exclusive=False,
        shortlist_size=0,
        known_grape_tokens=KNOWN,
        dictionary_tokens=0,
    )


def test_unique_shortlist_keeps_clear_visual_when_text_is_squashed():
    winner = _wine(
        slug="czitronnyj-magaracha-sovinon-blan",
        grape="Совиньон Блан, Цитронный Магарача",
        category="Белое",
        final_score=0.019,
        visual_score=0.728,
    )
    assert decide_match(
        winner,
        None,
        "цитрон совиньон белое",
        method="ocr_visual",
        exclusive=True,
        shortlist_size=1,
        known_grape_tokens=KNOWN,
        dictionary_tokens=2,
    )
    weak = _wine(
        slug="lookalike",
        grape="Совиньон Блан",
        category="Белое",
        final_score=0.02,
        visual_score=0.40,
    )
    assert not decide_match(
        weak,
        None,
        "цитрон совиньон белое",
        method="ocr_visual",
        exclusive=True,
        shortlist_size=1,
        known_grape_tokens=KNOWN,
        dictionary_tokens=2,
    )


def test_saturated_text_with_a_visual_lead_stays_found():
    winner = _wine(
        slug="denisov_rubin_klaret_krasnaya_strelka",
        grape="Рубин Голодриги",
        category="Красное",
        final_score=0.9844,
        visual_score=0.706,
    )
    runner = _wine(
        slug="denisov_pino_noir_klaret",
        grape="Пино Нуар",
        category="Красное",
        final_score=0.98,
        visual_score=0.692,
    )
    assert decide_match(
        winner,
        runner,
        "DENISOV красная стрелка",
        method="ocr_visual",
        exclusive=True,
        shortlist_size=2,
        known_grape_tokens=KNOWN,
        dictionary_tokens=3,
    )


def test_soft_unique_hit_with_clear_visual_stays_found():
    winner = _wine(
        slug="oleg",
        grape="Олег",
        category="Белое",
        final_score=0.019,
        visual_score=0.806,
    )
    runner = _wine(
        slug="other",
        grape="Шардоне",
        category="Белое",
        final_score=0.0,
        visual_score=0.777,
    )
    assert decide_match(
        winner,
        runner,
        "олег",
        method="visual_ocr",
        exclusive=False,
        shortlist_size=1,
        known_grape_tokens=KNOWN,
        dictionary_tokens=1,
    )


def test_color_conflict_blocks_a_high_score():
    winner = _wine(
        slug="white-blend",
        category="Белое",
        grape="Совиньон Блан",
        final_score=0.95,
        visual_score=0.9,
    )
    assert not decide_match(
        winner,
        None,
        "красное сухое каберне",
        method="ocr_visual",
        exclusive=True,
        shortlist_size=1,
        known_grape_tokens=KNOWN,
    )


def test_garbage_ocr_does_not_reorder_visual_neighbors():
    white = _wine(
        slug="white",
        name="Белое",
        category="Белое",
        grape="Совиньон Блан",
        visual_score=0.80,
        final_score=0.80,
    )
    red = _wine(
        slug="fanagoriya-kaberne-krasnoe-suhoe",
        name="Красное",
        category="Красное",
        grape="Каберне Совиньон",
        visual_score=0.74,
        final_score=0.74,
    )
    garbage = build_similar([white, red], "к8 фааговия хнко", KNOWN)
    assert [item["slug"] for item in garbage] == [
        "white",
        "fanagoriya-kaberne-krasnoe-suhoe",
    ]

    ranked = build_similar([white, red], "красное сухое каберне", KNOWN)
    assert ranked[0]["slug"] == "fanagoriya-kaberne-krasnoe-suhoe"
    assert "тот же цвет" in ranked[0]["reason"]
    assert "тот же сорт" in ranked[0]["reason"]
    assert "похожая сладость" in ranked[0]["reason"]


def test_latin_lookalike_color_is_not_garbage():
    white = _wine(
        slug="white",
        category="Белое",
        grape="Совиньон Блан",
        visual_score=0.82,
    )
    red = _wine(
        slug="red",
        category="Красное",
        grape="Каберне Совиньон",
        visual_score=0.70,
    )
    ranked = build_similar([white, red], "KPACHOE каберне", KNOWN)
    assert ranked[0]["slug"] == "red"
    assert "тот же цвет" in ranked[0]["reason"]
    assert "тот же сорт" in ranked[0]["reason"]


def test_description_keywords_are_a_weak_tie_break():
    plain = _wine(
        slug="plain",
        visual_score=0.60,
        description="Свежее вино.",
        category="Красное",
    )
    oak = _wine(
        slug="oak",
        visual_score=0.60,
        description="Во вкусе вишня и дуб.",
        category="Красное",
    )
    ranked = build_similar([plain, oak], "аромат вишни и дуба", KNOWN)
    assert ranked[0]["slug"] == "oak"


def test_external_profile_ranks_sauvignon_above_red():
    """Attribute match only. No visual score and no SigLIP."""

    profile = ExternalProfile(grape="Совиньон Блан", color="белое", name="Тест")
    white = _wine(
        slug="sovinon-blan",
        name="Совиньон Блан сухое",
        category="Белое",
        grape="Совиньон Блан",
        color="соломенный",
        visual_score=0.1,
    )
    red = _wine(
        slug="merlo-krasnoe",
        name="Мерло",
        category="Красное",
        grape="Мерло",
        color="рубиновый",
        visual_score=0.99,
    )
    ranked = rank_catalog_by_attributes([red, white], profile, KNOWN)
    assert ranked
    assert ranked[0]["slug"] == "sovinon-blan"
    assert ranked[0]["role"] == "primary"
    assert "тот же сорт" in ranked[0]["reason"]
    assert "тот же цвет" in ranked[0]["reason"]
    assert "визуально" not in ranked[0]["reason"]
    assert all(item["slug"] != "merlo-krasnoe" or "тот же сорт" not in item["reason"] for item in ranked)
    assert ranked[0]["slug"] != "merlo-krasnoe"


def _stem_encoder(texts: list[str]):
    """Stand-in for the SigLIP text tower: one axis per taste or grape stem."""

    import numpy as np

    stems = (
        "СОВИНЬОН",
        "АЛИГОТЕ",
        "МЕРЛО",
        "ЦИТРУС",
        "ЯБЛОК",
        "ВИШН",
        "БЕЛОЕ",
        "КРАСНОЕ",
    )
    rows = []
    for text in texts:
        folded = str(text).upper()
        rows.append([1.0 if stem in folded else 0.0 for stem in stems])
    return np.asarray(rows, dtype=np.float32)


def test_text_encoder_ranks_citrus_white_above_red():
    """Grape and taste in the blurb, not the label photo, pick the neighbor."""

    profile = ExternalProfile(
        name="Чужое белое",
        grape="Совиньон блан",
        taste="цитрус и яблоко",
        color="белое",
        text="сухое белое, цитрус, зеленое яблоко, совиньон",
    )
    white = _wine(
        slug="sovinon-citrus",
        name="Совиньон блан",
        category="Белое",
        grape="Совиньон Блан",
        color="соломенный",
        taste="цитрус, яблоко",
        description="Свежее белое с цитрусом и яблоком.",
        visual_score=0.05,
    )
    red = _wine(
        slug="merlo-krasnoe",
        name="Мерло",
        category="Красное",
        grape="Мерло",
        color="рубиновый",
        taste="вишня",
        description="Плотное красное, вишня.",
        visual_score=0.99,
    )
    ranked = rank_catalog_by_text([red, white], profile, _stem_encoder, KNOWN)
    assert ranked
    assert ranked[0]["slug"] == "sovinon-citrus"
    assert ranked[0]["role"] == "primary"
    assert ranked[-1]["slug"] != "sovinon-citrus" or len(ranked) == 1
    assert ranked[0]["text_score"] > next(
        item["text_score"] for item in ranked if item["slug"] == "merlo-krasnoe"
    )


def test_squashed_open_search_with_a_clear_visual_lead_is_found():
    winner = _wine(slug="pobeda", final_score=0.014, visual_score=0.784, category="Красное")
    runner = _wine(slug="other", final_score=0.010, visual_score=0.561, category="Красное")
    assert decide_match(
        winner,
        runner,
        "победа",
        method="visual_ocr",
        exclusive=False,
        shortlist_size=13,
        known_grape_tokens=KNOWN,
        dictionary_tokens=1,
    )
    close = _wine(slug="formula-q", final_score=0.015, visual_score=0.657)
    sibling = _wine(slug="dekanter", final_score=0.004, visual_score=0.621)
    assert not decide_match(
        close,
        sibling,
        "фанагория декантер",
        method="ocr_visual",
        exclusive=True,
        shortlist_size=3,
        known_grape_tokens=KNOWN,
        dictionary_tokens=2,
    )


def test_visual_tie_band_keeps_the_ordered_leader():
    winner = _wine(
        slug="aratti-kaberne-sovinon-2021-krasnoe-suhoe",
        grape="Каберне Совиньон",
        final_score=0.752,
        visual_score=0.752,
    )
    runner = _wine(
        slug="aratti-sovinon-blan-beloe-polusuhoe",
        grape="Совиньон Блан",
        final_score=0.769,
        visual_score=0.769,
    )
    assert decide_match(
        winner,
        runner,
        "аратти каберне 2021",
        method="ocr_visual",
        exclusive=True,
        shortlist_size=5,
        known_grape_tokens=KNOWN,
        dictionary_tokens=2,
        producer_only=False,
    )


def test_producer_clear_lead_over_a_different_bottle_is_found():
    from similar_match import distinct_runner

    leader = _wine(
        slug="aratti-kaberne-po-belomu-1",
        name="Каберне по-белому",
        visual_score=0.805,
        final_score=0.805,
    )
    copy = _wine(
        slug="aratti-kaberne-po-belomu",
        name="Каберне по-белому",
        visual_score=0.795,
        final_score=0.795,
    )
    other = _wine(
        slug="aratti-kaberne-sovinon-2021-krasnoe-suhoe",
        name="Каберне Совиньон 2021",
        visual_score=0.734,
        final_score=0.734,
    )
    assert distinct_runner([leader, copy, other])["slug"] == other["slug"]
    assert decide_match(
        leader,
        other,
        "аратти каберне",
        method="ocr_visual",
        exclusive=True,
        shortlist_size=4,
        known_grape_tokens=KNOWN,
        dictionary_tokens=2,
        producer_only=True,
    )


def main() -> None:
    tests = [
        test_zhemchuzhnaya_fixture_stays_found,
        test_low_confidence_weak_ocr_is_not_found_with_similar,
        test_close_sibling_scores_are_not_found,
        test_visual_tie_keeps_dictionary_winner,
        test_open_catalog_needs_a_large_gap,
        test_producer_only_flat_visual_is_not_a_match,
        test_unique_title_hit_stays_found,
        test_generic_grape_only_is_not_a_match,
        test_generic_grape_open_catalog_lead_can_match,
        test_unique_shortlist_keeps_clear_visual_when_text_is_squashed,
        test_saturated_text_with_a_visual_lead_stays_found,
        test_soft_unique_hit_with_clear_visual_stays_found,
        test_squashed_open_search_with_a_clear_visual_lead_is_found,
        test_visual_tie_band_keeps_the_ordered_leader,
        test_producer_clear_lead_over_a_different_bottle_is_found,
        test_full_catalog_without_tokens_needs_a_clear_lead,
        test_color_conflict_blocks_a_high_score,
        test_garbage_ocr_does_not_reorder_visual_neighbors,
        test_latin_lookalike_color_is_not_garbage,
        test_description_keywords_are_a_weak_tie_break,
        test_external_profile_ranks_sauvignon_above_red,
        test_text_encoder_ranks_citrus_white_above_red,
    ]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
