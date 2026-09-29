"""OCR shortlist without SigLIP.

Run:
  .\\venv\\Scripts\\python.exe tests\\test_ocr_shortlist.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_lexicon import OcrLexicon  # noqa: E402
from similar_match import decide_match  # noqa: E402
from ocr_shortlist import (  # noqa: E402
    build_ocr_shortlist,
    catalog_tokens,
    extract_years,
    has_second_distinctive,
    order_candidates,
    recover_names_inside_producer,
    text_tokens,
    text_visual_weights,
)


def _indexes():
    known = {"АЛИГОТЕ", "ШАРДОНЕ", "ЦИТРОННЫЙ", "МАГАРАЧА"}
    grape_to_slugs = {
        "АЛИГОТЕ": {"aligote"},
        "ШАРДОНЕ": {"chardonnay"},
        "ЦИТРОННЫЙ": {"aligote", "chardonnay"},
        "МАГАРАЧА": {"aligote", "chardonnay"},
    }
    name_to_slugs = {
        "ЖЕМЧУЖНАЯ": {"aligote"},
        "АРРАТТИ": {"other-name"},
    }
    token_df = {
        "АЛИГОТЕ": 1,
        "ШАРДОНЕ": 5,
        "ЦИТРОННЫЙ": 2,
        "МАГАРАЧА": 2,
        "ЖЕМЧУЖНАЯ": 1,
        "АРРАТТИ": 1,
        "ВИНО": 50,
        "БЕЛОЕ": 40,
    }
    return known, grape_to_slugs, name_to_slugs, token_df, 400


def _build(ocr: str):
    known, grape_to, name_to, token_df, count = _indexes()
    return build_ocr_shortlist(
        text_tokens(ocr),
        known,
        grape_to,
        name_to,
        token_df,
        count,
    )


def test_mixed_homoglyph_aligote_and_citron():
    # RapidOCR may emit Latin A/E inside a Cyrillic word.
    decision = _build("A\u043b\u0438\u0433\u043e\u0442E \u0446\u0438\u0442\u0440\u043e\u043d")

    assert decision.confident
    assert decision.grape_tokens == ["\u0410\u041b\u0418\u0413\u041e\u0422\u0415", "\u0426\u0418\u0422\u0420\u041e\u041d\u041d\u042b\u0419"] or (
        "\u0410\u041b\u0418\u0413\u041e\u0422\u0415" in decision.grape_tokens
        and "\u0426\u0418\u0422\u0420\u041e\u041d\u041d\u042b\u0419" in decision.grape_tokens
    )
    assert "aligote" in decision.slugs
    assert "chardonnay" not in decision.slugs


def test_aligote_excludes_chardonnay():
    decision = _build("вино АЛИГОТЕ цитронный белое")

    assert decision.confident
    assert "aligote" in decision.slugs
    assert "chardonnay" not in decision.slugs
    assert "АЛИГОТЕ" in decision.grape_tokens
    assert decision.reason.startswith("grape:")


def test_chardonnay_excludes_aligote_only():
    decision = _build("CHARDONNAY")

    assert decision.confident
    assert decision.slugs == ["chardonnay"]
    assert "ШАРДОНЕ" in decision.grape_tokens


def test_fuzzy_aligot_maps_to_aligote_not_chardonnay():
    decision = _build("АЛИГОТ")

    assert decision.confident
    assert "aligote" in decision.slugs
    assert "chardonnay" not in decision.slugs
    assert decision.grape_tokens == ["АЛИГОТЕ"]


def test_spurious_second_grape_does_not_wipe_anchor():
    decision = _build("АЛИГОТЕ ШАРДОНЕ")

    assert decision.confident
    assert "aligote" in decision.slugs
    assert "chardonnay" not in decision.slugs


def test_empty_and_weak_ocr_are_not_confident():
    empty = _build("")
    weak = _build("вино белое сухое")

    assert not empty.confident
    assert empty.slugs == []
    assert empty.reason == "empty"

    assert not weak.confident
    assert weak.slugs == []
    assert weak.reason == "weak"


def test_rare_name_token_shortlists_without_opening_other_grapes():
    decision = _build("ЖЕМЧУЖНАЯ")

    assert decision.confident
    assert decision.slugs == ["aligote"]
    assert decision.reason.startswith("name:")
    assert "chardonnay" not in decision.slugs


def test_leading_three_becomes_ze():
    assert "ЗОЛОТО" in text_tokens("3ОЛОТО")
    assert "ЗО24" not in text_tokens("вино 2024")


def test_spaced_winery_letters_join():
    tokens = text_tokens("т а б и я вино белое")
    assert "ТАБИЯ" in tokens


def test_krasnos_does_not_lock_krasnostop():
    known, grape_to, name_to, token_df, count = _indexes()
    known = set(known) | {"КРАСНОСТОП"}
    grape_to = dict(grape_to)
    grape_to["КРАСНОСТОП"] = {"krasnostop-wine"}

    decision = build_ocr_shortlist(
        text_tokens("РЕЛИКТА КРАСНОС сухое"),
        known,
        grape_to,
        name_to,
        token_df,
        count,
    )

    assert "КРАСНОСТОП" not in decision.grape_tokens
    assert "krasnostop-wine" not in decision.slugs


def test_exact_winery_shortlists_the_brand():
    known, grape_to, name_to, token_df, count = _indexes()
    decision = build_ocr_shortlist(
        text_tokens("т а б и я"),
        known,
        grape_to,
        name_to,
        token_df,
        count,
        winery_to_slugs={"ТАБИЯ": {"pobeda", "rozovoe-zoloto"}},
    )

    assert decision.confident
    assert decision.reason.startswith("winery:")
    assert set(decision.slugs) == {"pobeda", "rozovoe-zoloto"}
    assert decision.strong
    assert not decision.soft


def test_belaya_alone_is_not_a_wine():
    decision = _build("БЕЛАЯ вино")
    assert not decision.confident
    assert decision.slugs == []


def test_year_token_does_not_pull_in_another_wine():
    known, grape_to, name_to, token_df, count = _indexes()
    name_to = dict(name_to)
    token_df = dict(token_df)
    name_to["2О24"] = {"chardonnay"}
    token_df["2О24"] = 1

    decision = build_ocr_shortlist(
        text_tokens("ЖЕМЧУЖНАЯ 2024"),
        known,
        grape_to,
        name_to,
        token_df,
        count,
    )

    assert decision.confident
    assert decision.slugs == ["aligote"]
    assert "2О24" not in decision.name_tokens
    assert "chardonnay" not in decision.slugs


def _load_real_catalog():
    csv_path = ROOT / "data" / "catalog" / "strapi_output0709.csv"
    if not csv_path.is_file():
        return None

    known: set[str] = set()
    grape_to: dict[str, set[str]] = {}
    name_to: dict[str, set[str]] = {}
    winery_to: dict[str, set[str]] = {}
    token_df: dict[str, int] = {}
    slug_grapes: dict[str, set[str]] = {}
    seen_slugs: set[str] = set()

    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            slug = str(row.get("Slug") or "").strip()
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            grape_tokens = set(text_tokens(row.get("Сорт винограда") or ""))
            name_tokens = set(text_tokens(row.get("Название вина") or ""))
            winery_tokens = set(text_tokens(row.get("Винодельня") or ""))
            slug_grapes[slug] = grape_tokens
            known.update(grape_tokens)

            for token in grape_tokens:
                grape_to.setdefault(token, set()).add(slug)
            for token in name_tokens:
                name_to.setdefault(token, set()).add(slug)
            for token in winery_tokens:
                if len(token) >= 4:
                    winery_to.setdefault(token, set()).add(slug)
            for token in grape_tokens | name_tokens | winery_tokens:
                token_df[token] = token_df.get(token, 0) + 1

    return known, grape_to, name_to, token_df, slug_grapes, len(seen_slugs), winery_to


def test_real_catalog_aligote_excludes_chardonnay_only():
    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, grape_to, name_to, token_df, slug_grapes, count, _winery = loaded

    decision = build_ocr_shortlist(
        text_tokens("АЛИГОТЕ"),
        known,
        grape_to,
        name_to,
        token_df,
        count,
    )

    assert decision.confident, decision.reason
    assert decision.slugs, "expected Aligote wines in the real catalog"
    assert "АЛИГОТЕ" in decision.grape_tokens

    chardonnay_only = next(
        slug
        for slug, grapes in slug_grapes.items()
        if "ШАРДОНЕ" in grapes and "АЛИГОТЕ" not in grapes
    )

    assert chardonnay_only not in decision.slugs
    for slug in decision.slugs:
        assert "АЛИГОТЕ" in slug_grapes[slug], slug

    print(
        f"real catalog: shortlist={len(decision.slugs)} "
        f"reason={decision.reason} "
        f"excluded={chardonnay_only}"
    )


def test_zhemchuzhnaya_citron_ranks_above_aristov_blend():
    """OCR ЖЕМЧУЖНАЯ АЛИГОТЕ ЦИТРОН АРАТТИ.

    Цитрон on the label is Цитронный Магарача in the catalog.
    A blend that only contains Aligote must rank below the wine
    whose grape field has both. No SigLIP.
    """

    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, grape_to, name_to, token_df, slug_grapes, count, _winery = loaded
    ocr = "ЖЕМЧУЖНАЯ АЛИГОТЕ ЦИТРОН АРАТТИ вино белое сухое брют 2024"
    decision = build_ocr_shortlist(
        text_tokens(ocr),
        known,
        grape_to,
        name_to,
        token_df,
        count,
    )

    target = "zhemchuzhnaya-9-aligote-czitron"
    blend = "aristov-anima-millesimato-beloe-bryut"
    chardonnay_only = next(
        slug
        for slug, grapes in slug_grapes.items()
        if "ШАРДОНЕ" in grapes and "АЛИГОТЕ" not in grapes
    )

    assert decision.confident, decision.reason
    assert "АЛИГОТЕ" in decision.grape_tokens
    assert "ЦИТРОННЫЙ" in decision.grape_tokens
    assert target in decision.slugs

    def rank(slug: str) -> int:
        if slug in decision.slugs:
            return decision.slugs.index(slug)
        return len(decision.slugs) + 1

    assert rank(target) < rank(blend)
    assert rank(target) < rank(chardonnay_only)
    assert blend not in decision.slugs
    assert chardonnay_only not in decision.slugs

    for slug in decision.slugs:
        assert "АЛИГОТЕ" in slug_grapes[slug], slug
        assert "ЦИТРОННЫЙ" in slug_grapes[slug], slug

    text_weight, visual_weight = text_visual_weights(
        len(decision.grape_tokens) + len(decision.name_tokens)
    )
    assert text_weight == 0.75
    assert visual_weight == 0.25
    assert text_weight > visual_weight

    print(
        f"zhemchuzhnaya: shortlist={len(decision.slugs)} "
        f"reason={decision.reason} "
        f"grapes={decision.grape_tokens} "
        f"names={decision.name_tokens} "
        f"rank={rank(target)}"
    )


def _rank_zhemchuzhnaya(ocr: str):
    loaded = _load_real_catalog()
    if loaded is None:
        return None

    known, grape_to, name_to, token_df, slug_grapes, count, winery_to = loaded
    clean = catalog_tokens(
        ocr,
        known,
        name_to,
        token_df,
        count,
        winery_to,
    )
    decision = build_ocr_shortlist(
        clean,
        known,
        grape_to,
        name_to,
        token_df,
        count,
        winery_to,
    )
    target = "zhemchuzhnaya-9-aligote-czitron"
    blend = "aristov-anima-millesimato-beloe-bryut"
    chardonnay_only = next(
        slug
        for slug, grapes in slug_grapes.items()
        if "ШАРДОНЕ" in grapes and "АЛИГОТЕ" not in grapes
    )

    def rank(slug: str) -> int:
        if slug in decision.slugs:
            return decision.slugs.index(slug)
        return len(decision.slugs) + 1

    assert "АЛИГОТЕ" in clean
    assert "ЦИТРОННЫЙ" in clean
    assert decision.confident, decision.reason
    assert rank(target) < rank(blend)
    assert rank(target) < rank(chardonnay_only)
    assert blend not in decision.slugs
    assert chardonnay_only not in decision.slugs
    assert decision.strong
    return clean, decision


def test_mixed_script_garbage_does_not_lock():
    known, grape_to, name_to, token_df, count = _indexes()
    ocr = "XQZPLT ЖЖЖЖ HOBLESS BELOE 2024 Aрррр"
    clean = catalog_tokens(ocr, known, name_to, token_df, count)
    decision = build_ocr_shortlist(
        clean,
        known,
        grape_to,
        name_to,
        token_df,
        count,
    )

    assert clean == []
    assert not decision.confident
    assert not decision.strong
    assert decision.slugs == []


def test_latin_catalog_name_is_kept():
    known, grape_to, name_to, token_df, count = _indexes()
    name_to = dict(name_to)
    token_df = dict(token_df)
    name_to["GOLUBITSKOE"] = {"golubitskoe-estate-chardonnay"}
    token_df["GOLUBITSKOE"] = 1

    clean = catalog_tokens(
        "GOLUBITSKOE estate",
        known,
        name_to,
        token_df,
        count,
    )

    assert "GOLUBITSKOE" in clean
    decision = build_ocr_shortlist(
        clean,
        known,
        grape_to,
        name_to,
        token_df,
        count,
    )
    assert "golubitskoe-estate-chardonnay" in decision.slugs


def test_latin_token_not_in_catalog_is_ignored():
    known, grape_to, name_to, token_df, count = _indexes()
    name_to = dict(name_to)
    token_df = dict(token_df)
    name_to["GOLUBITSKOE"] = {"golubitskoe-estate-chardonnay"}
    token_df["GOLUBITSKOE"] = 1

    clean = catalog_tokens(
        "HOBLESS OBLIG GOLUBITSKOE",
        known,
        name_to,
        token_df,
        count,
    )

    assert clean == ["GOLUBITSKOE"]
    assert "HOBLESS" not in clean
    assert "OBLIG" not in clean


def test_citron_sauvignon_ranks_above_chardonnay_sibling():
    """ЦИТРОН + СОВИНЬОН БЛАН must beat the Цитрон+Шардоне sibling.

    The catalog slug is czitronnyj-magaracha-sovinon-blan
    (Жемчужная 9 Цитрон, Совиньон блан). Алиготе+Цитрон stays
    on its own wine when Совиньон is absent.
    """

    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, grape_to, name_to, token_df, slug_grapes, count, winery_to = loaded
    target = "czitronnyj-magaracha-sovinon-blan"
    sibling = "zhemchuzhnaya-9-czitron-shardone-1"
    aligote = "zhemchuzhnaya-9-aligote-czitron"

    def rank_of(ocr: str, slug: str) -> int:
        clean = catalog_tokens(
            ocr, known, name_to, token_df, count, winery_to,
        )
        decision = build_ocr_shortlist(
            clean, known, grape_to, name_to, token_df, count, winery_to,
        )
        assert "ЦИТРОННЫЙ" in clean, clean
        assert "СОВИНЬОН" in clean, clean
        assert decision.confident, decision.reason
        if slug in decision.slugs:
            return decision.slugs.index(slug)
        return len(decision.slugs) + 1

    plain = "ЖЕМЧУЖНАЯ ЦИТРОН СОВИНЬОН БЛАН АРАТТИ белое"
    messy = "ЖЕМЧУЖНАЯ ЦИТР0Н CОВИНЬОН БЛAН APA\u0422T\u0418"
    # RapidOCR on the real bottle: Ь read as Ђ, В dropped, БЛАН glued.
    detected = "\u0416\u0435\u043c\u0447\u0443\u0436\u043d\u0430\u044f \u0446\u0438\u0442\u0440\u043e\u043d \u0441\u043e\u0438\u043d\u0452\u043e\u043d\u043b\u0430\u043d"
    latin = "ZHEMCHUZHNAYA CITRON SAUVIGNON BLANC ARATTI"

    for ocr in (plain, messy, detected, latin):
        assert rank_of(ocr, target) < rank_of(ocr, sibling), ocr
        assert rank_of(ocr, sibling) > rank_of(ocr, target)

    clean = catalog_tokens(
        plain, known, name_to, token_df, count, winery_to,
    )
    decision = build_ocr_shortlist(
        clean, known, grape_to, name_to, token_df, count, winery_to,
    )
    assert sibling not in decision.slugs
    assert target in decision.slugs
    assert aligote not in decision.slugs

    aligote_ocr = "ЖЕМЧУЖНАЯ АЛИГОТЕ ЦИТРОН АРАТТИ белое"
    aligote_clean = catalog_tokens(
        aligote_ocr, known, name_to, token_df, count, winery_to,
    )
    aligote_decision = build_ocr_shortlist(
        aligote_clean, known, grape_to, name_to, token_df, count, winery_to,
    )
    assert aligote in aligote_decision.slugs
    assert target not in aligote_decision.slugs


def test_lexicon_zhemchuzhnaya_russian_and_mixed_latin():
    loaded = _load_real_catalog()
    if loaded is None:
        return

    russian = _rank_zhemchuzhnaya(
        "ЖЕМЧУЖНАЯ АЛИГОТЕ ЦИТРОН АРАТТИ вино белое сухое брют 2024"
    )
    assert russian is not None
    clean, _decision = russian
    assert "АРАТТИ" in clean

    mixed = _rank_zhemchuzhnaya(
        "ALIGOTE CITRON BELOE SUKHOE 2024 ARATTI"
    )
    assert mixed is not None
    clean, _decision = mixed
    assert "АРАТТИ" in clean
    assert "BELOE" not in clean
    assert "SUKHOE" not in clean

    # Known-photo mix: Cyrillic word with Latin lookalikes, plus Latin fillers.
    homoglyph = _rank_zhemchuzhnaya(
        "A\u043b\u0438\u0433\u043e\u0442E \u0446\u0438\u0442\u0440\u043e\u043d APA\u0422T\u0418 \u0431\u0435\u043b\u043e\u0435"
    )
    assert homoglyph is not None
    print(f"lexicon clean russian={russian[0]} latin={mixed[0]} mixed={homoglyph[0]}")


def test_fanagoria_leftover_ranks_lekif_above_saqra():
    """ФАНАГОРИЯ plus a near-miss of ЛЕКИФ stays inside that house.

    The real catalog has SAQRA and no Lekif title, so the snap is
    checked on a two-wine house that uses those tokens.
    """

    known: set[str] = set()
    grape_to: dict[str, set[str]] = {}
    name_to = {
        "ЛЕКИФ": {"fanagoriya-lekif"},
        "SAQRA": {"fanagoriya-saqra-saperavi-krasnoe-suhoe-135"},
    }
    token_df = {"ФАНАГОРИЯ": 2, "ЛЕКИФ": 1, "SAQRA": 1}
    winery_to = {
        "ФАНАГОРИЯ": {
            "fanagoriya-lekif",
            "fanagoriya-saqra-saperavi-krasnoe-suhoe-135",
        },
    }
    kept = recover_names_inside_producer(
        ["ФАНАГОРИЯ"],
        ["ЛЕКИП"],
        name_to_slugs=name_to,
        winery_to_slugs=winery_to,
        token_df=token_df,
    )
    assert "ЛЕКИФ" in kept
    decision = build_ocr_shortlist(
        kept, known, grape_to, name_to, token_df, 2, winery_to,
    )
    assert decision.strong
    assert "fanagoriya-lekif" in decision.slugs
    assert "fanagoriya-saqra-saperavi-krasnoe-suhoe-135" not in decision.slugs


def test_litavshchuk_excludes_other_producers():
    loaded = _load_real_catalog()
    if loaded is None:
        return
    known, grape_to, name_to, token_df, _slug_grapes, count, winery_to = loaded
    decision = build_ocr_shortlist(
        ["ЛИТАВЩУК"],
        known, grape_to, name_to, token_df, count, winery_to,
    )
    assert decision.strong
    assert decision.slugs
    assert "pozdnij-sbor-krasnoe" in decision.slugs
    assert "fanagoriya-saqra-saperavi-krasnoe-suhoe-135" not in decision.slugs
    for slug in decision.slugs:
        assert slug in winery_to["ЛИТАВЩУК"]


def test_alveus_locks_the_line_and_cuvee_does_not():
    loaded = _load_real_catalog()
    if loaded is None:
        return
    known, grape_to, name_to, token_df, _slug_grapes, count, winery_to = loaded
    lexicon = OcrLexicon.build(
        grapes=known,
        names=name_to.keys(),
        wineries=winery_to.keys(),
    )
    cleaned = lexicon.clean("NlVeus Kюbe")
    assert "ALVEUS" in cleaned.tokens
    decision = build_ocr_shortlist(
        cleaned.tokens, known, grape_to, name_to, token_df, count, winery_to,
    )
    assert decision.strong, decision.reason
    assert decision.slugs
    assert all("alveus" in slug for slug in decision.slugs)
    assert "usadba-divnomorskoe-grande-cuvee-shardone-beloe-bryut-125" not in decision.slugs

    cuvee_only = build_ocr_shortlist(
        ["КЮВЕ"], known, grape_to, name_to, token_df, count, winery_to,
    )
    assert not cuvee_only.strong


def test_belenkoe_ending_beats_generic_pino():
    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, grape_to, name_to, token_df, _slug_grapes, count, winery_to = loaded
    name_to = dict(name_to)
    token_df = dict(token_df)
    # The live catalog has no Беленькое. The recovery still has to rank
    # that title above the single Pinot that PINO used to lock.
    name_to.setdefault("БЕЛЕНЬКОЕ", set()).add("belenkoe")
    token_df["БЕЛЕНЬКОЕ"] = 1
    bad = "cloudy-winery-plohaya-devochka-sochnyy-pino-noir-pino-nuar-krasnoe-suhoe-12"
    lexicon = OcrLexicon.build(
        grapes=known,
        names=name_to.keys(),
        wineries=winery_to.keys(),
    )
    cleaned = lexicon.clean("zот сuс PiNOТ енькое еленькое")
    assert "БЕЛЕНЬКОЕ" in cleaned.tokens, cleaned.tokens
    assert "ПИНО" in cleaned.tokens
    decision = build_ocr_shortlist(
        cleaned.tokens, known, grape_to, name_to, token_df, count, winery_to,
    )
    assert "belenkoe" in decision.slugs, decision.reason
    assert decision.slugs != [bad]
    if bad in decision.slugs:
        assert decision.slugs.index("belenkoe") < decision.slugs.index(bad)


def test_aragti_filters_to_aratti():
    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, grape_to, name_to, token_df, _slug_grapes, count, winery_to = loaded
    assert "АРАТТИ" in winery_to
    lexicon = OcrLexicon.build(
        grapes=known,
        names=name_to.keys(),
        wineries=winery_to.keys(),
    )
    cleaned = lexicon.clean("AРАГТИ coвинo 2о21")
    assert "АРАТТИ" in cleaned.tokens, cleaned.tokens
    assert "СОВИНЬОН" in cleaned.tokens, cleaned.tokens
    decision = build_ocr_shortlist(
        cleaned.tokens, known, grape_to, name_to, token_df, count, winery_to,
    )
    house = winery_to["АРАТТИ"]
    assert decision.strong, decision.reason
    assert decision.slugs
    assert set(decision.slugs) <= house
    assert len(decision.slugs) > 1

    alone = build_ocr_shortlist(
        ["СОВИНЬОН"], known, grape_to, name_to, token_df, count, winery_to,
    )
    assert alone.generic_only
    assert not alone.strong
    assert len(alone.slugs) > 1

    assert extract_years("AРАГТИ coвинo 2о21") == ["2021"]
    ranked = order_candidates(
        [
            {
                "slug": "aratti-kaberne-sovinon-2020-krasnoe-suhoe",
                "name": "Каберне Совиньон 2020",
                "visual_score": 0.70,
                "final_score": 0.70,
            },
            {
                "slug": "aratti-kaberne-sovinon-2021-krasnoe-suhoe",
                "name": "Каберне Совиньон 2021",
                "visual_score": 0.70,
                "final_score": 0.70,
            },
        ],
        allow_rerank=False,
        years=["2021"],
    )
    assert ranked[0]["slug"] == "aratti-kaberne-sovinon-2021-krasnoe-suhoe"
    close = order_candidates(
        [
            {
                "slug": "aratti-sovinon-blan-beloe-polusuhoe",
                "name": "Совиньон Блан",
                "visual_score": 0.7694,
                "final_score": 0.7694,
            },
            {
                "slug": "aratti-kaberne-sovinon-2021-krasnoe-suhoe",
                "name": "Каберне Совиньон 2021",
                "visual_score": 0.7523,
                "final_score": 0.7523,
            },
        ],
        allow_rerank=False,
        years=["2021"],
    )
    assert close[0]["slug"] == "aratti-kaberne-sovinon-2021-krasnoe-suhoe"


def test_fanagoria_without_a_second_name_follows_visual():
    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, grape_to, name_to, token_df, _slug_grapes, count, winery_to = loaded
    decision = build_ocr_shortlist(
        ["ФАНАГОРИЯ"], known, grape_to, name_to, token_df, count, winery_to,
    )
    assert decision.strong
    assert decision.winery_token == "ФАНАГОРИЯ"
    assert not has_second_distinctive(
        decision.grape_tokens,
        decision.name_tokens,
        decision.winery_token,
    )
    green = "fanagoriya-zelyonoe-vino-risling-tsitronnyy-magaracha-beloe-polusuhoe-11"
    saqra = "fanagoriya-saqra-saperavi-krasnoe-suhoe-135"
    assert green in decision.slugs
    assert saqra in decision.slugs
    ranked = order_candidates(
        [
            {
                "slug": green,
                "name": "Зелёное вино",
                "winery": "Фанагория",
                "visual_score": 0.59,
                "final_score": 0.97,
            },
            {
                "slug": saqra,
                "name": "SAQRA",
                "winery": "Фанагория",
                "visual_score": 0.62,
                "final_score": 0.70,
            },
        ],
        allow_rerank=False,
        years=[],
    )
    assert ranked[0]["slug"] == saqra


def test_zhemchuzhnaya_aligote_citron_stays_one_wine():
    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, grape_to, name_to, token_df, _slug_grapes, count, winery_to = loaded
    lexicon = OcrLexicon.build(
        grapes=known,
        names=name_to.keys(),
        wineries=winery_to.keys(),
    )
    cleaned = lexicon.clean("жечужная АлиготЕ цитрон")
    decision = build_ocr_shortlist(
        cleaned.tokens, known, grape_to, name_to, token_df, count, winery_to,
    )
    assert "ЖЕМЧУЖНАЯ" in cleaned.tokens
    assert "АЛИГОТЕ" in cleaned.tokens
    assert "ЦИТРОННЫЙ" in cleaned.tokens
    assert decision.slugs == ["zhemchuzhnaya-9-aligote-czitron"]


def test_aratti_belaya_beats_stray_nuar():
    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, grape_to, name_to, token_df, _slug_grapes, count, winery_to = loaded
    decision = build_ocr_shortlist(
        ["АРАТТИ", "БЕЛАЯ", "НУАР"],
        known,
        grape_to,
        name_to,
        token_df,
        count,
        winery_to,
    )
    assert decision.slugs == ["belaya-lvicza"], decision.reason
    assert decision.strong
    assert not decision.grape_conflict

    lexicon = OcrLexicon.build(
        grapes=known,
        names=name_to.keys(),
        wineries=winery_to.keys(),
    )
    assert "НУАР" not in lexicon.clean("HYAP").tokens
    assert "НУАР" not in lexicon.clean("HVAP ING").tokens
    pair = lexicon.clean("Ino Hyap")
    assert "ПИНО" in pair.tokens and "НУАР" in pair.tokens

    photo = (
        "HVAP боой гир ARATT белая ђви ING HYAP "
        "ARATTI BE^AA"
    )
    cleaned = lexicon.clean(photo)
    assert "НУАР" not in cleaned.tokens, cleaned.tokens
    assert "АРАТТИ" in cleaned.tokens
    assert "БЕЛАЯ" in cleaned.tokens
    photo_decision = build_ocr_shortlist(
        cleaned.tokens,
        known,
        grape_to,
        name_to,
        token_df,
        count,
        winery_to,
    )
    assert photo_decision.slugs == ["belaya-lvicza"], photo_decision.reason
    assert not photo_decision.grape_conflict
    winner = {
        "slug": "belaya-lvicza",
        "name": "Белая Львица белое полусладкое",
        "grape": "Рислинг Рейнский, Совиньон Блан",
        "category": "Белое",
        "winery": "АРАТТИ",
        "visual_score": 0.753,
        "final_score": 0.9845,
    }
    assert decide_match(
        winner,
        None,
        photo,
        method="ocr_visual",
        exclusive=True,
        shortlist_size=1,
        known_grape_tokens=known,
        dictionary_tokens=2,
    )


def test_wont_does_not_force_a_latin_name():
    loaded = _load_real_catalog()
    if loaded is None:
        return
    known, grape_to, name_to, token_df, _slug_grapes, count, winery_to = loaded
    lexicon = OcrLexicon.build(
        grapes=known,
        names=name_to.keys(),
        wineries=winery_to.keys(),
    )
    cleaned = lexicon.clean("WONT DI ILUT WONT DE ILEUK")
    assert cleaned.tokens == []


if __name__ == "__main__":
    test_mixed_homoglyph_aligote_and_citron()
    test_aligote_excludes_chardonnay()
    test_chardonnay_excludes_aligote_only()
    test_fuzzy_aligot_maps_to_aligote_not_chardonnay()
    test_spurious_second_grape_does_not_wipe_anchor()
    test_empty_and_weak_ocr_are_not_confident()
    test_rare_name_token_shortlists_without_opening_other_grapes()
    test_leading_three_becomes_ze()
    test_spaced_winery_letters_join()
    test_krasnos_does_not_lock_krasnostop()
    test_exact_winery_shortlists_the_brand()
    test_belaya_alone_is_not_a_wine()
    test_year_token_does_not_pull_in_another_wine()
    test_real_catalog_aligote_excludes_chardonnay_only()
    test_zhemchuzhnaya_citron_ranks_above_aristov_blend()
    test_mixed_script_garbage_does_not_lock()
    test_latin_catalog_name_is_kept()
    test_latin_token_not_in_catalog_is_ignored()
    test_citron_sauvignon_ranks_above_chardonnay_sibling()
    test_lexicon_zhemchuzhnaya_russian_and_mixed_latin()
    test_fanagoria_leftover_ranks_lekif_above_saqra()
    test_litavshchuk_excludes_other_producers()
    test_alveus_locks_the_line_and_cuvee_does_not()
    test_belenkoe_ending_beats_generic_pino()
    test_aragti_filters_to_aratti()
    test_fanagoria_without_a_second_name_follows_visual()
    test_zhemchuzhnaya_aligote_citron_stays_one_wine()
    test_aratti_belaya_beats_stray_nuar()
    test_wont_does_not_force_a_latin_name()
    print("ok")
