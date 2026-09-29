"""Catalog lexicon cleanup before the OCR shortlist.

Run:
  .\\venv\\Scripts\\python.exe tests\\test_ocr_lexicon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from external_lookup.web_lookup import build_web_query  # noqa: E402
from ocr_lexicon import OcrLexicon  # noqa: E402
from ocr_shortlist import build_ocr_shortlist  # noqa: E402
from test_ocr_shortlist import _load_real_catalog  # noqa: E402


def _tiny():
    return OcrLexicon.build(
        grapes={"АЛИГОТЕ", "ШАРДОНЕ", "ЦИТРОННЫЙ"},
        names={"ЖЕМЧУЖНАЯ", "GOLUBITSKOE"},
        wineries={"АРАТТИ"},
    )


def test_garbage_latin_is_dropped():
    cleaned = _tiny().clean("xyzqwk HOBLESS 2024 вино")
    assert cleaned.tokens == []
    assert cleaned.cleaned_text == ""


def test_real_latin_catalog_token_is_kept():
    cleaned = _tiny().clean("xyzqwk GOLUBITSKOE estate")
    assert cleaned.tokens == ["GOLUBITSKOE"]
    assert cleaned.hits[0].kind == "name"
    assert cleaned.hits[0].score >= 0.99


def test_cyrillic_typos_snap_to_dictionary():
    cleaned = _tiny().clean("АЛИГАТЕ ЖЕМЧЖНАЯ")
    assert "АЛИГОТЕ" in cleaned.tokens
    assert "ЖЕМЧУЖНАЯ" in cleaned.tokens
    by_catalog = {hit.catalog: hit for hit in cleaned.hits}
    assert by_catalog["АЛИГОТЕ"].score >= 0.84
    assert by_catalog["ЖЕМЧУЖНАЯ"].score >= 0.84
    assert by_catalog["АЛИГОТЕ"].raw != "АЛИГОТЕ"


def test_homoglyph_soup_shortlists_zhemchuzhnaya():
    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, grape_to, name_to, token_df, slug_grapes, count, winery_to = loaded
    lexicon = OcrLexicon.build(
        grapes=known,
        names=name_to.keys(),
        wineries=winery_to.keys(),
    )
    ocr = (
        "xyzqwk жемчжная "
        "A\u043b\u0438\u0433\u043e\u0442E "
        "\u0446\u0438\u0442\u0440\u043e\u043d "
        "\u0410\u0420\u0410\u0422\u0422\u0418 "
        "вино белое"
    )
    cleaned = lexicon.clean(ocr)
    assert "АЛИГОТЕ" in cleaned.tokens
    assert "ЦИТРОННЫЙ" in cleaned.tokens
    assert "xyzqwk" not in cleaned.tokens
    assert "ВИНО" not in cleaned.tokens

    decision = build_ocr_shortlist(
        cleaned.tokens,
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

    assert decision.confident, decision.reason
    assert "АЛИГОТЕ" in decision.grape_tokens
    assert "ЦИТРОННЫЙ" in decision.grape_tokens
    assert rank(target) < rank(blend)
    assert rank(target) < rank(chardonnay_only)
    assert blend not in decision.slugs
    assert chardonnay_only not in decision.slugs
    assert len(decision.slugs) != 1 or decision.strong
    print(
        f"homoglyph soup: tokens={cleaned.tokens} "
        f"shortlist={len(decision.slugs)} rank={rank(target)}"
    )


def test_clipped_grape_and_one_edit_alias_snap():
    lexicon = OcrLexicon.build(
        grapes={"СИРА", "МУСКАТ", "КАБЕРНЕ"},
        names={"ЗОЛОТО", "ЗОВОТО"},
    )
    cleaned = lexicon.clean("сир MUSCAI зоNото xyzqwk")
    assert "СИРА" in cleaned.tokens
    assert "МУСКАТ" in cleaned.tokens
    assert "ЗОЛОТО" not in cleaned.tokens
    assert "xyzqwk" not in cleaned.tokens
    # Two names share the pattern, so the stray letter stays unmatched.
    alone = OcrLexicon.build(grapes={"СИРА"}, names={"ЗОЛОТО"})
    snapped = alone.clean("зоNото")
    assert snapped.tokens == ["ЗОЛОТО"]


def test_homoglyph_brand_snaps_when_the_word_is_in_the_dictionary():
    lexicon = OcrLexicon.build(
        grapes=set(),
        names={"АЛАЗАНСКАЯ"},
        wineries={"MOGZAURI"},
    )
    cleaned = lexicon.clean("MOGZAUR! AJA3AHCKAЯ долинА")
    assert "MOGZAURI" in cleaned.tokens
    assert "АЛАЗАНСКАЯ" in cleaned.tokens


def test_spaced_tabia_and_clipped_pinot():
    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, grape_to, name_to, token_df, _slug_grapes, count, winery_to = loaded
    assert "ТАБИЯ" in winery_to
    lexicon = OcrLexicon.build(
        grapes=known,
        names=name_to.keys(),
        wineries=winery_to.keys(),
    )
    cleaned = lexicon.clean("Tа б и я иноаеаьня пно уар полусухое 2025")
    assert "ТАБИЯ" in cleaned.tokens, cleaned
    assert "ПИНО" in cleaned.tokens, cleaned
    assert "НУАР" in cleaned.tokens, cleaned
    assert "ПОЛУСУХОЕ" not in cleaned.tokens
    assert "2025" not in cleaned.tokens
    assert "ВИНОДЕЛЬНЯ" not in cleaned.tokens

    decision = build_ocr_shortlist(
        cleaned.tokens,
        known,
        grape_to,
        name_to,
        token_df,
        count,
        winery_to,
    )
    # Табия does not bottle Pinot Noir. Conflict, not the house leader.
    assert decision.grape_conflict, decision.reason
    assert decision.slugs == []
    assert "pobeda" not in decision.slugs

    pino_only = build_ocr_shortlist(
        ["ПИНО"],
        known,
        grape_to,
        name_to,
        token_df,
        count,
        winery_to,
    )
    assert pino_only.generic_only
    assert not pino_only.strong
    assert len(pino_only.slugs) > 1


def test_absent_brand_still_builds_a_web_query():
    lexicon = OcrLexicon.build(grapes=set(), names=set(), wineries=set())
    cleaned = lexicon.clean("MOGZAURI АЛАЗАНСКАЯ долина xyzqwk")
    assert cleaned.tokens == []
    assert "MOGZAURI" in cleaned.readable
    assert "АЛАЗАНСКАЯ" in cleaned.readable
    assert "ДОЛИНА" in cleaned.readable
    assert "XYZQWK" not in cleaned.readable
    query = build_web_query(
        "",
        [{"catalog": token, "type": "extra"} for token in cleaned.readable],
    )
    folded = query.casefold()
    assert "mogzauri" in folded
    assert "алазанская" in folded
    assert "долина" in folded
    assert "xyzqwk" not in folded


def test_hyap_alone_is_not_noir_but_ino_hyap_is_pinot_noir():
    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, _grape_to, name_to, _token_df, _slug_grapes, _count, winery_to = loaded
    lexicon = OcrLexicon.build(
        grapes=known,
        names=name_to.keys(),
        wineries=winery_to.keys(),
    )
    assert "НУАР" not in lexicon.clean("HYAP").tokens
    assert "НУАР" not in lexicon.clean("HVAP").tokens
    assert "НУАР" not in lexicon.clean("ING").tokens

    pair = lexicon.clean("Ino Hyap")
    assert "ПИНО" in pair.tokens, pair.tokens
    assert "НУАР" in pair.tokens, pair.tokens

    clipped = lexicon.clean("по ђар")
    assert "ПИНО" in clipped.tokens, clipped.tokens
    assert "НУАР" in clipped.tokens, clipped.tokens

    cyr = lexicon.clean("пно уар")
    assert "ПИНО" in cyr.tokens, cyr.tokens
    assert "НУАР" in cyr.tokens, cyr.tokens


def test_garbage_does_not_lock_shortlist():
    loaded = _load_real_catalog()
    if loaded is None:
        return

    known, grape_to, name_to, token_df, _slug_grapes, count, winery_to = loaded
    lexicon = OcrLexicon.build(
        grapes=known,
        names=name_to.keys(),
        wineries=winery_to.keys(),
    )
    cleaned = lexicon.clean("xyzqwk qwertyuiop")
    decision = build_ocr_shortlist(
        cleaned.tokens,
        known,
        grape_to,
        name_to,
        token_df,
        count,
        winery_to,
    )
    assert cleaned.tokens == []
    assert not decision.confident
    assert decision.slugs == []


if __name__ == "__main__":
    test_clipped_grape_and_one_edit_alias_snap()
    test_garbage_latin_is_dropped()
    test_real_latin_catalog_token_is_kept()
    test_cyrillic_typos_snap_to_dictionary()
    test_homoglyph_soup_shortlists_zhemchuzhnaya()
    test_homoglyph_brand_snaps_when_the_word_is_in_the_dictionary()
    test_spaced_tabia_and_clipped_pinot()
    test_hyap_alone_is_not_noir_but_ino_hyap_is_pinot_noir()
    test_absent_brand_still_builds_a_web_query()
    test_garbage_does_not_lock_shortlist()
    print("ok")
