"""Rank our catalog by text, not by the label photo.

The query is an external blurb or leftover OCR. Each catalog row is
grape, color, taste, and description. Cosine is computed on vectors
from a caller-supplied encoder. Production uses the SigLIP text tower;
tests pass a small stand-in so they do not download a model.
"""

from __future__ import annotations

import re

import numpy as np

from ocr_shortlist import normalize_text, text_tokens
from similar_match import (
    ATTRIBUTE_SIMILAR_LIMIT,
    COLOR_TEXT_LIMIT,
    DESCRIPTION_LIMIT,
    ExternalProfile,
    attribute_reasons,
    coarse_color,
    color_of_token,
    extract_query_signals,
    short_text,
    signals_from_profile,
    sweetness_in_text,
    wine_signals,
)


def _join(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and str(part).strip())


def catalog_blob(wine: dict) -> str:
    return _join(
        str(wine.get("grape") or ""),
        str(wine.get("color") or ""),
        str(wine.get("category") or ""),
        str(wine.get("taste") or ""),
        str(wine.get("sugar") or ""),
        str(wine.get("description") or ""),
        str(wine.get("name") or ""),
    )


def profile_blob(profile: ExternalProfile | dict) -> str:
    item = profile if isinstance(profile, ExternalProfile) else ExternalProfile(
        name=str((profile or {}).get("name") or ""),
        grape=str((profile or {}).get("grape") or ""),
        color=str((profile or {}).get("color") or ""),
        sweetness=str((profile or {}).get("sweetness") or ""),
        taste=str((profile or {}).get("taste") or ""),
        region=str((profile or {}).get("region") or ""),
        winery=str((profile or {}).get("winery") or ""),
        source=str((profile or {}).get("source") or ""),
        text=str((profile or {}).get("text") or ""),
    )
    return _join(
        item.grape,
        item.color,
        item.sweetness,
        item.taste,
        item.text,
        item.name,
    )


def _attribute_boost(query, wine) -> float:
    """Small grape/color lift. Cosine stays the main rank signal."""

    bonus = 0.0
    if query.grapes and wine.grapes and (set(query.grapes) & set(wine.grapes)):
        bonus += 0.08
    if query.color and wine.color and query.color == wine.color:
        bonus += 0.04
    return bonus


def _normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def rank_catalog_by_vectors(
    catalog: list[dict],
    query_vector: np.ndarray,
    matrix: np.ndarray,
    known_grape_tokens: set[str] | None = None,
    profile: ExternalProfile | dict | None = None,
    limit: int = ATTRIBUTE_SIMILAR_LIMIT,
) -> list[dict]:
    """Cosine against a precomputed catalog matrix. Row i is catalog[i]."""

    if matrix is None or len(catalog) == 0 or len(matrix) != len(catalog):
        return []
    query_vec = _normalize(query_vector)[0]
    vectors = _normalize(matrix)
    scores = vectors @ query_vec
    grapes = known_grape_tokens or set()
    query_signals = signals_from_profile(profile or {}, grapes)
    boosted = np.array(scores, dtype=np.float32, copy=True)
    for index, wine in enumerate(catalog):
        boosted[index] += _attribute_boost(query_signals, wine_signals(wine, grapes))
    order = np.argsort(-boosted, kind="stable")
    cards: list[dict] = []
    seen: set[str] = set()
    for position in order:
        wine = catalog[int(position)]
        slug = str(wine.get("slug") or "")
        if slug and slug in seen:
            continue
        if slug:
            seen.add(slug)
        profile_wine = wine_signals(wine, grapes)
        reason = attribute_reasons(query_signals, profile_wine, wine)
        if not reason:
            reason = "близкое описание"
        cards.append({
            "slug": slug or None,
            "name": wine.get("name") or "",
            "winery": wine.get("winery") or "",
            "grape": wine.get("grape") or "",
            "color": short_text(wine.get("color") or "", COLOR_TEXT_LIMIT),
            "category": wine.get("category") or "",
            "region": wine.get("region") or "",
            "description": short_text(wine.get("description") or "", DESCRIPTION_LIMIT),
            "image": wine.get("image") or "",
            "taste": wine.get("taste") or "",
            "sugar": wine.get("sugar") or "",
            "food": wine.get("food") or "",
            "reason": reason,
            "role": "primary" if not cards else "alternative",
            "text_score": round(float(scores[int(position)]), 4),
        })
        if len(cards) >= limit:
            break
    return cards


def rank_catalog_by_text(
    catalog: list[dict],
    profile: ExternalProfile | dict,
    encode,
    known_grape_tokens: set[str] | None = None,
    limit: int = ATTRIBUTE_SIMILAR_LIMIT,
) -> list[dict]:
    """Cosine of encoded blurbs. `encode` maps a list of strings to vectors.

    Wines with an empty text blob are skipped. Ties keep catalog order.
    Reasons still name the shared grape or color when those fields match,
    otherwise the card says the description is close.
    """

    query = profile_blob(profile)
    if not query:
        return []

    rows: list[tuple[int, dict, str]] = []
    for index, wine in enumerate(catalog):
        blob = catalog_blob(wine)
        if blob:
            rows.append((index, wine, blob))
    if not rows:
        return []

    vectors = _normalize(encode([blob for _index, _wine, blob in rows]))
    query_vec = _normalize(encode([query]))[0]
    scores = vectors @ query_vec
    grapes = known_grape_tokens or set()
    query_signals = signals_from_profile(profile, grapes)
    boosted = np.array(scores, dtype=np.float32, copy=True)
    for position, (_index, wine, _blob) in enumerate(rows):
        boosted[position] += _attribute_boost(query_signals, wine_signals(wine, grapes))
    order = np.argsort(-boosted, kind="stable")
    cards: list[dict] = []
    seen: set[str] = set()
    for position in order:
        wine = rows[int(position)][1]
        slug = str(wine.get("slug") or "")
        if slug and slug in seen:
            continue
        if slug:
            seen.add(slug)
        profile_wine = wine_signals(wine, grapes)
        reason = attribute_reasons(query_signals, profile_wine, wine)
        if not reason:
            reason = "близкое описание"
        cards.append({
            "slug": slug or None,
            "name": wine.get("name") or "",
            "winery": wine.get("winery") or "",
            "grape": wine.get("grape") or "",
            "color": short_text(wine.get("color") or "", COLOR_TEXT_LIMIT),
            "category": wine.get("category") or "",
            "region": wine.get("region") or "",
            "description": short_text(wine.get("description") or "", DESCRIPTION_LIMIT),
            "image": wine.get("image") or "",
            "taste": wine.get("taste") or "",
            "sugar": wine.get("sugar") or "",
            "food": wine.get("food") or "",
            "reason": reason,
            "role": "primary" if not cards else "alternative",
            "text_score": round(float(scores[int(position)]), 4),
        })
        if len(cards) >= limit:
            break
    return cards


# Short Russian asks land in a narrow SigLIP band (unrelated sentences
# still cosine around 0.8). Color and grape are a hard filter; cosine
# only breaks ties inside that set. Food and sweetness are smaller lifts.
_COSINE_TIE = 0.2
_FOOD_LIFT = 1.6
_SWEET_LIFT = 1.1
_SWEET_MISS = 0.35
_OVERLAP_LIFT = 0.12
_OVERLAP_CAP = 3
_SAME_VECTOR = 0.9999

# Stems, not whole dictionary forms: "рыбе", "мясу", "сыра", "десерта".
_FOOD_STEMS = (
    ("РЫБ", "fish"),
    ("МЯС", "meat"),
    ("СЫР", "cheese"),
    ("ДЕСЕРТ", "dessert"),
)
MIN_RECOMMEND_CHARS = 3
SHORT_QUERY_MESSAGE = "Напишите чуть подробнее: сорт, вкус или к какой еде."

# Ordinary words that are not a grape or a dish. They must not lift every
# white or every "вино" in the catalog when two cosines are close.
_QUERY_STOP = frozenset({
    "ВИНО", "ВИНА", "ВИНУ", "ВИНОМ", "ВИНЕ", "ВИННЫЙ", "ВИННОЕ",
    "ХОЧУ", "ХОТЕЛ", "ХОТЕЛА", "НУЖЕН", "НУЖНА", "НУЖНО",
    "КАКОЕ", "КАКОЙ", "КАКАЯ", "КАКИЕ",
    "ПОЖАЛУЙСТА", "ОЧЕНЬ", "МЕНЯ", "МНЕ",
    "БУТЫЛКА", "БУТЫЛКУ", "БУТЫЛОК",
    "БЕЛОЕ", "БЕЛЫЙ", "БЕЛОГО", "БЕЛУЮ",
    "КРАСНОЕ", "КРАСНЫЙ", "КРАСНОГО",
    "РОЗОВОЕ", "РОЗОВЫЙ", "РОЗОВОГО",
    "ОРАНЖЕВОЕ", "ОРАНЖЕВЫЙ",
    "СУХОЕ", "СУХОЙ", "СУХОГО",
    "СЛАДКОЕ", "СЛАДКИЙ", "СЛАДКОГО",
    "ПОЛУСУХОЕ", "ПОЛУСУХОЙ", "ПОЛУСЛАДКОЕ", "ПОЛУСЛАДКИЙ",
    "ВКУС", "ВКУСА", "ВКУСЕ", "ВКУСОМ",
    "СТИЛЬ", "СТИЛЯ",
    "ЕДА", "ЕДУ", "ЕДЕ", "ЕДЫ",
    "ХОРОШО", "ХОРОШЕЕ", "ХОРОШИЙ", "ХОРОШАЯ",
    "ПОДОЙДЕТ", "ПОДОБРАТЬ", "ПОДБЕРИ", "ПОДБЕРИТЕ",
})


def query_too_short_message(text: str) -> str | None:
    """Russian 400 text when the ask is empty or only a couple of letters."""

    cleaned = " ".join(str(text or "").split())
    if len(cleaned) < MIN_RECOMMEND_CHARS:
        return SHORT_QUERY_MESSAGE
    return None


def _fold_tokens(text: str) -> list[str]:
    folded = str(text or "").upper().replace("Ё", "Е")
    return re.findall(r"[A-ZА-Я]{3,}", folded)


def _content_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in _fold_tokens(text):
        if len(token) < 4 or token in _QUERY_STOP or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _overlap_count(sentence: str, wine: dict) -> int:
    """Shared grape or food stems between the ask and this catalog row."""

    asked = _content_tokens(sentence)
    if not asked:
        return 0
    hay = _fold_tokens(_join(
        str(wine.get("name") or ""),
        str(wine.get("grape") or ""),
        str(wine.get("description") or ""),
    ))
    hay_stems = {token[:3] for token in hay if len(token) >= 3}
    hits = 0
    seen: set[str] = set()
    for token in asked:
        stem = token[:3]
        if stem in seen or stem not in hay_stems:
            continue
        seen.add(stem)
        hits += 1
    return hits


def _food_labels(text: str) -> set[str]:
    norm = normalize_text(text)
    if not norm:
        return set()
    return {label for stem, label in _FOOD_STEMS if stem in norm}


def _grape_lexicon(catalog: list[dict], known: set[str] | None) -> set[str]:
    lexicon = set(known or ())
    for wine in catalog:
        lexicon.update(text_tokens(wine.get("grape") or ""))
    return lexicon


def _wine_color_class(wine: dict) -> str | None:
    """Coarse white/red/rose/orange from category, slug, then the color column."""

    found = coarse_color(str(wine.get("category") or ""), str(wine.get("slug") or ""))
    if found:
        return found
    for token in text_tokens(str(wine.get("color") or "")):
        color = color_of_token(token)
        if color:
            return color
    return None


def _wine_sweetness(wine: dict, lexicon: set[str]) -> str | None:
    sweet = wine_signals(wine, lexicon).sweetness
    if sweet:
        return sweet
    return sweetness_in_text(_join(
        str(wine.get("sugar") or ""),
        str(wine.get("taste") or ""),
    ))


def _eligible_rows(
    catalog: list[dict],
    lexicon: set[str],
    color: str | None,
    grapes: list[str],
) -> list[int]:
    """Wines that match a stated color and grape, when any such wine exists."""

    asked = set(grapes)
    parsed: list[tuple[int, str | None, set[str]]] = []
    for index, wine in enumerate(catalog):
        wine_grapes = set(wine_signals(wine, lexicon).grapes)
        parsed.append((index, _wine_color_class(wine), wine_grapes))

    def color_ok(item: tuple[int, str | None, set[str]]) -> bool:
        if not color:
            return True
        return item[1] == color

    def grape_ok(item: tuple[int, str | None, set[str]], *, need_all: bool) -> bool:
        if not asked:
            return True
        overlap = asked & item[2]
        if not overlap:
            return False
        if need_all:
            return asked <= item[2]
        return True

    indexes = [item[0] for item in parsed]
    if color and asked:
        both = [item[0] for item in parsed if color_ok(item) and grape_ok(item, need_all=True)]
        if both:
            return both
        partial = [item[0] for item in parsed if color_ok(item) and grape_ok(item, need_all=False)]
        if partial:
            return partial
    if asked:
        full = [item[0] for item in parsed if grape_ok(item, need_all=True)]
        if full:
            return full
        some = [item[0] for item in parsed if grape_ok(item, need_all=False)]
        if some:
            return some
    if color:
        colored = [item[0] for item in parsed if item[1] == color]
        if colored:
            return colored
    return indexes


def rank_catalog_for_sentence(
    catalog: list[dict],
    query_vector: np.ndarray,
    matrix: np.ndarray,
    sentence: str,
    limit: int = ATTRIBUTE_SIMILAR_LIMIT,
    known_grape_tokens: set[str] | None = None,
) -> list[dict]:
    """Rank by color, grape, sweetness, and food parsed from the sentence.

    SigLIP cosine is only a tie-break: short Russian phrases sit too close
    together for cosine to choose the color or the grape. A stated color
    or grape drops wines that do not match, when the catalog has any that do.
    `matrix[i]` belongs to `catalog[i]`. Identical vectors are one wine.
    """

    if matrix is None or len(catalog) == 0 or len(matrix) != len(catalog):
        return []
    if not str(sentence or "").strip():
        return []

    lexicon = _grape_lexicon(catalog, known_grape_tokens)
    asked = extract_query_signals(sentence, lexicon)
    foods = _food_labels(sentence)
    eligible = set(_eligible_rows(catalog, lexicon, asked.color, asked.grapes))

    query_vec = _normalize(query_vector)[0]
    vectors = _normalize(matrix)
    cosines = vectors @ query_vec
    boosted = np.full(len(catalog), -1.0e9, dtype=np.float32)
    for index in eligible:
        wine = catalog[index]
        score = _COSINE_TIE * float(cosines[index])
        if foods:
            wine_food = _food_labels(_join(
                str(wine.get("description") or ""),
                str(wine.get("food") or ""),
                str(wine.get("name") or ""),
            ))
            if foods & wine_food:
                score += _FOOD_LIFT
        if asked.sweetness:
            sweet = _wine_sweetness(wine, lexicon)
            if sweet:
                score += _SWEET_LIFT if sweet == asked.sweetness else -_SWEET_MISS
        hits = _overlap_count(sentence, wine)
        if hits:
            score += _OVERLAP_LIFT * min(hits, _OVERLAP_CAP)
        boosted[index] = score

    order = np.argsort(-boosted, kind="stable")
    cards: list[dict] = []
    seen: set[str] = set()
    kept: list[np.ndarray] = []
    for position in order:
        row = int(position)
        if row not in eligible:
            break
        wine = catalog[row]
        slug = str(wine.get("slug") or "")
        if slug and slug in seen:
            continue
        vector = vectors[row]
        if any(float(np.dot(vector, previous)) >= _SAME_VECTOR for previous in kept):
            continue
        if slug:
            seen.add(slug)
        kept.append(vector)
        cards.append({
            "slug": slug or None,
            "name": wine.get("name") or "",
            "winery": wine.get("winery") or "",
            "grape": wine.get("grape") or "",
            "color": short_text(wine.get("color") or "", COLOR_TEXT_LIMIT),
            "category": wine.get("category") or "",
            "description": short_text(wine.get("description") or "", DESCRIPTION_LIMIT),
            "image": wine.get("image") or "",
            "score": round(float(boosted[row]), 4),
        })
        if len(cards) >= limit:
            break
    return cards
