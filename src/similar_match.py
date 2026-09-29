"""Found vs not-found, and similar catalog wines when the photo is not a match.

The photo does not need to be in the dataset. Similarity uses fields the
catalog already has (category as the coarse color, grape, sweetness words
inside the name, slug, or description, plus winery and region) together
with SigLIP neighbors. There is no separate taste model.

A wine is returned only when the evidence is strong: one shortlist
wine from a rare title plus its producer, or two distinctive grapes
that agree. A producer-only list with a flat visual top, a generic
grape or CHATEAU alone, a grape that producer does not bottle, and a
full-catalog search with no dictionary token stay not_in_catalog.
A generic grape can still match when the visual leader clears the
same open-catalog bar. The no-token search can still match when the
visual leader is clearly ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz.fuzz import ratio

from ocr_shortlist import (
    _LATIN_TO_CYRILLIC,
    LINE_FILLERS,
    _resolve_grape,
    fold_mixed_homoglyphs,
    is_brut_misread,
    is_generic_grape,
    is_generic_token,
    normalize_text,
    recovered_pinot_noir,
    text_tokens,
)


# Locked to one catalog wine (exclusive OCR shortlist).
SINGLE_MIN_SCORE = 0.55
# Same lock, when the logistic head squashes final_score near 0.
# The visual score of that single bottle still has to be clear.
UNIQUE_VISUAL_MIN = 0.65
# Several shortlist wines pinned near 1.0, so the final gap is tiny.
# A strict visual lead with this text score is still the leader.
SATURATED_TEXT_MIN = 0.90

# OCR shortlist has several wines. Need a visible gap.
EXCLUSIVE_MIN_SCORE = 0.58
EXCLUSIVE_MIN_GAP = 0.08

# Soft OCR or the full catalog. Only a lone leader counts.
OPEN_MIN_SCORE = 0.75
OPEN_MIN_GAP = 0.20

# SigLIP neighbor that is ahead of the runner-up. A modest lead is not
# enough on its own: see NO_DICTIONARY_* and PRODUCER_ONLY_MIN_GAP.
VISUAL_WIN_MIN = 0.50
VISUAL_WIN_GAP = 0.05
# Smaller than this, the two visual scores are a tie. A dictionary
# token may keep the final-score winner. A wider deficit still blocks it.
VISUAL_TIE = 0.02

# Both neighbors below this, and no dictionary token, is not a bottle.
WEAK_VISUAL = 0.45

# Producer name only, several bottles. A flatter visual top is not a hit.
PRODUCER_ONLY_MIN_GAP = 0.03
# Same house, but the leader is clearly ahead of a different bottle.
# Duplicate catalog rows of one wine do not count as that neighbor.
PRODUCER_CLEAR_GAP = 0.06
PRODUCER_CLEAR_VISUAL = 0.78
# Logistic head output near zero is not a low visual score.
SQUASHED_TEXT_MAX = 0.25

# Full catalog after OCR kept no dictionary token. Both must hold.
NO_DICTIONARY_MIN_GAP = 0.04
NO_DICTIONARY_MIN_SCORE = 0.70

SIMILAR_LIMIT = 3
# One primary catalog neighbor plus up to three alternatives.
ATTRIBUTE_SIMILAR_LIMIT = 4
DESCRIPTION_LIMIT = 180
COLOR_TEXT_LIMIT = 120

# Attribute bonuses stay smaller than a typical SigLIP gap inside the
# neighbor list, except a clear color/grape agreement which should
# outrank a slightly closer label of the wrong kind of wine.
COLOR_BONUS = 0.18
COLOR_PENALTY = 0.25
GRAPE_BONUS = 0.16
SWEET_BONUS = 0.10
SWEET_PENALTY = 0.08
WINERY_BONUS = 0.05
REGION_BONUS = 0.04
TASTE_WORD_BONUS = 0.02
TASTE_WORD_CAP = 0.06

_COLOR_EXACT = {
    "БЕЛОЕ": "white",
    "БЕЛЫЙ": "white",
    "БЕЛОГО": "white",
    "КРАСНОЕ": "red",
    "КРАСНЫЙ": "red",
    "КРАСНОГО": "red",
    "РОЗОВОЕ": "rose",
    "РОЗОВЫЙ": "rose",
    "РОЗОВОГО": "rose",
    "ОРАНЖЕВОЕ": "orange",
    "ОРАНЖЕВЫЙ": "orange",
}

_SWEET_EXACT = {
    "СУХОЕ": "dry",
    "СУХОЙ": "dry",
    "СУХОГО": "dry",
    "СЛАДКОЕ": "sweet",
    "СЛАДКИЙ": "sweet",
    "СЛАДКОГО": "sweet",
    "ПОЛУСУХОЕ": "semi_dry",
    "ПОЛУСУХОЙ": "semi_dry",
    "ПОЛУСЛАДКОЕ": "semi_sweet",
    "ПОЛУСЛАДКИЙ": "semi_sweet",
    "БРЮТ": "brut",
    "BRUT": "brut",
}

# Stems checked after the longer "полу…" forms are removed.
_SWEET_STEMS = (
    ("ПОЛУСЛАД", "semi_sweet"),
    ("POLUSLAD", "semi_sweet"),
    ("ПОЛУСУХ", "semi_dry"),
    ("POLUSUH", "semi_dry"),
    ("POLUSUKH", "semi_dry"),
    ("БРЮТ", "brut"),
    ("BRUT", "brut"),
    ("СЛАДК", "sweet"),
    ("SLADK", "sweet"),
    ("СУХОЕ", "dry"),
    ("СУХОЙ", "dry"),
    ("СУХОГО", "dry"),
    ("SUHOE", "dry"),
    ("SUHOY", "dry"),
    ("SUKH", "dry"),
)

_TASTE_STEMS = (
    "ЯГОД",
    "ДУБ",
    "ЦИТРУС",
    "ВАНИЛ",
    "ШОКОЛАД",
    "ВИШН",
    "СЛИВ",
    "ЯБЛОК",
    "ЦВЕТОЧ",
    "МИНЕРАЛ",
    "МЕДОВ",
    "ПЕРСИК",
    "СМОРОДИН",
    "МАЛИН",
    "КЛУБНИК",
    "ТАНИН",
    "КАРАМЕЛ",
    "ЛИМОН",
    "АПЕЛЬСИН",
    "ГРЕЙПФРУТ",
    "АБРИКОС",
    "ЧЕРЕШН",
)

_REASON_COLOR = "тот же цвет"
_REASON_GRAPE = "тот же сорт"
_REASON_SWEET = "похожая сладость"
_REASON_REGION = "тот же регион"
_REASON_BRAND = "общий бренд"
_REASON_TASTE = "похожий вкус"
_REASON_VISUAL = "визуально близкая этикетка"

_TASTE_RU = {
    "ЯГОД": "ягоды",
    "ДУБ": "дуб",
    "ЦИТРУС": "цитрус",
    "ВАНИЛ": "ваниль",
    "ШОКОЛАД": "шоколад",
    "ВИШН": "вишня",
    "СЛИВ": "слива",
    "ЯБЛОК": "яблоко",
    "ЦВЕТОЧ": "цветы",
    "МИНЕРАЛ": "минерал",
    "МЕДОВ": "мёд",
    "ПЕРСИК": "персик",
    "СМОРОДИН": "смородина",
    "МАЛИН": "малина",
    "КЛУБНИК": "клубника",
    "ТАНИН": "танин",
    "КАРАМЕЛ": "карамель",
    "ЛИМОН": "лимон",
    "АПЕЛЬСИН": "апельсин",
    "ГРЕЙПФРУТ": "грейпфрут",
    "АБРИКОС": "абрикос",
    "ЧЕРЕШН": "черешня",
}

_COLOR_RU = {
    "white": "белое",
    "red": "красное",
    "rose": "розовое",
    "orange": "оранжевое",
}
_SWEET_RU = {
    "dry": "сухое",
    "sweet": "сладкое",
    "semi_dry": "полусухое",
    "semi_sweet": "полусладкое",
    "brut": "брют",
}


@dataclass
class QuerySignals:
    """OCR cues that are real catalog attributes. Other tokens are ignored."""

    color: str | None = None
    sweetness: str | None = None
    grapes: list[str] = field(default_factory=list)
    taste_stems: set[str] = field(default_factory=set)
    brands: list[str] = field(default_factory=list)
    regions: set[str] = field(default_factory=set)


@dataclass
class WineSignals:
    color: str | None = None
    sweetness: str | None = None
    grapes: list[str] = field(default_factory=list)
    winery_tokens: set[str] = field(default_factory=set)
    region_tokens: set[str] = field(default_factory=set)
    taste_stems: set[str] = field(default_factory=set)


def _near(token: str, target: str, minimum: float = 0.80) -> bool:
    if abs(len(token) - len(target)) > 1 or len(token) < 5:
        return False
    return ratio(token, target) / 100.0 >= minimum


def _label_forms(token: str) -> list[str]:
    """Cyrillic form plus a Latin lookalike reading (KPACHOE -> КРАСНОЕ)."""

    forms = [token]
    mixed = fold_mixed_homoglyphs(token)
    if mixed not in forms:
        forms.append(mixed)
    if token.isascii():
        cyrillic = token.translate(_LATIN_TO_CYRILLIC)
        if cyrillic not in forms:
            forms.append(cyrillic)
    return forms


def color_of_token(token: str, known_grape_tokens: set[str] | None = None) -> str | None:
    """Map a label word to white/red/rose/orange. Grapes are not colors."""

    for form in _label_forms(token):
        if form in _COLOR_EXACT:
            return _COLOR_EXACT[form]
        if form.startswith("ОРАНЖ") or form.startswith("ORANGE"):
            return "orange"
        if form in {"WHITE", "RED", "ROSE"}:
            return {"WHITE": "white", "RED": "red", "ROSE": "rose"}[form]
    if known_grape_tokens and token in known_grape_tokens:
        return None
    # КРАСНОСТОП is a grape; КРАСНОС is a clipped "красное".
    if "СТОП" in token or token.startswith("КРАСНОСТ"):
        return None
    if token.startswith("КРАСНА"):
        return None
    if _near(token, "КРАСНОЕ") or _near(token, "КРАСНЫЙ"):
        return "red"
    if _near(token, "БЕЛОЕ") or _near(token, "БЕЛЫЙ"):
        return "white"
    if _near(token, "РОЗОВОЕ") or _near(token, "РОЗОВЫЙ"):
        return "rose"
    if _near(token, "ОРАНЖЕВОЕ"):
        return "orange"
    return None


def sweetness_of_token(token: str) -> str | None:
    if is_brut_misread(token):
        return "brut"
    for form in _label_forms(token):
        if form in _SWEET_EXACT:
            return _SWEET_EXACT[form]
        if form == "DRY":
            return "dry"
        if form == "SWEET":
            return "sweet"
        if form.startswith("ПОЛУСЛАД") or form.startswith("POLUSLAD"):
            return "semi_sweet"
        if form.startswith("ПОЛУСУХ") or form.startswith("POLUSUH"):
            return "semi_dry"
    if _near(token, "СУХОЕ") or _near(token, "СУХОЙ"):
        return "dry"
    if _near(token, "СЛАДКОЕ") or _near(token, "СЛАДКИЙ"):
        return "sweet"
    if _near(token, "ПОЛУСЛАДКОЕ"):
        return "semi_sweet"
    if _near(token, "ПОЛУСУХОЕ"):
        return "semi_dry"
    if _near(token, "БРЮТ"):
        return "brut"
    return None


def sweetness_in_text(text: str) -> str | None:
    """One sweetness class, or none when the text mentions several."""

    norm = normalize_text(text.replace("-", " "))
    if not norm:
        return None

    found: list[str] = []
    stripped = norm
    for stem, label in _SWEET_STEMS:
        if stem in stripped and label not in found:
            found.append(label)
            stripped = stripped.replace(stem, " ")
    if len(found) == 1:
        return found[0]
    return None


def coarse_color(category: str, slug: str = "") -> str | None:
    """Catalog 'Категория' is Белое/Красное/Розовое/Оранжевое.

    The free-text color column is a shade note, not the white/red/rosé bit.
    """

    norm = normalize_text(category)
    if "ОРАНЖ" in norm:
        return "orange"
    if "РОЗОВ" in norm:
        return "rose"
    if "КРАСН" in norm:
        return "red"
    if norm.startswith("БЕЛ") or " БЕЛ" in f" {norm}":
        return "white"

    slug_norm = normalize_text(slug.replace("-", " "))
    if "ORANZHEVOE" in slug_norm or "ORANGE" in slug_norm:
        return "orange"
    if "ROZOVOE" in slug_norm or "ROSE" in slug_norm:
        return "rose"
    if "KRASNOE" in slug_norm:
        return "red"
    if "BELOE" in slug_norm:
        return "white"
    return None


def _taste_stems(text: str) -> set[str]:
    norm = normalize_text(text)
    if not norm:
        return set()
    return {stem for stem in _TASTE_STEMS if stem in norm}


def taste_notes(text: str) -> str:
    """Short Russian taste words found in a blurb, in stem order."""

    found = _taste_stems(text)
    words = [_TASTE_RU[stem] for stem in _TASTE_STEMS if stem in found]
    return ", ".join(words[:5])


def brand_tokens(wine: dict) -> set[str]:
    """Name, slug, and winery. Alveus lives in the cuvee name, not the house."""

    blob = " ".join(
        part
        for part in (
            wine.get("name") or "",
            str(wine.get("slug") or "").replace("-", " "),
            wine.get("winery") or "",
        )
        if part
    )
    return set(text_tokens(blob))


def brand_hit(query: QuerySignals, wine: dict) -> bool:
    if not query.brands:
        return False
    return bool(set(query.brands) & brand_tokens(wine))


def _only(values: set[str]) -> str | None:
    if len(values) == 1:
        return next(iter(values))
    return None


def extract_query_signals(
    ocr_text: str,
    known_grape_tokens: set[str],
) -> QuerySignals:
    """Keep color, sweetness, and indexed grapes. Drop the rest.

    A garbage string adds no attribute score, so it cannot outrank
    the visual neighbor list.
    """

    tokens = text_tokens(ocr_text)
    colors: set[str] = set()
    sweets: set[str] = set()
    grapes: list[str] = []
    seen_grapes: set[str] = set()

    for token in tokens:
        color = color_of_token(token, known_grape_tokens)
        if color:
            colors.add(color)
        sweet = sweetness_of_token(token)
        if sweet:
            sweets.add(sweet)
        grape = _resolve_grape(token, known_grape_tokens)
        if grape and grape not in seen_grapes:
            seen_grapes.add(grape)
            grapes.append(grape)

    for grape in recovered_pinot_noir(tokens):
        if grape in known_grape_tokens and grape not in seen_grapes:
            seen_grapes.add(grape)
            grapes.append(grape)

    text_sweet = sweetness_in_text(ocr_text)
    if text_sweet:
        sweets.add(text_sweet)

    return QuerySignals(
        color=_only(colors),
        sweetness=_only(sweets),
        grapes=grapes,
        taste_stems=_taste_stems(ocr_text),
    )


def wine_signals(
    wine: dict,
    known_grape_tokens: set[str],
) -> WineSignals:
    grapes: list[str] = []
    seen: set[str] = set()
    for token in text_tokens(wine.get("grape") or ""):
        grape = _resolve_grape(token, known_grape_tokens)
        if grape and grape not in seen:
            seen.add(grape)
            grapes.append(grape)

    named = " ".join(
        part
        for part in (wine.get("name") or "", wine.get("slug") or "")
        if part
    )
    sweetness = sweetness_in_text(named)
    if sweetness is None:
        sweetness = sweetness_in_text(wine.get("description") or "")
    return WineSignals(
        color=coarse_color(wine.get("category") or "", wine.get("slug") or ""),
        sweetness=sweetness,
        grapes=grapes,
        winery_tokens=set(text_tokens(wine.get("winery") or "")),
        region_tokens=set(text_tokens(wine.get("region") or "")),
        taste_stems=_taste_stems(wine.get("description") or ""),
    )


def _attribute_bonus(query: QuerySignals, wine: WineSignals, ocr_tokens: list[str]) -> float:
    bonus = 0.0
    if query.color and wine.color:
        if query.color == wine.color:
            bonus += COLOR_BONUS
        else:
            bonus -= COLOR_PENALTY
    if query.sweetness and wine.sweetness:
        if query.sweetness == wine.sweetness:
            bonus += SWEET_BONUS
        else:
            bonus -= SWEET_PENALTY
    if query.grapes and wine.grapes:
        overlap = len(set(query.grapes) & set(wine.grapes))
        bonus += GRAPE_BONUS * (overlap / len(query.grapes))
    useful = [
        token
        for token in ocr_tokens
        if len(token) >= 5 and not is_generic_token(token)
    ]
    if any(token in wine.winery_tokens for token in useful):
        bonus += WINERY_BONUS
    if any(token in wine.region_tokens for token in useful):
        bonus += REGION_BONUS
    if query.taste_stems and wine.taste_stems:
        shared = len(query.taste_stems & wine.taste_stems)
        bonus += min(TASTE_WORD_CAP, TASTE_WORD_BONUS * shared)
    return bonus


def similarity_reasons(query: QuerySignals, wine: WineSignals) -> str:
    parts: list[str] = []
    if query.color and wine.color and query.color == wine.color:
        parts.append(_REASON_COLOR)
    if query.grapes and wine.grapes and (set(query.grapes) & set(wine.grapes)):
        parts.append(_REASON_GRAPE)
    if query.sweetness and wine.sweetness and query.sweetness == wine.sweetness:
        parts.append(_REASON_SWEET)
    parts.append(_REASON_VISUAL)
    return ", ".join(parts)


def attribute_reasons(
    query: QuerySignals,
    wine: WineSignals,
    wine_row: dict | None = None,
) -> str:
    """Shared catalog traits only. No visual-neighbor phrase."""

    parts: list[str] = []
    if query.grapes and wine.grapes and (set(query.grapes) & set(wine.grapes)):
        parts.append(_REASON_GRAPE)
    if query.color and wine.color and query.color == wine.color:
        parts.append(_REASON_COLOR)
        label = _COLOR_RU.get(query.color)
        if label:
            parts.append(label)
    if query.sweetness and wine.sweetness and query.sweetness == wine.sweetness:
        parts.append(_REASON_SWEET)
        label = _SWEET_RU.get(query.sweetness)
        if label:
            parts.append(label)
    if wine_row is not None and brand_hit(query, wine_row):
        parts.append(_REASON_BRAND)
    if query.taste_stems and wine.taste_stems and (query.taste_stems & wine.taste_stems):
        parts.append(_REASON_TASTE)
    if query.regions and wine.region_tokens and (query.regions & wine.region_tokens):
        parts.append(_REASON_REGION)
    return ", ".join(parts)


@dataclass
class ExternalProfile:
    """Attributes parsed from a web page, not from the label photo."""

    name: str = ""
    grape: str = ""
    color: str = ""
    sweetness: str = ""
    taste: str = ""
    region: str = ""
    winery: str = ""
    source: str = ""
    text: str = ""


def _as_profile(profile: ExternalProfile | dict) -> ExternalProfile:
    if isinstance(profile, ExternalProfile):
        return profile
    data = profile or {}
    return ExternalProfile(
        name=str(data.get("name") or ""),
        grape=str(data.get("grape") or ""),
        color=str(data.get("color") or ""),
        sweetness=str(data.get("sweetness") or data.get("taste") or ""),
        taste=str(data.get("taste") or ""),
        region=str(data.get("region") or ""),
        winery=str(data.get("winery") or ""),
        source=str(data.get("source") or ""),
        text=str(data.get("text") or ""),
    )


def signals_from_profile(
    profile: ExternalProfile | dict,
    known_grape_tokens: set[str],
) -> QuerySignals:
    """Grape, color, sweetness, and region taken from web attributes."""

    item = _as_profile(profile)
    blob = " ".join(
        part
        for part in (
            item.grape,
            item.color,
            item.sweetness,
            item.taste,
            item.region,
            item.name,
            item.text,
        )
        if part
    )
    signals = extract_query_signals(blob, known_grape_tokens)
    signals.regions = set(text_tokens(item.region))
    signals.brands = [
        token
        for token in text_tokens(item.winery)
        if len(token) >= 4 and not is_generic_token(token)
    ]
    return signals


def _diverse_neighbors(ranked, query: QuerySignals, limit: int):
    """Best overall, plus a color hit and a brand hit when the query has them."""

    picked = []
    seen: set[str] = set()

    def consider(item) -> None:
        slug = str(item[1].get("slug") or "")
        if slug and slug in seen:
            return
        if slug:
            seen.add(slug)
        picked.append(item)

    if not ranked:
        return []
    consider(ranked[0])
    if query.color:
        for item in ranked:
            if item[2].color == query.color:
                consider(item)
                break
    if query.brands:
        for item in ranked:
            if brand_hit(query, item[1]):
                consider(item)
                break
    if query.grapes:
        for item in ranked:
            if set(query.grapes) & set(item[2].grapes):
                consider(item)
                break
    for item in ranked:
        if len(picked) >= limit:
            break
        consider(item)
    return picked[:limit]


def rank_catalog_by_attributes(
    catalog: list[dict],
    profile: ExternalProfile | dict,
    known_grape_tokens: set[str],
    limit: int = ATTRIBUTE_SIMILAR_LIMIT,
) -> list[dict]:
    """Our catalog, ranked by attributes. No SigLIP score is read.

    Order: shared grape, then color, then sweetness, then region.
    The first card is the primary neighbor; the rest are alternatives.
    A wine with none of those traits is left out.
    """

    query = signals_from_profile(profile, known_grape_tokens)
    if not (
        query.grapes
        or query.color
        or query.sweetness
        or query.taste_stems
        or query.brands
        or query.regions
    ):
        return []

    ranked: list[tuple[tuple[int, int, int, int, int, int], dict, WineSignals]] = []
    for wine in catalog:
        profile_wine = wine_signals(wine, known_grape_tokens)
        grape_overlap = len(set(query.grapes) & set(profile_wine.grapes))
        color_match = int(
            bool(query.color and profile_wine.color and query.color == profile_wine.color)
        )
        sweet_match = int(
            bool(
                query.sweetness
                and profile_wine.sweetness
                and query.sweetness == profile_wine.sweetness
            )
        )
        house = int(brand_hit(query, wine))
        taste_overlap = len(query.taste_stems & profile_wine.taste_stems)
        region_overlap = len(query.regions & profile_wine.region_tokens)
        if grape_overlap + color_match + sweet_match + house + taste_overlap + region_overlap == 0:
            continue
        key = (
            grape_overlap,
            color_match,
            sweet_match,
            house,
            taste_overlap,
            region_overlap,
        )
        ranked.append((key, wine, profile_wine))

    ranked.sort(key=lambda item: item[0], reverse=True)
    picked = _diverse_neighbors(ranked, query, limit)
    cards: list[dict] = []
    seen: set[str] = set()
    for _key, wine, profile_wine in picked:
        slug = str(wine.get("slug") or "")
        if slug and slug in seen:
            continue
        if slug:
            seen.add(slug)
        reason = attribute_reasons(query, profile_wine, wine)
        if not reason:
            continue
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
        })
        if len(cards) >= limit:
            break
    return cards


def short_text(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit].rsplit(" ", 1)[0].rstrip(".,;:")
    return (cut or cleaned[:limit].rstrip()) + "…"


def conflicts_with_query(query: QuerySignals, wine: WineSignals) -> bool:
    """OCR and the leader name different color or disjoint grapes."""

    if query.color and wine.color and query.color != wine.color:
        return True
    if query.grapes and wine.grapes and not (set(query.grapes) & set(wine.grapes)):
        return True
    return False


def signals_are_generic_only(
    tokens: list[str],
    *,
    winery: str = "",
    names: list[str] | None = None,
    grapes: list[str] | None = None,
    flagged: bool = False,
) -> bool:
    """ПИНО, СОВИНЬОН, or CHATEAU alone is not a bottle.

    A producer or a rare title clears this. Empty OCR does not: the
    full-catalog visual rule still applies.
    """

    if flagged:
        return True
    if winery or names:
        return False
    grape_list = list(grapes or [])
    if any(not is_generic_grape(token) for token in grape_list):
        return False
    if grape_list:
        return True
    if not tokens:
        return False
    return all(
        is_generic_grape(token) or is_generic_token(token) or token in LINE_FILLERS
        for token in tokens
    )


def is_same_wine(
    *,
    method: str,
    top_score: float,
    second_score: float | None,
    exclusive: bool,
    shortlist_size: int,
    conflicts: bool,
    top_visual: float = 0.0,
    second_visual: float | None = None,
    dictionary_tokens: int = -1,
    generic_only: bool = False,
    producer_only: bool = False,
) -> bool:
    """True only when the leader is actually this bottle.

    A one-wine shortlist from a rare title, or two distinctive grapes
    with a visible score gap, is a match. A one-wine shortlist whose
    text score was squashed still matches when its visual score is
    clear. Several wines pinned near 1.0 still match when the leader
    is strictly ahead visually. A producer-only list whose visual top
    is flatter than PRODUCER_ONLY_MIN_GAP is not. A generic grape is
    not a bottle unless the open-catalog visual bar is cleared.
    CHATEAU is not. With no dictionary token, the visual leader must
    clear NO_DICTIONARY_MIN_GAP and NO_DICTIONARY_MIN_SCORE.
    """

    if method in {"barcode", "qr", "code"}:
        return True
    if conflicts:
        return False

    gap = 1.0 if second_score is None else top_score - second_score
    alone = second_score is None or shortlist_size <= 1
    visual_gap = 1.0 if second_visual is None else top_visual - second_visual
    both_weak = top_visual < WEAK_VISUAL and (
        second_visual is None or second_visual < WEAK_VISUAL
    )

    # ПИНО or КАБЕРНЕ alone is not a bottle. The same bar as a search
    # with no dictionary token still applies: a clear visual leader
    # on the open catalog can be this wine. A modest gap cannot.
    if generic_only:
        return (
            not exclusive
            and not both_weak
            and top_visual >= OPEN_MIN_SCORE
            and visual_gap >= OPEN_MIN_GAP
        )

    # Rare title locked to one catalog wine (АРАТТИ + БЕЛАЯ).
    # A squashed text score does not undo that lock when the photo
    # still looks like this one bottle.
    if exclusive and alone and not producer_only and (
        top_score >= SINGLE_MIN_SCORE or top_visual >= UNIQUE_VISUAL_MIN
    ):
        return True
    # Two agreeing distinctive tokens, and the scores are not tied.
    if (
        exclusive
        and not producer_only
        and top_score >= EXCLUSIVE_MIN_SCORE
        and gap >= EXCLUSIVE_MIN_GAP
    ):
        return True

    # House name only. A flat visual top is another bottle from the
    # same producer, not a confident hit.
    if producer_only and visual_gap < PRODUCER_ONLY_MIN_GAP:
        return False

    # Head pinned the shortlist near 1.0. The final gap is then noise.
    # An exact visual tie stays out (two siblings). A strict lead does not.
    if (
        exclusive
        and not producer_only
        and top_score >= SATURATED_TEXT_MIN
        and visual_gap > 0
        and top_visual >= VISUAL_WIN_MIN
    ):
        return True

    # One soft dictionary hit. The catalog was still searched. That
    # wine is the bottle when it leads visually by more than a tie.
    if (
        not exclusive
        and shortlist_size == 1
        and not producer_only
        and top_visual >= UNIQUE_VISUAL_MIN
        and visual_gap >= VISUAL_TIE
    ):
        return True

    # The head squashed every neighbor. Judge the open search on the
    # visual scores the thresholds were written for.
    if (
        not producer_only
        and top_score < SQUASHED_TEXT_MAX
        and top_visual >= OPEN_MIN_SCORE
        and visual_gap >= OPEN_MIN_GAP
    ):
        return True

    # Producer plus a generic grape. A flat top stays out. A clear
    # lead over a different bottle from that house is the label.
    if (
        producer_only
        and top_visual >= PRODUCER_CLEAR_VISUAL
        and visual_gap >= PRODUCER_CLEAR_GAP
    ):
        return True

    # No catalog word survived cleanup. A modest neighbor lead is noise.
    if dictionary_tokens == 0:
        return (
            not both_weak
            and top_visual >= NO_DICTIONARY_MIN_SCORE
            and visual_gap >= NO_DICTIONARY_MIN_GAP
        )

    # Inside the visual tie band the shortlist order already chose
    # (a vintage, or the grape). A slightly closer sibling is not
    # a different bottle. An exact tie still needs a real score gap.
    if (
        exclusive
        and not producer_only
        and top_score >= EXCLUSIVE_MIN_SCORE
        and -VISUAL_TIE <= visual_gap < 0
    ):
        return True

    # SigLIP clearly prefers the runner-up, or the two neighbors are tied.
    # A gap inside VISUAL_TIE is noise: the dictionary token already
    # chose the final-score winner, so this is not a different bottle.
    if visual_gap < -VISUAL_TIE:
        return False
    if visual_gap < 0 and dictionary_tokens <= 0:
        return False
    if visual_gap == 0 and gap < EXCLUSIVE_MIN_GAP:
        return False

    if exclusive and top_score >= EXCLUSIVE_MIN_SCORE and gap >= EXCLUSIVE_MIN_GAP:
        return True
    if top_score >= OPEN_MIN_SCORE and gap >= OPEN_MIN_GAP:
        return True
    return False


def _copy_stem(slug: str) -> str:
    """Drop a short -1 / -12 copy suffix. A vintage (-2021) stays."""

    stem = str(slug or "")
    cut = stem.rfind("-")
    suffix = stem[cut + 1 :] if cut >= 0 else ""
    if suffix.isdigit() and len(suffix) <= 2:
        return stem[:cut]
    return stem


def same_catalog_listing(left: dict, right: dict) -> bool:
    """True for a second row of the same card (slug-1, or the same name)."""

    if _copy_stem(str(left.get("slug") or "")) == _copy_stem(str(right.get("slug") or "")):
        left_slug = str(left.get("slug") or "")
        right_slug = str(right.get("slug") or "")
        if left_slug and right_slug and _copy_stem(left_slug):
            return True
    left_name = " ".join(str(left.get("name") or "").casefold().split())
    right_name = " ".join(str(right.get("name") or "").casefold().split())
    return bool(left_name) and left_name == right_name


def distinct_runner(candidates: list[dict]) -> dict | None:
    """First neighbor that is a different bottle, skipping duplicate rows."""

    if len(candidates) < 2:
        return None
    leader = candidates[0]
    for item in candidates[1:]:
        if not same_catalog_listing(leader, item):
            return item
    return None


def decide_match(
    winner: dict | None,
    runner_up: dict | None,
    ocr_text: str,
    *,
    method: str,
    exclusive: bool,
    shortlist_size: int,
    known_grape_tokens: set[str],
    dictionary_tokens: int = -1,
    generic_only: bool = False,
    producer_only: bool = False,
) -> bool:
    if not winner:
        return False
    if method in {"barcode", "qr", "code"}:
        return True
    query = extract_query_signals(ocr_text, known_grape_tokens)
    profile = wine_signals(winner, known_grape_tokens)
    second = None if runner_up is None else float(runner_up.get("final_score") or 0.0)
    second_visual = (
        None if runner_up is None else float(runner_up.get("visual_score") or 0.0)
    )
    return is_same_wine(
        method=method,
        top_score=float(winner.get("final_score") or winner.get("visual_score") or 0.0),
        second_score=second,
        exclusive=exclusive,
        shortlist_size=shortlist_size,
        conflicts=conflicts_with_query(query, profile),
        top_visual=float(winner.get("visual_score") or 0.0),
        second_visual=second_visual,
        dictionary_tokens=dictionary_tokens,
        generic_only=generic_only,
        producer_only=producer_only,
    )


def build_similar(
    candidates: list[dict],
    ocr_text: str,
    known_grape_tokens: set[str],
    limit: int = SIMILAR_LIMIT,
) -> list[dict]:
    """Re-rank visual neighbors by shared catalog attributes. Up to `limit`."""

    if not candidates:
        return []

    query = extract_query_signals(ocr_text, known_grape_tokens)
    ocr_tokens = text_tokens(ocr_text)
    ranked: list[tuple[float, float, dict]] = []

    for candidate in candidates:
        profile = wine_signals(candidate, known_grape_tokens)
        visual = float(candidate.get("visual_score") or 0.0)
        score = visual + _attribute_bonus(query, profile, ocr_tokens)
        ranked.append((score, visual, candidate))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    cards: list[dict] = []
    seen: set[str] = set()

    for _score, _visual, candidate in ranked:
        slug = str(candidate.get("slug") or "")
        if slug and slug in seen:
            continue
        if slug:
            seen.add(slug)
        profile = wine_signals(candidate, known_grape_tokens)
        cards.append({
            "slug": slug or None,
            "name": candidate.get("name") or "",
            "winery": candidate.get("winery") or "",
            "grape": candidate.get("grape") or "",
            "color": short_text(candidate.get("color") or "", COLOR_TEXT_LIMIT),
            "category": candidate.get("category") or "",
            "region": candidate.get("region") or "",
            "description": short_text(candidate.get("description") or "", DESCRIPTION_LIMIT),
            "image": candidate.get("image") or "",
            "reason": similarity_reasons(query, profile),
        })
        if len(cards) >= limit:
            break

    return cards


def public_view(
    *,
    status: str,
    ocr: str,
    method: str | None,
    ocr_clean: str = "",
    visual_compared: int,
    timing: dict | None,
    wine: dict | None,
    slug: str | None,
    confidence: float,
    similar: list[dict] | None,
    external: dict | None = None,
    message: str = "",
) -> dict:
    """API/CLI object. A miss has no slug or name of a guessed bottle.

    status not_in_catalog means the bottle is not a confident catalog
    hit. similar then lists attribute neighbors, and external is the
    web profile when a page was parsed.
    """

    if status != "found":
        shown = "not_in_catalog" if status == "not_in_catalog" else "not_found"
        payload = {
            "ocr": ocr or "",
            "ocr_normalized": ocr_clean or "",
            "ocr_clean": ocr_clean or "",
            "name": "",
            "slug": None,
            "confidence": round(float(confidence or 0.0), 4),
            "method": method,
            "visual_compared": int(visual_compared or 0),
            "timing": timing or {},
            "status": shown,
            "category": "",
            "color": "",
            "region": "",
            "grape": "",
            "winery": "",
            "description": "",
            "image": "",
            "similar": list(similar or []),
            "message": message or "",
        }
        if shown == "not_in_catalog":
            payload["external"] = external
        return payload

    card = wine or {}
    return {
        "ocr": ocr or "",
        "ocr_normalized": ocr_clean or "",
        "ocr_clean": ocr_clean or "",
        "name": card.get("name", ""),
        "slug": slug,
        "confidence": round(float(confidence or 0.0), 4),
        "method": method,
        "visual_compared": int(visual_compared or 0),
        "timing": timing or {},
        "status": "found",
        "category": card.get("category", ""),
        "color": card.get("color", ""),
        "region": card.get("region", ""),
        "grape": card.get("grape", ""),
        "winery": card.get("winery", ""),
        "description": card.get("description", ""),
        "image": card.get("image", ""),
    }
