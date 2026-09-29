"""OCR token shortlist for the catalog.

Thresholds (intentional and easy to explain):

* Grape first. An OCR token is a grape hit when it is an exact known
  grape token (length >= 4), an alias (ALIGOTE -> АЛИГОТЕ,
  ЦИТРОН / CITRON -> ЦИТРОННЫЙ), a unique longer prefix
  (ЦИТРОН -> ЦИТРОННЫЙ), or a fuzzy match with rapidfuzz ratio >= 0.90
  and a 0.04 gap over the runner-up. Fuzzy is only tried for tokens
  of length >= 5.
* Color and filler words that sit in the grape column (БЕЛЫЙ, СОРТА,
  ВИНОГРАДА) are not grapes. Those words, other generic wine words,
  and 4-digit numbers do not add a text score.
* Several grape hits: if some wines contain every detected grape, the
  shortlist is only that set (a blend missing one grape is dropped).
  If the full intersection is empty, start from the rarest and
  intersect the next grape only while the set stays non-empty, so one
  spurious extra variety does not wipe the list.
* If that grape set has at most SCORE_ALL_MAX wines, every one of them
  is a visual candidate. Above that, name-token overlap keeps the top
  SCORE_ALL_MAX; if no name token hits, the caller visually trims the
  grape set (still no other varieties).
* Name-only, when no grape hit: at least two distinctive name tokens,
  or one rare token (document frequency <= 12, length >= 5). Generic
  label words (ВИНО, СУХОЕ, …) do not count.
* Empty or weak OCR -> not confident, empty slug list. Caller must
  search the full catalog.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from rapidfuzz.fuzz import ratio


# Fuzzy grape: 90% character similarity, and clearly ahead of the
# second candidate so АЛИГОТ can map to АЛИГОТЕ without also sticking
# to a neighbor variety.
GRAPE_FUZZY_MIN = 0.90
GRAPE_FUZZY_MARGIN = 0.04
GRAPE_EXACT_MIN_LEN = 4
GRAPE_FUZZY_MIN_LEN = 5

# Whole grape-constrained set is scored (visual + text) up to this
# size. Larger sets are name-trimmed or left for a visual top-k.
SCORE_ALL_MAX = 200

NAME_MIN_LEN = 4
NAME_SINGLE_MIN_LEN = 5
NAME_RARE_DF = 12
NAME_MAX_DF_RATIO = 0.08

# Words that appear in the grape column but are not varieties.
GRAPE_STOPWORDS = frozenset({
    "СОРТА",
    "СОРТ",
    "ВИНОГРАДА",
    "ВИНОГРАД",
    "БЕЛЫЙ",
    "БЕЛОЕ",
    "ЧЕРНЫЙ",
    "ЧЕРНОЕ",
    "КРАСНЫЙ",
    "КРАСНОЕ",
    "РОЗОВЫЙ",
    "РОЗОВОЕ",
    "СУХОЕ",
    "СУХОЙ",
    "СЛАДКОЕ",
    "СЛАДКИЙ",
})

NAME_STOPWORDS = GRAPE_STOPWORDS | frozenset({
    "ВИНО",
    "ВИНА",
    "ВИНОМ",
    "WINE",
    "VIN",
    "ПОЛУСУХОЕ",
    "ПОЛУСЛАДКОЕ",
    "ПОЛУСУХОЙ",
    "БРЮТ",
    "BRUT",
    "EXTRA",
    "DRY",
    "ИГРИСТОЕ",
    "ТИХОЕ",
    "ВЫДЕРЖКА",
    "ВЫДЕРЖАННОЕ",
    "РЕЗЕРВ",
    "RESERVE",
    "КРЫМ",
    "РОССИЯ",
    "RUSSIA",
    "CRIMEA",
    "ДОЛИНА",
    "ГОД",
    "УРОЖАЙ",
    "VINTAGE",
    "ОБЪЕМ",
    "АЛКОГОЛЬ",
    "ЗГУ",
    "ЗНМП",
    "THE",
    "AND",
    "ДЛЯ",
    "ОРАНЖЕВОЕ",
    # The word "винодельня" is printed on many houses. It is not a brand.
    "ВИНОДЕЛЬНЯ",
    "ВИНОДЕЛЬНИ",
    "ВИНОДЕЛЕН",
    # Latin fillers printed on Russian labels. They are not catalog names.
    "BELOE",
    "BELOYE",
    "BELAYA",
    "SUKHOE",
    "SUHOE",
    "SUKH",
    "KRASNOE",
    "ROZOVOE",
    "SLADKOE",
    "POLUSUKHOE",
    "POLUSUHOE",
    "IGRISTOE",
})

# Style words printed on many houses. They may rerank inside a brand.
# They must not hard-lock the whole catalog (КЮВЕ → Grande Cuvee).
LINE_FILLERS = frozenset({
    "КЮВЕ",
    "CUVEE",
    "CUVE",
    "KYUVE",
    "ULTRA",
    "GRANDE",
    "BRUT",
    "БРЮТ",
    "ROSE",
    "РОЗЕ",
    "EXTRA",
    "ЭКСТРА",
    # Printed on many houses. Not a brand lock.
    "CHATEAU",
    "SHATO",
    "ШАТО",
})

# EasyOCR often emits the international name in Latin.
# ЦИТРОН on the label is stored as Цитронный Магарача.
LATIN_GRAPE_ALIASES = {
    "ALIGOTE": "АЛИГОТЕ",
    "CHARDONNAY": "ШАРДОНЕ",
    "CABERNET": "КАБЕРНЕ",
    "SAUVIGNON": "СОВИНЬОН",
    "MERLOT": "МЕРЛО",
    "RIESLING": "РИСЛИНГ",
    "PINOT": "ПИНО",
    "PINO": "ПИНО",
    "INO": "ПИНО",
    # HYAP / HVAP / ING are not НУАР by themselves. See recovered_pinot_noir.
    "NOIR": "НУАР",
    "COBUNSON": "СОВИНЬОН",
    "COBVNSON": "СОВИНЬОН",
    "BLANC": "БЛАН",
    "SYRAH": "СИРА",
    "SHIRAZ": "ШИРАЗ",
    "MUSCAT": "МУСКАТ",
    "CITRON": "ЦИТРОННЫЙ",
    "TSITRON": "ЦИТРОННЫЙ",
    "CITRONNY": "ЦИТРОННЫЙ",
    "CITRONNYI": "ЦИТРОННЫЙ",
}

# Latin spelling of a Cyrillic winery that shows up on the known photo.
# Applied only when the Cyrillic token is actually in the catalog.
LATIN_WINERY_ALIASES = {
    "ARATTI": "АРАТТИ",
}

# Catalog token is longer than the word printed on the label.
GRAPE_ALIASES = {
    "ЦИТРОН": "ЦИТРОННЫЙ",
    # Detector often drops В and glues БЛАН: соинђон / соинђонлан.
    "СОИНЬОН": "СОВИНЬОН",
    "СОИНЬОНЛАН": "СОВИНЬОН",
    "СОИНОН": "СОВИНЬОН",
    # Three-letter clip of Сира. Longer prefixes stay in _unique_prefix_grape.
    "СИР": "СИРА",
    # RapidOCR drops the И in ПИНО and the leading Н in НУАР.
    "ПНО": "ПИНО",
    "УАР": "НУАР",
    "ЬАР": "НУАР",
    "ПОАР": "НУАР",
    "СВШНЬОН": "СОВИНЬОН",
    "СВШНЬОН": "СОВИНЬОН",
}

# A lone fragment of these shapes is label noise (Белая львица reads
# HYAP). It is НУАР only when a Pinot token is on the same label.
NOIR_ONLY_WITH_PINOT = frozenset({"HYAP", "HVAP", "ING"})
PINOT_MARKERS = frozenset({"ПИНО", "PINOT", "PINO", "INO", "ПНО"})
# "по ђар" is ПИНО НУАР with the middle letters dropped. ПО alone is not Pinot.
_NOIR_STEMS = frozenset({
    "НУАР",
    "NOIR",
    "УАР",
    "ЬАР",
    "ПОАР",
    "HYAP",
    "HVAP",
    "ING",
})


def recovered_pinot_noir(tokens: list[str]) -> list[str]:
    """Catalog grapes implied by a Pinot token beside a noir fragment.

    HYAP, HVAP, and ING alone are not Pinot Noir. "Ino Hyap", "пно уар",
    and "по ђар" still are.
    """

    present = {token for token in tokens if token}
    has_pinot = bool(present & PINOT_MARKERS)
    has_gated = bool(present & NOIR_ONLY_WITH_PINOT)
    has_noir = bool(present & _NOIR_STEMS)
    found: list[str] = []
    if "ПО" in present and has_noir:
        found.extend(["ПИНО", "НУАР"])
    elif has_pinot and has_gated:
        found.append("НУАР")
    return found


# Varieties printed on dozens of labels. They may rerank inside a
# producer. Alone they must not shrink the catalog to one wine.
GENERIC_GRAPE_TOKENS = frozenset({
    "ПИНО",
    "СОВИНЬОН",
    "КАБЕРНЕ",
    "ШАРДОНЕ",
    "МЕРЛО",
    "РИСЛИНГ",
})

# Text score uses 0.75/0.25 once OCR has this many grape or rare-name hits.
TEXT_DOMINANT_MIN_HITS = 2
TEXT_DOMINANT_TEXT = 0.75
TEXT_DOMINANT_VISUAL = 0.25


def normalize_text(text: str) -> str:
    """Нормализация OCR/каталожного текста."""
    if not text:
        return ""

    text = str(text).upper()
    text = text.replace("Ё", "Е")
    # RapidOCR reads Ь in СОВИНЬОН as Serbian Ђ.
    text = text.replace("Ђ", "Ь").replace("Ћ", "Ь")

    # Типичные OCR-замены.
    text = text.replace("0", "О")
    text = text.replace("1", "I")

    text = re.sub(r"[^A-ZА-Я0-9]+", " ", text)
    return " ".join(text.split())


# RapidOCR sometimes emits a Cyrillic word with Latin lookalikes
# (AлиготE). Fold those only when the token already mixes scripts,
# so a pure Latin alias such as ALIGOTE or CITRON stays intact.
_LATIN_TO_CYRILLIC = str.maketrans({
    "A": "А",
    "B": "В",
    "C": "С",
    "E": "Е",
    "H": "Н",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
    "X": "Х",
    "Y": "У",
})

_CYR_TO_LATIN = str.maketrans({
    "А": "A",
    "В": "B",
    "С": "C",
    "Е": "E",
    "Н": "H",
    "К": "K",
    "М": "M",
    "О": "O",
    "Р": "P",
    "Т": "T",
    "Х": "X",
    "У": "Y",
})

# 5PIOT is RapidOCR reading БРЮТ/BRUT. Those letters sit next to PINOT
# in the fuzzy table, but they are sweetness, not a grape.
_BRUT_LETTER_FORMS = frozenset({"PIOT", "BRUT", "SPIOT", "BPIOT"})


def is_brut_misread(token: str) -> bool:
    """True for БРЮТ, BRUT, and the 5PIOT family. Never for PINOT."""

    raw = "".join(ch for ch in str(token or "").upper() if ch.isalnum())
    if not raw:
        return False
    cyrillic = "".join(ch for ch in raw if "А" <= ch <= "Я")
    if cyrillic == "БРЮТ":
        return True
    letters = "".join(ch for ch in raw.translate(_CYR_TO_LATIN) if ch.isalpha())
    return letters in _BRUT_LETTER_FORMS


# Lookalikes do not vote. Unambiguous letters decide the script.
_LATIN_LOOKALIKE = frozenset("ABCEHKMOPTXY")
_CYRILLIC_LOOKALIKE = frozenset("АВСЕНКМОРТХУ")
# Letters that are not Cyrillic lookalikes. A token with one of these
# keeps a Latin reading beside the Cyrillic fold.
_DISTINCTIVE_LATIN = frozenset("DFGIJLNQRSUVWZ")


def _script_votes(token: str) -> tuple[int, int, int, int]:
    """Return unambiguous Latin, unambiguous Cyrillic, Latin lookalikes, Cyrillic lookalikes."""

    latin = cyrillic = look_latin = look_cyrillic = 0
    for char in token:
        if "A" <= char <= "Z":
            if char in _LATIN_LOOKALIKE:
                look_latin += 1
            else:
                latin += 1
        elif "А" <= char <= "Я":
            if char in _CYRILLIC_LOOKALIKE:
                look_cyrillic += 1
            else:
                cyrillic += 1
    return latin, cyrillic, look_latin, look_cyrillic


def mostly_cyrillic(token: str) -> bool:
    """True when the token should be folded into Cyrillic before lookup.

    Confusable letters (A/А, E/Е, …) do not decide. A pure Latin token
    stays Latin even if every letter has a Cyrillic lookalike.
    """

    latin, cyrillic, look_latin, look_cyrillic = _script_votes(token)
    if cyrillic != latin:
        return cyrillic > latin
    return look_cyrillic > look_latin


def mostly_latin(token: str) -> bool:
    latin, cyrillic, look_latin, look_cyrillic = _script_votes(token)
    if latin != cyrillic:
        return latin > cyrillic
    return look_latin > look_cyrillic


def fold_mixed_homoglyphs(token: str) -> str:
    if mostly_cyrillic(token):
        return token.translate(_LATIN_TO_CYRILLIC)
    return token


def text_tokens(text: str, *, fold: bool = True) -> list[str]:
    text = normalize_text(text)
    raw = [
        fold_mixed_homoglyphs(token) if fold else token
        for token in text.split()
        if token
    ]

    # RapidOCR often splits a word into single letters ("т а б и я").
    merged: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if len(buffer) >= 4:
            joined = "".join(buffer)
            merged.append(fold_mixed_homoglyphs(joined) if fold else joined)
        buffer.clear()

    for token in raw:
        if len(token) == 1 and token.isalpha():
            buffer.append(token)
            continue
        flush()
        if len(token) >= 3:
            merged.append(_repair_leading_digit(token))
    flush()
    return [_repair_leading_digit(token) for token in merged]


def _repair_leading_digit(token: str) -> str:
    """OCR reads З as 3 (3ОЛОТО -> ЗОЛОТО). Years stay digits."""

    if len(token) >= 5 and token.startswith("3") and token[1:].isalpha():
        return "З" + token[1:]
    return token


def _idf(
    token: str,
    token_df: dict[str, int],
    catalog_count: int,
) -> float:
    df = token_df.get(token, 0)
    return float(
        math.log((catalog_count + 1) / (df + 1)) + 1.0
    )


@dataclass
class OcrShortlist:
    confident: bool
    slugs: list[str]
    reason: str
    grape_tokens: list[str] = field(default_factory=list)
    name_tokens: list[str] = field(default_factory=list)
    # True: grape set is larger than SCORE_ALL_MAX and name tokens did
    # not narrow it. Caller runs visual top-k inside this set only.
    visual_trim: bool = False
    # Two agreeing signals, or one exact rare name. Safe to keep a
    # one-wine shortlist. A single fuzzy grape is not strong.
    strong: bool = False
    # OCR hint only. Caller must union these slugs with visual top-K
    # instead of hiding the rest of the catalog.
    soft: bool = False
    # Only common grape names, no producer and no distinctive title.
    generic_only: bool = False
    winery_token: str = ""
    # Producer is known, and the grape on the label is not in that house.
    grape_conflict: bool = False


def _near_generic(token: str) -> bool:
    """One-character OCR slip of a filler word (КРАСНОС ~ КРАСНОЕ)."""

    if token in NAME_STOPWORDS or len(token) < 5:
        return False

    best = 0.0
    for stop in NAME_STOPWORDS:
        if abs(len(stop) - len(token)) > 1:
            continue
        score = ratio(token, stop) / 100.0
        if score > best:
            best = score
            if best >= 0.84:
                return True
    return False


def _resolve_grape_hit(
    token: str,
    known_grape_tokens: set[str],
) -> tuple[str, str] | None:
    """Return (catalog grape, quality). Quality is exact, alias, prefix, fuzzy."""

    if token in GRAPE_STOPWORDS:
        return None

    alias = LATIN_GRAPE_ALIASES.get(token) or GRAPE_ALIASES.get(token)
    if alias and alias in known_grape_tokens and alias not in GRAPE_STOPWORDS:
        return alias, "alias"

    if len(token) < GRAPE_EXACT_MIN_LEN:
        return None

    if _near_generic(token):
        return None

    if token in known_grape_tokens:
        return token, "exact"

    prefixed = _unique_prefix_grape(token, known_grape_tokens)
    if prefixed:
        return prefixed, "prefix"

    if len(token) < GRAPE_FUZZY_MIN_LEN or not known_grape_tokens:
        return None

    best_token: str | None = None
    best_score = 0.0
    second_score = 0.0

    for grape in known_grape_tokens:
        if grape in GRAPE_STOPWORDS or len(grape) < GRAPE_EXACT_MIN_LEN:
            continue
        if abs(len(grape) - len(token)) > 2:
            continue

        score = ratio(token, grape) / 100.0
        if score > best_score:
            second_score = best_score
            best_score = score
            best_token = grape
        elif score > second_score:
            second_score = score

    if best_token is None or best_score < GRAPE_FUZZY_MIN:
        return None

    if (
        second_score >= GRAPE_FUZZY_MIN
        and (best_score - second_score) < GRAPE_FUZZY_MARGIN
    ):
        return None

    return best_token, "fuzzy"


def _resolve_grape(
    token: str,
    known_grape_tokens: set[str],
) -> str | None:
    hit = _resolve_grape_hit(token, known_grape_tokens)
    if hit is None:
        return None
    return hit[0]


def _unique_prefix_grape(
    token: str,
    known_grape_tokens: set[str],
) -> str | None:
    """ЦИТРОН -> ЦИТРОННЫЙ when exactly one catalog grape extends it."""

    if len(token) < 6:
        return None

    hits: list[str] = []
    for grape in known_grape_tokens:
        if grape in GRAPE_STOPWORDS or len(grape) <= len(token):
            continue
        if len(grape) - len(token) > 4:
            continue
        if grape.startswith(token):
            hits.append(grape)
            if len(hits) > 1:
                return None

    if len(hits) == 1:
        return hits[0]

    return None


_CLEAN_GRAPE = frozenset({"exact", "alias"})
_QUALITY_RANK = {"exact": 3, "alias": 2, "prefix": 1, "fuzzy": 0}

# Fuzzy winery hits stay a hint. An exact winery may narrow the catalog.
WINERY_FUZZY_MIN = 0.85
WINERY_FUZZY_MARGIN = 0.04
WINERY_FUZZY_MAX_WINES = 40
NAME_INTERSECT_MAX_DF = 40


def _matched_grapes(
    ocr_tokens: list[str],
    known_grape_tokens: set[str],
) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    order: list[str] = []

    for token in ocr_tokens:
        hit = _resolve_grape_hit(token, known_grape_tokens)
        if hit is None:
            continue
        grape, quality = hit
        previous = found.get(grape)
        if previous is None:
            found[grape] = quality
            order.append(grape)
        elif _QUALITY_RANK[quality] > _QUALITY_RANK[previous]:
            found[grape] = quality

    return [(grape, found[grape]) for grape in order]


def _match_winery(
    ocr_tokens: list[str],
    winery_to_slugs: dict[str, set[str]] | None,
    known_grape_tokens: set[str],
) -> tuple[str, set[str], bool]:
    """Return (token, slugs, exact). Empty token when nothing matched."""

    if not winery_to_slugs:
        return "", set(), False

    exact: list[tuple[int, str, set[str]]] = []
    for token in ocr_tokens:
        if is_generic_token(token) or len(token) < 4:
            continue
        if token in known_grape_tokens:
            continue
        if token in LATIN_GRAPE_ALIASES or token in GRAPE_ALIASES:
            continue
        owners = winery_to_slugs.get(token)
        if owners:
            exact.append((len(owners), token, set(owners)))

    if exact:
        exact.sort()
        _, token, owners = exact[0]
        return token, owners, True

    best_token = ""
    best_owners: set[str] = set()
    best_score = 0.0
    second_score = 0.0

    for token in ocr_tokens:
        if is_generic_token(token) or len(token) < 5 or _near_generic(token):
            continue
        if token in known_grape_tokens:
            continue
        for winery, owners in winery_to_slugs.items():
            if not owners or len(owners) > WINERY_FUZZY_MAX_WINES:
                continue
            # Short names (AGORA) fuzzy-match fragments of a longer
            # brand (Fanagoria -> AGORIA). Keep this for long names only.
            if len(winery) < 7 or abs(len(winery) - len(token)) > 1:
                continue
            score = ratio(token, winery) / 100.0
            if score > best_score:
                second_score = best_score
                best_score = score
                best_token = winery
                best_owners = set(owners)
            elif score > second_score:
                second_score = score

    if (
        best_token
        and best_score >= WINERY_FUZZY_MIN
        and (best_score - second_score) >= WINERY_FUZZY_MARGIN
    ):
        return best_token, best_owners, False

    return "", set(), False


def is_generic_grape(token: str) -> bool:
    """A variety that matches dozens of wines. Not a producer and not a title."""

    if token in GENERIC_GRAPE_TOKENS:
        return True
    alias = LATIN_GRAPE_ALIASES.get(token) or GRAPE_ALIASES.get(token)
    return bool(alias and alias in GENERIC_GRAPE_TOKENS)


def extract_years(text: str) -> list[str]:
    """Four-digit vintages, including OCR 2о21 / 2О2I. Order preserved."""

    found: list[str] = []
    seen: set[str] = set()
    for token in normalize_text(text).split():
        folded = token.replace("О", "0").replace("I", "1")
        if not re.fullmatch(r"(19|20)\d{2}", folded):
            continue
        if folded in seen:
            continue
        seen.add(folded)
        found.append(folded)
    return found


def has_second_distinctive(
    grape_tokens: list[str],
    name_tokens: list[str],
    winery_token: str = "",
) -> bool:
    """True when a token other than the producer and a common grape matched.

    The reranker may reorder only then. A house name alone stays on
    the visual score.
    """

    if any(not is_generic_grape(token) for token in grape_tokens):
        return True
    for token in name_tokens:
        if token == winery_token or token in LINE_FILLERS:
            continue
        if is_generic_grape(token):
            continue
        return True
    return False


def order_candidates(
    candidates: list[dict],
    *,
    allow_rerank: bool,
    years: list[str] | None = None,
) -> list[dict]:
    """Visual order unless a second distinctive token matched.

    A single vintage breaks a tie (scores within 0.02) inside that order.
    It does not outrank a clearly closer label.
    """

    vintages = [year for year in (years or []) if year]

    def primary(item: dict) -> float:
        key = "final_score" if allow_rerank else "visual_score"
        return float(item.get(key) or 0.0)

    def year_hit(item: dict) -> int:
        if not vintages:
            return 0
        blob = f"{item.get('slug') or ''} {item.get('name') or ''}"
        return 1 if any(year in blob for year in vintages) else 0

    # Only a single vintage in the list may break a tie. Two 2020s
    # must not all jump ahead of a closer label.
    use_year = sum(year_hit(item) for item in candidates) == 1

    def band(score: float) -> float:
        # 0.02 is a tie. A vintage may win inside it and must not
        # jump a clearly closer label.
        return round(score / 0.02) * 0.02

    return sorted(
        candidates,
        key=lambda item: (
            band(primary(item)) if use_year else primary(item),
            year_hit(item) if use_year else 0,
            primary(item),
        ),
        reverse=True,
    )


def is_generic_token(token: str) -> bool:
    """Generic wine words and 4-digit numbers must not add score.

    2024 survives normalization as 2О24 (0 -> О, 1 -> I).
    """

    if token in NAME_STOPWORDS:
        return True

    folded = token.replace("О", "0").replace("I", "1")
    return bool(re.fullmatch(r"\d{4}", folded))


def _distinctive_name(token: str, owners: set[str], token_df: dict[str, int], catalog_count: int) -> bool:
    catalog_count = max(catalog_count, 1)
    df = token_df.get(token, len(owners))
    return df / catalog_count <= NAME_MAX_DF_RATIO


def _fuzzy_name(
    token: str,
    name_to_slugs: dict[str, set[str]],
    token_df: dict[str, int],
) -> str | None:
    """One-character slip of a rare title word. Cyrillic lexicon only."""

    if len(token) < 5 or _near_generic(token):
        return None

    best_name = ""
    best_score = 0.0
    second_score = 0.0
    for name, owners in name_to_slugs.items():
        if len(name) < 5 or abs(len(name) - len(token)) > 1:
            continue
        if token_df.get(name, len(owners)) > NAME_RARE_DF:
            continue
        score = ratio(token, name) / 100.0
        if score > best_score:
            second_score = best_score
            best_score = score
            best_name = name
        elif score > second_score:
            second_score = score
    if (
        best_name
        and best_score >= 0.86
        and (best_score - second_score) >= 0.05
    ):
        return best_name
    return None


def _fuzzy_winery(
    token: str,
    winery_to_slugs: dict[str, set[str]],
) -> str | None:
    if len(token) < 5 or _near_generic(token):
        return None

    best_token = ""
    best_score = 0.0
    second_score = 0.0
    for winery, owners in winery_to_slugs.items():
        if not owners or len(owners) > WINERY_FUZZY_MAX_WINES:
            continue
        if len(winery) < 7 or abs(len(winery) - len(token)) > 1:
            continue
        score = ratio(token, winery) / 100.0
        if score > best_score:
            second_score = best_score
            best_score = score
            best_token = winery
        elif score > second_score:
            second_score = score
    if (
        best_token
        and best_score >= WINERY_FUZZY_MIN
        and (best_score - second_score) >= WINERY_FUZZY_MARGIN
    ):
        return best_token
    return None


def lexicon_hit(
    token: str,
    known_grape_tokens: set[str],
    name_to_slugs: dict[str, set[str]],
    winery_to_slugs: dict[str, set[str]] | None,
    token_df: dict[str, int],
    catalog_count: int,
) -> str | None:
    """Catalog form of one OCR token, or None when it must be ignored.

    Mostly-Latin tokens are looked up only in the Latin lexicon
    (English names, brands, and grape aliases stored for the catalog).
    A miss is discarded. Mostly-Cyrillic tokens may use exact or careful
    fuzzy matches against Russian grapes, names, and wineries.
    """

    if not token or is_generic_token(token) or token.isdigit():
        return None

    wineries = winery_to_slugs or {}

    if mostly_latin(token):
        alias = LATIN_GRAPE_ALIASES.get(token)
        if alias and alias in known_grape_tokens and alias not in GRAPE_STOPWORDS:
            return alias
        winery_alias = LATIN_WINERY_ALIASES.get(token)
        if winery_alias and winery_alias in wineries:
            return winery_alias
        owners = name_to_slugs.get(token)
        if owners and _distinctive_name(token, owners, token_df, catalog_count):
            return token
        if token in wineries and len(token) >= 4:
            return token
        return None

    folded = fold_mixed_homoglyphs(token)
    if is_generic_token(folded):
        return None

    grape = _resolve_grape(folded, known_grape_tokens)
    if grape:
        return grape

    owners = name_to_slugs.get(folded)
    if (
        owners
        and len(folded) >= NAME_MIN_LEN
        and folded not in known_grape_tokens
        and _distinctive_name(folded, owners, token_df, catalog_count)
    ):
        return folded

    fuzzy_name = _fuzzy_name(folded, name_to_slugs, token_df)
    if fuzzy_name:
        return fuzzy_name

    if folded in wineries and len(folded) >= 4:
        return folded

    return _fuzzy_winery(folded, wineries)


def _token_owners(
    token: str,
    grape_to_slugs: dict[str, set[str]],
    name_to_slugs: dict[str, set[str]],
    winery_to_slugs: dict[str, set[str]],
) -> set[str]:
    return (
        set(grape_to_slugs.get(token, ()))
        | set(name_to_slugs.get(token, ()))
        | set(winery_to_slugs.get(token, ()))
    )


def merge_compatible_tokens(
    primary: list[str],
    secondary: list[str],
    grape_to_slugs: dict[str, set[str]],
    name_to_slugs: dict[str, set[str]],
    winery_to_slugs: dict[str, set[str]] | None = None,
) -> list[str]:
    """Union of two OCR passes.

    The Cyrillic pass is the anchor. An English token is kept when it
    shares a wine with that anchor (SAQRA inside Фанагория). A second
    producer read off Cyrillic shapes (CADET, TAMAGNE) is left out.
    """

    wineries = winery_to_slugs or {}
    if not primary:
        return list(secondary)

    groups = [
        _token_owners(token, grape_to_slugs, name_to_slugs, wineries)
        for token in primary
    ]
    groups = [group for group in groups if group]
    if not groups:
        anchor: set[str] = set()
    elif len(groups) == 1:
        anchor = groups[0]
    else:
        anchor = set.intersection(*groups) or set.union(*groups)

    merged = list(primary)
    seen = set(merged)
    for token in secondary:
        if token in seen:
            continue
        owned = _token_owners(token, grape_to_slugs, name_to_slugs, wineries)
        if anchor and not (owned & anchor):
            continue
        seen.add(token)
        merged.append(token)
    return merged


def catalog_tokens(
    ocr_text: str,
    known_grape_tokens: set[str],
    name_to_slugs: dict[str, set[str]],
    token_df: dict[str, int],
    catalog_count: int,
    winery_to_slugs: dict[str, set[str]] | None = None,
) -> list[str]:
    """Normalized catalog hits. Raw OCR tokens that miss the lexicon are dropped."""

    from ocr_lexicon import OcrLexicon

    lexicon = OcrLexicon.build(
        grapes=known_grape_tokens,
        names=name_to_slugs.keys(),
        wineries=(winery_to_slugs or {}).keys(),
    )
    return lexicon.clean(ocr_text).tokens


def text_visual_weights(distinctive_hits: int) -> tuple[float, float]:
    """Return (text_weight, visual_weight) inside an OCR shortlist."""

    if distinctive_hits >= TEXT_DOMINANT_MIN_HITS:
        return TEXT_DOMINANT_TEXT, TEXT_DOMINANT_VISUAL

    return 0.35, 0.65


def _name_tokens(
    ocr_tokens: list[str],
    known_grape_tokens: set[str],
    name_to_slugs: dict[str, set[str]],
    token_df: dict[str, int],
    catalog_count: int,
) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    catalog_count = max(catalog_count, 1)

    for token in ocr_tokens:
        if token in seen:
            continue
        if len(token) < NAME_MIN_LEN or token.isdigit():
            continue
        if is_generic_token(token) or token in LINE_FILLERS:
            continue
        if token in known_grape_tokens:
            continue
        if token in LATIN_GRAPE_ALIASES or token in GRAPE_ALIASES:
            continue
        if is_generic_grape(token):
            continue
        if _unique_prefix_grape(token, known_grape_tokens):
            continue

        owners = name_to_slugs.get(token)
        if not owners:
            continue

        df = token_df.get(token, len(owners))
        if df / catalog_count > NAME_MAX_DF_RATIO:
            continue

        seen.add(token)
        found.append(token)

    # One-character slips of a rare title word (ЗОАОТО -> ЗОЛОТО).
    for token in ocr_tokens:
        if token in seen or len(token) < 5 or is_generic_token(token):
            continue
        if token in known_grape_tokens:
            continue
        best_name = ""
        best_score = 0.0
        second_score = 0.0
        for name, owners in name_to_slugs.items():
            if len(name) < 5 or abs(len(name) - len(token)) > 1:
                continue
            if token_df.get(name, len(owners)) > NAME_RARE_DF:
                continue
            score = ratio(token, name) / 100.0
            if score > best_score:
                second_score = best_score
                best_score = score
                best_name = name
            elif score > second_score:
                second_score = score
        if (
            best_name
            and best_name not in seen
            and best_score >= 0.86
            and (best_score - second_score) >= 0.05
        ):
            seen.add(best_name)
            found.append(best_name)

    return found


def _name_scores(
    slugs: set[str],
    name_tokens: list[str],
    name_to_slugs: dict[str, set[str]],
    token_df: dict[str, int],
    catalog_count: int,
) -> dict[str, float]:
    scores = {slug: 0.0 for slug in slugs}

    for token in name_tokens:
        weight = _idf(token, token_df, catalog_count)
        for slug in name_to_slugs.get(token, ()):
            if slug in scores:
                scores[slug] += weight

    return scores


def _order_slugs(
    slugs: set[str],
    scores: dict[str, float],
) -> list[str]:
    return sorted(slugs, key=lambda slug: (-scores.get(slug, 0.0), slug))


def _trim_large_set(
    slugs: set[str],
    name_tokens: list[str],
    name_to_slugs: dict[str, set[str]],
    token_df: dict[str, int],
    catalog_count: int,
) -> tuple[list[str], bool]:
    scores = _name_scores(
        slugs,
        name_tokens,
        name_to_slugs,
        token_df,
        catalog_count,
    )
    ordered = _order_slugs(slugs, scores)

    if len(ordered) <= SCORE_ALL_MAX:
        return ordered, False

    named = [slug for slug in ordered if scores.get(slug, 0.0) > 0]
    if named:
        return named[:SCORE_ALL_MAX], False

    return ordered, True


def _slugs_for_grapes(
    grapes: list[str],
    grape_to_slugs: dict[str, set[str]],
) -> tuple[set[str], list[str]]:
    """Wines that contain every detected grape, else a greedy intersection.

    A blend that only has the first grape is dropped when at least one
    catalog wine contains all of them. An empty full intersection falls
    back to the rarest-first greedy pass.
    """

    present = [
        grape
        for grape in grapes
        if grape_to_slugs.get(grape)
    ]

    if present:
        full = set(grape_to_slugs[present[0]])
        for grape in present[1:]:
            full &= grape_to_slugs[grape]
            if not full:
                break
        if full:
            return full, present

    slugs: set[str] = set()
    used: list[str] = []

    for grape in grapes:
        owners = set(grape_to_slugs.get(grape, ()))
        if not owners:
            continue
        if not slugs:
            slugs = owners
            used.append(grape)
            continue
        overlapped = slugs & owners
        if overlapped:
            slugs = overlapped
            used.append(grape)

    return slugs, used


def _is_house_line(
    token: str,
    name_to_slugs: dict[str, set[str]],
    winery_to_slugs: dict[str, set[str]] | None,
) -> bool:
    """True when every wine with this name token sits in one winery.

    ALVEUS is only Fanagoria, so it hard-filters. КЮВЕ is printed by
    many houses, so it does not.
    """

    if token in LINE_FILLERS or is_generic_token(token) or not winery_to_slugs:
        return False
    owners = set(name_to_slugs.get(token, ()))
    if not owners or len(owners) > 80:
        return False
    return any(
        owners <= set(slugs)
        for winery, slugs in winery_to_slugs.items()
        if len(winery) >= 5 and not is_generic_token(winery)
    )


def recover_names_inside_producer(
    kept: list[str],
    dropped_raw: list[str],
    *,
    name_to_slugs: dict[str, set[str]],
    winery_to_slugs: dict[str, set[str]] | None,
    token_df: dict[str, int],
) -> list[str]:
    """Fuzzy-match leftover OCR against one producer's own name tokens.

    The dictionary is the distinctive titles of that house, so a
    mangled ЛЕКИФ can snap without matching the whole catalog.
    """

    if not kept or not dropped_raw or not winery_to_slugs:
        return list(kept)

    producer = ""
    producer_slugs: set[str] = set()
    for token in kept:
        owners = winery_to_slugs.get(token)
        if owners and (not producer_slugs or len(owners) < len(producer_slugs)):
            producer = token
            producer_slugs = set(owners)
    if not producer_slugs:
        return list(kept)

    titles = [
        name
        for name, owners in name_to_slugs.items()
        if len(name) >= 4
        and name not in kept
        and name not in LINE_FILLERS
        and not is_generic_token(name)
        and set(owners) & producer_slugs
        and token_df.get(name, len(owners)) <= 40
    ]
    if not titles:
        return list(kept)

    found = list(kept)
    seen = set(found)
    for raw in dropped_raw:
        forms = []
        for token in text_tokens(raw, fold=False):
            forms.append(token)
            repaired = token.translate(_LATIN_TO_CYRILLIC).replace("J", "Л").replace("3", "З")
            if repaired not in forms:
                forms.append(repaired)
        best_name = ""
        best_score = 0.0
        second_score = 0.0
        for form in forms:
            if len(form) < 4 or is_generic_token(form) or _near_generic(form):
                continue
            for name in titles:
                if abs(len(name) - len(form)) > 2:
                    continue
                score = ratio(form, name) / 100.0
                if score > best_score:
                    second_score = best_score
                    best_score = score
                    best_name = name
                elif score > second_score:
                    second_score = score
        if (
            best_name
            and best_name not in seen
            and best_score >= 0.75
            and (best_score - second_score) >= 0.05
        ):
            seen.add(best_name)
            found.append(best_name)
    return found


def _unique_producer_title(
    names: list[str],
    winery_token: str,
    winery_slugs: set[str],
    name_to_slugs: dict[str, set[str]],
    token_df: dict[str, int],
    catalog_count: int,
    *,
    exact: bool,
) -> set[str]:
    """One bottle named by the producer plus a rare title word.

    A stray token that is not in this house does not erase the hit.
    """

    if not exact or not winery_slugs:
        return set()
    rare = [
        token
        for token in names
        if token != winery_token
        and token_df.get(token, catalog_count) <= NAME_RARE_DF
    ]
    if not rare:
        return set()
    hit = set(winery_slugs)
    matched = False
    for token in rare:
        nxt = hit & set(name_to_slugs.get(token, ()))
        if not nxt:
            continue
        hit = nxt
        matched = True
    if matched and len(hit) == 1:
        return hit
    return set()


def _weak(reason: str) -> OcrShortlist:
    return OcrShortlist(
        confident=False,
        slugs=[],
        reason=reason,
    )


def build_ocr_shortlist(
    ocr_tokens: list[str],
    known_grape_tokens: set[str],
    grape_to_slugs: dict[str, set[str]],
    name_to_slugs: dict[str, set[str]],
    token_df: dict[str, int],
    catalog_count: int,
    winery_to_slugs: dict[str, set[str]] | None = None,
) -> OcrShortlist:
    """Return an OCR-constrained slug list, or a weak/empty decision."""

    if not ocr_tokens:
        return _weak("empty")

    extra_grapes = [
        token
        for token in recovered_pinot_noir(ocr_tokens)
        if token in known_grape_tokens and token not in ocr_tokens
    ]
    if extra_grapes:
        ocr_tokens = list(ocr_tokens) + extra_grapes

    catalog_count = max(catalog_count, 1)
    hits = _matched_grapes(ocr_tokens, known_grape_tokens)
    hits.sort(
        key=lambda item: (
            -_idf(item[0], token_df, catalog_count),
            token_df.get(item[0], 0),
            item[0],
        )
    )
    clean = [grape for grape, quality in hits if quality in _CLEAN_GRAPE]
    fuzzy = [grape for grape, quality in hits if quality not in _CLEAN_GRAPE]
    grapes = clean or fuzzy
    # Prefix/fuzzy varieties are a hint. They must not hide the catalog.
    soft_grapes = not clean and bool(fuzzy)

    names = _name_tokens(
        ocr_tokens,
        known_grape_tokens,
        name_to_slugs,
        token_df,
        catalog_count,
    )
    winery_token, winery_slugs, winery_exact = _match_winery(
        ocr_tokens,
        winery_to_slugs,
        known_grape_tokens,
    )

    if grapes:
        slugs, used = _slugs_for_grapes(grapes, grape_to_slugs)

        if not slugs:
            return _weak("grape_without_wines")

        # One grape, or only common grapes, that share no wine with a
        # rare title must not hide that title (НУАР vs БЕЛАЯ, ПИНО vs
        # БЕЛЕНЬКОЕ).
        generic_grapes = bool(used) and all(is_generic_grape(grape) for grape in used)
        if names and (len(used) == 1 or generic_grapes):
            rare_title = [
                token
                for token in names
                if token_df.get(token, catalog_count) <= NAME_RARE_DF
            ]
            title_slugs: set[str] = set()
            for token in rare_title:
                title_slugs |= set(name_to_slugs.get(token, ()))
            if rare_title and not (slugs & title_slugs):
                grapes = []

        # Producer plus a rare title already names one bottle
        # (АРАТТИ + БЕЛАЯ). A grape this bottle does not carry must
        # not veto it. grape_conflict stays for producer+grape alone.
        unique_title = _unique_producer_title(
            names,
            winery_token,
            set(winery_slugs),
            name_to_slugs,
            token_df,
            catalog_count,
            exact=winery_exact,
        )
        if unique_title and not (set(slugs) & unique_title):
            grapes = []

        if grapes:
            strong = bool(clean) and len(used) >= 2
            grape_slugs = set(slugs)
            # Title words shrink the grape set. The producer token goes
            # first, so a later misread (TAMAGNE) cannot widen it.
            # Golubitskoe Estate Chardonnay is filed under another
            # winery string; the name hit is the bottle, and a small
            # title set is kept when it does not sit in the house list.
            name_hit: set[str] | None = None
            if clean and names:
                rare_names = [
                    token
                    for token in names
                    if token_df.get(token, catalog_count) <= NAME_INTERSECT_MAX_DF
                ]
                ordered_names: list[str] = []
                if winery_token in rare_names:
                    ordered_names.append(winery_token)
                ordered_names.extend(
                    token for token in rare_names if token not in ordered_names
                )
                narrowed = set(grape_slugs)
                did_narrow = False
                for token in ordered_names:
                    hit = narrowed & set(name_to_slugs.get(token, ()))
                    if hit:
                        narrowed = hit
                        did_narrow = True
                if did_narrow:
                    name_hit = narrowed
            if winery_exact and winery_slugs:
                house = set(winery_slugs)
                both = grape_slugs & house
                if used and not both:
                    # The label names a grape this producer does not bottle.
                    # A rare title that already selected one bottle wins.
                    # Otherwise do not fall back to the visual leader.
                    if unique_title:
                        return OcrShortlist(
                            confident=True,
                            slugs=sorted(unique_title),
                            reason="name:" + "+".join(names),
                            grape_tokens=[],
                            name_tokens=names,
                            strong=True,
                            soft=False,
                            winery_token=winery_token,
                        )
                    return OcrShortlist(
                        confident=False,
                        slugs=[],
                        reason="conflict:" + winery_token + "+" + "+".join(used),
                        grape_tokens=used,
                        name_tokens=names,
                        winery_token=winery_token,
                        grape_conflict=True,
                    )
                if name_hit is not None:
                    together = name_hit & house
                    if together:
                        slugs = together
                    elif len(name_hit) <= 3 or not both:
                        slugs = name_hit
                    else:
                        slugs = both
                elif both:
                    slugs = both
                else:
                    slugs = house
                strong = True
            elif name_hit is not None:
                slugs = name_hit
                strong = True

            generic_only = (
                not winery_exact
                and not names
                and bool(used)
                and all(is_generic_grape(grape) for grape in used)
            )
            if generic_only:
                strong = False

            ordered, visual_trim = _trim_large_set(
                slugs,
                names,
                name_to_slugs,
                token_df,
                catalog_count,
            )
            # A common grape must not leave a one-wine shortlist.
            if generic_only and len(ordered) == 1 and len(slugs) > 1:
                ordered, visual_trim = _trim_large_set(
                    slugs,
                    [],
                    name_to_slugs,
                    token_df,
                    catalog_count,
                )
            reason = "grape:" + "+".join(used)
            if winery_token and (strong or winery_exact):
                reason += "|winery:" + winery_token
            if visual_trim:
                reason += "|visual_trim"
            locked = strong and not soft_grapes and not generic_only
            return OcrShortlist(
                confident=True,
                slugs=ordered,
                reason=reason,
                grape_tokens=used,
                name_tokens=names,
                visual_trim=visual_trim,
                strong=locked,
                # One grape, or a fuzzy grape, is a hint. Two grapes, or a
                # grape plus a rare name, may lock the comparison set.
                soft=not locked,
                generic_only=generic_only,
                winery_token=winery_token,
            )

    if winery_slugs:
        strong = False
        if names:
            # The winery word itself is not a second hit.
            rare_names = [
                token
                for token in names
                if token != winery_token
                and token_df.get(token, catalog_count) <= NAME_INTERSECT_MAX_DF
            ]
            name_slugs: set[str] = set()
            for token in rare_names:
                name_slugs |= set(name_to_slugs.get(token, ()))
            overlapped = winery_slugs & name_slugs
            if overlapped:
                winery_slugs = overlapped
                strong = True
        ordered = sorted(winery_slugs)
        visual_trim = len(ordered) > SCORE_ALL_MAX
        if visual_trim:
            ordered = ordered[:SCORE_ALL_MAX]
        # A recovered producer restricts the catalog. A fuzzy winery
        # stays a hint. Name overlap may shrink the set further.
        locked = bool(winery_exact)
        return OcrShortlist(
            confident=True,
            slugs=ordered,
            reason="winery:" + winery_token,
            grape_tokens=[],
            name_tokens=names,
            visual_trim=visual_trim,
            strong=locked,
            soft=not locked,
            winery_token=winery_token,
        )

    if not names:
        return _weak("weak")

    hit_tokens: dict[str, list[str]] = {}
    scores: dict[str, float] = {}

    for token in names:
        weight = _idf(token, token_df, catalog_count)
        rare_single = (
            len(token) >= NAME_SINGLE_MIN_LEN
            and token_df.get(token, NAME_RARE_DF + 1) <= NAME_RARE_DF
        )
        for slug in name_to_slugs.get(token, ()):
            hit_tokens.setdefault(slug, []).append(token)
            scores[slug] = scores.get(slug, 0.0) + weight
            if rare_single:
                scores.setdefault(slug, 0.0)

    qualified = [
        slug
        for slug, tokens in hit_tokens.items()
        if len(tokens) >= 2
        or (
            len(tokens) == 1
            and len(tokens[0]) >= NAME_SINGLE_MIN_LEN
            and token_df.get(tokens[0], NAME_RARE_DF + 1) <= NAME_RARE_DF
        )
    ]

    house_tokens = [
        token
        for token in names
        if _is_house_line(token, name_to_slugs, winery_to_slugs)
    ]
    if house_tokens:
        house_slugs: set[str] | None = None
        for token in house_tokens:
            owners = set(name_to_slugs.get(token, ()))
            house_slugs = owners if house_slugs is None else house_slugs & owners
        if house_slugs:
            narrowed = [slug for slug in qualified if slug in house_slugs]
            qualified = narrowed or sorted(house_slugs)

    if not qualified:
        return _weak("weak")

    ordered = sorted(
        qualified,
        key=lambda slug: (-scores.get(slug, 0.0), slug),
    )[:SCORE_ALL_MAX]

    rare_exact = (
        len(names) == 1
        and not is_generic_grape(names[0])
        and token_df.get(names[0], NAME_RARE_DF + 1) <= NAME_RARE_DF
        and 1 <= len(qualified) <= 3
    )
    # Two name tokens, one exact rare title, or a line that belongs
    # to a single house (ALVEUS) may lock. КЮВЕ spans houses, so it
    # stays a hint.
    locked = (
        any(len(hit_tokens.get(slug, ())) >= 2 for slug in qualified)
        or rare_exact
        or bool(house_tokens and qualified)
    )
    return OcrShortlist(
        confident=True,
        slugs=ordered,
        reason="name:" + "+".join(names),
        grape_tokens=[],
        name_tokens=names,
        visual_trim=False,
        strong=locked,
        soft=not locked,
    )
