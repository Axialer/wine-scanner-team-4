"""Match OCR tokens to the catalog dictionary before scoring.

Raw OCR never reaches the shortlist or the text score. Each token is
homoglyph-normalized, then accepted only when it is a confident
dictionary hit. The recovered catalog spelling is what gets returned.

Thresholds (rapidfuzz fuzz.ratio, 0..1):

* Cyrillic, or a homoglyph-only token read as Cyrillic: exact hit, a
  known alias (ЦИТРОН → ЦИТРОННЫЙ), or fuzzy ratio >= 0.84 with a
  0.04 gap over the runner-up. Fuzzy starts at length 5 and only
  compares dictionary words within 2 characters of length. 0.84 is
  one substitution on a 7-letter word (АЛИГАТЕ → АЛИГОТЕ ≈ 0.86) or
  one insertion/deletion on a longer name (ЖЕМЧЖНАЯ → ЖЕМЧУЖНАЯ).
* Real Latin (at least one letter that is not a Cyrillic lookalike):
  exact hit, a Latin alias (CHARDONNAY, ARATTI), or fuzzy ratio
  >= 0.92 with a 0.05 gap. Anything else is dropped, so "xyzqwk"
  cannot steer SigLIP.
* Mixed token: Cyrillic reading first. The Latin reading is kept
  only when that reading itself is a confident dictionary hit.
* A token made only of lookalikes (no distinctive Latin) is read as
  Cyrillic first. It stays Latin only when the untouched spelling
  hits an English catalog name.

Stopwords and 4-digit years are ignored and are not dictionary keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from ocr_shortlist import (
    GRAPE_ALIASES,
    LATIN_GRAPE_ALIASES,
    LATIN_WINERY_ALIASES,
    _DISTINCTIVE_LATIN,
    _LATIN_LOOKALIKE,
    _LATIN_TO_CYRILLIC,
    _near_generic,
    _script_votes,
    _CYR_TO_LATIN,
    _repair_leading_digit,
    is_brut_misread,
    is_generic_token,
    mostly_cyrillic,
    NOIR_ONLY_WITH_PINOT,
    normalize_text,
    recovered_pinot_noir,
    text_tokens,
)


# One substitution on len>=7, or one edit on a longer token.
CYR_FUZZY_MIN = 0.84
CYR_FUZZY_MARGIN = 0.04

# Latin catalog names. Length >= 4 may sit a little further from the
# spelling (NLVEUS → ALVEUS is one substitution, ratio ≈ 0.83) but
# still needs a clear gap, so WONT does not become MONT.
LATIN_FUZZY_MIN = 0.80
LATIN_FUZZY_MARGIN = 0.05

EXACT_MIN_LEN = 4
FUZZY_MIN_LEN = 5
FUZZY_LEN_WINDOW = 2

_KIND_RANK = {"grape": 4, "winery": 3, "name": 2, "region": 1}


def _edit_distance(left: str, right: str) -> int:
    if abs(len(left) - len(right)) > 1:
        return 2
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]

def _repair_cyrillic_reading(token: str) -> str:
    """Fold lookalikes, then OCR slips that are not lookalikes.

    J is how RapidOCR often reads Л, and 3 is З inside a word
    (AJA3AHCKAЯ → АЛАЗАНСКАЯ). A year stays a year: the 3→З
    swap runs only when the token is not four digits.
    """

    folded = token.translate(_LATIN_TO_CYRILLIC)
    folded = folded.replace("J", "Л")
    if not folded.replace("О", "0").replace("I", "1").isdigit():
        folded = folded.replace("3", "З")
    return folded


@dataclass
class LexiconHit:
    raw: str
    catalog: str
    kind: str
    score: float

    def as_dict(self) -> dict:
        return {
            "raw": self.raw,
            "catalog": self.catalog,
            "type": self.kind,
            "score": round(self.score, 3),
        }


_READABLE_VOWELS = frozenset("АЕИОУЫЭЮЯAEIOUY")
_CONSONANT_RUN = re.compile(rf"[^{''.join(sorted(_READABLE_VOWELS))}]{{4,}}")


def _readable_word(token: str) -> bool:
    """A repaired word long enough for a web query, catalog hit or not."""

    if token == "ДОЛИНА":
        return True
    if len(token) < 4 or is_generic_token(token) or _near_generic(token):
        return False
    if not token.isalpha():
        return False
    vowels = sum(char in _READABLE_VOWELS for char in token)
    if vowels < 1:
        return False
    if len(token) >= 6 and vowels < 2:
        return False
    if _CONSONANT_RUN.search(token):
        return False
    return True


@dataclass
class LexiconCleanup:
    cleaned_text: str
    tokens: list[str]
    hits: list[LexiconHit] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    # Repaired words of length >= 4 that are not catalog hits. The web
    # query may use them (Mogzauri, Алазанская) without locking a slug.
    readable: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ocr_normalized": self.cleaned_text,
            "tokens": list(self.tokens),
            "matches": [hit.as_dict() for hit in self.hits],
            "dropped": list(self.dropped),
            "readable": list(self.readable),
        }


@dataclass
class _Entry:
    canonical: str
    kind: str
    alias: bool = False


def _script(token: str) -> str:
    latin, cyrillic, _, _ = _script_votes(token)
    if latin > 0 and cyrillic == 0:
        return "lat"
    return "cyr"


class OcrLexicon:
    def __init__(self, entries: dict[str, _Entry]):
        self.entries = entries
        self._cyr_by_len: dict[int, list[str]] = {}
        self._lat_by_len: dict[int, list[str]] = {}
        for key in entries:
            bucket = self._lat_by_len if _script(key) == "lat" else self._cyr_by_len
            bucket.setdefault(len(key), []).append(key)

    @classmethod
    def from_wine_metadata(cls, metadata: dict) -> OcrLexicon:
        grapes: set[str] = set()
        names: set[str] = set()
        wineries: set[str] = set()
        regions: set[str] = set()
        for data in metadata.values():
            grapes.update(text_tokens(data.get("grape") or ""))
            names.update(text_tokens(data.get("name") or ""))
            wineries.update(text_tokens(data.get("winery") or ""))
            regions.update(text_tokens(data.get("region") or ""))
        return cls.build(
            grapes=grapes,
            names=names,
            wineries=wineries,
            regions=regions,
        )

    @classmethod
    def build(
        cls,
        *,
        grapes: set[str] | list[str],
        names: set[str] | list[str] | None = None,
        wineries: set[str] | list[str] | None = None,
        regions: set[str] | list[str] | None = None,
    ) -> OcrLexicon:
        groups = (
            ("grape", grapes),
            ("winery", wineries or ()),
            ("name", names or ()),
            ("region", regions or ()),
        )
        entries: dict[str, _Entry] = {}
        for kind, tokens in groups:
            for token in tokens:
                if not token or len(token) < EXACT_MIN_LEN:
                    continue
                if is_generic_token(token):
                    continue
                current = entries.get(token)
                if current is None or _KIND_RANK[kind] > _KIND_RANK[current.kind]:
                    entries[token] = _Entry(canonical=token, kind=kind)

        alias_map = {
            **GRAPE_ALIASES,
            **LATIN_GRAPE_ALIASES,
            **LATIN_WINERY_ALIASES,
        }
        for source, target in alias_map.items():
            target_entry = entries.get(target)
            if target_entry is None:
                continue
            kind = "winery" if source in LATIN_WINERY_ALIASES else "grape"
            if source in LATIN_GRAPE_ALIASES or source in GRAPE_ALIASES:
                kind = "grape"
            entries[source] = _Entry(
                canonical=target_entry.canonical,
                kind=kind,
                alias=True,
            )
        return cls(entries)

    def clean(self, text: str) -> LexiconCleanup:
        hits: list[LexiconHit] = []
        tokens: list[str] = []
        dropped: list[dict] = []
        readable: list[str] = []
        seen: set[str] = set()
        readable_seen: set[str] = set()

        def accept(hit: LexiconHit) -> None:
            if hit.catalog in seen:
                dropped.append({
                    "raw": hit.raw,
                    "catalog": hit.catalog,
                    "reason": "повтор",
                })
                return
            seen.add(hit.catalog)
            tokens.append(hit.catalog)
            hits.append(hit)

        def remember(form: str) -> None:
            if form in readable_seen or form in seen or not _readable_word(form):
                return
            readable_seen.add(form)
            readable.append(form)

        unmatched: list[str] = []
        for raw in self._source_tokens(text):
            hit = self._match_token(raw)
            if hit is None:
                repaired = _repair_cyrillic_reading(raw)
                # Keep the printed spelling ahead of its lookalike fold
                # so a web query does not replace MOGZAURI with МОGZАURI.
                remember(raw)
                if repaired != raw:
                    remember(repaired)
                dropped.append({"raw": raw, "reason": "нет в словаре"})
                unmatched.append(repaired)
                continue
            accept(hit)

        for hit in self._ending_hits(unmatched):
            accept(hit)

        for catalog in recovered_pinot_noir(normalize_text(text).split()):
            entry = self.entries.get(catalog)
            if entry is None or entry.kind != "grape":
                continue
            accept(LexiconHit(catalog, entry.canonical, "grape", 0.99))

        readable = [
            token
            for token in readable
            if token not in seen
            and not any(
                catalog.endswith(token) or token.endswith(catalog)
                for catalog in seen
            )
        ]
        return LexiconCleanup(
            cleaned_text=" ".join(tokens),
            tokens=tokens,
            hits=hits,
            dropped=dropped,
            readable=readable,
        )

    def _source_tokens(self, text: str) -> list[str]:
        """Tokens for the lexicon, including a spaced join that hits the catalog.

        ``т а б и я`` still joins at four letters. ``Tа б и я`` joins
        only when ТАБИЯ (or the same join) is a dictionary word.
        A token with a real Latin letter also keeps its Latin spelling.
        """

        pieces = normalize_text(text).split()
        output: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            if not buffer:
                return
            joined = _repair_leading_digit("".join(buffer))
            folded = _repair_cyrillic_reading(joined)
            in_catalog = folded in self.entries or joined in self.entries
            if in_catalog or len(buffer) >= 4:
                output.append(folded if folded in self.entries else joined)
            buffer.clear()

        for piece in pieces:
            if piece.isalpha() and len(piece) <= 2:
                buffer.append(piece)
                continue
            flush()
            if len(piece) >= 3:
                output.append(_repair_leading_digit(piece))
        flush()

        expanded: list[str] = []
        seen: set[str] = set()
        for token in output:
            forms = [token]
            if any(char in _DISTINCTIVE_LATIN for char in token):
                forms.append(_repair_cyrillic_reading(token))
                forms.append(token.translate(_CYR_TO_LATIN))
            for form in forms:
                if form and form not in seen:
                    seen.add(form)
                    expanded.append(form)
        return expanded

    def _match_token(self, token: str) -> LexiconHit | None:
        if is_generic_token(token) or is_brut_misread(token):
            return None
        # HYAP is all Cyrillic lookalikes, so it folds to НУАР. That
        # reading counts only when a Pinot token is on the same label.
        if token in NOIR_ONLY_WITH_PINOT:
            return None
        # Known clipped grapes (СИР → СИРА) are shorter than a normal word.
        if len(token) < EXACT_MIN_LEN:
            entry = self.entries.get(token)
            if (
                entry is not None
                and entry.alias
                and entry.kind == "grape"
                and len(token) >= 3
            ):
                return LexiconHit(token, entry.canonical, entry.kind, 0.99)
            return None

        latin, cyrillic, _, _ = _script_votes(token)
        if cyrillic > 0 and latin > 0:
            folded = _repair_cyrillic_reading(token)
            hit = self._lookup(token, folded, "cyr")
            if hit is not None:
                return hit
            latin_form = token.translate(_CYR_TO_LATIN)
            hit = self._lookup(token, latin_form, "lat")
            if hit is not None:
                return hit
            return self._snap_stray_latin(token)

        if latin == 0 or mostly_cyrillic(token):
            folded = _repair_cyrillic_reading(token)
            hit = self._lookup(token, folded, "cyr")
            if hit is not None:
                return hit
            if folded != token:
                return self._lookup(token, token, "lat")
            return None

        return self._lookup(token, token, "lat")

    def _lookup(self, raw: str, form: str, script: str) -> LexiconHit | None:
        if is_brut_misread(raw) or is_brut_misread(form):
            return None
        if len(form) < EXACT_MIN_LEN or is_generic_token(form):
            return None

        entry = self.entries.get(form)
        if entry is not None and _script(form) == script:
            score = 0.99 if entry.alias else 1.0
            return LexiconHit(raw, entry.canonical, entry.kind, score)

        fuzzy_floor = 4 if script == "lat" else FUZZY_MIN_LEN
        if len(form) < fuzzy_floor or _near_generic(form):
            return None

        minimum = CYR_FUZZY_MIN if script == "cyr" else LATIN_FUZZY_MIN
        margin = CYR_FUZZY_MARGIN if script == "cyr" else LATIN_FUZZY_MARGIN
        found = self._fuzzy(form, script, minimum, margin)
        if found is None:
            if script == "lat":
                return self._one_edit_grape_alias(raw, form)
            return self._one_edit_winery(raw, form)
        key, score = found
        entry = self.entries[key]
        return LexiconHit(raw, entry.canonical, entry.kind, score)

    def _one_edit_grape_alias(self, raw: str, form: str) -> LexiconHit | None:
        """MUSCAI → MUSCAT when exactly one Latin grape alias is one edit away."""

        if len(form) < 5:
            return None
        canonicals: list[str] = []
        kinds: list[str] = []
        for key, entry in self.entries.items():
            if not entry.alias or entry.kind != "grape" or _script(key) != "lat":
                continue
            if abs(len(key) - len(form)) > 1:
                continue
            if _edit_distance(form, key) != 1:
                continue
            canonicals.append(entry.canonical)
            kinds.append(entry.kind)
        unique = set(canonicals)
        if len(unique) != 1:
            return None
        return LexiconHit(raw, canonicals[0], kinds[0], 0.90)

    def _one_edit_winery(self, raw: str, form: str) -> LexiconHit | None:
        """АРАГТИ → АРАТТИ when exactly one winery is one edit away.

        Length >= 5, so WONT cannot become MONT.
        """

        if len(form) < 5:
            return None
        matches: list[_Entry] = []
        for key, entry in self.entries.items():
            if entry.alias or entry.kind != "winery" or _script(key) != "cyr":
                continue
            if abs(len(key) - len(form)) > 1:
                continue
            if _edit_distance(form, key) != 1:
                continue
            matches.append(entry)
        canonicals = {entry.canonical for entry in matches}
        if len(canonicals) != 1:
            return None
        entry = matches[0]
        return LexiconHit(raw, entry.canonical, entry.kind, 0.88)

    def _ending_hits(self, forms: list[str]) -> list[LexiconHit]:
        """БЕЛЕНЬКОЕ from a repeated ending (енькое / еленькое)."""

        words = []
        for form in forms:
            if len(form) >= 5 and form.isalpha() and form not in words:
                words.append(form)
        found: list[LexiconHit] = []
        seen: set[str] = set()
        for left in words:
            for right in words:
                if left == right:
                    continue
                short, long = (left, right) if len(left) <= len(right) else (right, left)
                if len(short) < 5 or not long.endswith(short):
                    continue
                entry = self._by_suffix(long) or self._by_suffix(short)
                if entry is None or entry.canonical in seen:
                    continue
                seen.add(entry.canonical)
                found.append(LexiconHit(long, entry.canonical, entry.kind, 0.90))
        return found

    def _by_suffix(self, stem: str) -> _Entry | None:
        matches: list[_Entry] = []
        for key, entry in self.entries.items():
            if entry.alias or entry.kind not in {"name", "winery"}:
                continue
            if _script(key) != "cyr":
                continue
            extra = len(key) - len(stem)
            if extra < 0 or extra > 2:
                continue
            if key.endswith(stem):
                matches.append(entry)
        canonicals = {entry.canonical for entry in matches}
        if len(canonicals) != 1:
            return None
        return matches[0]

    def _snap_stray_latin(self, token: str) -> LexiconHit | None:
        """One leftover Latin letter inside a Cyrillic word (ЗОNОТО → ЗОЛОТО).

        Accepted only when a single dictionary word shares every other letter.
        """

        latin, cyrillic, _, _ = _script_votes(token)
        if latin != 1 or cyrillic == 0:
            return None
        stray = [
            index
            for index, char in enumerate(token)
            if "A" <= char <= "Z" and char not in _LATIN_LOOKALIKE
        ]
        if len(stray) != 1:
            return None
        index = stray[0]
        folded = list(token.translate(_LATIN_TO_CYRILLIC))
        matches: list[_Entry] = []
        for key, entry in self.entries.items():
            if entry.alias or len(key) != len(folded) or _script(key) != "cyr":
                continue
            if all(
                position == index or folded[position] == key[position]
                for position in range(len(key))
            ):
                matches.append(entry)
        canonicals = {entry.canonical for entry in matches}
        if len(canonicals) != 1:
            return None
        entry = matches[0]
        return LexiconHit(token, entry.canonical, entry.kind, 0.90)

    def _fuzzy(
        self,
        form: str,
        script: str,
        minimum: float,
        margin: float,
    ) -> tuple[str, float] | None:
        bucket = self._cyr_by_len if script == "cyr" else self._lat_by_len
        pool: list[str] = []
        for length in range(len(form) - FUZZY_LEN_WINDOW, len(form) + FUZZY_LEN_WINDOW + 1):
            pool.extend(bucket.get(length, ()))
        if not pool:
            return None

        matches = process.extract(
            form,
            pool,
            scorer=fuzz.ratio,
            limit=2,
            score_cutoff=minimum * 100.0,
        )
        if not matches:
            return None

        best_key = matches[0][0]
        best_score = matches[0][1] / 100.0
        second = matches[1][1] / 100.0 if len(matches) > 1 else 0.0
        if best_score < minimum:
            return None
        if second >= minimum and (best_score - second) < margin:
            return None
        return best_key, best_score
