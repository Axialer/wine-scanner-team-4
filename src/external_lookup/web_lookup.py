"""Web lookup for a bottle that is not in our catalog.

One normalized query is cached. The search stays inside about 4 seconds:
DuckDuckGo HTML, then the first result pages, then Wikipedia if the
snippet still has no grape or taste. A transport error is returned and
not cached. An empty parse is cached for a day.
"""

from __future__ import annotations

import html
import json
import re
import time
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from rapidfuzz.fuzz import ratio

from external_lookup.cache import ProductCache
from external_lookup.http_util import fetch_url
from external_lookup.qr_lookup import (
    _join_text,
    _meta_content,
    _product_from_jsonld,
    _title,
)
from ocr_shortlist import (
    _CYR_TO_LATIN,
    is_brut_misread,
    is_generic_token,
    text_tokens,
)
from similar_match import extract_query_signals, taste_notes

WEB_CACHE_PREFIX = "web:"
WEB_DEADLINE = 4.0
WEB_EMPTY_TTL = 24 * 60 * 60
SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"
SEARCH_POST_URL = "https://html.duckduckgo.com/html/"
WIKI_URL = (
    "https://ru.wikipedia.org/w/api.php?action=opensearch"
    "&limit=2&namespace=0&format=json&search={query}"
)

_WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru,en;q=0.8",
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

_COLOR_QUERY = {
    "БЕЛОЕ": "белое",
    "БЕЛЫЙ": "белое",
    "КРАСНОЕ": "красное",
    "КРАСНЫЙ": "красное",
    "РОЗОВОЕ": "розовое",
    "РОЗОВЫЙ": "розовое",
    "ОРАНЖЕВОЕ": "оранж",
    "ОРАНЖЕВЫЙ": "оранж",
    "ОРАНЖ": "оранж",
    "ORANGE": "оранж",
}
_SWEET_QUERY = {
    "БРЮТ": "брют",
    "BRUT": "брют",
    "СУХОЕ": "сухое",
    "СУХОЙ": "сухое",
    "СЛАДКОЕ": "сладкое",
    "СЛАДКИЙ": "сладкое",
    "ПОЛУСУХОЕ": "полусухое",
    "ПОЛУСУХОЙ": "полусухое",
    "ПОЛУСЛАДКОЕ": "полусладкое",
    "ПОЛУСЛАДКИЙ": "полусладкое",
}
_GRAPE_PAIRS = {
    ("ПИНО", "НУАР"): "Пино нуар",
    ("ПИНО", "ГРИ"): "Пино гри",
    ("ПИНО", "БЛАН"): "Пино блан",
    ("СОВИНЬОН", "БЛАН"): "Совиньон блан",
    ("КАБЕРНЕ", "СОВИНЬОН"): "Каберне совиньон",
    ("КАБЕРНЕ", "ФРАН"): "Каберне фран",
}
_STYLE_READY = frozenset({
    "оранж",
    "белое",
    "красное",
    "розовое",
    "брют",
    "сухое",
    "сладкое",
    "полусухое",
    "полусладкое",
    "кюве",
    "ultra",
})

_REGION_RE = re.compile(
    r"(?:регион|region)\s*[:\-–]?\s*([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё \-]{2,40})",
    flags=re.I,
)
_RESULT_LINK_RE = re.compile(
    r'<a\b[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    flags=re.I | re.S,
)
_RESULT_LINK_HREF_FIRST_RE = re.compile(
    r'<a\b[^>]*href="([^"]+)"[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>',
    flags=re.I | re.S,
)
_SNIPPET_RE = re.compile(
    r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|td|span|div)>',
    flags=re.I | re.S,
)
_PINOT_KEYS = frozenset({"пино", "pinot", "pino"})

_QUERY_KINDS = ("name", "grape", "winery", "extra")
_WEB_KEEP = frozenset({"ДОЛИНА"})
_MAX_QUERY_WORDS = 6


def normalize_query(text: str) -> str:
    """Cache key: one case-folded line, no extra spaces."""

    cleaned = " ".join(str(text or "").replace("\xa0", " ").split())
    return cleaned.casefold()[:240]


def _recovered_token(item) -> tuple[str, str | None]:
    if isinstance(item, str):
        return item.strip(), None
    if isinstance(item, dict):
        token = item.get("catalog") or item.get("token") or ""
        kind = item.get("type") or item.get("kind")
        return str(token).strip(), (str(kind) if kind else None)
    token = getattr(item, "catalog", "") or ""
    kind = getattr(item, "kind", None) or getattr(item, "type", None)
    return str(token).strip(), (str(kind) if kind else None)


def _raw_of(item) -> str:
    if isinstance(item, dict):
        return str(item.get("raw") or "")
    return str(getattr(item, "raw", "") or "")


def _dedupe_key(token: str) -> str:
    upper = str(token or "").upper()
    return upper.translate(_CYR_TO_LATIN).casefold()


def _false_pinot(token: str, raw: str) -> bool:
    """ПИНО snapped from 5PIOT/БРЮТ is not a grape."""

    if _dedupe_key(token) not in _PINOT_KEYS:
        return False
    return is_brut_misread(raw) or is_brut_misread(token)


def _keep_query_token(token: str) -> bool:
    if not token or token.isdigit():
        return False
    if is_brut_misread(token):
        return False
    if token in _WEB_KEEP:
        return True
    if is_generic_token(token):
        return False
    return True


def _pretty_token(token: str) -> str:
    token = str(token or "").strip()
    if not token:
        return ""
    if token.casefold() in _STYLE_READY:
        return "Ultra" if token.casefold() == "ultra" else token.casefold()
    return token[:1] + token[1:].lower()


def _close_token(token: str, target: str, minimum: int = 80) -> bool:
    if abs(len(token) - len(target)) > 1:
        return False
    return ratio(token, target) >= minimum


def _style_words(raw: str) -> list[str]:
    """Color, sweetness, and line words printed on the label."""

    lines: list[str] = []
    colors: list[str] = []
    sweets: list[str] = []
    seen: set[str] = set()

    def add(bucket: list[str], word: str) -> None:
        key = word.casefold()
        if key in seen:
            return
        seen.add(key)
        bucket.append(word)

    for token in text_tokens(raw or ""):
        if token in _COLOR_QUERY:
            add(colors, _COLOR_QUERY[token])
        elif _close_token(token, "ОРАНЖ"):
            add(colors, "оранж")
        if is_brut_misread(token):
            add(sweets, "брют")
        elif token in _SWEET_QUERY:
            add(sweets, _SWEET_QUERY[token])
        if token in {"ULTRA", "УЛЬТРА"} or _close_token(token, "УЛЬТРА"):
            add(lines, "Ultra")
        if token in {"CUVEE", "CUVE", "КЮВЕ"}:
            add(lines, "кюве")
    return lines + colors + sweets


def build_web_query(raw_text: str = "", recovered=None) -> str:
    """Short query from lexicon words, plus color and sweetness on the label.

    Homoglyph twins collapse to one word. 5PIOT/БРЮТ never becomes Пино.
    Garbage and a color-only line produce an empty query.
    """

    grouped: dict[str, list[str]] = {kind: [] for kind in _QUERY_KINDS}
    extras: list[str] = []
    plain: list[str] = []
    seen: set[str] = set()
    typed = False

    def add(bucket: list[str], token: str) -> None:
        key = _dedupe_key(token)
        if not key or key in seen or not _keep_query_token(token):
            return
        seen.add(key)
        bucket.append(token)

    for item in recovered or []:
        token, kind = _recovered_token(item)
        raw = _raw_of(item)
        if _false_pinot(token, raw):
            continue
        if kind == "extra":
            add(extras, token)
        elif kind in _QUERY_KINDS:
            typed = True
            add(grouped[kind], token)
        elif kind is None:
            add(plain, token)

    if typed:
        ordered = grouped["name"] + grouped["winery"] + grouped["grape"]
    elif plain:
        ordered = plain
    else:
        ordered = extras
    if not ordered:
        return ""

    for word in _style_words(raw_text):
        key = _dedupe_key(word)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(word)

    words: list[str] = []
    for token in ordered:
        pretty = _pretty_token(token)
        if pretty and pretty.casefold() != "вино":
            words.append(pretty)
        if len(words) >= _MAX_QUERY_WORDS:
            break
    if not words:
        return ""
    return ("вино " + " ".join(words))[:300]


def best_lookup_query(
    ocr_clean: str = "",
    ocr_raw: str = "",
    external_text: str = "",
    recovered=None,
) -> str:
    """Lexicon words only. Raw OCR blur is not copied in as-is."""

    del external_text
    if recovered is not None:
        return build_web_query(ocr_raw, recovered)
    tokens = [part for part in str(ocr_clean or "").split() if _keep_query_token(part)]
    return build_web_query(ocr_raw, tokens)


def _empty(query: str, status: str) -> dict:
    return {
        "query": query,
        "name": "",
        "grape": "",
        "color": "",
        "taste": "",
        "sweetness": "",
        "region": "",
        "winery": "",
        "source": "",
        "text": "",
        "status": status,
        "from_cache": False,
        "where": "",
    }


def _strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(html.unescape(text).split())


def _visible_text(page: str, limit: int = 2500) -> str:
    cleaned = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        page or "",
        flags=re.I | re.S,
    )
    cleaned = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        cleaned,
        flags=re.I | re.S,
    )
    return _strip_tags(cleaned)[:limit]


def _result_url(href: str) -> str:
    href = html.unescape(href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    target = ""
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg") or []
        target = unquote(uddg[0]) if uddg else ""
    else:
        target = href
    if not target.startswith("http"):
        return ""
    host = urlparse(target).netloc.lower()
    if "duckduckgo.com" in host:
        return ""
    return target


def search_result_links(page: str, limit: int = 2) -> list[dict]:
    found: list[dict] = []
    seen: set[str] = set()
    pairs = list(_RESULT_LINK_RE.findall(page or ""))
    pairs.extend(_RESULT_LINK_HREF_FIRST_RE.findall(page or ""))
    for href, title_html in pairs:
        url = _result_url(href)
        if not url or url in seen:
            continue
        seen.add(url)
        found.append({"url": url, "title": _strip_tags(title_html)})
        if len(found) >= limit:
            break
    return found


def _format_grapes(grapes: list[str]) -> str:
    phrases: list[str] = []
    index = 0
    upper = [token.upper() for token in grapes]
    while index < len(grapes):
        if index + 1 < len(grapes):
            phrase = _GRAPE_PAIRS.get((upper[index], upper[index + 1]))
            if phrase:
                phrases.append(phrase)
                index += 2
                continue
        pretty = _pretty_token(grapes[index])
        if pretty:
            phrases.append(pretty)
        index += 1
    return ", ".join(phrases)


def attributes_from_text(text: str, known_grape_tokens: set[str] | None) -> dict:
    signals = extract_query_signals(text or "", known_grape_tokens or set())
    region = ""
    match = _REGION_RE.search(text or "")
    if match:
        region = " ".join(match.group(1).split())
    return {
        "grape": _format_grapes(signals.grapes),
        "color": _COLOR_RU.get(signals.color or "", ""),
        "taste": taste_notes(text or ""),
        "sweetness": _SWEET_RU.get(signals.sweetness or "", ""),
        "region": region,
    }


def parse_wine_html(page: str, url: str, known_grape_tokens: set[str] | None) -> dict:
    product = _product_from_jsonld(page or "")
    title = _title(page or "")
    og_title = _meta_content(page or "", "property", "og:title")
    og_description = _meta_content(page or "", "property", "og:description")
    description = _meta_content(page or "", "name", "description")
    visible = _visible_text(page or "")
    name = product.get("name") or og_title or title
    winery = product.get("winery") or product.get("brand") or ""
    body = _join_text(
        name,
        winery,
        product.get("description") or "",
        title,
        og_title,
        og_description,
        description,
        visible,
    )
    attrs = attributes_from_text(body, known_grape_tokens)
    return {
        "name": name,
        "winery": winery,
        "source": url,
        "text": body[:1500],
        **attrs,
    }


def _merge_page(base: dict, page: dict) -> None:
    for key in ("name", "grape", "color", "taste", "sweetness", "region", "winery", "source"):
        if not base.get(key) and page.get(key):
            base[key] = page[key]
    if page.get("text"):
        base["text"] = _join_text(base.get("text") or "", page["text"])[:1500]


def _has_detail(result: dict) -> bool:
    return bool(result.get("grape") or result.get("taste"))


def _useful(result: dict) -> bool:
    return any(
        result.get(key)
        for key in ("grape", "color", "sweetness", "taste", "winery")
    )


def lookup_reason(result: dict | None) -> str:
    """What the search actually found, in one Russian line."""

    data = result or {}
    status = str(data.get("status") or "")
    if status == "error":
        return "сеть не ответила"
    if status == "no_lexicon":
        return "нет словарных слов"
    bits: list[str] = []
    if data.get("grape"):
        bits.append("сорт " + str(data["grape"]))
    if data.get("color"):
        bits.append(str(data["color"]))
    if data.get("sweetness"):
        bits.append(str(data["sweetness"]))
    if data.get("taste"):
        bits.append(str(data["taste"]))
    if data.get("winery"):
        bits.append(str(data["winery"]))
    if bits:
        return "нашли " + ", ".join(bits)
    if status == "found" and data.get("name"):
        return "нашли «" + str(data["name"]) + "»"
    return "ничего ясного не нашлось"


def search_result_view(result: dict | None) -> dict | None:
    """Console object. Present only when a profile was parsed."""

    data = result or {}
    if str(data.get("status") or "") != "found":
        return None
    return {
        "name": data.get("name") or "",
        "grape": data.get("grape") or "",
        "color": data.get("color") or "",
        "sweetness": data.get("sweetness") or "",
        "taste": data.get("taste") or "",
        "source": data.get("source") or "",
    }


def _first_brand(recovered) -> str:
    for kind in ("name", "winery"):
        for item in recovered or []:
            token, item_kind = _recovered_token(item)
            raw = _raw_of(item)
            if item_kind != kind:
                continue
            if _false_pinot(token, raw) or not _keep_query_token(token):
                continue
            pretty = _pretty_token(token)
            if pretty:
                return pretty
    return ""


def profile_from_query(query: str, recovered, known_grape_tokens: set[str] | None) -> dict:
    """Catalog-similarity profile from the clean label words. No network."""

    attrs = attributes_from_text(query, known_grape_tokens)
    brand = _first_brand(recovered)
    return {
        "query": query,
        "name": brand,
        "grape": attrs["grape"],
        "color": attrs["color"],
        "taste": attrs["taste"],
        "sweetness": attrs["sweetness"],
        "region": attrs["region"],
        "winery": brand,
        "source": "",
        "text": query,
        "status": "ocr",
        "from_cache": False,
        "where": "",
    }


def fill_profile_gaps(web: dict, ocr: dict | None) -> dict:
    """Keep the page, and fill color or sweetness the label already had."""

    if not ocr:
        return web
    merged = dict(web)
    for key in ("grape", "color", "sweetness", "taste", "winery", "name", "region"):
        if not str(merged.get(key) or "").strip() and ocr.get(key):
            merged[key] = ocr[key]
    if ocr.get("text"):
        merged["text"] = _join_text(merged.get("text") or "", ocr["text"])[:1500]
    return merged


def _wiki_hits(raw: str) -> list[dict]:
    try:
        data = json.loads(raw or "")
    except Exception:
        return []
    if not isinstance(data, list) or len(data) < 4:
        return []
    titles, descriptions, urls = data[1], data[2], data[3]
    found: list[dict] = []
    for index, url in enumerate(urls or []):
        target = str(url or "")
        if not target.startswith("http"):
            continue
        title = titles[index] if index < len(titles) else ""
        text = descriptions[index] if index < len(descriptions) else ""
        found.append({
            "url": target,
            "title": " ".join(str(title).split()),
            "text": " ".join(str(text).split()),
        })
    return found


def _absorb_text(result: dict, text: str, source: str, title: str, grapes) -> None:
    parsed = {
        "name": title,
        "text": text,
        "source": source,
        **attributes_from_text(_join_text(title, text), grapes),
    }
    _merge_page(result, parsed)


def lookup_wine_query(
    query: str,
    cache: ProductCache,
    timeout: float = WEB_DEADLINE,
    known_grape_tokens: set[str] | None = None,
    fetch=None,
) -> dict:
    """Search the web for a wine name and parse a short attribute profile.

    Cache key is the normalized query. Repeats do not touch the network.
    Empty parses expire after a day. Transport errors are not cached.
    """

    fetch_fn = fetch or fetch_url
    normalized = normalize_query(query)
    if not normalized:
        return _empty("", "empty")

    cache_key = WEB_CACHE_PREFIX + normalized
    cached = cache.get_usable(cache_key)
    if cached is not None:
        return cached

    result = _empty(normalized, "empty")
    deadline = time.monotonic() + min(WEB_DEADLINE, max(0.4, float(timeout or WEB_DEADLINE)))
    grapes = known_grape_tokens

    def remaining() -> float:
        return deadline - time.monotonic()

    def fetch_within(url: str, data: bytes | None = None) -> str:
        left = remaining()
        if left < 0.35:
            raise TimeoutError("web lookup deadline")
        _status, _content_type, raw = fetch_fn(
            url,
            left,
            floor=0.3,
            ceiling=max(left, 0.3),
            headers=_WEB_HEADERS,
            data=data,
        )
        return raw.decode("utf-8", errors="ignore")

    search_query = normalized
    if "вино" not in search_query and "wine" not in search_query:
        search_query = search_query + " вино"

    try:
        search_html = fetch_within(SEARCH_URL.format(query=quote(search_query)))
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:300]
        result["where"] = "DuckDuckGo"
        return result

    links = search_result_links(search_html, limit=2)
    snippets = " ".join(_strip_tags(part) for part in _SNIPPET_RE.findall(search_html))
    if not links and not snippets and remaining() >= 0.35:
        try:
            posted = fetch_within(
                SEARCH_POST_URL,
                data=urlencode({"q": search_query, "kl": "ru-ru"}).encode("utf-8"),
            )
        except Exception:
            posted = ""
        if posted:
            search_html = posted
            links = search_result_links(search_html, limit=2)
            snippets = " ".join(_strip_tags(part) for part in _SNIPPET_RE.findall(search_html))

    titles = " ".join(item["title"] for item in links)
    result["where"] = "DuckDuckGo"
    if snippets or titles:
        _absorb_text(
            result,
            _join_text(titles, snippets),
            links[0]["url"] if links else "",
            links[0]["title"] if links else "",
            grapes,
        )

    for link in links:
        if remaining() < 0.35 or _has_detail(result):
            break
        try:
            page = fetch_within(link["url"])
        except Exception:
            continue
        parsed = parse_wine_html(page, link["url"], grapes)
        if not parsed.get("name"):
            parsed["name"] = link["title"]
        _merge_page(result, parsed)

    if not _has_detail(result) and remaining() >= 0.35:
        wiki_raw = ""
        try:
            wiki_raw = fetch_within(WIKI_URL.format(query=quote(search_query)))
        except Exception:
            wiki_raw = ""
        hits = _wiki_hits(wiki_raw)
        if hits:
            if result["where"] == "DuckDuckGo":
                result["where"] = "DuckDuckGo, Wikipedia"
            else:
                result["where"] = "Wikipedia"
            _absorb_text(
                result,
                hits[0]["text"],
                hits[0]["url"],
                hits[0]["title"],
                grapes,
            )
            if not _has_detail(result) and remaining() >= 0.35:
                try:
                    page = fetch_within(hits[0]["url"])
                except Exception:
                    page = ""
                if page:
                    parsed = parse_wine_html(page, hits[0]["url"], grapes)
                    if not parsed.get("name"):
                        parsed["name"] = hits[0]["title"]
                    _merge_page(result, parsed)

    result["status"] = "found" if _useful(result) else "empty"
    result["query"] = normalized
    if result["status"] == "empty":
        result["negative_ttl"] = WEB_EMPTY_TTL
    cache.set(cache_key, result)
    return result
