from __future__ import annotations

import html
import json
import re

from external_lookup.cache import ProductCache
from external_lookup.http_util import fetch_url

QR_CACHE_PREFIX = "qr:"

_GTIN_KEYS = ("gtin", "gtin8", "gtin12", "gtin13", "gtin14")


def _empty(url: str, status: str) -> dict:
    return {
        "url": url,
        "name": "",
        "brand": "",
        "winery": "",
        "gtin": "",
        "description": "",
        "text": "",
        "status": status,
        "from_cache": False,
    }


def _as_text(value) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, dict):
        return _as_text(value.get("name"))
    if isinstance(value, list):
        for item in value:
            text = _as_text(item)
            if text:
                return text
    return ""


def _types(node: dict) -> set[str]:
    raw = node.get("@type")
    values = raw if isinstance(raw, list) else [raw]
    return {str(item).lower() for item in values if item}


def _walk_products(node, found: list[dict]) -> None:
    if isinstance(node, dict):
        if "product" in _types(node):
            found.append(node)
        graph = node.get("@graph")
        if isinstance(graph, list):
            for child in graph:
                _walk_products(child, found)
        for key, child in node.items():
            if key == "@graph":
                continue
            if isinstance(child, (dict, list)):
                _walk_products(child, found)
    elif isinstance(node, list):
        for child in node:
            _walk_products(child, found)


def _product_from_jsonld(page: str) -> dict:
    scripts = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>'
        r"(.*?)</script>",
        page,
        flags=re.I | re.S,
    )
    products: list[dict] = []
    for script in scripts:
        try:
            data = json.loads(html.unescape(script))
        except Exception:
            continue
        _walk_products(data, products)

    if not products:
        return {}

    product = products[0]
    gtin = ""
    for key in _GTIN_KEYS:
        gtin = _as_text(product.get(key))
        if gtin:
            break

    return {
        "name": _as_text(product.get("name")),
        "brand": _as_text(product.get("brand")),
        "winery": _as_text(
            product.get("manufacturer") or product.get("brand")
        ),
        "gtin": re.sub(r"\D", "", gtin),
        "description": _as_text(product.get("description")),
    }


def _meta_content(page: str, attr: str, key: str) -> str:
    pattern = re.compile(r"<meta\b[^>]*>", flags=re.I)
    attr_re = re.compile(
        attr + r"\s*=\s*['\"]" + re.escape(key) + r"['\"]",
        flags=re.I,
    )
    content_re = re.compile(
        r"content\s*=\s*['\"](.*?)['\"]",
        flags=re.I | re.S,
    )
    for tag in pattern.findall(page):
        if not attr_re.search(tag):
            continue
        match = content_re.search(tag)
        if match:
            return " ".join(html.unescape(match.group(1)).split())
    return ""


def _title(page: str) -> str:
    match = re.search(
        r"<title[^>]*>(.*?)</title>",
        page,
        flags=re.I | re.S,
    )
    if not match:
        return ""
    return " ".join(html.unescape(match.group(1)).split())


def _join_text(*parts: str) -> str:
    seen: list[str] = []
    for part in parts:
        cleaned = " ".join(str(part or "").split())
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return " ".join(seen)[:5000]


def lookup_qr_url(
    url: str,
    cache: ProductCache,
    timeout: float = 2.0,
) -> dict:
    """Fetch a QR URL and pull product name, brand, GTIN, title."""

    url = (url or "").strip()
    cache_key = QR_CACHE_PREFIX + url
    cached = cache.get_usable(cache_key)
    if cached is not None:
        return cached

    result = _empty(url, "not_found")

    try:
        _status, content_type, raw = fetch_url(url, timeout)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:300]
        return result

    if "html" not in content_type.lower() and b"<html" not in raw[:500].lower():
        result["status"] = "non_html"
        cache.set(cache_key, result)
        return result

    page = raw.decode("utf-8", errors="ignore")
    product = _product_from_jsonld(page)
    title = _title(page)
    og_title = _meta_content(page, "property", "og:title")
    og_description = _meta_content(page, "property", "og:description")
    description = _meta_content(page, "name", "description")

    name = product.get("name") or og_title or title
    brand = product.get("brand") or ""
    winery = product.get("winery") or brand
    gtin = product.get("gtin") or ""
    product_description = (
        product.get("description") or og_description or description
    )
    text = _join_text(
        name,
        brand,
        winery,
        product_description,
        title,
        og_title,
        og_description,
        description,
    )

    result.update({
        "name": name,
        "brand": brand,
        "winery": winery,
        "gtin": gtin,
        "description": product_description,
        "text": text,
        "status": "found" if text else "empty",
    })
    cache.set(cache_key, result)
    return result
