from __future__ import annotations

import json
import re

from external_lookup.cache import ProductCache
from external_lookup.http_util import fetch_url

BARCODE_CACHE_PREFIX = "barcode:"
OFF_PRODUCT_URL = (
    "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
)


def _empty_product(code: str, gtin: str, status: str) -> dict:
    return {
        "code": code,
        "name": "",
        "brand": "",
        "winery": "",
        "gtin": gtin,
        "url": "",
        "text": "",
        "status": status,
        "from_cache": False,
    }


def _join_text(*parts: str) -> str:
    seen: list[str] = []
    for part in parts:
        cleaned = " ".join(str(part or "").split())
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return " ".join(seen)[:5000]


def lookup_barcode(
    code: str,
    cache: ProductCache,
    timeout: float = 2.0,
) -> dict:
    """GTIN lookup via Open Food Facts, cache-first.

    A cache hit does not call the network. GS1 is not used.
    Successes are cached; unknown codes are cached as negatives.
    Timeouts and transport errors are not cached.
    """

    code = (code or "").strip()
    cache_key = BARCODE_CACHE_PREFIX + code
    cached = cache.get_usable(cache_key)
    if cached is not None:
        return cached

    digits = re.sub(r"\D", "", code)
    gtin = digits if len(digits) in (8, 12, 13, 14) else ""
    if not gtin:
        result = _empty_product(code, "", "not_found")
        cache.set(cache_key, result)
        return result

    url = OFF_PRODUCT_URL.format(barcode=gtin)
    try:
        _status, content_type, body = fetch_url(url, timeout)
    except Exception as exc:
        result = _empty_product(code, gtin, "error")
        result["url"] = url
        result["error"] = str(exc)[:300]
        return result

    if "json" not in content_type.lower() and not body.lstrip().startswith(b"{"):
        result = _empty_product(code, gtin, "error")
        result["url"] = url
        result["error"] = "non_json"
        return result

    try:
        payload = json.loads(body.decode("utf-8", errors="ignore"))
    except Exception as exc:
        result = _empty_product(code, gtin, "error")
        result["url"] = url
        result["error"] = str(exc)[:300]
        return result

    product = payload.get("product") if isinstance(payload, dict) else None
    found = (
        isinstance(payload, dict)
        and int(payload.get("status") or 0) == 1
        and isinstance(product, dict)
    )
    if not found:
        result = _empty_product(code, gtin, "not_found")
        result["url"] = url
        cache.set(cache_key, result)
        return result

    name = str(
        product.get("product_name")
        or product.get("product_name_ru")
        or product.get("product_name_en")
        or ""
    ).strip()
    brand = str(product.get("brands") or "").strip()
    winery = str(
        product.get("manufacturing_places")
        or product.get("origins")
        or brand
    ).strip()
    gtin_value = str(product.get("code") or gtin).strip()
    page_url = str(
        product.get("url")
        or f"https://world.openfoodfacts.org/product/{gtin_value}"
    ).strip()
    categories = str(product.get("categories") or "").strip()
    quantity = str(product.get("quantity") or "").strip()
    text = _join_text(name, brand, winery, categories, quantity)

    result = {
        "code": code,
        "name": name,
        "brand": brand,
        "winery": winery,
        "gtin": gtin_value,
        "url": page_url,
        "text": text,
        "status": "found" if text else "empty",
        "from_cache": False,
    }
    cache.set(cache_key, result)
    return result
