"""Web lookup parser and deadline. No live scrape.

Run:
  .\\venv\\Scripts\\python.exe tests\\test_web_lookup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_lookup.cache import ProductCache  # noqa: E402
from external_lookup.http_util import clamp_timeout  # noqa: E402
from external_lookup.web_lookup import (  # noqa: E402
    build_web_query,
    lookup_wine_query,
    normalize_query,
    parse_wine_html,
    search_result_links,
)

PAGE = """
<html><head>
<title>Шато Белое Совиньон Блан</title>
<meta name="description" content="Белое сухое вино, сорт Совиньон Блан. Регион: Крым">
<script type="application/ld+json">
{"@type":"Product","name":"Шато Белое","brand":"Дом Крыма","description":"Белое сухое, Совиньон Блан"}
</script>
</head><body></body></html>
"""

SEARCH = """
<a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.test%2Fwine&amp;rut=1">Шато Белое</a>
<a class="result__snippet">Белое сухое Совиньон Блан</a>
"""


def test_parse_name_grape_color():
    parsed = parse_wine_html(PAGE, "https://example.test/wine", {"СОВИНЬОН", "БЛАН"})
    assert parsed["name"] == "Шато Белое"
    assert "Совиньон" in parsed["grape"]
    assert parsed["color"] == "белое"
    assert parsed["taste"] == "сухое"
    assert "Крым" in parsed["region"]
    assert parsed["winery"] == "Дом Крыма"


def test_search_links_unwrap_duckduckgo():
    links = search_result_links(SEARCH, limit=2)
    assert links[0]["url"] == "https://example.test/wine"
    assert links[0]["title"] == "Шато Белое"


def test_timeout_returns_error_without_cache():
    calls = {"n": 0}

    def fetch(url, timeout, floor=1.5):
        calls["n"] += 1
        assert timeout <= 2.5
        assert floor <= 0.3
        raise TimeoutError("timed out")

    cache = ProductCache(ROOT / "index" / "external_cache_test")
    query = "тестовый таймаут совиньон"
    result = lookup_wine_query(query, cache, timeout=2.5, fetch=fetch)
    assert result["status"] == "error"
    assert calls["n"] == 1
    again = lookup_wine_query(query, cache, timeout=2.5, fetch=fetch)
    assert again["status"] == "error"
    assert calls["n"] == 2
    assert clamp_timeout(9, floor=0.3) == 2.5
    assert normalize_query("  АА  бб ") == "аа бб"


def test_cache_skips_second_fetch(tmp_path=None):
    calls = {"n": 0}

    def fetch(url, timeout, floor=1.5):
        calls["n"] += 1
        if "duckduckgo" in url:
            return 200, "text/html", SEARCH.encode("utf-8")
        return 200, "text/html", PAGE.encode("utf-8")

    cache = ProductCache(ROOT / "index" / "external_cache_test")
    query = "шато белое совиньон блан кэш"
    first = lookup_wine_query(
        query,
        cache,
        known_grape_tokens={"СОВИНЬОН"},
        fetch=fetch,
    )
    assert first["status"] == "found"
    assert first["color"] == "белое"
    used = calls["n"]
    second = lookup_wine_query(
        query,
        cache,
        known_grape_tokens={"СОВИНЬОН"},
        fetch=fetch,
    )
    assert second.get("from_cache") is True
    assert calls["n"] == used


def test_web_query_uses_recovered_words_only():
    raw = "жжжккк ххыыы АЛИГОТЕ мусорЦЫФРА ЦИТРОННЫЙ гогй белое 2024"
    query = build_web_query(raw, ["АЛИГОТЕ", "ЦИТРОННЫЙ", "БЕЛОЕ", "2024"])
    folded = query.casefold()
    assert "алиготе" in folded
    assert "цитронный" in folded
    assert "гогй" not in folded
    assert "жжжккк" not in folded
    assert "мусорцыфра" not in folded
    assert "белое" not in folded
    assert "2024" not in query
    assert query.startswith("вино ")
    assert build_web_query("ххыы мусор гогй белое сухое вино 2019", []) == ""
    assert build_web_query("только белое сухое", ["ВИНО", "СУХОЕ", "БЕЛОЕ"]) == ""


def main() -> None:
    tests = [
        test_parse_name_grape_color,
        test_search_links_unwrap_duckduckgo,
        test_timeout_returns_error_without_cache,
        test_cache_skips_second_fetch,
        test_web_query_uses_recovered_words_only,
    ]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
