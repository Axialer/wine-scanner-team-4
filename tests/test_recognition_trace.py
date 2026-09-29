"""Compact recognition trace for the browser console.

Run:
  .\\venv\\Scripts\\python.exe tests\\test_recognition_trace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_recognizer import (  # noqa: E402
    build_recognition_trace,
    public_recognition_json,
    search_trace,
)
from ocr_lexicon import OcrLexicon  # noqa: E402


def test_lexicon_records_kept_and_dropped():
    lexicon = OcrLexicon.build(
        grapes={"АЛИГОТЕ"},
        names={"GOLUBITSKOE"},
        wineries=set(),
    )
    cleaned = lexicon.clean("xyzqwk GOLUBITSKOE вино 2024")
    assert cleaned.tokens == ["GOLUBITSKOE"]
    assert cleaned.dropped
    assert any(item["raw"] == "XYZQWK" or "XYZ" in item["raw"].upper() for item in cleaned.dropped)


def test_trace_builder_is_compact():
    trace = build_recognition_trace(
        ocr_text="сырой текст",
        ocr_normalized="АЛИГОТЕ",
        kept=[{"raw": "АЛИГАТЕ", "catalog": "АЛИГОТЕ", "type": "grape", "score": 0.86}],
        dropped=[{"raw": "XYZQWK", "reason": "нет в словаре"}],
        crop={"strategy": "center", "box": [0, 1, 2, 3], "deskew_deg": 1.5},
        visual_scope="full_catalog",
        shortlist_size=0,
        visual_compared=12,
        catalog_count=100,
        reranker_text="АЛИГОТЕ",
        grape_tokens=["АЛИГОТЕ"],
        reranker_used=True,
        text_encoded=True,
        neighbors=[
            {"slug": f"s{i}", "name": f"n{i}", "visual_score": 0.5, "final_score": 0.4}
            for i in range(8)
        ],
        winner_slug="s0",
        winner_name="n0",
        status="found",
    )
    trace["search"] = search_trace(None, skipped="вино есть в каталоге")
    assert trace["ocr"] == "сырой текст"
    assert trace["lexicon"]["ocr_normalized"] == "АЛИГОТЕ"
    assert len(trace["neighbors"]) == 5
    assert set(trace["neighbors"][0]) == {"slug", "name", "visual", "final"}
    assert trace["search"]["where"] == "не вызывался"
    assert trace["search"]["reason"] == "вино есть в каталоге"
    assert trace["neural"]["scope"] == "весь каталог"
    assert "center" in trace["neural"]["crop"]
    assert trace["neural"]["features"]["reranker"] is True

    web = search_trace({
        "query": "вино алиготе",
        "status": "found",
        "from_cache": True,
        "name": "Алиготе",
        "grape": "Алиготе",
        "taste": "сухое",
        "text": "длинный текст страницы не должен попасть в trace",
    })
    assert web["where"] == "кэш"
    assert web["result"] == {
        "name": "Алиготе",
        "grape": "Алиготе",
        "color": "",
        "sweetness": "",
        "taste": "сухое",
        "source": "",
    }
    assert "text" not in web["result"]

    skipped = search_trace({"status": "no_lexicon", "query": ""})
    assert skipped["where"] == "не вызывался"
    assert skipped["reason"] == "нет словарных слов"

    live = search_trace({
        "query": "вино алиготе",
        "status": "empty",
        "from_cache": False,
    })
    assert live["where"] == "DuckDuckGo"


def test_public_json_hides_trace_unless_asked():
    result = {
        "status": "found",
        "method": "visual",
        "slug": "s0",
        "wine": {"name": "Имя", "final_score": 0.9, "visual_score": 0.8},
        "ocr_text": "raw",
        "ocr_clean": "АЛИГОТЕ",
        "confidence": 0.9,
        "timing": {},
        "visual_compared": 3,
        "_trace": {"ocr": "raw", "neighbors": []},
        "_debug": {"candidates": [{"slug": "secret"}]},
    }
    plain = public_recognition_json(result, debug=False, trace=False)
    assert "trace" not in plain
    assert "_debug" not in plain
    asked = public_recognition_json(result, debug=False, trace=True)
    assert asked["trace"]["ocr"] == "raw"
    assert "_debug" not in asked


if __name__ == "__main__":
    test_lexicon_records_kept_and_dropped()
    test_trace_builder_is_compact()
    test_public_json_hides_trace_unless_asked()
    print("ok")
