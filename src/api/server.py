"""
Minimal FastAPI wrapper around HybridWineRecognizer.

Run from repo root:
  uvicorn api.server:app --app-dir src --host 0.0.0.0 --port 8091

Dev UI on this machine: http://127.0.0.1:8091/
Phone on the same Wi-Fi: http://<LAN-IPv4>:8091/

Catalog browsing reads CSV + catalog_meta.json and does not wait for
OCR or SigLIP. Heavy inference is serialized with a lock; for multi-user
production use a worker pool or dedicated GPU service and keep this API thin.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import socket
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

STATIC_DIR = Path(__file__).resolve().parent / "static"
SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
META_PATH = ROOT / "index" / "catalog_meta.json"
CSV_PATH = ROOT / "data" / "catalog" / "strapi_output0709.csv"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

app = FastAPI(title="Wine Scanner Hybrid API", version="0.1.0")

recognizer = None
inference_lock = asyncio.Semaphore(1)
_IMAGE_TYPES = {
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _cell(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _resolve_catalog_file(raw: str) -> Path | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_file():
        path = ROOT / raw.replace("\\", "/")
    if path.is_file():
        return path
    return None


def load_catalog() -> tuple[list[dict], dict[str, Path]]:
    """Wine cards from the same CSV and image index the recognizer uses."""

    files: dict[str, Path] = {}
    with META_PATH.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    for item in meta.get("items") or []:
        slug = _cell(item.get("slug"))
        if not slug or slug in files:
            continue
        path = _resolve_catalog_file(str(item.get("file") or ""))
        if path is not None:
            files[slug] = path

    wines: list[dict] = []
    seen: set[str] = set()
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            slug = _cell(row.get("Slug"))
            if not slug or slug in seen:
                continue
            seen.add(slug)
            card = {
                "name": _cell(row.get("Название вина")),
                "slug": slug,
                "winery": _cell(row.get("Винодельня")),
                "region": _cell(row.get("Регион")),
                "grape": _cell(row.get("Сорт винограда")),
                "color": _cell(row.get("Цвет")),
                "category": _cell(row.get("Категория")),
                "description": _cell(row.get("Описание")),
                "image": f"/catalog-image/{slug}" if slug in files else "",
            }
            for column, key in (
                ("Вкус", "taste"),
                ("Сахар", "sugar"),
                ("Содержание сахара", "sugar"),
                ("Гастрономия", "food"),
                ("Сочетание с едой", "food"),
            ):
                extra = _cell(row.get(column))
                if extra and not card.get(key):
                    card[key] = extra
            wines.append(card)
    return wines, files


CATALOG_WINES, CATALOG_FILES = load_catalog()


def _matches_query(wine: dict, needle: str) -> bool:
    if not needle:
        return True
    haystack = " ".join((
        wine.get("name") or "",
        wine.get("winery") or "",
        wine.get("grape") or "",
    )).casefold()
    return needle in haystack


def _boot_recognizer() -> None:
    global recognizer
    from hybrid_recognizer import HybridWineRecognizer

    instance = HybridWineRecognizer()
    instance.warmup()
    recognizer = instance


@app.on_event("startup")
def startup() -> None:
    threading.Thread(
        target=_boot_recognizer,
        name="recognizer-boot",
        daemon=True,
    ).start()
    port = "8091"
    if "--port" in sys.argv:
        index = sys.argv.index("--port")
        if index + 1 < len(sys.argv):
            port = sys.argv[index + 1]
    addresses = lan_ipv4_addresses()
    log = logging.getLogger("uvicorn.error")
    log.info("Dev UI: http://127.0.0.1:%s/", port)
    if addresses:
        log.info("Phone URL (same Wi-Fi, host 0.0.0.0):")
        for ip in addresses:
            log.info("  http://%s:%s/", ip, port)
    else:
        log.info("Phone URL: no LAN IPv4 found. Bind is still 0.0.0.0.")


_SKIP_ADAPTER = (
    "loopback",
    "vethernet",
    "hyper-v",
    "wsl",
    "vmware",
    "virtualbox",
    "bluetooth",
    "radmin",
    "tailscale",
    "docker",
    "npcap",
    "happ",
    "xray",
    "vpn",
)


def lan_ipv4_addresses() -> list[str]:
    """Local IPv4 addresses a phone on the same network can open."""

    found: list[str] = []
    try:
        import subprocess

        raw = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-NetIPAddress -AddressFamily IPv4 | "
                "ForEach-Object { $_.IPAddress + '|' + $_.InterfaceAlias }",
            ],
            text=True,
            timeout=8,
        )
        for line in raw.splitlines():
            if "|" not in line:
                continue
            ip, alias = line.split("|", 1)
            ip = ip.strip()
            alias_key = alias.strip().casefold()
            if not ip or ip.startswith("127.") or ip.startswith("169.254."):
                continue
            if any(marker in alias_key for marker in _SKIP_ADAPTER):
                continue
            if ip not in found:
                found.append(ip)
    except Exception:
        found = []

    if found:
        return found

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = probe.getsockname()[0]
        probe.close()
        if ip and not ip.startswith("127."):
            return [ip]
    except OSError:
        pass
    return []


def _dev_ui() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Dev UI not found")
    return FileResponse(index)


@app.get("/")
def dev_ui() -> FileResponse:
    return _dev_ui()


@app.get("/site.css")
def site_css() -> FileResponse:
    path = STATIC_DIR / "site.css"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Stylesheet not found")
    return FileResponse(path, media_type="text/css")


@app.get("/wine/{slug}")
def wine_page(slug: str) -> FileResponse:
    """Same UI. The page script reads /wine/{slug} and opens that card."""

    return _dev_ui()


@app.get("/health")
def health() -> dict:
    ready = recognizer is not None
    device = None
    ocr_backend = None
    if ready:
        from hybrid_recognizer import DEVICE

        device = DEVICE
        ocr_backend = getattr(getattr(recognizer, "ocr", None), "backend", None)
    return {
        "status": "ok" if ready else "loading",
        "recognizer": ready,
        "pipeline": "ocr+siglip" if ready else None,
        "ocr": ocr_backend,
        "device": device,
        "catalog": len(CATALOG_WINES),
    }


@app.get("/catalog")
def catalog(q: str = Query("")) -> dict:
    needle = q.strip().casefold()
    wines = [wine for wine in CATALOG_WINES if _matches_query(wine, needle)]
    return {
        "q": q.strip(),
        "total": len(CATALOG_WINES),
        "count": len(wines),
        "wines": wines,
    }


class RecommendIn(BaseModel):
    text: str = ""


@app.post("/recommend")
async def recommend(body: RecommendIn) -> dict:
    """Text ask against catalog embeddings. No OCR and no visual search."""

    from text_similarity import query_too_short_message

    text = " ".join((body.text or "").split())
    message = query_too_short_message(text)
    if message:
        raise HTTPException(status_code=400, detail=message)
    if recognizer is None:
        raise HTTPException(
            status_code=503,
            detail="Подбор ещё загружается. Подождите немного.",
        )

    async with inference_lock:
        wines = await asyncio.to_thread(recognizer.recommend_text, text)
    return {"text": text, "wines": wines}


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    debug: bool = Query(False),
    trace: bool = Query(False),
) -> dict:
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Recognizer not ready")

    from hybrid_recognizer import public_recognition_json

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(content))
        image = ImageOps.exif_transpose(image) or image
        image = image.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        ) from exc

    # Full hybrid path: codes, OCR shortlist, then SigLIP inside that shortlist.
    async with inference_lock:
        result = await asyncio.to_thread(
            recognizer.recognize,
            image,
            debug,
        )

    return public_recognition_json(result, debug=debug, trace=trace)


@app.get("/catalog-image/{slug}")
def catalog_image(slug: str) -> FileResponse:
    path = CATALOG_FILES.get(slug)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Catalog image not found")
    media = _IMAGE_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media)


app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
