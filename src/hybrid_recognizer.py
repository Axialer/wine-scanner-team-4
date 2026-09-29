from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import pandas as pd
import torch
import zxingcpp

from PIL import Image, ImageOps
from rapidfuzz.fuzz import ratio
from transformers import AutoModel, AutoProcessor

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from external_lookup import (  # noqa: E402
    ProductCache,
    lookup_barcode,
    lookup_qr_url,
)
from external_lookup.web_lookup import (  # noqa: E402
    SEARCH_URL,
    WEB_DEADLINE,
    build_web_query,
    fill_profile_gaps,
    lookup_reason,
    lookup_wine_query,
    profile_from_query,
    search_result_view,
)
from fast_ocr import FastOcr  # noqa: E402
from label_crop import prepare_label  # noqa: E402
from learned_reranker import candidate_features, load_reranker  # noqa: E402
from ocr_lexicon import OcrLexicon  # noqa: E402
from ocr_shortlist import (  # noqa: E402
    _resolve_grape,
    build_ocr_shortlist,
    extract_years,
    has_second_distinctive,
    is_generic_token,
    merge_compatible_tokens,
    normalize_text,
    order_candidates,
    recover_names_inside_producer,
    text_tokens,
    text_visual_weights,
)
from similar_match import (  # noqa: E402
    build_similar,
    decide_match,
    distinct_runner,
    extract_query_signals,
    public_view,
    rank_catalog_by_attributes,
    signals_are_generic_only,
)
from text_similarity import (  # noqa: E402
    catalog_blob,
    profile_blob,
    rank_catalog_by_vectors,
    rank_catalog_for_sentence,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_PATH = ROOT / "index" / "catalog_embeddings.npy"
TEXT_EMB_PATH = ROOT / "index" / "catalog_text_emb.npy"
TEXT_SLUGS_PATH = ROOT / "index" / "catalog_text_slugs.json"
FINETUNE_PROJ_PATH = ROOT / "index" / "finetune_proj.npy"
FINETUNE_FLAG_PATH = ROOT / "index" / "finetune_enabled.txt"
RERANKER_PATH = ROOT / "index" / "reranker.npy"
RERANKER_FLAG_PATH = ROOT / "index" / "reranker_enabled.txt"
META_PATH = ROOT / "index" / "catalog_meta.json"
CSV_PATH = ROOT / "data" / "catalog" / "strapi_output0709.csv"

CACHE_DIR = ROOT / "index" / "external_cache"

MODEL_NAME = "google/siglip2-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Сколько визуальных кандидатов оставляем для OCR rerank.
VISUAL_TOP_K = 30

# В JSON/API возвращаем только один итоговый вариант.
FINAL_TOP_K = 1

# Two OCR passes on a downscaled label crop: Cyrillic, then English.
OCR_MAX_SIDE = 800

# Hard cap so a slow Open Food Facts / QR fetch cannot stall the photo.
EXTERNAL_TIMEOUT = 2.0

CODE_MAX_SIDE = 1400

# Если найден QR/штрихкод и он уже есть локально — это абсолютный приоритет.
CODE_EXACT_MATCH = True

# Режим отладки включается через:
# python src\hybrid_recognizer.py PHOTO --debug
DEBUG = "--debug" in sys.argv


# ============================================================
# TEXT UTILS
# ============================================================

def compact_token(token: str) -> str:
    return re.sub(r"[^A-ZА-Я0-9]", "", normalize_text(token))


def _flag_on(path: Path, weight_path: Path) -> bool:
    if not path.is_file() or not weight_path.is_file():
        return False
    flag = path.read_text(encoding="utf-8").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _finetune_enabled() -> bool:
    return _flag_on(FINETUNE_FLAG_PATH, FINETUNE_PROJ_PATH)


def _apply_proj(matrix: np.ndarray, proj: np.ndarray) -> np.ndarray:
    mapped = np.asarray(matrix, dtype=np.float32) @ proj
    norms = np.linalg.norm(mapped, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mapped / norms).astype(np.float32)


def _first_cell(row, columns: tuple[str, ...]) -> str:
    for column in columns:
        text = _cell(row.get(column, ""))
        if text:
            return text
    return ""


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _cell(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


# ============================================================
# CODE DETECTION
# ============================================================

class CodeDetector:

    def __init__(
        self,
        metadata: dict[str, dict],
        catalog_text: dict[str, dict] | None = None,
    ):
        self.metadata = metadata
        self.catalog_text = catalog_text or {}
        self.product_cache = ProductCache(CACHE_DIR)

        self.code_index: dict[str, list[str]] = {}

        for slug, data in metadata.items():
            for value in data.values():
                if value is None:
                    continue

                value = str(value).strip()

                if not value:
                    continue

                digits = re.sub(r"\D", "", value)

                if len(digits) in (8, 12, 13, 14):
                    self.code_index.setdefault(
                        digits,
                        [],
                    ).append(slug)

        print(
            f"Code index: {len(self.code_index)} codes"
        )

    # --------------------------------------------------------
    # LOCAL CODE -> SLUG
    # --------------------------------------------------------

    def find_slug_by_code(
        self,
        code: str,
    ) -> str | None:

        if not code:
            return None

        code = code.strip()

        digits = re.sub(
            r"\D",
            "",
            code,
        )

        if len(digits) in (8, 12, 13, 14):
            matches = self.code_index.get(
                digits,
                [],
            )

            if matches:
                return matches[0]

        if code in self.metadata:
            return code

        if code.startswith(
            ("http://", "https://")
        ):
            try:
                parsed = urlparse(code)

                parts = [
                    unquote(x)
                    for x in parsed.path.split("/")
                    if x
                ]

                for part in parts:
                    part = part.strip()

                    if part in self.metadata:
                        return part

                    if "?" in part:
                        part = part.split("?")[0]

                    if part in self.metadata:
                        return part

                # Иногда slug находится в полном URL.
                for slug in self.metadata:
                    if slug in code:
                        return slug

            except Exception:
                pass

        for slug in self.metadata:
            if slug in code:
                return slug

        return None

    # --------------------------------------------------------
    # ALL CODES
    # --------------------------------------------------------

    def _scan_array(self, image: Image.Image) -> np.ndarray:
        """Downscale before zxing/QR so a 12MP frame stays cheap."""

        image = image.convert("RGB")
        width, height = image.size
        side = max(width, height)
        if side > CODE_MAX_SIDE:
            scale = CODE_MAX_SIDE / side
            image = image.resize(
                (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                ),
                Image.Resampling.BILINEAR,
            )
        return np.array(image)

    def detect_qr(self, image: Image.Image) -> list[dict]:
        image_np = self._scan_array(image)
        detector = cv2.QRCodeDetector()
        results = []

        try:
            data, _points, _ = detector.detectAndDecode(image_np)
            if data:
                results.append({
                    "type": "qr",
                    "data": data.strip(),
                })
        except Exception:
            pass

        unique = {}
        for item in results:
            unique[item["data"]] = item
        return list(unique.values())

    def detect_barcode(self, image: Image.Image) -> list[dict]:
        image_np = self._scan_array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        results = []

        for variant in (image_np, gray):
            try:
                detected = zxingcpp.read_barcodes(variant)
            except Exception as exc:
                if DEBUG:
                    print(f"Barcode warning: {exc}")
                continue

            for barcode in detected:
                text = str(barcode.text).strip()
                if not text:
                    continue
                results.append({
                    "type": "barcode",
                    "format": str(barcode.format),
                    "data": text,
                })

        unique = {}
        for item in results:
            key = (item["type"], item["data"])
            unique[key] = item
        return list(unique.values())

    def detect(
        self,
        image: Image.Image,
    ) -> list[dict]:
        """Decode codes, then resolve misses from the local cache or network.

        A cache hit never calls Open Food Facts or the QR host.
        The network budget is one short timeout for the whole photo.
        """

        codes = self.detect_qr(image) + self.detect_barcode(image)
        deadline = time.perf_counter() + EXTERNAL_TIMEOUT

        for item in codes:
            data = item["data"]
            slug = self.find_slug_by_code(data)
            item["matched_slug"] = slug
            item["match_source"] = "local" if slug else ""

            needs_external = not slug and (
                (
                    item["type"] == "qr"
                    and data.startswith(("http://", "https://"))
                )
                or item["type"] == "barcode"
            )
            if not needs_external:
                continue

            remaining = deadline - time.perf_counter()
            if remaining < 0.2:
                item["external_status"] = "skipped_deadline"
                item["product"] = {
                    "status": "error",
                    "text": "",
                    "name": "",
                    "brand": "",
                    "winery": "",
                    "gtin": "",
                    "url": data if item["type"] == "qr" else "",
                }
                continue

            if item["type"] == "qr":
                external = lookup_qr_url(
                    data,
                    self.product_cache,
                    timeout=min(EXTERNAL_TIMEOUT, remaining),
                )
            else:
                external = lookup_barcode(
                    data,
                    self.product_cache,
                    timeout=min(EXTERNAL_TIMEOUT, remaining),
                )

            item["product"] = external
            item["external_status"] = external.get("status")
            item["external_text"] = external.get("text", "")
            if external.get("gtin"):
                item["gtin"] = external.get("gtin")

        return codes


# ============================================================
# MAIN RECOGNIZER
# ============================================================

def _merge_visual(
    primary: list[dict],
    extra: list[dict],
) -> list[dict]:
    """Keep the higher SigLIP score when a slug is in both lists."""

    merged: dict[str, dict] = {}
    for item in primary + extra:
        slug = item.get("slug")
        if not slug:
            continue
        previous = merged.get(slug)
        if previous is None or item.get("visual_score", 0) > previous.get(
            "visual_score",
            0,
        ):
            merged[slug] = item
    return list(merged.values())


class HybridWineRecognizer:

    def __init__(self):
        print("Loading catalog embeddings...")

        self.embeddings = np.load(
            EMBEDDINGS_PATH
        ).astype(np.float32)

        # На всякий случай нормализуем.
        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0

        self.embeddings /= norms
        self.finetune_enabled = _finetune_enabled()
        self.finetune_proj = None
        if self.finetune_enabled and FINETUNE_PROJ_PATH.is_file():
            proj = np.load(FINETUNE_PROJ_PATH).astype(np.float32)
            if proj.ndim == 2 and proj.shape[0] == self.embeddings.shape[1]:
                self.finetune_proj = proj
                self.embeddings = _apply_proj(self.embeddings, proj)
                print(f"Fine-tune projection on: {proj.shape}")
            else:
                self.finetune_enabled = False
                print("Fine-tune file ignored: shape does not match embeddings")
        else:
            self.finetune_enabled = False

        self.reranker = None
        eval_on = os.environ.get("WINE_RERANKER_EVAL") == "1" and RERANKER_PATH.is_file()
        if eval_on or _flag_on(RERANKER_FLAG_PATH, RERANKER_PATH):
            self.reranker = load_reranker(RERANKER_PATH)
            if self.reranker is None:
                print("Reranker file ignored: unexpected shape")
            else:
                print("Learned reranker on")

        print(
            f"Catalog embeddings: "
            f"{self.embeddings.shape}"
        )

        with open(
            META_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            meta = json.load(f)

        self.meta = meta["items"]

        print(
            f"Catalog loaded: "
            f"{len(self.meta)} wines"
        )

        self.df = pd.read_csv(
            CSV_PATH
        )

        self.metadata: dict[str, dict] = {}

        for _, row in self.df.iterrows():
            slug = _cell(row.get("Slug", ""))

            if not slug:
                continue

            if slug not in self.metadata:
                self.metadata[slug] = {
                    "name": _cell(row.get("Название вина", "")),
                    "category": _cell(row.get("Категория", "")),
                    "color": _cell(row.get("Цвет", "")),
                    "region": _cell(row.get("Регион", "")),
                    "grape": _cell(row.get("Сорт винограда", "")),
                    "winery": _cell(row.get("Винодельня", "")),
                    "description": _cell(row.get("Описание", "")),
                    "image_name": _cell(row.get("Название фото", "")),
                    "taste": _first_cell(row, ("Вкус", "Дегустационные характеристики")),
                    "sugar": _first_cell(row, ("Сахар", "Содержание сахара")),
                    "food": _first_cell(row, ("Гастрономия", "Сочетание с едой")),
                }

        print(
            f"Wine metadata loaded: "
            f"{len(self.metadata)} slugs"
        )

        # ----------------------------------------------------
        # PRECOMPUTE TEXT INDEX
        # ----------------------------------------------------

        self.catalog_text: dict[str, dict] = {}
        self.grape_to_slugs: dict[str, set[str]] = defaultdict(set)
        self.name_to_slugs: dict[str, set[str]] = defaultdict(set)
        self.winery_to_slugs: dict[str, set[str]] = defaultdict(set)

        # Known grape tokens (built once; O(1) membership per OCR token).
        self.known_grape_tokens: set[str] = set()
        self.global_grape_tokens = self.known_grape_tokens

        # Частота токена по товарам — нужен IDF,
        # чтобы "ЖЕМЧУЖНАЯ" не весила так же,
        # как редкий "АЛИГОТЕ".
        self.token_df: dict[str, int] = {}

        for slug, data in self.metadata.items():

            name_tokens = set(
                text_tokens(data["name"])
            )

            grape_tokens = set(
                text_tokens(data["grape"])
            )

            winery_tokens = set(
                text_tokens(data["winery"])
            )

            region_tokens = set(
                text_tokens(data["region"])
            )

            all_tokens = (
                name_tokens
                | grape_tokens
                | winery_tokens
                | region_tokens
            )

            for token in all_tokens:
                self.token_df[token] = (
                    self.token_df.get(token, 0)
                    + 1
                )

            self.known_grape_tokens.update(
                grape_tokens
            )

            for token in grape_tokens:
                self.grape_to_slugs[token].add(slug)

            for token in name_tokens:
                self.name_to_slugs[token].add(slug)

            for token in winery_tokens:
                if len(token) >= 4 and not is_generic_token(token):
                    self.winery_to_slugs[token].add(slug)

            self.catalog_text[slug] = {
                "name": name_tokens,
                "grape": grape_tokens,
                "winery": winery_tokens,
                "region": region_tokens,
                "all": all_tokens,
            }

        self.catalog_count = max(
            len(self.metadata),
            1,
        )
        self.lexicon = OcrLexicon.from_wine_metadata(self.metadata)

        print(
            f"Indexed grape tokens: "
            f"{len(self.global_grape_tokens)}"
        )

        self.slug_to_rows: dict[str, list[int]] = defaultdict(list)

        for idx, item in enumerate(self.meta):
            slug = str(item.get("slug", "")).strip()
            if slug:
                self.slug_to_rows[slug].append(idx)

        self.catalog_files: dict[str, Path] = {}
        for item in self.meta:
            slug = str(item.get("slug", "")).strip()
            raw_file = str(item.get("file", "") or "").strip()
            if not slug or not raw_file or slug in self.catalog_files:
                continue
            path = Path(raw_file)
            if not path.is_file():
                path = ROOT / raw_file.replace("\\", "/")
            if path.is_file():
                self.catalog_files[slug] = path

        # ----------------------------------------------------
        # CODE
        # ----------------------------------------------------

        self.code_detector = CodeDetector(
            self.metadata,
            self.catalog_text,
        )

        # ----------------------------------------------------
        # SIGLIP
        # ----------------------------------------------------

        print(
            f"Loading {MODEL_NAME}..."
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                MODEL_NAME
            )
        )

        self.model = (
            AutoModel.from_pretrained(
                MODEL_NAME
            )
        )

        self.model.to(DEVICE)
        self.model.eval()

        print("SigLIP2 ready")

        # ----------------------------------------------------
        # OCR — RapidOCR stays resident; EasyOCR only if it fails
        # ----------------------------------------------------

        print("Loading RapidOCR...")
        self.ocr = FastOcr()
        print(
            f"OCR ready: {self.ocr.backend}"
        )
        print(
            f"Device: siglip={DEVICE}, ocr={self.ocr.device}",
            flush=True,
        )

    # ========================================================
    # SIGLIP
    # ========================================================

    def get_embedding(
        self,
        image: Image.Image,
    ) -> np.ndarray:

        image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output = self.model.get_image_features(
                **inputs
            )

        if isinstance(
            output,
            torch.Tensor,
        ):
            features = output

        elif hasattr(
            output,
            "pooler_output",
        ):
            features = output.pooler_output

        elif hasattr(
            output,
            "image_embeds",
        ):
            features = output.image_embeds

        elif hasattr(
            output,
            "last_hidden_state",
        ):
            features = output.last_hidden_state

            if features.ndim == 3:
                features = features.mean(
                    dim=1
                )

        else:
            raise RuntimeError(
                "Unknown SigLIP output type: "
                f"{type(output)}"
            )

        features = (
            features
            .detach()
            .cpu()
            .numpy()
        )

        if features.ndim > 1:
            features = features[0]

        norm = np.linalg.norm(features)

        if norm > 0:
            features = features / norm

        features = features.astype(np.float32)
        if self.finetune_proj is not None:
            features = _apply_proj(features.reshape(1, -1), self.finetune_proj)[0]
        return features

    def _image_url(self, slug: str) -> str:
        if slug and slug in self.catalog_files:
            return f"/catalog-image/{slug}"
        return ""

    # ========================================================
    # VISUAL SEARCH
    # ========================================================

    def _rows_for_slugs(self, slugs: list[str]) -> list[int]:
        rows: list[int] = []
        seen: set[int] = set()

        for slug in slugs:
            for idx in self.slug_to_rows.get(slug, ()):
                if idx not in seen:
                    seen.add(idx)
                    rows.append(idx)

        return rows

    def visual_search(
        self,
        image: Image.Image,
        top_k: int = VISUAL_TOP_K,
        row_indices: list[int] | None = None,
        embedding: np.ndarray | None = None,
    ) -> list[dict]:
        """SigLIP similarity.

        row_indices=None compares the full catalog (fallback).
        Otherwise only those embedding rows are scored.
        """

        if embedding is None:
            embedding = self.get_embedding(image)

        if row_indices is None:
            similarities = self.embeddings @ embedding
            chosen = np.argsort(similarities)[::-1][:top_k]
            pairs = [
                (int(idx), float(similarities[idx]))
                for idx in chosen
            ]
        else:
            if not row_indices:
                return []

            idx_arr = np.asarray(row_indices, dtype=np.int64)
            sims = self.embeddings[idx_arr] @ embedding
            k = min(int(top_k), int(idx_arr.size))
            order = np.argsort(sims)[::-1][:k]
            pairs = [
                (int(idx_arr[i]), float(sims[i]))
                for i in order
            ]

        results = []
        seen_slugs: set[str] = set()

        for idx, visual_score in pairs:
            item = self.meta[idx]
            slug = item["slug"]

            if slug in seen_slugs:
                continue

            seen_slugs.add(slug)
            data = self.metadata.get(slug, {})

            results.append({
                "slug": slug,
                "visual_score": visual_score,
                "name": data.get("name", ""),
                "category": data.get("category", ""),
                "color": data.get("color", ""),
                "region": data.get("region", ""),
                "grape": data.get("grape", ""),
                "winery": data.get("winery", ""),
                "description": data.get("description", ""),
                "image": self._image_url(slug),
            })

        return results

    def clean_ocr(self, ocr_text: str) -> list[str]:
        """Catalog lexicon hits only. Latin and Russian misses are dropped."""

        return self.lexicon.clean(ocr_text).tokens

    def shortlist_from_ocr(self, ocr_text: str):
        """Lexicon hits -> catalog slug shortlist. Raw OCR is not used."""

        return build_ocr_shortlist(
            ocr_tokens=self.clean_ocr(ocr_text),
            known_grape_tokens=self.known_grape_tokens,
            grape_to_slugs=self.grape_to_slugs,
            name_to_slugs=self.name_to_slugs,
            token_df=self.token_df,
            catalog_count=self.catalog_count,
            winery_to_slugs=self.winery_to_slugs,
        )

    # ========================================================
    # OCR
    # ========================================================

    def extract_ocr(
        self,
        image: Image.Image,
    ) -> tuple[str, dict, Image.Image]:
        """Label crop, then RapidOCR on the full crop and the varietal band."""

        prepared = prepare_label(image, OCR_MAX_SIDE)
        text = self.ocr.read(prepared["ocr"])
        prepared["debug"]["ocr_passes"] = getattr(self.ocr, "pass_mode", "")
        # Full-frame retry only when the label read is almost empty.
        if len([token for token in text_tokens(text) if len(token) >= 4]) < 3:
            width, height = image.size
            crop_h = int(height * 0.72)
            top = max(0, (height - crop_h) // 2)
            band = ImageOps.autocontrast(
                image.crop((0, top, width, min(height, top + crop_h)))
            )
            side = max(band.size)
            if side > 960:
                scale = 960 / side
                band = band.resize(
                    (
                        max(1, int(band.size[0] * scale)),
                        max(1, int(band.size[1] * scale)),
                    ),
                    Image.Resampling.BILINEAR,
                )
            extra = self.ocr.read(band)
            if len(text_tokens(extra)) > len(text_tokens(text)):
                text = extra
                prepared["debug"] = {
                    **prepared["debug"],
                    "ocr_retry": "center_autocontrast",
                }
        return text, prepared["debug"], prepared["visual"]

    def warmup(self) -> None:
        """Run one tiny OCR and SigLIP forward so the first photo is warm."""

        blank = Image.new("RGB", (224, 224), (244, 244, 244))
        self.ocr.read(blank)
        self.get_embedding(blank)

    def match_external_product(self, product: dict) -> str | None:
        """Confident catalog slug from an external name/brand/winery.

        A unique OCR shortlist wins. Otherwise a near-exact name match
        with a clear gap over the runner-up. Ambiguous text does not
        skip OCR and SigLIP.
        """

        pieces = [
            product.get("name") or "",
            product.get("brand") or "",
            product.get("winery") or "",
            product.get("text") or "",
        ]
        text = " ".join(part for part in pieces if part)
        decision = self.shortlist_from_ocr(text)
        if decision.confident and len(decision.slugs) == 1:
            return decision.slugs[0]

        name = normalize_text(product.get("name") or "")
        if len(name) < 10:
            return None

        best_slug = None
        best_score = 0.0
        second_score = 0.0
        for slug, data in self.metadata.items():
            catalog_name = normalize_text(data.get("name") or "")
            if len(catalog_name) < 6:
                continue
            score = ratio(name, catalog_name) / 100.0
            if score > best_score:
                second_score = best_score
                best_score = score
                best_slug = slug
            elif score > second_score:
                second_score = score

        if (
            best_slug
            and best_score >= 0.88
            and (best_score - second_score) >= 0.06
        ):
            return best_slug

        return None

    # ========================================================
    # OCR SCORE
    # ========================================================

    def _idf(self, token: str) -> float:
        """
        Rare token -> high weight.
        Common token -> low weight.

        Это ключевое исправление проблемы вида:
        ЖЕМЧУЖНАЯ + АРАТТИ + ЦИТРОН
        ошибочно перебивают АЛИГОТЕ.
        """
        df = self.token_df.get(
            token,
            0,
        )

        return float(
            np.log(
                (self.catalog_count + 1)
                / (df + 1)
            )
            + 1.0
        )

    def _best_token_match(
        self,
        ocr_token: str,
        candidate_tokens: set[str],
    ) -> tuple[str | None, float]:

        if not candidate_tokens:
            return None, 0.0

        best_token = None
        best_score = 0.0

        for candidate_token in candidate_tokens:

            score = ratio(
                ocr_token,
                candidate_token,
            ) / 100.0

            if score > best_score:
                best_score = score
                best_token = candidate_token

        return (
            best_token,
            best_score,
        )

    def calculate_text_score(
        self,
        ocr_text: str,
        candidate: dict,
        external_text: str = "",
        ocr_tokens: list[str] | None = None,
    ) -> dict:

        combined_ocr = " ".join(
            x
            for x in (
                ocr_text,
                external_text,
            )
            if x
        )

        if ocr_tokens is None:
            ocr_tokens = self.clean_ocr(combined_ocr)

        if not ocr_tokens:
            return {
                "score": 0.0,
                "name_score": 0.0,
                "grape_score": 0.0,
                "winery_score": 0.0,
                "region_score": 0.0,
                "matched_tokens": [],
                "negative_tokens": [],
                "strong_grape_matches": [],
            }

        slug = candidate["slug"]

        indexed = self.catalog_text.get(
            slug,
            {},
        )

        name_tokens = indexed.get(
            "name",
            set(),
        )

        grape_tokens = indexed.get(
            "grape",
            set(),
        )

        winery_tokens = indexed.get(
            "winery",
            set(),
        )

        region_tokens = indexed.get(
            "region",
            set(),
        )

        matched_tokens = []
        negative_tokens = []
        strong_grape_matches = []

        # ----------------------------------------------------
        # Positive token evidence.
        # ----------------------------------------------------

        field_hits = {
            "name": [],
            "grape": [],
            "winery": [],
            "region": [],
        }

        used_catalog_tokens = set()

        # Generic words (белое, сухое, вино, брют, …) and years
        # match almost every white wine and must not add score.
        scored_tokens: list[str] = []
        detected_grapes: list[str] = []
        seen_grapes: set[str] = set()
        resolved_by_token: dict[str, str | None] = {}

        for ocr_token in ocr_tokens:
            if is_generic_token(ocr_token):
                continue

            scored_tokens.append(ocr_token)
            resolved = _resolve_grape(
                ocr_token,
                self.known_grape_tokens,
            )
            resolved_by_token[ocr_token] = resolved

            if resolved and resolved not in seen_grapes:
                seen_grapes.add(resolved)
                detected_grapes.append(resolved)

        credited_grapes: set[str] = set()
        penalized_grapes: set[str] = set()

        for ocr_token in scored_tokens:

            resolved = resolved_by_token.get(ocr_token)

            # ЦИТРОН on the label is ЦИТРОННЫЙ in the catalog.
            # A blend that only has one of two detected grapes
            # must not collect a fuzzy name hit for that token.
            if resolved:
                if resolved in grape_tokens:
                    if resolved not in credited_grapes:
                        credited_grapes.add(resolved)
                        weight = self._idf(resolved)
                        field_hits["grape"].append(weight)
                        used_catalog_tokens.add(resolved)
                        matched_tokens.append({
                            "ocr": ocr_token,
                            "catalog": resolved,
                            "field": "grape",
                            "score": 1.0,
                            "idf": round(weight, 3),
                        })
                        strong_grape_matches.append(ocr_token)
                elif resolved not in penalized_grapes:
                    penalized_grapes.add(resolved)
                    penalty = (
                        1.0
                        if len(detected_grapes) >= 2
                        else min(
                            1.0,
                            self._idf(resolved) / 3.0,
                        )
                    )
                    negative_tokens.append({
                        "ocr": ocr_token,
                        "reason": "known_grape_missing",
                        "penalty": round(penalty, 3),
                    })
                continue

            candidates_by_field = {
                "name": name_tokens,
                "grape": grape_tokens,
                "winery": winery_tokens,
                "region": region_tokens,
            }

            best_field = None
            best_catalog = None
            best_score = 0.0

            for field, tokens in candidates_by_field.items():

                catalog_token, score = (
                    self._best_token_match(
                        ocr_token,
                        tokens,
                    )
                )

                if score > best_score:
                    best_score = score
                    best_field = field
                    best_catalog = catalog_token

            # 0.82 — нормальное fuzzy совпадение.
            if (
                best_field is not None
                and best_catalog is not None
                and best_score >= 0.82
            ):
                weight = self._idf(
                    ocr_token
                )

                field_hits[
                    best_field
                ].append(
                    best_score * weight
                )

                used_catalog_tokens.add(
                    best_catalog
                )

                matched_tokens.append({
                    "ocr": ocr_token,
                    "catalog": best_catalog,
                    "field": best_field,
                    "score": round(
                        best_score,
                        3,
                    ),
                    "idf": round(
                        weight,
                        3,
                    ),
                })

                if best_field == "grape":
                    strong_grape_matches.append(
                        ocr_token
                    )

        def aggregate(
            values: list[float],
        ) -> float:
            if not values:
                return 0.0

            # Берём не просто среднее всех слов:
            # сильные редкие совпадения должны иметь значение.
            values = sorted(
                values,
                reverse=True,
            )

            top = values[:4]

            return float(
                min(
                    1.0,
                    sum(top)
                    / max(
                        len(top),
                        1,
                    )
                    / 2.5,
                )
            )

        name_score = aggregate(
            field_hits["name"]
        )

        grape_score = aggregate(
            field_hits["grape"]
        )

        winery_score = aggregate(
            field_hits["winery"]
        )

        region_score = aggregate(
            field_hits["region"]
        )

        # ----------------------------------------------------
        # Exact grape token bonus.
        # ----------------------------------------------------

        exact_grape_matches = sum(
            1
            for grape in detected_grapes
            if grape in grape_tokens
        )

        grape_bonus = min(
            0.35,
            exact_grape_matches * 0.18,
        )

        # ----------------------------------------------------
        # Negative score.
        # ----------------------------------------------------

        negative_penalty = 0.0

        for item in negative_tokens:
            negative_penalty += (
                item["penalty"]
                * 0.20
            )

        negative_penalty = min(
            0.45,
            negative_penalty,
        )

        # Two distinctive grapes on the label: covering both is
        # worth more than a blend that only shares the common one.
        missing_grapes = [
            grape
            for grape in detected_grapes
            if grape not in grape_tokens
        ]

        if len(detected_grapes) >= 2 and not missing_grapes:
            grape_bonus = min(
                0.55,
                grape_bonus + 0.30,
            )
        elif missing_grapes and len(detected_grapes) >= 2:
            negative_penalty = min(
                0.90,
                negative_penalty + 0.55 * len(missing_grapes),
            )

        # ----------------------------------------------------
        # Final OCR score.
        # ----------------------------------------------------
        #
        # Сорт винограда — главный текстовый сигнал.
        # Название — второй.
        # Винодельня — слабее.
        # Регион — ещё слабее.
        # ----------------------------------------------------

        score = (
            grape_score * 0.58
            + name_score * 0.27
            + winery_score * 0.10
            + region_score * 0.05
            + grape_bonus
            - negative_penalty
        )

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        return {
            "score": float(score),
            "name_score": float(name_score),
            "grape_score": float(grape_score),
            "winery_score": float(winery_score),
            "region_score": float(region_score),
            "matched_tokens": matched_tokens,
            "negative_tokens": negative_tokens,
            "strong_grape_matches": strong_grape_matches,
        }

    # ========================================================
    # CODE CANDIDATE
    # ========================================================

    def _result_from_slug(
        self,
        slug: str,
        method: str,
        codes: list[dict],
        ocr_text: str = "",
        external_text: str = "",
        timing: dict | None = None,
    ) -> dict:

        data = self.metadata.get(
            slug,
            {},
        )

        wine = {
            "name": data.get(
                "name",
                "",
            ),
            "category": data.get(
                "category",
                "",
            ),
            "color": data.get(
                "color",
                "",
            ),
            "region": data.get(
                "region",
                "",
            ),
            "grape": data.get(
                "grape",
                "",
            ),
            "winery": data.get(
                "winery",
                "",
            ),
            "description": data.get(
                "description",
                "",
            ),
            "image": self._image_url(slug),
        }

        return {
            "method": method,
            "status": "found",
            "slug": slug,
            "wine": wine,
            "codes": codes,
            "ocr_text": ocr_text,
            "external_text": external_text,
            "visual_compared": 0,
            # В обычном режиме только один вариант.
            "candidates": [
                {
                    "slug": slug,
                    **wine,
                }
            ],
            "timing": timing or {},
        }

    # ========================================================
    # HYBRID RECOGNITION
    # ========================================================

    def recognize(
        self,
        image: Image.Image,
        debug: bool = False,
    ) -> dict:
        result = self._recognize_pipeline(image, debug=debug)
        trace = result.get("_trace")
        if trace:
            try:
                _print_recognition_log(trace)
            except Exception:
                pass
        return result

    def _recognize_pipeline(
        self,
        image: Image.Image,
        debug: bool = False,
    ) -> dict:

        total_start = time.perf_counter()
        image = ImageOps.exif_transpose(image) or image
        image = image.convert("RGB")

        # ----------------------------------------------------
        # STEP 1 — QR / BARCODE, then external resolve
        # ----------------------------------------------------

        code_start = time.perf_counter()

        detected_codes = self.code_detector.detect(image)

        winning_code = None
        if CODE_EXACT_MATCH:
            for code in detected_codes:
                if code.get("match_source") == "local" and code.get("matched_slug"):
                    winning_code = code
                    break

            if winning_code is None:
                for code in detected_codes:
                    product = code.get("product") or {}
                    if product.get("status") != "found":
                        continue
                    slug = self.match_external_product(product)
                    if not slug:
                        continue
                    code["matched_slug"] = slug
                    code["match_source"] = "external"
                    winning_code = code
                    break

        code_time = time.perf_counter() - code_start

        external_parts = []
        for code in detected_codes:
            text = (code.get("product") or {}).get("text") or code.get("external_text") or ""
            if text:
                external_parts.append(text)
        external_text = " ".join(external_parts)

        if winning_code is not None:
            total_time = time.perf_counter() - total_start
            method = "qr" if winning_code.get("type") == "qr" else "barcode"
            payload = self._result_from_slug(
                slug=winning_code["matched_slug"],
                method=method,
                codes=detected_codes,
                external_text=external_text,
                timing={
                    "codes": round(code_time, 3),
                    "visual": 0.0,
                    "ocr": 0.0,
                    "total": round(total_time, 3),
                },
            )
            wine = payload.get("wine") or {}
            payload["_trace"] = build_recognition_trace(
                search_skipped="совпадение по коду",
                neural_skipped="совпадение по коду, нейросеть не запускалась",
                catalog_count=self.catalog_count,
                winner_slug=payload.get("slug") or "",
                winner_name=wine.get("name") or "",
                status="found",
            )
            return payload

        # ----------------------------------------------------
        # STEP 2 — dominant crop + one OCR pass, then shortlist
        # ----------------------------------------------------

        ocr_start = time.perf_counter()

        ocr_text, crop_debug, visual_image = self.extract_ocr(
            image
        )

        ocr_time = (
            time.perf_counter()
            - ocr_start
        )

        cyr_text = getattr(self.ocr, "last_cyrillic", "") or ""
        en_text = getattr(self.ocr, "last_english", "") or ""
        if cyr_text or en_text:
            cyr_cleanup = self.lexicon.clean(cyr_text)
            en_cleanup = self.lexicon.clean(en_text) if en_text else cyr_cleanup
            merged_tokens = merge_compatible_tokens(
                cyr_cleanup.tokens,
                [] if en_text == "" else en_cleanup.tokens,
                self.grape_to_slugs,
                self.name_to_slugs,
                self.winery_to_slugs,
            )
            dropped = list(cyr_cleanup.dropped)
            if en_text:
                dropped.extend(en_cleanup.dropped)
            readable = list(cyr_cleanup.readable)
            for token in en_cleanup.readable if en_text else []:
                if token not in readable:
                    readable.append(token)
            ocr_cleanup = cyr_cleanup
            ocr_cleanup.tokens = merged_tokens
            ocr_cleanup.cleaned_text = " ".join(merged_tokens)
            ocr_cleanup.dropped = dropped
            ocr_cleanup.readable = [
                token for token in readable if token not in set(merged_tokens)
            ]
            kept_catalog = {hit.catalog for hit in cyr_cleanup.hits}
            if en_text:
                for hit in en_cleanup.hits:
                    if hit.catalog in merged_tokens and hit.catalog not in kept_catalog:
                        ocr_cleanup.hits.append(hit)
                        kept_catalog.add(hit.catalog)
            ocr_cleanup.hits = [
                hit for hit in ocr_cleanup.hits if hit.catalog in set(merged_tokens)
            ]
        else:
            ocr_cleanup = self.lexicon.clean(ocr_text)
        clean_tokens = recover_names_inside_producer(
            ocr_cleanup.tokens,
            [item.get("raw") or "" for item in ocr_cleanup.dropped],
            name_to_slugs=self.name_to_slugs,
            winery_to_slugs=self.winery_to_slugs,
            token_df=self.token_df,
        )
        if external_text:
            score_tokens = self.clean_ocr(
                " ".join(part for part in (ocr_text, external_text) if part)
            )
        else:
            score_tokens = clean_tokens
        ocr_clean = " ".join(clean_tokens)

        decision = build_ocr_shortlist(
            ocr_tokens=clean_tokens,
            known_grape_tokens=self.known_grape_tokens,
            grape_to_slugs=self.grape_to_slugs,
            name_to_slugs=self.name_to_slugs,
            token_df=self.token_df,
            catalog_count=self.catalog_count,
            winery_to_slugs=self.winery_to_slugs,
        )

        slugs = [
            slug
            for slug in decision.slugs
            if slug in self.slug_to_rows
        ]

        # Two strong dictionary hits may hide the rest of the catalog
        # (grape+grape, or grape/winery plus a distinctive name).
        # One recovered token only re-ranks: SigLIP still sees the
        # full catalog, so a weak token cannot erase the visual neighbor.
        exclusive = bool(
            decision.confident
            and slugs
            and decision.strong
            and not decision.soft
        )

        shortlist_reason = decision.reason

        if decision.confident and not slugs:
            shortlist_reason = (
                f"{decision.reason}:no_embeddings"
            )
        elif decision.confident and slugs and not exclusive:
            shortlist_reason = f"{decision.reason}:soft"

        # ----------------------------------------------------
        # STEP 3 — VISUAL on the shortlist, or full catalog
        # ----------------------------------------------------

        visual_start = time.perf_counter()
        visual_compared = 0

        if exclusive:
            rows = self._rows_for_slugs(slugs)
            visual_compared = len({
                self.meta[idx]["slug"]
                for idx in rows
            })
            top_k = (
                VISUAL_TOP_K
                if decision.visual_trim
                else len(rows)
            )
            visual_candidates = self.visual_search(
                visual_image,
                top_k=top_k,
                row_indices=rows,
            )

            if not visual_candidates:
                exclusive = False
                shortlist_reason = (
                    f"{shortlist_reason}:empty_visual"
                )
                visual_candidates = self.visual_search(
                    visual_image,
                    VISUAL_TOP_K,
                )
                visual_scope = "full_catalog"
                visual_compared = len(self.metadata)
            else:
                visual_scope = "filtered"
        else:
            embedding = self.get_embedding(visual_image)
            visual_candidates = self.visual_search(
                visual_image,
                VISUAL_TOP_K,
                embedding=embedding,
            )
            if slugs:
                rows = self._rows_for_slugs(slugs)
                hinted = self.visual_search(
                    visual_image,
                    top_k=min(12, len(rows)),
                    row_indices=rows,
                    embedding=embedding,
                )
                visual_candidates = _merge_visual(
                    visual_candidates,
                    hinted,
                )
                visual_scope = "visual_union"
            else:
                visual_scope = "full_catalog"
            visual_compared = len(self.metadata)

        use_filter = exclusive

        visual_time = (
            time.perf_counter()
            - visual_start
        )

        shortlist_size = (
            len(slugs) if decision.confident and slugs else 0
        )

        # ----------------------------------------------------
        # STEP 4 — RERANK
        # ----------------------------------------------------

        final_candidates = []

        # OCR weighting depends on whether we actually got
        # useful text.
        has_ocr = bool(clean_tokens)
        lexicon_hits = self.lexicon.clean(
            " ".join(part for part in (ocr_text, external_text) if part)
        ).hits
        winery_tokens = [
            hit.catalog for hit in lexicon_hits if hit.kind == "winery"
        ]
        text_query = None
        if self.reranker is not None and ocr_clean:
            try:
                self.ensure_text_embeddings()
                text_query = self.encode_texts([ocr_clean])[0]
            except Exception as exc:
                print(f"reranker text encode skipped: {exc}")
                text_query = None

        for candidate in visual_candidates:

            text_result = (
                self.calculate_text_score(
                    ocr_text,
                    candidate,
                    external_text,
                    ocr_tokens=score_tokens,
                )
            )

            visual_score = safe_float(
                candidate["visual_score"]
            )

            text_score = safe_float(
                text_result["score"]
            )

            distinctive_hits = (
                len(decision.grape_tokens)
                + len(decision.name_tokens)
            )

            if self.reranker is not None:
                text_cosine = 0.0
                if text_query is not None:
                    row = getattr(self, "_text_slug_index", {}).get(candidate["slug"])
                    if row is not None:
                        text_cosine = float(np.dot(
                            text_query,
                            self._text_embeddings[row],
                        ))
                final_score = self.reranker.score(candidate_features(
                    visual=visual_score,
                    grape_tokens=list(decision.grape_tokens),
                    name_tokens=list(decision.name_tokens),
                    winery_tokens=winery_tokens,
                    grape=candidate.get("grape") or "",
                    name=candidate.get("name") or "",
                    winery=candidate.get("winery") or "",
                    text_cosine=text_cosine,
                ))
            elif (
                use_filter
                and has_ocr
                and decision.confident
                and distinctive_hits >= 2
            ):
                # Two label tokens already narrowed the catalog.
                # Do not let a lookalike bottle outrank that text.
                text_weight, visual_weight = (
                    text_visual_weights(distinctive_hits)
                )
                final_score = (
                    visual_score * visual_weight
                    + text_score * text_weight
                )
            elif use_filter and has_ocr and decision.confident:
                final_score = (
                    visual_score * 0.65
                    + text_score * 0.35
                )
                if text_result["matched_tokens"]:
                    final_score += (
                        text_result["score"] * 0.10
                    )
            elif has_ocr and decision.confident:
                # One dictionary token re-ranks the open catalog.
                # Visual stays the larger share so it cannot be erased.
                final_score = (
                    visual_score * 0.65
                    + text_score * 0.35
                )
                if text_result["matched_tokens"]:
                    final_score += text_result["score"] * 0.10
            else:
                # Empty OCR, or text that matched no catalog token.
                # On the labeled photos a 0.10 hit on "винодельня"
                # outranked the visually closer Победа. That stray
                # score is not a ranking signal.
                final_score = visual_score

            final_score = max(
                0.0,
                min(
                    1.0,
                    final_score,
                ),
            )

            final_candidates.append({
                **candidate,
                "ocr_score": round(
                    text_score,
                    4,
                ),
                "name_score": round(
                    text_result[
                        "name_score"
                    ],
                    4,
                ),
                "grape_score": round(
                    text_result[
                        "grape_score"
                    ],
                    4,
                ),
                "winery_score": round(
                    text_result[
                        "winery_score"
                    ],
                    4,
                ),
                "region_score": round(
                    text_result[
                        "region_score"
                    ],
                    4,
                ),
                "matched_tokens":
                    text_result[
                        "matched_tokens"
                    ],
                "negative_tokens":
                    text_result[
                        "negative_tokens"
                    ],
                "strong_grape_matches":
                    text_result[
                        "strong_grape_matches"
                    ],
                "final_score": round(
                    final_score,
                    4,
                ),
            })

        allow_rerank = has_second_distinctive(
            list(decision.grape_tokens),
            list(decision.name_tokens),
            decision.winery_token,
        )
        # No second title: the logistic head must not lift a lookalike
        # (Зелёное вино 0.97) over a closer label from the same house.
        if not allow_rerank:
            for item in final_candidates:
                item["final_score"] = round(
                    float(item.get("visual_score") or 0.0),
                    4,
                )
        final_candidates = order_candidates(
            final_candidates,
            allow_rerank=allow_rerank,
            years=extract_years(ocr_text) if decision.winery_token else [],
        )

        # The logistic head can squash every neighbor to ~0.01. That
        # must not let a different winery outrank a clearly closer
        # label. Inside one house the head may still reorder.
        if self.reranker is not None and len(final_candidates) >= 2:
            def _house(item: dict) -> str:
                return str(item.get("winery") or "")

            leader = final_candidates[0]
            closest = max(
                final_candidates,
                key=lambda item: float(item.get("visual_score") or 0.0),
            )
            visual_lead = float(closest.get("visual_score") or 0.0) - float(
                leader.get("visual_score") or 0.0
            )
            if (
                closest is not leader
                and _house(closest)
                and _house(closest) != _house(leader)
                and visual_lead >= 0.05
            ):
                final_candidates.remove(closest)
                final_candidates.insert(0, closest)

        # A tiny text bump must not bury a clearly closer label.
        if self.reranker is None and not use_filter and len(final_candidates) >= 2:
            leader = final_candidates[0]
            closest = max(
                final_candidates,
                key=lambda item: item.get("visual_score") or 0.0,
            )
            visual_lead = (closest.get("visual_score") or 0.0) - (
                leader.get("visual_score") or 0.0
            )
            final_lead = (leader.get("final_score") or 0.0) - (
                closest.get("final_score") or 0.0
            )
            if closest is not leader and visual_lead >= 0.04 and final_lead <= 0.02:
                final_candidates.remove(closest)
                final_candidates.insert(0, closest)

        total_time = (
            time.perf_counter()
            - total_start
        )

        timing = {
            "codes": round(
                code_time,
                3,
            ),
            "preprocess": round(
                float(crop_debug.get("preprocess_ms") or 0.0) / 1000.0,
                3,
            ),
            "visual": round(
                visual_time,
                3,
            ),
            "ocr": round(
                ocr_time,
                3,
            ),
            "total": round(
                total_time,
                3,
            ),
        }

        debug_block = {
            "ocr_text": ocr_text,
            "ocr_normalized": ocr_clean,
            "shortlist_size": shortlist_size,
            "visual_scope": visual_scope,
            "shortlist_reason": shortlist_reason,
            "visual_trim": decision.visual_trim,
            "ocr_grape_tokens": list(
                decision.grape_tokens
            ),
            "ocr_name_tokens": list(
                decision.name_tokens
            ),
            "visual_compared": visual_compared,
            "crop": crop_debug,
            "ocr_backend": self.ocr.backend,
            "candidates": final_candidates[:10],
        }

        crop_w, crop_h = visual_image.size
        trace = build_recognition_trace(
            ocr_text=ocr_text,
            ocr_normalized=ocr_clean,
            kept=[hit.as_dict() for hit in ocr_cleanup.hits],
            dropped=list(ocr_cleanup.dropped),
            crop=crop_debug,
            crop_px=(int(crop_w), int(crop_h)),
            embed_dim=int(self.embeddings.shape[1]),
            visual_scope=visual_scope,
            shortlist_size=shortlist_size,
            shortlist_reason=shortlist_reason,
            visual_compared=visual_compared,
            catalog_count=self.catalog_count,
            reranker_text=ocr_clean,
            grape_tokens=list(decision.grape_tokens),
            name_tokens=list(decision.name_tokens),
            winery_tokens=winery_tokens,
            reranker_used=self.reranker is not None,
            text_encoded=text_query is not None,
            neighbors=final_candidates,
        )

        if not final_candidates:
            return self._not_in_catalog(
                method="none",
                ocr_text=ocr_text,
                ocr_clean=ocr_clean,
                external_text=external_text,
                visual_compared=visual_compared,
                timing=timing,
                codes=detected_codes,
                confidence=0.0,
                debug=debug,
                debug_block=debug_block,
                trace=trace,
            )

        winner = final_candidates[0]
        runner_up = (
            final_candidates[1]
            if len(final_candidates) > 1
            else None
        )

        if use_filter:
            method = "ocr_visual"
        elif has_ocr:
            method = "visual_ocr"
        else:
            method = "visual"

        generic_only = signals_are_generic_only(
            score_tokens,
            winery=decision.winery_token,
            names=list(decision.name_tokens),
            grapes=list(decision.grape_tokens),
            flagged=bool(decision.generic_only),
        )
        # A house name with no second distinctive token. One rare title
        # (АРАТТИ + БЕЛАЯ) is not this case: the shortlist is a single wine.
        # A single vintage on the winner (Каберне 2021) is that second token.
        years = extract_years(ocr_text) if decision.winery_token else []
        winner_blob = f"{winner.get('slug') or ''} {winner.get('name') or ''}"
        vintage_hits = 0
        if years:
            for item in final_candidates:
                blob = f"{item.get('slug') or ''} {item.get('name') or ''}"
                if any(year in blob for year in years):
                    vintage_hits += 1
        vintage_lock = (
            vintage_hits == 1
            and any(year in winner_blob for year in years)
        )
        producer_only = (
            bool(decision.winery_token)
            and shortlist_size > 1
            and not vintage_lock
            and not has_second_distinctive(
                list(decision.grape_tokens),
                list(decision.name_tokens),
                decision.winery_token,
            )
        )
        # Two rows of one wine must not look like a flat house-only top.
        if producer_only:
            runner_up = distinct_runner(final_candidates)
        if decision.grape_conflict:
            found = False
        else:
            found = decide_match(
                winner,
                runner_up,
                ocr_text,
                method=method,
                exclusive=use_filter,
                shortlist_size=shortlist_size,
                known_grape_tokens=self.known_grape_tokens,
                dictionary_tokens=len(score_tokens),
                generic_only=generic_only,
                producer_only=producer_only,
            )
        confidence = safe_float(
            winner.get(
                "final_score",
                winner.get("visual_score", 0.0),
            )
        )
        if not found:
            house_similar = None
            if producer_only and final_candidates:
                house_similar = build_similar(
                    final_candidates,
                    ocr_text,
                    self.known_grape_tokens,
                )
            return self._not_in_catalog(
                method=method,
                ocr_text=ocr_text,
                ocr_clean=ocr_clean,
                external_text=external_text,
                visual_compared=visual_compared,
                timing=timing,
                codes=detected_codes,
                confidence=confidence,
                debug=debug,
                debug_block=debug_block,
                trace=trace,
                pinned_similar=house_similar,
            )

        payload = {
            "method": method,
            "status": "found",
            "slug": winner["slug"],
            "wine": winner,
            "similar": [],
            "confidence": round(confidence, 4),
            "codes": detected_codes,
            "ocr_text": ocr_text,
            "ocr_clean": ocr_clean,
            "external_text": external_text,
            "visual_compared": visual_compared,
            "timing": timing,
        }

        if debug:
            payload["_debug"] = debug_block

        if use_filter:
            trace["search"] = search_trace(None, skipped="вино есть в каталоге")
        else:
            trace["search"] = search_trace(
                None,
                skipped="словарный набор не сузил каталог",
            )
        trace["winner"] = {
            "slug": winner.get("slug") or "",
            "name": winner.get("name") or "",
            "status": "found",
        }
        payload["_trace"] = trace

        return payload

    def _catalog_cards(self) -> list[dict]:
        cached = getattr(self, "_catalog_cards_cache", None)
        if cached is not None:
            return cached
        cards = []
        for slug, data in self.metadata.items():
            cards.append({
                "slug": slug,
                "name": data.get("name", ""),
                "category": data.get("category", ""),
                "color": data.get("color", ""),
                "region": data.get("region", ""),
                "grape": data.get("grape", ""),
                "winery": data.get("winery", ""),
                "description": data.get("description", ""),
                "taste": data.get("taste", ""),
                "sugar": data.get("sugar", ""),
                "food": data.get("food", ""),
                "image": self._image_url(slug),
            })
        self._catalog_cards_cache = cards
        return cards

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """One SigLIP text-tower pass. Vectors are L2-normalized."""

        cleaned = [text if str(text).strip() else "вино" for text in texts]
        inputs = self.processor(
            text=cleaned,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
            if torch.is_tensor(value)
        }
        with torch.inference_mode():
            output = self.model.get_text_features(**inputs)
        if hasattr(output, "pooler_output"):
            features = output.pooler_output
        elif hasattr(output, "text_embeds"):
            features = output.text_embeds
        else:
            features = output
        features = features.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return features / norms

    def ensure_text_embeddings(self) -> np.ndarray | None:
        """Load catalog text vectors, or build them once and store on disk."""

        cached = getattr(self, "_text_embeddings", None)
        if cached is not None:
            return cached
        cards = self._catalog_cards()
        slugs = [card["slug"] for card in cards]
        if TEXT_EMB_PATH.is_file() and TEXT_SLUGS_PATH.is_file():
            stored = json.loads(TEXT_SLUGS_PATH.read_text(encoding="utf-8"))
            matrix = np.load(TEXT_EMB_PATH).astype(np.float32)
            if stored == slugs and len(matrix) == len(slugs):
                # catalog_blob already folds description into each row, so the
                # file on disk is the pairing index. Do not rebuild it.
                self._text_embeddings = matrix
                self._text_slug_index = {slug: i for i, slug in enumerate(slugs)}
                return matrix
        blobs = [catalog_blob(card) or card["name"] or card["slug"] for card in cards]
        rows: list[np.ndarray] = []
        batch = 32
        for start in range(0, len(blobs), batch):
            rows.append(self.encode_texts(blobs[start:start + batch]))
        matrix = np.concatenate(rows, axis=0).astype(np.float32)
        TEXT_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(TEXT_EMB_PATH, matrix)
        TEXT_SLUGS_PATH.write_text(
            json.dumps(slugs, ensure_ascii=False),
            encoding="utf-8",
        )
        self._text_embeddings = matrix
        self._text_slug_index = {slug: i for i, slug in enumerate(slugs)}
        return matrix

    def recommend_text(self, text: str) -> list[dict]:
        """One SigLIP text-tower encode, then cosine over catalog text rows.

        Does not run OCR or the visual recognizer. Food pairing comes from
        description text already stored in those rows.
        """

        matrix = self.ensure_text_embeddings()
        if matrix is None:
            return []
        query = self.encode_texts([text])[0]
        return rank_catalog_for_sentence(
            self._catalog_cards(),
            query,
            matrix,
            text,
            limit=4,
            known_grape_tokens=self.known_grape_tokens,
        )

    def _ocr_profile(self, ocr_clean: str, ocr_text: str) -> dict:
        blob = " ".join(part for part in (ocr_clean, ocr_text) if part).strip()
        signals = extract_query_signals(blob, self.known_grape_tokens)
        color = {
            "white": "белое",
            "red": "красное",
            "rose": "розовое",
            "orange": "оранжевое",
        }.get(signals.color or "", "")
        return {
            "query": blob[:300],
            "name": "",
            "grape": ", ".join(signals.grapes),
            "color": color,
            "taste": "",
            "sweetness": signals.sweetness or "",
            "region": "",
            "winery": "",
            "source": "",
            "text": blob,
            "status": "ocr",
            "from_cache": False,
        }

    def _rank_similar(self, profile: dict) -> list[dict]:
        cards = self._catalog_cards()
        ranked = rank_catalog_by_attributes(
            cards,
            profile,
            self.known_grape_tokens,
        )
        if ranked:
            return ranked
        matrix = None
        try:
            matrix = self.ensure_text_embeddings()
        except Exception as exc:
            print(f"text embeddings unavailable: {exc}")
        if matrix is not None and profile_blob(profile):
            query = self.encode_texts([profile_blob(profile)])[0]
            return rank_catalog_by_vectors(
                cards,
                query,
                matrix,
                known_grape_tokens=self.known_grape_tokens,
                profile=profile,
            )
        return []

    def _not_in_catalog(
        self,
        *,
        method: str,
        ocr_text: str,
        ocr_clean: str,
        external_text: str,
        visual_compared: int,
        timing: dict,
        codes: list,
        confidence: float,
        debug: bool,
        debug_block: dict,
        trace: dict | None = None,
        pinned_similar: list | None = None,
    ) -> dict:
        """Web profile, then attribute neighbors in our catalog.

        Not called for a confident in-catalog hit, so that path does
        not wait on the network. Visual neighbors are not used here.
        """

        web_start = time.perf_counter()
        lexicon_blob = " ".join(
            part for part in (ocr_text, external_text) if part
        )
        cleanup = self.lexicon.clean(lexicon_blob)
        recovered = [hit.as_dict() for hit in cleanup.hits]
        for token in cleanup.readable:
            recovered.append({
                "catalog": token,
                "raw": token,
                "type": "extra",
            })
        query = build_web_query(ocr_text, recovered)
        external = None
        similar: list[dict] = []
        ocr_profile = (
            profile_from_query(query, recovered, self.known_grape_tokens)
            if query
            else None
        )
        if not query:
            external = {
                "query": "",
                "name": "",
                "grape": "",
                "color": "",
                "taste": "",
                "sweetness": "",
                "region": "",
                "winery": "",
                "source": "",
                "text": "",
                "status": "no_lexicon",
                "from_cache": False,
            }
        elif query:
            try:
                external = lookup_wine_query(
                    query,
                    self.code_detector.product_cache,
                    timeout=WEB_DEADLINE,
                    known_grape_tokens=self.known_grape_tokens,
                )
            except Exception as exc:
                external = {
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
                    "status": "error",
                    "error": str(exc)[:300],
                    "from_cache": False,
                }
        if external and external.get("status") == "found":
            profile = fill_profile_gaps(external, ocr_profile)
        else:
            profile = ocr_profile
        if pinned_similar:
            similar = list(pinned_similar)
        elif profile and profile_blob(profile):
            try:
                similar = self._rank_similar(profile)
            except Exception as exc:
                print(f"similar rank failed: {exc}")
                similar = []
        if similar:
            message = ""
        elif not query:
            message = "С этикетки не получилось ни одного слова из каталога."
        else:
            message = "В нашем каталоге нет близкого вина по вкусу и сорту."
        web_time = time.perf_counter() - web_start
        timing = dict(timing)
        timing["web"] = round(web_time, 3)
        timing["total"] = round(float(timing.get("total") or 0.0) + web_time, 3)

        payload = {
            "method": method,
            "status": "not_in_catalog",
            "slug": None,
            "wine": None,
            "similar": similar,
            "external": external,
            "message": message,
            "confidence": round(float(confidence or 0.0), 4),
            "codes": codes,
            "ocr_text": ocr_text,
            "ocr_clean": ocr_clean,
            "external_text": external_text,
            "visual_compared": visual_compared,
            "candidates": [],
            "timing": timing,
        }
        if debug:
            payload["_debug"] = debug_block
        if trace is not None:
            traced = dict(trace)
            traced["search"] = search_trace(external, query=query)
            traced["winner"] = {
                "slug": "",
                "name": "",
                "status": "not_in_catalog",
            }
            payload["_trace"] = traced
        return payload


# ============================================================
# PUBLIC JSON
# ============================================================

def _crop_note(crop: dict | None) -> str:
    if not crop:
        return ""
    parts = [str(crop.get("strategy") or "crop")]
    box = crop.get("box")
    if box:
        parts.append(f"box {box}")
    deskew = crop.get("deskew_deg")
    if deskew:
        parts.append(f"deskew {deskew}°")
    retry = crop.get("ocr_retry")
    if retry:
        parts.append(str(retry))
    passes = crop.get("ocr_passes")
    if passes:
        parts.append(str(passes))
    return ", ".join(parts)


def _scope_label(scope: str) -> str:
    return {
        "full_catalog": "весь каталог",
        "filtered": "короткий список",
        "visual_union": "весь каталог и короткий список",
    }.get(scope or "", scope or "")


def search_trace(
    external: dict | None,
    *,
    skipped: str = "",
    query: str = "",
) -> dict:
    """Where the web query went, without the page body."""

    if skipped:
        return {
            "query": "",
            "where": "не вызывался",
            "reason": skipped,
            "result": None,
        }
    external = external or {}
    status = str(external.get("status") or "")
    query = str(query or external.get("query") or "")
    if status == "no_lexicon" or not query:
        return {
            "query": query,
            "where": "не вызывался",
            "reason": "нет словарных слов",
            "result": None,
        }
    where = "кэш" if external.get("from_cache") else (
        str(external.get("where") or "") or "DuckDuckGo"
    )
    if status == "error":
        result = {"error": str(external.get("error") or "ошибка")[:300]}
        reason = "сеть не ответила"
    elif status == "found":
        result = search_result_view(external)
        reason = lookup_reason(external)
    else:
        result = None
        reason = "ничего ясного не нашлось"
    return {
        "query": query,
        "where": where,
        "reason": reason,
        "result": result,
    }


def build_recognition_trace(
    *,
    ocr_text: str = "",
    ocr_normalized: str = "",
    kept: list | None = None,
    dropped: list | None = None,
    search_skipped: str = "",
    crop: dict | None = None,
    crop_px: tuple[int, int] | None = None,
    embed_dim: int = 0,
    visual_scope: str = "",
    shortlist_size: int = 0,
    shortlist_reason: str = "",
    visual_compared: int = 0,
    catalog_count: int = 0,
    reranker_text: str = "",
    grape_tokens: list | None = None,
    name_tokens: list | None = None,
    winery_tokens: list | None = None,
    reranker_used: bool = False,
    text_encoded: bool = False,
    neighbors: list | None = None,
    neural_skipped: str = "",
    winner_slug: str = "",
    winner_name: str = "",
    status: str = "",
) -> dict:
    """Compact pipeline log. Ranking is unchanged; this only records it."""

    nearest = []
    for item in (neighbors or [])[:5]:
        nearest.append({
            "slug": item.get("slug") or "",
            "name": item.get("name") or "",
            "visual": item.get("visual_score"),
            "final": item.get("final_score"),
        })
    return {
        "ocr": ocr_text or "",
        "lexicon": {
            "ocr_normalized": ocr_normalized or "",
            "kept": list(kept or []),
            "dropped": list(dropped or []),
        },
        "search": search_trace(None, skipped=search_skipped or "ещё не искали"),
        "neural": {
            "crop": _crop_note(crop),
            "crop_px": [int(crop_px[0]), int(crop_px[1])] if crop_px else None,
            "dim": int(embed_dim or 0),
            "scope": _scope_label(visual_scope),
            "catalog": int(catalog_count or 0),
            "shortlist": int(shortlist_size or 0),
            "shortlist_reason": shortlist_reason or "",
            "compared": int(visual_compared or 0),
            "text": reranker_text or "",
            "features": {
                "grape": list(grape_tokens or []),
                "name": list(name_tokens or []),
                "winery": list(winery_tokens or []),
                "reranker": bool(reranker_used),
                "text_encoded": bool(text_encoded),
            },
            "skipped": neural_skipped or "",
        },
        "neighbors": nearest,
        "winner": {
            "slug": winner_slug or "",
            "name": winner_name or "",
            "status": status or "",
        },
    }


def _emit_console(lines: list[str]) -> None:
    """Print Russian text even when the Windows console is cp1251."""

    text = "\n".join(lines)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "cp1251"
        data = (text + "\n").encode(encoding, errors="replace")
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            buffer.write(data)
            buffer.flush()


def _print_recognition_log(trace: dict) -> None:
    """Readable stdout block. The API response stays JSON without this."""

    lines = ["--- распознавание ---"]
    neural = trace.get("neural") or {}
    winner = trace.get("winner") or {}
    if neural.get("skipped"):
        name = winner.get("name") or winner.get("slug") or ""
        lines.append(f"Код: {name}" if name else "Код: найдено")
        lines.append("OCR: не читали")
        lines.append("Поиск: не вызывался (вино найдено в каталоге)")
        lines.append(f"Нейросеть: {neural['skipped']}")
        lines.append("---")
        _emit_console(lines)
        return

    ocr = " ".join(str(trace.get("ocr") or "").split()) or "(пусто)"
    if len(ocr) > 400:
        ocr = ocr[:400] + "..."
    lines.append(f"OCR: {ocr}")

    lexicon = trace.get("lexicon") or {}
    kept: list[str] = []
    for item in lexicon.get("kept") or []:
        if isinstance(item, dict):
            token = item.get("catalog") or item.get("raw") or ""
        else:
            token = str(item)
        if token:
            kept.append(str(token))
    lines.append(
        "После словаря: " + (", ".join(kept) if kept else "(нет слов каталога)")
    )

    dropped: list[str] = []
    for item in lexicon.get("dropped") or []:
        if isinstance(item, dict):
            if item.get("reason") == "повтор":
                continue
            raw = str(item.get("raw") or "")
        else:
            raw = str(item)
        if raw:
            dropped.append(raw)
    if dropped:
        shown = ", ".join(dropped[:12])
        if len(dropped) > 12:
            shown += f" (+{len(dropped) - 12})"
        lines.append(f"Отброшено: {shown}")

    search = trace.get("search") or {}
    where = str(search.get("where") or "")
    reason = str(search.get("reason") or "")
    if where == "не вызывался":
        if "словар" in reason:
            lines.append("Поиск: не вызывался (нет чистых слов)")
        else:
            lines.append("Поиск: не вызывался (вино найдено в каталоге)")
    else:
        host = urlparse(SEARCH_URL).netloc or "html.duckduckgo.com"
        place = f"lookup_wine_query → {host}"
        if where == "кэш":
            place += ", из кэша"
        lines.append(f"Поиск: {place}")
        lines.append(f"  запрос: «{search.get('query') or ''}»")
        result = search.get("result") or {}
        if reason == "сеть не ответила":
            lines.append("  ответ: ошибка сети")
        elif any(result.get(key) for key in ("name", "grape", "color", "sweetness", "taste")):
            bits = []
            if result.get("name"):
                bits.append(str(result["name"])[:80])
            if result.get("grape"):
                bits.append("сорт " + str(result["grape"]))
            if result.get("color"):
                bits.append(str(result["color"]))
            if result.get("sweetness"):
                bits.append(str(result["sweetness"]))
            if result.get("taste"):
                bits.append("вкус " + str(result["taste"])[:80])
            lines.append("  ответ: " + "; ".join(bits))
        else:
            lines.append("  ответ: ничего")

    features = neural.get("features") or {}
    rerank = "реранкер включён" if features.get("reranker") else "реранкер выключен"
    crop_px = neural.get("crop_px") or [0, 0]
    try:
        crop_w, crop_h = int(crop_px[0]), int(crop_px[1])
    except (TypeError, ValueError, IndexError):
        crop_w, crop_h = 0, 0
    dim = int(neural.get("dim") or 0)
    crop_bit = f"кроп {crop_w}x{crop_h}" if crop_w and crop_h else "кроп ?"
    dim_bit = f"SigLIP {dim}" if dim else "SigLIP"
    lines.append(f"Нейросеть вход: {crop_bit}, {dim_bit}, {rerank}")
    token_bits = []
    if features.get("grape"):
        token_bits.append("сорта: " + ", ".join(features["grape"]))
    if features.get("name"):
        token_bits.append("названия: " + ", ".join(features["name"]))
    if features.get("winery"):
        token_bits.append("винодельня: " + ", ".join(features["winery"]))
    lines.append("  токены: " + ("; ".join(token_bits) if token_bits else "нет"))

    compared = int(neural.get("compared") or 0)
    catalog = int(neural.get("catalog") or 0)
    if neural.get("scope") == "короткий список":
        lines.append(f"Сравнили {compared} из {catalog} (шортлист)")
    else:
        lines.append(f"Сравнили {compared} (весь каталог)")

    lines.append("Ближайшие:")
    neighbors = trace.get("neighbors") or []
    winner_slug = winner.get("slug") or ""
    found = winner.get("status") == "found"
    if not neighbors:
        lines.append("  нет")
    for index, item in enumerate(neighbors, 1):
        label = item.get("name") or item.get("slug") or "?"
        try:
            visual_s = f"{float(item.get('visual')):.2f}"
        except (TypeError, ValueError):
            visual_s = "—"
        try:
            final_s = f"{float(item.get('final')):.2f}"
        except (TypeError, ValueError):
            final_s = "—"
        mark = ""
        if found and item.get("slug") and item.get("slug") == winner_slug:
            mark = "  <- результат"
        lines.append(f"  {index}. {label} | visual {visual_s} | итог {final_s}{mark}")
    if not found:
        lines.append("Итог: не в каталоге")
    lines.append("---")
    _emit_console(lines)


def public_recognition_json(
    result: dict,
    debug: bool = False,
    trace: bool = False,
) -> dict:
    """One catalog card when the match is confident.

    Otherwise status is not_found: slug and name stay empty, and up to
    three similar catalog wines are in `similar`. Top candidates stay
    in `_debug` only.
    """

    wine = result.get("wine") or {}
    status = result.get("status") or "not_found"
    if result.get("method") in ("code", "barcode", "qr") and status == "found":
        confidence = 1.0
    elif status == "found":
        confidence = safe_float(
            wine.get(
                "final_score",
                wine.get("visual_score", 0.0),
            )
        )
    else:
        confidence = safe_float(result.get("confidence", 0.0))

    miss = status != "found"
    payload = public_view(
        status="not_in_catalog" if miss else "found",
        ocr=result.get("ocr_text") or "",
        ocr_clean=result.get("ocr_clean") or "",
        method=result.get("method"),
        visual_compared=int(result.get("visual_compared") or 0),
        timing=result.get("timing", {}),
        wine=wine if status == "found" else None,
        slug=result.get("slug") if status == "found" else None,
        confidence=confidence,
        similar=result.get("similar") or [],
        external=result.get("external") if miss else None,
        message=result.get("message") or "",
    )

    if debug and result.get("_debug"):
        payload["_debug"] = result["_debug"]

    if trace and result.get("_trace"):
        payload["trace"] = result["_trace"]

    return payload


# ============================================================
# PRINT
# ============================================================

def print_result(
    result: dict,
    debug: bool = False,
) -> None:

    print()
    print("=" * 80)
    print("RECOGNITION RESULT")
    print("=" * 80)

    print(
        f"Method: {result['method']}"
    )

    print(
        f"Status: {result['status']}"
    )

    print()

    # --------------------------------------------------------
    # FINAL RESULT — ONE WINE
    # --------------------------------------------------------

    wine = result.get("wine")

    if wine:

        print(
            "RESULT:"
        )

        print(
            f"Name:   {wine.get('name', '')}"
        )

        print(
            f"Slug:   {result.get('slug', '')}"
        )

        print(
            f"Winery: {wine.get('winery', '')}"
        )

        print(
            f"Region: {wine.get('region', '')}"
        )

        print(
            f"Grape:  {wine.get('grape', '')}"
        )

        if wine.get("description"):
            print(
                f"Description: "
                f"{wine['description']}"
            )

        if "final_score" in wine:
            print(
                f"Confidence: "
                f"{wine['final_score']:.4f}"
            )

    else:
        print(
            "RESULT: NOT FOUND"
        )
        similar = result.get("similar") or []
        if similar:
            print()
            print("SIMILAR:")
            for index, item in enumerate(similar, 1):
                print(
                    f"  {index}. {item.get('name', '')} "
                    f"({item.get('reason', '')})"
                )

    # --------------------------------------------------------
    # CODES
    # --------------------------------------------------------

    if result.get("codes"):
        print()
        print("CODES:")

        for code in result["codes"]:
            print(
                f"  {code['type']}: "
                f"{code['data']}"
            )

            if code.get("matched_slug"):
                print(
                    f"    local slug: "
                    f"{code['matched_slug']}"
                )

            if code.get("external_status"):
                print(
                    f"    external: "
                    f"{code['external_status']}"
                )

    # --------------------------------------------------------
    # TIMING
    # --------------------------------------------------------

    print()

    timing = result.get(
        "timing",
        {},
    )

    print(
        f"Time: "
        f"{timing.get('total', 0)}s"
    )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    if debug:

        print()
        print("=" * 80)
        print("DEBUG CANDIDATES")
        print("=" * 80)

        debug_block = result.get("_debug", {})

        scope = debug_block.get(
            "visual_scope",
            "",
        )
        print(
            f"Visual scope: {scope}"
        )
        print(
            "Shortlist: "
            f"{debug_block.get('shortlist_size', 0)} "
            f"({debug_block.get('shortlist_reason', '')})"
        )
        print(
            "Grape tokens: "
            f"{debug_block.get('ocr_grape_tokens', [])}"
        )

        # В debug можно увидеть top-10,
        # но обычный результат всегда один.
        candidates = debug_block.get(
            "candidates",
            [],
        )

        for i, candidate in enumerate(
            candidates[:10],
            1,
        ):
            print()
            print(
                f"{i}. "
                f"{candidate.get('name', '')}"
            )

            print(
                f"   slug:       "
                f"{candidate.get('slug', '')}"
            )

            print(
                f"   visual:     "
                f"{candidate.get('visual_score', 0):.4f}"
            )

            print(
                f"   OCR:        "
                f"{candidate.get('ocr_score', 0):.4f}"
            )

            print(
                f"   grape:      "
                f"{candidate.get('grape_score', 0):.4f}"
            )

            print(
                f"   final:      "
                f"{candidate.get('final_score', 0):.4f}"
            )

            if candidate.get(
                "matched_tokens"
            ):
                print(
                    "   matched:"
                )

                for item in candidate[
                    "matched_tokens"
                ]:
                    print(
                        f"      "
                        f"{item['ocr']} -> "
                        f"{item['catalog']} "
                        f"({item['field']})"
                    )

            if candidate.get(
                "negative_tokens"
            ):
                print(
                    "   negative:"
                )

                for item in candidate[
                    "negative_tokens"
                ]:
                    print(
                        f"      "
                        f"{item['ocr']} "
                        f"({item['reason']})"
                    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = [
        arg
        for arg in sys.argv[1:]
        if arg != "--debug"
    ]

    if not args:
        print(
            "Usage:"
        )
        print(
            "python "
            "src\\hybrid_recognizer.py "
            "PHOTO [--debug]"
        )
        return

    image_path = Path(
        args[0]
    )

    if not image_path.exists():
        print(
            f"File not found: "
            f"{image_path}"
        )
        return

    print(
        f"Image: {image_path}"
    )

    try:
        image = Image.open(
            image_path
        ).convert("RGB")
    except Exception as exc:
        print(
            f"Cannot open image: {exc}"
        )
        return

    recognizer = (
        HybridWineRecognizer()
    )

    result = recognizer.recognize(
        image,
        debug=DEBUG,
    )

    ocr_text = result.get("ocr_text") or ""

    print()
    print("OCR:")
    print(ocr_text if ocr_text else "(пусто)")
    normalized = result.get("ocr_clean") or ""
    print("OCR normalized:")
    print(normalized if normalized else "(нет надежных токенов)")
    print()

    print(
        json.dumps(
            public_recognition_json(
                result,
                debug=DEBUG,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )

    if DEBUG:
        print_result(
            result,
            debug=True,
        )


if __name__ == "__main__":
    main()
