"""Two RapidOCR reads of the same crop: Cyrillic, then English.

Detection and recognition both run twice when the English mobile model
loads. That file is the small PP-OCRv4 rec head, not a second EasyOCR.
If the English model cannot be downloaded, the Cyrillic read is kept
and tokens that contain a real Latin letter are also emitted in Latin.
EasyOCR stays the last resort and is not loaded for the second pass.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from ocr_shortlist import _DISTINCTIVE_LATIN, normalize_text


def onnx_cuda_available() -> bool:
    try:
        import onnxruntime as ort
    except Exception:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


# The inverse of the lookalike table. Used when the English
# model is missing and a mixed token still needs a Latin spelling.
_CYR_LOOKALIKE_TO_LATIN = str.maketrans({
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


def interpret_latin(text: str) -> str:
    """Latin spellings of tokens that contain a non-lookalike Latin letter."""

    extra: list[str] = []
    seen: set[str] = set()
    for token in normalize_text(text).split():
        if not any(char in _DISTINCTIVE_LATIN for char in token):
            continue
        latin = token.translate(_CYR_LOOKALIKE_TO_LATIN)
        if latin and latin not in seen and latin != token:
            seen.add(latin)
            extra.append(latin)
    return " ".join(extra)


class FastOcr:
    def __init__(self) -> None:
        self.backend = "rapidocr"
        self.device = "cuda" if onnx_cuda_available() else "cpu"
        self._easy = None
        self._engine = self._load_rapid("cyrillic")
        self._engine_en = self._load_rapid("en") if self._engine is not None else None
        self.pass_mode = (
            "rapidocr-cyrillic+en"
            if self._engine_en is not None
            else "rapidocr-cyrillic-interpret"
        )
        self.last_seconds = 0.0
        self.last_cyrillic = ""
        self.last_english = ""
        if self._engine is None:
            self.backend = "easyocr"
            self.device = "cpu"
            self.pass_mode = "easyocr"
            self._easy = self._load_easy()

    def _load_rapid(self, lang: str):
        engine = self._try_rapid(lang, cuda=self.device == "cuda")
        if engine is None and self.device == "cuda":
            print(f"RapidOCR {lang} CUDA failed, using CPU")
            self.device = "cpu"
            engine = self._try_rapid(lang, cuda=False)
        return engine

    def _try_rapid(self, lang: str, *, cuda: bool):
        try:
            from rapidocr import ModelType, OCRVersion, RapidOCR
        except Exception as exc:
            print(f"RapidOCR import failed: {exc}")
            return None

        params = {
            "Global.use_cls": False,
            "Global.max_side_len": 960,
            "Global.log_level": "error",
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Det.model_type": ModelType.MOBILE,
            "Det.limit_side_len": 640,
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.lang_type": lang,
        }
        if cuda:
            params["EngineConfig.onnxruntime.use_cuda"] = True

        try:
            return RapidOCR(params=params)
        except Exception as exc:
            print(f"RapidOCR {lang} init failed: {exc}")
            return None

    def _load_easy(self):
        import easyocr

        gpu = False
        try:
            import torch

            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
        self.device = "cuda" if gpu else "cpu"
        print("Loading EasyOCR fallback...")
        reader = easyocr.Reader(["ru", "en"], gpu=gpu, verbose=False)
        print("EasyOCR ready")
        return reader

    def read(self, image: Image.Image) -> str:
        started = time.perf_counter()
        if self._engine is not None:
            try:
                cyrillic = self._read_engine(self._engine, image)
            except Exception as exc:
                print(f"RapidOCR failed, falling back to EasyOCR: {exc}")
                self._engine = None
                self._engine_en = None
                self.backend = "easyocr"
                self.pass_mode = "easyocr"
                if self._easy is None:
                    self._easy = self._load_easy()
                cyrillic = ""
            else:
                english = ""
                if self._engine_en is not None:
                    try:
                        english = self._read_engine(self._engine_en, image)
                    except Exception as exc:
                        print(f"RapidOCR English pass failed: {exc}")
                        self._engine_en = None
                        self.pass_mode = "rapidocr-cyrillic-interpret"
                if not english:
                    english = interpret_latin(cyrillic)
                    if self._engine_en is None:
                        self.pass_mode = "rapidocr-cyrillic-interpret"
                self.last_cyrillic = cyrillic
                self.last_english = english
                self.last_seconds = time.perf_counter() - started
                return " ".join(part for part in (cyrillic, english) if part)

        if self._easy is None:
            self.last_seconds = time.perf_counter() - started
            return ""
        text = self._read_easy(image)
        self.last_cyrillic = text
        self.last_english = ""
        self.last_seconds = time.perf_counter() - started
        return text

    def _read_engine(self, engine, image: Image.Image) -> str:
        # A lower box threshold keeps the smaller varietal line
        # (СОВИНЬОН БЛАН under ЦИТРОН).
        result = engine(
            image.convert("RGB"),
            use_cls=False,
            box_thresh=0.25,
            text_score=0.3,
            unclip_ratio=2.0,
        )
        texts = getattr(result, "txts", None) or ()
        parts = []
        for text in texts:
            cleaned = str(text).strip()
            if cleaned:
                parts.append(cleaned)
        return " ".join(parts)

    def _read_easy(self, image: Image.Image) -> str:
        array = np.array(image.convert("RGB"))
        result = self._easy.readtext(
            array,
            detail=1,
            paragraph=False,
            width_ths=0.7,
            mag_ratio=1.0,
        )
        parts = []
        for detection in result:
            if len(detection) < 2:
                continue
            text = str(detection[1]).strip()
            if text:
                parts.append(text)
        return " ".join(parts)
