import sys
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import easyocr
from rapidfuzz import fuzz


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "index"
CSV_PATH = BASE_DIR / "data" / "catalog" / "strapi_output0709.csv"

EMBEDDINGS_PATH = INDEX_DIR / "catalog_embeddings.npy"
META_PATH = INDEX_DIR / "catalog_meta.json"

MODEL_NAME = "google/siglip2-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Берём больше кандидатов для reranking.
TOP_K = 20

# После эксперимента:
# visual = 60%
# OCR    = 40%
VISUAL_WEIGHT = 0.60
OCR_WEIGHT = 0.40

# OCR запускаем, если есть хоть какая-то неоднозначность.
# На первом этапе лучше не экономить OCR:
# нам сейчас важнее качество.
VISUAL_CONFIDENCE_THRESHOLD = 0.82
MARGIN_THRESHOLD = 0.025

# Не принимаем совсем слабый результат.
FINAL_THRESHOLD = 0.35

DEBUG_TOP_K = 10


# ============================================================
# TEXT UTILS
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).upper()
    text = text.replace("Ё", "Е")

    text = re.sub(
        r"[^A-ZА-Я0-9]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text):

    text = normalize_text(text)

    if not text:
        return []

    return [
        token
        for token in text.split()
        if len(token) >= 3
    ]


# ============================================================
# RECOGNIZER
# ============================================================

class WineRecognizer:

    def __init__(self):

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        print("Loading catalog embeddings...")

        self.embeddings = np.load(
            EMBEDDINGS_PATH
        ).astype(np.float32)

        print(
            f"Catalog embeddings: {self.embeddings.shape}"
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        with open(
            META_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            meta_data = json.load(f)

        self.meta = meta_data["items"]

        print(
            f"Catalog loaded: {len(self.meta)} wines"
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        self.load_catalog_metadata()

        # ----------------------------------------------------
        # SigLIP
        # ----------------------------------------------------

        print("Loading SigLIP2...")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModel.from_pretrained(
            MODEL_NAME
        )

        self.model.to(DEVICE)
        self.model.eval()

        print("SigLIP2 ready")

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        print("Loading EasyOCR...")

        self.reader = easyocr.Reader(
            ["ru", "en"],
            gpu=torch.cuda.is_available()
        )

        print("EasyOCR ready")

    # ========================================================
    # CATALOG
    # ========================================================

    def load_catalog_metadata(self):

        df = pd.read_csv(
            CSV_PATH
        )

        self.catalog_by_slug = {}

        for _, row in df.iterrows():

            slug = str(
                row.get("Slug", "")
            ).strip()

            if not slug:
                continue

            if slug in self.catalog_by_slug:
                continue

            self.catalog_by_slug[slug] = {

                "name": str(
                    row.get(
                        "Название вина",
                        ""
                    )
                ),

                "category": str(
                    row.get(
                        "Категория",
                        ""
                    )
                ),

                "color": str(
                    row.get(
                        "Цвет",
                        ""
                    )
                ),

                "region": str(
                    row.get(
                        "Регион",
                        ""
                    )
                ),

                "grape": str(
                    row.get(
                        "Сорт винограда",
                        ""
                    )
                ),

                "description": str(
                    row.get(
                        "Описание",
                        ""
                    )
                ),

                "winery": str(
                    row.get(
                        "Винодельня",
                        ""
                    )
                )
            }

        print(
            f"Wine metadata loaded: "
            f"{len(self.catalog_by_slug)} slugs"
        )

    # ========================================================
    # SIGLIP
    # ========================================================

    @torch.no_grad()
    def get_embedding(self, image):

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            image_rgb
        )

        inputs = self.processor(
            images=pil_image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        outputs = self.model.get_image_features(
            **inputs
        )

        if isinstance(
            outputs,
            torch.Tensor
        ):

            features = outputs

        elif hasattr(
            outputs,
            "pooler_output"
        ):

            features = outputs.pooler_output

        elif hasattr(
            outputs,
            "last_hidden_state"
        ):

            features = outputs.last_hidden_state[:, 0]

        else:

            raise RuntimeError(
                "Неизвестный формат результата "
                f"SigLIP: {type(outputs)}"
            )

        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

        return features[0].cpu().numpy()

    # ========================================================
    # VISUAL SEARCH
    # ========================================================

    def visual_search(
        self,
        embedding,
        top_k=TOP_K
    ):

        embedding = embedding.astype(
            np.float32
        )

        scores = np.dot(
            self.embeddings,
            embedding
        )

        indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in indices:

            item = self.meta[index]

            results.append({

                "slug": item["slug"],

                "score": float(
                    scores[index]
                )
            })

        return results

    # ========================================================
    # OCR
    # ========================================================

    def extract_ocr(self, image):

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.reader.readtext(
            image_rgb,
            detail=1,
            paragraph=False,
            min_size=10,
            text_threshold=0.4,
            low_text=0.2,
            link_threshold=0.2,
            mag_ratio=1.5
        )

        texts = []

        for item in results:

            if len(item) < 2:
                continue

            text = str(
                item[1]
            ).strip()

            if not text:
                continue

            texts.append(text)

        return " ".join(texts)

    # ========================================================
    # OCR SCORE
    # ========================================================

    def calculate_ocr_score(
        self,
        ocr_text,
        candidate
    ):

        if not ocr_text:
            return 0.0

        catalog = self.catalog_by_slug.get(
            candidate["slug"]
        )

        if not catalog:
            return 0.0

        ocr_tokens = tokenize(
            ocr_text
        )

        if not ocr_tokens:
            return 0.0

        # Название и винодельня самые важные.
        fields = [

            catalog.get(
                "name",
                ""
            ),

            catalog.get(
                "winery",
                ""
            ),

            catalog.get(
                "grape",
                ""
            ),

            catalog.get(
                "region",
                ""
            )
        ]

        field_tokens = []

        for field in fields:

            field_tokens.extend(
                tokenize(field)
            )

        if not field_tokens:
            return 0.0

        best_scores = []

        for ocr_token in ocr_tokens:

            best = 0.0

            for catalog_token in field_tokens:

                if ocr_token == catalog_token:

                    best = 1.0
                    break

                score = fuzz.ratio(
                    ocr_token,
                    catalog_token
                ) / 100.0

                if score > best:
                    best = score

            best_scores.append(
                best
            )

        if not best_scores:
            return 0.0

        best_scores.sort(
            reverse=True
        )

        if len(best_scores) == 1:

            return best_scores[0]

        if len(best_scores) == 2:

            return (
                best_scores[0] * 0.70
                + best_scores[1] * 0.30
            )

        return (
            best_scores[0] * 0.60
            + best_scores[1] * 0.25
            + best_scores[2] * 0.15
        )

    # ========================================================
    # RERANK
    # ========================================================

    def rerank_with_ocr(
        self,
        visual_results,
        ocr_text
    ):

        if not visual_results:
            return []

        # ----------------------------------------------------
        # Нормализуем visual score внутри Top-K.
        #
        # Именно так считался эксперимент,
        # который дал 82% Top-1.
        # ----------------------------------------------------

        scores = [
            item["score"]
            for item in visual_results
        ]

        min_score = min(scores)
        max_score = max(scores)

        reranked = []

        for candidate in visual_results:

            visual_score = candidate["score"]

            if max_score > min_score:

                visual_normalized = (
                    visual_score - min_score
                ) / (
                    max_score - min_score
                )

            else:

                visual_normalized = 1.0

            ocr_score = self.calculate_ocr_score(
                ocr_text,
                candidate
            )

            final_score = (
                VISUAL_WEIGHT
                * visual_normalized
                +
                OCR_WEIGHT
                * ocr_score
            )

            reranked.append({

                "slug":
                    candidate["slug"],

                "visual_score":
                    float(visual_score),

                "visual_normalized":
                    float(visual_normalized),

                "ocr_score":
                    float(ocr_score),

                "final_score":
                    float(final_score)
            })

        reranked.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return reranked

    # ========================================================
    # RECOGNITION
    # ========================================================

    def recognize(
        self,
        image_path
    ):

        total_start = time.perf_counter()

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                f"Не удалось открыть изображение: "
                f"{image_path}"
            )

        # ----------------------------------------------------
        # RAW IMAGE
        #
        # ВАЖНО:
        # normalization здесь НЕ используется.
        #
        # На наших 100 тестах:
        # raw = 77%
        # normalized = 76%
        # ----------------------------------------------------

        start = time.perf_counter()

        embedding = self.get_embedding(
            image
        )

        visual_results = self.visual_search(
            embedding,
            TOP_K
        )

        visual_time = (
            time.perf_counter() - start
        )

        # ----------------------------------------------------
        # Visual statistics
        # ----------------------------------------------------

        visual_top1 = visual_results[0]

        visual_top2 = visual_results[1]

        visual_score = visual_top1["score"]

        margin = (
            visual_top1["score"]
            -
            visual_top2["score"]
        )

        # ----------------------------------------------------
        # OCR
        #
        # Пока запускаем OCR для всех запросов.
        #
        # Сначала получаем максимально честное качество.
        # После этого отдельно оптимизируем SLA.
        # ----------------------------------------------------

        start = time.perf_counter()

        ocr_text = self.extract_ocr(
            image
        )

        ocr_time = (
            time.perf_counter() - start
        )

        # ----------------------------------------------------
        # RERANK
        # ----------------------------------------------------

        reranked = self.rerank_with_ocr(
            visual_results,
            ocr_text
        )

        best = reranked[0]

        final_score = best["final_score"]

        # ----------------------------------------------------
        # FOUND / NOT FOUND
        # ----------------------------------------------------

        # Не используем старый visual threshold 0.72,
        # потому что final_score теперь находится
        # в другой шкале.
        #
        # Если визуальный кандидат совсем слабый,
        # всё равно считаем это not_found.
        #
        # Для текущего этапа основное решение:
        # есть ли кандидат вообще.
        #
        # 0.35 — консервативный нижний порог.
        # Позже откалибруем отдельно.
        # ----------------------------------------------------

        status = (
            "found"
            if final_score >= FINAL_THRESHOLD
            else "not_found"
        )

        total_time = (
            time.perf_counter()
            - total_start
        )

        # ----------------------------------------------------
        # NOT FOUND
        # ----------------------------------------------------

        if status == "not_found":

            recommendations = []

            for item in reranked[:5]:

                recommendations.append({

                    "slug":
                        item["slug"],

                    "similarity":
                        round(
                            item["final_score"],
                            4
                        )
                })

            return {

                "slug": None,

                "status": "not_found",

                "message":
                    "К сожалению, такое вино "
                    "не найдено в каталоге.",

                "recommendations":
                    recommendations,

                "_debug": {

                    "visual_score":
                        float(visual_score),

                    "final_score":
                        float(final_score),

                    "ocr_score":
                        float(best["ocr_score"]),

                    "margin":
                        float(margin),

                    "ocr_used":
                        True,

                    "ocr_text":
                        ocr_text,

                    "timing": {

                        "visual":
                            round(
                                visual_time,
                                3
                            ),

                        "ocr":
                            round(
                                ocr_time,
                                3
                            ),

                        "total":
                            round(
                                total_time,
                                3
                            )
                    },

                    "top_candidates": [
                        {
                            "slug":
                                item["slug"],

                            "visual_score":
                                round(
                                    item["visual_score"],
                                    4
                                ),

                            "visual_normalized":
                                round(
                                    item["visual_normalized"],
                                    4
                                ),

                            "ocr_score":
                                round(
                                    item["ocr_score"],
                                    4
                                ),

                            "final_score":
                                round(
                                    item["final_score"],
                                    4
                                )
                        }

                        for item in reranked[
                            :DEBUG_TOP_K
                        ]
                    ]
                }
            }

        # ----------------------------------------------------
        # FOUND
        # ----------------------------------------------------

        return {

            "slug":
                best["slug"],

            "status":
                "found",

            "_debug": {

                "visual_score":
                    float(visual_score),

                "final_score":
                    float(final_score),

                "ocr_score":
                    float(best["ocr_score"]),

                "margin":
                    float(margin),

                "ocr_used":
                    True,

                "ocr_text":
                    ocr_text,

                "timing": {

                    "visual":
                        round(
                            visual_time,
                            3
                        ),

                    "ocr":
                        round(
                            ocr_time,
                            3
                        ),

                    "total":
                        round(
                            total_time,
                            3
                        )
                },

                "top_candidates": [

                    {

                        "slug":
                            item["slug"],

                        "visual_score":
                            round(
                                item["visual_score"],
                                4
                            ),

                        "visual_normalized":
                            round(
                                item["visual_normalized"],
                                4
                            ),

                        "ocr_score":
                            round(
                                item["ocr_score"],
                                4
                            ),

                        "final_score":
                            round(
                                item["final_score"],
                                4
                            )
                    }

                    for item in reranked[
                        :DEBUG_TOP_K
                    ]
                ]
            }
        }


# ============================================================
# CLI
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Использование:"
        )

        print(
            "python src\\recognizer.py test.jpg"
        )

        return

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print(
            f"Файл не найден: {image_path}"
        )

        return

    recognizer = WineRecognizer()

    result = recognizer.recognize(
        image_path
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()