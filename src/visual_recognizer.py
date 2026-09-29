import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_PATH = (
    ROOT / "index" / "catalog_embeddings.npy"
)

META_PATH = (
    ROOT / "index" / "catalog_meta.json"
)

CSV_PATH = (
    ROOT
    / "data"
    / "catalog"
    / "strapi_output0709.csv"
)

MODEL_NAME = "google/siglip2-base-patch16-224"

TOP_K = 10


class VisualWineRecognizer:

    def __init__(self):
        print("Loading catalog embeddings...")

        self.embeddings = np.load(
            EMBEDDINGS_PATH
        ).astype(np.float32)

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

        self.items = meta["items"]

        print(
            f"Catalog loaded: "
            f"{len(self.items)} wines"
        )

        self.csv = pd.read_csv(
            CSV_PATH,
            dtype=str,
            keep_default_na=False,
        )

        self.csv_by_slug = {}

        for _, row in self.csv.iterrows():
            slug = row.get("Slug", "")

            if slug and slug not in self.csv_by_slug:
                self.csv_by_slug[slug] = row.to_dict()

        print(
            f"Wine metadata loaded: "
            f"{len(self.csv_by_slug)} slugs"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(
            f"Loading {MODEL_NAME}..."
        )

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModel.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        print("SigLIP2 ready")

    @torch.inference_mode()
    def get_embedding(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        features = self.model.get_image_features(
            **inputs
        )
        
        # Совместимость с разными версиями transformers:
        # в одних возвращается Tensor, в других — BaseModelOutputWithPooling
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        
        features = features.detach().cpu().numpy()
        
        features /= (
            np.linalg.norm(
                features,
                axis=1,
                keepdims=True,
            )
            + 1e-12
        )

        return features[0]

    def search(
        self,
        image,
        top_k=TOP_K,
    ):
        start = time.perf_counter()

        embedding = self.get_embedding(
            image
        )

        scores = (
            self.embeddings
            @ embedding
        )

        top_k = min(
            top_k,
            len(scores),
        )

        indices = np.argpartition(
            -scores,
            top_k - 1,
        )[:top_k]

        indices = indices[
            np.argsort(
                -scores[indices]
            )
        ]

        results = []

        for index in indices:
            item = self.items[index]

            slug = item["slug"]

            metadata = self.csv_by_slug.get(
                slug,
                {},
            )

            results.append({
                "slug": slug,
                "score": float(scores[index]),
                "name": metadata.get(
                    "Название вина",
                    "",
                ),
                "category": metadata.get(
                    "Категория",
                    "",
                ),
                "color": metadata.get(
                    "Цвет",
                    "",
                ),
                "region": metadata.get(
                    "Регион",
                    "",
                ),
                "grape": metadata.get(
                    "Сорт винограда",
                    "",
                ),
                "winery": metadata.get(
                    "Винодельня",
                    "",
                ),
                "description": metadata.get(
                    "Описание",
                    "",
                ),
            })

        elapsed = (
            time.perf_counter()
            - start
        )

        return {
            "results": results,
            "time": elapsed,
        }


def main():
    if len(sys.argv) < 2:
        print(
            "Использование:"
        )
        print(
            "python src\\visual_recognizer.py photo.webp"
        )
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(
            f"Файл не найден: {image_path}"
        )
        return

    recognizer = VisualWineRecognizer()

    result = recognizer.search(
        image_path,
        top_k=10,
    )

    print()
    print("=" * 70)
    print("RECOGNITION RESULT")
    print("=" * 70)

    print(
        f"Time: "
        f"{result['time']:.3f}s"
    )

    print()

    for index, wine in enumerate(
        result["results"],
        1,
    ):
        print(
            f"{index}. "
            f"{wine['name']}"
        )

        print(
            f"   slug:       {wine['slug']}"
        )

        print(
            f"   similarity: "
            f"{wine['score']:.4f}"
        )

        print(
            f"   winery:     "
            f"{wine['winery']}"
        )

        print(
            f"   region:     "
            f"{wine['region']}"
        )

        print(
            f"   grape:      "
            f"{wine['grape']}"
        )

        print()


if __name__ == "__main__":
    main()