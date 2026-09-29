from pathlib import Path
import json
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATALOG_DIR = PROJECT_ROOT / "data" / "catalog_clean"
INDEX_DIR = PROJECT_ROOT / "index"

MODEL_NAME = "google/siglip2-base-patch16-224"

# CPU laptop
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Не ставим большой batch на CPU.
BATCH_SIZE = 8 if DEVICE == "cpu" else 32

EMBEDDINGS_FILE = INDEX_DIR / "catalog_embeddings.npy"
META_FILE = INDEX_DIR / "catalog_meta.json"


# ============================================================
# HELPERS
# ============================================================

def get_catalog_images():
    """
    Получает список изображений каталога.

    Формат:
        data/catalog_clean/<slug>.webp
    """

    if not CATALOG_DIR.exists():
        print(f"ERROR: catalog directory not found:")
        print(f"  {CATALOG_DIR}")
        sys.exit(1)

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }

    images = [
        path
        for path in CATALOG_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    ]

    images.sort(key=lambda p: p.name.lower())

    return images


def load_image(path):
    """
    Загружает изображение и приводит к RGB.
    """

    try:
        with Image.open(path) as image:
            return image.convert("RGB")

    except Exception as exc:
        print()
        print(f"WARNING: failed to load image:")
        print(f"  {path}")
        print(f"  {exc}")

        return None


def extract_slug(path):
    """
    Slug хранится в имени файла:

        slug.webp
        slug.jpg
        slug.png
    """

    return path.stem


def get_features(model, processor, images):
    """
    Получает image embeddings для batch изображений.

    В зависимости от версии Transformers
    get_image_features() может вернуть:

        Tensor

    либо:

        BaseModelOutputWithPooling

    В последнем случае используем pooler_output.
    """

    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # --------------------------------------------------------
    # Transformers может вернуть Tensor
    # --------------------------------------------------------

    if isinstance(outputs, torch.Tensor):
        features = outputs

    # --------------------------------------------------------
    # Или BaseModelOutputWithPooling
    # --------------------------------------------------------

    elif hasattr(outputs, "pooler_output"):
        features = outputs.pooler_output

    # --------------------------------------------------------
    # На всякий случай поддерживаем last_hidden_state
    # --------------------------------------------------------

    elif hasattr(outputs, "last_hidden_state"):
        features = outputs.last_hidden_state[:, 0]

    else:
        raise RuntimeError(
            "Unknown output type from model.get_image_features(): "
            f"{type(outputs)}"
        )

    # --------------------------------------------------------
    # Приводим к float32
    # --------------------------------------------------------

    features = features.float()

    # --------------------------------------------------------
    # L2 normalization
    #
    # После этого:
    # cosine similarity == dot product
    # --------------------------------------------------------

    norms = features.norm(
        dim=-1,
        keepdim=True,
    )

    features = features / torch.clamp(
        norms,
        min=1e-12,
    )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILD WINE CATALOG EMBEDDINGS")
    print("=" * 70)
    print()

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # --------------------------------------------------------
    # Catalog
    # --------------------------------------------------------

    catalog_images = get_catalog_images()

    if not catalog_images:
        print("ERROR: no catalog images found.")
        print()
        print(f"Expected directory:")
        print(f"  {CATALOG_DIR}")
        sys.exit(1)

    print(f"Catalog images: {len(catalog_images)}")
    print()

    # --------------------------------------------------------
    # Index directory
    # --------------------------------------------------------

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("[1] Loading model...")

    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

        model = AutoModel.from_pretrained(
            MODEL_NAME
        )

    except Exception as exc:
        print()
        print("ERROR: failed to load SigLIP2.")
        print()
        print(exc)
        print()
        sys.exit(1)

    model = model.to(DEVICE)
    model.eval()

    print("    Model loaded.")
    print()

    # --------------------------------------------------------
    # Build embeddings
    # --------------------------------------------------------

    print("[2] Building embeddings...")

    all_embeddings = []
    metadata = []

    total_batches = (
        len(catalog_images) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    processed = 0
    failed = 0

    for batch_start in tqdm(
        range(
            0,
            len(catalog_images),
            BATCH_SIZE,
        ),
        desc="Embedding",
        total=total_batches,
    ):

        batch_paths = catalog_images[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        images = []
        valid_paths = []

        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        for path in batch_paths:

            image = load_image(path)

            if image is None:
                failed += 1
                continue

            images.append(image)
            valid_paths.append(path)

        if not images:
            continue

        # ----------------------------------------------------
        # Get embeddings
        # ----------------------------------------------------

        try:

            features = get_features(
                model,
                processor,
                images,
            )

        except Exception as exc:

            print()
            print("ERROR while building embeddings.")
            print()

            for path in valid_paths:
                print(f"  {path}")

            print()
            print(exc)

            sys.exit(1)

        # ----------------------------------------------------
        # Save embeddings in memory
        # ----------------------------------------------------

        all_embeddings.append(
            features.cpu().numpy()
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        for path in valid_paths:

            slug = extract_slug(path)

            metadata.append(
                {
                    "slug": slug,
                    "file": str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "filename": path.name,
                }
            )

        processed += len(valid_paths)

    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    if not all_embeddings:

        print()
        print("ERROR: no embeddings were generated.")
        sys.exit(1)

    # --------------------------------------------------------
    # Combine batches
    # --------------------------------------------------------

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    # --------------------------------------------------------
    # Final normalization
    #
    # This is intentionally repeated to guarantee that
    # every stored vector is normalized.
    # --------------------------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    embeddings = embeddings / np.clip(
        norms,
        1e-12,
        None,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if len(embeddings) != len(metadata):

        print()
        print("ERROR:")
        print(
            f"Embeddings: {len(embeddings)}"
        )
        print(
            f"Metadata:   {len(metadata)}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    print()
    print("[3] Saving index...")

    np.save(
        EMBEDDINGS_FILE,
        embeddings,
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_payload = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "count": len(metadata),
        "dimension": int(
            embeddings.shape[1]
        ),
        "normalized": True,
        "items": metadata,
    }

    with open(
        META_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()

    print(
        f"Embeddings: {embeddings.shape}"
    )

    print(
        f"Dimension:  {embeddings.shape[1]}"
    )

    print(
        f"Processed:  {processed}"
    )

    print(
        f"Failed:     {failed}"
    )

    print()

    print("Files:")

    print(
        f"  {EMBEDDINGS_FILE}"
    )

    print(
        f"  {META_FILE}"
    )

    print()

    # --------------------------------------------------------
    # Quick normalization check
    # --------------------------------------------------------

    sample_norms = np.linalg.norm(
        embeddings[: min(100, len(embeddings))],
        axis=1,
    )

    print(
        "Normalization check:"
    )

    print(
        f"  min norm: {sample_norms.min():.6f}"
    )

    print(
        f"  max norm: {sample_norms.max():.6f}"
    )

    print()

    print("Catalog embedding index successfully built.")


if __name__ == "__main__":
    main()