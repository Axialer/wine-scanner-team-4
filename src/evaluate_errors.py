import json
import time
from pathlib import Path

import numpy as np

from recognizer import WineRecognizer


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "data"
    / "catalog"
    / "eval_generated"
    / "manifest.json"
)

QUERY_DIR = (
    ROOT
    / "data"
    / "catalog"
    / "eval_generated"
    / "queries"
)


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError(
            f"Ожидался список в manifest.json, "
            f"получено: {type(data).__name__}"
        )

    return data


def get_value(item, names):
    for name in names:
        if name in item:
            return item[name]

    return None


def extract_test_info(item):
    """
    Автоматически пытаемся найти:
      - имя изображения
      - правильный slug
    """

    image_name = get_value(
        item,
        [
            "image",
            "filename",
            "file",
            "query",
            "path",
        ],
    )

    target_slug = get_value(
        item,
        [
            "target_slug",
            "expected_slug",
            "ground_truth",
            "target",
            "wine_slug",
            "correct_slug",
            "label",
        ],
    )

    return image_name, target_slug


def get_slug(result):
    if isinstance(result, dict):
        return result.get("slug")

    return None


def get_score(result):
    if not isinstance(result, dict):
        return None

    for key in ["score", "similarity", "visual_score"]:
        if key in result:
            try:
                return float(result[key])
            except Exception:
                pass

    return None


def rank_of(results, target_slug):
    if not target_slug:
        return None

    for i, result in enumerate(results, 1):
        if get_slug(result) == target_slug:
            return i

    return None


def print_top(results, target_slug, count=5):
    for i, result in enumerate(results[:count], 1):
        slug = get_slug(result)
        score = get_score(result)

        marker = "  <==" if slug == target_slug else ""

        if score is None:
            print(
                f"      {i}. {slug}{marker}"
            )
        else:
            print(
                f"      {i}. {slug} "
                f"[{score:.4f}]{marker}"
            )


def calculate_metrics(ranks):
    total = len(ranks)

    top1 = sum(
        r == 1
        for r in ranks
    )

    top5 = sum(
        r is not None and r <= 5
        for r in ranks
    )

    reciprocal = []

    for r in ranks:
        if r is None:
            reciprocal.append(0.0)
        else:
            reciprocal.append(1.0 / r)

    return {
        "top1": top1 / total if total else 0,
        "top5": top5 / total if total else 0,
        "mrr": float(np.mean(reciprocal))
        if reciprocal
        else 0,
    }


def main():

    print()
    print("=" * 70)
    print("WINE RECOGNITION — VISUAL EVALUATION")
    print("=" * 70)
    print()

    # ------------------------------------------------------------
    # MANIFEST
    # ------------------------------------------------------------

    tests = load_manifest()

    print(
        f"Тестов в manifest: {len(tests)}"
    )

    print()
    print("Проверяю структуру manifest...")

    first = tests[0]

    print(
        "Первый элемент:"
    )

    print(
        json.dumps(
            first,
            ensure_ascii=False,
            indent=2,
        )
    )

    image_name, target_slug = extract_test_info(
        first
    )

    print()
    print(
        f"Изображение: {image_name}"
    )
    print(
        f"Ожидаемый slug: {target_slug}"
    )

    if not image_name or not target_slug:
        print()
        print(
            "ОШИБКА: не удалось автоматически определить "
            "поля manifest."
        )
        print()
        print(
            "Покажи структуру первого элемента выше."
        )
        return

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    print()
    print(
        "Инициализация WineRecognizer..."
    )

    recognizer = WineRecognizer()

    print()
    print("Модель готова.")
    print()

    # ------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------

    ranks = []

    errors = []

    times = []

    for index, item in enumerate(
        tests,
        1,
    ):

        image_name, target_slug = (
            extract_test_info(item)
        )

        if not image_name:
            print(
                f"[{index:03d}] "
                f"SKIP — нет имени изображения"
            )
            continue

        if not target_slug:
            print(
                f"[{index:03d}] "
                f"SKIP — нет target slug"
            )
            continue

        image_path = (
            QUERY_DIR
            / image_name
        )

        if not image_path.exists():

            print(
                f"[{index:03d}] "
                f"SKIP — файл не найден: "
                f"{image_path.name}"
            )

            continue

        print(
            f"[{index:03d}/{len(tests)}] "
            f"{target_slug}"
        )

        start = time.perf_counter()

        results = recognizer.visual_search(
            str(image_path),
            normalize=False,
            top_k=20,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(elapsed)

        rank = rank_of(
            results,
            target_slug,
        )

        ranks.append(rank)

        print(
            f"    rank = {rank}"
            f"    time = {elapsed:.2f}s"
        )

        if rank != 1:

            errors.append(
                {
                    "image": image_name,
                    "target": target_slug,
                    "rank": rank,
                    "results": results[:5],
                }
            )

            print(
                "    TOP-5:"
            )

            print_top(
                results,
                target_slug,
                5,
            )

    # ------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------

    metrics = calculate_metrics(
        ranks
    )

    print()
    print("=" * 70)
    print("РЕЗУЛЬТАТ")
    print("=" * 70)
    print()

    print(
        f"Тестов:   {len(ranks)}"
    )

    print(
        f"Top-1:    "
        f"{metrics['top1'] * 100:.2f}%"
    )

    print(
        f"Top-5:    "
        f"{metrics['top5'] * 100:.2f}%"
    )

    print(
        f"MRR:      "
        f"{metrics['mrr']:.4f}"
    )

    print()

    if times:
        print(
            f"Время:    "
            f"{np.mean(times):.3f}s avg"
        )

        print(
            f"Медиана:  "
            f"{np.median(times):.3f}s"
        )

    # ------------------------------------------------------------
    # ERROR SUMMARY
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"ОШИБКИ TOP-1: {len(errors)}"
    )
    print("=" * 70)

    for error in errors:

        print()
        print(
            f"TARGET: {error['target']}"
        )

        print(
            f"RANK:   {error['rank']}"
        )

        print(
            "TOP-5:"
        )

        print_top(
            error["results"],
            error["target"],
            5,
        )

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()