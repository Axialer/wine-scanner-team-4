import json
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "data" / "real_photo"
REPORT_DIR = ROOT / "reports"
REPORT_FILE = REPORT_DIR / "real_photo_dataset.json"


SUPPORTED = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def main():
    print("=" * 70)
    print("REAL PHOTO DATASET INSPECTOR")
    print("=" * 70)
    print()

    if not REAL_DIR.exists():
        print(f"ОШИБКА: папка не найдена:")
        print(REAL_DIR)
        return

    files = [
        p
        for p in REAL_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED
    ]

    print(f"Папка: {REAL_DIR}")
    print(f"Изображений: {len(files)}")
    print()

    if not files:
        print("Изображений нет.")
        return

    extensions = Counter()
    sizes = Counter()

    min_width = None
    min_height = None
    max_width = None
    max_height = None

    total_pixels = 0
    valid = 0
    broken = 0

    records = []

    for index, path in enumerate(files, 1):
        relative = str(path.relative_to(REAL_DIR))

        record = {
            "file": relative,
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
        }

        extensions[path.suffix.lower()] += 1

        try:
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                fmt = image.format

                record.update({
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "format": fmt,
                })

                sizes[(width, height)] += 1

                total_pixels += width * height

                if min_width is None or width < min_width:
                    min_width = width

                if max_width is None or width > max_width:
                    max_width = width

                if min_height is None or height < min_height:
                    min_height = height

                if max_height is None or height > max_height:
                    max_height = height

                valid += 1

        except Exception as exc:
            record["error"] = str(exc)
            broken += 1

        records.append(record)

    print("=" * 70)
    print("ОБЩАЯ ИНФОРМАЦИЯ")
    print("=" * 70)

    print(f"Всего файлов:       {len(files)}")
    print(f"Корректных:         {valid}")
    print(f"Повреждённых:       {broken}")

    print()

    print(f"Минимальный размер: {min_width} x {min_height}")
    print(f"Максимальный размер: {max_width} x {max_height}")

    if valid:
        print(
            f"Среднее разрешение: "
            f"{int(total_pixels / valid):,} px"
        )

    print()

    print("=" * 70)
    print("РАСШИРЕНИЯ")
    print("=" * 70)

    for ext, count in extensions.most_common():
        print(f"{ext:<10} {count}")

    print()

    print("=" * 70)
    print("ТОП РАЗМЕРОВ")
    print("=" * 70)

    for (width, height), count in sizes.most_common(20):
        print(
            f"{width} x {height:<6} "
            f"{count} шт."
        )

    print()

    print("=" * 70)
    print("ПРИМЕРЫ")
    print("=" * 70)

    for path in files[:20]:
        print(
            f"  {path.relative_to(REAL_DIR)}"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "dataset": str(REAL_DIR),
        "count": len(files),
        "valid": valid,
        "broken": broken,
        "extensions": dict(extensions),
        "min_width": min_width,
        "max_width": max_width,
        "min_height": min_height,
        "max_height": max_height,
        "images": records,
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("ОТЧЁТ СОХРАНЁН")
    print("=" * 70)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()