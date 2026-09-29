import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = (
    ROOT
    / "data"
    / "catalog"
    / "strapi_output0709.csv"
)

OUTPUT_FILE = (
    ROOT
    / "index"
    / "text_catalog.json"
)


def normalize_text(text):
    if not text:
        return ""

    text = str(text).lower()

    text = text.replace("ё", "е")

    text = re.sub(
        r"[^a-zа-я0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def main():
    rows = []

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        seen = set()

        for row in reader:
            slug = (
                row.get("Slug") or ""
            ).strip()

            if not slug:
                continue

            if slug in seen:
                continue

            seen.add(slug)

            wine_name = (
                row.get("Название вина")
                or ""
            )

            category = (
                row.get("Категория")
                or ""
            )

            color = (
                row.get("Цвет")
                or ""
            )

            region = (
                row.get("Регион")
                or ""
            )

            grape = (
                row.get("Сорт винограда")
                or ""
            )

            winery = (
                row.get("Винодельня")
                or ""
            )

            description = (
                row.get("Описание")
                or ""
            )

            text_parts = [
                wine_name,
                category,
                color,
                region,
                grape,
                winery,
                description,
            ]

            text = " ".join(
                normalize_text(x)
                for x in text_parts
                if x
            )

            rows.append(
                {
                    "slug": slug,
                    "name": wine_name,
                    "category": category,
                    "color": color,
                    "region": region,
                    "grape": grape,
                    "winery": winery,
                    "description": description,
                    "text": text,
                }
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            rows,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Catalog entries: {len(rows)}"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()