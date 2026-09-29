import re
from difflib import SequenceMatcher


class WineReranker:
    """
    Переранжирование Top-K кандидатов SigLIP2
    с использованием OCR и метаданных каталога.
    """

    def __init__(self, recognizer):
        self.recognizer = recognizer

        # catalog_meta.json содержит 2037 изображений,
        # а CSV — 2103 записи.
        #
        # Для визуальных кандидатов используем meta,
        # где есть slug -> файл.
        self.meta_by_slug = {}

        for item in recognizer.meta:
            slug = item.get("slug")

            if slug:
                self.meta_by_slug[slug] = item

        # Попробуем использовать расширенные wine metadata,
        # если они уже загружены recognizer'ом.
        self.wine_metadata = getattr(
            recognizer,
            "wine_metadata",
            {}
        )

    # =========================================================
    # TEXT
    # =========================================================

    @staticmethod
    def normalize_text(text):
        if text is None:
            return ""

        text = str(text).lower()

        # Ё -> Е
        text = text.replace("ё", "е")

        # латиница/кириллица/цифры
        text = re.sub(
            r"[^a-zа-я0-9]+",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    @classmethod
    def text_similarity(cls, a, b):
        a = cls.normalize_text(a)
        b = cls.normalize_text(b)

        if not a or not b:
            return 0.0

        if a == b:
            return 1.0

        # Полное вхождение
        if a in b or b in a:
            return 0.9

        return SequenceMatcher(
            None,
            a,
            b
        ).ratio()

    @classmethod
    def token_similarity(cls, ocr_text, target_text):
        ocr = cls.normalize_text(ocr_text)
        target = cls.normalize_text(target_text)

        if not ocr or not target:
            return 0.0

        ocr_tokens = set(ocr.split())
        target_tokens = set(target.split())

        if not ocr_tokens or not target_tokens:
            return 0.0

        exact = len(
            ocr_tokens.intersection(target_tokens)
        )

        if exact:
            return exact / len(target_tokens)

        # Для OCR с ошибками сравниваем токены
        best = 0.0

        for ot in ocr_tokens:
            for tt in target_tokens:
                similarity = SequenceMatcher(
                    None,
                    ot,
                    tt
                ).ratio()

                if similarity > best:
                    best = similarity

        return best

    # =========================================================
    # METADATA
    # =========================================================

    def get_metadata(self, slug):
        """
        Возвращает метаданные вина.

        Поддерживает несколько возможных форматов,
        чтобы не зависеть от конкретной реализации recognizer.py.
        """

        metadata = self.wine_metadata

        if isinstance(metadata, dict):

            item = metadata.get(slug)

            if item:
                return item

        # Если recognizer хранит список
        if isinstance(metadata, list):

            for item in metadata:

                if not isinstance(item, dict):
                    continue

                if item.get("Slug") == slug:
                    return item

                if item.get("slug") == slug:
                    return item

        return {}

    # =========================================================
    # OCR
    # =========================================================

    def calculate_ocr_score(
        self,
        ocr_text,
        metadata
    ):
        if not ocr_text:
            return 0.0

        if isinstance(ocr_text, dict):

            # Возможные варианты результата OCR
            text = (
                ocr_text.get("text")
                or ocr_text.get("raw_text")
                or ocr_text.get("ocr_text")
                or ""
            )

        else:
            text = str(ocr_text)

        if not text:
            return 0.0

        fields = [
            metadata.get("Название вина"),
            metadata.get("Винодельня"),
            metadata.get("Сорт винограда"),
            metadata.get("Регион"),
            metadata.get("Цвет"),
            metadata.get("Категория"),
        ]

        fields = [
            x for x in fields
            if x
        ]

        if not fields:
            return 0.0

        scores = []

        for field in fields:

            score = max(
                self.text_similarity(text, field),
                self.token_similarity(text, field)
            )

            scores.append(score)

        if not scores:
            return 0.0

        # Самое информативное совпадение имеет больший вес
        scores.sort(reverse=True)

        if len(scores) >= 3:
            return (
                scores[0] * 0.60
                + scores[1] * 0.25
                + scores[2] * 0.15
            )

        if len(scores) == 2:
            return (
                scores[0] * 0.70
                + scores[1] * 0.30
            )

        return scores[0]

    # =========================================================
    # RERANK
    # =========================================================

    def rerank(
        self,
        candidates,
        ocr_text=None
    ):
        """
        candidates:

        [
            {
                "slug": "...",
                "score": 0.81
            }
        ]

        Возвращает новый рейтинг.
        """

        if not candidates:
            return []

        result = []

        # -----------------------------------------------------
        # Нормализуем visual score
        # -----------------------------------------------------

        visual_scores = []

        for candidate in candidates:

            score = candidate.get("score")

            if score is not None:
                visual_scores.append(float(score))

        if visual_scores:

            min_score = min(visual_scores)
            max_score = max(visual_scores)

        else:
            min_score = 0.0
            max_score = 1.0

        # -----------------------------------------------------
        # Каждый кандидат
        # -----------------------------------------------------

        for candidate in candidates:

            slug = candidate.get("slug")

            if not slug:
                continue

            raw_visual = candidate.get("score")

            if raw_visual is None:
                raw_visual = 0.0

            raw_visual = float(raw_visual)

            # Нормализация visual score относительно Top-K
            if max_score > min_score:
                visual_norm = (
                    raw_visual - min_score
                ) / (
                    max_score - min_score
                )
            else:
                visual_norm = 1.0

            metadata = self.get_metadata(slug)

            ocr_score = self.calculate_ocr_score(
                ocr_text,
                metadata
            )

            # -------------------------------------------------
            # Финальный score
            #
            # Основной вес всё ещё у изображения.
            # OCR пока только корректирует рейтинг.
            # -------------------------------------------------

            final_score = (
                visual_norm * 0.85
                + ocr_score * 0.15
            )

            result.append({
                "slug": slug,
                "visual_score": round(
                    raw_visual,
                    6
                ),
                "visual_normalized": round(
                    visual_norm,
                    6
                ),
                "ocr_score": round(
                    ocr_score,
                    6
                ),
                "final_score": round(
                    final_score,
                    6
                )
            })

        result.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return result