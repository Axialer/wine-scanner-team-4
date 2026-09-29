from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


# Misses stay cached so the same unknown code is not fetched again.
NEGATIVE_TTL_SECONDS = 7 * 24 * 60 * 60
NEGATIVE_STATUSES = frozenset({"not_found", "empty", "non_html"})


class ProductCache:
    """Local JSON cache for barcode/QR lookup results."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key_path(cache_dir: Path, key: str) -> Path:
        digest = hashlib.sha256(
            key.encode("utf-8", errors="ignore")
        ).hexdigest()
        return cache_dir / f"{digest}.json"

    def get(self, key: str) -> dict | None:
        path = self._key_path(self.cache_dir, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def get_usable(self, key: str) -> dict | None:
        """Return a hit without touching the network.

        Successes are kept. Negative lookups expire after about 7 days.
        Transport errors are not reused.
        """

        data = self.get(key)
        if not data:
            return None

        status = str(data.get("status") or "")
        if status == "found":
            data["from_cache"] = True
            return data

        if status in NEGATIVE_STATUSES:
            cached_at = float(data.get("cached_at") or 0)
            try:
                ttl = float(data.get("negative_ttl") or NEGATIVE_TTL_SECONDS)
            except (TypeError, ValueError):
                ttl = float(NEGATIVE_TTL_SECONDS)
            age = time.time() - cached_at
            if cached_at and age <= ttl:
                data["from_cache"] = True
                return data
            return None

        return None

    def set(self, key: str, data: dict) -> None:
        path = self._key_path(self.cache_dir, key)
        payload = dict(data)
        payload["cached_at"] = time.time()
        payload["from_cache"] = False
        try:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
