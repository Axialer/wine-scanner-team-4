"""Short HTTP GET used by barcode and QR lookups."""

from __future__ import annotations

from urllib.request import Request, urlopen


USER_AGENT = "WineScanner/1.0 (local catalog matcher)"
MAX_BYTES = 1_500_000


def clamp_timeout(
    timeout: float,
    *,
    floor: float = 1.5,
    ceiling: float = 2.5,
) -> float:
    """External calls stay inside a short window. Default floor is 1.5s.

    A multi-step web search passes a lower floor so the last fetch can
    use whatever time is left before the shared 2.5s deadline.
    """

    try:
        value = float(timeout)
    except (TypeError, ValueError):
        value = 2.0
    return min(ceiling, max(floor, value))


def fetch_url(
    url: str,
    timeout: float,
    *,
    floor: float = 1.5,
    ceiling: float = 2.5,
    data: bytes | None = None,
    headers: dict | None = None,
) -> tuple[int, str, bytes]:
    hdrs = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    }
    if headers:
        hdrs.update(headers)
    if data is not None and "Content-Type" not in hdrs:
        hdrs["Content-Type"] = "application/x-www-form-urlencoded"
    request = Request(url, data=data, headers=hdrs)
    with urlopen(request, timeout=clamp_timeout(timeout, floor=floor, ceiling=ceiling)) as response:
        status = int(getattr(response, "status", 200) or 200)
        content_type = response.headers.get("Content-Type", "")
        body = response.read(MAX_BYTES)
    return status, content_type, body
