"""Pick the dominant bottle or label before OCR.

OpenCV contours only: no extra detector download. When two bottle-like
regions are visible, the larger (or, if sizes are close, the more
central) one is kept and the partial second bottle is ignored.
"""

from __future__ import annotations

import time

from PIL import Image

import cv2
import numpy as np


DETECT_MAX_SIDE = 480
MIN_REGION_AREA = 0.08
MAX_REGION_AREA = 0.96
PARTIAL_WIDTH = 0.18

# Rotate only a clear tilt. Near-zero is noise; a steep angle is the
# bottle silhouette, not a line of text.
DESKEW_MIN_DEG = 2.0
DESKEW_MAX_DEG = 25.0

# Lower-middle of the label, where the second grape is printed
# (ЦИТРОН / СОВИНЬОН БЛАН). Same fractions for every photo.


def prepare_label(
    image: Image.Image,
    ocr_max_side: int = 800,
) -> dict:
    """Return an OCR-sized crop and the image used for SigLIP."""

    image = image.convert("RGB")
    width, height = image.size
    rgb = np.array(image)

    boxes, strategy = _regions(rgb)
    chosen = _choose_box(boxes, width, height)
    trimmed_sides = 0

    if chosen is None:
        crop_box = _center_box(width, height)
        strategy = "center"
        region_count = 0
    else:
        chosen, trimmed_sides = _trim_side_bottles(rgb, chosen)
        crop_box = _pad_box(chosen, width, height, pad=0.02)
        region_count = len(boxes) + trimmed_sides
        # A single close-up that fills the frame still benefits from
        # a label band; two regions keep the primary bottle intact.
        if region_count < 2 and _coverage(crop_box, width, height) > 0.82:
            crop_box = _center_box(width, height)
            strategy = "center"
        elif trimmed_sides:
            strategy = "primary_bottle"

    x1, y1, x2, y2 = crop_box
    cropped = _downscale(image.crop((x1, y1, x2, y2)), ocr_max_side)
    started = time.perf_counter()
    aligned, deskew_deg = _deskew(cropped)
    ocr_image = _photometric(aligned)
    preprocess_ms = (time.perf_counter() - started) * 1000.0

    return {
        "visual": aligned,
        "ocr": ocr_image,
        "debug": {
            "box": [x1, y1, x2, y2],
            "regions": region_count,
            "strategy": strategy,
            "deskew_deg": round(deskew_deg, 2),
            "preprocess_ms": round(preprocess_ms, 1),
        },
    }


def _deskew(image: Image.Image) -> tuple[Image.Image, float]:
    """Rotate when label text is clearly off-horizontal. Otherwise leave it."""

    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if width < 40 or height < 40:
        return image, 0.0

    angle = _text_angle(rgb)
    if angle is None or abs(angle) < DESKEW_MIN_DEG or abs(angle) > DESKEW_MAX_DEG:
        return image, 0.0

    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(rotated), angle


def _text_angle(rgb: np.ndarray) -> float | None:
    """Median angle of long near-horizontal edges. None when the vote is thin."""

    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    min_len = max(24, int(width * 0.12))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=30,
        minLineLength=min_len,
        maxLineGap=12,
    )
    if lines is None:
        return None

    lines = np.asarray(lines).reshape(-1, 4)
    angles: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in lines:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = (dx * dx + dy * dy) ** 0.5
        if length < min_len:
            continue
        degrees = float(np.degrees(np.arctan2(dy, dx)))
        if degrees > 90.0:
            degrees -= 180.0
        elif degrees < -90.0:
            degrees += 180.0
        if abs(degrees) > DESKEW_MAX_DEG + 5.0:
            continue
        angles.append(degrees)
        weights.append(length)

    if len(angles) < 4:
        return None

    order = np.argsort(angles)
    sorted_w = np.asarray(weights, dtype=np.float32)[order]
    cumulative = np.cumsum(sorted_w)
    midpoint = float(cumulative[-1]) * 0.5
    index = int(np.searchsorted(cumulative, midpoint))
    index = min(index, len(order) - 1)
    return float(np.asarray(angles)[order][index])


def _photometric(image: Image.Image) -> Image.Image:
    """CLAHE on luminance only, so yellow phone photos stay color for OCR."""

    rgb = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    lab = cv2.merge((lightness, a, b))
    boosted = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(boosted, (0, 0), 1.0)
    sharp = cv2.addWeighted(boosted, 1.3, blur, -0.3, 0)
    return Image.fromarray(sharp)


def _downscale(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    if side <= max_side:
        return image
    scale = max_side / side
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BILINEAR,
    )


def _center_box(width: int, height: int) -> tuple[int, int, int, int]:
    crop_h = int(height * 0.72)
    top = max(0, (height - crop_h) // 2)
    return 0, top, width, min(height, top + crop_h)


def _coverage(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> float:
    x1, y1, x2, y2 = box
    return ((x2 - x1) * (y2 - y1)) / max(width * height, 1)


def _regions(
    rgb: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], str]:
    height, width = rgb.shape[:2]
    scale = 1.0
    view = rgb
    side = max(width, height)
    if side > DETECT_MAX_SIDE:
        scale = DETECT_MAX_SIDE / side
        view = cv2.resize(
            rgb,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    edge_boxes = _edge_boxes(view)
    bright_boxes = _bright_boxes(view)
    boxes = edge_boxes if len(edge_boxes) >= len(bright_boxes) else bright_boxes
    strategy = "edges" if boxes is edge_boxes and boxes else "bright"
    if not boxes:
        return [], "none"

    mapped = [_scale_box(box, 1.0 / scale, width, height) for box in boxes]
    mapped = _drop_partials(mapped, width)
    return mapped, strategy if mapped else "none"


def _edge_boxes(
    rgb: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 17))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return _filter_contours(contours, width, height)


def _bright_boxes(
    rgb: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 165, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return _filter_contours(contours, width, height)


def _filter_contours(
    contours,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    area_total = float(width * height)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area_ratio = (box_w * box_h) / area_total
        if area_ratio < MIN_REGION_AREA or area_ratio > MAX_REGION_AREA:
            continue
        if box_h < height * 0.28:
            continue
        boxes.append((x, y, x + box_w, y + box_h))
    return _merge_overlaps(boxes)


def _merge_overlaps(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    if len(boxes) < 2:
        return boxes

    pending = list(boxes)
    merged: list[tuple[int, int, int, int]] = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            rest: list[tuple[int, int, int, int]] = []
            for other in pending:
                if _iou(current, other) >= 0.35:
                    current = _union(current, other)
                    changed = True
                else:
                    rest.append(other)
            pending = rest
        merged.append(current)
    return merged


def _drop_partials(
    boxes: list[tuple[int, int, int, int]],
    width: int,
) -> list[tuple[int, int, int, int]]:
    """Ignore a thin sliver of a second bottle at the frame edge."""

    if len(boxes) < 2:
        return boxes

    widest = max(box[2] - box[0] for box in boxes)
    kept = []
    for box in boxes:
        box_w = box[2] - box[0]
        if box_w < PARTIAL_WIDTH * width and box_w < widest * 0.45:
            continue
        kept.append(box)
    return kept or boxes


def _trim_side_bottles(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], int]:
    """Drop a narrow second bottle touching the left or right edge.

    A real gap is a vertical edge-valley much quieter than the label,
    with a leftover strip that is clearly the smaller object.
    """

    x1, y1, x2, y2 = box
    crop = rgb[y1:y2, x1:x2]
    height, width = crop.shape[:2]
    if width < 80 or height < 80:
        return box, 0

    view = crop
    if max(width, height) > DETECT_MAX_SIDE:
        scale = DETECT_MAX_SIDE / max(width, height)
        view = cv2.resize(
            crop,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 130)
    column = edges.mean(axis=0).astype(np.float32)
    kernel = max(5, column.size // 40)
    smooth = np.convolve(column, np.ones(kernel) / kernel, mode="same")
    view_w = int(smooth.size)
    center = smooth[int(view_w * 0.35):int(view_w * 0.65)]
    center_mean = float(center.mean()) if center.size else float(smooth.mean())
    if center_mean <= 1.0:
        return box, 0

    removed = 0
    span = x2 - x1

    def cut_at(fraction: float, side: str) -> None:
        nonlocal x1, x2, removed
        if side == "right":
            cut = x1 + int(span * fraction)
            side_w = x2 - cut
            main_w = cut - x1
            if main_w > side_w and 0.08 * span <= side_w <= 0.32 * span:
                x2 = cut
                removed += 1
        else:
            cut = x1 + int(span * fraction)
            side_w = cut - x1
            main_w = x2 - cut
            if main_w > side_w and 0.08 * span <= side_w <= 0.32 * span:
                x1 = cut
                removed += 1

    right_lo = int(view_w * 0.70)
    right_hi = int(view_w * 0.94)
    if right_hi > right_lo + 2:
        segment = smooth[right_lo:right_hi]
        index = int(np.argmin(segment)) + right_lo
        if float(smooth[index]) < center_mean * 0.40:
            cut_at(index / view_w, "right")

    left_lo = int(view_w * 0.06)
    left_hi = int(view_w * 0.30)
    if left_hi > left_lo + 2 and removed == 0:
        segment = smooth[left_lo:left_hi]
        index = int(np.argmin(segment)) + left_lo
        if float(smooth[index]) < center_mean * 0.40:
            cut_at(index / view_w, "left")

    return (x1, y1, x2, y2), removed


def _choose_box(
    boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]

    def score(box: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        area = float((x2 - x1) * (y2 - y1))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = (cx - width / 2.0) / max(width, 1)
        dy = (cy - height / 2.0) / max(height, 1)
        distance = (dx * dx + dy * dy) ** 0.5
        return area * (1.0 - 0.35 * min(distance, 1.0))

    ranked = sorted(boxes, key=score, reverse=True)
    largest = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    largest_area = (largest[2] - largest[0]) * (largest[3] - largest[1])
    best = ranked[0]
    best_area = (best[2] - best[0]) * (best[3] - best[1])
    if best_area < largest_area * 0.85:
        return largest
    return best


def _pad_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    pad: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    px = int(width * pad)
    py = int(height * pad)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(width, x2 + px),
        min(height, y2 + py),
    )


def _scale_box(
    box: tuple[int, int, int, int],
    scale: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, int(x1 * scale))),
        max(0, min(height, int(y1 * scale))),
        max(0, min(width, int(x2 * scale))),
        max(0, min(height, int(y2 * scale))),
    )


def _iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _union(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )
