import os
from typing import List, Tuple

import cv2 as cv
import numpy as np


def _preprocess(gray: np.ndarray) -> np.ndarray:
    if len(gray.shape) == 3:
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh


def _iou(a: dict, b: dict) -> float:
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    return inter / float(area_a + area_b - inter)


class ScoreReader:
    def __init__(self, templates_dir: str, match_threshold: float = 0.7):
        self.templates_dir = templates_dir
        self.match_threshold = match_threshold
        self.templates = self._load_templates()

    def _load_templates(self) -> List[Tuple[str, np.ndarray]]:
        templates: List[Tuple[str, np.ndarray]] = []
        for digit in range(10):
            path = os.path.join(self.templates_dir, f"{digit}.png")
            if not os.path.exists(path):
                continue
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            templates.append((str(digit), _preprocess(img)))
        if not templates:
            raise ValueError(f"No digit templates found in {self.templates_dir}")
        return templates

    def read(self, image: np.ndarray) -> int | None:
        proc = _preprocess(image)
        matches = []
        for digit, tmpl in self.templates:
            res = cv.matchTemplate(proc, tmpl, cv.TM_CCOEFF_NORMED)
            loc = np.where(res >= self.match_threshold)
            for pt in zip(*loc[::-1]):
                matches.append(
                    {
                        "digit": digit,
                        "x": int(pt[0]),
                        "y": int(pt[1]),
                        "w": int(tmpl.shape[1]),
                        "h": int(tmpl.shape[0]),
                        "score": float(res[pt[1], pt[0]]),
                    }
                )
        if not matches:
            return None

        matches.sort(key=lambda m: m["score"], reverse=True)
        kept: List[dict] = []
        for m in matches:
            if all(_iou(m, k) < 0.3 for k in kept):
                kept.append(m)

        kept.sort(key=lambda m: m["x"])
        digits = [m["digit"] for m in kept]
        if not digits:
            return None
        return int("".join(digits))
