"""Digit recognition — same logic as the provided reference (``template.pkl`` + matchTemplate)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

_DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "resource" / "template.pkl"

_template: Optional[Any] = None


def load_templates(path: Optional[Path] = None) -> Any:
    """Load ``template.pkl`` (same as ``template = pickle.load(open('template.pkl','rb'))``)."""
    global _template
    p = path or _DEFAULT_TEMPLATE_PATH
    if _template is not None:
        return _template
    if not p.is_file():
        _template = {}
        return _template
    with open(p, "rb") as f:
        _template = pickle.load(f)
    return _template


def recognize_digit(image: np.ndarray, templates: Optional[Any] = None) -> int:
    template = templates if templates is not None else load_templates()
    if not template or image.size == 0:
        return 0

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_ = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    scores = np.zeros(10)
    for number, template_img in template.items():
        score = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
        scores[int(number)] = np.max(score)
    if np.max(scores) < 200000:
        # Reference: print('识别出错！')
        print("recognition error: low match score")
    return int(np.argmax(scores))
