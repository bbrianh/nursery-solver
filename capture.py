"""Capture the game window: screenshot and normalize to internal resolution (matches ``sqinfo`` calibration)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui

# Canonical size used by the upstream WeChat mini-game bot (calibrate ``sqinfo`` on this image).
INTERNAL_WIDTH = 450
INTERNAL_HEIGHT = 844

@dataclass
class CaptureConfig:
    """Fixed screen region: ``(left, top, width, height)`` in screen coordinates."""

    region: Tuple[int, int, int, int]
    resize_to: Optional[Tuple[int, int]] = None
    pause_before: float = 0.25


@dataclass(frozen=True)
class WindowCaptureResult:
    """BGR image at internal size and how to map its coordinates to the real window."""

    image_bgr: np.ndarray
    anchor: Tuple[float, float]
    owidth: int
    oheight: int
    internal_width: int
    internal_height: int
    stretch: bool
    letterbox_pad_x: int
    letterbox_pad_y: int
    letterbox_scale: float


def capture_bgr(config: CaptureConfig) -> np.ndarray:
    """Return a BGR ``uint8`` image. Optionally resize."""
    time.sleep(config.pause_before)
    left, top, w, h = config.region
    shot = pyautogui.screenshot(region=(left, top, w, h))
    bgr = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    if config.resize_to is not None:
        rw, rh = config.resize_to
        bgr = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return bgr


def screen_origin_from_region(config: CaptureConfig) -> Tuple[float, float]:
    left, top, _, _ = config.region
    return (float(left), float(top))


def find_windows_game_window(title: str = "开局托儿所") -> Tuple[Tuple[int, int], int, int]:
    """Windows: find window by title, return ``(anchor_left, anchor_top), width, height``."""
    try:
        import pygetwindow as gw
    except ImportError as e:
        raise RuntimeError(
            "Window capture on Windows requires PyGetWindow. Install with: pip install PyGetWindow"
        ) from e

    matches = gw.getWindowsWithTitle(title)
    if not matches:
        raise RuntimeError(
            f"No window with title {title!r}. Open the mini-game in WeChat and try again."
        )
    win = matches[0]
    anchor = (int(win.left), int(win.top))
    return anchor, int(win.width), int(win.height)


def capture_game_window(
    *,
    window_title: str = "开局托儿所",
    internal_width: int = INTERNAL_WIDTH,
    internal_height: int = INTERNAL_HEIGHT,
    pause_before: float = 1.0,
    stretch_to_internal: bool = False,
) -> WindowCaptureResult:
    """
    Locate the game window, screenshot it, map to ``internal_width`` × ``internal_height``.

    By default uses **letterboxing** (uniform scale + black padding) so the image is not squashed.
    Pass ``stretch_to_internal=True`` for independent X/Y resize to fill the canvas (reference-bot style;
    can distort aspect ratio).
    """
    if sys.platform.startswith("win"):
        anchor, owidth, oheight = find_windows_game_window(window_title)
    else:
        raise RuntimeError(
            "Automatic window capture is only implemented for Windows. "
            "Use capture_bgr(CaptureConfig(region=(l,t,w,h), resize_to=(iw, ih))) manually."
        )

    # resize the window to the internal width and height
    scale = owidth / internal_width * 1.
    if int(scale * internal_height) != int(oheight):
        delta = int(oheight - scale * internal_height)
        anchor = (anchor[0], anchor[1] + delta//2)
        oheight = int(oheight - delta//2)

    time.sleep(pause_before)
    left, top = anchor
    shot = pyautogui.screenshot(region=(left, top, owidth, oheight))
    bgr = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)

    image = cv2.resize(
        bgr, (internal_width, internal_height), interpolation=cv2.INTER_LINEAR
    )
    return WindowCaptureResult(
        image_bgr=image,
        anchor=(float(anchor[0]), float(anchor[1])),
        owidth=owidth,
        oheight=oheight,
        internal_width=internal_width,
        internal_height=internal_height,
        stretch=True,
        letterbox_pad_x=0,
        letterbox_pad_y=0,
        letterbox_scale=1.0,
    )
