"""Execute planned grid moves as mouse drags in screen coordinates."""

from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import pyautogui

from models import GridGeometry, Move


class MoveExecutor:
    """Maps each ``Move`` to a drag from top-left to bottom-right of the rectangle."""

    def __init__(
        self,
        geometry: GridGeometry,
        screen_origin: Tuple[float, float],
        *,
        scale_xy: Tuple[float, float] = (1.0, 1.0),
        letterbox: Optional[Tuple[float, float, float]] = None,
        drag_duration: float = 0.15,
        pause_between_moves: float = 0.08,
        failsafe: bool = True,
    ) -> None:
        """
        :param geometry: Calibration in **canvas** pixel space (same as ``sqinfo``).
        :param screen_origin: Window top-left ``(left, top)`` in screen coordinates.
        :param scale_xy: Used when ``letterbox`` is ``None``: map canvas x/y by these
            factors then add origin (independent x/y; can squash if aspect ratios differ).
        :param letterbox: If set, ``(pad_x, pad_y, uniform_scale)`` from
            :func:`capture.letterbox_to_internal` — maps canvas coords to screenshot
            pixels without distortion.
        """
        self.geometry = geometry
        self._ox, self._oy = screen_origin
        self._sx, self._sy = scale_xy
        self._letterbox = letterbox
        self.drag_duration = drag_duration
        self.pause_between_moves = pause_between_moves
        pyautogui.FAILSAFE = failsafe

    def _to_screen(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        x, y = xy
        if self._letterbox is not None:
            pad_x, pad_y, scale = self._letterbox
            ox = (x - pad_x) / scale
            oy = (y - pad_y) / scale
            return int(self._ox + round(ox)), int(self._oy + round(oy))
        sx = self._ox + round(x * self._sx)
        sy = self._oy + round(y * self._sy)
        return int(sx), int(sy)

    def execute_moves(self, moves: Sequence[Move]) -> None:
        """Perform each move as a left-button drag, in order."""
        for move in moves:
            start, end = self.geometry.move_drag_endpoints(move)
            sx, sy = self._to_screen(start)
            ex, ey = self._to_screen(end)
            pyautogui.moveTo(sx, sy, duration=0.05)
            pyautogui.dragTo(ex, ey, duration=self.drag_duration, button="left")
            time.sleep(self.pause_between_moves)
