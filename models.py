"""Shared types for grid moves and screen geometry (decouples solver from vision/input)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class Move:
    """Axis-aligned rectangle on the digit grid, inclusive indices.

    Rows 0..ROWS-1, columns 0..COLS-1. A move eliminates this rectangle when
    the sum of cell values (0 for empty) equals 10.
    """

    r1: int
    c1: int
    r2: int
    c2: int

    def __post_init__(self) -> None:
        if self.r1 > self.r2 or self.c1 > self.c2:
            raise ValueError(f"Move corners out of order: {self}")


# Default board shape (开局托儿所–style grid)
ROWS = 16
COLS = 10


@dataclass
class GridGeometry:
    """Maps grid indices to pixel coordinates in the calibrated screenshot space."""

    anchor_x: float
    anchor_y: float
    h: float
    v: float
    hwidth: float
    vwidth: float
    rows: int = ROWS
    cols: int = COLS

    @classmethod
    def from_sqinfo(cls, sqinfo: Mapping[str, Any]) -> GridGeometry:
        return cls(
            anchor_x=float(sqinfo["anchor_x"]),
            anchor_y=float(sqinfo["anchor_y"]),
            h=float(sqinfo["h"]),
            v=float(sqinfo["v"]),
            hwidth=float(sqinfo["hwidth"]),
            vwidth=float(sqinfo["vwidth"]),
        )

    def cell_bbox(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """Top-left inclusive, bottom-right exclusive style corners (x1, y1, x2, y2)."""
        x1 = self.anchor_x + col * self.h
        y1 = self.anchor_y + row * self.v
        x2 = self.anchor_x + self.hwidth + col * self.h
        y2 = self.anchor_y + self.vwidth + row * self.v
        return (x1, y1, x2, y2)

    def move_bbox(self, move: Move) -> Tuple[float, float, float, float]:
        """Bounding box covering cells (r1,c1) through (r2,c2) inclusive."""
        x1, y1, _, _ = self.cell_bbox(move.r1, move.c1)
        _, _, x2, y2 = self.cell_bbox(move.r2, move.c2)
        return (x1, y1, x2, y2)

    def move_drag_endpoints(self, move: Move) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Start at top-left of selection, end at bottom-right (for drag selection)."""
        x1, y1, x2, y2 = self.move_bbox(move)
        return (x1, y1), (x2, y2)
