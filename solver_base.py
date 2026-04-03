"""Abstract solver API: strategies return an ordered list of moves with no I/O."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from models import COLS, ROWS


class Solver(ABC):
    """Pluggable strategy: given the current grid, produce moves in play order.

    - ``grid`` is shape (ROWS, COLS); cleared cells are ``0``.
    - Returned moves are applied sequentially by the game; subclasses must not
      perform mouse, file, or network operations.
    """

    @abstractmethod
    def plan(self, grid: np.ndarray) -> List[Move]:
        raise NotImplementedError

    def validate_grid_shape(self, grid: np.ndarray) -> None:
        if grid.shape != (ROWS, COLS):
            raise ValueError(f"Expected grid shape ({ROWS}, {COLS}), got {grid.shape}")
