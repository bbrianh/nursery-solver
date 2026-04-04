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
    def __init__(self, grid: np.ndarray, debug: bool = False) -> None:
        self.grid = grid
        self.solution = []
        self.validate_grid_shape()
        self.debug = debug

    @abstractmethod
    def plan(self) -> None:
        raise NotImplementedError
    
    def get_solution(self) -> List[Move]:
        return self.solution

    def validate_grid_shape(self) -> None:
        if self.grid.shape != (ROWS, COLS):
            raise ValueError(f"Expected grid shape ({ROWS}, {COLS}), got {grid.shape}")
    
    def is_valid_move(self, move: Move) -> None:
        return np.sum(self.grid[move.r1:move.r2, move.c1:move.c2]) == 10
    
    def apply_move(self, move: Move) -> None:
        assert self.is_valid_move(move)
        self.grid[move.r1:move.r2, move.c1:move.c2] = 0
        self.solution.append(move)
        if self.debug:
            print(f"Applied move: {move}")
            print(f"Grid after move: \n{self.grid}")

    def score(self) -> int:
        return ROWS * COLS - np.count_nonzero(self.grid)