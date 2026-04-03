from __future__ import annotations

from typing import List
import random

import numpy as np

from models import Move
from solver_base import Solver


class RandomSolver(Solver):
    def __init__(self, grid: np.ndarray, debug: bool = False) -> None:
        super().__init__(grid, debug)

    def plan(self) -> List[Move]:
        next_moves = self.find_all_moves()
        while next_moves:
            move = random.choice(next_moves)
            if self.is_valid_move(move):
                self.apply_move(move)
            next_moves = self.find_all_moves()
        return self.solution

    def compute_prefix(self):
        return self.grid.cumsum(axis=0).cumsum(axis=1)

    def rect_sum(self, prefix, r1, c1, r2, c2):
        total = prefix[r2, c2]
        if r1 > 0: total -= prefix[r1-1, c2]
        if c1 > 0: total -= prefix[r2, c1-1]
        if r1 > 0 and c1 > 0: total += prefix[r1-1, c1-1]
        return total

    def trim_bbox(self, r1, r2, c1, c2):
        sub = self.grid[r1:r2+1, c1:c2+1]
        rows_nonzero = np.any(sub != 0, axis=1)
        cols_nonzero = np.any(sub != 0, axis=0)

        if not rows_nonzero.any() or not cols_nonzero.any():
            return None

        rr1 = r1 + np.argmax(rows_nonzero)
        rr2 = r1 + len(rows_nonzero) - np.argmax(rows_nonzero[::-1])
        cc1 = c1 + np.argmax(cols_nonzero)
        cc2 = c1 + len(cols_nonzero) - np.argmax(cols_nonzero[::-1])

        return (rr1, rr2, cc1, cc2)

    def find_all_moves(self):
        rows, cols = self.grid.shape
        prefix = self.compute_prefix()
        moves_set = set()

        for r1 in range(rows):
            for r2 in range(r1, rows):
                for c1 in range(cols):
                    for c2 in range(c1, cols):
                        if self.rect_sum(prefix, r1, c1, r2, c2) == 10:
                            trimmed = self.trim_bbox(r1, r2, c1, c2)
                            if trimmed:
                                moves_set.add(trimmed)

        moves = []
        for rr1, rr2, cc1, cc2 in moves_set:
            sub = self.grid[rr1:rr2, cc1:cc2]
            moves.append(Move(r1=rr1, r2=rr2, c1=cc1, c2=cc2))

        return moves
