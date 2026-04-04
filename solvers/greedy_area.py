from __future__ import annotations
from typing import List, Tuple
import numpy as np
from models import Move, ROWS, COLS
from solver_base import Solver


class GreedyAreaSolver(Solver):
    """Greedy solver: remove smallest-area rectangles/pairs first to avoid blocking."""

    def __init__(self, grid: np.ndarray, debug: bool = False):
        super().__init__(grid, debug)

    def plan(self) -> None:
        self._plan_recursive(self.grid)

    def _plan_recursive(self, grid: np.ndarray) -> None:
        moves: List[Tuple[int, str, Tuple[int,int,int,int]]] = []

        # 1. Rectangles
        for x_len in range(1, ROWS+1):
            for y_len in range(1, COLS+1):
                area = x_len * y_len
                for bx in range(ROWS - x_len + 1):
                    for by in range(COLS - y_len + 1):
                        if np.sum(grid[bx:bx+x_len, by:by+y_len]) == 10:
                            moves.append((area, 'block', (bx, bx+x_len, by, by+y_len)))

        # 2. Straight-line pairs (right/down)
        for bx in range(ROWS):
            for by in range(COLS):
                val1 = grid[bx, by]
                if val1 == 0:
                    continue
                for dx, dy in [(1, 0), (0, 1)]:
                    nx, ny = bx + dx, by + dy
                    while 0 <= nx < ROWS and 0 <= ny < COLS:
                        val2 = grid[nx, ny]
                        if val2 == 0:
                            nx += dx
                            ny += dy
                            continue
                        if val1 + val2 == 10:
                            area = (abs(nx - bx) + 1) * (abs(ny - by) + 1)
                            moves.append((area, 'pair', (bx, nx, by, ny)))
                        break

        if not moves:
            return  # no moves left

        # Sort by area ascending (smallest area first)
        moves.sort(key=lambda x: x[0])

        # Execute moves greedily
        for _, m_type, coords in moves:
            if m_type == 'block':
                bx, ex, by, ey = coords
                if np.sum(grid[bx:ex, by:ey]) != 10:
                    continue
                move = Move(bx, by, ex, ey)
                self.apply_move(move)
                grid[bx:ex, by:ey] = 0
                self._plan_recursive(grid)
                return
            elif m_type == 'pair':
                bx, nx, by, ny = coords
                if grid[bx, by] == 0 or grid[nx, ny] == 0 or grid[bx, by] + grid[nx, ny] != 10:
                    continue
                move = Move(min(bx, nx), min(by, ny), max(bx, nx)+1, max(by, ny)+1)
                self.apply_move(move)
                grid[bx, by] = 0
                grid[nx, ny] = 0
                self._plan_recursive(grid)
                return