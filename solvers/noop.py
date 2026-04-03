"""Baseline solver that plans nothing (useful for testing wiring)."""

from __future__ import annotations

from typing import List

import numpy as np

from models import Move
from solver_base import Solver


class NoOpSolver(Solver):
    def plan(self, grid: np.ndarray) -> List[Move]:
        self.validate_grid_shape(grid)
        return []
