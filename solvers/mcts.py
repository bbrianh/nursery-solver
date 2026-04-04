from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import random
import math
from models import Move, ROWS, COLS
from solver_base import Solver

@dataclass
class MCTSNode:
    grid: np.ndarray
    parent: Optional[MCTSNode] = None
    move: Optional[Move] = None
    children: List[MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_moves: List[Move] = field(default_factory=list)

    def ucb1(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.total_reward / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)


class MCTSSolver(Solver):
    """Monte Carlo Tree Search solver with fast move finder and beam pruning."""

    def __init__(
        self,
        grid: np.ndarray,
        debug: bool = False,
        rollout_depth: int = 20,
        rollout_count: int = 100,
        beam_width: int = 20,
    ):
        super().__init__(grid, debug)
        self.rollout_depth = rollout_depth
        self.rollout_count = rollout_count
        self.beam_width = beam_width

    def plan(self) -> None:
        """Plan all moves until grid is cleared."""
        root = MCTSNode(grid=self.grid.copy())
        root.untried_moves = self.find_moves(root.grid)

        while True:
            if not root.untried_moves and not root.children:
                break  # no moves left

            # Perform MCTS iterations for this step
            iterations = max(20, len(root.untried_moves) * 2)
            for _ in range(iterations):
                node = self.select(root)
                reward = self.rollout(node.grid)
                self.backpropagate(node, reward)

            # Choose the best child from root
            if not root.children:
                break
            best_child = max(root.children, key=lambda n: n.visits)
            if best_child.move is None:
                break

            # Apply the move to actual solution
            self.apply_move(best_child.move)
            root = MCTSNode(grid=best_child.grid.copy())
            root.untried_moves = self.find_moves(root.grid)

    # -----------------------------
    # MCTS core methods
    # -----------------------------
    def select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCB1."""
        while True:
            if node.untried_moves:
                return self.expand(node)
            if not node.children:
                return node
            node = max(node.children, key=lambda n: n.ucb1())

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by trying one untried move."""
        move_score, (r1, r2, c1, c2) = node.untried_moves.pop(0)
        move = Move(r1, c1, r2, c2)
        new_grid = node.grid.copy()
        new_grid[r1:r2, c1:c2] = 0
        child = MCTSNode(grid=new_grid, parent=node, move=move)
        child.untried_moves = self.find_moves(new_grid)
        node.children.append(child)
        return child

    def rollout(self, grid: np.ndarray) -> int:
        """Simulate a random sequence of moves from the grid."""
        g = grid.copy()
        total = 0
        for _ in range(self.rollout_depth):
            moves = self.find_moves(g)
            if not moves:
                break
            _, (r1, r2, c1, c2) = random.choice(moves)
            removed = np.count_nonzero(g[r1:r2, c1:c2])
            g[r1:r2, c1:c2] = 0
            total += removed
        return total

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Propagate reward back to all ancestors."""
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    # -----------------------------
    # Fast move finder
    # -----------------------------
    def find_moves(self, grid: np.ndarray) -> List[Tuple[int, Tuple[int,int,int,int]]]:
        moves_set = set()
        rows, cols = grid.shape

        # Rectangles
        for r1 in range(rows):
            col_sums = [0] * cols
            for r2 in range(r1, rows):
                for c in range(cols):
                    col_sums[c] += grid[r2, c]
                c1 = 0
                while c1 < cols:
                    s = 0
                    non_zero = []
                    for c2 in range(c1, cols):
                        val = col_sums[c2]
                        s += val
                        if val != 0:
                            non_zero.append(c2)
                        if s > 10:
                            break
                        if s == 10 and non_zero:
                            rr1, rr2 = r1, r2 + 1
                            cc1, cc2 = non_zero[0], non_zero[-1] + 1
                            moves_set.add((rr1, rr2, cc1, cc2))
                    c1 += 1

        # Straight-line pairs
        for r in range(rows):
            for c in range(cols):
                v1 = grid[r, c]
                if v1 == 0:
                    continue
                # horizontal
                for nc in range(c + 1, cols):
                    v2 = grid[r, nc]
                    if v2 == 0:
                        continue
                    if v1 + v2 == 10:
                        moves_set.add((r, r + 1, c, nc + 1))
                    break
                # vertical
                for nr in range(r + 1, rows):
                    v2 = grid[nr, c]
                    if v2 == 0:
                        continue
                    if v1 + v2 == 10:
                        moves_set.add((r, nr + 1, c, c + 1))
                    break

        # Score = max digit
        moves = []
        for rr1, rr2, cc1, cc2 in moves_set:
            score = np.max(grid[rr1:rr2, cc1:cc2])
            moves.append((score, (rr1, rr2, cc1, cc2)))

        # Beam width
        moves.sort(key=lambda x: x[0], reverse=True)
        return moves[:self.beam_width]