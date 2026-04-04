"""Orchestrate window capture → recognize → plan → execute (reference-style, no manual region)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from capture import INTERNAL_HEIGHT, INTERNAL_WIDTH, capture_game_window
from executor import MoveExecutor
from models import GridGeometry
from recognizer import Recognizer
from solvers import __all__ as solver_names
from solvers import SOLVERS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Nursery bot: find the game window, read the grid, plan moves, execute drags. "
            "Uses resource/sqinfo.json (use --calibrate once to create/update it). "
            f"Default capture letterboxes into {INTERNAL_WIDTH}×{INTERNAL_HEIGHT} (no squash). "
            "Use --stretch for reference-style stretch fill. Recalibrate after switching modes."
        )
    )
    p.add_argument(
        "--window-title",
        default="开局托儿所",
        help='WeChat window title substring (default: "%(default)s").',
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan moves only; do not move the mouse.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Write tmp/capture.png and tmp/matrix.txt for inspection.",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Open the grid calibration UI and write resource/sqinfo.json. "
            "If omitted, existing sqinfo is used (no interactive calibration)."
        ),
    )
    p.add_argument(
        "--solver",
        required=True,
        help="Solver to use (required).",
        choices=solver_names
    )
    return p.parse_args()


def _debug_dir() -> Path:
    d = Path(__file__).resolve().parent / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_debug_capture(image_bgr: np.ndarray) -> None:
    path = _debug_dir() / "capture.png"
    cv2.imwrite(str(path), image_bgr)
    print(f"Debug: wrote {path}")


def _save_debug_matrix(grid: np.ndarray) -> None:
    path = _debug_dir() / "matrix.txt"
    np.savetxt(path, grid, fmt="%d")
    print(f"Debug: wrote {path}")


def main() -> None:
    args = parse_args()

    print("Locating game window and capturing…")
    cap = capture_game_window(
        window_title=args.window_title
    )
    if args.debug:
        _save_debug_capture(cap.image_bgr)

    recognizer = Recognizer(force_calibration=args.calibrate)
    matrix, _squares = recognizer.get_matrix(cap.image_bgr)
    if matrix is None:
        print("Failed to read grid (missing/invalid sqinfo or recognition).")
        return

    grid = np.array(matrix, dtype=np.int16)
    print("Grid shape:", grid.shape)
    if args.debug:
        _save_debug_matrix(grid)

    print("Solving grid...")
    solver = SOLVERS[args.solver](grid, args.debug)
    solver.plan()
    moves = solver.get_solution()
    print(f"Planned {len(moves)} move(s).")
    print(f"Score: {solver.score()}")

    if args.dry_run:
        print("Dry run: skipping executor.")
        return

    if recognizer.sqinfo is None:
        print("Missing sqinfo after recognition; cannot execute.")
        return

    print("Executing moves...")
    geometry = GridGeometry.from_sqinfo(recognizer.sqinfo)
    executor = MoveExecutor(
        geometry,
        cap.anchor,
        scale_xy=(
            cap.owidth / float(cap.internal_width),
            cap.oheight / float(cap.internal_height),
        ),
    )
    executor.execute(moves)

    print("All moves executed!")


if __name__ == "__main__":
    main()
