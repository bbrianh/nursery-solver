"""Grid calibration and digit matrix extraction from a screenshot."""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from digits import recognize_digit
from models import COLS, ROWS

# Keys persisted in resource/sqinfo.json (interactive calibration output).
_REQUIRED_SQINFO_KEYS = (
    "anchor_x",
    "anchor_y",
    "hwidth",
    "vwidth",
    "hgap",
    "vgap",
    "settings_x",
    "settings_y",
)


class Recognizer:
    """Loads or interactively builds ``sqinfo``, then reads a 16×10 digit grid from an image."""

    def __init__(
        self, sqinfo_path: str = "resource/sqinfo.json", *, force_calibration: bool = False
    ) -> None:
        self.sqinfo: Optional[dict[str, Any]] = None
        self.sqinfo_path = sqinfo_path
        self.image: np.ndarray
        self.crop_images: List[np.ndarray] = []
        self.digits_matrix: List[List[int]] = []
        if force_calibration:
            print(
                f"Calibration mode: ignoring existing {sqinfo_path}; "
                "confirm in the UI to overwrite it."
            )
        else:
            self._try_load_sqinfo()

    def _try_load_sqinfo(self) -> None:
        try:
            with open(self.sqinfo_path, encoding="utf-8") as f:
                self.sqinfo = json.load(f)
            print(f"Loaded calibration from {self.sqinfo_path}.")
        except FileNotFoundError:
            print(f"{self.sqinfo_path} not found; interactive calibration will run on first use.")
            self.sqinfo = None
        except json.JSONDecodeError:
            print(f"Invalid JSON in {self.sqinfo_path}; interactive calibration will run.")
            self.sqinfo = None
        except OSError as e:
            print(f"Could not read {self.sqinfo_path}: {e}; interactive calibration will run.")
            self.sqinfo = None

    def _save_sqinfo(self) -> None:
        if not self.sqinfo:
            return
        try:
            with open(self.sqinfo_path, "w", encoding="utf-8") as f:
                json.dump(self.sqinfo, f, indent=4)
            print(f"Saved calibration to {self.sqinfo_path}.")
        except OSError as e:
            print(f"Could not save calibration to {self.sqinfo_path}: {e}.")

    def _ensure_sqinfo_derived_fields(self) -> None:
        assert self.sqinfo is not None
        if "h" not in self.sqinfo or "v" not in self.sqinfo:
            self.sqinfo["h"] = float(self.sqinfo["hwidth"]) + float(self.sqinfo["hgap"])
            self.sqinfo["v"] = float(self.sqinfo["vwidth"]) + float(self.sqinfo["vgap"])

    def _validate_loaded_sqinfo(self) -> bool:
        if self.sqinfo is None:
            return False
        if not all(k in self.sqinfo for k in _REQUIRED_SQINFO_KEYS):
            print(
                f"Calibration in {self.sqinfo_path} is incomplete; "
                "will re-run interactive calibration."
            )
            self.sqinfo = None
            return False
        self._ensure_sqinfo_derived_fields()
        return True

    def get_sqinfo(self, image: np.ndarray) -> dict[str, Any]:
        """Return grid geometry; run interactive calibration if missing or invalid."""
        if self.sqinfo is not None and self._validate_loaded_sqinfo():
            print("Using saved calibration.")
            assert self.sqinfo is not None
            return self.sqinfo

        print("\nCalibration: drag lines to outer/inner grid edges, drag the blue marker to Settings.")
        calibrator = _InteractiveGridCalibration(image, rows=ROWS, cols=COLS)
        plt.show()

        if calibrator.results is None:
            raise ValueError("Calibration was cancelled or closed without confirming.")

        self.sqinfo = calibrator.results
        self._save_sqinfo()
        print(f"Calibration saved: {self.sqinfo}")
        return self.sqinfo

    def _crop_region(self, square: Tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = square
        return self.image[int(y1) : int(y2), int(x1) : int(x2)]

    def get_matrix(
        self, image: np.ndarray
    ) -> Tuple[Optional[List[List[int]]], Optional[List[Tuple[float, float, float, float]]]]:
        """
        Parse ``image`` (BGR) into a ``ROWS`` × ``COLS`` digit matrix and per-cell pixel boxes.

        Returns ``(None, None)`` if calibration fails; ``(None, squares)`` if cell layout is wrong.
        """
        self.image = image
        try:
            sqinfo = self.get_sqinfo(image)
        except ValueError as e:
            print(f"get_sqinfo failed in get_matrix: {e}")
            return None, None

        squares: List[Tuple[float, float, float, float]] = []
        for i in range(ROWS):
            for j in range(COLS):
                squares.append(
                    (
                        sqinfo["anchor_x"] + j * sqinfo["h"],
                        sqinfo["anchor_y"] + i * sqinfo["v"],
                        sqinfo["anchor_x"] + sqinfo["hwidth"] + j * sqinfo["h"],
                        sqinfo["anchor_y"] + sqinfo["vwidth"] + i * sqinfo["v"],
                    )
                )

        expected = ROWS * COLS
        if len(squares) != expected:
            print(squares)
            print(f"Expected {expected} cell boxes, got {len(squares)}.")
            return None, squares

        self.crop_images = [self._crop_region(sq) for sq in squares]
        recognized_digits = [recognize_digit(crop) for crop in self.crop_images]
        self.digits_matrix = [
            recognized_digits[r * COLS : (r + 1) * COLS] for r in range(ROWS)
        ]
        return self.digits_matrix, squares


class _InteractiveGridCalibration:
    """Matplotlib UI: align boundary lines and settings marker, output ``sqinfo`` dict."""

    def __init__(self, img: np.ndarray, rows: int, cols: int) -> None:
        self.image = img
        self.rows = rows
        self.cols = cols
        self.fig, self.ax = plt.subplots(figsize=(8, 12))
        self.fig.canvas.manager.set_window_title("Grid calibration")

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        self.ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.ax.set_title(
            "1. Drag solid lines to the outer/inner edges of the grid\n"
            "2. Drag the blue marker to the center of the Settings button\n"
            "(Red: outer bounds; yellow: inner; purple: cell preview)",
            pad=20,
        )

        img_h, img_w = img.shape[:2]

        self.x1_pos, self.x2_pos = img_w * 0.05, img_w * 0.14
        self.x3_pos, self.x4_pos = img_w * 0.86, img_w * 0.95
        self.y1_pos, self.y2_pos = img_h * 0.20, img_h * 0.24
        self.y3_pos, self.y4_pos = img_h * 0.80, img_h * 0.84

        self.settings_marker, = self.ax.plot(
            [img_w * 0.08],
            [img_h * 0.08],
            marker="*",
            color="blue",
            markersize=20,
            linestyle="None",
            picker=5,
        )

        self.lines = {
            "x1": self.ax.axvline(self.x1_pos, color="r", linewidth=2.5, picker=5),
            "x2": self.ax.axvline(self.x2_pos, color="y", linewidth=2.5, picker=5),
            "x3": self.ax.axvline(self.x3_pos, color="y", linewidth=2.5, picker=5),
            "x4": self.ax.axvline(self.x4_pos, color="r", linewidth=2.5, picker=5),
            "y1": self.ax.axhline(self.y1_pos, color="r", linewidth=2.5, picker=5),
            "y2": self.ax.axhline(self.y2_pos, color="y", linewidth=2.5, picker=5),
            "y3": self.ax.axhline(self.y3_pos, color="y", linewidth=2.5, picker=5),
            "y4": self.ax.axhline(self.y4_pos, color="r", linewidth=2.5, picker=5),
        }

        n_v = 2 * cols
        n_h = 2 * rows
        self.v_grid_lines = [
            self.ax.axvline(0, color="purple", linestyle="--", linewidth=1, alpha=0.7) for _ in range(n_v)
        ]
        self.h_grid_lines = [
            self.ax.axhline(0, color="purple", linestyle="--", linewidth=1, alpha=0.7) for _ in range(n_h)
        ]

        self.active_line: Optional[str] = None
        self.results: Optional[dict[str, float]] = None

        self.update_grid()

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        ax_btn = plt.axes([0.8, 0.01, 0.15, 0.05])
        self.btn = Button(ax_btn, "Confirm")
        self.btn.on_clicked(self.on_confirm)

    def update_grid(self) -> None:
        x1, x2, x3, x4 = sorted([self.lines[k].get_xdata()[0] for k in ["x1", "x2", "x3", "x4"]])
        y1, y2, y3, y4 = sorted([self.lines[k].get_ydata()[0] for k in ["y1", "y2", "y3", "y4"]])

        hwidth = ((x2 - x1) + (x4 - x3)) / 2.0
        vwidth = ((y2 - y1) + (y4 - y3)) / 2.0

        hgap = (x4 - x1 - self.cols * hwidth) / (self.cols - 1) if x4 > x1 and self.cols > 1 else 0.0
        vgap = (y4 - y1 - self.rows * vwidth) / (self.rows - 1) if y4 > y1 and self.rows > 1 else 0.0

        for j in range(self.cols):
            left = x1 + j * (hwidth + hgap)
            right = left + hwidth
            self.v_grid_lines[j * 2].set_xdata([left, left])
            self.v_grid_lines[j * 2 + 1].set_xdata([right, right])

        for i in range(self.rows):
            top = y1 + i * (vwidth + vgap)
            bottom = top + vwidth
            self.h_grid_lines[i * 2].set_ydata([top, top])
            self.h_grid_lines[i * 2 + 1].set_ydata([bottom, bottom])

    def on_press(self, event: Any) -> None:
        if event.inaxes != self.ax:
            return

        contains, _ = self.settings_marker.contains(event)
        if contains:
            self.active_line = "settings"
            return

        min_dist = float("inf")
        for name, line in self.lines.items():
            line_contains, _ = line.contains(event)
            if line_contains:
                if name.startswith("x"):
                    dist = abs(line.get_xdata()[0] - event.xdata)
                else:
                    dist = abs(line.get_ydata()[0] - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    self.active_line = name

    def on_motion(self, event: Any) -> None:
        if self.active_line is None or event.inaxes != self.ax:
            return

        if self.active_line == "settings":
            self.settings_marker.set_data([event.xdata], [event.ydata])
            self.fig.canvas.draw_idle()
            return

        line = self.lines[self.active_line]
        if self.active_line.startswith("x"):
            line.set_xdata([event.xdata, event.xdata])
        else:
            line.set_ydata([event.ydata, event.ydata])

        self.update_grid()
        self.fig.canvas.draw_idle()

    def on_release(self, event: Any) -> None:
        self.active_line = None

    def on_confirm(self, event: Any) -> None:
        x1, x2, x3, x4 = sorted([self.lines[k].get_xdata()[0] for k in ["x1", "x2", "x3", "x4"]])
        y1, y2, y3, y4 = sorted([self.lines[k].get_ydata()[0] for k in ["y1", "y2", "y3", "y4"]])

        hwidth = ((x2 - x1) + (x4 - x3)) / 2.0
        vwidth = ((y2 - y1) + (y4 - y3)) / 2.0
        hgap = (x4 - x1 - self.cols * hwidth) / (self.cols - 1) if self.cols > 1 else 0.0
        vgap = (y4 - y1 - self.rows * vwidth) / (self.rows - 1) if self.rows > 1 else 0.0

        self.results = {
            "anchor_x": float(x1),
            "anchor_y": float(y1),
            "hwidth": float(hwidth),
            "vwidth": float(vwidth),
            "hgap": float(hgap),
            "vgap": float(vgap),
            "h": float(hwidth + hgap),
            "v": float(vwidth + vgap),
            "settings_x": float(self.settings_marker.get_xdata()[0]),
            "settings_y": float(self.settings_marker.get_ydata()[0]),
        }
        plt.close(self.fig)
