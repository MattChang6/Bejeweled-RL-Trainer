import json
import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional, List

import cv2 as cv
import numpy as np
import pyautogui

from windowCapture import WindowCapture


@dataclass
class Calibration:
    board_left: int
    board_top: int
    board_right: int
    board_bottom: int
    grid_size: int = 8
    colors: int = 7

    @property
    def board_width(self) -> int:
        return self.board_right - self.board_left

    @property
    def board_height(self) -> int:
        return self.board_bottom - self.board_top

    @property
    def cell_w(self) -> int:
        return int(self.board_width / self.grid_size)

    @property
    def cell_h(self) -> int:
        return int(self.board_height / self.grid_size)


@dataclass
class ScoreCalibration:
    score_left: int
    score_top: int
    score_right: int
    score_bottom: int

    @property
    def width(self) -> int:
        return self.score_right - self.score_left

    @property
    def height(self) -> int:
        return self.score_bottom - self.score_top


class BoardVision:
    def __init__(
        self,
        window_title: str,
        calibration: Calibration,
        classifier_path: Optional[str] = None,
        classifier_device: str = "cpu",
        confidence_threshold: float = 0.55,
        smoothing_alpha: float = 0.65,
    ):
        self.window_title = window_title
        self.calibration = calibration
        self.wincap = WindowCapture(window_title)
        self.centroids: Optional[np.ndarray] = None
        self.classifier_bundle = None
        self.classifier_path = classifier_path
        self.classifier_device = classifier_device
        self.confidence_threshold = confidence_threshold
        self.smoothing_alpha = smoothing_alpha
        self.prev_labels: Optional[np.ndarray] = None
        self.prev_confidence: Optional[np.ndarray] = None
        self.last_confidence_map: Optional[np.ndarray] = None

        if self.classifier_path and os.path.exists(self.classifier_path):
            self._load_classifier(self.classifier_path, self.classifier_device)

    @staticmethod
    def load_calibration(path: str) -> Optional[Calibration]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Calibration(**data)
        except FileNotFoundError:
            return None

    @staticmethod
    def save_calibration(path: str, calibration: Calibration) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(calibration.__dict__, f, indent=2)

    @staticmethod
    def load_score_calibration(path: str) -> Optional[ScoreCalibration]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ScoreCalibration(**data)
        except FileNotFoundError:
            return None

    @staticmethod
    def save_score_calibration(path: str, calibration: ScoreCalibration) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(calibration.__dict__, f, indent=2)

    @staticmethod
    def run_calibration(window_title: str) -> Calibration:
        wincap = WindowCapture(window_title)
        print("Calibration: move mouse to TOP-LEFT of the board, then press Enter.")
        input()
        tl = pyautogui.position()
        print("Calibration: move mouse to BOTTOM-RIGHT of the board, then press Enter.")
        input()
        br = pyautogui.position()

        left = int(tl.x - wincap.offset_x)
        top = int(tl.y - wincap.offset_y)
        right = int(br.x - wincap.offset_x)
        bottom = int(br.y - wincap.offset_y)

        if right <= left or bottom <= top:
            raise ValueError("Invalid calibration rectangle. Try again.")

        return Calibration(
            board_left=left,
            board_top=top,
            board_right=right,
            board_bottom=bottom,
        )

    @staticmethod
    def run_score_calibration(window_title: str) -> ScoreCalibration:
        wincap = WindowCapture(window_title)
        print("Score calibration: move mouse to TOP-LEFT of the score area, then press Enter.")
        input()
        tl = pyautogui.position()
        print("Score calibration: move mouse to BOTTOM-RIGHT of the score area, then press Enter.")
        input()
        br = pyautogui.position()

        left = int(tl.x - wincap.offset_x)
        top = int(tl.y - wincap.offset_y)
        right = int(br.x - wincap.offset_x)
        bottom = int(br.y - wincap.offset_y)

        if right <= left or bottom <= top:
            raise ValueError("Invalid score calibration rectangle. Try again.")

        return ScoreCalibration(
            score_left=left,
            score_top=top,
            score_right=right,
            score_bottom=bottom,
        )

    def capture_board(self) -> np.ndarray:
        screenshot = self.wincap.get_screenshot()
        c = self.calibration
        board_img = screenshot[c.board_top:c.board_bottom, c.board_left:c.board_right]
        return board_img

    def capture_score(self, score_calibration: ScoreCalibration) -> np.ndarray:
        screenshot = self.wincap.get_screenshot()
        s = score_calibration
        return screenshot[s.score_top:s.score_bottom, s.score_left:s.score_right]

    def cell_images(self, board_img: np.ndarray) -> List[np.ndarray]:
        c = self.calibration
        cells: List[np.ndarray] = []
        for r in range(c.grid_size):
            for col in range(c.grid_size):
                x0 = col * c.cell_w
                y0 = r * c.cell_h
                x1 = x0 + c.cell_w
                y1 = y0 + c.cell_h
                cells.append(board_img[y0:y1, x0:x1])
        return cells

    def annotate_board(self, board_img: np.ndarray, labels: np.ndarray) -> np.ndarray:
        c = self.calibration
        annotated = board_img.copy()
        for r in range(c.grid_size):
            for col in range(c.grid_size):
                x0 = col * c.cell_w
                y0 = r * c.cell_h
                x1 = x0 + c.cell_w
                y1 = y0 + c.cell_h
                cv.rectangle(annotated, (x0, y0), (x1, y1), (50, 50, 50), 1)
                label = int(labels[r, col])
                cv.putText(
                    annotated,
                    str(label),
                    (x0 + 4, y0 + 16),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )
        return annotated

    def _cell_hsv_means(self, board_img: np.ndarray) -> np.ndarray:
        c = self.calibration
        hsv = cv.cvtColor(board_img, cv.COLOR_BGR2HSV)
        cells = []
        for r in range(c.grid_size):
            for col in range(c.grid_size):
                x0 = col * c.cell_w
                y0 = r * c.cell_h
                x1 = x0 + c.cell_w
                y1 = y0 + c.cell_h
                cell = hsv[y0:y1, x0:x1]
                mean = cell.reshape(-1, 3).mean(axis=0)
                cells.append(mean)
        return np.array(cells, dtype=np.float32)

    def _init_centroids(self, cell_hsv: np.ndarray) -> None:
        k = self.calibration.colors
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        attempts = 5
        _, _, centers = cv.kmeans(
            cell_hsv,
            k,
            None,
            criteria,
            attempts,
            cv.KMEANS_PP_CENTERS,
        )
        self.centroids = centers

    def _load_classifier(self, classifier_path: str, device: str) -> None:
        from gem_classifier import load_gem_classifier

        self.classifier_bundle = load_gem_classifier(classifier_path, device=device)

    def _board_state_classifier(self, board_img: np.ndarray) -> np.ndarray:
        from gem_classifier import infer_cells

        c = self.calibration
        cells = self.cell_images(board_img)
        labels, confidence = infer_cells(self.classifier_bundle, cells)
        labels = labels.reshape(c.grid_size, c.grid_size)
        confidence = confidence.reshape(c.grid_size, c.grid_size)

        if self.prev_labels is None or self.prev_confidence is None:
            self.prev_labels = labels.copy()
            self.prev_confidence = confidence.copy()
        else:
            smoothed = self.smoothing_alpha * confidence + (1.0 - self.smoothing_alpha) * self.prev_confidence
            keep_prev = smoothed < self.confidence_threshold
            labels = np.where(keep_prev, self.prev_labels, labels)
            confidence = smoothed
            self.prev_labels = labels.copy()
            self.prev_confidence = confidence.copy()

        self.last_confidence_map = confidence
        return labels

    def board_state(self, reinit: bool = False) -> np.ndarray:
        board_img = self.capture_board()
        if self.classifier_bundle is not None:
            if reinit:
                self.prev_labels = None
                self.prev_confidence = None
            return self._board_state_classifier(board_img)

        cell_hsv = self._cell_hsv_means(board_img)
        if self.centroids is None or reinit:
            self._init_centroids(cell_hsv)
        dists = np.linalg.norm(cell_hsv[:, None, :] - self.centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        return labels.reshape(self.calibration.grid_size, self.calibration.grid_size)

    def cell_center_screen(self, row: int, col: int) -> Tuple[int, int]:
        c = self.calibration
        x = c.board_left + int((col + 0.5) * c.cell_w)
        y = c.board_top + int((row + 0.5) * c.cell_h)
        return self.wincap.get_screen_position((x, y))

    def wait_for_settle(self, delay: float) -> None:
        time.sleep(delay)
