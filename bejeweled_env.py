import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import cv2 as cv
import numpy as np
import pyautogui

from bejeweled_vision import BoardVision, Calibration


@dataclass
class RewardConfig:
    match_reward: float = 1.0
    invalid_penalty: float = 1.0
    step_penalty: float = 0.01
    speed_target_sec: float = 1.5
    speed_bonus_max: float = 0.5


@dataclass
class TransitionConfig:
    enabled: bool = True
    pause_seconds: float = 10.0
    confidence_threshold: float = 0.58
    motion_threshold: float = 0.11
    consecutive_frames: int = 4


class BejeweledEnv:
    def __init__(
        self,
        window_title: str,
        calibration_path: str = "calibration.json",
        classifier_path: str = "models/gem_classifier.pt",
        classifier_device: str = "cpu",
        swap_delay: float = 0.1,
        settle_delay: float = 0.6,
        reward_cfg: RewardConfig = RewardConfig(),
        transition_cfg: TransitionConfig = TransitionConfig(),
    ):
        self.window_title = window_title
        self.calibration_path = calibration_path
        self.swap_delay = swap_delay
        self.settle_delay = settle_delay
        self.reward_cfg = reward_cfg
        self.transition_cfg = transition_cfg

        calibration = BoardVision.load_calibration(calibration_path)
        if calibration is None:
            calibration = BoardVision.run_calibration(window_title)
            BoardVision.save_calibration(calibration_path, calibration)

        self.vision = BoardVision(
            window_title,
            calibration,
            classifier_path=classifier_path,
            classifier_device=classifier_device,
        )
        self.last_board = None
        self.last_match_time = time.time()
        self.prev_gray = None
        self.transition_streak = 0
        self.transition_until = 0.0

        self.action_count = calibration.grid_size * calibration.grid_size * 4
        self.grid_size = calibration.grid_size
        self.colors = calibration.colors

    def reset(self) -> np.ndarray:
        self.last_board = self.vision.board_state(reinit=True)
        self.last_match_time = time.time()
        self.prev_gray = None
        self.transition_streak = 0
        self.transition_until = 0.0
        return self._obs_from_board(self.last_board)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self._transition_active():
            board = self.vision.board_state()
            self.last_board = board
            obs = self._obs_from_board(board)
            remaining = max(0.0, self.transition_until - time.time())
            return obs, 0.0, False, {"transition": True, "skip_replay": True, "transition_remaining_sec": remaining}

        if self._transition_detected():
            self.transition_streak = 0
            self.transition_until = time.time() + self.transition_cfg.pause_seconds
            board = self.vision.board_state()
            self.last_board = board
            obs = self._obs_from_board(board)
            return obs, 0.0, False, {"transition": True, "skip_replay": True, "transition_triggered": True}

        r1, c1, r2, c2 = self._decode_action(action)
        if not self._valid_swap(r1, c1, r2, c2):
            obs = self._obs_from_board(self.last_board)
            return obs, -self.reward_cfg.invalid_penalty, False, {"invalid": True}

        self._perform_swap(r1, c1, r2, c2)
        time.sleep(self.swap_delay)
        self.vision.wait_for_settle(self.settle_delay)

        new_board = self.vision.board_state()
        if self._transition_detected():
            self.transition_streak = 0
            self.transition_until = time.time() + self.transition_cfg.pause_seconds
            self.last_board = new_board
            obs = self._obs_from_board(new_board)
            return obs, 0.0, False, {"transition": True, "skip_replay": True, "transition_triggered": True}

        reward, info = self._compute_reward(self.last_board, new_board)
        self.last_board = new_board
        obs = self._obs_from_board(new_board)
        return obs, reward, False, info

    def _transition_active(self) -> bool:
        return self.transition_cfg.enabled and time.time() < self.transition_until

    def _transition_detected(self) -> bool:
        if not self.transition_cfg.enabled:
            return False

        board_img = self.vision.capture_board()
        gray = cv.cvtColor(board_img, cv.COLOR_BGR2GRAY)
        motion = 0.0
        if self.prev_gray is not None:
            diff = cv.absdiff(gray, self.prev_gray)
            motion = float(np.mean(diff) / 255.0)
        self.prev_gray = gray

        _ = self.vision.board_state()
        conf_map = self.vision.last_confidence_map
        if conf_map is None:
            low_confidence = True
        else:
            low_confidence = float(np.mean(conf_map)) < self.transition_cfg.confidence_threshold

        high_motion = motion > self.transition_cfg.motion_threshold
        if low_confidence and high_motion:
            self.transition_streak += 1
        else:
            self.transition_streak = 0

        return self.transition_streak >= self.transition_cfg.consecutive_frames

    def _perform_swap(self, r1: int, c1: int, r2: int, c2: int) -> None:
        x1, y1 = self.vision.cell_center_screen(r1, c1)
        x2, y2 = self.vision.cell_center_screen(r2, c2)
        pyautogui.click(x=x1, y=y1)
        time.sleep(0.03)
        pyautogui.click(x=x2, y=y2)

    def _compute_reward(self, prev: np.ndarray, new: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        matches = self._find_matches(new)
        match_count = len(matches)
        reward = -self.reward_cfg.step_penalty
        info: Dict[str, Any] = {"match_count": match_count}

        if match_count == 0 or np.array_equal(prev, new):
            reward -= self.reward_cfg.invalid_penalty
            info["invalid"] = True
            return reward, info

        reward += self.reward_cfg.match_reward * match_count

        now = time.time()
        dt = now - self.last_match_time
        self.last_match_time = now
        if dt <= self.reward_cfg.speed_target_sec:
            bonus = self.reward_cfg.speed_bonus_max * (1.0 - dt / self.reward_cfg.speed_target_sec)
            reward += max(0.0, bonus)
            info["speed_bonus"] = bonus

        return reward, info

    def _find_matches(self, board: np.ndarray) -> set:
        matches = set()
        n = self.grid_size

        for r in range(n):
            run = 1
            for c in range(1, n):
                if board[r, c] == board[r, c - 1]:
                    run += 1
                else:
                    if run >= 3:
                        for k in range(c - run, c):
                            matches.add((r, k))
                    run = 1
            if run >= 3:
                for k in range(n - run, n):
                    matches.add((r, k))

        for c in range(n):
            run = 1
            for r in range(1, n):
                if board[r, c] == board[r - 1, c]:
                    run += 1
                else:
                    if run >= 3:
                        for k in range(r - run, r):
                            matches.add((k, c))
                    run = 1
            if run >= 3:
                for k in range(n - run, n):
                    matches.add((k, c))

        return matches

    def _obs_from_board(self, board: np.ndarray) -> np.ndarray:
        one_hot = np.eye(self.colors, dtype=np.float32)[board]
        return np.transpose(one_hot, (2, 0, 1))

    def debug_frame(self) -> np.ndarray:
        board_img = self.vision.capture_board()
        labels = self.vision.board_state()
        return self.vision.annotate_board(board_img, labels)

    def _decode_action(self, action: int) -> Tuple[int, int, int, int]:
        n = self.grid_size
        cell = action // 4
        direction = action % 4
        r = cell // n
        c = cell % n

        if direction == 0:  # up
            return r, c, r - 1, c
        if direction == 1:  # down
            return r, c, r + 1, c
        if direction == 2:  # left
            return r, c, r, c - 1
        return r, c, r, c + 1

    def _valid_swap(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        n = self.grid_size
        if r2 < 0 or r2 >= n or c2 < 0 or c2 >= n:
            return False
        return abs(r1 - r2) + abs(c1 - c2) == 1
