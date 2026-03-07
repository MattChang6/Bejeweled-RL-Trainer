import sys
import time
from typing import Optional, Tuple, List
import pyautogui
from PyQt6.QtWidgets import (QApplication, QMainWindow, QMessageBox, QFileDialog,
                             QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox,
                             QPushButton, QWidget, QLabel, QSlider, QGroupBox,
                             QRadioButton, QCheckBox, QScrollArea, QSizePolicy)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2 as cv
import numpy as np
import win32gui
import win32con

from windowCapture import WindowCapture
from vision import Vision
from hsvfilter import hsvFilter
from openCV_GUI import Ui_MainWindow

class CaptureThread(QThread):
    update_signal = pyqtSignal(np.ndarray)
    action_signal = pyqtSignal(list)

    def __init__(self, window_name: str, hsv_filter_type: str,
                 target_image_path: Optional[str] = None,
                 draw_rectangles: bool = False,
                 threshold: float = 0.75,
                 crop_values: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        super().__init__()
        self.window_name = window_name
        self.hsv_filter_type = hsv_filter_type
        self.target_image_path = target_image_path
        self.targets = []
        self.draw_rectangles = draw_rectangles
        self.threshold = threshold
        self.crop_values = crop_values
        self.running = True
        self.wincap: Optional[WindowCapture] = None
        self.current_rectangles = []
        self.show_fps = False
        self.last_time = time.time()
        self.fps = 0

        self.hsv_filter_color = hsvFilter(0, 0, 0, 179, 255, 255, 0, 0, 0, 0)
        self.hsv_filter_grayscale = hsvFilter(0, 0, 0, 179, 255, 255, 0, 255, 0, 0)

        self.target_vision = Vision(self.target_image_path) if self.target_image_path else None

    def run(self):
        try:
            self.wincap = WindowCapture(self.window_name)
            if not self.wincap or not self.wincap.hwnd:
                print(f"Failed to initialize window capture for: {self.window_name}")
                return

            vision = Vision(None)
            retry_count = 0
            max_retries = 3

            while self.running:
                screenshot = self.wincap.get_screenshot()

                if screenshot is None:
                    retry_count += 1
                    if retry_count > max_retries:
                        print("Failed to capture screenshot after multiple attempts")
                        break
                    self.msleep(100)
                    continue

                retry_count = 0

                # Calculate FPS
                current_time = time.time()
                if self.show_fps:
                    self.fps = 1 / (current_time - self.last_time)
                    self.last_time = current_time

                # Apply cropping
                if any(self.crop_values):
                    h, w = screenshot.shape[:2]
                    left = int(w * self.crop_values[0] / 100)
                    right = int(w * (1 - self.crop_values[1] / 100))
                    top = int(h * self.crop_values[2] / 100)
                    bottom = int(h * (1 - self.crop_values[3] / 100))
                    if right > left and bottom > top:
                        screenshot = screenshot[top:bottom, left:right]

                # Convert BGRA to BGR if needed
                if screenshot.shape[2] == 4:
                    screenshot = cv.cvtColor(screenshot, cv.COLOR_BGRA2BGR)

                # Process image with HSV filter
                hsv_image = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)

                if self.hsv_filter_type == 'color':
                    processed_image = vision.apply_hsv_filter(hsv_image, self.hsv_filter_color)
                    processed_image = cv.cvtColor(processed_image, cv.COLOR_HSV2BGR)
                else:
                    processed_image = vision.apply_hsv_filter(hsv_image, self.hsv_filter_grayscale)
                    processed_image = cv.cvtColor(processed_image, cv.COLOR_BGR2GRAY)
                    processed_image = cv.cvtColor(processed_image, cv.COLOR_GRAY2BGR)

                # Search for targets
                if self.targets:
                    for vision_obj, threshold in self.targets:
                        search_img = processed_image.copy()
                        rectangles = vision_obj.find(search_img, threshold)

                        if rectangles.any():
                            self.current_rectangles = rectangles
                            self.action_signal.emit(rectangles.tolist())

                        if self.draw_rectangles and len(rectangles) > 0:
                            processed_image = vision_obj.draw_rectangles(processed_image, rectangles)

                # Add FPS text to image if enabled
                if self.show_fps:
                    cv.putText(processed_image, f"FPS: {self.fps:.0f}", (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.update_signal.emit(processed_image)
                self.msleep(30)  # ~33 FPS

        except Exception as e:
            print(f"Thread error: {e}")

    def stop(self):
        self.running = False
        self.wait(500)

    def add_target(self, vision_obj, threshold):
        """Add a target for detection"""
        self.targets.append((vision_obj, threshold))

    def clear_targets(self):
        """Clear all targets"""
        self.targets.clear()

    def get_window_position(self):
        """Safely get the window position"""
        if not self.wincap or not hasattr(self.wincap, 'hwnd') or not self.wincap.hwnd:
            return None

        try:
            # Get window rect using win32gui
            rect = win32gui.GetWindowRect(self.wincap.hwnd)
            if rect:
                return rect[0], rect[1]  # left, top
        except Exception as e:
            print(f"Error getting window position: {e}")

        return None