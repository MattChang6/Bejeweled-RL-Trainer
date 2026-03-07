import os
from dataclasses import replace
from typing import Optional

import win32gui
from train import TrainingConfig, TrainingControl, load_checkpoint, play_session, train_session
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from bejeweled_env import RewardConfig
from bejeweled_vision import BoardVision
from dqn import DQNConfig


class RewardChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.rewards = []

    def set_rewards(self, rewards):
        self.rewards = rewards
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.white)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = 30
        w = max(1, self.width() - 2 * margin)
        h = max(1, self.height() - 2 * margin)
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        painter.drawRect(margin, margin, w, h)

        if len(self.rewards) < 2:
            return

        rmin = min(self.rewards)
        rmax = max(self.rewards)
        if rmax == rmin:
            rmax += 1.0

        points = []
        for i, reward in enumerate(self.rewards):
            x = margin + int(i * w / (len(self.rewards) - 1))
            y = margin + int((rmax - reward) * h / (rmax - rmin))
            points.append((x, y))

        painter.setPen(QPen(Qt.GlobalColor.blue, 2))
        for i in range(1, len(points)):
            painter.drawLine(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1])

        painter.setPen(QPen(Qt.GlobalColor.darkGray, 1))
        painter.drawText(6, margin + 6, f"{rmax:.2f}")
        painter.drawText(6, margin + h, f"{rmin:.2f}")


class OptionsDialog(QDialog):
    def __init__(self, cfg: TrainingConfig, dqn_cfg: DQNConfig, reward_cfg: RewardConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Options")
        self.cfg = cfg
        self.dqn_cfg = dqn_cfg
        self.reward_cfg = reward_cfg

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.episodes = QSpinBox()
        self.episodes.setRange(1, 100000)
        self.episodes.setValue(cfg.episodes)
        form.addRow("Episodes", self.episodes)

        self.max_steps = QSpinBox()
        self.max_steps.setRange(1, 10000)
        self.max_steps.setValue(cfg.max_steps)
        form.addRow("Max Steps", self.max_steps)

        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(1e-6, 1.0)
        self.lr.setSingleStep(1e-4)
        self.lr.setValue(dqn_cfg.lr)
        form.addRow("Learning Rate", self.lr)

        self.gamma = QDoubleSpinBox()
        self.gamma.setDecimals(4)
        self.gamma.setRange(0.0, 0.9999)
        self.gamma.setSingleStep(0.01)
        self.gamma.setValue(dqn_cfg.gamma)
        form.addRow("Gamma", self.gamma)

        self.batch = QSpinBox()
        self.batch.setRange(1, 4096)
        self.batch.setValue(dqn_cfg.batch_size)
        form.addRow("Batch Size", self.batch)

        self.target_update = QSpinBox()
        self.target_update.setRange(1, 1000000)
        self.target_update.setValue(dqn_cfg.target_update)
        form.addRow("Target Update Steps", self.target_update)

        self.replay_size = QSpinBox()
        self.replay_size.setRange(100, 2000000)
        self.replay_size.setValue(dqn_cfg.replay_size)
        form.addRow("Replay Size", self.replay_size)

        self.min_replay = QSpinBox()
        self.min_replay.setRange(1, 2000000)
        self.min_replay.setValue(dqn_cfg.min_replay)
        form.addRow("Min Replay", self.min_replay)

        self.match_reward = QDoubleSpinBox()
        self.match_reward.setRange(0.0, 100.0)
        self.match_reward.setValue(reward_cfg.match_reward)
        form.addRow("Match Reward", self.match_reward)

        self.invalid_penalty = QDoubleSpinBox()
        self.invalid_penalty.setRange(0.0, 100.0)
        self.invalid_penalty.setValue(reward_cfg.invalid_penalty)
        form.addRow("Invalid Penalty", self.invalid_penalty)

        self.step_penalty = QDoubleSpinBox()
        self.step_penalty.setDecimals(4)
        self.step_penalty.setRange(0.0, 10.0)
        self.step_penalty.setValue(reward_cfg.step_penalty)
        form.addRow("Step Penalty", self.step_penalty)

        self.speed_target = QDoubleSpinBox()
        self.speed_target.setDecimals(3)
        self.speed_target.setRange(0.05, 10.0)
        self.speed_target.setValue(reward_cfg.speed_target_sec)
        form.addRow("Speed Target (s)", self.speed_target)

        self.speed_bonus_max = QDoubleSpinBox()
        self.speed_bonus_max.setRange(0.0, 20.0)
        self.speed_bonus_max.setValue(reward_cfg.speed_bonus_max)
        form.addRow("Speed Bonus Max", self.speed_bonus_max)

        self.log_steps = QCheckBox("Log every step")
        self.log_steps.setChecked(cfg.log_steps)
        form.addRow("", self.log_steps)

        self.debug_view = QCheckBox("Show debug board window")
        self.debug_view.setChecked(cfg.debug_view)
        form.addRow("", self.debug_view)

        self.poll_hotkeys = QCheckBox("Enable global hotkeys (P pause/resume, Q stop)")
        self.poll_hotkeys.setChecked(cfg.poll_hotkeys)
        form.addRow("", self.poll_hotkeys)

        self.transition_enabled = QCheckBox("Enable transition detection auto-pause")
        self.transition_enabled.setChecked(cfg.transition_enabled)
        form.addRow("", self.transition_enabled)

        self.transition_pause_seconds = QDoubleSpinBox()
        self.transition_pause_seconds.setRange(0.0, 60.0)
        self.transition_pause_seconds.setDecimals(1)
        self.transition_pause_seconds.setValue(cfg.transition_pause_seconds)
        form.addRow("Transition Pause (s)", self.transition_pause_seconds)

        self.transition_confidence = QDoubleSpinBox()
        self.transition_confidence.setRange(0.0, 1.0)
        self.transition_confidence.setDecimals(3)
        self.transition_confidence.setSingleStep(0.01)
        self.transition_confidence.setValue(cfg.transition_confidence_threshold)
        form.addRow("Transition Conf Thresh", self.transition_confidence)

        self.transition_motion = QDoubleSpinBox()
        self.transition_motion.setRange(0.0, 1.0)
        self.transition_motion.setDecimals(3)
        self.transition_motion.setSingleStep(0.01)
        self.transition_motion.setValue(cfg.transition_motion_threshold)
        form.addRow("Transition Motion Thresh", self.transition_motion)

        self.transition_frames = QSpinBox()
        self.transition_frames.setRange(1, 20)
        self.transition_frames.setValue(cfg.transition_consecutive_frames)
        form.addRow("Transition Frames", self.transition_frames)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self):
        cfg = replace(
            self.cfg,
            episodes=self.episodes.value(),
            max_steps=self.max_steps.value(),
            log_steps=self.log_steps.isChecked(),
            debug_view=self.debug_view.isChecked(),
            poll_hotkeys=self.poll_hotkeys.isChecked(),
            transition_enabled=self.transition_enabled.isChecked(),
            transition_pause_seconds=self.transition_pause_seconds.value(),
            transition_confidence_threshold=self.transition_confidence.value(),
            transition_motion_threshold=self.transition_motion.value(),
            transition_consecutive_frames=self.transition_frames.value(),
        )
        dqn_cfg = replace(
            self.dqn_cfg,
            lr=self.lr.value(),
            gamma=self.gamma.value(),
            batch_size=self.batch.value(),
            target_update=self.target_update.value(),
            replay_size=self.replay_size.value(),
            min_replay=self.min_replay.value(),
        )
        reward_cfg = replace(
            self.reward_cfg,
            match_reward=self.match_reward.value(),
            invalid_penalty=self.invalid_penalty.value(),
            step_penalty=self.step_penalty.value(),
            speed_target_sec=self.speed_target.value(),
            speed_bonus_max=self.speed_bonus_max.value(),
        )
        return cfg, dqn_cfg, reward_cfg


class TrainingWorker(QThread):
    progress = pyqtSignal(dict)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        cfg: TrainingConfig,
        dqn_cfg: DQNConfig,
        reward_cfg: RewardConfig,
        checkpoint_in: Optional[str],
        checkpoint_out: Optional[str],
        run_mode: str,
        model_in: Optional[str],
        control: TrainingControl,
        parent=None,
    ):
        super().__init__(parent)
        self.cfg = cfg
        self.dqn_cfg = dqn_cfg
        self.reward_cfg = reward_cfg
        self.checkpoint_in = checkpoint_in
        self.checkpoint_out = checkpoint_out
        self.run_mode = run_mode
        self.model_in = model_in
        self.control = control

    def run(self):
        try:
            if self.run_mode == "play":
                result = play_session(
                    cfg=self.cfg,
                    model_in=self.model_in or self.cfg.model_out,
                    reward_cfg=self.reward_cfg,
                    control=self.control,
                    progress_cb=lambda m: self.progress.emit(m),
                )
            else:
                result = train_session(
                    cfg=self.cfg,
                    dqn_cfg=self.dqn_cfg,
                    reward_cfg=self.reward_cfg,
                    control=self.control,
                    progress_cb=lambda m: self.progress.emit(m),
                    checkpoint_in=self.checkpoint_in,
                    save_checkpoint_path=self.checkpoint_out,
                )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class WindowCaptureGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bejeweled 3 RL Trainer")
        self.resize(980, 760)

        self.training_cfg = TrainingConfig(window_title="", poll_hotkeys=True)
        self.dqn_cfg = DQNConfig()
        self.reward_cfg = RewardConfig()

        self.control: Optional[TrainingControl] = None
        self.worker: Optional[TrainingWorker] = None
        self.checkpoint_in: Optional[str] = None
        self.reward_history = []
        self.episode_index = 0
        self.current_episode_reward = 0.0

        self._build_ui()
        self.populate_window_list()
        self._set_idle_state()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top_group = QGroupBox("Run")
        top_grid = QGridLayout(top_group)
        layout.addWidget(top_group)

        self.window_box = QComboBox()
        refresh_windows_btn = QPushButton("Refresh")
        refresh_windows_btn.clicked.connect(self.populate_window_list)
        top_grid.addWidget(QLabel("Window"), 0, 0)
        top_grid.addWidget(self.window_box, 0, 1)
        top_grid.addWidget(refresh_windows_btn, 0, 2)

        self.model_out_edit = QLineEdit("models/bejeweled_dqn.pt")
        model_btn = QPushButton("Browse")
        model_btn.clicked.connect(self.select_model_out)
        top_grid.addWidget(QLabel("Model Output"), 1, 0)
        top_grid.addWidget(self.model_out_edit, 1, 1)
        top_grid.addWidget(model_btn, 1, 2)

        self.checkpoint_edit = QLineEdit("models/bejeweled_checkpoint.pt")
        checkpoint_btn = QPushButton("Browse")
        checkpoint_btn.clicked.connect(self.select_checkpoint_out)
        top_grid.addWidget(QLabel("Checkpoint Output"), 2, 0)
        top_grid.addWidget(self.checkpoint_edit, 2, 1)
        top_grid.addWidget(checkpoint_btn, 2, 2)

        self.classifier_edit = QLineEdit("models/gem_classifier.pt")
        classifier_btn = QPushButton("Browse")
        classifier_btn.clicked.connect(self.select_classifier_path)
        top_grid.addWidget(QLabel("Gem Classifier"), 3, 0)
        top_grid.addWidget(self.classifier_edit, 3, 1)
        top_grid.addWidget(classifier_btn, 3, 2)

        self.start_btn = QPushButton("Start Training")
        self.play_btn = QPushButton("Play Trained Model")
        self.stop_btn = QPushButton("Stop")
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.options_btn = QPushButton("Options")
        self.calibrate_btn = QPushButton("Recalibrate Board")
        self.save_training_btn = QPushButton("Save Training Data")
        self.load_training_btn = QPushButton("Load Training Data")

        self.start_btn.clicked.connect(self.start_training)
        self.play_btn.clicked.connect(self.start_play_mode)
        self.stop_btn.clicked.connect(self.stop_training)
        self.pause_btn.clicked.connect(self.pause_training)
        self.resume_btn.clicked.connect(self.resume_training)
        self.options_btn.clicked.connect(self.open_options)
        self.calibrate_btn.clicked.connect(self.recalibrate_board)
        self.save_training_btn.clicked.connect(self.save_training_data)
        self.load_training_btn.clicked.connect(self.load_training_data)

        button_row = QHBoxLayout()
        for btn in [
            self.start_btn,
            self.play_btn,
            self.stop_btn,
            self.pause_btn,
            self.resume_btn,
            self.options_btn,
            self.calibrate_btn,
            self.save_training_btn,
            self.load_training_btn,
        ]:
            button_row.addWidget(btn)
        layout.addLayout(button_row)

        metrics_group = QGroupBox("Metrics")
        metrics_layout = QGridLayout(metrics_group)
        layout.addWidget(metrics_group)

        self.status_label = QLabel("Idle")
        self.episode_label = QLabel("0")
        self.step_label = QLabel("0")
        self.ep_reward_label = QLabel("0.00")
        self.last_reward_label = QLabel("0.00")
        self.total_steps_label = QLabel("0")
        self.epsilon_label = QLabel("0.000")
        self.loss_label = QLabel("0.0000")

        metrics_layout.addWidget(QLabel("Status"), 0, 0)
        metrics_layout.addWidget(self.status_label, 0, 1)
        metrics_layout.addWidget(QLabel("Episode"), 0, 2)
        metrics_layout.addWidget(self.episode_label, 0, 3)
        metrics_layout.addWidget(QLabel("Step"), 0, 4)
        metrics_layout.addWidget(self.step_label, 0, 5)

        metrics_layout.addWidget(QLabel("Episode Reward"), 1, 0)
        metrics_layout.addWidget(self.ep_reward_label, 1, 1)
        metrics_layout.addWidget(QLabel("Last Reward"), 1, 2)
        metrics_layout.addWidget(self.last_reward_label, 1, 3)
        metrics_layout.addWidget(QLabel("Total Steps"), 1, 4)
        metrics_layout.addWidget(self.total_steps_label, 1, 5)

        metrics_layout.addWidget(QLabel("Epsilon"), 2, 0)
        metrics_layout.addWidget(self.epsilon_label, 2, 1)
        metrics_layout.addWidget(QLabel("Loss"), 2, 2)
        metrics_layout.addWidget(self.loss_label, 2, 3)

        self.chart = RewardChartWidget()
        layout.addWidget(self.chart)

    def populate_window_list(self):
        current = self.window_box.currentText()
        self.window_box.clear()

        def win_enum_handler(hwnd, _ctx):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    self.window_box.addItem(title)

        win32gui.EnumWindows(win_enum_handler, None)
        if current:
            idx = self.window_box.findText(current)
            if idx >= 0:
                self.window_box.setCurrentIndex(idx)

    def select_model_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Model output", self.model_out_edit.text(), "PyTorch (*.pt *.pth)")
        if path:
            self.model_out_edit.setText(path)

    def select_checkpoint_out(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Checkpoint output", self.checkpoint_edit.text(), "PyTorch checkpoint (*.pt *.pth)"
        )
        if path:
            self.checkpoint_edit.setText(path)

    def select_classifier_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Gem classifier model", self.classifier_edit.text(), "PyTorch (*.pt *.pth)"
        )
        if path:
            self.classifier_edit.setText(path)

    def _set_idle_state(self):
        self.start_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.status_label.setText("Idle")

    def _set_running_state(self, mode_label: str = "Running"):
        self.start_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.status_label.setText(mode_label)

    def start_training(self):
        self._start_session(run_mode="train")

    def start_play_mode(self):
        self._start_session(run_mode="play")

    def _start_session(self, run_mode: str):
        if self.worker and self.worker.isRunning():
            return

        window_title = self.window_box.currentText().strip()
        if not window_title:
            QMessageBox.warning(self, "Missing Window", "Select a Bejeweled 3 window first.")
            return

        model_path = self.model_out_edit.text().strip() or "models/bejeweled_dqn.pt"
        if run_mode == "play" and not os.path.exists(model_path):
            QMessageBox.warning(self, "Missing Model", "Select an existing trained model file for play mode.")
            return

        self.training_cfg = replace(
            self.training_cfg,
            window_title=window_title,
            model_out=model_path,
            classifier_path=self.classifier_edit.text().strip() or "models/gem_classifier.pt",
        )
        checkpoint_out = self.checkpoint_edit.text().strip() or None
        self.control = TrainingControl()
        self.worker = TrainingWorker(
            cfg=self.training_cfg,
            dqn_cfg=self.dqn_cfg,
            reward_cfg=self.reward_cfg,
            checkpoint_in=self.checkpoint_in,
            checkpoint_out=checkpoint_out,
            run_mode=run_mode,
            model_in=model_path,
            control=self.control,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_training_finished)
        self.worker.failed.connect(self.on_training_failed)
        self.worker.start()
        if run_mode == "play":
            self._set_running_state("Playing")
        else:
            self._set_running_state("Training")

    def stop_training(self):
        if self.control:
            self.control.request_stop()
            self.status_label.setText("Stopping...")

    def pause_training(self):
        if self.control:
            self.control.request_pause()
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)
            self.status_label.setText("Paused")

    def resume_training(self):
        if self.control:
            self.control.request_resume()
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)
            self.status_label.setText("Running")

    def open_options(self):
        dlg = OptionsDialog(self.training_cfg, self.dqn_cfg, self.reward_cfg, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.training_cfg, self.dqn_cfg, self.reward_cfg = dlg.values()

    def recalibrate_board(self):
        window_title = self.window_box.currentText().strip()
        if not window_title:
            QMessageBox.warning(self, "Missing Window", "Select a window first.")
            return
        try:
            calibration = BoardVision.run_calibration(window_title)
            BoardVision.save_calibration("calibration.json", calibration)
            QMessageBox.information(self, "Calibration", "Calibration saved to calibration.json")
        except Exception as exc:
            QMessageBox.critical(self, "Calibration Failed", str(exc))

    def save_training_data(self):
        if not self.worker or not self.worker.isRunning():
            QMessageBox.information(self, "Info", "Training data is saved automatically when training stops.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save training data", "models/session_checkpoint.pt", "PyTorch (*.pt *.pth)")
        if not path:
            return
        self.checkpoint_edit.setText(path)
        QMessageBox.information(self, "Info", "Checkpoint path updated. It will be saved when run completes.")

    def load_training_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load training data", "", "PyTorch (*.pt *.pth)")
        if not path:
            return
        try:
            payload = load_checkpoint(path, self.training_cfg.device)
            train_payload = payload.get("training_config", {})
            dqn_payload = payload.get("dqn_config", {})
            reward_payload = payload.get("reward_config", {})
            self.training_cfg = replace(
                self.training_cfg,
                **{k: train_payload[k] for k in train_payload if hasattr(self.training_cfg, k)},
            )
            self.dqn_cfg = replace(self.dqn_cfg, **{k: dqn_payload[k] for k in dqn_payload if hasattr(self.dqn_cfg, k)})
            self.reward_cfg = replace(
                self.reward_cfg, **{k: reward_payload[k] for k in reward_payload if hasattr(self.reward_cfg, k)}
            )
            self.reward_history = list(payload.get("reward_history", []))
            self.chart.set_rewards(self.reward_history)
            self.checkpoint_in = path
            QMessageBox.information(self, "Loaded", "Training data loaded. Next start will resume from checkpoint.")
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def on_progress(self, metrics: dict):
        ep = int(metrics.get("episode", 0))
        step = int(metrics.get("step", 0))
        ep_reward = float(metrics.get("episode_reward", 0.0))
        if ep != self.episode_index:
            if self.episode_index > 0:
                self.reward_history.append(self.current_episode_reward)
            self.episode_index = ep
        self.current_episode_reward = ep_reward

        self.episode_label.setText(str(ep))
        self.step_label.setText(str(step))
        self.ep_reward_label.setText(f"{ep_reward:.2f}")
        self.last_reward_label.setText(f"{float(metrics.get('last_reward', 0.0)):.2f}")
        self.total_steps_label.setText(str(int(metrics.get("total_steps", 0))))
        self.epsilon_label.setText(f"{float(metrics.get('epsilon', 0.0)):.3f}")
        self.loss_label.setText(f"{float(metrics.get('loss', 0.0)):.4f}")
        if int(metrics.get("transition", 0)) == 1:
            self.status_label.setText("Transition Pause")
        plot_rewards = self.reward_history + [self.current_episode_reward]
        self.chart.set_rewards(plot_rewards)

    def on_training_finished(self, result: dict):
        self._set_idle_state()
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        rewards = result.get("reward_history", [])
        self.reward_history = rewards
        self.current_episode_reward = 0.0
        self.episode_index = 0
        self.chart.set_rewards(rewards)
        self.checkpoint_in = self.checkpoint_edit.text().strip() or self.checkpoint_in
        mode = result.get("run_mode", "train")
        title = "Play Complete" if mode == "play" else "Training Complete"
        QMessageBox.information(
            self,
            title,
            f"Episodes completed: {result.get('episodes_completed', 0)}\nModel: {result.get('model_out', '')}",
        )

    def on_training_failed(self, message: str):
        self._set_idle_state()
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        QMessageBox.critical(self, "Training Failed", message)

    def closeEvent(self, event):
        if self.control:
            self.control.request_stop()
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)
        super().closeEvent(event)
