import json
import os
from dataclasses import asdict, replace
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

        note = QLabel(
            "Training always uses score-based rewards. Reward comes from the scoreboard change after each move."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        layout.addLayout(form)

        self.episodes = QSpinBox()
        self.episodes.setRange(1, 100000)
        self.episodes.setValue(cfg.episodes)
        self.episodes.setToolTip("Total number of episodes to run before training stops.")
        form.addRow("Episodes", self.episodes)

        self.max_steps = QSpinBox()
        self.max_steps.setRange(1, 10000)
        self.max_steps.setValue(cfg.max_steps)
        self.max_steps.setToolTip("Maximum number of scored moves allowed per episode.")
        form.addRow("Max Steps", self.max_steps)

        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(1e-6, 1.0)
        self.lr.setSingleStep(1e-4)
        self.lr.setValue(dqn_cfg.lr)
        self.lr.setToolTip("Gradient step size used when updating the Q-network.")
        form.addRow("Learning Rate", self.lr)

        self.gamma = QDoubleSpinBox()
        self.gamma.setDecimals(4)
        self.gamma.setRange(0.0, 0.9999)
        self.gamma.setSingleStep(0.01)
        self.gamma.setValue(dqn_cfg.gamma)
        self.gamma.setToolTip("Discount factor for future rewards. Higher values favor long-term reward.")
        form.addRow("Gamma", self.gamma)

        self.batch = QSpinBox()
        self.batch.setRange(1, 4096)
        self.batch.setValue(dqn_cfg.batch_size)
        self.batch.setToolTip("Number of replay samples used per optimizer update.")
        form.addRow("Batch Size", self.batch)

        self.target_update = QSpinBox()
        self.target_update.setRange(1, 1000000)
        self.target_update.setValue(dqn_cfg.target_update)
        self.target_update.setToolTip("How many training steps occur before copying weights to the target network.")
        form.addRow("Target Update Steps", self.target_update)

        self.replay_size = QSpinBox()
        self.replay_size.setRange(100, 2000000)
        self.replay_size.setValue(dqn_cfg.replay_size)
        self.replay_size.setToolTip("Maximum number of experiences stored in the replay buffer.")
        form.addRow("Replay Size", self.replay_size)

        self.min_replay = QSpinBox()
        self.min_replay.setRange(1, 2000000)
        self.min_replay.setValue(dqn_cfg.min_replay)
        self.min_replay.setToolTip("Minimum replay samples required before gradient updates begin.")
        form.addRow("Min Replay", self.min_replay)

        self.invalid_penalty = QDoubleSpinBox()
        self.invalid_penalty.setRange(0.0, 100.0)
        self.invalid_penalty.setValue(reward_cfg.invalid_penalty)
        self.invalid_penalty.setToolTip("Penalty applied when the selected swap is not a valid move.")
        form.addRow("Invalid Penalty", self.invalid_penalty)

        self.step_penalty = QDoubleSpinBox()
        self.step_penalty.setDecimals(4)
        self.step_penalty.setRange(0.0, 10.0)
        self.step_penalty.setValue(reward_cfg.step_penalty)
        self.step_penalty.setToolTip("Small cost applied to each move to favor efficient play.")
        form.addRow("Step Penalty", self.step_penalty)

        self.log_steps = QCheckBox("Log every step")
        self.log_steps.setChecked(cfg.log_steps)
        self.log_steps.setToolTip("Print a console log for every environment step.")
        form.addRow("", self.log_steps)

        self.debug_view = QCheckBox("Show debug board window")
        self.debug_view.setChecked(cfg.debug_view)
        self.debug_view.setToolTip("Show the detected board overlay window during training or play.")
        form.addRow("", self.debug_view)

        self.poll_hotkeys = QCheckBox("Enable global hotkeys (P pause/resume, Q stop)")
        self.poll_hotkeys.setChecked(cfg.poll_hotkeys)
        self.poll_hotkeys.setToolTip("Allow global P and Q key handling even when the console is not focused.")
        form.addRow("", self.poll_hotkeys)

        self.transition_enabled = QCheckBox("Enable transition detection auto-pause")
        self.transition_enabled.setChecked(cfg.transition_enabled)
        self.transition_enabled.setToolTip("Pause actions automatically when the scoreboard becomes unreadable during a level transition.")
        form.addRow("", self.transition_enabled)

        self.transition_pause_seconds = QDoubleSpinBox()
        self.transition_pause_seconds.setRange(0.0, 60.0)
        self.transition_pause_seconds.setDecimals(1)
        self.transition_pause_seconds.setValue(cfg.transition_pause_seconds)
        self.transition_pause_seconds.setToolTip("How long to pause mouse actions after a transition is detected.")
        form.addRow("Transition Pause (s)", self.transition_pause_seconds)

        self.transition_frames = QSpinBox()
        self.transition_frames.setRange(1, 20)
        self.transition_frames.setValue(cfg.transition_consecutive_frames)
        self.transition_frames.setToolTip("Number of consecutive unreadable score captures required before a transition pause starts.")
        form.addRow("Transition Frames", self.transition_frames)

        self.score_stable_frames = QSpinBox()
        self.score_stable_frames.setRange(1, 20)
        self.score_stable_frames.setValue(cfg.score_stable_frames)
        self.score_stable_frames.setToolTip("Number of matching scoreboard captures required before a score read is accepted as stable.")
        form.addRow("Score Stable Frames", self.score_stable_frames)

        self.score_stable_threshold = QDoubleSpinBox()
        self.score_stable_threshold.setRange(0.0, 20.0)
        self.score_stable_threshold.setDecimals(2)
        self.score_stable_threshold.setValue(cfg.score_stable_threshold)
        self.score_stable_threshold.setToolTip("Maximum average pixel difference allowed for two score captures to count as the same frame.")
        form.addRow("Score Stable Threshold", self.score_stable_threshold)

        self.score_reward_scale = QDoubleSpinBox()
        self.score_reward_scale.setRange(0.0, 1000.0)
        self.score_reward_scale.setDecimals(2)
        self.score_reward_scale.setValue(cfg.score_reward_scale)
        self.score_reward_scale.setToolTip("Multiplier applied to the raw score increase before it becomes reward.")
        form.addRow("Score Reward Scale", self.score_reward_scale)

        self.score_match_threshold = QDoubleSpinBox()
        self.score_match_threshold.setRange(0.1, 1.0)
        self.score_match_threshold.setDecimals(2)
        self.score_match_threshold.setValue(cfg.score_match_threshold)
        self.score_match_threshold.setToolTip("Minimum template match score required for the OCR reader to accept a digit.")
        form.addRow("Score OCR Match Thresh", self.score_match_threshold)

        self.score_debug_print = QCheckBox("Print score to console")
        self.score_debug_print.setChecked(cfg.score_debug_print)
        self.score_debug_print.setToolTip("Print each OCR score read to the console for debugging.")
        form.addRow("", self.score_debug_print)

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
            transition_consecutive_frames=self.transition_frames.value(),
            score_enabled=True,
            score_stable_frames=self.score_stable_frames.value(),
            score_stable_threshold=self.score_stable_threshold.value(),
            score_reward_scale=self.score_reward_scale.value(),
            score_match_threshold=self.score_match_threshold.value(),
            score_debug_print=self.score_debug_print.isChecked(),
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
            invalid_penalty=self.invalid_penalty.value(),
            step_penalty=self.step_penalty.value(),
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
        initial_model_in: Optional[str],
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
        self.initial_model_in = initial_model_in
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
                    initial_model_in=self.initial_model_in,
                )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class WindowCaptureGUI(QMainWindow):
    GAME_WINDOW_TITLE = "Bejeweled 3"
    SETTINGS_FILENAME = "gui_settings.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bejeweled 3 RL Trainer")
        self.resize(980, 760)

        self.training_cfg = TrainingConfig(window_title=self.GAME_WINDOW_TITLE, poll_hotkeys=True)
        self.dqn_cfg = DQNConfig()
        self.reward_cfg = RewardConfig()

        self.control: Optional[TrainingControl] = None
        self.worker: Optional[TrainingWorker] = None
        self.checkpoint_in: Optional[str] = None
        self.reward_history = []
        self.episode_index = 0
        self.current_episode_reward = 0.0

        self._build_ui()
        self._load_settings()
        self._set_idle_state()

    def _build_ui(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #f4f6fb; }
            QGroupBox {
                background: white;
                border: 1px solid #d6dbe7;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                min-height: 32px;
                padding: 4px 12px;
            }
            QLabel#titleLabel {
                font-size: 20px;
                font-weight: 700;
                color: #172033;
            }
            QLabel#mutedLabel {
                color: #51607a;
            }
            QLabel#valueLabel {
                font-weight: 600;
                color: #172033;
            }
            QLabel#statusValue {
                font-weight: 700;
            }
            """
        )

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        overview_group = QGroupBox("Overview")
        overview_layout = QVBoxLayout(overview_group)
        title = QLabel("Bejeweled 3 RL Trainer")
        title.setObjectName("titleLabel")
        subtitle = QLabel(
            "Workflow: 1. Verify the game window. 2. Choose files. 3. Optionally load a checkpoint or warm-start model. 4. Train or play."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("mutedLabel")
        overview_layout.addWidget(title)
        overview_layout.addWidget(subtitle)
        layout.addWidget(overview_group)

        content_row = QHBoxLayout()
        content_row.setSpacing(12)
        layout.addLayout(content_row, 1)

        left_col = QVBoxLayout()
        left_col.setSpacing(12)
        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        content_row.addLayout(left_col, 3)
        content_row.addLayout(right_col, 2)

        status_group = QGroupBox("Game Status")
        status_grid = QGridLayout(status_group)
        self.game_window_value = QLabel(self.GAME_WINDOW_TITLE)
        self.game_window_value.setObjectName("valueLabel")
        self.window_status_value = QLabel("Checking...")
        self.window_status_value.setObjectName("statusValue")
        self.window_check_btn = QPushButton("Check Window")
        self.window_check_btn.setToolTip("Verify that the Bejeweled 3 window is currently visible.")
        self.window_check_btn.clicked.connect(self.refresh_window_status)
        self.calibrate_btn = QPushButton("Recalibrate Board")
        self.calibrate_btn.setToolTip("Re-select the board corners if the board capture is misaligned.")
        self.calibrate_score_btn = QPushButton("Calibrate Scoreboard")
        self.calibrate_score_btn.setToolTip("Re-select the scoreboard region used for OCR-based rewards.")
        self.calibrate_btn.clicked.connect(self.recalibrate_board)
        self.calibrate_score_btn.clicked.connect(self.recalibrate_scoreboard)

        status_help = QLabel("The trainer only runs when a visible window titled Bejeweled 3 is detected.")
        status_help.setWordWrap(True)
        status_help.setObjectName("mutedLabel")
        status_grid.addWidget(QLabel("Game Window"), 0, 0)
        status_grid.addWidget(self.game_window_value, 0, 1)
        status_grid.addWidget(self.window_check_btn, 0, 2)
        status_grid.addWidget(QLabel("Detection"), 1, 0)
        status_grid.addWidget(self.window_status_value, 1, 1)
        status_grid.addWidget(status_help, 2, 0, 1, 3)
        status_grid.addWidget(self.calibrate_btn, 3, 0, 1, 2)
        status_grid.addWidget(self.calibrate_score_btn, 3, 2)
        left_col.addWidget(status_group)

        files_group = QGroupBox("Training Files")
        files_grid = QGridLayout(files_group)
        files_help = QLabel(
            "Checkpoint = full training state resume. Warm-start model = weights only. Saved model = file used for play mode and exported after training."
        )
        files_help.setWordWrap(True)
        files_help.setObjectName("mutedLabel")
        files_grid.addWidget(files_help, 0, 0, 1, 3)

        self.model_out_edit = QLineEdit("models/bejeweled_dqn.pt")
        self.model_out_edit.setToolTip("Where trained model weights are saved and which model play mode will load.")
        self.model_out_edit.setPlaceholderText("Path to saved model weights")
        model_btn = QPushButton("Browse")
        model_btn.clicked.connect(self.select_model_out)
        model_btn.setToolTip("Choose the model weights file used for saving and play mode.")
        files_grid.addWidget(QLabel("Saved Model"), 1, 0)
        files_grid.addWidget(self.model_out_edit, 1, 1)
        files_grid.addWidget(model_btn, 1, 2)

        self.checkpoint_edit = QLineEdit("models/bejeweled_checkpoint.pt")
        self.checkpoint_edit.setToolTip("Where the full checkpoint is written when training saves session state.")
        self.checkpoint_edit.setPlaceholderText("Path to training checkpoint")
        checkpoint_btn = QPushButton("Browse")
        checkpoint_btn.clicked.connect(self.select_checkpoint_out)
        checkpoint_btn.setToolTip("Choose the checkpoint path used to save training state.")
        files_grid.addWidget(QLabel("Checkpoint Save"), 2, 0)
        files_grid.addWidget(self.checkpoint_edit, 2, 1)
        files_grid.addWidget(checkpoint_btn, 2, 2)

        self.warm_start_edit = QLineEdit("")
        self.warm_start_edit.setToolTip("Optional model weights to initialize a new training run from an already-trained model.")
        self.warm_start_edit.setPlaceholderText("Optional: continue training from an existing model file")
        warm_start_btn = QPushButton("Browse")
        warm_start_btn.clicked.connect(self.select_warm_start_model)
        warm_start_btn.setToolTip("Choose a model file whose weights should seed the next training run.")
        files_grid.addWidget(QLabel("Warm-Start Model"), 3, 0)
        files_grid.addWidget(self.warm_start_edit, 3, 1)
        files_grid.addWidget(warm_start_btn, 3, 2)

        self.classifier_edit = QLineEdit("models/gem_classifier.pt")
        self.classifier_edit.setToolTip("Gem classifier model used by the vision system.")
        self.classifier_edit.setPlaceholderText("Path to gem classifier model")
        classifier_btn = QPushButton("Browse")
        classifier_btn.clicked.connect(self.select_classifier_path)
        classifier_btn.setToolTip("Choose the gem classifier model used for board recognition.")
        files_grid.addWidget(QLabel("Gem Classifier"), 4, 0)
        files_grid.addWidget(self.classifier_edit, 4, 1)
        files_grid.addWidget(classifier_btn, 4, 2)

        self.resume_source_value = QLabel("New training run")
        self.resume_source_value.setObjectName("valueLabel")
        files_grid.addWidget(QLabel("Training Source"), 5, 0)
        files_grid.addWidget(self.resume_source_value, 5, 1, 1, 2)

        self.model_out_edit.editingFinished.connect(self._save_settings)
        self.checkpoint_edit.editingFinished.connect(self._save_settings)
        self.classifier_edit.editingFinished.connect(self._save_settings)
        self.warm_start_edit.editingFinished.connect(self._on_warm_start_changed)
        left_col.addWidget(files_group)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        actions_help = QLabel(
            "Use Load Checkpoint to continue an exact prior run, or set Warm-Start Model to begin a fresh run from saved weights."
        )
        actions_help.setWordWrap(True)
        actions_help.setObjectName("mutedLabel")
        actions_layout.addWidget(actions_help)

        self.start_btn = QPushButton("Start Training")
        self.play_btn = QPushButton("Play Saved Model")
        self.stop_btn = QPushButton("Stop")
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.options_btn = QPushButton("Options")
        self.save_training_btn = QPushButton("Set Checkpoint Save Path")
        self.load_training_btn = QPushButton("Load Checkpoint")
        self.load_model_btn = QPushButton("Use Warm-Start Model")
        self.clear_resume_btn = QPushButton("Clear Resume Source")

        self.start_btn.setToolTip("Start or continue training using the current files and options.")
        self.play_btn.setToolTip("Run the saved model without learning updates.")
        self.stop_btn.setToolTip("Stop the current training or play session.")
        self.pause_btn.setToolTip("Pause mouse actions so you can regain control.")
        self.resume_btn.setToolTip("Resume a paused training or play session.")
        self.options_btn.setToolTip("Open training hyperparameters and score OCR settings.")
        self.save_training_btn.setToolTip("Choose where the full training checkpoint should be saved.")
        self.load_training_btn.setToolTip("Load a full checkpoint to resume an existing training run.")
        self.load_model_btn.setToolTip("Use a saved model's weights to seed a new training run.")
        self.clear_resume_btn.setToolTip("Clear any loaded checkpoint or warm-start model selection.")

        self.start_btn.clicked.connect(self.start_training)
        self.play_btn.clicked.connect(self.start_play_mode)
        self.stop_btn.clicked.connect(self.stop_training)
        self.pause_btn.clicked.connect(self.pause_training)
        self.resume_btn.clicked.connect(self.resume_training)
        self.options_btn.clicked.connect(self.open_options)
        self.save_training_btn.clicked.connect(self.save_training_data)
        self.load_training_btn.clicked.connect(self.load_training_data)
        self.load_model_btn.clicked.connect(self.select_warm_start_model)
        self.clear_resume_btn.clicked.connect(self.clear_resume_source)

        primary_row = QHBoxLayout()
        primary_row.addWidget(self.start_btn)
        primary_row.addWidget(self.play_btn)
        primary_row.addWidget(self.stop_btn)
        primary_row.addWidget(self.pause_btn)
        primary_row.addWidget(self.resume_btn)
        actions_layout.addLayout(primary_row)

        secondary_row = QHBoxLayout()
        secondary_row.addWidget(self.load_training_btn)
        secondary_row.addWidget(self.load_model_btn)
        secondary_row.addWidget(self.clear_resume_btn)
        secondary_row.addWidget(self.options_btn)
        actions_layout.addLayout(secondary_row)

        utility_row = QHBoxLayout()
        utility_row.addWidget(self.save_training_btn)
        utility_row.addStretch(1)
        actions_layout.addLayout(utility_row)
        left_col.addWidget(actions_group)
        left_col.addStretch(1)

        metrics_group = QGroupBox("Live Metrics")
        metrics_layout = QGridLayout(metrics_group)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("valueLabel")
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
        metrics_layout.addWidget(QLabel("Step"), 1, 0)
        metrics_layout.addWidget(self.step_label, 1, 1)
        metrics_layout.addWidget(QLabel("Total Steps"), 1, 2)
        metrics_layout.addWidget(self.total_steps_label, 1, 3)
        metrics_layout.addWidget(QLabel("Episode Reward"), 2, 0)
        metrics_layout.addWidget(self.ep_reward_label, 2, 1)
        metrics_layout.addWidget(QLabel("Last Reward"), 2, 2)
        metrics_layout.addWidget(self.last_reward_label, 2, 3)
        metrics_layout.addWidget(QLabel("Epsilon"), 3, 0)
        metrics_layout.addWidget(self.epsilon_label, 3, 1)
        metrics_layout.addWidget(QLabel("Loss"), 3, 2)
        metrics_layout.addWidget(self.loss_label, 3, 3)
        right_col.addWidget(metrics_group)

        chart_group = QGroupBox("Reward Trend")
        chart_layout = QVBoxLayout(chart_group)
        chart_help = QLabel("Episode reward history for the current session or loaded checkpoint.")
        chart_help.setWordWrap(True)
        chart_help.setObjectName("mutedLabel")
        self.chart = RewardChartWidget()
        chart_layout.addWidget(chart_help)
        chart_layout.addWidget(self.chart)
        right_col.addWidget(chart_group, 1)

    def _settings_path(self) -> str:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(repo_root, self.SETTINGS_FILENAME)

    def _replace_dataclass(self, current, payload: dict):
        if not isinstance(payload, dict):
            return current
        allowed = set(getattr(current, "__dataclass_fields__", {}).keys())
        updates = {key: value for key, value in payload.items() if key in allowed}
        if not updates:
            return current
        return replace(current, **updates)

    def _find_game_window(self) -> int:
        matches = []

        def win_enum_handler(hwnd, _ctx):
            if not win32gui.IsWindowVisible(hwnd) or not win32gui.IsWindowEnabled(hwnd):
                return
            title = win32gui.GetWindowText(hwnd).strip()
            if title == self.GAME_WINDOW_TITLE:
                matches.append(hwnd)

        win32gui.EnumWindows(win_enum_handler, None)
        return matches[0] if matches else 0

    def refresh_window_status(self) -> bool:
        available = self._find_game_window() != 0
        if available:
            self.window_status_value.setText("Detected")
            self.window_status_value.setStyleSheet("color: #146c2e;")
        else:
            self.window_status_value.setText("Not detected")
            self.window_status_value.setStyleSheet("color: #b42318;")
        return available

    def _sync_config_from_inputs(self) -> None:
        self.training_cfg = replace(
            self.training_cfg,
            window_title=self.GAME_WINDOW_TITLE,
            score_enabled=True,
            model_out=self.model_out_edit.text().strip() or self.training_cfg.model_out,
            classifier_path=self.classifier_edit.text().strip() or self.training_cfg.classifier_path,
        )

    def _warm_start_path(self) -> Optional[str]:
        path = self.warm_start_edit.text().strip()
        return path or None

    def _on_warm_start_changed(self) -> None:
        if self._warm_start_path():
            self.checkpoint_in = None
        self._update_resume_source()
        self._save_settings()

    def _update_resume_source(self) -> None:
        if self.checkpoint_in:
            label = f"Checkpoint resume: {os.path.basename(self.checkpoint_in)}"
            color = "#146c2e"
        else:
            warm_start = self._warm_start_path()
            if warm_start:
                if os.path.exists(warm_start):
                    label = f"Warm-start model: {os.path.basename(warm_start)}"
                    color = "#146c2e"
                else:
                    label = f"Warm-start model missing: {os.path.basename(warm_start)}"
                    color = "#b42318"
            else:
                label = "New training run"
                color = "#172033"
        self.resume_source_value.setText(label)
        self.resume_source_value.setToolTip(label)
        self.resume_source_value.setStyleSheet(f"color: {color};")

    def _load_settings(self) -> None:
        self._sync_config_from_inputs()
        path = self._settings_path()
        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"[settings] failed to load {path}: {exc}")
            return

        self.training_cfg = self._replace_dataclass(self.training_cfg, payload.get("training_config", {}))
        self.training_cfg = replace(self.training_cfg, window_title=self.GAME_WINDOW_TITLE, score_enabled=True)
        self.dqn_cfg = self._replace_dataclass(self.dqn_cfg, payload.get("dqn_config", {}))
        self.reward_cfg = self._replace_dataclass(self.reward_cfg, payload.get("reward_config", {}))

        paths = payload.get("paths", {})
        model_out = paths.get("model_out") or self.training_cfg.model_out
        checkpoint_out = paths.get("checkpoint_out") or self.checkpoint_edit.text()
        classifier_path = paths.get("classifier_path") or self.training_cfg.classifier_path
        warm_start_model = paths.get("warm_start_model") or ""

        if isinstance(model_out, str):
            self.model_out_edit.setText(model_out)
        if isinstance(checkpoint_out, str):
            self.checkpoint_edit.setText(checkpoint_out)
        if isinstance(classifier_path, str):
            self.classifier_edit.setText(classifier_path)
        if isinstance(warm_start_model, str):
            self.warm_start_edit.setText(warm_start_model)

        self._sync_config_from_inputs()
        self._update_resume_source()

    def _save_settings(self) -> None:
        self._sync_config_from_inputs()
        payload = {
            "training_config": asdict(self.training_cfg),
            "dqn_config": asdict(self.dqn_cfg),
            "reward_config": asdict(self.reward_cfg),
            "paths": {
                "model_out": self.model_out_edit.text().strip(),
                "checkpoint_out": self.checkpoint_edit.text().strip(),
                "classifier_path": self.classifier_edit.text().strip(),
                "warm_start_model": self.warm_start_edit.text().strip(),
            },
        }

        path = self._settings_path()
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            print(f"[settings] failed to save {path}: {exc}")

    def _require_game_window(self) -> bool:
        if self.refresh_window_status():
            return True
        QMessageBox.warning(
            self,
            "Game Window Missing",
            f'Could not find a visible window titled "{self.GAME_WINDOW_TITLE}". Open the game before running.',
        )
        return False

    def select_model_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Model output", self.model_out_edit.text(), "PyTorch (*.pt *.pth)")
        if path:
            self.model_out_edit.setText(path)
            self._save_settings()

    def select_checkpoint_out(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Checkpoint output", self.checkpoint_edit.text(), "PyTorch checkpoint (*.pt *.pth)"
        )
        if path:
            self.checkpoint_edit.setText(path)
            self._save_settings()

    def select_classifier_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Gem classifier model", self.classifier_edit.text(), "PyTorch (*.pt *.pth)"
        )
        if path:
            self.classifier_edit.setText(path)
            self._save_settings()

    def select_warm_start_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Warm-start model weights", self.warm_start_edit.text(), "PyTorch (*.pt *.pth)"
        )
        if path:
            self.warm_start_edit.setText(path)
            self.checkpoint_in = None
            self._update_resume_source()
            self._save_settings()

    def clear_resume_source(self):
        self.checkpoint_in = None
        self.warm_start_edit.clear()
        self._update_resume_source()
        self._save_settings()

    def _set_idle_state(self):
        window_available = self.refresh_window_status()
        self.start_btn.setEnabled(window_available)
        self.play_btn.setEnabled(window_available)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.calibrate_btn.setEnabled(window_available)
        self.calibrate_score_btn.setEnabled(window_available)
        self.window_check_btn.setEnabled(True)
        self.load_training_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        self.clear_resume_btn.setEnabled(True)
        self.save_training_btn.setEnabled(True)
        self.options_btn.setEnabled(True)
        self.status_label.setText("Idle")
        self._update_resume_source()

    def _set_running_state(self, mode_label: str = "Running"):
        self.start_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.calibrate_btn.setEnabled(False)
        self.calibrate_score_btn.setEnabled(False)
        self.window_check_btn.setEnabled(False)
        self.load_training_btn.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        self.clear_resume_btn.setEnabled(False)
        self.save_training_btn.setEnabled(False)
        self.options_btn.setEnabled(False)
        self.status_label.setText(mode_label)

    def start_training(self):
        self._start_session(run_mode="train")

    def start_play_mode(self):
        self._start_session(run_mode="play")

    def _start_session(self, run_mode: str):
        if self.worker and self.worker.isRunning():
            return
        if not self._require_game_window():
            self._set_idle_state()
            return

        model_path = self.model_out_edit.text().strip() or "models/bejeweled_dqn.pt"
        warm_start_path = self._warm_start_path()
        if run_mode == "play" and not os.path.exists(model_path):
            QMessageBox.warning(self, "Missing Model", "Select an existing trained model file for play mode.")
            return
        if run_mode == "train" and self.checkpoint_in and not os.path.exists(self.checkpoint_in):
            QMessageBox.warning(self, "Missing Checkpoint", "The loaded checkpoint file no longer exists. Load it again or clear the resume source.")
            self.checkpoint_in = None
            self._update_resume_source()
            return
        if run_mode == "train" and not self.checkpoint_in and warm_start_path and not os.path.exists(warm_start_path):
            QMessageBox.warning(self, "Missing Warm-Start Model", "The selected warm-start model file does not exist.")
            self._update_resume_source()
            return

        self._sync_config_from_inputs()
        checkpoint_out = self.checkpoint_edit.text().strip() or None
        initial_model_in = warm_start_path if run_mode == "train" and not self.checkpoint_in else None
        self.control = TrainingControl()
        self.worker = TrainingWorker(
            cfg=self.training_cfg,
            dqn_cfg=self.dqn_cfg,
            reward_cfg=self.reward_cfg,
            checkpoint_in=self.checkpoint_in,
            checkpoint_out=checkpoint_out,
            run_mode=run_mode,
            model_in=model_path,
            initial_model_in=initial_model_in,
            control=self.control,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_training_finished)
        self.worker.failed.connect(self.on_training_failed)
        self.worker.start()
        self._save_settings()
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
            self.training_cfg = replace(self.training_cfg, window_title=self.GAME_WINDOW_TITLE, score_enabled=True)
            self._update_resume_source()
            self._save_settings()

    def recalibrate_board(self):
        if not self._require_game_window():
            return
        try:
            calibration = BoardVision.run_calibration(self.GAME_WINDOW_TITLE)
            BoardVision.save_calibration("calibration.json", calibration)
            QMessageBox.information(self, "Calibration", "Calibration saved to calibration.json")
        except Exception as exc:
            QMessageBox.critical(self, "Calibration Failed", str(exc))

    def recalibrate_scoreboard(self):
        if not self._require_game_window():
            return
        try:
            calibration = BoardVision.run_score_calibration(self.GAME_WINDOW_TITLE)
            BoardVision.save_score_calibration("score_calibration.json", calibration)
            QMessageBox.information(self, "Score Calibration", "Score calibration saved to score_calibration.json")
        except Exception as exc:
            QMessageBox.critical(self, "Score Calibration Failed", str(exc))

    def save_training_data(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Set checkpoint save path",
            self.checkpoint_edit.text() or "models/session_checkpoint.pt",
            "PyTorch (*.pt *.pth)",
        )
        if not path:
            return
        self.checkpoint_edit.setText(path)
        self._save_settings()
        QMessageBox.information(self, "Checkpoint Path", "Checkpoint save path updated.")

    def load_training_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load checkpoint", "", "PyTorch (*.pt *.pth)")
        if not path:
            return
        try:
            payload = load_checkpoint(path, self.training_cfg.device)
            train_payload = payload.get("training_config", {})
            dqn_payload = payload.get("dqn_config", {})
            reward_payload = payload.get("reward_config", {})
            self.training_cfg = self._replace_dataclass(self.training_cfg, train_payload)
            self.training_cfg = replace(self.training_cfg, window_title=self.GAME_WINDOW_TITLE, score_enabled=True)
            self.dqn_cfg = self._replace_dataclass(self.dqn_cfg, dqn_payload)
            self.reward_cfg = self._replace_dataclass(self.reward_cfg, reward_payload)
            self.reward_history = list(payload.get("reward_history", []))
            self.chart.set_rewards(self.reward_history)
            self.checkpoint_in = path
            self.checkpoint_edit.setText(path)
            self.warm_start_edit.clear()
            self.model_out_edit.setText(self.training_cfg.model_out)
            self.classifier_edit.setText(self.training_cfg.classifier_path)
            self._update_resume_source()
            self._save_settings()
            QMessageBox.information(self, "Loaded", "Checkpoint loaded. The next training run will resume from this saved session state.")
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
        mode = result.get("run_mode", "train")
        checkpoint_path = self.checkpoint_edit.text().strip()
        if mode == "train" and checkpoint_path and os.path.exists(checkpoint_path):
            self.checkpoint_in = checkpoint_path
        self._update_resume_source()
        self._save_settings()
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
        self._update_resume_source()
        QMessageBox.critical(self, "Training Failed", message)

    def closeEvent(self, event):
        self._save_settings()
        if self.control:
            self.control.request_stop()
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)
        super().closeEvent(event)
