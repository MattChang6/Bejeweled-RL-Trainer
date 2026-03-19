import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vision"))

from train import train_agent
from PyQt6.QtWidgets import QApplication
from window_capture_gui import WindowCaptureGUI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bejeweled 3 RL Trainer")
    parser.add_argument("--cli", action="store_true", help="Run command-line training instead of GUI")
    parser.add_argument("--window", help="Exact window title for Bejeweled 3 (CLI mode)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--model-out", default="models/bejeweled_dqn.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-steps", action="store_true")
    parser.add_argument("--debug-view", action="store_true")
    parser.add_argument("--reward-log", default="training_rewards.csv")
    parser.add_argument("--reward-plot", default="training_rewards.png")
    parser.add_argument("--classifier-path", default="models/gem_classifier.pt")
    parser.add_argument("--classifier-device", default="cpu")
    return parser.parse_args()


def run_gui() -> int:
    app = QApplication(sys.argv)
    window = WindowCaptureGUI()
    window.show()
    return app.exec()


if __name__ == "__main__":
    args = parse_args()
    if args.cli:
        if not args.window:
            raise ValueError("--window is required in --cli mode")
        train_agent(
            window_title=args.window,
            episodes=args.episodes,
            max_steps=args.max_steps,
            model_out=args.model_out,
            device=args.device,
            log_steps=args.log_steps,
            debug_view=args.debug_view,
            reward_log_path=args.reward_log,
            plot_path=args.reward_plot,
            classifier_path=args.classifier_path,
            classifier_device=args.classifier_device,
        )
    else:
        sys.exit(run_gui())
