import argparse

from bejeweled_vision import BoardVision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate score area for OCR.")
    parser.add_argument("--window", required=True, help="Exact game window title")
    parser.add_argument("--out", default="score_calibration.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = BoardVision.run_score_calibration(args.window)
    BoardVision.save_score_calibration(args.out, calibration)
    print(f"Saved score calibration to {args.out}")


if __name__ == "__main__":
    main()
