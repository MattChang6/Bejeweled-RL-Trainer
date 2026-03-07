import argparse
import os
import time

import cv2 as cv

from bejeweled_vision import BoardVision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Bejeweled gem cells into an unlabeled dataset.")
    parser.add_argument("--window", required=True, help="Exact game window title")
    parser.add_argument("--out", default="dataset/unlabeled")
    parser.add_argument("--calibration", default="calibration.json")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between captures")
    parser.add_argument("--preview", action="store_true", help="Show live preview while capturing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = BoardVision.load_calibration(args.calibration)
    if calibration is None:
        calibration = BoardVision.run_calibration(args.window)
        BoardVision.save_calibration(args.calibration, calibration)

    os.makedirs(args.out, exist_ok=True)
    vision = BoardVision(args.window, calibration)

    for frame_idx in range(args.frames):
        board = vision.capture_board()
        cells = vision.cell_images(board)
        ts = int(time.time() * 1000)
        for i, cell in enumerate(cells):
            row = i // calibration.grid_size
            col = i % calibration.grid_size
            name = f"{ts}_{frame_idx:04d}_r{row}_c{col}.png"
            cv.imwrite(os.path.join(args.out, name), cell)

        if args.preview:
            labels = vision.board_state()
            annotated = vision.annotate_board(board, labels)
            cv.imshow("Dataset Capture Preview", annotated)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        time.sleep(args.interval)

    if args.preview:
        cv.destroyAllWindows()
    print(f"Saved captures to {args.out}")


if __name__ == "__main__":
    main()
