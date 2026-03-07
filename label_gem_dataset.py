import argparse
import os
import shutil

import cv2 as cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label captured gem cell images with keyboard.")
    parser.add_argument("--source", default="dataset/unlabeled")
    parser.add_argument("--dest", default="dataset/labeled")
    parser.add_argument("--classes", type=int, default=7, help="Number of gem classes. Uses keys 0..classes-1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.source):
        raise FileNotFoundError(f"Source folder not found: {args.source}")

    for i in range(args.classes):
        os.makedirs(os.path.join(args.dest, str(i)), exist_ok=True)

    files = sorted([f for f in os.listdir(args.source) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    total = len(files)
    if total == 0:
        print("No files to label.")
        return

    print("Keys: 0-9 label | s skip | q quit")
    idx = 0
    while idx < total:
        filename = files[idx]
        src_path = os.path.join(args.source, filename)
        img = cv.imread(src_path)
        if img is None:
            idx += 1
            continue

        display = cv.resize(img, (256, 256), interpolation=cv.INTER_NEAREST)
        cv.putText(display, f"{idx+1}/{total}: {filename}", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.imshow("Label Gem Cells", display)
        key = cv.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            idx += 1
            continue
        if ord("0") <= key <= ord("9"):
            label = key - ord("0")
            if label >= args.classes:
                print(f"Invalid class {label}; max class is {args.classes - 1}")
                continue
            dst_path = os.path.join(args.dest, str(label), filename)
            shutil.move(src_path, dst_path)
            idx += 1
            continue

    cv.destroyAllWindows()
    print("Labeling finished.")


if __name__ == "__main__":
    main()
