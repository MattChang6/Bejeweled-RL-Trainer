import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gem_classifier import GemClassifierCNN, preprocess_cell


class GemDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], input_size: int):
        self.samples = samples
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        x = preprocess_cell(img, self.input_size)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a gem classifier from labeled cell images.")
    parser.add_argument("--dataset", default="dataset/labeled")
    parser.add_argument("--out", default="models/gem_classifier.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def discover_samples(dataset_dir: str) -> Tuple[List[str], List[Tuple[str, int]]]:
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if not class_names:
        raise ValueError(f"No class folders in {dataset_dir}")
    samples: List[Tuple[str, int]] = []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        for name in os.listdir(class_dir):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(class_dir, name), idx))
    if not samples:
        raise ValueError(f"No images found in {dataset_dir}")
    return class_names, samples


def split_samples(samples: List[Tuple[str, int]], val_split: float) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    rng = np.random.default_rng(42)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    split = int(len(samples) * (1.0 - val_split))
    train_idx = idx[:split]
    val_idx = idx[split:]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * x.shape[0]
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += x.shape[0]
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    class_names, samples = discover_samples(args.dataset)
    train_samples, val_samples = split_samples(samples, args.val_split)
    if not train_samples:
        raise ValueError("Not enough samples for training.")

    train_ds = GemDataset(train_samples, args.img_size)
    val_ds = GemDataset(val_samples, args.img_size) if val_samples else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds else None

    model = GemClassifierCNN(num_classes=len(class_names)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * x.shape[0]
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += x.shape[0]

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, args.device)
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )
        else:
            print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "input_size": args.img_size,
    }
    torch.save(payload, args.out)
    print(f"Saved classifier to {args.out}")


if __name__ == "__main__":
    main()
