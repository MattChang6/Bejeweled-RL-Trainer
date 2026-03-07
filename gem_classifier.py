from dataclasses import dataclass
from typing import List, Tuple

import cv2 as cv
import numpy as np
import torch
from torch import nn


@dataclass
class GemClassifierBundle:
    model: nn.Module
    class_names: List[str]
    input_size: int
    device: str


class GemClassifierCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def preprocess_cell(cell_bgr: np.ndarray, input_size: int) -> np.ndarray:
    resized = cv.resize(cell_bgr, (input_size, input_size), interpolation=cv.INTER_AREA)
    rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return chw


def load_gem_classifier(checkpoint_path: str, device: str = "cpu") -> GemClassifierBundle:
    payload = torch.load(checkpoint_path, map_location=device)
    class_names = payload["class_names"]
    input_size = int(payload.get("input_size", 32))
    model = GemClassifierCNN(num_classes=len(class_names))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return GemClassifierBundle(model=model, class_names=class_names, input_size=input_size, device=device)


def infer_cells(bundle: GemClassifierBundle, cells_bgr: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    batch = np.stack([preprocess_cell(cell, bundle.input_size) for cell in cells_bgr], axis=0)
    tensor = torch.tensor(batch, dtype=torch.float32, device=bundle.device)
    with torch.no_grad():
        logits = bundle.model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return idx.cpu().numpy().astype(np.int32), conf.cpu().numpy().astype(np.float32)
