from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

IMG_SIZE = (224, 224)
MODEL_PATH = "models/classifier.pth"


class PepperClassifier(nn.Module):
    def __init__(self, pretrained: bool = False, num_classes: int = 1) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone.last_channel, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_pepper_model(
    checkpoint_path: str = MODEL_PATH,
    device: torch.device | None = None,
) -> PepperClassifier:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PepperClassifier(pretrained=False)
    model.to(device)

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint}. "
            "Train the model first with `python src/train.py` or place a valid checkpoint there."
        )

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model
