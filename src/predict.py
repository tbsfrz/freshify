from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from torchvision import transforms

from src.model import IMG_SIZE, load_pepper_model

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_model: Optional[torch.nn.Module] = None


def _prepare_image(image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    return TRANSFORM(image).unsqueeze(0)


def _get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        _model = load_pepper_model()
    return _model


def predict(image_input: Union[str, Path, Image.Image]) -> tuple[str, float, float]:
    model = _get_model()
    tensor = _prepare_image(image_input).to(next(model.parameters()).device)

    with torch.no_grad():
        prediction = model(tensor).item()

    if prediction >= 0.5:
        label = "non_edible"
        confidence = float(prediction)
    else:
        label = "edible"
        confidence = float(1.0 - prediction)

    return label, confidence, float(prediction)
