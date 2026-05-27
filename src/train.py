import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import IMG_SIZE, MODEL_PATH, PepperClassifier

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-3


def build_data_loaders(
    data_root: Path,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, list[str]]:
    transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(data_root / "train", transform=transform)
    val_dataset = datasets.ImageFolder(data_root / "val", transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.unsqueeze(1).float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.unsqueeze(1).float().to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)

            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader.dataset), accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pepper freshness classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to the data root containing train/ and val/ folders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(MODEL_PATH),
        help="Path where the trained PyTorch checkpoint will be saved.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = build_data_loaders(args.data_dir, args.batch_size)

    print("Classes:", class_names)
    print("Device:", device)
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}.")

    model = PepperClassifier(pretrained=True)
    model.to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}"
        )

    checkpoint_path = args.model_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
