"""
Trains a model (custom CNN for now; pretrained models plug in later
via the same loop) and saves the best checkpoint by validation accuracy.

Usage:
    python train.py --model custom_cnn
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_dataloaders
from models.custom_cnn import CustomCNN


def build_model(model_name: str) -> nn.Module:
    if model_name == "custom_cnn":
        return CustomCNN(num_classes=config.NUM_CLASSES)
    # Placeholder hooks for the next phase of the project:
    # elif model_name == "mobilenet_v2": ...
    # elif model_name == "resnet50": ...
    raise ValueError(f"Unknown model name: {model_name}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    running_loss, running_correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)

    torch.set_grad_enabled(True)
    return running_loss / total, running_correct / total


def train_model(model_name: str = "custom_cnn"):
    torch.manual_seed(config.RANDOM_SEED)

    device = config.DEVICE
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders()

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:2d}/{config.EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{config.EARLY_STOPPING_PATIENCE} epochs).")
                break

    # Save best checkpoint + training history
    ckpt_path = Path(config.CHECKPOINT_DIR) / f"{model_name}_best.pt"
    torch.save(best_state, ckpt_path)
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Saved best checkpoint to: {ckpt_path}")

    history_path = Path(config.OUTPUT_DIR) / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")

    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="custom_cnn",
        choices=["custom_cnn"],  # more choices added in the transfer-learning phase
        help="Which model to train",
    )
    args = parser.parse_args()
    train_model(args.model)
