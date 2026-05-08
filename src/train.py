import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import CLASSES, TrainConfig
from .dataset import create_dataloaders
from .models import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(preferred: str = "auto"):
    preferred = preferred.lower()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda").fill_(1).cpu()
            return torch.device("cuda")
        except RuntimeError as exc:
            print(f"CUDA is visible but unusable, falling back to CPU: {exc}")
    return torch.device("cpu")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            for target, pred in zip(labels.cpu(), preds.cpu()):
                confusion[target.long(), pred.long()] += 1

    tp = confusion.diag().float()
    precision = tp / confusion.sum(dim=0).clamp(min=1).float()
    recall = tp / confusion.sum(dim=1).clamp(min=1).float()
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    return {
        "loss": total_loss / total_samples,
        "accuracy": tp.sum().item() / total_samples,
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "confusion_matrix": confusion,
    }


def save_history(history, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_accuracy", "valid_loss", "valid_accuracy"],
        )
        writer.writeheader()
        writer.writerows(history)


def serializable_config(config: TrainConfig):
    result = {}
    for key, value in config.__dict__.items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def train_model(config: TrainConfig, model_name: str = "basic"):
    set_seed(config.seed)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader, _ = create_dataloaders(config)
    device = get_device(config.device)
    model = build_model(model_name, config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_valid_acc = 0.0
    history = []

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}, valid batches: {len(valid_loader)}")

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, criterion, device, optimizer)
        valid_metrics = run_one_epoch(model, valid_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"valid_loss={row['valid_loss']:.4f} "
            f"valid_acc={row['valid_accuracy']:.4f}"
        )

        if valid_metrics["accuracy"] > best_valid_acc:
            best_valid_acc = valid_metrics["accuracy"]
            torch.save(
                {
                    "model_name": model_name,
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "config": serializable_config(config),
                    "best_valid_accuracy": best_valid_acc,
                    "epoch": epoch,
                },
                config.checkpoint_path,
            )

    save_history(history, config.output_dir / "logs" / f"{model_name}_history.csv")
    print(f"Best valid accuracy: {best_valid_acc:.4f}")
    print(f"Saved checkpoint: {config.checkpoint_path}")
    return model


def evaluate_checkpoint(config: TrainConfig, model_name: str = "basic"):
    _, _, test_loader = create_dataloaders(config)
    device = get_device(config.device)
    model = build_model(model_name, config.num_classes).to(device)

    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device, config.num_classes)

    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['precision']:.4f}")
    print(f"Macro recall: {metrics['recall']:.4f}")
    print(f"Macro F1: {metrics['f1']:.4f}")
    print("Confusion matrix rows=true labels, cols=pred labels:")
    print(metrics["confusion_matrix"].numpy())
    return metrics
