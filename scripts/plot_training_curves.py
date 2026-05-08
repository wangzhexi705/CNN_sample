import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "valid_loss": float(row["valid_loss"]),
                    "valid_accuracy": float(row["valid_accuracy"]),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot train/valid curves from history csv.")
    parser.add_argument("--model", default="basic", choices=["basic", "augmented"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    history_path = root / "outputs" / "logs" / f"{args.model}_history.csv"
    figure_dir = root / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = load_history(history_path)
    epochs = [row["epoch"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    valid_loss = [row["valid_loss"] for row in rows]
    train_acc = [row["train_accuracy"] for row in rows]
    valid_acc = [row["valid_accuracy"] for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)

    axes[0].plot(epochs, train_loss, marker="o", markersize=3, linewidth=1.8, label="Train Loss")
    axes[0].plot(epochs, valid_loss, marker="s", markersize=3, linewidth=1.8, label="Valid Loss")
    model_title = "AugmentedCNN" if args.model == "augmented" else "BasicCNN"

    axes[0].set_title(f"{model_title} Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", markersize=3, linewidth=1.8, label="Train Accuracy")
    axes[1].plot(epochs, valid_acc, marker="s", markersize=3, linewidth=1.8, label="Valid Accuracy")
    axes[1].set_title(f"{model_title} Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.15, 0.72)
    axes[1].legend()

    output_path = figure_dir / f"{args.model}_training_curves.png"
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")

    best_valid_acc = max(valid_acc)
    best_epoch = epochs[valid_acc.index(best_valid_acc)]
    print(f"Saved figure: {output_path}")
    print(f"Best valid accuracy: {best_valid_acc:.4f} at epoch {best_epoch}")
    print(f"Last train loss/accuracy: {train_loss[-1]:.4f}/{train_acc[-1]:.4f}")
    print(f"Last valid loss/accuracy: {valid_loss[-1]:.4f}/{valid_acc[-1]:.4f}")


if __name__ == "__main__":
    main()
