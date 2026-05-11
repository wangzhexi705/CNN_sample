from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "STL10"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "outputs"
GRADCAM_OUTPUT_DIR = OUTPUT_DIR / "gradcam"

CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def build_run_name(
    model_name: str,
    activation: str = "relu",
    pooling: str = "avg",
    normalization: str = "batchnorm",
    run_name: Optional[str] = None,
) -> str:
    if run_name is not None:
        return run_name
    model_name = model_name.lower()
    if model_name != "improved":
        return model_name
    norm_tag = "bn" if normalization == "batchnorm" else "no_bn"
    return f"improved_{activation}_{pooling}_{norm_tag}"


@dataclass
class TrainConfig:
    data_dir: Path = DATA_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    output_dir: Path = OUTPUT_DIR
    image_size: int = 96
    num_classes: int = len(CLASSES)
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adamw"
    run_name: Optional[str] = None
    val_ratio: float = 0.15
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    use_augmentation: bool = False
    activation: str = "relu"
    pooling: str = "avg"
    normalization: str = "batchnorm"
    max_train_samples: Optional[int] = None
    max_valid_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    checkpoint_name: str = "basic_cnn_best.pt"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name
