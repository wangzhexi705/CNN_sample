import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import CLASSES, GRADCAM_OUTPUT_DIR, TrainConfig, build_run_name
from .dataset import STL10FolderDataset, build_transforms
from .models import build_model
from .train import get_device


MEAN = torch.tensor((0.4467, 0.4398, 0.4066), dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor((0.2603, 0.2566, 0.2713), dtype=torch.float32).view(3, 1, 1)


def resolve_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    return model.features[3].block[3]


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradients):
        self.gradients = gradients.detach()

    def remove(self):
        self._forward_handle.remove()

    def __call__(self, image_tensor: torch.Tensor, target_index: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits[:, target_index].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.squeeze(0).squeeze(0).cpu()


def tensor_to_rgb_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu()
    image = image * STD + MEAN
    image = image.clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def apply_colormap(cam: np.ndarray) -> np.ndarray:
    cam = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 * cam - 0.5, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * cam - 1.0) * 1.5, 0.0, 1.0)
    blue = np.clip(1.0 - 1.5 * cam, 0.0, 1.0)
    heatmap = np.stack([red, green, blue], axis=-1)
    return (heatmap * 255).astype(np.uint8)


def blend_overlay(original: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    blended = (1.0 - alpha) * original.astype(np.float32) + alpha * heatmap.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_panel(original: np.ndarray, heatmap: np.ndarray, overlay: np.ndarray) -> Image.Image:
    original_img = Image.fromarray(original)
    heatmap_img = Image.fromarray(heatmap)
    overlay_img = Image.fromarray(overlay)
    width, height = original_img.size
    panel = Image.new("RGB", (width * 3, height), color=(255, 255, 255))
    panel.paste(original_img, (0, 0))
    panel.paste(heatmap_img, (width, 0))
    panel.paste(overlay_img, (width * 2, 0))
    return panel


def _load_from_dataset(
    config: TrainConfig,
    split: str,
    sample_index: int,
    class_name: Optional[str],
):
    dataset = STL10FolderDataset(
        config.data_dir / split,
        transform=build_transforms(config, train=False),
    )
    if class_name is not None:
        filtered = [sample for sample in dataset.samples if sample[0].parent.name == class_name]
        if not filtered:
            raise ValueError(f"No samples found for class '{class_name}' in split '{split}'")
        if sample_index < 0 or sample_index >= len(filtered):
            raise IndexError(f"sample_index {sample_index} out of range for class '{class_name}'")
        image_path, label = filtered[sample_index]
        image = Image.open(image_path).convert("RGB")
        tensor = dataset.transform(image)
        return image_path, tensor, label

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} out of range for split '{split}'")
    tensor, label = dataset[sample_index]
    image_path, _ = dataset.samples[sample_index]
    return image_path, tensor, label


def _load_from_path(config: TrainConfig, image_path: Path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    tensor = build_transforms(config, train=False)(image)
    return image_path, tensor, None


def _resolve_target_class(target_class: Optional[str], predicted_index: int) -> int:
    if target_class is None:
        return predicted_index
    if target_class.isdigit():
        index = int(target_class)
        if index < 0 or index >= len(CLASSES):
            raise ValueError(f"target class index out of range: {index}")
        return index
    if target_class not in CLASSES:
        raise ValueError(f"Unknown target class: {target_class}")
    return CLASSES.index(target_class)


def _build_file_stem(split: Optional[str], sample_index: int, true_label: Optional[int], image_path: Path, output_tag: Optional[str]) -> str:
    if split is not None:
        class_part = CLASSES[true_label] if true_label is not None else image_path.parent.name
        stem = f"{split}_{sample_index:03d}_{class_part}"
    else:
        stem = image_path.stem
    if output_tag:
        stem = f"{stem}_{output_tag}"
    return stem


def generate_gradcam(
    config: TrainConfig,
    model_name: str,
    image_path: Optional[str] = None,
    split: str = "test",
    sample_index: int = 0,
    class_name: Optional[str] = None,
    target_class: Optional[str] = None,
    alpha: float = 0.45,
    output_tag: Optional[str] = None,
):
    device = get_device(config.device)
    run_name = build_run_name(
        model_name,
        activation=config.activation,
        pooling=config.pooling,
        normalization=config.normalization,
        run_name=config.run_name,
    )
    model = build_model(
        model_name,
        config.num_classes,
        activation=config.activation,
        pooling=config.pooling,
        normalization=config.normalization,
    ).to(device)

    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if image_path is not None:
        source_path, input_tensor, true_label = _load_from_path(config, image_path)
        split_name = None
    else:
        source_path, input_tensor, true_label = _load_from_dataset(config, split, sample_index, class_name)
        split_name = split

    image_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(probabilities.argmax(dim=1).item())
        predicted_confidence = float(probabilities[0, predicted_index].item())

    target_index = _resolve_target_class(target_class, predicted_index)

    gradcam = GradCAM(model, resolve_target_layer(model))
    try:
        cam = gradcam(image_batch, target_index)
    finally:
        gradcam.remove()

    original = tensor_to_rgb_image(input_tensor)
    heatmap = apply_colormap(cam.numpy())
    overlay = blend_overlay(original, heatmap, alpha=alpha)
    panel = build_panel(original, heatmap, overlay)

    output_dir = GRADCAM_OUTPUT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = _build_file_stem(split_name, sample_index, true_label, source_path, output_tag)

    original_path = output_dir / f"{file_stem}_original.png"
    heatmap_path = output_dir / f"{file_stem}_heatmap.png"
    overlay_path = output_dir / f"{file_stem}_overlay.png"
    panel_path = output_dir / f"{file_stem}_panel.png"
    meta_path = output_dir / f"{file_stem}_meta.json"

    Image.fromarray(original).save(original_path)
    Image.fromarray(heatmap).save(heatmap_path)
    Image.fromarray(overlay).save(overlay_path)
    panel.save(panel_path)

    metadata = {
        "model_name": model_name,
        "run_name": run_name,
        "checkpoint_path": str(config.checkpoint_path),
        "image_path": str(source_path),
        "split": split_name,
        "sample_index": sample_index if split_name is not None else None,
        "class_filter": class_name,
        "true_label": CLASSES[true_label] if true_label is not None else None,
        "predicted_label": CLASSES[predicted_index],
        "predicted_confidence": predicted_confidence,
        "target_label": CLASSES[target_index],
        "hook_layer": "features[3].block[3]",
        "artifacts": {
            "original": str(original_path),
            "heatmap": str(heatmap_path),
            "overlay": str(overlay_path),
            "panel": str(panel_path),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved Grad-CAM panel: {panel_path}")
    print(f"Predicted label: {CLASSES[predicted_index]} ({predicted_confidence:.4f})")
    print(f"Target label: {CLASSES[target_index]}")
    print(f"Source image: {source_path}")

    return metadata
