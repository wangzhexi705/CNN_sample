# STL-10 CNN Image Classification

This project trains convolutional neural networks on the local STL-10 folder dataset.
Current experiments include `BasicCNN`, `AugmentedCNN`, and `ImprovedCNN`.

## Dataset

Expected structure:

```text
STL10/
|-- train/
|   |-- airplane/
|   |-- bird/
|   `-- ...
`-- test/
    |-- airplane/
    |-- bird/
    `-- ...
```

## Models

### BasicCNN

`BasicCNN` is the baseline model for STL-10 RGB images of size `96x96`.

- Four convolution blocks: `32 -> 64 -> 128 -> 256`
- Each block uses `Conv2d + BatchNorm + ReLU` twice, then `MaxPool2d`
- `Dropout2d` is used to reduce overfitting
- `AdaptiveAvgPool2d` keeps the classifier independent of exact feature map size
- Final classifier outputs 10 STL-10 classes

### AugmentedCNN

`AugmentedCNN` keeps the same CNN backbone as `BasicCNN`, but enables stronger data augmentation during training.

- Training transform adds `RandomCrop`, `RandomHorizontalFlip`, and `ColorJitter`
- Validation and test still use deterministic preprocessing
- Useful for studying whether stronger augmentation improves generalization

### ImprovedCNN

`ImprovedCNN` focuses on model-structure experiments instead of only changing the input pipeline.
It supports configurable activation, pooling, and normalization settings.

Available options:

- Activation: `relu`, `sigmoid`, `tanh`
- Pooling: `max`, `avg`
- Normalization: `batchnorm`, `none`
- Regularization: convolution blocks use `Dropout2d`, classifier uses `Dropout`

This makes it suitable for the course requirement of comparing:

- different activation functions
- different pooling methods
- normalization and regularization strategies

Detailed design notes are in `docs/ImprovedCNN设计说明.md`.

## Commands

### BasicCNN / AugmentedCNN

```bash
python main.py --mode train --model basic
python main.py --mode eval --model basic
python main.py --mode train --model augmented
python main.py --mode eval --model augmented
```

### ImprovedCNN

Train a default ImprovedCNN configuration:

```bash
python main.py --mode train --model improved --activation relu --pooling avg --normalization batchnorm
```

Evaluate the same configuration:

```bash
python main.py --mode eval --model improved --activation relu --pooling avg --normalization batchnorm
```

Example comparison runs:

```bash
python main.py --mode train --model improved --activation relu --pooling avg --normalization batchnorm
python main.py --mode train --model improved --activation sigmoid --pooling avg --normalization batchnorm
python main.py --mode train --model improved --activation tanh --pooling avg --normalization batchnorm

python main.py --mode train --model improved --activation relu --pooling max --normalization batchnorm
python main.py --mode train --model improved --activation relu --pooling avg --normalization batchnorm

python main.py --mode train --model improved --activation relu --pooling avg --normalization none
```

### Hyperparameter and Optimizer Experiments

Use `--run-name` to save learning-rate or optimizer experiments without overwriting the default `basic_history.csv` and `basic_cnn_best.pt` files.

Learning-rate examples:

```bash
python main.py --mode train --model basic --optimizer adamw --lr 0.0001 --run-name basic_adamw_lr1e-4
python main.py --mode train --model basic --optimizer adamw --lr 0.0005 --run-name basic_adamw_lr5e-4
python main.py --mode train --model basic --optimizer adamw --lr 0.001  --run-name basic_adamw_lr1e-3
python main.py --mode train --model basic --optimizer adamw --lr 0.003  --run-name basic_adamw_lr3e-3
```

Optimizer comparison examples:

```bash
python main.py --mode train --model basic --optimizer adamw --lr 0.001 --weight-decay 0.0001 --run-name basic_adamw_lr1e-3
python main.py --mode train --model basic --optimizer adam  --lr 0.001 --weight-decay 0.0001 --run-name basic_adam_lr1e-3
python main.py --mode train --model basic --optimizer sgd   --lr 0.01  --momentum 0.9 --weight-decay 0.0005 --run-name basic_sgd_lr1e-2
```

Evaluate a specific experiment by passing the same `--run-name`:

```bash
python main.py --mode eval --model basic --run-name basic_adamw_lr1e-3
```

A run named `basic_adamw_lr1e-3` saves:

```text
checkpoints/basic_adamw_lr1e-3_cnn_best.pt
outputs/logs/basic_adamw_lr1e-3_history.csv
```

Useful quick test:

```bash
python main.py --mode train --model basic --epochs 1 --batch-size 32 --max-train-samples 256 --max-valid-samples 128
python main.py --mode train --model improved --activation relu --pooling avg --normalization batchnorm --epochs 1 --batch-size 8 --max-train-samples 32 --max-valid-samples 16 --device cpu
```

If your installed PyTorch does not support your GPU, force CPU:

```bash
python main.py --mode train --model basic --device cpu
python main.py --mode train --model improved --activation relu --pooling avg --normalization batchnorm --device cpu
```

## Training Outputs

Training saves:

```text
checkpoints/basic_cnn_best.pt
checkpoints/augmented_cnn_best.pt
outputs/logs/basic_history.csv
outputs/logs/augmented_history.csv
```

ImprovedCNN uses configuration-specific names so different experiments do not overwrite each other.
For example, the configuration

```text
activation = relu
pooling = avg
normalization = batchnorm
```

saves:

```text
checkpoints/improved_relu_avg_bn_cnn_best.pt
outputs/logs/improved_relu_avg_bn_history.csv
outputs/figures/improved_relu_avg_bn_training_curves.png
```

## Grad-CAM Visualization

Grad-CAM can be used to visualize which image regions contribute most to a model prediction.
Detailed notes are in `docs/GradCAM可视化说明.md`.

Generate Grad-CAM for a BasicCNN test sample:

```bash
python main.py --mode gradcam --model basic --split test --sample-index 0 --device cpu
```

Generate Grad-CAM for AugmentedCNN:

```bash
python main.py --mode gradcam --model augmented --split test --sample-index 0 --device cpu
```

Generate Grad-CAM for an ImprovedCNN configuration:

```bash
python main.py --mode gradcam --model improved --activation relu --pooling avg --normalization none --split test --sample-index 0 --device cpu
```

You can also select a sample within a specific class:

```bash
python main.py --mode gradcam --model basic --split test --class-name airplane --sample-index 1 --device cpu
```

Or target a specific class instead of the predicted class:

```bash
python main.py --mode gradcam --model basic --split test --sample-index 0 --target-class airplane --device cpu
```

Grad-CAM saves artifacts under:

```text
outputs/gradcam/<run_name>/
```

Each run writes:

```text
*_original.png
*_heatmap.png
*_overlay.png
*_panel.png
*_meta.json
```

## Plot Curves

```bash
python scripts/plot_training_curves.py --model basic
python scripts/plot_training_curves.py --model augmented
python scripts/plot_training_curves.py --model improved --history-name improved_relu_avg_bn
```

If curve plotting fails with `ModuleNotFoundError: No module named 'matplotlib'`, install `matplotlib` first.
