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

## Plot Curves

```bash
python scripts/plot_training_curves.py --model basic
python scripts/plot_training_curves.py --model augmented
python scripts/plot_training_curves.py --model improved --history-name improved_relu_avg_bn
```

If curve plotting fails with `ModuleNotFoundError: No module named 'matplotlib'`, install `matplotlib` first.
