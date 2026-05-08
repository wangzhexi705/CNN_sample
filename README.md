# STL-10 BasicCNN Image Classification

This project trains a compact convolutional neural network on the local STL-10
folder dataset.

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

## Model

`BasicCNN` is designed for STL-10 RGB images of size `96x96`.

- Four convolution blocks: `32 -> 64 -> 128 -> 256`
- Each block uses `Conv2d + BatchNorm + ReLU` twice, then `MaxPool2d`
- `Dropout2d` is used to reduce overfitting
- `AdaptiveAvgPool2d` keeps the classifier independent of exact feature map size
- Final classifier outputs 10 STL-10 classes

## Commands

```bash
python main.py --mode train --model basic
python main.py --mode eval --model basic
```

Useful quick test:

```bash
python main.py --mode train --model basic --epochs 1 --batch-size 32 --max-train-samples 256 --max-valid-samples 128
```

If your installed PyTorch does not support your GPU, force CPU:

```bash
python main.py --mode train --model basic --device cpu
```

Training saves:

```text
checkpoints/basic_cnn_best.pt
outputs/logs/basic_history.csv
```
