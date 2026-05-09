import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConfigurableConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        pooling: str,
        use_batchnorm: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            self._make_normalization(out_channels, use_batchnorm),
            self._make_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            self._make_normalization(out_channels, use_batchnorm),
            self._make_activation(activation),
            self._make_pooling(pooling),
            nn.Dropout2d(dropout),
        )

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation: {name}")

    @staticmethod
    def _make_pooling(name: str) -> nn.Module:
        name = name.lower()
        if name == "max":
            return nn.MaxPool2d(kernel_size=2)
        if name == "avg":
            return nn.AvgPool2d(kernel_size=2)
        raise ValueError(f"Unsupported pooling: {name}")

    @staticmethod
    def _make_normalization(out_channels: int, use_batchnorm: bool) -> nn.Module:
        if use_batchnorm:
            return nn.BatchNorm2d(out_channels)
        return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BasicCNN(nn.Module):
    """A compact CNN designed for STL-10 96x96 RGB classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.05),    # 96 -> 48
            ConvBlock(32, 64, dropout=0.10),   # 48 -> 24
            ConvBlock(64, 128, dropout=0.15),  # 24 -> 12
            ConvBlock(128, 256, dropout=0.20), # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)


class AugmentedCNN(BasicCNN):
    """BasicCNN trained with stronger data augmentation."""


class ImprovedCNN(nn.Module):
    """CNN variant for activation, pooling, normalization, and dropout experiments."""

    def __init__(
        self,
        num_classes: int = 10,
        activation: str = "relu",
        pooling: str = "avg",
        normalization: str = "batchnorm",
    ):
        super().__init__()
        self.activation = activation.lower()
        normalization = normalization.lower()
        if normalization not in {"batchnorm", "none"}:
            raise ValueError(f"Unsupported normalization: {normalization}")
        use_batchnorm = normalization == "batchnorm"
        self.features = nn.Sequential(
            ConfigurableConvBlock(3, 32, activation, pooling, use_batchnorm, dropout=0.08),
            ConfigurableConvBlock(32, 64, activation, pooling, use_batchnorm, dropout=0.12),
            ConfigurableConvBlock(64, 128, activation, pooling, use_batchnorm, dropout=0.18),
            ConfigurableConvBlock(128, 256, activation, pooling, use_batchnorm, dropout=0.24),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            ConfigurableConvBlock._make_activation(activation),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if self.activation == "relu":
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if self.activation == "relu":
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)


def build_model(
    model_name: str,
    num_classes: int,
    activation: str = "relu",
    pooling: str = "avg",
    normalization: str = "batchnorm",
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "basic":
        return BasicCNN(num_classes=num_classes)
    if model_name == "augmented":
        return AugmentedCNN(num_classes=num_classes)
    if model_name == "improved":
        return ImprovedCNN(
            num_classes=num_classes,
            activation=activation,
            pooling=pooling,
            normalization=normalization,
        )
    raise ValueError(f"Unknown model: {model_name}")
