# AugmentedCNN 数据增强实验说明

本文档说明 AugmentedCNN 的设计目的、代码实现方式和实验运行方法。AugmentedCNN 用于研究数据增强对 STL-10 图像分类任务的影响。

## 1. 实验目的

BasicCNN 关注基础卷积神经网络结构本身，而 AugmentedCNN 关注训练数据处理策略。两者使用相同的 CNN 网络骨架，主要区别在于：

```text
BasicCNN:
只使用 Resize、ToTensor、Normalize 等基础预处理。

AugmentedCNN:
在训练阶段额外加入 RandomCrop、RandomHorizontalFlip、ColorJitter 等数据增强。
```

这样设计可以让实验对比更加清晰：如果 AugmentedCNN 的验证集或测试集表现优于 BasicCNN，则可以说明数据增强提升了模型泛化能力。

## 2. 数据增强策略

AugmentedCNN 在训练集上使用如下 transform：

```text
RandomCrop(96, padding=8)
RandomHorizontalFlip()
ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03)
ToTensor()
Normalize(mean, std)
```

验证集和测试集不使用随机增强，只使用：

```text
Resize((96, 96))
ToTensor()
Normalize(mean, std)
```

这样可以保证验证和测试结果稳定，同时避免把随机增强带入评估过程。

## 3. 各数据增强方法的作用

### 3.1 RandomCrop

`RandomCrop(96, padding=8)` 会先在图像边缘补充像素，再随机裁剪回 `96 x 96`。它可以模拟目标在图像中位置发生轻微偏移的情况，使模型不依赖物体固定出现在图像中央。

### 3.2 RandomHorizontalFlip

`RandomHorizontalFlip` 会以一定概率水平翻转图像。STL-10 中大部分物体类别，如汽车、船、动物等，左右翻转后类别不变，因此该增强方式可以有效扩充训练样本的姿态变化。

### 3.3 ColorJitter

`ColorJitter` 会对亮度、对比度、饱和度和色调进行轻微扰动。它可以模拟不同光照、拍摄条件和颜色变化，减少模型对固定颜色分布的依赖。

## 4. 代码实现方式

代码中通过 `TrainConfig.use_augmentation` 控制是否启用训练增强：

```python
use_augmentation=args.model == "augmented"
```

当命令行参数为 `--model augmented` 时，训练集会启用增强策略；当命令行参数为 `--model basic` 时，只使用基础预处理。

模型注册在 `src/models.py` 中：

```python
class AugmentedCNN(BasicCNN):
    """BasicCNN trained with stronger data augmentation."""
```

这里 AugmentedCNN 继承 BasicCNN，表示它的网络结构与 BasicCNN 相同。这样做的原因是：本实验希望控制变量，只观察数据增强对模型性能的影响。

## 5. 运行命令

训练 AugmentedCNN：

```bash
python main.py --mode train --model augmented --epochs 30 --batch-size 64
```

评估 AugmentedCNN：

```bash
python main.py --mode eval --model augmented
```

绘制训练曲线：

```bash
python scripts/plot_training_curves.py --model augmented
```

训练完成后会生成：

```text
checkpoints/augmented_cnn_best.pt
outputs/logs/augmented_history.csv
outputs/figures/augmented_training_curves.png
```

## 6. 与 BasicCNN 的对比方式

建议在相同训练设置下分别运行：

```bash
python main.py --mode train --model basic --epochs 30 --batch-size 64
python main.py --mode train --model augmented --epochs 30 --batch-size 64
```

然后分别评估：

```bash
python main.py --mode eval --model basic
python main.py --mode eval --model augmented
```

对比时重点关注：

```text
1. 验证集 accuracy 是否提升
2. 测试集 accuracy 是否提升
3. 训练集和验证集之间的差距是否缩小
4. 验证 loss 是否更加稳定
5. 混淆矩阵中容易混淆的类别是否有所改善
```

如果 AugmentedCNN 的训练准确率低于 BasicCNN，但验证或测试准确率更高，这是正常且积极的现象。因为数据增强会让训练样本更难，短期内可能降低训练集准确率，但通常能提升模型对未见样本的泛化能力。

## 7. 报告中可使用的表述

可以在报告中这样描述：

```text
为了提升模型的泛化能力，本文在 BasicCNN 的基础上设计了 AugmentedCNN。该模型保持网络结构不变，仅在训练阶段加入 RandomCrop、RandomHorizontalFlip 和 ColorJitter 等数据增强方法。通过控制网络结构一致，可以更直接地分析数据增强对分类性能的影响。

RandomCrop 用于模拟目标位置偏移，RandomHorizontalFlip 用于扩充左右方向的姿态变化，ColorJitter 用于增强模型对光照和颜色变化的鲁棒性。验证集和测试集不使用随机增强，以保证评估结果的稳定性。
```
