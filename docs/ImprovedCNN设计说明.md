# ImprovedCNN 模型结构改进实验说明

本文档说明 ImprovedCNN 阶段的设计目标、代码实现方式、运行命令以及当前已验证的实验入口。ImprovedCNN 主要针对模型结构本身进行改进和对比，包括激活函数、池化方式以及归一化与正则化方法。

## 1. 实验目标

在课程项目的前两个阶段中：

```text
BasicCNN:      作为基础 CNN baseline，使用固定结构完成 STL-10 分类。
AugmentedCNN:  保持网络结构不变，主要研究数据增强对泛化能力的影响。
ImprovedCNN:   在网络结构层面进行改进，研究不同结构选择对模型性能的影响。
```

ImprovedCNN 阶段重点围绕以下三个方向展开：

```text
1. 不同激活函数：ReLU、Sigmoid、Tanh
2. 不同池化方式：Max Pooling、Average Pooling
3. 引入正则化和归一化：Dropout、Batch Normalization
```

这样可以把“数据处理策略的影响”和“模型结构设计的影响”区分开来，便于在实验报告中分别分析。

## 2. ImprovedCNN 的设计思路

ImprovedCNN 沿用 BasicCNN 的整体四层卷积骨架，但将卷积块内部设计改为可配置形式，使同一套训练流程下可以切换不同结构选项。

当前实现支持以下可配置项：

### 2.1 激活函数

支持三种激活函数：

```text
ReLU
Sigmoid
Tanh
```

- `ReLU` 计算简单，收敛通常较快，是当前 CNN 中最常用的激活函数。
- `Sigmoid` 输出范围在 `0~1`，更容易出现梯度饱和，因此通常训练速度较慢，但适合作为对比实验。
- `Tanh` 输出范围在 `-1~1`，相比 Sigmoid 具有更中心化的输出，也适合作为经典激活函数对照组。

### 2.2 池化方式

支持两种池化方式：

```text
Max Pooling
Average Pooling
```

- `Max Pooling` 倾向于保留局部区域中最强的响应，更适合突出显著纹理和边缘特征。
- `Average Pooling` 倾向于保留区域平均信息，使特征图更平滑，可能对整体轮廓和背景变化更稳定。

### 2.3 归一化与正则化

ImprovedCNN 中包含两类常用结构技巧：

```text
Batch Normalization
Dropout / Dropout2d
```

- `Batch Normalization` 可以稳定中间层特征分布，加快训练并改善收敛稳定性。
- `Dropout2d` 用于卷积特征层，随机屏蔽部分通道特征，降低过拟合风险。
- `Dropout` 用于全连接分类头，减少分类器对单一特征的过度依赖。

当前实现中，`Batch Normalization` 支持开启或关闭；`Dropout` 默认保留，用于体现课程要求中的正则化方法。

## 3. 当前网络结构

ImprovedCNN 的整体结构仍然保持清晰、易解释的 CNN 形式：

```text
Input: 3 x 96 x 96

ConvBlock 1: 3   -> 32   96 x 96 -> 48 x 48
ConvBlock 2: 32  -> 64   48 x 48 -> 24 x 24
ConvBlock 3: 64  -> 128  24 x 24 -> 12 x 12
ConvBlock 4: 128 -> 256  12 x 12 -> 6 x 6

AdaptiveAvgPool2d: 256 x 6 x 6 -> 256 x 1 x 1
Flatten: 256
Linear: 256 -> 128
Activation
Dropout
Linear: 128 -> 10
```

其中每个 `ConfigurableConvBlock` 的内部形式为：

```text
Conv2d
[BatchNorm2d or Identity]
[Activation]
Conv2d
[BatchNorm2d or Identity]
[Activation]
[MaxPool2d or AvgPool2d]
Dropout2d
```

这意味着我们可以在不改训练主流程的前提下，直接比较不同结构组合的效果。

## 4. 代码实现方式

ImprovedCNN 相关实现已经接入到现有代码框架中。

### 4.1 模型定义

模型定义位于：

```text
src/models.py
```

新增的主要结构包括：

```python
class ConfigurableConvBlock(nn.Module):
    ...

class ImprovedCNN(nn.Module):
    ...
```

其中：

- `ConfigurableConvBlock` 负责根据命令行参数动态构造激活函数、池化方式和是否使用 BatchNorm。
- `ImprovedCNN` 基于该卷积块堆叠四层特征提取模块，并保留分类头中的 Dropout。

### 4.2 训练配置

训练配置位于：

```text
src/config.py
```

当前新增了以下配置项：

```text
activation      # relu / sigmoid / tanh
pooling         # max / avg
normalization   # batchnorm / none
```

同时为了避免不同 ImprovedCNN 实验相互覆盖，代码会自动构造区分配置的运行名，例如：

```text
improved_relu_avg_bn
improved_tanh_max_no_bn
```

### 4.3 命令行入口

命令行入口位于：

```text
main.py
```

当前已经支持：

```bash
--model improved
--activation relu|sigmoid|tanh
--pooling max|avg
--normalization batchnorm|none
```

### 4.4 训练曲线脚本

训练曲线脚本位于：

```text
scripts/plot_training_curves.py
```

当前脚本支持：

```bash
--model improved
--history-name improved_relu_avg_bn
```

用于绘制具体某组 ImprovedCNN 配置对应的训练曲线。

## 5. 运行命令示例

### 5.1 训练一个默认 ImprovedCNN 配置

```bash
python main.py --mode train --model improved --activation relu --pooling avg --normalization batchnorm --epochs 30 --batch-size 64
```

### 5.2 评估默认 ImprovedCNN 配置

```bash
python main.py --mode eval --model improved --activation relu --pooling avg --normalization batchnorm
```

### 5.3 比较不同激活函数

```bash
python main.py --mode train --model improved --activation relu    --pooling avg --normalization batchnorm --epochs 30 --batch-size 64
python main.py --mode train --model improved --activation sigmoid --pooling avg --normalization batchnorm --epochs 30 --batch-size 64
python main.py --mode train --model improved --activation tanh    --pooling avg --normalization batchnorm --epochs 30 --batch-size 64
```

### 5.4 比较不同池化方式

```bash
python main.py --mode train --model improved --activation relu --pooling max --normalization batchnorm --epochs 30 --batch-size 64
python main.py --mode train --model improved --activation relu --pooling avg --normalization batchnorm --epochs 30 --batch-size 64
```

### 5.5 比较是否使用 BatchNorm

```bash
python main.py --mode train --model improved --activation relu --pooling avg --normalization batchnorm --epochs 30 --batch-size 64
python main.py --mode train --model improved --activation relu --pooling avg --normalization none      --epochs 30 --batch-size 64
```

### 5.6 绘制某一组 ImprovedCNN 的训练曲线

```bash
python scripts/plot_training_curves.py --model improved --history-name improved_relu_avg_bn
```

## 6. 输出文件命名方式

ImprovedCNN 为不同结构组合单独保存 checkpoint、日志和训练曲线，便于做对比实验。

例如，对于配置：

```text
activation = relu
pooling = avg
normalization = batchnorm
```

会生成：

```text
checkpoints/improved_relu_avg_bn_cnn_best.pt
outputs/logs/improved_relu_avg_bn_history.csv
outputs/figures/improved_relu_avg_bn_training_curves.png
```

这种命名方式可以避免不同实验相互覆盖，便于后续整理表格和实验报告。

## 7. 当前已完成的接入验证

为了确认 ImprovedCNN 已经正确接入训练、评估和模型构造流程，当前已经完成以下验证：

```text
1. 使用小样本在 CPU 上完成了 1 轮 ImprovedCNN 训练
2. 使用保存的 checkpoint 完成了 eval 流程测试
3. 对所有激活函数 × 池化方式 × 归一化选项组合完成了前向传播检查
```

已验证通过的结构组合包括：

```text
relu/max/batchnorm
relu/max/none
relu/avg/batchnorm
relu/avg/none
sigmoid/max/batchnorm
sigmoid/max/none
sigmoid/avg/batchnorm
sigmoid/avg/none
tanh/max/batchnorm
tanh/max/none
tanh/avg/batchnorm
tanh/avg/none
```

这说明当前代码层面已经支持课程要求中的三类结构改进实验。

## 8. 当前验证阶段的说明

当前已经跑通的只是 **接入验证和 smoke test**，不是正式实验结果。

例如，使用以下命令进行了 1 轮小样本训练：

```bash
python main.py --mode train --model improved --activation relu --pooling avg --normalization batchnorm --epochs 1 --batch-size 8 --max-train-samples 32 --max-valid-samples 16 --device cpu
```

该次验证生成了：

```text
checkpoints/improved_relu_avg_bn_cnn_best.pt
outputs/logs/improved_relu_avg_bn_history.csv
```

由于训练轮数和样本数都非常小，因此这组结果**不能作为最终性能结论**，它的意义只是确认：

```text
1. 新模型能够成功构建
2. 训练流程能够正常执行
3. checkpoint 能够保存和加载
4. eval 流程能够正常运行
5. 不同结构选项不会导致维度错误或模型构造错误
```

## 9. 建议的正式实验方式

如果你要把 ImprovedCNN 作为课程报告的正式部分，建议至少设计以下三组对比实验：

### 9.1 激活函数对比实验

固定：

```text
pooling = avg
normalization = batchnorm
```

对比：

```text
ReLU vs Sigmoid vs Tanh
```

观察：

```text
训练收敛速度
验证集 accuracy
测试集 accuracy
是否出现梯度饱和导致训练困难
```

### 9.2 池化方式对比实验

固定：

```text
activation = relu
normalization = batchnorm
```

对比：

```text
Max Pooling vs Average Pooling
```

观察：

```text
分类精度差异
对动物类和交通工具类的影响是否不同
训练曲线是否更稳定
```

### 9.3 归一化与正则化实验

固定：

```text
activation = relu
pooling = avg
```

对比：

```text
BatchNorm vs No BatchNorm
```

观察：

```text
收敛速度
验证集 loss 波动情况
是否更容易过拟合
```

同时，由于 ImprovedCNN 默认保留 Dropout，可以在报告中说明：

```text
本实验在卷积块和分类头中均保留 Dropout，以满足正则化实验要求；同时通过控制是否启用 Batch Normalization，进一步比较归一化方法对训练稳定性和泛化能力的影响。
```

## 10. 报告中可使用的表述

可以在报告中这样描述：

```text
为了进一步研究模型结构设计对图像分类性能的影响，本文在 BasicCNN 的基础上实现了 ImprovedCNN。该模型保持总体卷积网络框架不变，但将卷积块设计为可配置结构，从而支持不同激活函数、不同池化方式以及归一化方法的对比实验。

具体而言，ImprovedCNN 支持 ReLU、Sigmoid 和 Tanh 三种激活函数，支持 Max Pooling 和 Average Pooling 两种池化方式，并支持是否启用 Batch Normalization。同时，模型在卷积层和分类头中保留 Dropout，以增强正则化效果。通过控制变量实验，可以更系统地分析不同结构设计对收敛速度、稳定性和最终分类精度的影响。
```

## 11. 当前 ImprovedCNN 实验结果汇总

目前已经完成了 `5` 组 ImprovedCNN 配置训练，每组均训练 `30` 轮。训练日志保存在：

```text
outputs/logs/improved_relu_avg_bn_history.csv
outputs/logs/improved_relu_avg_no_bn_history.csv
outputs/logs/improved_relu_max_bn_history.csv
outputs/logs/improved_sigmoid_avg_bn_history.csv
outputs/logs/improved_tanh_avg_bn_history.csv
```

对应 checkpoint 保存在：

```text
checkpoints/improved_relu_avg_bn_cnn_best.pt
checkpoints/improved_relu_avg_no_bn_cnn_best.pt
checkpoints/improved_relu_max_bn_cnn_best.pt
checkpoints/improved_sigmoid_avg_bn_cnn_best.pt
checkpoints/improved_tanh_avg_bn_cnn_best.pt
```

### 11.1 训练与验证结果

根据各配置的训练日志，可以得到以下关键结果：

| 配置 | 第 1 轮 train acc | 第 1 轮 valid acc | 第 30 轮 train acc | 第 30 轮 valid acc | 最佳 valid acc | 最低 valid loss |
| --- | --- | --- | --- | --- | --- | --- |
| ReLU + AvgPool + BatchNorm | 18.22% | 26.95% | 54.24% | 59.62% | 60.76%（第 28 轮） | 1.0871（第 29 轮） |
| ReLU + AvgPool + No BatchNorm | 16.37% | 19.71% | 65.78% | 65.05% | 65.33%（第 26 轮） | 0.9724（第 30 轮） |
| ReLU + MaxPool + BatchNorm | 15.65% | 22.76% | 50.94% | 54.76% | 54.95%（第 28 轮） | 1.1582（第 29 轮） |
| Sigmoid + AvgPool + BatchNorm | 15.24% | 18.76% | 30.81% | 32.19% | 32.38%（第 23 轮） | 1.6787（第 28 轮） |
| Tanh + AvgPool + BatchNorm | 20.59% | 24.29% | 61.16% | 60.10% | 60.10%（第 30 轮） | 1.1408（第 26 轮） |

从训练与验证曲线数据可以观察到：

1. **ReLU + AvgPool + No BatchNorm 的验证集表现最好。** 其最佳验证准确率达到 `65.33%`，第 `30` 轮验证准确率也保持在 `65.05%`，说明该配置在当前训练设置下收敛更充分。
2. **Sigmoid 配置明显训练不足。** 第 `30` 轮训练准确率只有 `30.81%`，验证准确率只有 `32.19%`，远低于其他配置。这符合 Sigmoid 容易梯度饱和、深层 CNN 中收敛较慢的特点。
3. **Tanh 表现明显优于 Sigmoid，但仍低于最佳 ReLU 配置。** Tanh 配置最终验证准确率达到 `60.10%`，说明其具备一定非线性表达能力，但在当前任务上仍不如 ReLU 稳定。
4. **MaxPool 在当前 ImprovedCNN 中不如 AvgPool。** 在同样使用 ReLU 和 BatchNorm 的条件下，AvgPool 的最佳验证准确率为 `60.76%`，MaxPool 只有 `54.95%`。
5. **BatchNorm 在当前配置下没有带来提升。** ReLU + AvgPool 条件下，关闭 BatchNorm 的最佳验证准确率为 `65.33%`，高于启用 BatchNorm 的 `60.76%`。这说明 BatchNorm 的效果会受到初始化方式、Dropout 强度、学习率和训练轮数共同影响，并不一定在所有设置下都更优。

### 11.2 测试集结果

使用各配置对应的最佳 checkpoint 在 `STL10/test` 上进行评估，结果如下：

| 配置 | Test loss | Test accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| ReLU + AvgPool + BatchNorm | 1.1104 | 56.80% | 57.02% | 56.80% | 56.75% |
| ReLU + AvgPool + No BatchNorm | 0.9947 | 62.00% | 62.30% | 62.00% | 61.89% |
| ReLU + MaxPool + BatchNorm | 1.1635 | 54.60% | 54.90% | 54.60% | 54.30% |
| Sigmoid + AvgPool + BatchNorm | 1.6662 | 33.30% | 29.01% | 33.30% | 29.60% |
| Tanh + AvgPool + BatchNorm | 1.1673 | 58.20% | 59.46% | 58.20% | 58.20% |

测试集结果与验证集趋势基本一致：

```text
最佳测试配置: ReLU + AvgPool + No BatchNorm
Test accuracy = 62.00%
Macro F1 = 61.89%
```

该配置相比其他 ImprovedCNN 配置具有更低的测试损失和更高的综合分类指标，说明它在当前训练设置下泛化能力最好。

## 12. 三类结构改进实验分析

### 12.1 激活函数对比：ReLU vs Sigmoid vs Tanh

为了比较激活函数的影响，固定：

```text
pooling = avg
normalization = batchnorm
```

对比结果如下：

| 激活函数 | 最佳 valid accuracy | Test accuracy | Macro F1 |
| --- | --- | --- | --- |
| ReLU | 60.76% | 56.80% | 56.75% |
| Sigmoid | 32.38% | 33.30% | 29.60% |
| Tanh | 60.10% | 58.20% | 58.20% |

从结果看，`Sigmoid` 明显不适合当前较深的 CNN 结构。它的训练准确率和验证准确率都提升缓慢，最终测试准确率只有 `33.30%`，说明模型没有充分学习到有效特征。

`ReLU` 和 `Tanh` 的验证集准确率接近，其中 ReLU 的最佳验证准确率略高，Tanh 的测试集准确率略高。整体来看，ReLU 仍然是更常用、更稳定的 CNN 激活函数；Tanh 可以作为有效对照组，但 Sigmoid 在本实验中表现较差。

### 12.2 池化方式对比：Average Pooling vs Max Pooling

为了比较池化方式的影响，固定：

```text
activation = relu
normalization = batchnorm
```

对比结果如下：

| 池化方式 | 最佳 valid accuracy | Test accuracy | Macro F1 |
| --- | --- | --- | --- |
| Average Pooling | 60.76% | 56.80% | 56.75% |
| Max Pooling | 54.95% | 54.60% | 54.30% |

在当前 ImprovedCNN 设置下，`Average Pooling` 整体优于 `Max Pooling`。可能原因是 STL-10 图像分辨率较高、背景和物体姿态变化较大，Average Pooling 能保留更平滑的区域统计信息，而 Max Pooling 只保留局部最大响应，可能丢失了一部分整体轮廓信息。

不过，这一结论只针对当前网络结构和训练参数成立。如果后续调整 Dropout、学习率或训练轮数，Max Pooling 的表现仍可能发生变化。

### 12.3 BatchNorm 对比：BatchNorm vs No BatchNorm

为了比较归一化方法的影响，固定：

```text
activation = relu
pooling = avg
```

对比结果如下：

| 归一化设置 | 最佳 valid accuracy | Test accuracy | Macro F1 |
| --- | --- | --- | --- |
| BatchNorm | 60.76% | 56.80% | 56.75% |
| No BatchNorm | 65.33% | 62.00% | 61.89% |

从当前结果看，关闭 BatchNorm 的配置反而取得了更好的验证集和测试集结果。该现象说明 BatchNorm 并不是在所有设置下都必然提升性能。对于当前 ImprovedCNN，模型已经使用了 Dropout2d、Dropout、AdamW 和学习率调度器，BatchNorm 与这些训练策略组合后可能没有带来额外收益。

因此，在报告中可以将这一结果作为实验发现：**归一化方法的作用需要结合具体结构和训练超参数分析，而不是简单认为加入 BatchNorm 一定更好。**

## 13. 最佳 ImprovedCNN 配置分析

当前 5 组实验中，综合验证集和测试集表现最好的配置为：

```text
activation = relu
pooling = avg
normalization = none
```

对应文件为：

```text
outputs/logs/improved_relu_avg_no_bn_history.csv
checkpoints/improved_relu_avg_no_bn_cnn_best.pt
```

该配置的关键结果为：

```text
第 1 轮:
train_loss = 2.1830, train_accuracy = 0.1637
valid_loss = 1.9671, valid_accuracy = 0.1971

第 30 轮:
train_loss = 0.9252, train_accuracy = 0.6578
valid_loss = 0.9724, valid_accuracy = 0.6505

验证集最高准确率:
valid_accuracy = 0.6533, epoch = 26

测试集:
Test loss = 0.9947
Test accuracy = 0.6200
Macro precision = 0.6230
Macro recall = 0.6200
Macro F1 = 0.6189
```

该配置的混淆矩阵如下，行表示真实类别，列表示预测类别：

```text
[[71  7  2  3  0  0  0  0 11  6]
 [ 4 54  1 15  1  9  1 12  1  2]
 [ 1  1 76  4  0  0  1  0  3 14]
 [ 0 12  2 55  6  7  1 17  0  0]
 [ 0  3  2 11 68  3  4  9  0  0]
 [ 3  7  0  8 14 29 14 25  0  0]
 [ 0  1  0  2 11 17 62  5  0  2]
 [ 0  3  0 13 10 11  1 62  0  0]
 [10  5  4  1  0  0  0  0 77  3]
 [ 1  0 19  1  0  1  3  0  9 66]]
```

从混淆矩阵看，该配置对部分交通工具类别识别较好，例如：

```text
car:   76 / 100
ship:  77 / 100
truck: 66 / 100
airplane: 71 / 100
```

动物类别仍然是主要难点，例如：

```text
dog 正确识别: 29 / 100
dog 被预测为 monkey: 25 次
dog 被预测为 horse: 14 次
cat 被预测为 monkey: 17 次
horse 被预测为 dog: 17 次
```

这与 BasicCNN 和 AugmentedCNN 中观察到的现象一致：模型能够较好地区分外观差异明显的交通工具类，但对于猫、狗、猴子、马等外观和姿态相近的动物类别，仍然存在较多混淆。

## 14. 与 BasicCNN 和 AugmentedCNN 的对比

结合当前已有实验结果，可以得到以下对比：

| 模型/配置 | 最佳 valid accuracy | Test accuracy | Macro F1 |
| --- | --- | --- | --- |
| BasicCNN | 67.33% | 65.80% | 65.58% |
| AugmentedCNN | 66.48% | 64.10% | 63.89% |
| ImprovedCNN 最佳配置（ReLU + AvgPool + No BatchNorm） | 65.33% | 62.00% | 61.89% |

从结果看，当前 ImprovedCNN 的最佳配置仍低于 BasicCNN 和 AugmentedCNN。原因可能包括：

1. ImprovedCNN 当前主要用于结构对比实验，而不是专门调参后的最优模型。
2. Average Pooling、No BatchNorm 和更高 Dropout 组合虽然在 ImprovedCNN 内部最优，但相对 BasicCNN 的原始结构仍可能损失部分局部强响应特征。
3. ImprovedCNN 的不同配置之间差异明显，说明结构选择对性能影响很大，但当前还没有进一步针对最佳配置进行学习率、Dropout 和训练轮数优化。

因此，ImprovedCNN 阶段的主要价值在于完成模型结构改进与消融分析，而不是替代 BasicCNN 成为最终最佳模型。

## 15. 当前限制说明

当前代码已经支持 ImprovedCNN 的训练与评估，但训练曲线绘图脚本依赖 `matplotlib`。如果当前环境未安装该库，则会在绘图时出现：

```text
ModuleNotFoundError: No module named 'matplotlib'
```

这不会影响模型训练和评估本身，但会影响训练曲线图片的自动生成。如果后续需要实际绘图，需要先安装 `matplotlib`。
