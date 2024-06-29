# ResNet50-VS-ViT-on-CIFAR-100

复旦大学 DATA620004 神经网络和深度学习 期末作业

## 任务2：在 CIFAR-100 数据集上比较基于 Transformer 和 CNN 的图像分类模型

### 基本要求：
1. **分别基于 CNN 和 Transformer 架构实现具有相近参数量的图像分类网络**。
2. **在 CIFAR-100 数据集上采用相同的训练策略对二者进行训练**，其中数据增强策略中应包含 CutMix。
3. **尝试不同的超参数组合**，尽可能提升各架构在 CIFAR-100 上的性能以进行合理的比较。

### 准备
请确保安装以下依赖：
- torch
- torchvision
- tensorboard
- tqdm

### 训练
1. **训练 ResNet50**
   ```sh
   python train_resnet.py
   ```
   根据自己的需求调整参数搜索即可。

2. **训练 Vision Transformer**
   ```sh
   python train_vit.py
   ```
   同样根据自己的需求调整参数搜索即可。

### 测试
运行以下脚本对模型进行测试：
```sh
python testor.py
```

### 附
CutMix 数据增强在 `CutMix.py` 中实现。
