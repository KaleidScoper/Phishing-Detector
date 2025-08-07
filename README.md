# Phishing-Detector

这是一个用于钓鱼网站识别的项目，包含数据预处理、特征提取、模型训练与评估等完整流程。项目主要由 Python 实现，并使用了 PyTorch 框架进行深度学习模型的构建与训练。

## 主要模块

### `FrontEnd`
Chrome浏览器扩展。

### `BackEnd/artificial/iv_cal.py`
提供数据分箱、计算 WOE（Weight of Evidence）和 IV（Information Value）的函数，用于特征选择与评估。

### `BackEnd/artificial/sa_bs.py`
包含双向特征选择算法，用于优化模型输入特征，提高模型性能。

### `BackEnd/automatic/artificial.py`
提供 URL 特征提取函数，如 URL 长度、点的数量、是否包含 IP 地址等，用于将原始 URL 转换为可用于模型训练的特征。

### `BackEnd/automatic/feature_extraction.py`
实现用于 URL 分类的卷积神经网络（CNN）模型结构，并包含数据处理与特征提取函数。

### `BackEnd/automatic/pytorchtools.py`
包含 `EarlyStopping` 类，用于在模型训练过程中根据验证集损失停止训练，防止过拟合。

### `BackEnd/automatic/test.py` 和 `BackEnd/automatic/train.py`
分别用于模型的测试与训练，包含数据集类 `CnnDataset` 和 CNN 模型定义。

### `BackEnd/automatic/utils.py`
提供通用工具函数，如字符 n-gram 提取、数据加载、URL 分段等。

### `BackEnd/automatic/vocab.py`
实现词汇表类 `Vocab`，用于将字符或词转换为对应的 ID，便于模型处理。

## 数据

- `BackEnd/automatic/data/legtimate-58w.txt`：合法网站 URL 数据集。
- `BackEnd/automatic/data/phish-58w.txt`：钓鱼网站 URL 数据集。
- `BackEnd/automatic/data/cnn_features_*.csv`：提取的 CNN 特征数据。

## 模型

- 使用卷积神经网络（CNN）进行 URL 分类。
- 模型保存路径：`BackEnd/automatic/model_storage/model-cc.pkl`。

## 使用方法

1. 数据预处理与特征提取
2. 在项目`BackEnd`目录使用 `python -m automatic.train.py` 进行模型训练
3. 在项目`BackEnd`目录使用 `python -m automatic.test` 进行模型评估
4. 使用训练好的模型进行预测

## 依赖库

- PyTorch
- scikit-learn
- pandas
- numpy
