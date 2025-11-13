# Kaggle LLM Classification Finetuning Competition

## 📌 竞赛说明
这是 Kaggle LLM Classification Finetuning 竞赛的解决方案。

**竞赛目标**: 预测人类在两个LLM响应之间的偏好选择
**竞赛链接**: https://www.kaggle.com/competitions/llm-classification-finetuning/
**评估指标**: 多分类对数损失 (Log Loss)

## 🗂️ 项目文件

- `simple_baseline.py` - **推荐**: 简单基线方法，仅使用Python标准库，无需额外依赖
- `train_and_predict.py` - 完整的transformer模型方案（需要安装依赖）
- `create_sample_data.py` - 生成示例数据用于测试
- `requirements.txt` - Python依赖包列表
- `SUBMISSION_GUIDE.md` - 详细的提交指南

## 🚀 快速开始

### 方法 1: 简单基线（推荐，无需安装依赖）

```bash
# 1. 下载竞赛数据到此目录
#    访问 https://www.kaggle.com/competitions/llm-classification-finetuning/data
#    下载 train.csv 和 test.csv

# 2. 运行基线脚本
python3 simple_baseline.py

# 3. 生成的 submission.csv 可以直接提交到 Kaggle
```

### 方法 2: Transformer模型（需要GPU和依赖）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据（同上）

# 3. 运行训练脚本
python3 train_and_predict.py
```

## 📊 数据格式

### 训练集 (train.csv)
- `prompt`: 输入提示词
- `response_a`: 模型A的回复
- `response_b`: 模型B的回复
- `model_a`, `model_b`: 模型标识
- `winner_model_a`, `winner_model_b`, `winner_tie`: 获胜标签（互斥，仅一个为1）

### 测试集 (test.csv)
- `id`: 样本ID
- `prompt`: 输入提示词
- `response_a`: 模型A的回复
- `response_b`: 模型B的回复

### 提交文件 (submission.csv)
- `id`: 样本ID
- `winner_model_a`: 模型A获胜概率 [0-1]
- `winner_model_b`: 模型B获胜概率 [0-1]
- `winner_tie`: 平局概率 [0-1]

注意：每行三个概率之和应为1.0

## 🧪 测试

如果你想在没有真实数据的情况下测试脚本：

```bash
# 生成示例数据
python3 create_sample_data.py

# 运行基线方法
python3 simple_baseline.py
```

## 📤 提交

详细的提交指南请参见 [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)

**快速提交**:
1. 访问 https://www.kaggle.com/competitions/llm-classification-finetuning/submit
2. 上传生成的 `submission.csv`
3. 点击提交

## 🎯 当前状态

- ✅ 简单基线已实现并测试（使用启发式规则）
- ✅ 完整的transformer模型代码已准备
- ✅ 生成了有效的提交文件格式
- 📝 基线准确率: ~24% (在示例数据上)

## 🔍 下一步改进

1. 使用预训练语言模型（BERT, RoBERTa等）
2. 实现更复杂的特征工程
3. 尝试集成学习方法
4. 优化超参数

## 📝 注意事项

- 当前环境网络受限，无法直接下载Kaggle数据
- 需要手动下载数据文件
- 简单基线使用启发式规则，性能有限但可以快速生成提交
- 建议在有GPU的环境中运行transformer模型以获得更好的性能
