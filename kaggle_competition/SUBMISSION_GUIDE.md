# Kaggle 提交指南

## 📋 竞赛信息
- **竞赛名称**: LLM Classification Finetuning
- **竞赛链接**: https://www.kaggle.com/competitions/llm-classification-finetuning/
- **目标**: 预测人类在两个LLM回复之间的偏好选择

## 🚀 快速开始

### 方法 1: 使用简单基线（无需额外依赖）

```bash
# 1. 下载竞赛数据
# 访问 https://www.kaggle.com/competitions/llm-classification-finetuning/data
# 下载 train.csv 和 test.csv 到当前目录

# 2. 运行基线脚本
python3 simple_baseline.py

# 3. 上传 submission.csv 到 Kaggle
```

### 方法 2: 使用高级模型（需要安装依赖）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载竞赛数据（同上）

# 3. 运行训练脚本
python3 train_and_predict.py

# 4. 上传 submission.csv 到 Kaggle
```

## 📤 如何提交到 Kaggle

### 通过网页提交

1. 访问竞赛提交页面: https://www.kaggle.com/competitions/llm-classification-finetuning/submit
2. 点击 "Submit Predictions"
3. 上传生成的 `submission.csv` 文件
4. 添加描述（可选）
5. 点击 "Make Submission"

### 通过 Kaggle API 提交

如果你已经安装了 Kaggle API：

```bash
# 配置 Kaggle API credentials
# 从 https://www.kaggle.com/settings 下载 kaggle.json
# 放置在 ~/.kaggle/kaggle.json

# 提交
kaggle competitions submit -c llm-classification-finetuning -f submission.csv -m "Baseline submission"
```

## 📊 提交文件格式

submission.csv 应该包含以下列：

- `id`: 测试样本ID
- `winner_model_a`: 模型A获胜的概率 (0-1)
- `winner_model_b`: 模型B获胜的概率 (0-1)
- `winner_tie`: 平局的概率 (0-1)

注意：每行的三个概率应该加起来等于1.0

示例：
```
id,winner_model_a,winner_model_b,winner_tie
0,0.7,0.2,0.1
1,0.1,0.8,0.1
2,0.2,0.2,0.6
```

## 🎯 改进建议

当前的简单基线使用启发式规则，准确率较低。以下是改进方向：

1. **使用预训练语言模型**
   - DistilBERT, RoBERTa, DeBERTa
   - 对prompt和responses进行编码

2. **特征工程**
   - 回复长度、复杂度
   - 情感分析
   - 语法正确性

3. **集成学习**
   - 结合多个模型的预测
   - Voting 或 Stacking

4. **数据增强**
   - 使用更多的训练数据
   - 对抗性训练

5. **微调策略**
   - 使用对话数据预训练
   - 任务特定的损失函数

## 🔍 调试

如果遇到问题：

1. **检查数据文件**
   ```bash
   head train.csv
   head test.csv
   ```

2. **验证提交文件**
   ```bash
   head submission.csv
   wc -l submission.csv
   ```

3. **检查概率和**
   ```python
   import csv
   with open('submission.csv') as f:
       reader = csv.DictReader(f)
       for row in reader:
           total = float(row['winner_model_a']) + float(row['winner_model_b']) + float(row['winner_tie'])
           print(f"Row {row['id']}: sum = {total}")
   ```

## 📝 文件说明

- `simple_baseline.py`: 简单基线方法（仅用标准库）
- `train_and_predict.py`: 完整的transformer模型方案
- `create_sample_data.py`: 生成示例数据用于测试
- `requirements.txt`: Python依赖包列表
- `train.csv`: 训练数据（需要下载）
- `test.csv`: 测试数据（需要下载）
- `submission.csv`: 生成的提交文件

## 🏆 Good Luck!

祝你在竞赛中取得好成绩！
