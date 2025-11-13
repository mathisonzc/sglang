#!/usr/bin/env python3
"""
Kaggle LLM Classification Finetuning - 预测人类偏好
使用轻量级transformer模型进行三分类（model_a获胜、model_b获胜、平局）
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("警告: transformers 或 torch 未安装，将使用基础方法")


class PreferenceDataset(Dataset):
    """人类偏好数据集"""
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def prepare_text_input(row):
    """准备输入文本：组合prompt和两个responses"""
    text = f"Question: {row['prompt']}\n\n"
    text += f"Response A: {row['response_a']}\n\n"
    text += f"Response B: {row['response_b']}"
    return text


def prepare_labels(row):
    """准备标签：0=model_a获胜, 1=model_b获胜, 2=平局"""
    if row['winner_model_a'] == 1:
        return 0
    elif row['winner_model_b'] == 1:
        return 1
    else:  # winner_tie == 1
        return 2


def train_transformer_model(train_df, test_df):
    """使用transformer模型训练"""
    print("正在加载预训练模型...")

    # 使用轻量级模型以节省资源
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # 三分类
    )

    # 准备数据
    print("准备训练数据...")
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()

    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )

    train_dataset = PreferenceDataset(train_texts, train_labels, tokenizer)
    val_dataset = PreferenceDataset(val_texts, val_labels, tokenizer)

    # 训练配置
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
    )

    # 训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # 预测
    print("生成预测...")
    test_texts = test_df['text'].tolist()
    test_dataset = PreferenceDataset(test_texts, None, tokenizer)

    predictions = trainer.predict(test_dataset)
    pred_probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

    return pred_probs


def train_baseline_model(train_df, test_df):
    """基础基线模型：基于简单特征的分类"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    print("使用基线模型（TF-IDF + Logistic Regression）...")

    # 准备数据
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['text'].tolist()

    # TF-IDF特征
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # 训练逻辑回归
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, train_labels)

    # 预测概率
    pred_probs = model.predict_proba(X_test)

    return pred_probs


def main():
    """主函数"""
    print("=" * 60)
    print("Kaggle LLM Classification Finetuning 竞赛")
    print("=" * 60)

    # 检查数据文件
    if not os.path.exists('train.csv'):
        print("\n错误: 未找到 train.csv")
        print("请从 Kaggle 下载数据文件并放置在此目录下")
        print("下载地址: https://www.kaggle.com/competitions/llm-classification-finetuning/data")
        return

    if not os.path.exists('test.csv'):
        print("\n错误: 未找到 test.csv")
        print("请从 Kaggle 下载数据文件并放置在此目录下")
        return

    # 加载数据
    print("\n加载数据...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}")

    # 数据探索
    print("\n数据探索:")
    print(f"训练集列: {train_df.columns.tolist()}")

    if 'winner_model_a' in train_df.columns:
        winner_dist = pd.Series({
            'model_a': train_df['winner_model_a'].sum(),
            'model_b': train_df['winner_model_b'].sum(),
            'tie': train_df['winner_tie'].sum() if 'winner_tie' in train_df.columns else 0
        })
        print(f"\n获胜者分布:\n{winner_dist}")

    # 准备输入文本
    print("\n准备输入文本...")
    train_df['text'] = train_df.apply(prepare_text_input, axis=1)
    test_df['text'] = test_df.apply(prepare_text_input, axis=1)

    # 准备标签
    train_df['label'] = train_df.apply(prepare_labels, axis=1)

    # 选择训练方法
    if HAS_TRANSFORMERS and torch.cuda.is_available():
        print("\n使用 GPU 加速的 Transformer 模型")
        pred_probs = train_transformer_model(train_df, test_df)
    elif HAS_TRANSFORMERS:
        print("\n使用 CPU 的 Transformer 模型（较慢）")
        pred_probs = train_transformer_model(train_df, test_df)
    else:
        print("\n使用基线模型")
        pred_probs = train_baseline_model(train_df, test_df)

    # 创建提交文件
    print("\n生成提交文件...")
    submission = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
        'winner_model_a': pred_probs[:, 0],
        'winner_model_b': pred_probs[:, 1],
        'winner_tie': pred_probs[:, 2]
    })

    submission.to_csv('submission.csv', index=False)
    print(f"\n提交文件已生成: submission.csv")
    print(f"样本预测（前5行）:")
    print(submission.head())

    print("\n" + "=" * 60)
    print("完成！现在可以提交 submission.csv 到 Kaggle")
    print("=" * 60)


if __name__ == "__main__":
    main()
