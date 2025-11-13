#!/usr/bin/env python3
"""
简单基线方法 - 仅使用Python标准库
基于启发式规则预测人类偏好
"""

import csv
import re
from collections import Counter


def load_csv(filename):
    """加载CSV文件"""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def calculate_quality_score(text):
    """计算回复质量分数（简单启发式）"""
    score = 0.0

    # 长度得分（更长通常更详细）
    words = text.split()
    word_count = len(words)
    score += min(word_count / 10, 5.0)  # 最多5分

    # 标点符号（表示结构化）
    punctuation_count = len(re.findall(r'[.,!?;:]', text))
    score += min(punctuation_count / 5, 2.0)  # 最多2分

    # 大写字母开头（表示规范）
    if text and text[0].isupper():
        score += 1.0

    # 包含数字（可能是具体信息）
    if re.search(r'\d', text):
        score += 1.0

    # 避免过短回复
    if word_count < 3:
        score -= 2.0

    # 避免全大写（可能是喊叫）
    if text.isupper() and len(text) > 10:
        score -= 1.0

    return max(score, 0.0)


def predict_winner(row):
    """预测获胜者"""
    prompt = row['prompt']
    response_a = row['response_a']
    response_b = row['response_b']

    # 计算两个回复的质量分数
    score_a = calculate_quality_score(response_a)
    score_b = calculate_quality_score(response_b)

    # 如果差距很小，判定为平局
    diff = abs(score_a - score_b)
    if diff < 1.0:
        # 平局
        return [0.2, 0.2, 0.6]  # [prob_a, prob_b, prob_tie]
    elif score_a > score_b:
        # A获胜
        confidence = min(diff / 5.0, 0.8)
        return [confidence, (1-confidence)/2, (1-confidence)/2]
    else:
        # B获胜
        confidence = min(diff / 5.0, 0.8)
        return [(1-confidence)/2, confidence, (1-confidence)/2]


def main():
    """主函数"""
    print("=" * 60)
    print("简单基线方法 - Kaggle LLM Classification")
    print("=" * 60)

    # 检查文件
    try:
        train_data = load_csv('train.csv')
        test_data = load_csv('test.csv')
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保 train.csv 和 test.csv 在当前目录")
        return

    print(f"\n训练集样本数: {len(train_data)}")
    print(f"测试集样本数: {len(test_data)}")

    # 分析训练集（可选）
    print("\n分析训练集...")
    correct = 0
    total = len(train_data)

    for row in train_data:
        pred = predict_winner(row)
        true_winner = 0 if row['winner_model_a'] == '1' else (1 if row['winner_model_b'] == '1' else 2)
        pred_winner = pred.index(max(pred))

        if pred_winner == true_winner:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"训练集准确率: {accuracy:.2%}")

    # 生成预测
    print("\n生成测试集预测...")
    predictions = []

    for row in test_data:
        pred_probs = predict_winner(row)
        predictions.append({
            'id': row['id'],
            'winner_model_a': pred_probs[0],
            'winner_model_b': pred_probs[1],
            'winner_tie': pred_probs[2]
        })

    # 保存提交文件
    print("保存提交文件...")
    with open('submission.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'winner_model_a', 'winner_model_b', 'winner_tie']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"\n提交文件已生成: submission.csv")
    print(f"样本预测（前5行）:")

    # 显示前5行
    for i, pred in enumerate(predictions[:5]):
        print(f"  ID {pred['id']}: "
              f"A={pred['winner_model_a']:.3f}, "
              f"B={pred['winner_model_b']:.3f}, "
              f"Tie={pred['winner_tie']:.3f}")

    print("\n" + "=" * 60)
    print("完成！现在可以提交 submission.csv 到 Kaggle")
    print("=" * 60)


if __name__ == "__main__":
    main()
