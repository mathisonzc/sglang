#!/usr/bin/env python3
"""
创建示例数据用于测试
"""

import csv
import random

# 示例prompts
sample_prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "How do I make chocolate chip cookies?",
    "What are the benefits of exercise?",
    "Write a haiku about spring.",
]

# 示例responses
sample_responses_good = [
    "Paris is the capital of France.",
    "Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously.",
    "Mix butter, sugar, eggs, flour, and chocolate chips. Bake at 350°F for 12 minutes.",
    "Exercise improves cardiovascular health, strengthens muscles, and boosts mental well-being.",
    "Cherry blossoms bloom,\nSoft petals dance in the breeze,\nSpring awakens life.",
]

sample_responses_bad = [
    "The capital is London.",
    "It's computers that are really fast.",
    "Buy them from the store.",
    "Exercise is tiring.",
    "Spring is nice.",
]

# 创建训练数据
print("创建示例 train.csv...")
with open('train.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'prompt', 'response_a', 'response_b', 'model_a', 'model_b',
                     'winner_model_a', 'winner_model_b', 'winner_tie'])

    for i in range(100):
        prompt_idx = i % len(sample_prompts)
        prompt = sample_prompts[prompt_idx]

        # 随机选择response质量
        if random.random() < 0.5:
            response_a = sample_responses_good[prompt_idx]
            response_b = sample_responses_bad[prompt_idx]
            winner = 0  # model_a wins
        else:
            response_a = sample_responses_bad[prompt_idx]
            response_b = sample_responses_good[prompt_idx]
            winner = 1  # model_b wins

        # 10% 的情况是平局
        if random.random() < 0.1:
            winner = 2  # tie

        winner_a = 1 if winner == 0 else 0
        winner_b = 1 if winner == 1 else 0
        winner_tie = 1 if winner == 2 else 0

        writer.writerow([
            i,
            prompt,
            response_a,
            response_b,
            'model_a',
            'model_b',
            winner_a,
            winner_b,
            winner_tie
        ])

# 创建测试数据
print("创建示例 test.csv...")
with open('test.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'prompt', 'response_a', 'response_b'])

    for i in range(20):
        prompt_idx = i % len(sample_prompts)
        prompt = sample_prompts[prompt_idx]

        if random.random() < 0.5:
            response_a = sample_responses_good[prompt_idx]
            response_b = sample_responses_bad[prompt_idx]
        else:
            response_a = sample_responses_bad[prompt_idx]
            response_b = sample_responses_good[prompt_idx]

        writer.writerow([
            i,
            prompt,
            response_a,
            response_b
        ])

print("示例数据创建完成!")
print("- train.csv: 100 样本")
print("- test.csv: 20 样本")
