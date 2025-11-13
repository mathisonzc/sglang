#!/bin/bash
# Kaggle竞赛自动提交脚本

set -e

echo "============================================================"
echo "Kaggle LLM Classification Finetuning - 自动提交"
echo "============================================================"

# 切换到脚本目录
cd "$(dirname "$0")"

echo ""
echo "需要你的Kaggle API凭证才能继续"
echo "获取方式:"
echo "1. 访问 https://www.kaggle.com/settings"
echo "2. 滚动到 'API' 部分"
echo "3. 点击 'Create New Token' 下载 kaggle.json"
echo ""

# 检查是否已有凭证
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✓ 找到现有的Kaggle凭证"
else
    echo "请提供Kaggle凭证:"
    read -p "Kaggle用户名: " KAGGLE_USERNAME
    read -p "Kaggle API Key: " KAGGLE_KEY

    # 创建凭证文件
    mkdir -p ~/.kaggle
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    echo "✓ Kaggle凭证已配置"
fi

# 设置环境变量
export PATH="$HOME/.local/bin:$PATH"

# 运行Python自动提交脚本
echo ""
echo "开始自动化流程..."
python3 auto_submit.py

echo ""
echo "完成!"
