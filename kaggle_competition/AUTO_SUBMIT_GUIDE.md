# 🚀 Kaggle竞赛自动提交指南

## 快速开始

### 方法 1: 交互式脚本（推荐）

```bash
cd /home/user/sglang/kaggle_competition
./setup_and_submit.sh
```

脚本会提示你输入Kaggle凭证，然后自动完成：
- ✓ 配置Kaggle API
- ✓ 下载竞赛数据
- ✓ 运行预测
- ✓ 提交到Kaggle

### 方法 2: 提供凭证后自动运行

```bash
cd /home/user/sglang/kaggle_competition

# 设置环境变量
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# 运行自动提交
python3 auto_submit.py
```

### 方法 3: 手动配置凭证文件

```bash
# 1. 创建目录
mkdir -p ~/.kaggle

# 2. 创建凭证文件
cat > ~/.kaggle/kaggle.json <<EOF
{
  "username": "your_username",
  "key": "your_api_key"
}
EOF

# 3. 设置权限
chmod 600 ~/.kaggle/kaggle.json

# 4. 运行自动提交
python3 auto_submit.py
```

## 📋 获取Kaggle API凭证

### 步骤 1: 登录Kaggle
访问 https://www.kaggle.com 并登录你的账号

### 步骤 2: 打开账号设置
访问 https://www.kaggle.com/settings 或点击右上角头像 → Settings

### 步骤 3: 生成API Token
1. 滚动到 **API** 部分
2. 点击 **"Create New Token"** 按钮
3. 会自动下载 `kaggle.json` 文件

### 步骤 4: 查看凭证
打开下载的 `kaggle.json` 文件，内容类似：
```json
{
  "username": "your_username",
  "key": "1234567890abcdef1234567890abcdef"
}
```

## 🔐 配置凭证的三种方式

### 选项 A: 直接告诉我你的凭证

把 `kaggle.json` 的内容发给我，我会帮你配置。格式：
```
username: your_username
key: your_api_key
```

### 选项 B: 运行交互式脚本

运行 `./setup_and_submit.sh`，按提示输入用户名和API key

### 选项 C: 手动配置

将 `kaggle.json` 放到 `~/.kaggle/kaggle.json`

## 🎯 完整自动化流程

一旦凭证配置完成，自动化脚本会：

1. **验证凭证** - 检查Kaggle API访问
2. **安装依赖** - 安装kaggle包（如需要）
3. **下载数据** - 从Kaggle下载train.csv和test.csv
4. **生成预测** - 运行预测脚本
5. **提交结果** - 自动上传submission.csv到Kaggle
6. **显示状态** - 显示提交状态和分数

## 📊 查看结果

提交完成后，访问：
https://www.kaggle.com/competitions/llm-classification-finetuning/submissions

## ⚠️ 常见问题

### Q: 网络连接失败？
A: 确保你的环境可以访问 kaggle.com

### Q: 权限错误？
A: 确保 kaggle.json 权限是 600：
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Q: 数据下载失败？
A: 可以手动下载数据：
1. 访问 https://www.kaggle.com/competitions/llm-classification-finetuning/data
2. 下载train.csv和test.csv到当前目录

### Q: 提交失败？
A: 检查：
- 凭证是否正确
- 是否接受了竞赛规则（需要先在网页上Accept规则）
- submission.csv格式是否正确

## 🔄 重新提交

如果想改进模型并重新提交：

```bash
# 修改预测脚本后
python3 auto_submit.py

# 或者只提交已有的submission.csv
kaggle competitions submit -c llm-classification-finetuning -f submission.csv -m "Second submission"
```

## 📝 提交信息

每次提交会自动添加时间戳，例如：
```
Automated submission - 2025-11-13 12:34:56
```

你也可以自定义提交信息，编辑 `auto_submit.py` 中的 `submission_message` 变量。
