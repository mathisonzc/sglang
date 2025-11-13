#!/usr/bin/env python3
"""
自动化Kaggle竞赛提交脚本
自动下载数据、生成预测、提交结果
"""

import os
import sys
import json
import subprocess
import time

def check_kaggle_credentials():
    """检查Kaggle API凭证"""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')

    if os.path.exists(kaggle_json):
        print(f"✓ 找到Kaggle凭证: {kaggle_json}")
        return True
    else:
        print(f"✗ 未找到Kaggle凭证")
        print(f"\n请按照以下步骤获取Kaggle API凭证:")
        print(f"1. 访问 https://www.kaggle.com/settings")
        print(f"2. 滚动到 'API' 部分")
        print(f"3. 点击 'Create New Token'")
        print(f"4. 下载 kaggle.json 文件")
        print(f"5. 将文件内容提供给我，我会帮你配置\n")
        return False

def setup_kaggle_credentials(username, key):
    """配置Kaggle API凭证"""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)

    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    credentials = {
        "username": username,
        "key": key
    }

    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f)

    # 设置正确的权限
    os.chmod(kaggle_json, 0o600)
    print(f"✓ Kaggle凭证已配置: {kaggle_json}")
    return True

def install_kaggle():
    """安装Kaggle CLI"""
    print("\n正在安装Kaggle CLI...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle', '--user'],
                      check=True, capture_output=True, text=True)
        print("✓ Kaggle CLI安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Kaggle CLI安装失败: {e.stderr}")
        return False

def download_competition_data(competition_name):
    """下载竞赛数据"""
    print(f"\n正在下载竞赛数据: {competition_name}...")

    try:
        # 下载所有竞赛文件
        result = subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', competition_name],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print("✓ 数据下载成功")

            # 解压文件
            print("正在解压数据...")
            import zipfile

            zip_file = f"{competition_name}.zip"
            if os.path.exists(zip_file):
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall('.')
                os.remove(zip_file)
                print("✓ 数据解压完成")

            return True
        else:
            print(f"✗ 数据下载失败: {result.stderr}")
            return False

    except FileNotFoundError:
        print("✗ Kaggle CLI未找到，请先安装")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

def run_prediction_script():
    """运行预测脚本"""
    print("\n正在生成预测...")

    # 优先使用简单基线
    if os.path.exists('simple_baseline.py'):
        script = 'simple_baseline.py'
    elif os.path.exists('train_and_predict.py'):
        script = 'train_and_predict.py'
    else:
        print("✗ 未找到预测脚本")
        return False

    try:
        result = subprocess.run([sys.executable, script],
                              capture_output=True, text=True)
        print(result.stdout)

        if result.returncode == 0 and os.path.exists('submission.csv'):
            print("✓ 预测生成成功")
            return True
        else:
            print(f"✗ 预测生成失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

def submit_to_kaggle(competition_name, submission_file, message):
    """提交结果到Kaggle"""
    print(f"\n正在提交到Kaggle: {competition_name}...")

    try:
        result = subprocess.run(
            ['kaggle', 'competitions', 'submit',
             '-c', competition_name,
             '-f', submission_file,
             '-m', message],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print("✓ 提交成功!")
            print(result.stdout)
            return True
        else:
            print(f"✗ 提交失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

def check_submission_status(competition_name):
    """检查提交状态"""
    print(f"\n检查提交状态...")

    try:
        result = subprocess.run(
            ['kaggle', 'competitions', 'submissions', '-c', competition_name],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"检查状态失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    """主函数"""
    print("=" * 70)
    print("Kaggle LLM Classification Finetuning - 自动提交脚本")
    print("=" * 70)

    competition_name = "llm-classification-finetuning"

    # 步骤 1: 检查Kaggle凭证
    print("\n[步骤 1/5] 检查Kaggle API凭证...")
    if not check_kaggle_credentials():
        print("\n请提供Kaggle API凭证:")
        print("方法 1: 手动配置")
        print("  将 kaggle.json 放到 ~/.kaggle/kaggle.json")
        print("\n方法 2: 通过环境变量")
        print("  export KAGGLE_USERNAME=your_username")
        print("  export KAGGLE_KEY=your_key")

        # 尝试从环境变量获取
        username = os.environ.get('KAGGLE_USERNAME')
        key = os.environ.get('KAGGLE_KEY')

        if username and key:
            print(f"\n从环境变量获取凭证...")
            setup_kaggle_credentials(username, key)
        else:
            print("\n请先配置Kaggle凭证后再运行此脚本")
            sys.exit(1)

    # 步骤 2: 安装Kaggle CLI
    print("\n[步骤 2/5] 安装Kaggle CLI...")
    if not install_kaggle():
        print("尝试继续...")

    # 步骤 3: 下载数据
    print("\n[步骤 3/5] 下载竞赛数据...")
    if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
        if not download_competition_data(competition_name):
            print("数据下载失败，请手动下载数据")
            sys.exit(1)
    else:
        print("✓ 数据文件已存在")

    # 步骤 4: 生成预测
    print("\n[步骤 4/5] 运行预测脚本...")
    if not run_prediction_script():
        print("预测生成失败")
        sys.exit(1)

    # 步骤 5: 提交到Kaggle
    print("\n[步骤 5/5] 提交到Kaggle...")
    submission_message = f"Automated submission - {time.strftime('%Y-%m-%d %H:%M:%S')}"

    if submit_to_kaggle(competition_name, 'submission.csv', submission_message):
        print("\n" + "=" * 70)
        print("🎉 提交成功!")
        print("=" * 70)

        # 显示提交状态
        time.sleep(2)
        check_submission_status(competition_name)

        print(f"\n查看结果:")
        print(f"https://www.kaggle.com/competitions/{competition_name}/submissions")
    else:
        print("\n提交失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
