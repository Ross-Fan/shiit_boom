#!/bin/bash
# 部署脚本

set -e

echo "======================================"
echo "动态滚动前向验证系统 - 部署"
echo "======================================"

# 检测Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "检测到 Python $PYTHON_VERSION"

# 0. Ubuntu/Debian系统：安装必要的系统包
if [ -f /etc/debian_version ]; then
    echo ""
    echo "[0/4] 检查系统依赖 (Ubuntu/Debian)..."

    # 检查是否需要安装python3-venv
    if ! dpkg -l | grep -q python3-venv && ! dpkg -l | grep -q "python${PYTHON_VERSION}-venv"; then
        echo "需要安装 python3-venv，请输入密码..."
        sudo apt update
        sudo apt install -y python3-venv python3-pip
    else
        echo "系统依赖已满足"
    fi
fi

# 1. 创建虚拟环境
if [ ! -d "venv" ]; then
    echo ""
    echo "[1/4] 创建虚拟环境..."
    python3 -m venv venv
else
    echo ""
    echo "[1/4] 虚拟环境已存在，跳过创建"
fi

# 2. 激活虚拟环境
echo ""
echo "[2/4] 激活虚拟环境..."
source venv/bin/activate

# 3. 升级pip
echo ""
echo "[3/4] 升级pip..."
pip install --upgrade pip -q

# 4. 安装依赖
echo ""
echo "[4/4] 安装依赖..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "部署完成！"
echo "======================================"
echo ""
echo "使用方法:"
echo "  source venv/bin/activate"
echo "  python run_backtest.py --dynamic --start 2026-01-01 --end 2026-01-30"
echo ""
echo "后台运行:"
echo "  nohup python run_backtest.py --dynamic --start 2026-01-01 --end 2026-01-30 > backtest.log 2>&1 &"
echo ""
