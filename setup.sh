#!/bin/bash
# 部署脚本

set -e

echo "======================================"
echo "动态滚动前向验证系统 - 部署"
echo "======================================"

# 1. 创建虚拟环境（可选但推荐）
if [ ! -d "venv" ]; then
    echo "[1/3] 创建虚拟环境..."
    python3 -m venv venv
fi

# 2. 激活虚拟环境
echo "[2/3] 激活虚拟环境..."
source venv/bin/activate

# 3. 安装依赖
echo "[3/3] 安装依赖..."
pip install --upgrade pip
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
