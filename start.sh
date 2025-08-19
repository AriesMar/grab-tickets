#!/bin/bash

# 抢票项目一键启动脚本
# 使用方法: ./start.sh

echo "🚀 启动抢票项目Web启动器..."

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，正在创建..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 检查依赖
echo "📦 检查依赖包..."
if ! python3 -c "import fastapi, uvicorn, psutil" 2>/dev/null; then
    echo "📥 安装依赖包..."
    pip install fastapi uvicorn psutil
fi

# 启动Web启动器
echo "🌐 启动Web启动器..."
echo "📱 访问地址: http://localhost:8081"
echo "📊 健康检查: http://localhost:8081/health"
echo ""
echo "按 Ctrl+C 停止启动器"
echo ""

python3 start_web.py
