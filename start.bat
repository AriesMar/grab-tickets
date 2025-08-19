@echo off
chcp 65001 >nul
title 抢票项目一键启动器

echo 🚀 启动抢票项目Web启动器...
echo.

REM 检查虚拟环境
if not exist "venv" (
    echo ❌ 虚拟环境不存在，正在创建...
    python -m venv venv
    echo ✅ 虚拟环境创建完成
)

REM 激活虚拟环境
echo 🔧 激活虚拟环境...
call venv\Scripts\activate.bat

REM 检查依赖
echo 📦 检查依赖包...
python -c "import fastapi, uvicorn, psutil" 2>nul
if errorlevel 1 (
    echo 📥 安装依赖包...
    pip install fastapi uvicorn psutil
)

REM 启动Web启动器
echo.
echo 🌐 启动Web启动器...
echo 📱 访问地址: http://localhost:8081
echo 📊 健康检查: http://localhost:8081/health
echo.
echo 按 Ctrl+C 停止启动器
echo.

python start_web.py

pause
