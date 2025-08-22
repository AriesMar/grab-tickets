#!/usr/bin/env python3
"""
🎫 Grab Tickets 一键启动脚本
启动后台管理中心，打开浏览器访问统一管理界面
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("🎫 Grab Tickets 后台管理中心")
    print("=" * 60)
    print("🚀 正在启动系统...")
    print()

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        sys.exit(1)
    print(f"✅ Python版本检查通过: {sys.version.split()[0]}")

def check_venv():
    """检查虚拟环境"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("📦 创建虚拟环境...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ 虚拟环境创建成功")
        except subprocess.CalledProcessError:
            print("❌ 虚拟环境创建失败")
            sys.exit(1)
    else:
        print("✅ 虚拟环境已存在")

def activate_venv():
    """激活虚拟环境"""
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate.bat"
        if os.path.exists(activate_script):
            os.environ['VIRTUAL_ENV'] = str(Path("venv").absolute())
            os.environ['PATH'] = f"{Path('venv/Scripts').absolute()};{os.environ['PATH']}"
            print("✅ 虚拟环境已激活 (Windows)")
        else:
            print("❌ 虚拟环境激活脚本未找到")
            sys.exit(1)
    else:  # macOS/Linux
        activate_script = "venv/bin/activate"
        if os.path.exists(activate_script):
            # 在Unix系统中，我们需要source激活脚本
            # 这里我们直接设置环境变量
            os.environ['VIRTUAL_ENV'] = str(Path("venv").absolute())
            os.environ['PATH'] = f"{Path('venv/bin').absolute()}:{os.environ['PATH']}"
            print("✅ 虚拟环境已激活 (Unix)")
        else:
            print("❌ 虚拟环境激活脚本未找到")
            sys.exit(1)

def get_venv_python():
    """获取虚拟环境中的Python解释器路径"""
    if os.name == 'nt':  # Windows
        return Path("venv/Scripts/python.exe")
    else:  # macOS/Linux
        return Path("venv/bin/python")

def get_venv_pip():
    """获取虚拟环境中的pip路径"""
    if os.name == 'nt':  # Windows
        return Path("venv/Scripts/pip.exe")
    else:  # macOS/Linux
        return Path("venv/bin/pip")

def install_dependencies():
    """安装依赖包"""
    print("📦 检查并安装依赖包...")
    try:
        # 使用虚拟环境中的pip
        venv_pip = get_venv_pip()
        if not venv_pip.exists():
            print("❌ 虚拟环境中的pip未找到")
            sys.exit(1)
        
        # 检查pip是否可用
        subprocess.run([str(venv_pip), "--version"], check=True, capture_output=True)
        
        # 安装依赖
        requirements_file = "requirements.txt"
        if os.path.exists(requirements_file):
            print("   正在安装依赖包...")
            result = subprocess.run([
                str(venv_pip), "install", "-r", requirements_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 依赖包安装成功")
            else:
                print("⚠️  依赖包安装可能有问题，但继续启动...")
                print(f"   错误信息: {result.stderr}")
        else:
            print("⚠️  requirements.txt文件未找到，跳过依赖安装")
    except Exception as e:
        print(f"⚠️  依赖检查失败: {e}")
        print("   继续启动...")

def start_main_system():
    """启动主系统"""
    print("🚀 启动Grab Tickets主系统...")
    
    # 检查main.py是否存在
    if not os.path.exists("main.py"):
        print("❌ main.py文件未找到")
        sys.exit(1)
    
    try:
        # 使用虚拟环境中的Python启动主系统
        venv_python = get_venv_python()
        if not venv_python.exists():
            print("❌ 虚拟环境中的Python未找到")
            sys.exit(1)
        
        print("   正在启动后台服务...")
        process = subprocess.Popen([
            str(venv_python), "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待系统启动
        print("   等待系统启动...")
        time.sleep(8)  # 等待8秒让系统完全启动
        
        # 检查进程是否还在运行
        if process.poll() is None:
            print("✅ 主系统启动成功")
            return process
        else:
            stdout, stderr = process.communicate()
            print("❌ 主系统启动失败")
            print(f"   错误信息: {stderr.decode()}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

def wait_for_service(port=8080, timeout=30):
    """等待服务启动"""
    print(f"⏳ 等待服务在端口 {port} 启动...")
    
    import socket
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                print(f"✅ 服务已在端口 {port} 启动")
                return True
        except:
            pass
        
        time.sleep(1)
        print(f"   等待中... ({int(time.time() - start_time)}s)")
    
    print(f"❌ 服务启动超时 (端口 {port})")
    return False

def open_browser():
    """打开浏览器"""
    print("🌐 打开浏览器...")
    try:
        webbrowser.open(f"http://localhost:8080")
        print("✅ 浏览器已打开")
    except Exception as e:
        print(f"⚠️  浏览器打开失败: {e}")
        print("   请手动访问: http://localhost:8080")

def print_success_info():
    """打印成功信息"""
    print()
    print("🎉 启动完成！")
    print("=" * 60)
    print("📱 后台管理中心: http://localhost:8080")
    print("📊 系统监控指标: http://localhost:8001")
    print("=" * 60)
    print("💡 提示:")
    print("   - 按 Ctrl+C 停止系统")
    print("   - 系统会自动在后台运行")
    print("   - 关闭终端不会影响系统运行")
    print()

def main():
    """主函数"""
    try:
        # 打印横幅
        print_banner()
        
        # 检查Python版本
        check_python_version()
        
        # 检查虚拟环境
        check_venv()
        
        # 激活虚拟环境
        activate_venv()
        
        # 安装依赖
        install_dependencies()
        
        # 启动主系统
        process = start_main_system()
        
        # 等待服务启动
        if wait_for_service(8080):
            # 打开浏览器
            open_browser()
            
            # 打印成功信息
            print_success_info()
            
            # 保持进程运行
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 收到停止信号，正在关闭系统...")
                process.terminate()
                process.wait()
                print("✅ 系统已关闭")
        else:
            print("❌ 服务启动失败，请检查日志")
            process.terminate()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 启动被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 启动过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
