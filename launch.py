#!/usr/bin/env python3
"""
🚀 抢票项目一键启动器
整合了Web启动器和项目启动功能
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional

# 项目配置
PROJECT_NAME = "抢票项目"
PROJECT_ROOT = Path(__file__).parent
MAIN_SCRIPT = "main.py"
VENV_ACTIVATE = "venv/bin/activate"
PID_FILE = "project.pid"
WEB_LAUNCHER_PORT = 8081

class ProjectLauncher:
    """项目启动器"""
    
    def __init__(self):
        self.project_process: Optional[subprocess.Popen] = None
        self.web_launcher_process: Optional[subprocess.Popen] = None
        
    def check_dependencies(self) -> bool:
        """检查依赖包"""
        print("📦 检查依赖包...")
        
        required_packages = ["fastapi", "uvicorn"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
            print("🔧 正在安装...")
            
            try:
                # 激活虚拟环境
                if os.path.exists(VENV_ACTIVATE):
                    os.system(f"source {VENV_ACTIVATE} && pip install {' '.join(missing_packages)}")
                else:
                    os.system(f"pip install {' '.join(missing_packages)}")
                
                # 再次检查
                for package in missing_packages:
                    try:
                        __import__(package)
                    except ImportError:
                        print(f"❌ 安装失败: {package}")
                        return False
                        
                print("✅ 依赖包安装完成")
                return True
                
            except Exception as e:
                print(f"❌ 安装失败: {e}")
                return False
        
        print("✅ 依赖包检查完成")
        return True
    
    def check_virtual_env(self) -> bool:
        """检查虚拟环境"""
        print("🔧 检查虚拟环境...")
        
        if not os.path.exists(VENV_ACTIVATE):
            print("❌ 虚拟环境不存在，正在创建...")
            try:
                os.system("python3 -m venv venv")
                print("✅ 虚拟环境创建完成")
            except Exception as e:
                print(f"❌ 创建失败: {e}")
                return False
        
        print("✅ 虚拟环境检查完成")
        return True
    
    def start_web_launcher(self) -> bool:
        """启动Web启动器"""
        print("🌐 启动Web启动器...")
        
        try:
            # 启动Web启动器
            cmd = f"python3 {PROJECT_ROOT}/start_web_simple.py"
            
            self.web_launcher_process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=PROJECT_ROOT
            )
            
            # 等待启动
            time.sleep(3)
            
            if self.web_launcher_process.poll() is None:
                print("✅ Web启动器启动成功")
                return True
            else:
                print("❌ Web启动器启动失败")
                return False
                
        except Exception as e:
            print(f"❌ 启动Web启动器失败: {e}")
            return False
    
    def start_project(self) -> bool:
        """启动抢票项目"""
        print("🚀 启动抢票项目...")
        
        try:
            # 检查是否已在运行
            if self.is_project_running():
                print("⚠️ 项目已在运行中")
                return True
            
            # 启动命令
            cmd = f"source {VENV_ACTIVATE} && python3 {MAIN_SCRIPT}"
            
            self.project_process = subprocess.Popen(
                cmd,
                shell=True,
                executable="/bin/bash",
                cwd=PROJECT_ROOT
            )
            
            # 保存PID
            with open(PID_FILE, 'w') as f:
                f.write(str(self.project_process.pid))
            
            # 等待启动
            time.sleep(5)
            
            if self.project_process.poll() is None:
                print("✅ 抢票项目启动成功")
                return True
            else:
                print("❌ 抢票项目启动失败")
                return False
                
        except Exception as e:
            print(f"❌ 启动项目失败: {e}")
            return False
    
    def is_project_running(self) -> bool:
        """检查项目是否在运行"""
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, 'r') as f:
                    pid = f.read().strip()
                return os.path.exists(f"/proc/{pid}") if os.name == 'posix' else True
            except:
                pass
        return False
    
    def open_browser(self):
        """打开浏览器"""
        print("🌐 打开浏览器...")
        
        # 等待Web启动器启动
        time.sleep(2)
        
        try:
            # 打开Web启动器
            webbrowser.open(f"http://localhost:{WEB_LAUNCHER_PORT}")
            print("✅ 浏览器已打开")
        except Exception as e:
            print(f"⚠️ 无法自动打开浏览器: {e}")
            print(f"请手动访问: http://localhost:{WEB_LAUNCHER_PORT}")
    
    def show_status(self):
        """显示状态信息"""
        print("\n" + "="*60)
        print(f"🎯 {PROJECT_NAME} 启动状态")
        print("="*60)
        
        # 项目状态
        project_status = "🟢 运行中" if self.is_project_running() else "🔴 已停止"
        print(f"📱 抢票项目: {project_status}")
        
        # Web启动器状态
        web_status = "🟢 运行中" if self.web_launcher_process and self.web_launcher_process.poll() is None else "🔴 已停止"
        print(f"🌐 Web启动器: {web_status}")
        
        # 访问地址
        print(f"\n🌐 访问地址:")
        print(f"   Web启动器: http://localhost:{WEB_LAUNCHER_PORT}")
        print(f"   抢票系统: http://localhost:8080")
        print(f"   监控指标: http://localhost:8001")
        
        print("\n💡 使用说明:")
        print("   1. 在Web启动器中点击'启动项目'开始抢票")
        print("   2. 使用抢票系统控制台管理任务")
        print("   3. 查看监控指标了解系统状态")
        print("   4. 按 Ctrl+C 停止启动器")
        
        print("="*60)
    
    def run(self):
        """运行启动器"""
        print(f"🚀 启动 {PROJECT_NAME} 一键启动器...")
        print(f"📁 项目路径: {PROJECT_ROOT}")
        
        # 检查依赖
        if not self.check_dependencies():
            print("❌ 依赖检查失败，请手动安装")
            return
        
        # 检查虚拟环境
        if not self.check_virtual_env():
            print("❌ 虚拟环境检查失败")
            return
        
        # 启动Web启动器
        if not self.start_web_launcher():
            print("❌ Web启动器启动失败")
            return
        
        # 显示状态
        self.show_status()
        
        # 打开浏览器
        self.open_browser()
        
        print("\n🎉 启动完成！请在浏览器中操作...")
        print("按 Ctrl+C 停止启动器")
        
        try:
            # 保持运行
            while True:
                time.sleep(1)
                
                # 检查Web启动器状态
                if self.web_launcher_process and self.web_launcher_process.poll() is not None:
                    print("⚠️ Web启动器已停止")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 正在停止启动器...")
            self.cleanup()
            print("👋 启动器已停止")
    
    def cleanup(self):
        """清理资源"""
        try:
            # 停止Web启动器
            if self.web_launcher_process:
                self.web_launcher_process.terminate()
                self.web_launcher_process.wait(timeout=5)
            
            # 停止项目进程
            if self.is_project_running():
                os.system(f"kill $(cat {PID_FILE})")
                if os.path.exists(PID_FILE):
                    os.remove(PID_FILE)
                    
        except Exception as e:
            print(f"⚠️ 清理资源时出错: {e}")

def main():
    """主函数"""
    launcher = ProjectLauncher()
    
    try:
        launcher.run()
    except KeyboardInterrupt:
        print("\n👋 启动器已停止")
    except Exception as e:
        print(f"❌ 启动器运行出错: {e}")
        launcher.cleanup()

if __name__ == "__main__":
    main()
