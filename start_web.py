#!/usr/bin/env python3
"""
一键启动脚本 - 通过Web界面启动抢票项目
支持启动、停止、重启、状态查看等功能
"""

import os
import sys
import time
import json
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional
import psutil

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from fastapi import FastAPI, HTTPException, Request, Form
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    from pydantic import BaseModel
except ImportError as e:
    print(f"缺少依赖包: {e}")
    print("请运行: pip install fastapi uvicorn")
    sys.exit(1)

# 项目配置
PROJECT_NAME = "抢票项目"
PROJECT_ROOT = project_root
MAIN_SCRIPT = "main.py"
VENV_ACTIVATE = "venv/bin/activate"
LOG_FILE = "logs/startup.log"
PID_FILE = "project.pid"

# 创建日志目录
os.makedirs("logs", exist_ok=True)

class ProjectStatus(BaseModel):
    """项目状态模型"""
    is_running: bool
    pid: Optional[int]
    start_time: Optional[str]
    uptime: Optional[str]
    memory_usage: Optional[str]
    cpu_usage: Optional[str]
    ports: Dict[str, int]
    log_tail: List[str]

class StartupManager:
    """项目启动管理器"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.log_buffer: List[str] = []
        
    def start_project(self) -> Dict[str, str]:
        """启动项目"""
        try:
            if self.is_project_running():
                return {"status": "error", "message": "项目已在运行中"}
            
            # 检查虚拟环境
            if not os.path.exists(VENV_ACTIVATE):
                return {"status": "error", "message": "虚拟环境不存在，请先创建虚拟环境"}
            
            # 启动命令
            cmd = f"source {VENV_ACTIVATE} && python3 {MAIN_SCRIPT}"
            
            # 使用bash启动
            self.process = subprocess.Popen(
                cmd,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=PROJECT_ROOT,
                preexec_fn=os.setsid  # 创建新进程组
            )
            
            self.start_time = time.time()
            
            # 保存PID
            with open(PID_FILE, 'w') as f:
                f.write(str(self.process.pid))
            
            # 等待启动
            time.sleep(3)
            
            if self.process.poll() is None:
                return {"status": "success", "message": f"项目启动成功，PID: {self.process.pid}"}
            else:
                return {"status": "error", "message": "项目启动失败"}
                
        except Exception as e:
            return {"status": "error", "message": f"启动失败: {str(e)}"}
    
    def stop_project(self) -> Dict[str, str]:
        """停止项目"""
        try:
            if not self.is_project_running():
                return {"status": "error", "message": "项目未在运行"}
            
            # 停止进程
            if self.process:
                os.killpg(os.getpgid(self.process.pid), 9)
                self.process = None
            
            # 清理PID文件
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
            
            self.start_time = None
            return {"status": "success", "message": "项目已停止"}
            
        except Exception as e:
            return {"status": "error", "message": f"停止失败: {str(e)}"}
    
    def restart_project(self) -> Dict[str, str]:
        """重启项目"""
        stop_result = self.stop_project()
        if stop_result["status"] == "error":
            return stop_result
        
        time.sleep(2)
        return self.start_project()
    
    def is_project_running(self) -> bool:
        """检查项目是否在运行"""
        # 检查PID文件
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, 'r') as f:
                    pid = int(f.read().strip())
                
                # 检查进程是否存在
                if psutil.pid_exists(pid):
                    # 检查是否是我们的项目进程
                    try:
                        process = psutil.Process(pid)
                        cmdline = " ".join(process.cmdline())
                        if MAIN_SCRIPT in cmdline:
                            return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (ValueError, FileNotFoundError):
                pass
        
        # 清理无效的PID文件
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        
        return False
    
    def get_project_status(self) -> ProjectStatus:
        """获取项目状态"""
        is_running = self.is_project_running()
        pid = None
        start_time = None
        uptime = None
        memory_usage = None
        cpu_usage = None
        ports = {}
        
        if is_running and os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, 'r') as f:
                    pid = int(f.read().strip())
                
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    
                    # 获取启动时间
                    start_time = time.strftime(
                        "%Y-%m-%d %H:%M:%S", 
                        time.localtime(process.create_time())
                    )
                    
                    # 计算运行时间
                    uptime_seconds = time.time() - process.create_time()
                    hours = int(uptime_seconds // 3600)
                    minutes = int((uptime_seconds % 3600) // 60)
                    seconds = int(uptime_seconds % 60)
                    uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    # 获取资源使用情况
                    try:
                        memory_info = process.memory_info()
                        memory_usage = f"{memory_info.rss / 1024 / 1024:.1f} MB"
                        
                        cpu_percent = process.cpu_percent()
                        cpu_usage = f"{cpu_percent:.1f}%"
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    
                    # 检查端口使用情况
                    try:
                        connections = process.connections()
                        for conn in connections:
                            if conn.status == 'LISTEN':
                                ports[f"Port {conn.laddr.port}"] = conn.laddr.port
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
            except (ValueError, FileNotFoundError):
                pass
        
        # 获取日志尾部
        log_tail = self.get_log_tail()
        
        return ProjectStatus(
            is_running=is_running,
            pid=pid,
            start_time=start_time,
            uptime=uptime,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            ports=ports,
            log_tail=log_tail
        )
    
    def get_log_tail(self, lines: int = 20) -> List[str]:
        """获取日志尾部"""
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                    return [line.strip() for line in all_lines[-lines:]]
        except Exception:
            pass
        return []

# 创建FastAPI应用
app = FastAPI(title=f"{PROJECT_NAME} 启动器", version="1.0.0")

# 创建启动管理器
startup_manager = StartupManager()

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{PROJECT_NAME} 启动器</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {{
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #56ab2f;
            --danger-color: #ff416c;
            --warning-color: #f093fb;
            --info-color: #4facfe;
            --dark-color: #2d3748;
            --light-color: #f8f9fa;
            --white-color: #ffffff;
            --shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            --border-radius: 20px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
            overflow-x: hidden;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            overflow: hidden;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .header {{
            background: linear-gradient(135deg, var(--info-color) 0%, #00f2fe 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            opacity: 0.3;
        }}

        .header h1 {{
            font-size: 3.5em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            position: relative;
            z-index: 1;
        }}

        .header p {{
            font-size: 1.4em;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }}

        .content {{
            padding: 40px;
        }}

        .control-panel {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}

        .btn {{
            padding: 20px 30px;
            border: none;
            border-radius: 15px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            text-transform: uppercase;
            letter-spacing: 1px;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }}

        .btn::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }}

        .btn:hover::before {{
            left: 100%;
        }}

        .btn-primary {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
        }}

        .btn-primary:hover {{
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
        }}

        .btn-success {{
            background: linear-gradient(135deg, var(--success-color) 0%, #a8e6cf 100%);
            color: white;
        }}

        .btn-success:hover {{
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
        }}

        .btn-danger {{
            background: linear-gradient(135deg, var(--danger-color) 0%, #ff4b2b 100%);
            color: white;
        }}

        .btn-danger:hover {{
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
        }}

        .btn-warning {{
            background: linear-gradient(135deg, var(--warning-color) 0%, #f5576c 100%);
            color: white;
        }}

        .btn-warning:hover {{
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
        }}

        .status-panel {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: var(--border-radius);
            padding: 30px;
            margin-bottom: 40px;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }}

        .status-panel h2 {{
            color: #333;
            margin-bottom: 25px;
            font-size: 1.8em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
        }}

        .status-item {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(0, 0, 0, 0.05);
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }}

        .status-item::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        }}

        .status-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
        }}

        .status-item h3 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.1em;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .status-value {{
            font-size: 1.5em;
            font-weight: 600;
            color: var(--primary-color);
        }}

        .status-running {{
            color: var(--success-color) !important;
        }}

        .status-stopped {{
            color: var(--danger-color) !important;
        }}

        .log-panel {{
            background: linear-gradient(135deg, var(--dark-color) 0%, #1a202c 100%);
            border-radius: var(--border-radius);
            padding: 30px;
            color: #e2e8f0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .log-panel h3 {{
            color: #f7fafc;
            margin-bottom: 20px;
            font-size: 1.4em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .log-controls {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .refresh-btn {{
            background: var(--info-color);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .refresh-btn:hover {{
            background: #3182ce;
            transform: translateY(-2px);
        }}

        .auto-refresh {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .auto-refresh label {{
            color: #f7fafc;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .auto-refresh input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            accent-color: var(--info-color);
        }}

        .log-content {{
            background: #0f1419;
            border-radius: 15px;
            padding: 25px;
            max-height: 500px;
            overflow-y: auto;
            font-family: 'Courier New', 'Monaco', monospace;
            font-size: 0.95em;
            line-height: 1.6;
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
        }}

        .log-content::-webkit-scrollbar {{
            width: 8px;
        }}

        .log-content::-webkit-scrollbar-track {{
            background: #1a202c;
            border-radius: 4px;
        }}

        .log-content::-webkit-scrollbar-thumb {{
            background: var(--primary-color);
            border-radius: 4px;
        }}

        .log-content::-webkit-scrollbar-thumb:hover {{
            background: var(--secondary-color);
        }}

        .log-line {{
            margin-bottom: 8px;
            padding: 8px 12px;
            border-radius: 8px;
            transition: var(--transition);
            position: relative;
        }}

        .log-line:nth-child(even) {{
            background: rgba(255, 255, 255, 0.03);
        }}

        .log-line:hover {{
            background: rgba(255, 255, 255, 0.08);
            transform: translateX(5px);
        }}

        .log-line::before {{
            content: '>';
            color: var(--info-color);
            margin-right: 10px;
            font-weight: bold;
        }}

        .floating-actions {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            display: flex;
            flex-direction: column;
            gap: 15px;
            z-index: 1000;
        }}

        .floating-btn {{
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            font-size: 1.5em;
            cursor: pointer;
            transition: var(--transition);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .floating-btn:hover {{
            transform: scale(1.1) rotate(5deg);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }}

        .notification {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
            border-left: 5px solid var(--success-color);
            transform: translateX(400px);
            transition: var(--transition);
            z-index: 1001;
            max-width: 350px;
        }}

        .notification.show {{
            transform: translateX(0);
        }}

        .notification.success {{
            border-left-color: var(--success-color);
        }}

        .notification.error {{
            border-left-color: var(--danger-color);
        }}

        .notification h4 {{
            margin-bottom: 8px;
            color: #333;
        }}

        .notification p {{
            color: #666;
            margin: 0;
        }}

        @media (max-width: 768px) {{
            .control-panel {{
                grid-template-columns: 1fr;
            }}
            .status-grid {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 2.5em;
            }}
            .content {{
                padding: 20px;
            }}
            .floating-actions {{
                bottom: 20px;
                right: 20px;
            }}
        }}

        .loading {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        .pulse {{
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-rocket"></i> {PROJECT_NAME} 启动器</h1>
            <p>🚀 一键启动、停止、重启和管理你的抢票项目</p>
        </div>
        
        <div class="content">
            <!-- 控制面板 -->
            <div class="control-panel">
                <button class="btn btn-success" onclick="startProject()">
                    <i class="fas fa-play"></i> 启动项目
                </button>
                <button class="btn btn-danger" onclick="stopProject()">
                    <i class="fas fa-stop"></i> 停止项目
                </button>
                <button class="btn btn-warning" onclick="restartProject()">
                    <i class="fas fa-redo"></i> 重启项目
                </button>
                <button class="btn btn-primary" onclick="refreshStatus()">
                    <i class="fas fa-sync-alt"></i> 刷新状态
                </button>
            </div>
            
            <!-- 状态面板 -->
            <div class="status-panel">
                <h2><i class="fas fa-chart-line"></i> 项目状态</h2>
                <div class="status-grid" id="statusGrid">
                    <!-- 状态信息将通过JavaScript动态更新 -->
                </div>
            </div>
            
            <!-- 日志面板 -->
            <div class="log-panel">
                <div class="log-controls">
                    <div class="auto-refresh">
                        <label>
                            <input type="checkbox" id="autoRefresh" onchange="toggleAutoRefresh()">
                            <i class="fas fa-clock"></i> 自动刷新 (每5秒)
                        </label>
                    </div>
                    <button class="refresh-btn" onclick="refreshLogs()">
                        <i class="fas fa-sync-alt"></i> 刷新日志
                    </button>
                </div>
                <h3><i class="fas fa-terminal"></i> 实时日志</h3>
                <div class="log-content" id="logContent">
                    <!-- 日志内容将通过JavaScript动态更新 -->
                </div>
            </div>
        </div>
    </div>

    <!-- 浮动操作按钮 -->
    <div class="floating-actions">
        <button class="floating-btn" onclick="scrollToTop()" title="回到顶部">
            <i class="fas fa-arrow-up"></i>
        </button>
        <button class="floating-btn" onclick="refreshAll()" title="刷新所有">
            <i class="fas fa-sync-alt"></i>
        </button>
    </div>

    <!-- 通知提示 -->
    <div id="notification" class="notification">
        <h4 id="notificationTitle"></h4>
        <p id="notificationMessage"></p>
    </div>

    <script>
        let autoRefreshInterval = null;
        let isRefreshing = false;
        
        // 页面加载完成后自动刷新状态
        document.addEventListener('DOMContentLoaded', function() {
            refreshStatus();
            refreshLogs();
            showNotification('欢迎使用', '系统已准备就绪，可以开始操作！', 'success');
        });
        
        // 显示通知
        function showNotification(title, message, type = 'success') {
            const notification = document.getElementById('notification');
            const titleEl = document.getElementById('notificationTitle');
            const messageEl = document.getElementById('notificationMessage');
            
            titleEl.textContent = title;
            messageEl.textContent = message;
            
            notification.className = `notification ${type}`;
            notification.classList.add('show');
            
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }
        
        // 启动项目
        async function startProject() {
            if (isRefreshing) return;
            
            try {
                isRefreshing = true;
                const btn = event.target.closest('.btn');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<span class="loading"></span> 启动中...';
                btn.disabled = true;
                
                const response = await fetch('/api/start', { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    showNotification('启动成功', result.message, 'success');
                    refreshStatus();
                } else {
                    showNotification('启动失败', result.message, 'error');
                }
            } catch (error) {
                showNotification('请求失败', error.message, 'error');
            } finally {
                isRefreshing = false;
                const btn = event.target.closest('.btn');
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }
        
        // 停止项目
        async function stopProject() {
            if (!confirm('确定要停止项目吗？')) return;
            if (isRefreshing) return;
            
            try {
                isRefreshing = true;
                const btn = event.target.closest('.btn');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<span class="loading"></span> 停止中...';
                btn.disabled = true;
                
                const response = await fetch('/api/stop', { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    showNotification('停止成功', result.message, 'success');
                    refreshStatus();
                } else {
                    showNotification('停止失败', result.message, 'error');
                }
            } catch (error) {
                showNotification('请求失败', error.message, 'error');
            } finally {
                isRefreshing = false;
                const btn = event.target.closest('.btn');
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }
        
        // 重启项目
        async function restartProject() {
            if (!confirm('确定要重启项目吗？')) return;
            if (isRefreshing) return;
            
            try {
                isRefreshing = true;
                const btn = event.target.closest('.btn');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<span class="loading"></span> 重启中...';
                btn.disabled = true;
                
                const response = await fetch('/api/restart', { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    showNotification('重启成功', result.message, 'success');
                    refreshStatus();
                } else {
                    showNotification('重启失败', result.message, 'error');
                }
            } catch (error) {
                showNotification('请求失败', error.message, 'error');
            } finally {
                isRefreshing = false;
                const btn = event.target.closest('.btn');
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }
        
        // 刷新状态
        async function refreshStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                updateStatusDisplay(status);
            } catch (error) {
                console.error('刷新状态失败:', error);
            }
        }
        
        // 刷新日志
        async function refreshLogs() {
            try {
                const response = await fetch('/api/logs');
                const logs = await response.json();
                updateLogDisplay(logs);
            } catch (error) {
                console.error('刷新日志失败:', error);
            }
        }
        
        // 更新状态显示
        function updateStatusDisplay(status) {
            const statusGrid = document.getElementById('statusGrid');
            
            const statusItems = [
                {
                    title: '运行状态',
                    icon: 'fas fa-circle',
                    value: status.is_running ? '🟢 运行中' : '🔴 已停止',
                    className: status.is_running ? 'status-running' : 'status-stopped'
                },
                {
                    title: '进程ID',
                    icon: 'fas fa-cog',
                    value: status.pid || 'N/A',
                    className: ''
                },
                {
                    title: '启动时间',
                    icon: 'fas fa-clock',
                    value: status.start_time || 'N/A',
                    className: ''
                },
                {
                    title: '运行时长',
                    icon: 'fas fa-hourglass-half',
                    value: status.uptime || 'N/A',
                    className: ''
                },
                {
                    title: '内存使用',
                    icon: 'fas fa-memory',
                    value: status.memory_usage || 'N/A',
                    className: ''
                },
                {
                    title: 'CPU使用',
                    icon: 'fas fa-microchip',
                    value: status.cpu_usage || 'N/A',
                    className: ''
                }
            ];
            
            statusGrid.innerHTML = statusItems.map(item => `
                <div class="status-item">
                    <h3><i class="${item.icon}"></i> ${item.title}</h3>
                    <div class="status-value ${item.className}">${item.value}</div>
                </div>
            `).join('');
            
            // 显示端口信息
            if (Object.keys(status.ports).length > 0) {
                const portsHtml = Object.entries(status.ports).map(([name, port]) => 
                    `<div class="status-item">
                        <h3><i class="fas fa-network-wired"></i> ${name}</h3>
                        <div class="status-value">${port}</div>
                    </div>`
                ).join('');
                statusGrid.innerHTML += portsHtml;
            }
        }
        
        // 更新日志显示
        function updateLogDisplay(logs) {
            const logContent = document.getElementById('logContent');
            
            if (logs.length === 0) {
                logContent.innerHTML = '<div class="log-line">暂无日志</div>';
                return;
            }
            
            logContent.innerHTML = logs.map(log => 
                `<div class="log-line">${log}</div>`
            ).join('');
            
            // 滚动到底部
            logContent.scrollTop = logContent.scrollHeight;
        }
        
        // 切换自动刷新
        function toggleAutoRefresh() {
            const checkbox = document.getElementById('autoRefresh');
            
            if (checkbox.checked) {
                autoRefreshInterval = setInterval(() => {
                    refreshStatus();
                    refreshLogs();
                }, 5000);
            } else {
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                    autoRefreshInterval = null;
                }
            }
        }
        
        // 回到顶部
        function scrollToTop() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        
        // 刷新所有
        function refreshAll() {
            refreshStatus();
            refreshLogs();
            showNotification('刷新完成', '所有数据已更新', 'success');
        }
        
        // 页面卸载时清理定时器
        window.addEventListener('beforeunload', function() {
            if (autoRefreshInterval) {
                clearInterval(autoRefreshInterval);
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """主页"""
    return HTMLResponse(content=HTML_TEMPLATE.format(PROJECT_NAME=PROJECT_NAME))

@app.post("/api/start")
async def start_project():
    """启动项目API"""
    result = startup_manager.start_project()
    return JSONResponse(content=result)

@app.post("/api/stop")
async def stop_project():
    """停止项目API"""
    result = startup_manager.stop_project()
    return JSONResponse(content=result)

@app.post("/api/restart")
async def restart_project():
    """重启项目API"""
    result = startup_manager.restart_project()
    return JSONResponse(content=result)

@app.get("/api/status")
async def get_status():
    """获取项目状态API"""
    status = startup_manager.get_project_status()
    return JSONResponse(content=status.dict())

@app.get("/api/logs")
async def get_logs():
    """获取日志API"""
    logs = startup_manager.get_log_tail(50)
    return JSONResponse(content=logs)

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": time.time()}

def main():
    """主函数"""
    print(f"🚀 启动 {PROJECT_NAME} 启动器...")
    print(f"📁 项目路径: {PROJECT_ROOT}")
    print(f"🌐 Web界面: http://localhost:8081")
    print(f"📊 健康检查: http://localhost:8081/health")
    print("\n按 Ctrl+C 停止启动器")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8081,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 启动器已停止")

if __name__ == "__main__":
    main()
