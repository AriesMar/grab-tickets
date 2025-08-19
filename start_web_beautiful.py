#!/usr/bin/env python3
"""
美化版Web启动器 - 一键启动抢票项目
使用现代化的设计风格和动画效果
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    print("❌ 缺少依赖包，正在安装...")
    os.system("pip install fastapi uvicorn")
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        import uvicorn
    except ImportError:
        print("❌ 安装失败，请手动运行: pip install fastapi uvicorn")
        sys.exit(1)

# 项目配置
PROJECT_NAME = "抢票项目"
MAIN_SCRIPT = "main.py"
VENV_ACTIVATE = "venv/bin/activate"
PID_FILE = "project.pid"

class BeautifulStartupManager:
    """美化版项目启动管理器"""
    
    def __init__(self):
        self.process = None
        
    def start_project(self):
        """启动项目"""
        try:
            if self.is_project_running():
                return {"status": "error", "message": "项目已在运行中"}
            
            if not os.path.exists(VENV_ACTIVATE):
                return {"status": "error", "message": "虚拟环境不存在"}
            
            cmd = f"source {VENV_ACTIVATE} && python3 {MAIN_SCRIPT}"
            
            self.process = subprocess.Popen(
                cmd,
                shell=True,
                executable="/bin/bash",
                cwd=project_root
            )
            
            with open(PID_FILE, 'w') as f:
                f.write(str(self.process.pid))
            
            time.sleep(2)
            
            if self.process.poll() is None:
                return {"status": "success", "message": f"项目启动成功！PID: {self.process.pid}"}
            else:
                return {"status": "error", "message": "项目启动失败"}
                
        except Exception as e:
            return {"status": "error", "message": f"启动失败: {str(e)}"}
    
    def stop_project(self):
        """停止项目"""
        try:
            if not self.is_project_running():
                return {"status": "error", "message": "项目未在运行"}
            
            if os.path.exists(PID_FILE):
                with open(PID_FILE, 'r') as f:
                    pid = f.read().strip()
                os.system(f"kill {pid}")
                os.remove(PID_FILE)
            
            return {"status": "success", "message": "项目已停止"}
            
        except Exception as e:
            return {"status": "error", "message": f"停止失败: {str(e)}"}
    
    def is_project_running(self):
        """检查项目是否在运行"""
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, 'r') as f:
                    pid = f.read().strip()
                return os.path.exists(f"/proc/{pid}") if os.name == 'posix' else True
            except:
                pass
        return False

# 创建FastAPI应用
app = FastAPI(title=f"{PROJECT_NAME} 启动器", version="1.0.0")

# 创建启动管理器
startup_manager = BeautifulStartupManager()

# 美化版HTML界面
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 抢票项目启动器</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --success: #56ab2f;
            --danger: #ff416c;
            --warning: #f093fb;
            --info: #4facfe;
            --dark: #2d3748;
            --light: #f8f9fa;
            --white: #ffffff;
            --shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            --radius: 20px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
            overflow-x: hidden;
        }

        .container {
            max-width: 600px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            overflow: hidden;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header {
            background: linear-gradient(135deg, var(--info) 0%, #00f2fe 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
            position: relative;
        }

        .header h1 {
            font-size: 2.8em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .content {
            padding: 40px;
        }

        .status {
            background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
            border-radius: 15px;
            padding: 25px;
            margin: 25px 0;
            text-align: center;
            font-size: 1.3em;
            border: 1px solid rgba(0, 0, 0, 0.05);
            position: relative;
            overflow: hidden;
        }

        .status::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }

        .running { 
            color: var(--success);
            animation: pulse 2s infinite;
        }

        .stopped { 
            color: var(--danger);
        }

        .btn {
            padding: 18px 35px;
            margin: 15px;
            border: none;
            border-radius: 15px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            text-transform: uppercase;
            letter-spacing: 1px;
            position: relative;
            overflow: hidden;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            min-width: 160px;
            justify-content: center;
        }

        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }

        .btn:hover::before {
            left: 100%;
        }

        .btn-start {
            background: linear-gradient(135deg, var(--success) 0%, #a8e6cf 100%);
            color: white;
        }

        .btn-stop {
            background: linear-gradient(135deg, var(--danger) 0%, #ff4b2b 100%);
            color: white;
        }

        .btn:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .info {
            margin-top: 35px;
            padding: 25px;
            background: #e3f2fd;
            border-radius: 15px;
            color: #1976d2;
            border: 1px solid rgba(25, 118, 210, 0.2);
        }

        .info h3 {
            margin-bottom: 15px;
            color: #1565c0;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .info ul {
            text-align: left;
            margin: 15px 0;
            padding-left: 20px;
        }

        .info li {
            margin: 8px 0;
            line-height: 1.5;
        }

        .floating-actions {
            position: fixed;
            bottom: 30px;
            right: 30px;
            display: flex;
            flex-direction: column;
            gap: 15px;
            z-index: 1000;
        }

        .floating-btn {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            font-size: 1.5em;
            cursor: pointer;
            transition: var(--transition);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .floating-btn:hover {
            transform: scale(1.1) rotate(5deg);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }

        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
            border-left: 5px solid var(--success);
            transform: translateX(400px);
            transition: var(--transition);
            z-index: 1001;
            max-width: 350px;
        }

        .notification.show {
            transform: translateX(0);
        }

        .notification.success {
            border-left-color: var(--success);
        }

        .notification.error {
            border-left-color: var(--danger);
        }

        .notification h4 {
            margin-bottom: 8px;
            color: #333;
        }

        .notification p {
            color: #666;
            margin: 0;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 2.2em;
            }
            .content {
                padding: 25px;
            }
            .btn {
                margin: 10px;
                padding: 15px 25px;
                min-width: 140px;
            }
            .floating-actions {
                bottom: 20px;
                right: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-rocket"></i> 抢票项目启动器</h1>
            <p>🚀 一键启动、停止和管理你的抢票系统</p>
        </div>
        
        <div class="content">
            <div class="status" id="status">
                <i class="fas fa-spinner fa-spin"></i> 正在检查状态...
            </div>
            
            <div style="text-align: center;">
                <button class="btn btn-start" onclick="startProject()" id="startBtn">
                    <i class="fas fa-play"></i> 启动项目
                </button>
                <button class="btn btn-stop" onclick="stopProject()" id="stopBtn">
                    <i class="fas fa-stop"></i> 停止项目
                </button>
            </div>
            
            <div class="info">
                <h3><i class="fas fa-info-circle"></i> 使用说明</h3>
                <ul>
                    <li>点击"启动项目"开始运行抢票系统</li>
                    <li>点击"停止项目"安全停止系统</li>
                    <li>系统会自动管理进程和资源</li>
                    <li>支持后台运行和状态监控</li>
                </ul>
            </div>
            
            <div class="info">
                <h3><i class="fas fa-globe"></i> 访问地址</h3>
                <ul>
                    <li><strong>抢票系统控制台:</strong> <a href="http://localhost:8080" target="_blank">http://localhost:8080</a></li>
                    <li><strong>系统监控指标:</strong> <a href="http://localhost:8001" target="_blank">http://localhost:8001</a></li>
                </ul>
            </div>
        </div>
    </div>

    <!-- 浮动操作按钮 -->
    <div class="floating-actions">
        <button class="floating-btn" onclick="scrollToTop()" title="回到顶部">
            <i class="fas fa-arrow-up"></i>
        </button>
        <button class="floating-btn" onclick="refreshStatus()" title="刷新状态">
            <i class="fas fa-sync-alt"></i>
        </button>
    </div>

    <!-- 通知提示 -->
    <div id="notification" class="notification">
        <h4 id="notificationTitle"></h4>
        <p id="notificationMessage"></p>
    </div>

    <script>
        let isRefreshing = false;
        
        // 页面加载时检查状态
        document.addEventListener('DOMContentLoaded', function() {
            checkStatus();
            // 每5秒自动刷新状态
            setInterval(checkStatus, 5000);
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
        
        // 检查项目状态
        async function checkStatus() {
            try {
                const response = await fetch('/api/status');
                const result = await response.json();
                updateStatus(result.is_running);
            } catch (error) {
                console.error('检查状态失败:', error);
            }
        }
        
        // 更新状态显示
        function updateStatus(isRunning) {
            const statusDiv = document.getElementById('status');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            
            if (isRunning) {
                statusDiv.innerHTML = '<i class="fas fa-check-circle"></i> <span class="running">项目正在运行中</span>';
                statusDiv.className = 'status running';
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } else {
                statusDiv.innerHTML = '<i class="fas fa-times-circle"></i> <span class="stopped">项目已停止</span>';
                statusDiv.className = 'status stopped';
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        }
        
        // 启动项目
        async function startProject() {
            if (isRefreshing) return;
            
            try {
                isRefreshing = true;
                const btn = document.getElementById('startBtn');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<span class="loading"></span> 启动中...';
                btn.disabled = true;
                
                const response = await fetch('/api/start', { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    showNotification('启动成功', result.message, 'success');
                    checkStatus();
                } else {
                    showNotification('启动失败', result.message, 'error');
                }
            } catch (error) {
                showNotification('请求失败', error.message, 'error');
            } finally {
                isRefreshing = false;
                const btn = document.getElementById('startBtn');
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
                const btn = document.getElementById('stopBtn');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<span class="loading"></span> 停止中...';
                btn.disabled = true;
                
                const response = await fetch('/api/stop', { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    showNotification('停止成功', result.message, 'success');
                    checkStatus();
                } else {
                    showNotification('停止失败', result.message, 'error');
                }
            } catch (error) {
                showNotification('请求失败', error.message, 'error');
            } finally {
                isRefreshing = false;
                const btn = document.getElementById('stopBtn');
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }
        
        // 回到顶部
        function scrollToTop() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        
        // 刷新状态
        function refreshStatus() {
            checkStatus();
            showNotification('刷新完成', '状态已更新', 'success');
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """主页"""
    return HTMLResponse(content=HTML_CONTENT)

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

@app.get("/api/status")
async def get_status():
    """获取项目状态API"""
    is_running = startup_manager.is_project_running()
    return JSONResponse(content={"is_running": is_running})

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": time.time()}

def main():
    """主函数"""
    print("🚀 启动美化版Web启动器...")
    print("📁 项目路径:", project_root)
    print("🌐 Web界面: http://localhost:8081")
    print("📊 健康检查: http://localhost:8081/health")
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
