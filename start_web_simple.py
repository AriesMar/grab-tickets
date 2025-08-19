#!/usr/bin/env python3
"""
简化版Web启动器 - 一键启动抢票项目
"""

import os
import sys
import time
import subprocess
from pathlib import Path

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

class SimpleStartupManager:
    """简化的项目启动管理器"""
    
    def __init__(self):
        self.process = None
        
    def start_project(self):
        """启动项目"""
        try:
            # 检查是否已在运行
            if self.is_project_running():
                return {"status": "error", "message": "项目已在运行中"}
            
            # 检查虚拟环境
            if not os.path.exists(VENV_ACTIVATE):
                return {"status": "error", "message": "虚拟环境不存在"}
            
            # 启动命令
            cmd = f"source {VENV_ACTIVATE} && python3 {MAIN_SCRIPT}"
            
            # 启动项目
            self.process = subprocess.Popen(
                cmd,
                shell=True,
                executable="/bin/bash",
                cwd=project_root
            )
            
            # 保存PID
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
            
            # 停止进程
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
startup_manager = SimpleStartupManager()

# 简化的HTML界面
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 抢票项目启动器</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            max-width: 500px;
            width: 100%;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .status {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            font-size: 1.2em;
        }
        .running { color: #28a745; }
        .stopped { color: #dc3545; }
        .btn {
            padding: 15px 30px;
            margin: 10px;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .btn-start {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
            color: white;
        }
        .btn-stop {
            background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
            color: white;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        .info {
            margin-top: 30px;
            padding: 20px;
            background: #e3f2fd;
            border-radius: 10px;
            color: #1976d2;
        }
        .info h3 {
            margin-top: 0;
            color: #1565c0;
        }
        .info ul {
            text-align: left;
            margin: 10px 0;
        }
        .info li {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 抢票项目启动器</h1>
        
        <div class="status" id="status">
            正在检查状态...
        </div>
        
        <div>
            <button class="btn btn-start" onclick="startProject()">▶️ 启动项目</button>
            <button class="btn btn-stop" onclick="stopProject()">⏹️ 停止项目</button>
        </div>
        
        <div class="info">
            <h3>📋 使用说明</h3>
            <ul>
                <li>点击"启动项目"开始运行抢票系统</li>
                <li>点击"停止项目"安全停止系统</li>
                <li>系统会自动管理进程和资源</li>
                <li>支持后台运行和状态监控</li>
            </ul>
        </div>
        
        <div class="info">
            <h3>🌐 访问地址</h3>
            <ul>
                <li><strong>抢票系统控制台:</strong> <a href="http://localhost:8080" target="_blank">http://localhost:8080</a></li>
                <li><strong>系统监控指标:</strong> <a href="http://localhost:8001" target="_blank">http://localhost:8001</a></li>
            </ul>
        </div>
    </div>

    <script>
        // 页面加载时检查状态
        document.addEventListener('DOMContentLoaded', function() {
            checkStatus();
            // 每5秒自动刷新状态
            setInterval(checkStatus, 5000);
        });
        
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
            if (isRunning) {
                statusDiv.innerHTML = '🟢 <span class="running">项目正在运行中</span>';
                statusDiv.className = 'status running';
            } else {
                statusDiv.innerHTML = '🔴 <span class="stopped">项目已停止</span>';
                statusDiv.className = 'status stopped';
            }
        }
        
        // 启动项目
        async function startProject() {
            try {
                const response = await fetch('/api/start', { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    alert('✅ ' + result.message);
                    checkStatus();
                } else {
                    alert('❌ ' + result.message);
                }
            } catch (error) {
                alert('❌ 请求失败: ' + error.message);
            }
        }
        
        // 停止项目
        async function stopProject() {
            if (!confirm('确定要停止项目吗？')) return;
            
            try {
                const response = await fetch('/api/stop', { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    alert('✅ ' + result.message);
                    checkStatus();
                } else {
                    alert('❌ ' + result.message);
                }
            } catch (error) {
                alert('❌ 请求失败: ' + error.message);
            }
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
    print("🚀 启动简化版Web启动器...")
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
