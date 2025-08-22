"""
轻量 Web 控制台：
 - 查看健康检查、设置、任务列表、ADB 设备
 - 手动触发任务执行（立即执行）
"""
from typing import List, Optional
import threading
import asyncio
import subprocess
import time

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.websockets import WebSocket
from pydantic import BaseModel

from ..data.config_manager import ConfigManager
from ..data.models import TicketRequest
from ..core.grab_engine import GrabEngine
from ..utils.logger import get_logger
from ..utils.log_buffer import install_log_sink, get_latest_text, subscribe_queue, unsubscribe_queue
from ..utils.result_buffer import latest as latest_results, summary as results_summary


logger = get_logger("web_console")


class TriggerBody(BaseModel):
    event_ids: Optional[List[str]] = None


def _auth_dep(config_manager: ConfigManager):
    def _inner(token: str | None = None):
        expected = config_manager.get_setting("web_token", None)
        if expected and token != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return _inner


def create_app(config_manager: ConfigManager, grab_engine: GrabEngine) -> FastAPI:
    app = FastAPI(title="Grab Tickets Web Console")
    auth = _auth_dep(config_manager)
    install_log_sink()

    def _token_qs(token: str | None) -> str:
        return f"?token={token}" if token else ""

    @app.get("/", response_class=HTMLResponse)
    def index(token: None = Depends(auth)):
        t = config_manager.get_setting("web_token", "")
        qs = _token_qs(t)
        
        # 获取系统状态数据
        tasks = config_manager.load_ticket_requests()
        out = get_devices(token)
        output = out.get("output") if isinstance(out, dict) else str(out)
        devices = []
        if output:
            lines = output.strip().split('\n')
            for line in lines[1:]:
                if line.strip() and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        device_id = parts[0].strip()
                        status = parts[1].strip()
                        devices.append({
                            'id': device_id,
                            'status': status,
                            'is_online': 'device' in status.lower()
                        })
        
        # 获取结果统计
        results_summary_data = results_summary()
        
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🎫 Grab Tickets 后台管理中心</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                :root {{
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
                }}

                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    min-height: 100vh;
                    color: #333;
                    overflow-x: hidden;
                }}

                .sidebar {{
                    position: fixed;
                    left: 0;
                    top: 0;
                    width: 280px;
                    height: 100vh;
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(20px);
                    border-right: 1px solid rgba(255, 255, 255, 0.2);
                    z-index: 1000;
                    transition: var(--transition);
                    overflow-y: auto;
                }}

                .sidebar-header {{
                    background: linear-gradient(135deg, var(--dark) 0%, #1a202c 100%);
                    color: white;
                    padding: 30px 20px;
                    text-align: center;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }}

                .sidebar-header h1 {{
                    font-size: 1.8em;
                    margin-bottom: 10px;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                }}

                .sidebar-header p {{
                    font-size: 0.9em;
                    opacity: 0.8;
                }}

                .nav-menu {{
                    padding: 20px 0;
                }}

                .nav-item {{
                    margin: 5px 20px;
                }}

                .nav-link {{
                    display: flex;
                    align-items: center;
                    gap: 15px;
                    padding: 15px 20px;
                    color: #555;
                    text-decoration: none;
                    border-radius: 12px;
                    transition: var(--transition);
                    font-weight: 500;
                }}

                .nav-link:hover, .nav-link.active {{
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    color: white;
                    transform: translateX(5px);
                }}

                .nav-link i {{
                    font-size: 1.2em;
                    width: 20px;
                    text-align: center;
                }}

                .main-content {{
                    margin-left: 280px;
                    padding: 20px;
                    min-height: 100vh;
                }}

                .top-bar {{
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(20px);
                    border-radius: var(--radius);
                    padding: 20px 30px;
                    margin-bottom: 30px;
                    box-shadow: var(--shadow);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    flex-wrap: wrap;
                    gap: 20px;
                }}

                .page-title {{
                    font-size: 2.2em;
                    color: #333;
                    font-weight: 600;
                }}

                .status-indicator {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    font-size: 0.9em;
                    color: #666;
                }}

                .status-dot {{
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background: var(--success);
                    animation: pulse 2s infinite;
                }}

                @keyframes pulse {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0.5; }}
                    100% {{ opacity: 1; }}
                }}

                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 25px;
                    margin-bottom: 30px;
                }}

                .dashboard-card {{
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(20px);
                    border-radius: var(--radius);
                    padding: 25px;
                    box-shadow: var(--shadow);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    transition: var(--transition);
                }}

                .dashboard-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
                }}

                .card-header {{
                    display: flex;
                    align-items: center;
                    gap: 15px;
                    margin-bottom: 20px;
                }}

                .card-icon {{
                    width: 50px;
                    height: 50px;
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.5em;
                    color: white;
                }}

                .card-title {{
                    font-size: 1.3em;
                    font-weight: 600;
                    color: #333;
                }}

                .card-content {{
                    margin-bottom: 20px;
                }}

                .stat-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 15px;
                }}

                .stat-item {{
                    text-align: center;
                    padding: 15px;
                    background: rgba(0, 0, 0, 0.03);
                    border-radius: 10px;
                }}

                .stat-value {{
                    font-size: 1.8em;
                    font-weight: 700;
                    color: var(--primary);
                    margin-bottom: 5px;
                }}

                .stat-label {{
                    font-size: 0.85em;
                    color: #666;
                    font-weight: 500;
                }}

                .card-actions {{
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                }}

                .btn {{
                    padding: 10px 20px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: var(--transition);
                    font-size: 0.9em;
                    font-weight: 600;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    color: white;
                    text-decoration: none;
                    flex: 1;
                    justify-content: center;
                }}

                .btn-primary {{
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                }}

                .btn-success {{
                    background: linear-gradient(135deg, var(--success) 0%, #a8e6cf 100%);
                }}

                .btn-warning {{
                    background: linear-gradient(135deg, var(--warning) 0%, #f5576c 100%);
                }}

                .btn-info {{
                    background: linear-gradient(135deg, var(--info) 0%, #00f2fe 100%);
                }}

                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
                }}

                .content-section {{
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(20px);
                    border-radius: var(--radius);
                    padding: 30px;
                    box-shadow: var(--shadow);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    margin-bottom: 30px;
                }}

                .section-header {{
                    display: flex;
                    align-items: center;
                    gap: 15px;
                    margin-bottom: 25px;
                }}

                .section-title {{
                    font-size: 1.5em;
                    font-weight: 600;
                    color: #333;
                }}

                .table-container {{
                    overflow-x: auto;
                    border-radius: 10px;
                    border: 1px solid #e1e5e9;
                }}

                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9em;
                }}

                .data-table th {{
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    color: white;
                    padding: 15px 12px;
                    text-align: left;
                    font-weight: 600;
                    border-radius: 10px;
                }}

                .data-table td {{
                    padding: 12px;
                    border-bottom: 1px solid #eee;
                    vertical-align: middle;
                }}

                .data-table tr:hover {{
                    background: #f8f9fa;
                }}

                .badge {{
                    padding: 4px 8px;
                    border-radius: 20px;
                    font-size: 0.8em;
                    font-weight: 600;
                    color: white;
                }}

                .badge-success {{
                    background: var(--success);
                }}

                .badge-danger {{
                    background: var(--danger);
                }}

                .badge-warning {{
                    background: var(--warning);
                }}

                .badge-info {{
                    background: var(--info);
                }}

                .mobile-toggle {{
                    display: none;
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 1001;
                    background: var(--primary);
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 1.2em;
                }}

                @media (max-width: 1024px) {{
                    .sidebar {{
                        transform: translateX(-100%);
                    }}
                    
                    .sidebar.open {{
                        transform: translateX(0);
                    }}
                    
                    .main-content {{
                        margin-left: 0;
                    }}
                    
                    .mobile-toggle {{
                        display: block;
                    }}
                }}

                @media (max-width: 768px) {{
                    .dashboard-grid {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .stat-grid {{
                        grid-template-columns: repeat(2, 1fr);
                    }}
                    
                    .top-bar {{
                        flex-direction: column;
                        text-align: center;
                    }}
                    
                    .page-title {{
                        font-size: 1.8em;
                    }}
                }}

                .hidden {{
                    display: none;
                }}
            </style>
        </head>
        <body>
            <!-- 移动端菜单切换按钮 -->
            <button class="mobile-toggle" onclick="toggleSidebar()">
                <i class="fas fa-bars"></i>
            </button>

            <!-- 侧边栏 -->
            <div class="sidebar" id="sidebar">
                <div class="sidebar-header">
                    <h1><i class="fas fa-ticket-alt"></i> Grab Tickets</h1>
                    <p>后台管理中心</p>
                </div>
                
                <nav class="nav-menu">
                    <div class="nav-item">
                        <a href="#dashboard" class="nav-link active" onclick="showSection('dashboard')">
                            <i class="fas fa-tachometer-alt"></i>
                            <span>仪表盘</span>
                        </a>
                    </div>
                    <div class="nav-item">
                        <a href="#tasks" class="nav-link" onclick="showSection('tasks')">
                            <i class="fas fa-tasks"></i>
                            <span>任务管理</span>
                        </a>
                    </div>
                    <div class="nav-item">
                        <a href="#devices" class="nav-link" onclick="showSection('devices')">
                            <i class="fas fa-mobile-alt"></i>
                            <span>设备管理</span>
                        </a>
                    </div>
                    <div class="nav-item">
                        <a href="#results" class="nav-link" onclick="showSection('results')">
                            <i class="fas fa-chart-bar"></i>
                            <span>结果查看</span>
                        </a>
                    </div>
                    <div class="nav-item">
                        <a href="#logs" class="nav-link" onclick="showSection('logs')">
                            <i class="fas fa-terminal"></i>
                            <span>系统日志</span>
                        </a>
                        </a>
                    </div>
                    <div class="nav-item">
                        <a href="#settings" class="nav-link" onclick="showSection('settings')">
                            <i class="fas fa-cog"></i>
                            <span>系统设置</span>
                        </a>
                    </div>
                </nav>
            </div>

            <!-- 主内容区域 -->
            <div class="main-content">
                <!-- 顶部状态栏 -->
                <div class="top-bar">
                    <div>
                        <h1 class="page-title" id="pageTitle">仪表盘</h1>
                        <div class="status-indicator">
                            <div class="status-dot"></div>
                            <span>系统运行中</span>
                        </div>
                    </div>
                    <div>
                        <span style="color: #666; font-size: 0.9em;">
                            <i class="fas fa-clock"></i> {time.strftime('%Y-%m-%d %H:%M:%S')}
                        </span>
                    </div>
                </div>

                <!-- 仪表盘 -->
                <div id="dashboard" class="content-section">
                    <div class="dashboard-grid">
                        <!-- 任务统计卡片 -->
                        <div class="dashboard-card">
                            <div class="card-header">
                                <div class="card-icon" style="background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);">
                                    <i class="fas fa-tasks"></i>
                                </div>
                                <div class="card-title">任务统计</div>
                            </div>
                            <div class="card-content">
                                <div class="stat-grid">
                                    <div class="stat-item">
                                        <div class="stat-value">{len(tasks)}</div>
                                        <div class="stat-label">总任务</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value">{len([t for t in tasks if t.auto_start])}</div>
                                        <div class="stat-label">自动任务</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value">{len([t for t in tasks if not t.auto_start])}</div>
                                        <div class="stat-label">手动任务</div>
                                    </div>
                                </div>
                            </div>
                            <div class="card-actions">
                                <a href="#tasks" class="btn btn-primary" onclick="showSection('tasks')">
                                    <i class="fas fa-edit"></i> 管理任务
                                </a>
                            </div>
                        </div>

                        <!-- 设备统计卡片 -->
                        <div class="dashboard-card">
                            <div class="card-header">
                                <div class="card-icon" style="background: linear-gradient(135deg, var(--info) 0%, #00f2fe 100%);">
                                    <i class="fas fa-mobile-alt"></i>
                                </div>
                                <div class="card-title">设备状态</div>
                            </div>
                            <div class="card-content">
                                <div class="stat-grid">
                                    <div class="stat-item">
                                        <div class="stat-value">{len(devices)}</div>
                                        <div class="stat-label">总设备</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value">{len([d for d in devices if d['is_online']])}</div>
                                        <div class="stat-label">在线</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value">{len([d for d in devices if not d['is_online']])}</div>
                                        <div class="stat-label">离线</div>
                                    </div>
                                </div>
                            </div>
                            <div class="card-actions">
                                <a href="#devices" class="btn btn-info" onclick="showSection('devices')">
                                    <i class="fas fa-cog"></i> 管理设备
                                </a>
                            </div>
                        </div>

                        <!-- 结果统计卡片 -->
                        <div class="dashboard-card">
                            <div class="card-header">
                                <div class="card-icon" style="background: linear-gradient(135deg, var(--success) 0%, #a8e6cf 100%);">
                                    <i class="fas fa-chart-bar"></i>
                                </div>
                                <div class="card-title">执行结果</div>
                            </div>
                            <div class="card-content">
                                <div class="stat-grid">
                                    <div class="stat-item">
                                        <div class="stat-value">{results_summary_data.get('total', 0)}</div>
                                        <div class="stat-label">总执行</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value">{results_summary_data.get('success', 0)}</div>
                                        <div class="stat-label">成功</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value">{results_summary_data.get('success_rate', 0):.1%}</div>
                                        <div class="stat-label">成功率</div>
                                    </div>
                                </div>
                            </div>
                            <div class="card-actions">
                                <a href="#results" class="btn btn-success" onclick="showSection('results')">
                                    <i class="fas fa-eye"></i> 查看详情
                                </a>
                            </div>
                        </div>

                        <!-- 快速操作卡片 -->
                        <div class="dashboard-card">
                            <div class="card-header">
                                <div class="card-icon" style="background: linear-gradient(135deg, var(--warning) 0%, #f5576c 100%);">
                                    <i class="fas fa-bolt"></i>
                                </div>
                                <div class="card-title">快速操作</div>
                            </div>
                            <div class="card-content">
                                <p style="color: #666; margin-bottom: 15px;">常用功能快速访问</p>
                            </div>
                            <div class="card-actions">
                                <button class="btn btn-warning" onclick="triggerAllTasks()">
                                    <i class="fas fa-play"></i> 触发全部
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 任务管理 -->
                <div id="tasks" class="content-section hidden">
                    <div class="section-header">
                        <i class="fas fa-tasks" style="color: var(--primary); font-size: 1.5em;"></i>
                        <h2 class="section-title">任务管理</h2>
                    </div>
                    
                    <div class="card-actions" style="margin-bottom: 20px;">
                        <button class="btn btn-success" onclick="showAddTaskForm()">
                            <i class="fas fa-plus"></i> 新增任务
                        </button>
                        <button class="btn btn-primary" onclick="triggerAllTasks()">
                            <i class="fas fa-play"></i> 触发全部
                        </button>
                    </div>
                    
                    <div class="table-container">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>活动ID</th>
                                    <th>平台</th>
                                    <th>场次ID</th>
                                    <th>设备ID</th>
                                    <th>数量</th>
                                    <th>自动开始</th>
                                    <th>状态</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody>
                                {('\\n'.join([f'''
                                    <tr>
                                        <td>{i}</td>
                                        <td>{task.event_id}</td>
                                        <td><span class="badge badge-info">{task.platform.value}</span></td>
                                        <td>{task.performance_id or '-'}</td>
                                        <td>{task.device_id or '-'}</td>
                                        <td>{task.quantity}</td>
                                        <td><span class="badge {'badge-success' if task.auto_start else 'badge-warning'}">{'是' if task.auto_start else '否'}</span></td>
                                        <td><span class="badge badge-info">待执行</span></td>
                                        <td>
                                            <button class="btn btn-primary btn-sm" onclick="triggerTask('{task.event_id}')" style="padding: 5px 10px; font-size: 0.8em;">
                                                <i class="fas fa-play"></i>
                                            </button>
                                            <button class="btn btn-warning btn-sm" onclick="editTask('{task.event_id}')" style="padding: 5px 10px; font-size: 0.8em;">
                                                <i class="fas fa-edit"></i>
                                            </button>
                                            <button class="btn btn-danger btn-sm" onclick="deleteTask('{task.event_id}', '{task.platform.value}')" style="padding: 5px 10px; font-size: 0.8em;">
                                                <i class="fas fa-trash"></i>
                                            </button>
                                        </td>
                                    </tr>
                                ''' for i, task in enumerate(tasks, 1)]) if tasks else '<tr><td colspan="9" style="text-align: center; padding: 40px; color: #666;">暂无任务</td></tr>')}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- 设备管理 -->
                <div id="devices" class="content-section hidden">
                    <div class="section-header">
                        <i class="fas fa-mobile-alt" style="color: var(--info); font-size: 1.5em;"></i>
                        <h2 class="section-title">设备管理</h2>
                    </div>
                    
                    <div class="card-actions" style="margin-bottom: 20px;">
                        <button class="btn btn-info" onclick="refreshDevices()">
                            <i class="fas fa-sync-alt"></i> 刷新设备
                        </button>
                    </div>
                    
                    <div class="table-container">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>设备ID</th>
                                    <th>状态</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody>
                                {('\\n'.join([f'''
                                    <tr>
                                        <td><code>{device['id']}</code></td>
                                        <td><span class="badge {'badge-success' if device['is_online'] else 'badge-danger'}">{device['status']}</span></td>
                                        <td>
                                            <button class="btn btn-danger btn-sm" onclick="disconnectDevice('{device['id']}')" style="padding: 5px 10px; font-size: 0.8em;">
                                                <i class="fas fa-unlink"></i> 断开
                                            </button>
                                        </td>
                                    </tr>
                                ''' for device in devices]) if devices else '<tr><td colspan="3" style="text-align: center; padding: 40px; color: #666;">暂无设备连接</td></tr>')}
                            </tbody>
                        </table>
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <h3 style="margin-bottom: 15px; color: #333;">
                            <i class="fas fa-wifi"></i> Wi-Fi ADB 连接
                        </h3>
                        <div style="display: flex; gap: 15px; align-items: end; flex-wrap: wrap;">
                            <div style="flex: 1; min-width: 250px;">
                                <label style="display: block; margin-bottom: 8px; font-weight: 600; color: #555;">设备IP地址和端口</label>
                                <input type="text" id="deviceIP" placeholder="192.168.1.23:5555" style="width: 100%; padding: 10px; border: 2px solid #e1e5e9; border-radius: 8px; font-size: 0.9em;">
                            </div>
                            <button class="btn btn-info" onclick="connectDevice()">
                                <i class="fas fa-plug"></i> 连接设备
                            </button>
                        </div>
                    </div>
                </div>

                <!-- 结果查看 -->
                <div id="results" class="content-section hidden">
                    <div class="section-header">
                        <i class="fas fa-chart-bar" style="color: var(--success); font-size: 1.5em;"></i>
                        <h2 class="section-title">结果查看</h2>
                    </div>
                    
                    <div class="card-actions" style="margin-bottom: 20px;">
                        <button class="btn btn-success" onclick="exportResults()">
                            <i class="fas fa-download"></i> 导出数据
                        </button>
                    </div>
                    
                    <div class="table-container">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>时间</th>
                                    <th>活动ID</th>
                                    <th>平台</th>
                                    <th>设备ID</th>
                                    <th>状态</th>
                                    <th>订单ID</th>
                                    <th>消息</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td colspan="7" style="text-align: center; padding: 40px; color: #666;">
                                        结果数据加载中...
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- 系统日志 -->
                <div id="logs" class="content-section hidden">
                    <div class="section-header">
                        <i class="fas fa-terminal" style="color: var(--dark); font-size: 1.5em;"></i>
                        <h2 class="section-title">系统日志</h2>
                    </div>
                    
                    <div class="card-actions" style="margin-bottom: 20px;">
                        <button class="btn btn-primary" onclick="refreshLogs()">
                            <i class="fas fa-sync-alt"></i> 刷新日志
                        </button>
                        <button class="btn btn-warning" onclick="clearLogs()">
                            <i class="fas fa-trash"></i> 清空显示
                        </button>
                    </div>
                    
                    <div style="background: #0f1419; border-radius: 12px; padding: 20px; max-height: 500px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 0.9em; line-height: 1.6; color: #e2e8f0;">
                        <div id="logsContent">
                            {logs.replace(chr(10), '<br>') if logs else '<div style="color: #666;">暂无日志</div>'}
                        </div>
                    </div>
                </div>

                <!-- 系统设置 -->
                <div id="settings" class="content-section hidden">
                    <div class="section-header">
                        <i class="fas fa-cog" style="color: var(--warning); font-size: 1.5em;"></i>
                        <h2 class="section-title">系统设置</h2>
                    </div>
                    
                    <div style="color: #666; text-align: center; padding: 40px;">
                        <i class="fas fa-cog" style="font-size: 3em; margin-bottom: 20px; display: block; color: var(--warning);"></i>
                        <p>系统设置功能开发中...</p>
                        <p style="font-size: 0.9em; margin-top: 10px;">未来版本将支持更多配置选项</p>
                    </div>
                </div>
            </div>

            <script>
                // 显示指定部分
                function showSection(sectionId) {{
                    // 隐藏所有部分
                    document.querySelectorAll('.content-section').forEach(section => {{
                        section.classList.add('hidden');
                    }});
                    
                    // 显示指定部分
                    document.getElementById(sectionId).classList.remove('hidden');
                    
                    // 更新页面标题
                    const titles = {{
                        'dashboard': '仪表盘',
                        'tasks': '任务管理',
                        'devices': '设备管理',
                        'results': '结果查看',
                        'logs': '系统日志',
                        'settings': '系统设置'
                    }};
                    document.getElementById('pageTitle').textContent = titles[sectionId] || '管理中心';
                    
                    // 更新导航状态
                    document.querySelectorAll('.nav-link').forEach(link => {{
                        link.classList.remove('active');
                    }});
                    event.target.classList.add('active');
                }}
                
                // 切换侧边栏（移动端）
                function toggleSidebar() {{
                    const sidebar = document.getElementById('sidebar');
                    sidebar.classList.toggle('open');
                }}
                
                // 触发任务
                function triggerTask(eventId) {{
                    if (confirm(`确定要立即执行任务 "${{eventId}}" 吗？`)) {{
                        // 这里添加触发任务的逻辑
                        alert('任务触发功能需要后端API支持');
                    }}
                }}
                
                // 编辑任务
                function editTask(eventId) {{
                    alert('编辑任务功能需要后端API支持');
                }}
                
                // 删除任务
                function deleteTask(eventId, platform) {{
                    if (confirm(`确定要删除任务 "${{eventId}}" 吗？此操作不可恢复！`)) {{
                        // 这里添加删除任务的逻辑
                        alert('删除任务功能需要后端API支持');
                    }}
                }}
                
                // 触发全部任务
                function triggerAllTasks() {{
                    if (confirm('确定要触发全部任务吗？')) {{
                        // 这里添加触发全部任务的逻辑
                        alert('触发全部任务功能需要后端API支持');
                    }}
                }}
                
                // 刷新设备
                function refreshDevices() {{
                    window.location.reload();
                }}
                
                // 断开设备
                function disconnectDevice(deviceId) {{
                    if (confirm(`确定要断开设备 "${{deviceId}}" 吗？`)) {{
                        alert('断开设备功能需要后端API支持');
                    }}
                }}
                
                // 连接设备
                function connectDevice() {{
                    const deviceIP = document.getElementById('deviceIP').value;
                    if (!deviceIP) {{
                        alert('请输入设备IP地址和端口');
                        return;
                    }}
                    // 这里添加连接设备的逻辑
                    alert('连接设备功能需要后端API支持');
                }}
                
                // 导出结果
                function exportResults() {{
                    alert('导出结果功能需要后端API支持');
                }}
                
                // 刷新日志
                function refreshLogs() {{
                    window.location.reload();
                }}
                
                // 清空日志
                function clearLogs() {{
                    if (confirm('确定要清空日志显示吗？')) {{
                        document.getElementById('logsContent').innerHTML = '';
                    }}
                }}
                
                // 显示新增任务表单
                function showAddTaskForm() {{
                    alert('新增任务功能需要后端API支持');
                }}
                
                // 页面加载完成后初始化
                document.addEventListener('DOMContentLoaded', function() {{
                    // 默认显示仪表盘
                    showSection('dashboard');
                }});
            </script>
        </body>
        </html>
        """
        return HTMLResponse(html)

    @app.get("/health")
    def health(token: None = Depends(auth)):
        return {"status": "ok"}

    @app.get("/settings")
    def get_settings(token: None = Depends(auth)):
        return config_manager.load_settings()

    @app.get("/tasks")
    def get_tasks(token: None = Depends(auth)):
        tasks: List[TicketRequest] = config_manager.load_ticket_requests()
        return [t.dict() for t in tasks]

    @app.get("/tasks/html", response_class=HTMLResponse)
    def get_tasks_html(token: None = Depends(auth)):
        tasks: List[TicketRequest] = config_manager.load_ticket_requests()
        t = config_manager.get_setting("web_token", "")
        qs = _token_qs(t)
        
        # 生成任务表格行
        rows = []
        for i, r in enumerate(tasks, 1):
            status_class = "status-running" if r.auto_start else "status-stopped"
            status_text = "🟢 自动" if r.auto_start else "🔴 手动"
            
            rows.append(f"""
                <tr class="task-row">
                    <td class="task-id">{i}</td>
                    <td class="event-id">{r.event_id}</td>
                    <td class="platform"><span class="platform-badge platform-{r.platform.value}">{r.platform.value}</span></td>
                    <td class="performance">{r.performance_id or '-'}</td>
                    <td class="keywords">{', '.join(r.performance_keywords) if r.performance_keywords else '-'}</td>
                    <td class="device">{r.device_id or '-'}</td>
                    <td class="quantity">{r.quantity}</td>
                    <td class="price">{r.target_price or '-'}</td>
                    <td class="auto-start"><span class="{status_class}">{status_text}</span></td>
                    <td class="start-time">{r.start_time or (f"+{r.start_offset_seconds}s" if r.start_offset_seconds else '-')}</td>
                    <td class="actions">
                        <div class="action-buttons">
                            <button class="btn btn-primary btn-sm" onclick="triggerTask('{r.event_id}')" title="立即执行">
                                <i class="fas fa-play"></i>
                            </button>
                            <button class="btn btn-warning btn-sm" onclick="editTask('{r.event_id}')" title="编辑任务">
                                <i class="fas fa-edit"></i>
                            </button>
                            <button class="btn btn-danger btn-sm" onclick="deleteTask('{r.event_id}', '{r.platform.value}')" title="删除任务">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </td>
                </tr>
            """)
        
        # 美化版HTML模板
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>📋 任务管理 - Grab Tickets</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                :root {{
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
                }}

                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    min-height: 100vh;
                    padding: 20px;
                    color: #333;
                }}

                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: var(--radius);
                    box-shadow: var(--shadow);
                    overflow: hidden;
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}

                .header {{
                    background: linear-gradient(135deg, var(--info) 0%, #00f2fe 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}

                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                }}

                .header p {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}

                .content {{
                    padding: 30px;
                }}

                .back-btn {{
                    display: inline-block;
                    padding: 10px 20px;
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                    text-decoration: none;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    transition: var(--transition);
                }}

                .back-btn:hover {{
                    background: rgba(255, 255, 255, 0.3);
                    transform: translateY(-2px);
                }}

                .stats-bar {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}

                .stat-card {{
                    background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
                    padding: 20px;
                    border-radius: 15px;
                    text-align: center;
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }}

                .stat-card h3 {{
                    color: var(--primary);
                    font-size: 2em;
                    margin-bottom: 5px;
                }}

                .stat-card p {{
                    color: #666;
                    font-size: 0.9em;
                }}

                .table-container {{
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
                    overflow-x: auto;
                }}

                .tasks-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9em;
                }}

                .tasks-table th {{
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    color: white;
                    padding: 15px 10px;
                    text-align: left;
                    font-weight: 600;
                    border-radius: 10px;
                }}

                .tasks-table td {{
                    padding: 12px 10px;
                    border-bottom: 1px solid #eee;
                    vertical-align: middle;
                }}

                .tasks-table tr:hover {{
                    background: #f8f9fa;
                }}

                .platform-badge {{
                    padding: 4px 8px;
                    border-radius: 20px;
                    font-size: 0.8em;
                    font-weight: 600;
                    color: white;
                }}

                .platform-damai {{
                    background: var(--success);
                }}

                .status-running {{
                    color: var(--success);
                    font-weight: 600;
                }}

                .status-stopped {{
                    color: var(--danger);
                    font-weight: 600;
                }}

                .action-buttons {{
                    display: flex;
                    gap: 5px;
                    justify-content: center;
                }}

                .btn {{
                    padding: 8px 12px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: var(--transition);
                    font-size: 0.9em;
                    display: inline-flex;
                    align-items: center;
                    gap: 5px;
                }}

                .btn-sm {{
                    padding: 6px 10px;
                    font-size: 0.8em;
                }}

                .btn-primary {{
                    background: var(--primary);
                    color: white;
                }}

                .btn-warning {{
                    background: var(--warning);
                    color: white;
                }}

                .btn-danger {{
                    background: var(--danger);
                    color: white;
                }}

                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                }}

                .form-section {{
                    background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 30px;
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }}

                .form-section h3 {{
                    color: #333;
                    margin-bottom: 20px;
                    font-size: 1.5em;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .form-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }}

                .form-group {{
                    display: flex;
                    flex-direction: column;
                }}

                .form-group label {{
                    margin-bottom: 5px;
                    font-weight: 600;
                    color: #555;
                }}

                .form-group input, .form-group select {{
                    padding: 10px;
                    border: 2px solid #e1e5e9;
                    border-radius: 8px;
                    font-size: 0.9em;
                    transition: var(--transition);
                }}

                .form-group input:focus, .form-group select:focus {{
                    outline: none;
                    border-color: var(--primary);
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                }}

                .form-actions {{
                    text-align: center;
                }}

                .btn-submit {{
                    background: linear-gradient(135deg, var(--success) 0%, #a8e6cf 100%);
                    color: white;
                    padding: 12px 30px;
                    font-size: 1.1em;
                    font-weight: 600;
                }}

                .btn-trigger-all {{
                    background: linear-gradient(135deg, var(--info) 0%, #00f2fe 100%);
                    color: white;
                    padding: 12px 25px;
                    font-size: 1em;
                    font-weight: 600;
                    margin-left: 15px;
                }}

                .notification {{
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
                }}

                .notification.show {{
                    transform: translateX(0);
                }}

                .notification.success {{
                    border-left-color: var(--success);
                }}

                .notification.error {{
                    border-left-color: var(--danger);
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
                    .header h1 {{
                        font-size: 2em;
                    }}
                    .content {{
                        padding: 20px;
                    }}
                    .form-grid {{
                        grid-template-columns: 1fr;
                    }}
                    .table-container {{
                        padding: 15px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1><i class="fas fa-tasks"></i> 任务管理</h1>
                    <p>管理抢票任务，监控执行状态</p>
                </div>
                
                <div class="content">
                    <a href="/{qs}" class="back-btn">
                        <i class="fas fa-arrow-left"></i> 返回控制台
                    </a>
                    
                    <div class="stats-bar">
                        <div class="stat-card">
                            <h3>{len(tasks)}</h3>
                            <p>总任务数</p>
                        </div>
                        <div class="stat-card">
                            <h3>{len([t for t in tasks if t.auto_start])}</h3>
                            <p>自动任务</p>
                        </div>
                        <div class="stat-card">
                            <h3>{len([t for t in tasks if not t.auto_start])}</h3>
                            <p>手动任务</p>
                        </div>
                        <div class="stat-card">
                            <h3>{len(set(t.platform.value for t in tasks))}</h3>
                            <p>平台数量</p>
                        </div>
                    </div>
                    
                    <div class="table-container">
                        <h3 style="margin-bottom: 20px; color: #333;">
                            <i class="fas fa-list"></i> 任务列表
                        </h3>
                        <table class="tasks-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>活动ID</th>
                                    <th>平台</th>
                                    <th>场次ID</th>
                                    <th>场次关键词</th>
                                    <th>设备ID</th>
                                    <th>数量</th>
                                    <th>目标价</th>
                                    <th>自动开始</th>
                                    <th>开始时间</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody>
                                {('\\n'.join(rows) if rows else '<tr><td colspan="11" style="text-align: center; padding: 40px; color: #666;">暂无任务</td></tr>')}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fas fa-plus-circle"></i> 新增任务</h3>
                        <form method="post" action="/tasks/add{qs}" id="addTaskForm">
                            <div class="form-grid">
                                <div class="form-group">
                                    <label>活动ID *</label>
                                    <input name="event_id" required placeholder="请输入活动ID"/>
                                </div>
                                <div class="form-group">
                                    <label>平台</label>
                                    <select name="platform">
                                        <option value="damai">大麦网</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>场次ID</label>
                                    <input name="performance_id" placeholder="场次ID（可选）"/>
                                </div>
                                <div class="form-group">
                                    <label>场次关键词</label>
                                    <input name="performance_keywords" placeholder="用逗号分隔多个关键词"/>
                                </div>
                                <div class="form-group">
                                    <label>设备ID</label>
                                    <input name="device_id" placeholder="ADB设备ID（可选）"/>
                                </div>
                                <div class="form-group">
                                    <label>数量</label>
                                    <input name="quantity" value="1" type="number" min="1"/>
                                </div>
                                <div class="form-group">
                                    <label>目标价</label>
                                    <input name="target_price" placeholder="目标价格（可选）"/>
                                </div>
                                <div class="form-group">
                                    <label>重试次数</label>
                                    <input name="retry_times" value="3" type="number" min="1"/>
                                </div>
                                <div class="form-group">
                                    <label>重试间隔(秒)</label>
                                    <input name="retry_interval" value="1.0" type="number" step="0.1" min="0.1"/>
                                </div>
                                <div class="form-group">
                                    <label>自动开始</label>
                                    <input type="checkbox" name="auto_start"/>
                                </div>
                                <div class="form-group">
                                    <label>开始时间</label>
                                    <input name="start_time" type="datetime-local"/>
                                </div>
                                <div class="form-group">
                                    <label>偏移秒数</label>
                                    <input name="start_offset_seconds" placeholder="相对开始时间的偏移秒数"/>
                                </div>
                                <div class="form-group">
                                    <label>座位偏好</label>
                                    <input name="seat_preference" placeholder="用逗号分隔多个偏好"/>
                                </div>
                            </div>
                            <div class="form-actions">
                                <button type="submit" class="btn btn-submit">
                                    <i class="fas fa-plus"></i> 添加任务
                                </button>
                            </div>
                        </form>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fas fa-cogs"></i> 批量操作</h3>
                        <div class="form-actions">
                            <form method="post" action="/tasks/trigger{qs}" style="display: inline;">
                                <button type="submit" class="btn btn-trigger-all">
                                    <i class="fas fa-play"></i> 触发全部任务
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 通知提示 -->
            <div id="notification" class="notification">
                <h4 id="notificationTitle"></h4>
                <p id="notificationMessage"></p>
            </div>

            <script>
                // 显示通知
                function showNotification(title, message, type = 'success') {{
                    const notification = document.getElementById('notification');
                    const titleEl = document.getElementById('notificationTitle');
                    const messageEl = document.getElementById('notificationMessage');
                    
                    titleEl.textContent = title;
                    messageEl.textContent = message;
                    
                    notification.className = `notification ${{type}}`;
                    notification.classList.add('show');
                    
                    setTimeout(() => {{
                        notification.classList.remove('show');
                    }}, 3000);
                }}
                
                // 触发任务
                function triggerTask(eventId) {{
                    if (confirm(`确定要立即执行任务 "${{eventId}}" 吗？`)) {{
                        const form = document.createElement('form');
                        form.method = 'POST';
                        form.action = '/tasks/trigger_one{qs}';
                        
                        const input = document.createElement('input');
                        input.type = 'hidden';
                        input.name = 'event_id';
                        input.value = eventId;
                        
                        form.appendChild(input);
                        document.body.appendChild(form);
                        form.submit();
                    }}
                }}
                
                // 编辑任务
                function editTask(eventId) {{
                    window.location.href = `/tasks/edit?event_id=${{eventId}}{qs}`;
                }}
                
                // 删除任务
                function deleteTask(eventId, platform) {{
                    if (confirm(`确定要删除任务 "${{eventId}}" 吗？此操作不可恢复！`)) {{
                        const form = document.createElement('form');
                        form.method = 'POST';
                        form.action = '/tasks/delete{qs}';
                        
                        const eventInput = document.createElement('input');
                        eventInput.type = 'hidden';
                        eventInput.name = 'event_id';
                        eventInput.value = eventId;
                        
                        const platformInput = document.createElement('input');
                        platformInput.type = 'hidden';
                        platformInput.name = 'platform';
                        platformInput.value = platform;
                        
                        form.appendChild(eventInput);
                        form.appendChild(platformInput);
                        document.body.appendChild(form);
                        form.submit();
                    }}
                }}
                
                // 表单提交成功提示
                document.getElementById('addTaskForm').addEventListener('submit', function() {{
                    showNotification('任务添加中', '正在处理您的请求...', 'success');
                }});
            </script>
        </body>
        </html>
        """
        return HTMLResponse(html)

    @app.get("/tasks/edit", response_class=HTMLResponse)
    def edit_task(event_id: str, token: None = Depends(auth)):
        tasks: List[TicketRequest] = config_manager.load_ticket_requests()
        tr = next((x for x in tasks if x.event_id == event_id), None)
        if not tr:
            return HTMLResponse("未找到任务", status_code=404)
        def _csv(lst):
            return ", ".join(lst) if lst else ""
        t = config_manager.get_setting("web_token", "")
        qs = _token_qs(t)
        form = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>✏️ 编辑任务 - Grab Tickets</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                :root {{
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
                }}

                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    min-height: 100vh;
                    padding: 20px;
                    color: #333;
                }}

                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: var(--radius);
                    box-shadow: var(--shadow);
                    overflow: hidden;
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}

                .header {{
                    background: linear-gradient(135deg, var(--warning) 0%, #f5576c 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}

                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                }}

                .header p {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}

                .content {{
                    padding: 30px;
                }}

                .back-btn {{
                    display: inline-block;
                    padding: 10px 20px;
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                    text-decoration: none;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    transition: var(--transition);
                }}

                .back-btn:hover {{
                    background: rgba(255, 255, 255, 0.3);
                    transform: translateY(-2px);
                }}

                .edit-form {{
                    background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
                    border-radius: 15px;
                    padding: 30px;
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }}

                .edit-form h3 {{
                    color: #333;
                    margin-bottom: 25px;
                    font-size: 1.8em;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    text-align: center;
                    justify-content: center;
                }}

                .form-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 25px;
                }}

                .form-group {{
                    display: flex;
                    flex-direction: column;
                }}

                .form-group label {{
                    margin-bottom: 8px;
                    font-weight: 600;
                    color: #555;
                    font-size: 0.95em;
                }}

                .form-group input, .form-group select {{
                    padding: 12px;
                    border: 2px solid #e1e5e9;
                    border-radius: 10px;
                    font-size: 0.95em;
                    transition: var(--transition);
                    background: white;
                }}

                .form-group input:focus, .form-group select:focus {{
                    outline: none;
                    border-color: var(--warning);
                    box-shadow: 0 0 0 3px rgba(240, 147, 251, 0.1);
                }}

                .form-group input[type="checkbox"] {{
                    width: 20px;
                    height: 20px;
                    accent-color: var(--warning);
                }}

                .checkbox-group {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .checkbox-group label {{
                    margin-bottom: 0;
                    cursor: pointer;
                }}

                .form-actions {{
                    text-align: center;
                    margin-top: 30px;
                }}

                .btn {{
                    padding: 15px 35px;
                    border: none;
                    border-radius: 12px;
                    cursor: pointer;
                    transition: var(--transition);
                    font-size: 1.1em;
                    font-weight: 600;
                    display: inline-flex;
                    align-items: center;
                    gap: 10px;
                    margin: 0 10px;
                }}

                .btn-save {{
                    background: linear-gradient(135deg, var(--success) 0%, #a8e6cf 100%);
                    color: white;
                }}

                .btn-cancel {{
                    background: linear-gradient(135deg, var(--danger) 0%, #ff4b2b 100%);
                    color: white;
                }}

                .btn:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
                }}

                .task-info {{
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 25px;
                    text-align: center;
                    color: white;
                }}

                .task-info h4 {{
                    margin-bottom: 10px;
                    font-size: 1.2em;
                }}

                .task-id {{
                    background: rgba(255, 255, 255, 0.2);
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-family: monospace;
                    font-size: 1.1em;
                }}

                @media (max-width: 768px) {{
                    .header h1 {{
                        font-size: 2em;
                    }}
                    .content {{
                        padding: 20px;
                    }}
                    .form-grid {{
                        grid-template-columns: 1fr;
                    }}
                    .edit-form {{
                        padding: 20px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1><i class="fas fa-edit"></i> 编辑任务</h1>
                    <p>修改任务配置参数</p>
                </div>
                
                <div class="content">
                    <a href="/tasks/html{qs}" class="back-btn">
                        <i class="fas fa-arrow-left"></i> 返回任务管理
                    </a>
                    
                    <div class="task-info">
                        <h4>正在编辑任务</h4>
                        <div class="task-id">{tr.event_id}</div>
                    </div>
                    
                    <div class="edit-form">
                        <h3><i class="fas fa-cog"></i> 任务配置</h3>
                        <form method="post" action="/tasks/save{qs}">
                            <input type="hidden" name="event_id" value="{tr.event_id}"/>
                            
                            <div class="form-grid">
                                <div class="form-group">
                                    <label>平台</label>
                                    <select name="platform">
                                        <option value="damai" {'selected' if tr.platform.value == 'damai' else ''}>大麦网</option>
                                    </select>
                                </div>
                                
                                <div class="form-group">
                                    <label>场次ID</label>
                                    <input name="performance_id" value="{tr.performance_id or ''}" placeholder="场次ID（可选）"/>
                                </div>
                                
                                <div class="form-group">
                                    <label>场次关键词</label>
                                    <input name="performance_keywords" value="{_csv(tr.performance_keywords)}" placeholder="用逗号分隔多个关键词"/>
                                </div>
                                
                                <div class="form-group">
                                    <label>设备ID</label>
                                    <input name="device_id" value="{tr.device_id or ''}" placeholder="ADB设备ID（可选）"/>
                                </div>
                                
                                <div class="form-group">
                                    <label>数量</label>
                                    <input name="quantity" value="{tr.quantity}" type="number" min="1"/>
                                </div>
                                
                                <div class="form-group">
                                    <label>目标价</label>
                                    <input name="target_price" value="{tr.target_price or ''}" placeholder="目标价格（可选）"/>
                                </div>
                                
                                <div class="form-group">
                                    <label>重试次数</label>
                                    <input name="retry_times" value="{tr.retry_times}" type="number" min="1"/>
                                </div>
                                
                                <div class="form-group">
                                    <label>重试间隔(秒)</label>
                                    <input name="retry_interval" value="{tr.retry_interval}" type="number" step="0.1" min="0.1"/>
                                </div>
                                
                                <div class="form-group">
                                    <label>座位偏好</label>
                                    <input name="seat_preference" value="{_csv(tr.seat_preference)}" placeholder="用逗号分隔多个偏好"/>
                                </div>
                                
                                <div class="form-group">
                                    <label>开始时间</label>
                                    <input name="start_time" value="{tr.start_time or ''}" type="datetime-local"/>
                                </div>
                                
                                <div class="form-group">
                                    <label>偏移秒数</label>
                                    <input name="start_offset_seconds" value="{tr.start_offset_seconds or ''}" placeholder="相对开始时间的偏移秒数"/>
                                </div>
                                
                                <div class="form-group">
                                    <div class="checkbox-group">
                                        <input type="checkbox" name="auto_start" id="auto_start" {'checked' if tr.auto_start else ''}/>
                                        <label for="auto_start">自动开始</label>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="form-actions">
                                <button type="submit" class="btn btn-save">
                                    <i class="fas fa-save"></i> 保存修改
                                </button>
                                <a href="/tasks/html{qs}" class="btn btn-cancel">
                                    <i class="fas fa-times"></i> 取消
                                </a>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(form)

    @app.post("/tasks/save")
    async def save_task(request: Request, token: None = Depends(auth)):
        form = await request.form()
        event_id = str(form.get('event_id')).strip()
        tasks: List[TicketRequest] = config_manager.load_ticket_requests()
        idx = next((i for i, x in enumerate(tasks) if x.event_id == event_id), -1)
        if idx < 0:
            return HTMLResponse("未找到任务", status_code=404)
        from ..data.models import TicketRequest, PlatformType
        def _split_csv(val: str):
            v = (val or "").strip()
            return [s.strip() for s in v.split(',') if s.strip()] if v else None
        tr = tasks[idx]
        # 更新字段
        tr.platform = PlatformType(str(form.get('platform','damai')).strip())
        tr.performance_id = (str(form.get('performance_id')).strip() or None)
        tr.performance_keywords = _split_csv(form.get('performance_keywords'))
        tr.device_id = (str(form.get('device_id')).strip() or None)
        tr.quantity = int(form.get('quantity','1'))
        tr.target_price = (float(form.get('target_price')) if form.get('target_price') else None)
        tr.retry_times = int(form.get('retry_times','3'))
        tr.retry_interval = float(form.get('retry_interval','1.0'))
        tr.auto_start = bool(form.get('auto_start'))
        tr.start_time = (str(form.get('start_time')).strip() or None)
        tr.start_offset_seconds = (int(form.get('start_offset_seconds')) if form.get('start_offset_seconds') else None)
        tr.seat_preference = _split_csv(form.get('seat_preference'))
        # 保存
        config_manager.save_ticket_requests(tasks)
        t = config_manager.get_setting("web_token", "")
        return RedirectResponse(url=f"/tasks/html{_token_qs(t)}", status_code=303)

    @app.get("/logs", response_class=HTMLResponse)
    def get_logs(max_lines: int = 200, token: None = Depends(auth)):
        t = config_manager.get_setting("web_token", "")
        qs = _token_qs(t)
        logs = get_latest_text(max_lines)
        
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>📋 系统日志 - Grab Tickets</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                :root {{
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
                }}

                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    min-height: 100vh;
                    padding: 20px;
                    color: #333;
                }}

                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: var(--radius);
                    box-shadow: var(--shadow);
                    overflow: hidden;
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}

                .header {{
                    background: linear-gradient(135deg, var(--dark) 0%, #1a202c 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}

                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                }}

                .header p {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}

                .content {{
                    padding: 30px;
                }}

                .back-btn {{
                    display: inline-block;
                    padding: 10px 20px;
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                    text-decoration: none;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    transition: var(--transition);
                }}

                .back-btn:hover {{
                    background: rgba(255, 255, 255, 0.3);
                    transform: translateY(-2px);
                }}

                .controls-section {{
                    background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 30px;
                    border: 1px solid rgba(0, 0, 0, 0.05);
                    display: flex;
                    align-items: center;
                    gap: 20px;
                    flex-wrap: wrap;
                }}

                .control-group {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .control-group label {{
                    font-weight: 600;
                    color: #555;
                    font-size: 0.95em;
                }}

                .control-group input, .control-group select {{
                    padding: 8px 12px;
                    border: 2px solid #e1e5e9;
                    border-radius: 8px;
                    font-size: 0.9em;
                    transition: var(--transition);
                }}

                .control-group input:focus, .control-group select:focus {{
                    outline: none;
                    border-color: var(--primary);
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                }}

                .btn {{
                    padding: 10px 20px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: var(--transition);
                    font-size: 0.95em;
                    font-weight: 600;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    color: white;
                }}

                .btn-primary {{
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                }}

                .btn-success {{
                    background: linear-gradient(135deg, var(--success) 0%, #a8e6cf 100%);
                }}

                .btn-warning {{
                    background: linear-gradient(135deg, var(--warning) 0%, #f5576c 100%);
                }}

                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
                }}

                .logs-container {{
                    background: var(--dark);
                    border-radius: 15px;
                    padding: 25px;
                    color: #e2e8f0;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    position: relative;
                }}

                .logs-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    flex-wrap: wrap;
                    gap: 15px;
                }}

                .logs-title {{
                    color: #f7fafc;
                    font-size: 1.4em;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .logs-actions {{
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                }}

                .logs-content {{
                    background: #0f1419;
                    border-radius: 12px;
                    padding: 20px;
                    max-height: 600px;
                    overflow-y: auto;
                    font-family: 'Courier New', 'Monaco', monospace;
                    font-size: 0.9em;
                    line-height: 1.6;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    position: relative;
                }}

                .logs-content::-webkit-scrollbar {{
                    width: 8px;
                }}

                .logs-content::-webkit-scrollbar-track {{
                    background: #1a202c;
                    border-radius: 4px;
                }}

                .logs-content::-webkit-scrollbar-thumb {{
                    background: var(--primary);
                    border-radius: 4px;
                }}

                .logs-content::-webkit-scrollbar-thumb:hover {{
                    background: var(--secondary);
                }}

                .log-line {{
                    margin-bottom: 8px;
                    padding: 8px 12px;
                    border-radius: 8px;
                    transition: var(--transition);
                    position: relative;
                    word-wrap: break-word;
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
                    color: var(--info);
                    margin-right: 10px;
                    font-weight: bold;
                }}

                .log-line.info {{
                    border-left: 3px solid var(--info);
                }}

                .log-line.warning {{
                    border-left: 3px solid var(--warning);
                }}

                .log-line.error {{
                    border-left: 3px solid var(--danger);
                }}

                .log-line.success {{
                    border-left: 3px solid var(--success);
                }}

                .filter-section {{
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}

                .filter-section h4 {{
                    color: #f7fafc;
                    margin-bottom: 15px;
                    font-size: 1.1em;
                }}

                .filter-controls {{
                    display: flex;
                    gap: 15px;
                    flex-wrap: wrap;
                    align-items: center;
                }}

                .filter-controls input {{
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    color: #f7fafc;
                    padding: 8px 12px;
                    border-radius: 6px;
                }}

                .filter-controls input::placeholder {{
                    color: rgba(255, 255, 255, 0.6);
                }}

                .status-indicator {{
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                    animation: pulse 2s infinite;
                }}

                .status-connected {{
                    background: var(--success);
                }}

                .status-disconnected {{
                    background: var(--danger);
                }}

                @keyframes pulse {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0.5; }}
                    100% {{ opacity: 1; }}
                }}

                @media (max-width: 768px) {{
                    .header h1 {{
                        font-size: 2em;
                    }}
                    .content {{
                        padding: 20px;
                    }}
                    .controls-section {{
                        flex-direction: column;
                        align-items: stretch;
                    }}
                    .logs-header {{
                        flex-direction: column;
                        align-items: stretch;
                    }}
                    .filter-controls {{
                        flex-direction: column;
                        align-items: stretch;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1><i class="fas fa-terminal"></i> 系统日志</h1>
                    <p>实时监控系统运行状态和日志信息</p>
                </div>
                
                <div class="content">
                    <a href="/{qs}" class="back-btn">
                        <i class="fas fa-arrow-left"></i> 返回控制台
                    </a>
                    
                    <div class="controls-section">
                        <div class="control-group">
                            <label>最大行数:</label>
                            <select id="maxLines" onchange="changeMaxLines()">
                                <option value="100" {'selected' if max_lines == 100 else ''}>100行</option>
                                <option value="200" {'selected' if max_lines == 200 else ''}>200行</option>
                                <option value="500" {'selected' if max_lines == 500 else ''}>500行</option>
                                <option value="1000" {'selected' if max_lines == 1000 else ''}>1000行</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label>自动刷新:</label>
                            <input type="checkbox" id="autoRefresh" onchange="toggleAutoRefresh()" checked/>
                        </div>
                        
                        <div class="control-group">
                            <label>刷新间隔:</label>
                            <select id="refreshInterval">
                                <option value="5">5秒</option>
                                <option value="10" selected>10秒</option>
                                <option value="30">30秒</option>
                                <option value="60">1分钟</option>
                            </select>
                        </div>
                        
                        <button class="btn btn-primary" onclick="refreshLogs()">
                            <i class="fas fa-sync-alt"></i> 刷新日志
                        </button>
                        
                        <button class="btn btn-success" onclick="clearLogs()">
                            <i class="fas fa-trash"></i> 清空显示
                        </button>
                    </div>
                    
                    <div class="logs-container">
                        <div class="logs-header">
                            <div class="logs-title">
                                <i class="fas fa-file-alt"></i> 实时日志
                                <span class="status-indicator status-connected" id="connectionStatus"></span>
                                <span id="connectionText">WebSocket已连接</span>
                            </div>
                            <div class="logs-actions">
                                <button class="btn btn-warning" onclick="scrollToBottom()">
                                    <i class="fas fa-arrow-down"></i> 滚动到底部
                                </button>
                                <button class="btn btn-primary" onclick="copyLogs()">
                                    <i class="fas fa-copy"></i> 复制日志
                                </button>
                            </div>
                        </div>
                        
                        <div class="filter-section">
                            <h4><i class="fas fa-filter"></i> 日志过滤</h4>
                            <div class="filter-controls">
                                <input type="text" id="searchFilter" placeholder="搜索关键词..." onkeyup="filterLogs()"/>
                                <select id="levelFilter" onchange="filterLogs()">
                                    <option value="">所有级别</option>
                                    <option value="INFO">INFO</option>
                                    <option value="WARNING">WARNING</option>
                                    <option value="ERROR">ERROR</option>
                                    <option value="SUCCESS">SUCCESS</option>
                                </select>
                                <button class="btn btn-success" onclick="clearFilters()">
                                    <i class="fas fa-times"></i> 清除过滤
                                </button>
                            </div>
                        </div>
                        
                        <div class="logs-content" id="logsContent">
                            {logs.replace(chr(10), '<br>') if logs else '<div class="log-line">暂无日志</div>'}
                        </div>
                    </div>
                </div>
            </div>

            <script>
                let autoRefreshInterval = null;
                let ws = null;
                let originalLogs = `{logs.replace(chr(10), '\\n') if logs else ''}`;
                
                // 页面加载完成后初始化
                document.addEventListener('DOMContentLoaded', function() {{
                    initWebSocket();
                    startAutoRefresh();
                }});
                
                // 初始化WebSocket连接
                function initWebSocket() {{
                    try {{
                        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                        const wsUrl = `${{protocol}}//${{window.location.host}}/ws/logs`;
                        ws = new WebSocket(wsUrl);
                        
                        ws.onopen = function() {{
                            updateConnectionStatus(true);
                        }};
                        
                        ws.onmessage = function(event) {{
                            appendLog(event.data);
                        }};
                        
                        ws.onclose = function() {{
                            updateConnectionStatus(false);
                            // 尝试重连
                            setTimeout(initWebSocket, 5000);
                        }};
                        
                        ws.onerror = function() {{
                            updateConnectionStatus(false);
                        }};
                    }} catch (error) {{
                        console.error('WebSocket连接失败:', error);
                        updateConnectionStatus(false);
                    }}
                }}
                
                // 更新连接状态
                function updateConnectionStatus(connected) {{
                    const statusEl = document.getElementById('connectionStatus');
                    const textEl = document.getElementById('connectionText');
                    
                    if (connected) {{
                        statusEl.className = 'status-indicator status-connected';
                        textEl.textContent = 'WebSocket已连接';
                    }} else {{
                        statusEl.className = 'status-indicator status-disconnected';
                        textEl.textContent = 'WebSocket已断开';
                    }}
                }}
                
                // 添加日志
                function appendLog(logText) {{
                    const logsContent = document.getElementById('logsContent');
                    const logLine = document.createElement('div');
                    logLine.className = 'log-line';
                    logLine.innerHTML = '> ' + logText;
                    
                    // 根据日志内容添加样式类
                    if (logText.includes('ERROR') || logText.includes('错误')) {{
                        logLine.classList.add('error');
                    }} else if (logText.includes('WARNING') || logText.includes('警告')) {{
                        logLine.classList.add('warning');
                    }} else if (logText.includes('SUCCESS') || logText.includes('成功')) {{
                        logLine.classList.add('success');
                    }} else {{
                        logLine.classList.add('info');
                    }}
                    
                    logsContent.appendChild(logLine);
                    
                    // 自动滚动到底部
                    if (document.getElementById('autoRefresh').checked) {{
                        logsContent.scrollTop = logsContent.scrollHeight;
                    }}
                }}
                
                // 切换自动刷新
                function toggleAutoRefresh() {{
                    const autoRefresh = document.getElementById('autoRefresh');
                    if (autoRefresh.checked) {{
                        startAutoRefresh();
                    }} else {{
                        stopAutoRefresh();
                    }}
                }}
                
                // 开始自动刷新
                function startAutoRefresh() {{
                    const interval = parseInt(document.getElementById('refreshInterval').value) * 1000;
                    autoRefreshInterval = setInterval(refreshLogs, interval);
                }}
                
                // 停止自动刷新
                function stopAutoRefresh() {{
                    if (autoRefreshInterval) {{
                        clearInterval(autoRefreshInterval);
                        autoRefreshInterval = null;
                    }}
                }}
                
                // 刷新日志
                function refreshLogs() {{
                    window.location.reload();
                }}
                
                // 清空日志显示
                function clearLogs() {{
                    if (confirm('确定要清空日志显示吗？')) {{
                        document.getElementById('logsContent').innerHTML = '';
                    }}
                }}
                
                // 滚动到底部
                function scrollToBottom() {{
                    const logsContent = document.getElementById('logsContent');
                    logsContent.scrollTop = logsContent.scrollHeight;
                }}
                
                // 复制日志
                function copyLogs() {{
                    const logsContent = document.getElementById('logsContent');
                    const text = logsContent.innerText;
                    
                    navigator.clipboard.writeText(text).then(function() {{
                        alert('日志已复制到剪贴板');
                    }}).catch(function() {{
                        // 降级方案
                        const textArea = document.createElement('textarea');
                        textArea.value = text;
                        document.body.appendChild(textArea);
                        textArea.select();
                        document.execCommand('copy');
                        document.body.removeChild(textArea);
                        alert('日志已复制到剪贴板');
                    }});
                }}
                
                // 过滤日志
                function filterLogs() {{
                    const searchFilter = document.getElementById('searchFilter').value.toLowerCase();
                    const levelFilter = document.getElementById('levelFilter').value;
                    const logLines = document.querySelectorAll('.log-line');
                    
                    logLines.forEach(line => {{
                        const text = line.textContent.toLowerCase();
                        const matchesSearch = !searchFilter || text.includes(searchFilter);
                        const matchesLevel = !levelFilter || text.includes(levelFilter.toLowerCase());
                        
                        if (matchesSearch && matchesLevel) {{
                            line.style.display = 'block';
                        }} else {{
                            line.style.display = 'none';
                        }}
                    }});
                }}
                
                // 清除过滤
                function clearFilters() {{
                    document.getElementById('searchFilter').value = '';
                    document.getElementById('levelFilter').value = '';
                    document.querySelectorAll('.log-line').forEach(line => {{
                        line.style.display = 'block';
                    }});
                }}
                
                // 改变最大行数
                function changeMaxLines() {{
                    const maxLines = document.getElementById('maxLines').value;
                    window.location.href = `/logs?max_lines=${{maxLines}}`;
                }}
                
                // 页面卸载时清理
                window.addEventListener('beforeunload', function() {{
                    stopAutoRefresh();
                    if (ws) {{
                        ws.close();
                    }}
                }});
            </script>
        </body>
        </html>
        """
        return HTMLResponse(html)

    @app.get("/results")
    def get_results(n: int = 50, token: None = Depends(auth)):
        return {"summary": results_summary(), "items": latest_results(n)}

    @app.get("/results/html", response_class=HTMLResponse)
    def get_results_html(n: int = 50, token: None = Depends(auth)):
        data = latest_results(n)
        s = results_summary()
        t = config_manager.get_setting("web_token", "")
        qs = _token_qs(t)
        
        # 生成结果表格行
        rows = []
        for i, r in enumerate(data, 1):
            status_class = "success" if r.get('success') else "failure"
            status_text = "✅ 成功" if r.get('success') else "❌ 失败"
            status_icon = "check-circle" if r.get('success') else "times-circle"
            
            rows.append(f"""
                <tr class="result-row">
                    <td class="result-id">{i}</td>
                    <td class="result-time">{r.get('time', '-')}</td>
                    <td class="event-id">{r.get('event_id', '-')}</td>
                    <td class="platform"><span class="platform-badge platform-{r.get('platform', 'unknown')}">{r.get('platform', '-')}</span></td>
                    <td class="device-id">{r.get('device_id') or '-'}</td>
                    <td class="status"><span class="status-badge status-{status_class}"><i class="fas fa-{status_icon}"></i> {status_text}</span></td>
                    <td class="order-id">{r.get('order_id') or '-'}</td>
                    <td class="message">{r.get('message') or '-'}</td>
                </tr>
            """)
        
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>📊 结果查看 - Grab Tickets</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                :root {{
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
                }}

                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    min-height: 100vh;
                    padding: 20px;
                    color: #333;
                }}

                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: var(--radius);
                    box-shadow: var(--shadow);
                    overflow: hidden;
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}

                .header {{
                    background: linear-gradient(135deg, var(--success) 0%, #a8e6cf 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}

                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                }}

                .header p {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}

                .content {{
                    padding: 30px;
                }}

                .back-btn {{
                    display: inline-block;
                    padding: 10px 20px;
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                    text-decoration: none;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    transition: var(--transition);
                }}

                .back-btn:hover {{
                    background: rgba(255, 255, 255, 0.3);
                    transform: translateY(-2px);
                }}

                .summary-section {{
                    background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 30px;
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }}

                .summary-section h3 {{
                    color: #333;
                    margin-bottom: 20px;
                    font-size: 1.5em;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                }}

                .summary-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
                    border: 1px solid rgba(0, 0, 0, 0.05);
                    position: relative;
                    overflow: hidden;
                }}

                .summary-card::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: linear-gradient(90deg, var(--primary), var(--secondary));
                }}

                .summary-card h4 {{
                    color: var(--primary);
                    font-size: 2.5em;
                    margin-bottom: 5px;
                }}

                .summary-card p {{
                    color: #666;
                    font-size: 0.9em;
                    font-weight: 600;
                }}

                .summary-card.success h4 {{
                    color: var(--success);
                }}

                .summary-card.failure h4 {{
                    color: var(--danger);
                }}

                .summary-card.rate h4 {{
                    color: var(--info);
                }}

                .results-section {{
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
                }}

                .results-section h3 {{
                    color: #333;
                    margin-bottom: 20px;
                    font-size: 1.5em;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .results-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9em;
                    overflow-x: auto;
                }}

                .results-table th {{
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    color: white;
                    padding: 15px 10px;
                    text-align: left;
                    font-weight: 600;
                    border-radius: 10px;
                }}

                .results-table td {{
                    padding: 12px 10px;
                    border-bottom: 1px solid #eee;
                    vertical-align: middle;
                }}

                .results-table tr:hover {{
                    background: #f8f9fa;
                }}

                .platform-badge {{
                    padding: 4px 8px;
                    border-radius: 20px;
                    font-size: 0.8em;
                    font-weight: 600;
                    color: white;
                }}

                .platform-damai {{
                    background: var(--success);
                }}

                .platform-unknown {{
                    background: var(--dark);
                }}

                .status-badge {{
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 0.85em;
                    font-weight: 600;
                    color: white;
                    display: inline-flex;
                    align-items: center;
                    gap: 5px;
                }}

                .status-success {{
                    background: var(--success);
                }}

                .status-failure {{
                    background: var(--danger);
                }}

                .controls-section {{
                    background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 30px;
                    border: 1px solid rgba(0, 0, 0, 0.05);
                    display: flex;
                    align-items: center;
                    gap: 20px;
                    flex-wrap: wrap;
                }}

                .control-group {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .control-group label {{
                    font-weight: 600;
                    color: #555;
                    font-size: 0.95em;
                }}

                .control-group input, .control-group select {{
                    padding: 8px 12px;
                    border: 2px solid #e1e5e9;
                    border-radius: 8px;
                    font-size: 0.9em;
                    transition: var(--transition);
                }}

                .control-group input:focus, .control-group select:focus {{
                    outline: none;
                    border-color: var(--primary);
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                }}

                .btn {{
                    padding: 10px 20px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: var(--transition);
                    font-size: 0.95em;
                    font-weight: 600;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    color: white;
                    text-decoration: none;
                }}

                .btn-primary {{
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                }}

                .btn-success {{
                    background: linear-gradient(135deg, var(--success) 0%, #a8e6cf 100%);
                }}

                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
                }}

                .chart-section {{
                    background: linear-gradient(135deg, var(--dark) 0%, #1a202c 100%);
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 30px;
                    color: #e2e8f0;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}

                .chart-section h3 {{
                    color: #f7fafc;
                    margin-bottom: 20px;
                    font-size: 1.5em;
                    text-align: center;
                }}

                .chart-container {{
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}

                .chart-placeholder {{
                    font-size: 1.2em;
                    color: #a0aec0;
                    padding: 40px;
                }}

                @media (max-width: 768px) {{
                    .header h1 {{
                        font-size: 2em;
                    }}
                    .content {{
                        padding: 20px;
                    }}
                    .summary-grid {{
                        grid-template-columns: 1fr;
                    }}
                    .controls-section {{
                        flex-direction: column;
                        align-items: stretch;
                    }}
                    .results-table {{
                        font-size: 0.8em;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1><i class="fas fa-chart-bar"></i> 结果查看</h1>
                    <p>分析抢票结果，监控成功率和性能数据</p>
                </div>
                
                <div class="content">
                    <a href="/{qs}" class="back-btn">
                        <i class="fas fa-arrow-left"></i> 返回控制台
                    </a>
                    
                    <div class="summary-section">
                        <h3><i class="fas fa-chart-pie"></i> 结果摘要</h3>
                        <div class="summary-grid">
                            <div class="summary-card">
                                <h4>{s.get('total', 0)}</h4>
                                <p>总任务数</p>
                            </div>
                            <div class="summary-card success">
                                <h4>{s.get('success', 0)}</h4>
                                <p>成功任务</p>
                            </div>
                            <div class="summary-card failure">
                                <h4>{s.get('failure', 0)}</h4>
                                <p>失败任务</p>
                            </div>
                            <div class="summary-card rate">
                                <h4>{s.get('success_rate', 0):.1%}</h4>
                                <p>成功率</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="controls-section">
                        <div class="control-group">
                            <label>显示数量:</label>
                            <select id="resultCount" onchange="changeResultCount()">
                                <option value="25" {'selected' if n == 25 else ''}>25条</option>
                                <option value="50" {'selected' if n == 50 else ''}>50条</option>
                                <option value="100" {'selected' if n == 100 else ''}>100条</option>
                                <option value="200" {'selected' if n == 200 else ''}>200条</option>
                            </select>
                        </div>
                        
                        <button class="btn btn-primary" onclick="refreshResults()">
                            <i class="fas fa-sync-alt"></i> 刷新结果
                        </button>
                        
                        <button class="btn btn-success" onclick="exportResults()">
                            <i class="fas fa-download"></i> 导出数据
                        </button>
                    </div>
                    
                    <div class="results-section">
                        <h3><i class="fas fa-list"></i> 最近结果</h3>
                        <div style="overflow-x: auto;">
                            <table class="results-table">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>时间</th>
                                        <th>活动ID</th>
                                        <th>平台</th>
                                        <th>设备ID</th>
                                        <th>状态</th>
                                        <th>订单ID</th>
                                        <th>消息</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {('\\n'.join(rows) if rows else '<tr><td colspan="8" style="text-align: center; padding: 40px; color: #666;">暂无结果数据</td></tr>')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    
                    <div class="chart-section">
                        <h3><i class="fas fa-chart-line"></i> 数据可视化</h3>
                        <div class="chart-container">
                            <div class="chart-placeholder">
                                <i class="fas fa-chart-area" style="font-size: 3em; margin-bottom: 20px; display: block; color: var(--info);"></i>
                                <p>图表功能开发中...</p>
                                <p style="font-size: 0.9em; margin-top: 10px;">未来版本将支持实时图表、趋势分析等功能</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                // 改变结果数量
                function changeResultCount() {{
                    const count = document.getElementById('resultCount').value;
                    window.location.href = `/results/html?n=${{count}}`;
                }}
                
                // 刷新结果
                function refreshResults() {{
                    window.location.reload();
                }}
                
                // 导出数据
                function exportResults() {{
                    const table = document.querySelector('.results-table');
                    const rows = Array.from(table.querySelectorAll('tr'));
                    
                    let csv = [];
                    rows.forEach(row => {{
                        const cols = Array.from(row.querySelectorAll('td, th'));
                        const rowData = cols.map(col => {{
                            let text = col.textContent || col.innerText;
                            // 移除图标和状态标签
                            text = text.replace(/[✅❌]/g, '').replace(/成功|失败/g, '').trim();
                            return `"${{text}}"`;
                        }});
                        csv.push(rowData.join(','));
                    }});
                    
                    const csvContent = csv.join('\\n');
                    const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
                    const link = document.createElement('a');
                    const url = URL.createObjectURL(blob);
                    link.setAttribute('href', url);
                    link.setAttribute('download', 'grab_tickets_results.csv');
                    link.style.visibility = 'hidden';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }}
                
                // 自动刷新（每60秒）
                setInterval(() => {{
                    // 可以在这里添加AJAX请求来更新结果
                }}, 60000);
            </script>
        </body>
        </html>
        """
        return HTMLResponse(html)

    @app.get("/devices")
    def get_devices(token: None = Depends(auth)):
        try:
            res = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)
            return {"output": res.stdout}
        except Exception as e:
            return {"error": str(e)}

    @app.get("/devices/html", response_class=HTMLResponse)
    def get_devices_html(token: None = Depends(auth)):
        t = config_manager.get_setting("web_token", "")
        qs = _token_qs(t)
        out = get_devices(token)
        output = out.get("output") if isinstance(out, dict) else str(out)
        
        # 解析设备信息
        devices = []
        if output:
            lines = output.strip().split('\n')
            for line in lines[1:]:  # 跳过标题行
                if line.strip() and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        device_id = parts[0].strip()
                        status = parts[1].strip()
                        devices.append({
                            'id': device_id,
                            'status': status,
                            'is_online': 'device' in status.lower()
                        })
        
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>📱 设备管理 - Grab Tickets</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                :root {{
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
                }}

                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    min-height: 100vh;
                    padding: 20px;
                    color: #333;
                }}

                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: var(--radius);
                    box-shadow: var(--shadow);
                    overflow: hidden;
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}

                .header {{
                    background: linear-gradient(135deg, var(--info) 0%, #00f2fe 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}

                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                }}

                .header p {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}

                .content {{
                    padding: 30px;
                }}

                .back-btn {{
                    display: inline-block;
                    padding: 10px 20px;
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                    text-decoration: none;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    transition: var(--transition);
                }}

                .back-btn:hover {{
                    background: rgba(255, 255, 255, 0.3);
                    transform: translateY(-2px);
                }}

                .stats-section {{
                    background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 30px;
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }}

                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                }}

                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }}

                .stat-card h3 {{
                    color: var(--primary);
                    font-size: 2em;
                    margin-bottom: 5px;
                }}

                .stat-card p {{
                    color: #666;
                    font-size: 0.9em;
                }}

                .devices-section {{
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
                }}

                .devices-section h3 {{
                    color: #333;
                    margin-bottom: 20px;
                    font-size: 1.5em;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .devices-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}

                .devices-table th {{
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    color: white;
                    padding: 15px 10px;
                    text-align: left;
                    font-weight: 600;
                    border-radius: 10px;
                }}

                .devices-table td {{
                    padding: 12px 10px;
                    border-bottom: 1px solid #eee;
                    vertical-align: middle;
                }}

                .devices-table tr:hover {{
                    background: #f8f9fa;
                }}

                .status-badge {{
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 0.85em;
                    font-weight: 600;
                    color: white;
                }}

                .status-online {{
                    background: var(--success);
                }}

                .status-offline {{
                    background: var(--danger);
                }}

                .status-unauthorized {{
                    background: var(--warning);
                }}

                .connect-section {{
                    background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
                    border-radius: 15px;
                    padding: 25px;
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }}

                .connect-section h4 {{
                    color: #333;
                    margin-bottom: 20px;
                    font-size: 1.3em;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .connect-form {{
                    display: flex;
                    gap: 15px;
                    align-items: end;
                    flex-wrap: wrap;
                }}

                .form-group {{
                    display: flex;
                    flex-direction: column;
                    flex: 1;
                    min-width: 250px;
                }}

                .form-group label {{
                    margin-bottom: 8px;
                    font-weight: 600;
                    color: #555;
                    font-size: 0.9em;
                }}

                .form-group input {{
                    padding: 12px;
                    border: 2px solid #e1e5e9;
                    border-radius: 10px;
                    font-size: 0.95em;
                    transition: var(--transition);
                }}

                .form-group input:focus {{
                    outline: none;
                    border-color: var(--info);
                    box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
                }}

                .btn {{
                    padding: 12px 25px;
                    border: none;
                    border-radius: 10px;
                    cursor: pointer;
                    transition: var(--transition);
                    font-size: 1em;
                    font-weight: 600;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    background: linear-gradient(135deg, var(--info) 0%, #00f2fe 100%);
                    color: white;
                }}

                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                }}

                .raw-output {{
                    background: var(--dark);
                    color: #e2e8f0;
                    border-radius: 15px;
                    padding: 25px;
                    margin-top: 30px;
                    font-family: 'Courier New', 'Monaco', monospace;
                    font-size: 0.9em;
                    line-height: 1.6;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    overflow-x: auto;
                }}

                .raw-output h4 {{
                    color: #f7fafc;
                    margin-bottom: 15px;
                    font-size: 1.2em;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .raw-output pre {{
                    background: rgba(255, 255, 255, 0.05);
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}

                .action-buttons {{
                    display: flex;
                    gap: 10px;
                    justify-content: center;
                }}

                .btn-refresh {{
                    background: linear-gradient(135deg, var(--success) 0%, #a8e6cf 100%);
                }}

                .btn-disconnect {{
                    background: linear-gradient(135deg, var(--danger) 0%, #ff4b2b 100%);
                }}

                @media (max-width: 768px) {{
                    .header h1 {{
                        font-size: 2em;
                    }}
                    .content {{
                        padding: 20px;
                    }}
                    .connect-form {{
                        flex-direction: column;
                        align-items: stretch;
                    }}
                    .form-group {{
                        min-width: auto;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1><i class="fas fa-mobile-alt"></i> 设备管理</h1>
                    <p>管理ADB设备连接，监控设备状态</p>
                </div>
                
                <div class="content">
                    <a href="/{qs}" class="back-btn">
                        <i class="fas fa-arrow-left"></i> 返回控制台
                    </a>
                    
                    <div class="stats-section">
                        <div class="stats-grid">
                            <div class="stat-card">
                                <h3>{len(devices)}</h3>
                                <p>总设备数</p>
                            </div>
                            <div class="stat-card">
                                <h3>{len([d for d in devices if d['is_online']])}</h3>
                                <p>在线设备</p>
                            </div>
                            <div class="stat-card">
                                <h3>{len([d for d in devices if not d['is_online']])}</h3>
                                <p>离线设备</p>
                            </div>
                            <div class="stat-card">
                                <h3>{len([d for d in devices if 'unauthorized' in d['status'].lower()])}</h3>
                                <p>未授权设备</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="devices-section">
                        <h3><i class="fas fa-list"></i> 设备列表</h3>
                        <div class="action-buttons" style="margin-bottom: 20px;">
                            <button class="btn btn-refresh" onclick="refreshDevices()">
                                <i class="fas fa-sync-alt"></i> 刷新设备列表
                            </button>
                        </div>
                        
                        <table class="devices-table">
                            <thead>
                                <tr>
                                    <th>设备ID</th>
                                    <th>状态</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody>
                                {('\\n'.join([f'''
                                    <tr>
                                        <td><code>{device['id']}</code></td>
                                        <td><span class="status-badge status-{'online' if device['is_online'] else 'offline'}">{device['status']}</span></td>
                                        <td>
                                            <button class="btn btn-sm" onclick="disconnectDevice('{device['id']}')" title="断开连接">
                                                <i class="fas fa-unlink"></i>
                                            </button>
                                        </td>
                                    </tr>
                                ''' for device in devices]) if devices else '<tr><td colspan="3" style="text-align: center; padding: 40px; color: #666;">暂无设备连接</td></tr>')}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="connect-section">
                        <h4><i class="fas fa-wifi"></i> Wi-Fi ADB 连接</h4>
                        <form method="post" action="/devices/connect{qs}" class="connect-form">
                            <div class="form-group">
                                <label>设备IP地址和端口</label>
                                <input name="serial" placeholder="192.168.1.23:5555" required/>
                            </div>
                            <button type="submit" class="btn">
                                <i class="fas fa-plug"></i> 连接设备
                            </button>
                        </form>
                        <div style="margin-top: 15px; color: #666; font-size: 0.9em;">
                            <i class="fas fa-info-circle"></i> 提示：确保设备已开启Wi-Fi ADB调试，格式为 IP:端口
                        </div>
                    </div>
                    
                    <div class="raw-output">
                        <h4><i class="fas fa-terminal"></i> 原始ADB输出</h4>
                        <pre>{output}</pre>
                    </div>
                </div>
            </div>

            <script>
                // 刷新设备列表
                function refreshDevices() {{
                    window.location.reload();
                }}
                
                // 断开设备连接
                function disconnectDevice(deviceId) {{
                    if (confirm(`确定要断开设备 "${{deviceId}}" 吗？`)) {{
                        // 这里可以添加断开连接的API调用
                        alert('断开连接功能需要后端API支持');
                    }}
                }}
                
                // 自动刷新设备状态（每30秒）
                setInterval(() => {{
                    // 可以在这里添加AJAX请求来更新设备状态
                }}, 30000);
            </script>
        </body>
        </html>
        """
        return HTMLResponse(html)

    @app.post("/tasks/trigger")
    async def trigger_tasks(body: TriggerBody, token: None = Depends(auth)):
        tasks: List[TicketRequest] = config_manager.load_ticket_requests()
        selected = tasks
        if body.event_ids:
            selected = [t for t in tasks if t.event_id in set(body.event_ids)]
        if not selected:
            return {"triggered": 0}
        # 异步后台执行
        asyncio.create_task(grab_engine.grab_tickets_batch(selected))
        return {"triggered": len(selected)}

    @app.post("/tasks/trigger_one")
    async def trigger_one(request: Request, token: None = Depends(auth)):
        form = await request.form()
        event_id = str(form.get("event_id", "")).strip()
        tasks: List[TicketRequest] = config_manager.load_ticket_requests()
        selected = [t for t in tasks if t.event_id == event_id]
        if selected:
            asyncio.create_task(grab_engine.grab_tickets_batch(selected))
        t = config_manager.get_setting("web_token", "")
        return RedirectResponse(url=f"/tasks/html{_token_qs(t)}", status_code=303)

    @app.post("/tasks/delete")
    async def delete_task(request: Request, token: None = Depends(auth)):
        form = await request.form()
        event_id = str(form.get("event_id", "")).strip()
        platform = str(form.get("platform", "damai")).strip()
        from ..data.models import PlatformType
        ok = config_manager.remove_ticket_request(event_id, PlatformType(platform))
        t = config_manager.get_setting("web_token", "")
        return RedirectResponse(url=f"/tasks/html{_token_qs(t)}", status_code=303)

    @app.post("/tasks/add")
    async def add_task(request: Request, token: None = Depends(auth)):
        form = await request.form()
        from ..data.models import TicketRequest, PlatformType
        def _split_csv(val: str):
            v = (val or "").strip()
            return [s.strip() for s in v.split(',') if s.strip()] if v else None
        try:
            tr = TicketRequest(
                event_id=str(form.get('event_id')).strip(),
                platform=PlatformType(str(form.get('platform','damai')).strip()),
                performance_id=(str(form.get('performance_id')).strip() or None),
                performance_keywords=_split_csv(form.get('performance_keywords')),
                device_id=(str(form.get('device_id')).strip() or None),
                quantity=int(form.get('quantity','1')),
                target_price=(float(form.get('target_price')) if form.get('target_price') else None),
                retry_times=int(form.get('retry_times','3')),
                retry_interval=float(form.get('retry_interval','1.0')),
                auto_start=bool(form.get('auto_start')),
                start_time=(str(form.get('start_time')).strip() or None),
                start_offset_seconds=(int(form.get('start_offset_seconds')) if form.get('start_offset_seconds') else None),
                seat_preference=_split_csv(form.get('seat_preference')),
            )
            ok = config_manager.add_ticket_request(tr)
        except Exception as e:
            logger.error(f"添加任务失败: {e}")
        t = config_manager.get_setting("web_token", "")
        return RedirectResponse(url=f"/tasks/html{_token_qs(t)}", status_code=303)

    @app.post("/devices/connect")
    async def devices_connect(request: Request, token: None = Depends(auth)):
        form = await request.form()
        serial = str(form.get('serial','')).strip()
        if serial:
            try:
                subprocess.run(["adb", "connect", serial], capture_output=True, text=True, timeout=5)
            except Exception:
                pass
        t = config_manager.get_setting("web_token", "")
        return RedirectResponse(url=f"/devices/html{_token_qs(t)}", status_code=303)

    return app


class WebConsoleServer:
    def __init__(self, config_manager: ConfigManager, grab_engine: GrabEngine, host: str = "0.0.0.0", port: int = 8080):
        self.config_manager = config_manager
        self.grab_engine = grab_engine
        self.host = host
        self.port = port
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Web 控制台已启动: http://{self.host}:{self.port}")

    def _run(self):
        import uvicorn
        app = create_app(self.config_manager, self.grab_engine)
        uvicorn.run(app, host=self.host, port=self.port, log_level="warning")



