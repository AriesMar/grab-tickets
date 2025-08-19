"""
轻量 Web 控制台：
 - 查看健康检查、设置、任务列表、ADB 设备
 - 手动触发任务执行（立即执行）
"""
from typing import List, Optional
import threading
import asyncio
import subprocess

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
        html = f"""
        <h2>Grab Tickets 控制台</h2>
        <ul>
          <li><a href='/health{qs}'>健康检查</a></li>
          <li><a href='/settings{qs}'>当前设置(JSON)</a></li>
          <li><a href='/tasks/html{qs}'>任务管理(HTML)</a></li>
          <li><a href='/devices/html{qs}'>设备(HTML)</a></li>
          <li><a href='/tasks{qs}'>任务(JSON)</a></li>
          <li><a href='/devices{qs}'>设备(JSON)</a></li>
        </ul>
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
        rows = []
        for i, r in enumerate(tasks, 1):
            rows.append(
                f"<tr><td>{i}</td><td>{r.event_id}</td><td>{r.platform.value}</td><td>{r.performance_id or '-'}</td>"
                f"<td>{', '.join(r.performance_keywords) if r.performance_keywords else '-'}</td>"
                f"<td>{r.device_id or '-'}</td><td>{r.quantity}</td><td>{r.target_price or '-'}</td>"
                f"<td>{'是' if r.auto_start else '否'}</td><td>{r.start_time or r.start_offset_seconds or '-'}</td>"
                f"<td><form method='post' action='/tasks/trigger_one{qs}'><input type='hidden' name='event_id' value='{r.event_id}'/><button type='submit'>触发</button></form></td>"
                f"<td><form method='get' action='/tasks/edit{qs}'><input type='hidden' name='event_id' value='{r.event_id}'/><button type='submit'>编辑</button></form></td>"
                f"<td><form method='post' action='/tasks/delete{qs}'><input type='hidden' name='event_id' value='{r.event_id}'/><input type='hidden' name='platform' value='{r.platform.value}'/><button type='submit'>删除</button></form></td>"
                f"</tr>"
            )
        table = """
        <table border='1' cellspacing='0' cellpadding='6'>
          <tr><th>#</th><th>活动ID</th><th>平台</th><th>场次ID</th><th>场次关键词</th><th>设备ID</th><th>数量</th><th>目标价</th><th>自动开始</th><th>开始时间/偏移</th><th>手动触发</th><th>删除</th></tr>
          %s
        </table>
        """ % ("\n".join(rows) if rows else "<tr><td colspan='12'>暂无任务</td></tr>")

        add_form = f"""
        <h3>新增任务</h3>
        <form method='post' action='/tasks/add{qs}'>
          活动ID:<input name='event_id' required/> 平台:<input name='platform' value='damai'/>
          场次ID:<input name='performance_id'/> 场次关键词(逗号):<input name='performance_keywords'/>
          设备ID:<input name='device_id'/> 数量:<input name='quantity' value='1'/>
          目标价:<input name='target_price'/> 重试次数:<input name='retry_times' value='3'/>
          重试间隔(秒):<input name='retry_interval' value='1.0'/><br/>
          自动开始:<input type='checkbox' name='auto_start'/> 开始时间(YYYY-MM-DD HH:MM:SS):<input name='start_time'/> 或 偏移秒:<input name='start_offset_seconds'/><br/>
          座位偏好(逗号):<input name='seat_preference'/>
          <button type='submit'>添加</button>
        </form>
        """
        ops = f"""
        <h3>批量操作</h3>
        <form method='post' action='/tasks/trigger{qs}'><button type='submit'>触发全部</button></form>
        """
        return HTMLResponse(table + add_form + ops)

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
        <h3>编辑任务: {tr.event_id}</h3>
        <form method='post' action='/tasks/save{qs}'>
          <input type='hidden' name='event_id' value='{tr.event_id}'/>
          平台:<input name='platform' value='{tr.platform.value}'/>
          场次ID:<input name='performance_id' value='{tr.performance_id or ''}'/>
          场次关键词(逗号):<input name='performance_keywords' value='{_csv(tr.performance_keywords)}'/>
          设备ID:<input name='device_id' value='{tr.device_id or ''}'/>
          数量:<input name='quantity' value='{tr.quantity}'/>
          目标价:<input name='target_price' value='{tr.target_price or ''}'/>
          重试次数:<input name='retry_times' value='{tr.retry_times}'/>
          重试间隔(秒):<input name='retry_interval' value='{tr.retry_interval}'/>
          自动开始:<input type='checkbox' name='auto_start' {'checked' if tr.auto_start else ''}/>
          开始时间:<input name='start_time' value='{tr.start_time or ''}'/>
          偏移秒:<input name='start_offset_seconds' value='{tr.start_offset_seconds or ''}'/>
          座位偏好(逗号):<input name='seat_preference' value='{_csv(tr.seat_preference)}'/>
          <button type='submit'>保存</button>
        </form>
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

    @app.get("/logs", response_class=PlainTextResponse)
    def get_logs(max_lines: int = 200, token: None = Depends(auth)):
        return PlainTextResponse(get_latest_text(max_lines))

    @app.websocket("/ws/logs")
    async def ws_logs(ws: WebSocket):
        await ws.accept()
        q = subscribe_queue()
        try:
            while True:
                msg = await q.get()
                await ws.send_text(msg)
        except Exception:
            pass
        finally:
            unsubscribe_queue(q)
            try:
                await ws.close()
            except Exception:
                pass

    @app.get("/results")
    def get_results(n: int = 50, token: None = Depends(auth)):
        return {"summary": results_summary(), "items": latest_results(n)}

    @app.get("/results/html", response_class=HTMLResponse)
    def get_results_html(n: int = 50, token: None = Depends(auth)):
        data = latest_results(n)
        s = results_summary()
        rows = []
        for i, r in enumerate(data, 1):
            rows.append(
                f"<tr><td>{i}</td><td>{r.get('time')}</td><td>{r.get('event_id')}</td><td>{r.get('platform')}</td>"
                f"<td>{r.get('device_id') or '-'}</td><td>{'成功' if r.get('success') else '失败'}</td>"
                f"<td>{r.get('order_id') or '-'}</td><td>{r.get('message') or ''}</td></tr>"
            )
        table = """
        <h3>结果摘要</h3>
        <div>总数: {total} 成功: {success} 失败: {failure} 成功率: {sr:.2%}</div>
        <h3>最近结果</h3>
        <table border='1' cellspacing='0' cellpadding='6'>
          <tr><th>#</th><th>时间</th><th>活动ID</th><th>平台</th><th>设备ID</th><th>状态</th><th>订单ID</th><th>消息</th></tr>
          {rows}
        </table>
        """.format(total=s.get('total'), success=s.get('success'), failure=s.get('failure'), sr=s.get('success_rate'), rows="\n".join(rows) if rows else "<tr><td colspan='8'>暂无数据</td></tr>")
        return HTMLResponse(table)

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
        html = f"""
        <h3>ADB 设备</h3>
        <pre>{output}</pre>
        <h4>Wi‑Fi ADB 连接</h4>
        <form method='post' action='/devices/connect{qs}'>
          设备IP:端口:<input name='serial' placeholder='192.168.1.23:5555'/> <button type='submit'>连接</button>
        </form>
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


