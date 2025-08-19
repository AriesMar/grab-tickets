"""
简单任务调度器：按 start_time 到点自动触发抢票
"""
import threading
import time
from datetime import datetime
from typing import List

from ..data.models import TicketRequest
from ..core.grab_engine import GrabEngine
from ..utils.logger import get_logger
from ..utils.metrics import scheduler_due_total, scheduler_loop_errors_total
from ..data.config_manager import ConfigManager


class TaskScheduler:
    def __init__(self, config_manager: ConfigManager, grab_engine: GrabEngine, interval_seconds: int = 1):
        self.config_manager = config_manager
        self.grab_engine = grab_engine
        self.interval_seconds = int(self.config_manager.get_setting("scheduler_interval_seconds", interval_seconds))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.logger = get_logger("task_scheduler")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.logger.info("任务调度器已启动")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.logger.info("任务调度器已停止")

    def _parse_time(self, time_str: str) -> datetime | None:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    def _due_requests(self, requests: List[TicketRequest]) -> List[TicketRequest]:
        now = datetime.now()
        due: List[TicketRequest] = []
        changed = False
        for r in requests:
            if not getattr(r, "auto_start", False):
                continue
            # 若未设置绝对时间但设置了相对秒数，则在首次扫描时转换为绝对时间
            if not getattr(r, "start_time", None) and getattr(r, "start_offset_seconds", None):
                target = now.timestamp() + int(r.start_offset_seconds)
                r.start_time = datetime.fromtimestamp(target).strftime("%Y-%m-%d %H:%M:%S")
                changed = True
            dt = self._parse_time(r.start_time) if getattr(r, "start_time", None) else None
            if dt and now >= dt:
                due.append(r)
        if changed:
            # 保存转换后的绝对时间，避免重复换算
            self.config_manager.save_ticket_requests(requests)
        return due

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                requests = self.config_manager.load_ticket_requests()
                due = self._due_requests(requests)
                if due:
                    self.logger.info(f"到点自动开始抢票: {len(due)} 个任务")
                    scheduler_due_total.inc(len(due))
                    # 异步批量执行
                    import asyncio
                    asyncio.run(self.grab_engine.grab_tickets_batch(due))
                    # 触发一次后防止重复，关闭这些任务的 auto_start
                    for r in requests:
                        if r in due:
                            r.auto_start = False
                    self.config_manager.save_ticket_requests(requests)
            except Exception as e:
                self.logger.error(f"调度器循环异常: {e}")
                scheduler_loop_errors_total.inc()
            finally:
                time.sleep(self.interval_seconds)


