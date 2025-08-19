"""
Prometheus 指标
"""
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from typing import Optional
from ..utils.logger import get_logger

logger = get_logger("metrics")

# 任务级指标
grab_requests_total = Counter("grab_requests_total", "发起的抢票请求数")
grab_success_total = Counter("grab_success_total", "抢票成功数")
grab_failure_total = Counter("grab_failure_total", "抢票失败数")
grab_duration_seconds = Histogram("grab_duration_seconds", "抢票单次耗时秒", buckets=(0.5, 1, 2, 5, 10, 20, 30, 60))

# 调度器指标
scheduler_due_total = Counter("scheduler_due_total", "到点触发的任务数")
scheduler_loop_errors_total = Counter("scheduler_loop_errors_total", "调度循环错误数")

_server_started = False

def start_metrics_server(port: int = 8001):
    global _server_started
    if _server_started:
        return
    try:
        start_http_server(port)
        logger.info(f"Prometheus 指标已启动: 0.0.0.0:{port}")
        _server_started = True
    except Exception as e:
        logger.error(f"启动 Prometheus 指标端口失败: {e}")


