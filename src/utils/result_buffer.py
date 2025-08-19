"""
运行结果缓冲与摘要
"""
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Any

_buf: Deque[Dict[str, Any]] = deque(maxlen=500)


def add_result(entry: Dict[str, Any]):
    _buf.append(entry)


def add_result_safely(request, result):
    try:
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_id": getattr(request, "event_id", None),
            "platform": getattr(getattr(request, "platform", None), "value", None),
            "device_id": getattr(request, "device_id", None),
            "success": bool(getattr(result, "success", False)),
            "order_id": getattr(result, "order_id", None),
            "message": getattr(result, "message", ""),
        }
        add_result(entry)
    except Exception:
        pass


def latest(n: int = 50) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    return list(_buf)[-n:]


def summary() -> Dict[str, Any]:
    total = len(_buf)
    success = sum(1 for x in _buf if x.get("success"))
    failure = total - success
    return {
        "total": total,
        "success": success,
        "failure": failure,
        "success_rate": (success / total) if total else 0.0,
    }


