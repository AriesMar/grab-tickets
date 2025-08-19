"""
内存日志缓冲与日志接入
"""
from collections import deque
from typing import Deque, List, Set
import asyncio
from loguru import logger


class LogBuffer:
    def __init__(self, max_lines: int = 2000):
        self.max_lines = max_lines
        self._buf: Deque[str] = deque(maxlen=max_lines)

    def append(self, line: str):
        if line is None:
            return
        self._buf.append(line.rstrip("\n"))

    def latest(self, n: int = 200) -> List[str]:
        if n <= 0:
            return []
        return list(self._buf)[-n:]


_buffer = LogBuffer()
_installed = False
_subscribers: Set[asyncio.Queue] = set()


def install_log_sink():
    global _installed
    if _installed:
        return

    def _sink(message):
        try:
            _buffer.append(message)
            # 推送给订阅者
            text = message.rstrip("\n") if isinstance(message, str) else str(message)
            for q in list(_subscribers):
                try:
                    q.put_nowait(text)
                except Exception:
                    pass
        except Exception:
            pass

    # 使用格式化后的文本
    logger.add(_sink, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    _installed = True


def get_latest_text(max_lines: int = 200) -> str:
    return "\n".join(_buffer.latest(max_lines))


def subscribe_queue(maxsize: int = 1000) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    _subscribers.add(q)
    return q


def unsubscribe_queue(q: asyncio.Queue):
    try:
        _subscribers.discard(q)
    except Exception:
        pass


