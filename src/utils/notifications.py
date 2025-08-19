"""
简单通知：控制台/系统提示为主，后续可扩展到企业微信/飞书/钉钉
"""
from ..utils.logger import get_logger
import json
import urllib.request
import urllib.error

logger = get_logger("notify")

def notify_info(msg: str):
    logger.info(f"通知: {msg}")

def notify_warn(msg: str):
    logger.warning(f"通知: {msg}")

def notify_error(msg: str):
    logger.error(f"通知: {msg}")

def notify_webhook(webhook_url: str, title: str, text: str):
    """简单Webhook，向任意HTTP端点POST文本。
    兼容飞书/企业微信自定义入群机器人（如需可调整payload格式）。"""
    if not webhook_url:
        return
    payload = {"title": title, "text": text}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            logger.info(f"Webhook 发送成功: {resp.status}")
    except urllib.error.URLError as e:
        logger.error(f"Webhook 发送失败: {e}")


