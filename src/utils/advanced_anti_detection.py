"""
高级反检测模块 - 确保完全隐蔽和反追踪
"""
import time
import random
import hashlib
import hmac
import base64
import json
import uuid
import string
import platform
import socket
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import requests
from fake_useragent import UserAgent
from loguru import logger
import os
import sys


class StealthManager:
    """隐身管理器 - 确保完全隐蔽"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.logger = logger.bind(name="stealth_manager")
        
        # 生成随机设备标识
        self.device_id = self._generate_random_device_id()
        self.session_id = self._generate_session_id()
        
        # 环境伪装
        self.environment_masks = {
            "webdriver": False,
            "automation": False,
            "selenium": False,
            "phantom": False,
            "headless": False
        }
    
    def _generate_random_device_id(self) -> str:
        """生成随机设备ID"""
        # 使用多种随机源生成设备ID
        random_sources = [
            str(random.randint(100000000000, 999999999999)),
            str(int(time.time() * 1000)),
            str(uuid.uuid4()),
            platform.node(),
            socket.gethostname()
        ]
        
        combined = "".join(random_sources)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return str(uuid.uuid4())
    
    def create_stealth_headers(self) -> Dict[str, str]:
        """创建隐身请求头"""
        # 随机化所有可能的指纹信息
        screen_width = random.choice([1920, 1366, 1440, 1536, 1600, 1680, 1920, 2560])
        screen_height = random.choice([1080, 768, 900, 864, 900, 1050, 1080, 1440])
        
        # 随机化浏览器信息
        browsers = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"
        ]
        
        user_agent = random.choice(browsers)
        
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "X-Forwarded-For": self._generate_random_ip(),
            "X-Real-IP": self._generate_random_ip(),
            "X-Device-ID": self.device_id,
            "X-Session-ID": self.session_id,
            "X-Request-ID": str(uuid.uuid4()),
            "X-Timestamp": str(int(time.time() * 1000))
        }
        
        return headers
    
    def _generate_random_ip(self) -> str:
        """生成随机IP地址"""
        # 生成私有IP地址，避免追踪
        private_ranges = [
            (10, 0, 0, 0, 10, 255, 255, 255),
            (172, 16, 0, 0, 172, 31, 255, 255),
            (192, 168, 0, 0, 192, 168, 255, 255)
        ]
        
        range_choice = random.choice(private_ranges)
        ip_parts = []
        for i in range(4):
            start = range_choice[i * 2]
            end = range_choice[i * 2 + 1]
            # 确保start <= end
            if start > end:
                start, end = end, start
            ip_parts.append(str(random.randint(start, end)))
        
        return ".".join(ip_parts)
    
    def create_stealth_cookies(self) -> Dict[str, str]:
        """创建隐身Cookie"""
        cookies = {
            "session_id": self.session_id,
            "device_id": self.device_id,
            "timestamp": str(int(time.time())),
            "random": str(random.randint(100000, 999999))
        }
        
        # 添加一些常见的Cookie
        common_cookies = {
            "language": random.choice(["en", "zh-CN", "zh-TW", "ja", "ko"]),
            "timezone": random.choice(["Asia/Shanghai", "America/New_York", "Europe/London"]),
            "theme": random.choice(["light", "dark", "auto"]),
            "notifications": random.choice(["true", "false"])
        }
        
        cookies.update(common_cookies)
        return cookies
    
    def create_stealth_params(self) -> Dict[str, str]:
        """创建隐身请求参数"""
        params = {
            "_t": str(int(time.time() * 1000)),
            "_r": str(random.randint(100000, 999999)),
            "device": self.device_id,
            "session": self.session_id,
            "v": str(random.randint(1, 100)),
            "ts": str(int(time.time()))
        }
        
        return params
    
    def add_stealth_delay(self):
        """添加隐身延迟"""
        # 使用更自然的延迟模式
        base_delay = random.uniform(0.5, 2.0)
        
        # 添加微小的随机变化
        variation = random.uniform(-0.1, 0.1)
        final_delay = max(0.1, base_delay + variation)
        
        time.sleep(final_delay)
        self.logger.debug(f"隐身延迟: {final_delay:.3f}秒")
    
    def rotate_stealth_identity(self):
        """轮换隐身身份"""
        # 重新生成设备ID和会话ID
        self.device_id = self._generate_random_device_id()
        self.session_id = self._generate_session_id()
        
        self.logger.info("隐身身份已轮换")
    
    def create_stealth_fingerprint(self) -> Dict[str, Any]:
        """创建隐身设备指纹"""
        # 随机化所有可能的指纹信息
        canvas_fingerprint = self._generate_canvas_fingerprint()
        webgl_fingerprint = self._generate_webgl_fingerprint()
        
        fingerprint = {
            "userAgent": self.ua.random,
            "language": random.choice(["en-US", "zh-CN", "ja-JP", "ko-KR"]),
            "platform": random.choice(["Win32", "MacIntel", "Linux x86_64"]),
            "screenWidth": random.choice([1920, 1366, 1440, 1536, 1600, 1680]),
            "screenHeight": random.choice([1080, 768, 900, 864, 900, 1050]),
            "colorDepth": random.choice([24, 32]),
            "pixelDepth": random.choice([24, 32]),
            "deviceMemory": random.choice([4, 8, 16, 32]),
            "hardwareConcurrency": random.choice([2, 4, 8, 12, 16]),
            "timezone": random.choice(["Asia/Shanghai", "America/New_York", "Europe/London"]),
            "canvas": canvas_fingerprint,
            "webgl": webgl_fingerprint,
            "deviceId": self.device_id,
            "sessionId": self.session_id,
            "timestamp": int(time.time() * 1000)
        }
        
        return fingerprint
    
    def _generate_canvas_fingerprint(self) -> str:
        """生成Canvas指纹"""
        # 模拟Canvas指纹
        canvas_data = f"canvas_{random.randint(100000, 999999)}"
        return hashlib.md5(canvas_data.encode()).hexdigest()
    
    def _generate_webgl_fingerprint(self) -> str:
        """生成WebGL指纹"""
        # 模拟WebGL指纹
        webgl_data = f"webgl_{random.randint(100000, 999999)}"
        return hashlib.md5(webgl_data.encode()).hexdigest()


class TrackingProtection:
    """追踪保护器"""
    
    def __init__(self):
        self.logger = logger.bind(name="tracking_protection")
        self.blocked_domains = set()
        self.blocked_scripts = set()
        
        # 常见的追踪域名和脚本
        self.tracking_domains = {
            "google-analytics.com", "googletagmanager.com", "facebook.com",
            "doubleclick.net", "adnxs.com", "googlesyndication.com",
            "amazon-adsystem.com", "criteo.com", "taboola.com"
        }
        
        self.tracking_scripts = {
            "gtag", "ga", "fbq", "twq", "snap", "pintrk", "ttq"
        }
    
    def is_tracking_domain(self, domain: str) -> bool:
        """检查是否为追踪域名"""
        return any(tracking_domain in domain.lower() for tracking_domain in self.tracking_domains)
    
    def is_tracking_script(self, script: str) -> bool:
        """检查是否为追踪脚本"""
        return any(tracking_script in script.lower() for tracking_script in self.tracking_scripts)
    
    def block_tracking_request(self, url: str) -> bool:
        """阻止追踪请求"""
        if self.is_tracking_domain(url):
            self.blocked_domains.add(url)
            self.logger.warning(f"阻止追踪域名: {url}")
            return True
        return False
    
    def sanitize_response(self, response_text: str) -> str:
        """清理响应中的追踪代码"""
        # 移除常见的追踪脚本
        for script in self.tracking_scripts:
            response_text = response_text.replace(f'"{script}"', '""')
            response_text = response_text.replace(f"'{script}'", "''")
        
        return response_text


class PrivacyProtection:
    """隐私保护器"""
    
    def __init__(self):
        self.logger = logger.bind(name="privacy_protection")
        
        # 敏感信息模式
        self.sensitive_patterns = [
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # 信用卡
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 邮箱
            r'\b\d{10,11}\b',  # 手机号
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 邮箱
        ]
    
    def sanitize_data(self, data: Any) -> Any:
        """清理敏感数据"""
        if isinstance(data, dict):
            return {k: self.sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_data(item) for item in data]
        elif isinstance(data, str):
            return self._sanitize_string(data)
        else:
            return data
    
    def _sanitize_string(self, text: str) -> str:
        """清理字符串中的敏感信息"""
        import re
        
        # 替换敏感信息
        for pattern in self.sensitive_patterns:
            text = re.sub(pattern, "[REDACTED]", text)
        
        return text
    
    def encrypt_sensitive_data(self, data: str, key: str = None) -> str:
        """加密敏感数据"""
        if key is None:
            key = os.urandom(32).hex()
        
        # 使用HMAC-SHA256加密
        signature = hmac.new(
            key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{signature[:16]}{data[:4]}***{data[-4:]}"
    
    def anonymize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """匿名化用户数据"""
        anonymized = {}
        
        for key, value in user_data.items():
            if key.lower() in ['password', 'token', 'secret', 'key', 'id']:
                anonymized[key] = "***"
            elif key.lower() in ['email', 'phone', 'address']:
                anonymized[key] = self._anonymize_value(value)
            else:
                anonymized[key] = value
        
        return anonymized
    
    def _anonymize_value(self, value: str) -> str:
        """匿名化单个值"""
        if not value:
            return value
        
        if '@' in value:  # 邮箱
            parts = value.split('@')
            return f"{parts[0][:2]}***@{parts[1]}"
        elif len(value) > 4:  # 其他敏感信息
            return f"{value[:2]}***{value[-2:]}"
        else:
            return "***"


class StealthSession:
    """隐身会话管理器"""
    
    def __init__(self):
        self.stealth_manager = StealthManager()
        self.tracking_protection = TrackingProtection()
        self.privacy_protection = PrivacyProtection()
        self.logger = logger.bind(name="stealth_session")
        
        # 会话状态
        self.session_start_time = time.time()
        self.request_count = 0
        self.last_rotation_time = time.time()
        
        # 轮换间隔
        self.rotation_interval = random.randint(300, 600)  # 5-10分钟
    
    def create_stealth_session(self) -> requests.Session:
        """创建隐身会话"""
        session = requests.Session()
        
        # 设置隐身请求头
        headers = self.stealth_manager.create_stealth_headers()
        session.headers.update(headers)
        
        # 设置隐身Cookie
        cookies = self.stealth_manager.create_stealth_cookies()
        for name, value in cookies.items():
            session.cookies.set(name, value)
        
        # 设置代理（如果需要）
        # session.proxies = self._get_stealth_proxy()
        
        return session
    
    def should_rotate_session(self) -> bool:
        """判断是否需要轮换会话"""
        current_time = time.time()
        return current_time - self.last_rotation_time > self.rotation_interval
    
    def rotate_session(self, session: requests.Session) -> requests.Session:
        """轮换会话"""
        # 轮换隐身身份
        self.stealth_manager.rotate_stealth_identity()
        
        # 创建新会话
        new_session = self.create_stealth_session()
        
        self.last_rotation_time = time.time()
        self.rotation_interval = random.randint(300, 600)  # 重新设置轮换间隔
        
        self.logger.info("隐身会话已轮换")
        return new_session
    
    def make_stealth_request(self, session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
        """发送隐身请求"""
        # 检查是否需要轮换会话
        if self.should_rotate_session():
            session = self.rotate_session(session)
        
        # 阻止追踪请求
        if self.tracking_protection.block_tracking_request(url):
            raise Exception("追踪请求被阻止")
        
        # 添加隐身延迟
        self.stealth_manager.add_stealth_delay()
        
        # 添加隐身参数
        if 'params' not in kwargs:
            kwargs['params'] = {}
        kwargs['params'].update(self.stealth_manager.create_stealth_params())
        
        # 发送请求
        response = session.request(method, url, **kwargs)
        
        # 清理响应中的追踪代码
        if hasattr(response, 'text'):
            response._content = self.tracking_protection.sanitize_response(response.text).encode()
        
        self.request_count += 1
        return response
    
    def _get_stealth_proxy(self) -> Optional[Dict[str, str]]:
        """获取隐身代理"""
        # 这里可以集成代理池
        # 目前返回None，表示不使用代理
        return None
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        return {
            "session_id": self.stealth_manager.session_id,
            "device_id": self.stealth_manager.device_id,
            "request_count": self.request_count,
            "session_duration": time.time() - self.session_start_time,
            "last_rotation": self.last_rotation_time
        } 