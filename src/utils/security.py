"""
安全和反检测模块
"""
import time
import random
import hashlib
import hmac
import base64
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import requests
from fake_useragent import UserAgent
from loguru import logger


class SecurityManager:
    """安全管理器"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.request_history = []
        self.fingerprint_cache = {}
        self.logger = logger.bind(name="security_manager")
        
        # 反检测配置
        self.anti_detection_config = {
            "enable_random_delay": True,
            "enable_fingerprint_randomization": True,
            "enable_request_rotation": True,
            "enable_session_rotation": True,
            "max_requests_per_minute": 30,
            "min_delay_between_requests": 1.0,
            "max_delay_between_requests": 3.0
        }
    
    def generate_device_fingerprint(self) -> Dict[str, Any]:
        """生成设备指纹"""
        try:
            # 模拟真实设备的指纹信息
            screen_resolutions = [
                (1080, 1920), (1440, 2560), (720, 1280), (800, 1280)
            ]
            screen_res = random.choice(screen_resolutions)
            
            # 生成随机设备ID
            device_id = hashlib.md5(f"device_{random.randint(100000, 999999)}".encode()).hexdigest()
            
            # 生成随机时区
            timezones = ["Asia/Shanghai", "Asia/Beijing", "Asia/Hong_Kong"]
            timezone = random.choice(timezones)
            
            fingerprint = {
                "userAgent": self.ua.random,
                "screenWidth": screen_res[0],
                "screenHeight": screen_res[1],
                "colorDepth": random.choice([24, 32]),
                "pixelDepth": random.choice([24, 32]),
                "deviceMemory": random.choice([4, 8, 16]),
                "hardwareConcurrency": random.choice([4, 8, 12, 16]),
                "timezone": timezone,
                "language": random.choice(["zh-CN", "zh-TW", "en-US"]),
                "platform": random.choice(["Win32", "MacIntel", "Linux x86_64"]),
                "deviceId": device_id,
                "timestamp": int(time.time() * 1000)
            }
            
            return fingerprint
        except Exception as e:
            self.logger.error(f"生成设备指纹失败: {e}")
            return {}
    
    def generate_request_signature(self, data: Dict[str, Any], secret_key: str) -> str:
        """生成请求签名"""
        try:
            # 按键排序
            sorted_data = dict(sorted(data.items()))
            
            # 构建签名字符串
            sign_string = "&".join([f"{k}={v}" for k, v in sorted_data.items()])
            
            # 使用HMAC-SHA256生成签名
            signature = hmac.new(
                secret_key.encode('utf-8'),
                sign_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
        except Exception as e:
            self.logger.error(f"生成请求签名失败: {e}")
            return ""
    
    def add_random_delay(self, min_delay: float = None, max_delay: float = None):
        """添加随机延迟"""
        if not self.anti_detection_config["enable_random_delay"]:
            return
        
        min_delay = min_delay or self.anti_detection_config["min_delay_between_requests"]
        max_delay = max_delay or self.anti_detection_config["max_delay_between_requests"]
        
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
        self.logger.debug(f"添加随机延迟: {delay:.2f}秒")
    
    def rotate_user_agent(self) -> str:
        """轮换User-Agent"""
        user_agents = [
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
            "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
            "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Mobile Safari/537.36",
            "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Mobile Safari/537.36"
        ]
        return random.choice(user_agents)
    
    def generate_headers(self, include_fingerprint: bool = True) -> Dict[str, str]:
        """生成请求头"""
        headers = {
            "User-Agent": self.rotate_user_agent(),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "X-Requested-With": "XMLHttpRequest"
        }
        
        if include_fingerprint:
            fingerprint = self.generate_device_fingerprint()
            headers["X-Device-Fingerprint"] = base64.b64encode(
                json.dumps(fingerprint).encode()
            ).decode()
        
        return headers
    
    def check_rate_limit(self, max_requests: int = None) -> bool:
        """检查请求频率限制"""
        max_requests = max_requests or self.anti_detection_config["max_requests_per_minute"]
        current_time = time.time()
        
        # 清理超过1分钟的历史记录
        self.request_history = [
            req_time for req_time in self.request_history 
            if current_time - req_time < 60
        ]
        
        # 检查是否超过限制
        if len(self.request_history) >= max_requests:
            self.logger.warning(f"请求频率超限: {len(self.request_history)}/{max_requests}")
            return False
        
        self.request_history.append(current_time)
        return True
    
    def validate_response(self, response: requests.Response) -> bool:
        """验证响应安全性"""
        try:
            # 检查状态码
            if response.status_code not in [200, 201, 202]:
                self.logger.warning(f"响应状态码异常: {response.status_code}")
                return False
            
            # 检查响应头
            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type and 'text/html' not in content_type:
                self.logger.warning(f"响应内容类型异常: {content_type}")
                return False
            
            # 检查响应大小
            if len(response.content) > 10 * 1024 * 1024:  # 10MB
                self.logger.warning("响应内容过大")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"验证响应失败: {e}")
            return False
    
    def detect_captcha(self, response_text: str) -> bool:
        """检测验证码"""
        captcha_indicators = [
            "验证码", "captcha", "验证", "verify", "图片验证", "滑动验证",
            "请验证", "安全验证", "人机验证", "reCAPTCHA"
        ]
        
        for indicator in captcha_indicators:
            if indicator.lower() in response_text.lower():
                self.logger.warning(f"检测到验证码: {indicator}")
                return True
        
        return False
    
    def handle_captcha(self, response: requests.Response) -> Dict[str, Any]:
        """处理验证码"""
        self.logger.info("检测到验证码，需要人工处理")
        
        # 这里可以集成验证码识别服务
        # 目前返回需要人工处理的信息
        return {
            "requires_manual": True,
            "captcha_type": "unknown",
            "message": "检测到验证码，请人工处理"
        }


class AntiDetectionManager:
    """反检测管理器"""
    
    def __init__(self):
        self.security_manager = SecurityManager()
        self.logger = logger.bind(name="anti_detection")
        
        # 会话轮换配置
        self.session_rotation_interval = 300  # 5分钟
        self.last_session_rotation = time.time()
        
        # 请求模式分析
        self.request_patterns = []
        self.suspicious_activities = []
    
    def should_rotate_session(self) -> bool:
        """判断是否需要轮换会话"""
        current_time = time.time()
        return current_time - self.last_session_rotation > self.session_rotation_interval
    
    def rotate_session(self, session: requests.Session) -> requests.Session:
        """轮换会话"""
        try:
            # 创建新会话
            new_session = requests.Session()
            
            # 更新请求头
            new_headers = self.security_manager.generate_headers()
            new_session.headers.update(new_headers)
            
            # 设置代理（如果需要）
            # new_session.proxies = self.get_proxy()
            
            self.last_session_rotation = time.time()
            self.logger.info("会话轮换完成")
            
            return new_session
        except Exception as e:
            self.logger.error(f"会话轮换失败: {e}")
            return session
    
    def analyze_request_pattern(self, request_data: Dict[str, Any]):
        """分析请求模式"""
        pattern = {
            "timestamp": time.time(),
            "method": request_data.get("method", "GET"),
            "url": request_data.get("url", ""),
            "headers": request_data.get("headers", {}),
            "params": request_data.get("params", {})
        }
        
        self.request_patterns.append(pattern)
        
        # 保持最近100个请求的记录
        if len(self.request_patterns) > 100:
            self.request_patterns.pop(0)
    
    def detect_suspicious_activity(self) -> List[Dict[str, Any]]:
        """检测可疑活动"""
        suspicious = []
        current_time = time.time()
        
        # 检查请求频率
        recent_requests = [
            p for p in self.request_patterns 
            if current_time - p["timestamp"] < 60
        ]
        
        if len(recent_requests) > 50:
            suspicious.append({
                "type": "high_frequency",
                "message": f"请求频率过高: {len(recent_requests)}/分钟",
                "severity": "high"
            })
        
        # 检查请求模式
        unique_urls = set(p["url"] for p in recent_requests)
        if len(unique_urls) < 3 and len(recent_requests) > 10:
            suspicious.append({
                "type": "repetitive_pattern",
                "message": "请求模式过于单一",
                "severity": "medium"
            })
        
        return suspicious
    
    def get_proxy(self) -> Optional[Dict[str, str]]:
        """获取代理"""
        # 这里可以集成代理池服务
        # 目前返回None，表示不使用代理
        return None
    
    def encrypt_sensitive_data(self, data: str, key: str) -> str:
        """加密敏感数据"""
        try:
            # 使用简单的base64编码（实际应用中应使用更强的加密）
            encoded = base64.b64encode(data.encode()).decode()
            return encoded
        except Exception as e:
            self.logger.error(f"加密数据失败: {e}")
            return data
    
    def decrypt_sensitive_data(self, encrypted_data: str, key: str) -> str:
        """解密敏感数据"""
        try:
            # 使用简单的base64解码
            decoded = base64.b64decode(encrypted_data.encode()).decode()
            return decoded
        except Exception as e:
            self.logger.error(f"解密数据失败: {e}")
            return encrypted_data


class SecurityValidator:
    """安全验证器"""
    
    def __init__(self):
        self.logger = logger.bind(name="security_validator")
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证配置安全性"""
        errors = []
        
        # 检查敏感信息
        sensitive_fields = ["password", "token", "secret", "key"]
        for field in sensitive_fields:
            if field in str(config).lower():
                errors.append(f"配置中包含敏感字段: {field}")
        
        # 检查URL安全性
        urls = self.extract_urls(config)
        for url in urls:
            if not url.startswith(("http://", "https://")):
                errors.append(f"不安全的URL: {url}")
        
        # 检查超时设置
        timeout = config.get("timeout", 30)
        if timeout < 1 or timeout > 300:
            errors.append(f"超时设置不合理: {timeout}")
        
        return len(errors) == 0, errors
    
    def extract_urls(self, data: Any) -> List[str]:
        """提取URL"""
        urls = []
        if isinstance(data, dict):
            for value in data.values():
                urls.extend(self.extract_urls(value))
        elif isinstance(data, list):
            for item in data:
                urls.extend(self.extract_urls(item))
        elif isinstance(data, str) and ("http://" in data or "https://" in data):
            urls.append(data)
        
        return urls
    
    def sanitize_log_data(self, data: Any) -> Any:
        """清理日志数据中的敏感信息"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in ["password", "token", "secret", "key"]):
                    sanitized[key] = "***"
                else:
                    sanitized[key] = self.sanitize_log_data(value)
            return sanitized
        elif isinstance(data, list):
            return [self.sanitize_log_data(item) for item in data]
        else:
            return data 