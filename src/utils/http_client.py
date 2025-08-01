"""
HTTP客户端工具模块
"""
import time
import random
from typing import Dict, Any, Optional, Union
import requests
from fake_useragent import UserAgent
from loguru import logger

from ..data.models import PlatformConfig
from .security import SecurityManager, AntiDetectionManager, SecurityValidator
from .advanced_anti_detection import StealthSession, PrivacyProtection


class HttpClient:
    """HTTP客户端"""
    
    def __init__(self, config: PlatformConfig):
        self.config = config
        self.session = requests.Session()
        self.ua = UserAgent()
        self.security_manager = SecurityManager()
        self.anti_detection = AntiDetectionManager()
        self.security_validator = SecurityValidator()
        self.stealth_session = StealthSession()
        self.privacy_protection = PrivacyProtection()
        self._setup_session()
    
    def _setup_session(self):
        """设置会话配置"""
        # 使用隐身会话管理器创建会话
        self.session = self.stealth_session.create_stealth_session()
        
        # 合并配置中的请求头
        self.session.headers.update(self.config.headers)
        
        # 设置超时
        self.session.timeout = self.config.timeout
        
        # 验证配置安全性
        is_safe, errors = self.security_validator.validate_config(self.config.dict())
        if not is_safe:
            logger.warning(f"配置安全性问题: {errors}")
        
        # 清理敏感信息
        self.config.headers = self.privacy_protection.sanitize_data(self.config.headers)
    
    def _rate_limit(self):
        """请求频率限制"""
        # 检查频率限制
        if not self.security_manager.check_rate_limit():
            raise Exception("请求频率超限")
        
        # 添加随机延迟
        self.security_manager.add_random_delay()
        
        if hasattr(self, '_last_request_time'):
            elapsed = time.time() - self._last_request_time
            if elapsed < self.config.rate_limit:
                sleep_time = self.config.rate_limit - elapsed + random.uniform(0.1, 0.5)
                time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def _update_user_agent(self):
        """更新User-Agent"""
        self.session.headers["User-Agent"] = self.security_manager.rotate_user_agent()
    
    def get(self, url: str, params: Optional[Dict[str, Any]] = None, 
            headers: Optional[Dict[str, str]] = None, **kwargs) -> requests.Response:
        """GET请求"""
        self._rate_limit()
        
        # 清理敏感参数
        if params:
            params = self.privacy_protection.sanitize_data(params)
        if headers:
            headers = self.privacy_protection.sanitize_data(headers)
        
        # 分析请求模式
        request_data = {
            "method": "GET",
            "url": url,
            "headers": headers or {},
            "params": params or {}
        }
        self.anti_detection.analyze_request_pattern(request_data)
        
        try:
            # 使用隐身请求
            response = self.stealth_session.make_stealth_request(
                self.session, "GET", url, 
                params=params, headers=headers, **kwargs
            )
            
            # 验证响应安全性
            if not self.security_manager.validate_response(response):
                raise Exception("响应安全性验证失败")
            
            # 检测验证码
            if self.security_manager.detect_captcha(response.text):
                captcha_result = self.security_manager.handle_captcha(response)
                logger.warning(f"检测到验证码: {captcha_result}")
            
            response.raise_for_status()
            logger.debug(f"GET请求成功: {url}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"GET请求失败: {url}, 错误: {e}")
            raise
    
    def post(self, url: str, data: Optional[Dict[str, Any]] = None, 
             json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, 
             **kwargs) -> requests.Response:
        """POST请求"""
        self._rate_limit()
        self._update_user_agent()
        
        try:
            response = self.session.post(
                url, 
                data=data, 
                json=json, 
                headers=headers,
                **kwargs
            )
            response.raise_for_status()
            logger.debug(f"POST请求成功: {url}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"POST请求失败: {url}, 错误: {e}")
            raise
    
    def put(self, url: str, data: Optional[Dict[str, Any]] = None, 
            json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, 
            **kwargs) -> requests.Response:
        """PUT请求"""
        self._rate_limit()
        self._update_user_agent()
        
        try:
            response = self.session.put(
                url, 
                data=data, 
                json=json, 
                headers=headers,
                **kwargs
            )
            response.raise_for_status()
            logger.debug(f"PUT请求成功: {url}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"PUT请求失败: {url}, 错误: {e}")
            raise
    
    def delete(self, url: str, headers: Optional[Dict[str, str]] = None, 
               **kwargs) -> requests.Response:
        """DELETE请求"""
        self._rate_limit()
        self._update_user_agent()
        
        try:
            response = self.session.delete(url, headers=headers, **kwargs)
            response.raise_for_status()
            logger.debug(f"DELETE请求成功: {url}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"DELETE请求失败: {url}, 错误: {e}")
            raise
    
    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """通用请求方法"""
        self._rate_limit()
        self._update_user_agent()
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            logger.debug(f"{method}请求成功: {url}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"{method}请求失败: {url}, 错误: {e}")
            raise
    
    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None, 
                 headers: Optional[Dict[str, str]] = None, **kwargs) -> Dict[str, Any]:
        """GET请求并返回JSON数据"""
        response = self.get(url, params, headers, **kwargs)
        return response.json()
    
    def post_json(self, url: str, data: Optional[Dict[str, Any]] = None, 
                  json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, 
                  **kwargs) -> Dict[str, Any]:
        """POST请求并返回JSON数据"""
        response = self.post(url, data, json, headers, **kwargs)
        return response.json()
    
    def download_file(self, url: str, file_path: str, chunk_size: int = 8192) -> bool:
        """下载文件"""
        try:
            response = self.get(url, stream=True)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            logger.info(f"文件下载成功: {file_path}")
            return True
        except Exception as e:
            logger.error(f"文件下载失败: {url}, 错误: {e}")
            return False
    
    def close(self):
        """关闭会话"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RetryHttpClient(HttpClient):
    """带重试机制的HTTP客户端"""
    
    def __init__(self, config: PlatformConfig, max_retries: int = None):
        super().__init__(config)
        self.max_retries = max_retries or config.max_retries
    
    def _retry_request(self, request_func, *args, **kwargs):
        """重试请求"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return request_func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"请求失败，{wait_time:.2f}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"请求最终失败，已重试 {self.max_retries} 次: {e}")
        
        raise last_exception
    
    def get(self, url: str, params: Optional[Dict[str, Any]] = None, 
            headers: Optional[Dict[str, str]] = None, **kwargs) -> requests.Response:
        """带重试的GET请求"""
        return self._retry_request(super().get, url, params, headers, **kwargs)
    
    def post(self, url: str, data: Optional[Dict[str, Any]] = None, 
             json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, 
             **kwargs) -> requests.Response:
        """带重试的POST请求"""
        return self._retry_request(super().post, url, data, json, headers, **kwargs) 