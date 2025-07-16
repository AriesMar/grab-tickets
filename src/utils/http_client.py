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


class HttpClient:
    """HTTP客户端"""
    
    def __init__(self, config: PlatformConfig):
        self.config = config
        self.session = requests.Session()
        self.ua = UserAgent()
        self._setup_session()
    
    def _setup_session(self):
        """设置会话配置"""
        # 设置默认请求头
        default_headers = {
            "User-Agent": self.ua.random,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # 合并配置中的请求头
        default_headers.update(self.config.headers)
        self.session.headers.update(default_headers)
        
        # 设置超时
        self.session.timeout = self.config.timeout
    
    def _rate_limit(self):
        """请求频率限制"""
        if hasattr(self, '_last_request_time'):
            elapsed = time.time() - self._last_request_time
            if elapsed < self.config.rate_limit:
                sleep_time = self.config.rate_limit - elapsed + random.uniform(0.1, 0.5)
                time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def _update_user_agent(self):
        """更新User-Agent"""
        self.session.headers["User-Agent"] = self.ua.random
    
    def get(self, url: str, params: Optional[Dict[str, Any]] = None, 
            headers: Optional[Dict[str, str]] = None, **kwargs) -> requests.Response:
        """GET请求"""
        self._rate_limit()
        self._update_user_agent()
        
        try:
            response = self.session.get(
                url, 
                params=params, 
                headers=headers,
                **kwargs
            )
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