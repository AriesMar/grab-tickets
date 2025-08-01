"""
安全配置模块
"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field
import time


class SecurityConfig(BaseModel):
    """安全配置"""
    
    # 反检测配置
    enable_anti_detection: bool = Field(default=True, description="启用反检测")
    enable_random_delay: bool = Field(default=True, description="启用随机延迟")
    enable_fingerprint_randomization: bool = Field(default=True, description="启用指纹随机化")
    enable_session_rotation: bool = Field(default=True, description="启用会话轮换")
    
    # 频率限制
    max_requests_per_minute: int = Field(default=30, description="每分钟最大请求数")
    min_delay_between_requests: float = Field(default=1.0, description="请求间最小延迟(秒)")
    max_delay_between_requests: float = Field(default=3.0, description="请求间最大延迟(秒)")
    
    # 人性化配置
    enable_human_behavior_simulation: bool = Field(default=True, description="启用人性化行为模拟")
    human_delay_probability: float = Field(default=0.3, description="人性化延迟概率")
    max_human_delay: float = Field(default=2.0, description="最大人性化延迟(秒)")
    
    # 验证码处理
    enable_captcha_detection: bool = Field(default=True, description="启用验证码检测")
    captcha_retry_times: int = Field(default=3, description="验证码重试次数")
    captcha_timeout: int = Field(default=30, description="验证码超时时间(秒)")
    
    # 安全验证
    enable_security_check_detection: bool = Field(default=True, description="启用安全验证检测")
    security_check_timeout: int = Field(default=60, description="安全验证超时时间(秒)")
    
    # 代理配置
    enable_proxy: bool = Field(default=False, description="启用代理")
    proxy_list: List[str] = Field(default=[], description="代理列表")
    proxy_rotation_interval: int = Field(default=300, description="代理轮换间隔(秒)")
    
    # 日志安全
    enable_sensitive_data_protection: bool = Field(default=True, description="启用敏感数据保护")
    log_sanitization: bool = Field(default=True, description="日志数据清理")
    
    # 设备指纹
    enable_device_fingerprint: bool = Field(default=True, description="启用设备指纹")
    fingerprint_rotation_interval: int = Field(default=600, description="指纹轮换间隔(秒)")
    
    # 会话管理
    session_timeout: int = Field(default=3600, description="会话超时时间(秒)")
    session_rotation_interval: int = Field(default=300, description="会话轮换间隔(秒)")
    
    # 错误处理
    max_consecutive_failures: int = Field(default=5, description="最大连续失败次数")
    failure_backoff_multiplier: float = Field(default=2.0, description="失败退避倍数")
    
    class Config:
        schema_extra = {
            "example": {
                "enable_anti_detection": True,
                "enable_random_delay": True,
                "max_requests_per_minute": 30,
                "min_delay_between_requests": 1.0,
                "max_delay_between_requests": 3.0,
                "enable_human_behavior_simulation": True,
                "enable_captcha_detection": True,
                "enable_security_check_detection": True
            }
        }


class SecurityRules:
    """安全规则"""
    
    @staticmethod
    def get_default_rules() -> Dict[str, Any]:
        """获取默认安全规则"""
        return {
            "request_patterns": {
                "max_same_url_requests": 10,  # 同一URL最大请求次数
                "max_concurrent_requests": 5,  # 最大并发请求数
                "min_request_interval": 1.0,   # 最小请求间隔
            },
            "behavior_patterns": {
                "enable_mouse_movement": True,  # 启用鼠标移动模拟
                "enable_keyboard_delay": True,  # 启用键盘延迟
                "enable_scroll_simulation": True,  # 启用滚动模拟
            },
            "detection_evasion": {
                "enable_webdriver_detection_evasion": True,  # 避免WebDriver检测
                "enable_automation_detection_evasion": True,  # 避免自动化检测
                "enable_fingerprint_randomization": True,  # 指纹随机化
            }
        }
    
    @staticmethod
    def get_suspicious_patterns() -> List[Dict[str, Any]]:
        """获取可疑行为模式"""
        return [
            {
                "name": "high_frequency",
                "description": "高频请求",
                "threshold": 50,
                "time_window": 60,
                "severity": "high"
            },
            {
                "name": "repetitive_pattern",
                "description": "重复模式",
                "threshold": 10,
                "time_window": 60,
                "severity": "medium"
            },
            {
                "name": "rapid_succession",
                "description": "快速连续操作",
                "threshold": 5,
                "time_window": 10,
                "severity": "high"
            }
        ]
    
    @staticmethod
    def get_safety_checks() -> List[Dict[str, Any]]:
        """获取安全检查列表"""
        return [
            {
                "name": "rate_limit_check",
                "description": "频率限制检查",
                "enabled": True,
                "priority": "high"
            },
            {
                "name": "captcha_detection",
                "description": "验证码检测",
                "enabled": True,
                "priority": "high"
            },
            {
                "name": "security_check_detection",
                "description": "安全验证检测",
                "enabled": True,
                "priority": "medium"
            },
            {
                "name": "response_validation",
                "description": "响应验证",
                "enabled": True,
                "priority": "medium"
            },
            {
                "name": "session_rotation",
                "description": "会话轮换",
                "enabled": True,
                "priority": "low"
            }
        ]


class SecurityMonitor:
    """安全监控器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.violation_count = 0
        self.last_violation_time = 0
        self.suspicious_activities = []
    
    def check_security_violation(self, activity: Dict[str, Any]) -> bool:
        """检查安全违规"""
        # 检查频率限制
        if self._check_rate_limit_violation(activity):
            return True
        
        # 检查可疑模式
        if self._check_suspicious_pattern(activity):
            return True
        
        # 检查安全验证
        if self._check_security_verification(activity):
            return True
        
        return False
    
    def _check_rate_limit_violation(self, activity: Dict[str, Any]) -> bool:
        """检查频率限制违规"""
        # 实现频率限制检查逻辑
        return False
    
    def _check_suspicious_pattern(self, activity: Dict[str, Any]) -> bool:
        """检查可疑模式"""
        # 实现可疑模式检查逻辑
        return False
    
    def _check_security_verification(self, activity: Dict[str, Any]) -> bool:
        """检查安全验证"""
        # 实现安全验证检查逻辑
        return False
    
    def record_violation(self, violation_type: str, details: Dict[str, Any]):
        """记录违规"""
        self.violation_count += 1
        self.last_violation_time = time.time()
        
        violation = {
            "type": violation_type,
            "details": details,
            "timestamp": time.time(),
            "count": self.violation_count
        }
        
        self.suspicious_activities.append(violation)
    
    def get_security_report(self) -> Dict[str, Any]:
        """获取安全报告"""
        return {
            "total_violations": self.violation_count,
            "last_violation_time": self.last_violation_time,
            "suspicious_activities": self.suspicious_activities,
            "security_level": self._calculate_security_level()
        }
    
    def _calculate_security_level(self) -> str:
        """计算安全等级"""
        if self.violation_count == 0:
            return "safe"
        elif self.violation_count < 3:
            return "warning"
        elif self.violation_count < 10:
            return "danger"
        else:
            return "critical" 