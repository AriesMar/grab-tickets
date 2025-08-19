"""
数据模型定义
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TicketStatus(str, Enum):
    """票务状态枚举"""
    AVAILABLE = "available"      # 有票
    SOLD_OUT = "sold_out"        # 售罄
    BOOKING = "booking"          # 预订中
    UNKNOWN = "unknown"          # 未知状态


class PlatformType(str, Enum):
    """平台类型枚举"""
    DAMAI = "damai"              # 大麦网
    TAOBAO = "taobao"            # 淘宝
    JD = "jd"                    # 京东
    CUSTOM = "custom"            # 自定义平台


class TicketInfo(BaseModel):
    """票务信息"""
    event_id: str = Field(..., description="活动ID")
    event_name: str = Field(..., description="活动名称")
    venue: str = Field(..., description="场馆")
    date: datetime = Field(..., description="演出日期")
    price_range: str = Field(..., description="价格范围")
    status: TicketStatus = Field(default=TicketStatus.UNKNOWN, description="票务状态")
    available_seats: Optional[int] = Field(default=None, description="可用座位数")
    total_seats: Optional[int] = Field(default=None, description="总座位数")
    url: str = Field(..., description="票务链接")
    platform: PlatformType = Field(..., description="平台类型")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TicketRequest(BaseModel):
    """抢票请求"""
    event_id: str = Field(..., description="活动ID")
    performance_id: Optional[str] = Field(default=None, description="场次ID")
    performance_keywords: Optional[List[str]] = Field(default=None, description="场次关键词（如日期/时间/关键字）")
    device_id: Optional[str] = Field(default=None, description="设备序列号(ADB序列/IP:端口)，用于多设备并发")
    auto_start: bool = Field(default=False, description="是否自动开始")
    start_time: Optional[str] = Field(default=None, description="自动开始时间（本地时间，格式 YYYY-MM-DD HH:MM:SS）")
    start_offset_seconds: Optional[int] = Field(default=None, description="相对当前的开始时间（秒），与 start_time 二选一")
    platform: PlatformType = Field(..., description="平台类型")
    target_price: Optional[float] = Field(default=None, description="目标价格")
    seat_preference: Optional[List[str]] = Field(default=None, description="座位偏好")
    quantity: int = Field(default=1, description="抢票数量")
    retry_times: int = Field(default=3, description="重试次数")
    retry_interval: float = Field(default=1.0, description="重试间隔(秒)")
    
    class Config:
        schema_extra = {
            "example": {
                "event_id": "123456",
                "performance_id": "987654",
                "performance_keywords": ["2024-12-31", "19:30"],
                "device_id": "192.168.1.23:5555",
                "auto_start": True,
                "start_time": "2024-12-31 19:00:00",
                "start_offset_seconds": 300,
                "platform": "damai",
                "target_price": 580.0,
                "seat_preference": ["A区", "B区"],
                "quantity": 2,
                "retry_times": 3,
                "retry_interval": 1.0
            }
        }


class GrabResult(BaseModel):
    """抢票结果"""
    success: bool = Field(..., description="是否成功")
    ticket_id: Optional[str] = Field(default=None, description="票务ID")
    order_id: Optional[str] = Field(default=None, description="订单ID")
    message: str = Field(default="", description="结果消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    details: Optional[Dict[str, Any]] = Field(default=None, description="详细信息")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserConfig(BaseModel):
    """用户配置"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    phone: Optional[str] = Field(default=None, description="手机号")
    email: Optional[str] = Field(default=None, description="邮箱")
    auto_login: bool = Field(default=True, description="自动登录")
    session_timeout: int = Field(default=3600, description="会话超时时间(秒)")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "your_username",
                "password": "your_password",
                "phone": "13800138000",
                "email": "user@example.com",
                "auto_login": True,
                "session_timeout": 3600
            }
        }


class PlatformConfig(BaseModel):
    """平台配置"""
    platform: PlatformType = Field(..., description="平台类型")
    base_url: str = Field(..., description="基础URL")
    api_endpoints: Dict[str, str] = Field(default_factory=dict, description="API端点")
    headers: Dict[str, str] = Field(default_factory=dict, description="请求头")
    timeout: int = Field(default=30, description="超时时间(秒)")
    max_retries: int = Field(default=3, description="最大重试次数")
    rate_limit: float = Field(default=1.0, description="请求频率限制(秒)")
    # 以下为部分平台（如大麦APP）所需的可选字段
    app_package: Optional[str] = Field(default=None, description="移动端应用包名")
    app_activity: Optional[str] = Field(default=None, description="移动端应用主活动/入口")
    device_id: Optional[str] = Field(default=None, description="设备ID")
    session_token: Optional[str] = Field(default=None, description="会话令牌")
    user_info: Optional[Dict[str, Any]] = Field(default=None, description="用户信息")
    
    class Config:
        schema_extra = {
            "example": {
                "platform": "damai",
                "base_url": "https://www.damai.cn",
                "api_endpoints": {
                    "search": "/api/search",
                    "detail": "/api/detail",
                    "order": "/api/order"
                },
                "headers": {
                    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
                },
                "timeout": 30,
                "max_retries": 3,
                "rate_limit": 1.0
            }
        } 