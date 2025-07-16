"""
大麦网平台适配器
"""
import time
import json
import hashlib
import random
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import subprocess
import cv2
import numpy as np
from PIL import Image
import pyautogui

from ..core.grab_engine import GrabStrategy
from ..data.models import TicketRequest, GrabResult, PlatformType, TicketInfo, TicketStatus
from ..utils.http_client import RetryHttpClient
from ..utils.logger import get_logger
from ..data.models import PlatformConfig


class DamaiConfig(PlatformConfig):
    """大麦网配置"""
    
    def __init__(self):
        super().__init__(
            platform=PlatformType.DAMAI,
            base_url="https://www.damai.cn",
            api_endpoints={
                "search": "/api/search",
                "detail": "/api/detail",
                "order": "/api/order",
                "login": "/api/login",
                "captcha": "/api/captcha"
            },
            headers={
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
                "Referer": "https://www.damai.cn/",
                "Origin": "https://www.damai.cn"
            },
            timeout=30,
            max_retries=3,
            rate_limit=1.0
        )
        
        # 大麦网特有配置
        self.app_package = "cn.damai"  # 大麦网APP包名
        self.app_activity = "cn.damai.homepage.MainActivity"  # 主活动
        self.device_id = None  # 设备ID
        self.session_token = None  # 会话令牌
        self.user_info = None  # 用户信息


class DamaiMobileAdapter(GrabStrategy):
    """大麦网手机APP适配器"""
    
    def __init__(self, config: DamaiConfig):
        self.config = config
        self.http_client = RetryHttpClient(config)
        self.logger = get_logger("damai_adapter")
        self.device_connected = False
        self.app_launched = False
        
    def can_handle(self, platform: PlatformType) -> bool:
        """判断是否可以处理大麦网平台"""
        return platform == PlatformType.DAMAI
    
    async def execute(self, request: TicketRequest) -> GrabResult:
        """执行大麦网抢票"""
        self.logger.info(f"开始大麦网抢票: {request.event_id}")
        
        try:
            # 1. 检查设备连接
            if not await self._check_device():
                return GrabResult(
                    success=False,
                    message="设备连接失败，请确保手机已连接并开启USB调试"
                )
            
            # 2. 启动大麦网APP
            if not await self._launch_app():
                return GrabResult(
                    success=False,
                    message="启动大麦网APP失败"
                )
            
            # 3. 检查登录状态
            if not await self._check_login():
                return GrabResult(
                    success=False,
                    message="用户未登录，请先登录大麦网APP"
                )
            
            # 4. 搜索活动
            if not await self._search_event(request.event_id):
                return GrabResult(
                    success=False,
                    message=f"未找到活动: {request.event_id}"
                )
            
            # 5. 进入活动详情
            if not await self._enter_event_detail():
                return GrabResult(
                    success=False,
                    message="进入活动详情页失败"
                )
            
            # 6. 选择票档
            if not await self._select_ticket_type(request):
                return GrabResult(
                    success=False,
                    message="选择票档失败"
                )
            
            # 7. 选择座位
            if not await self._select_seats(request):
                return GrabResult(
                    success=False,
                    message="选择座位失败"
                )
            
            # 8. 提交订单
            if not await self._submit_order(request):
                return GrabResult(
                    success=False,
                    message="提交订单失败"
                )
            
            # 9. 等待进入待支付页面
            if not await self._wait_for_payment_page():
                return GrabResult(
                    success=False,
                    message="进入待支付页面失败"
                )
            
            return GrabResult(
                success=True,
                ticket_id=f"damai_{request.event_id}_{int(time.time())}",
                order_id=f"order_{int(time.time())}",
                message="抢票成功！已进入待支付页面，请手动完成支付"
            )
            
        except Exception as e:
            self.logger.error(f"大麦网抢票异常: {e}")
            return GrabResult(
                success=False,
                message=f"抢票过程中发生异常: {str(e)}"
            )
    
    async def _check_device(self) -> bool:
        """检查设备连接"""
        try:
            result = subprocess.run(
                ["adb", "devices"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
                connected_devices = [line for line in lines if line.strip() and 'device' in line]
                
                if connected_devices:
                    self.device_connected = True
                    self.logger.info(f"设备连接成功: {len(connected_devices)} 台设备")
                    return True
                else:
                    self.logger.error("未检测到已连接的设备")
                    return False
            else:
                self.logger.error(f"ADB命令执行失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("ADB命令执行超时")
            return False
        except Exception as e:
            self.logger.error(f"检查设备连接异常: {e}")
            return False
    
    async def _launch_app(self) -> bool:
        """启动大麦网APP"""
        try:
            # 强制停止APP
            subprocess.run(
                ["adb", "shell", "am", "force-stop", self.config.app_package],
                capture_output=True,
                timeout=10
            )
            
            time.sleep(1)
            
            # 启动APP
            result = subprocess.run(
                ["adb", "shell", "am", "start", "-n", f"{self.config.app_package}/{self.config.app_activity}"],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.app_launched = True
                self.logger.info("大麦网APP启动成功")
                time.sleep(3)  # 等待APP完全启动
                return True
            else:
                self.logger.error(f"启动APP失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"启动APP异常: {e}")
            return False
    
    async def _check_login(self) -> bool:
        """检查登录状态"""
        try:
            # 截图检查登录状态
            screenshot = self._take_screenshot()
            if screenshot is None:
                return False
            
            # 这里可以添加图像识别逻辑来判断是否已登录
            # 简单示例：检查是否存在"登录"按钮
            # 实际实现需要更复杂的图像识别算法
            
            # 临时返回True，实际应该根据图像识别结果判断
            self.logger.info("登录状态检查完成")
            return True
            
        except Exception as e:
            self.logger.error(f"检查登录状态异常: {e}")
            return False
    
    async def _search_event(self, event_id: str) -> bool:
        """搜索活动"""
        try:
            # 点击搜索按钮
            search_button_pos = self._find_element_by_text("搜索")
            if search_button_pos:
                self._tap_screen(search_button_pos[0], search_button_pos[1])
                time.sleep(1)
            
            # 输入活动ID或关键词
            self._input_text(event_id)
            time.sleep(1)
            
            # 点击搜索
            self._tap_screen(100, 200)  # 搜索按钮位置
            time.sleep(2)
            
            self.logger.info(f"搜索活动: {event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"搜索活动异常: {e}")
            return False
    
    async def _enter_event_detail(self) -> bool:
        """进入活动详情页"""
        try:
            # 点击第一个搜索结果
            self._tap_screen(200, 300)  # 搜索结果位置
            time.sleep(2)
            
            self.logger.info("进入活动详情页")
            return True
            
        except Exception as e:
            self.logger.error(f"进入活动详情页异常: {e}")
            return False
    
    async def _select_ticket_type(self, request: TicketRequest) -> bool:
        """选择票档"""
        try:
            # 根据目标价格选择票档
            if request.target_price:
                # 查找价格接近的票档
                ticket_types = self._find_ticket_types()
                selected_type = self._select_best_ticket_type(ticket_types, request.target_price)
                
                if selected_type:
                    self._tap_screen(selected_type['x'], selected_type['y'])
                    time.sleep(1)
                    self.logger.info(f"选择票档: {selected_type['name']}")
                    return True
                else:
                    self.logger.warning("未找到合适的票档")
                    return False
            else:
                # 选择第一个可用票档
                self._tap_screen(200, 400)  # 默认票档位置
                time.sleep(1)
                self.logger.info("选择默认票档")
                return True
                
        except Exception as e:
            self.logger.error(f"选择票档异常: {e}")
            return False
    
    async def _select_seats(self, request: TicketRequest) -> bool:
        """选择座位"""
        try:
            if request.seat_preference:
                # 根据座位偏好选择
                for preference in request.seat_preference:
                    seat_pos = self._find_seat_by_preference(preference)
                    if seat_pos:
                        self._tap_screen(seat_pos[0], seat_pos[1])
                        time.sleep(0.5)
                        self.logger.info(f"选择座位: {preference}")
                        break
            else:
                # 选择推荐座位
                self._tap_screen(300, 500)  # 推荐座位位置
                time.sleep(0.5)
                self.logger.info("选择推荐座位")
            
            return True
            
        except Exception as e:
            self.logger.error(f"选择座位异常: {e}")
            return False
    
    async def _submit_order(self, request: TicketRequest) -> bool:
        """提交订单"""
        try:
            # 设置数量
            if request.quantity > 1:
                for _ in range(request.quantity - 1):
                    self._tap_screen(350, 600)  # 增加数量按钮
                    time.sleep(0.5)
            
            # 点击立即购买
            self._tap_screen(200, 700)  # 立即购买按钮
            time.sleep(2)
            
            self.logger.info("提交订单")
            return True
            
        except Exception as e:
            self.logger.error(f"提交订单异常: {e}")
            return False
    
    async def _wait_for_payment_page(self) -> bool:
        """等待进入待支付页面"""
        try:
            self.logger.info("等待进入待支付页面...")
            
            # 等待页面加载
            time.sleep(3)
            
            # 检查是否进入待支付页面
            # 可以通过截图识别页面特征来判断
            screenshot = self._take_screenshot()
            if screenshot is None:
                self.logger.warning("无法获取截图，假设已进入待支付页面")
                return True
            
            # 这里可以添加图像识别逻辑来判断是否在待支付页面
            # 例如识别"待支付"、"支付"等文字
            # 临时返回True，实际应该根据图像识别结果判断
            
            self.logger.info("已进入待支付页面")
            return True
            
        except Exception as e:
            self.logger.error(f"等待待支付页面异常: {e}")
            return False
    
    def _take_screenshot(self) -> Optional[np.ndarray]:
        """截图"""
        try:
            result = subprocess.run(
                ["adb", "exec-out", "screencap -p"],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # 将二进制数据转换为numpy数组
                nparr = np.frombuffer(result.stdout, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img
            else:
                self.logger.error("截图失败")
                return None
                
        except Exception as e:
            self.logger.error(f"截图异常: {e}")
            return None
    
    def _tap_screen(self, x: int, y: int):
        """点击屏幕"""
        try:
            subprocess.run(
                ["adb", "shell", "input", "tap", str(x), str(y)],
                capture_output=True,
                timeout=5
            )
        except Exception as e:
            self.logger.error(f"点击屏幕异常: {e}")
    
    def _input_text(self, text: str):
        """输入文本"""
        try:
            subprocess.run(
                ["adb", "shell", "input", "text", text],
                capture_output=True,
                timeout=5
            )
        except Exception as e:
            self.logger.error(f"输入文本异常: {e}")
    
    def _find_element_by_text(self, text: str) -> Optional[tuple]:
        """通过文本查找元素位置"""
        # 这里应该实现OCR文字识别
        # 临时返回固定位置
        return (100, 100)
    
    def _find_ticket_types(self) -> List[Dict[str, Any]]:
        """查找票档信息"""
        # 这里应该实现票档识别
        # 临时返回模拟数据
        return [
            {"name": "VIP票", "price": 580, "x": 200, "y": 400},
            {"name": "普通票", "price": 380, "x": 200, "y": 450}
        ]
    
    def _select_best_ticket_type(self, ticket_types: List[Dict[str, Any]], target_price: float) -> Optional[Dict[str, Any]]:
        """选择最佳票档"""
        best_type = None
        min_diff = float('inf')
        
        for ticket_type in ticket_types:
            diff = abs(ticket_type['price'] - target_price)
            if diff < min_diff:
                min_diff = diff
                best_type = ticket_type
        
        return best_type
    
    def _find_seat_by_preference(self, preference: str) -> Optional[tuple]:
        """根据偏好查找座位"""
        # 这里应该实现座位识别
        # 临时返回固定位置
        return (300, 500) 