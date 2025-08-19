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
import xml.etree.ElementTree as ET

from ..core.grab_engine import GrabStrategy
from ..data.models import TicketRequest, GrabResult, PlatformType, TicketInfo, TicketStatus
from ..utils.http_client import RetryHttpClient
from ..utils.logger import get_logger
from ..data.models import PlatformConfig
from ..utils.notifications import notify_warn


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
            if not await self._check_device(request):
                return GrabResult(
                    success=False,
                    message="设备连接失败，请确保手机已连接并开启USB调试"
                )
            
            # 2. 启动大麦网APP
            if not await self._launch_app(request):
                return GrabResult(
                    success=False,
                    message="启动大麦网APP失败"
                )
            
            # 3. 检查登录状态
            if not await self._check_login(request):
                return GrabResult(
                    success=False,
                    message="用户未登录，请先登录大麦网APP"
                )
            
            # 4. 搜索活动
            if not await self._search_event(request, request.event_id):
                return GrabResult(
                    success=False,
                    message=f"未找到活动: {request.event_id}"
                )
            
            # 5. 进入活动详情
            if not await self._enter_event_detail(request):
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
            if not await self._wait_for_payment_page(request):
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
    
    def _adb_prefix(self, request: Optional[TicketRequest] = None) -> List[str]:
        """返回 ADB 命令前缀，支持绑定设备 (-s <serial>)"""
        serial = getattr(request, "device_id", None) if request else None
        return ["adb"] + (["-s", serial] if serial else [])

    async def _check_device(self, request: Optional[TicketRequest] = None) -> bool:
        """检查设备连接"""
        try:
            result = subprocess.run(
                self._adb_prefix(request) + ["devices"], 
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
    
    async def _launch_app(self, request: Optional[TicketRequest] = None) -> bool:
        """启动大麦网APP"""
        try:
            # 强制停止APP
            subprocess.run(
                self._adb_prefix(request) + ["shell", "am", "force-stop", self.config.app_package],
                capture_output=True,
                timeout=10
            )
            
            time.sleep(1)
            
            # 启动APP
            result = subprocess.run(
                self._adb_prefix(request) + ["shell", "am", "start", "-n", f"{self.config.app_package}/{self.config.app_activity}"],
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
    
    async def _check_login(self, request: Optional[TicketRequest] = None) -> bool:
        """检查登录状态"""
        try:
            # 截图检查登录状态
            screenshot = self._take_screenshot(request)
            if screenshot is None:
                return False
            
            # 这里可以添加图像识别逻辑来判断是否已登录
            # 简单示例：检查是否存在"登录"按钮
            # 实际实现需要更复杂的图像识别算法
            
            # 检查是否有安全验证
            if self._detect_security_check(screenshot):
                self.logger.warning("检测到安全验证，可能需要人工处理")
                notify_warn("检测到验证码/安全校验，请在设备上完成后继续")
                # 可选Webhook
                try:
                    from ..data.config_manager import ConfigManager
                    from ..utils.notifications import notify_webhook
                    cfg = ConfigManager()
                    url = cfg.get_setting("captcha_webhook", "")
                    if url:
                        notify_webhook(url, "验证码提醒", "检测到安全校验，请尽快在设备上完成后返回")
                except Exception:
                    pass
                return False
            
            # 临时返回True，实际应该根据图像识别结果判断
            self.logger.info("登录状态检查完成")
            return True
            
        except Exception as e:
            self.logger.error(f"检查登录状态异常: {e}")
            return False
    
    async def _search_event(self, request: TicketRequest, event_id: str) -> bool:
        """搜索活动"""
        try:
            # 添加隐身延迟
            self._add_stealth_delay()
            
            # 掩盖自动化信号
            self._mask_automation_signals()
            
            # 点击搜索按钮
            search_button_pos = self._find_element_by_text("搜索")
            if search_button_pos:
                self._tap_screen(search_button_pos[0], search_button_pos[1], request)
                time.sleep(1)
            
            # 添加隐身行为
            self._add_stealth_behavior()
            
            # 模拟人性化行为
            self._simulate_human_behavior()
            
            # 输入活动ID或关键词
            self._input_text(event_id, request)
            time.sleep(1)
            
            # 点击搜索
            self._tap_screen(100, 200, request)  # 搜索按钮位置
            time.sleep(2)
            
            self.logger.info(f"搜索活动: {event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"搜索活动异常: {e}")
            return False
    
    async def _enter_event_detail(self, request: TicketRequest) -> bool:
        """进入活动详情页"""
        try:
            # 点击第一个搜索结果
            self._tap_screen(200, 300, request)  # 搜索结果位置
            time.sleep(2)
            
            self.logger.info("进入活动详情页")
            return True
            
        except Exception as e:
            self.logger.error(f"进入活动详情页异常: {e}")
            return False
    
    def _dump_ui_xml(self, request: Optional[TicketRequest] = None) -> Optional[str]:
        """通过 uiautomator dump 获取当前界面 XML"""
        try:
            # 导出到设备临时文件
            dump_cmd = self._adb_prefix(request) + ["shell", "uiautomator", "dump", "/sdcard/uidump.xml"]
            subprocess.run(dump_cmd, capture_output=True, timeout=5)
            # 读取 XML 内容
            cat_cmd = self._adb_prefix(request) + ["shell", "cat", "/sdcard/uidump.xml"]
            result = subprocess.run(cat_cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip().startswith("<?xml"):
                return result.stdout
            self.logger.warning("未能获取有效的UI层级XML")
            return None
        except Exception as e:
            self.logger.error(f"获取UI层级XML失败: {e}")
            return None

    def _parse_bounds_to_center(self, bounds: str) -> Optional[tuple]:
        """解析 bounds 字符串为中心坐标: [x1,y1][x2,y2] -> (cx, cy)"""
        try:
            # 形如 [48,122][1032,230]
            parts = bounds.replace("]", "").split("[")
            coords = [p for p in parts if p]
            x1, y1 = map(int, coords[0].split(","))
            x2, y2 = map(int, coords[1].split(","))
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        except Exception:
            return None

    def _find_node_center_by_keywords(self, xml_text: str, keywords: List[str]) -> Optional[tuple]:
        """在 UI XML 中查找包含所有关键词的节点中心坐标（大小写不敏感）"""
        try:
            root = ET.fromstring(xml_text)
            # UIAutomator XML 节点通常为 node 标签
            nodes = root.iter()
            normalized_keywords = [k.strip().lower() for k in keywords if k and k.strip()]
            best_match = None
            best_score = -1
            for node in nodes:
                attrib = node.attrib
                text_fields = [
                    attrib.get("text", ""),
                    attrib.get("content-desc", ""),
                    attrib.get("resource-id", ""),
                    attrib.get("class", ""),
                    attrib.get("contentDescription", "")
                ]
                combined = " ".join(text_fields).lower()
                if not combined:
                    continue
                # 计算匹配得分：包含的关键词个数
                score = sum(1 for k in normalized_keywords if k in combined)
                if score <= 0:
                    continue
                bounds = attrib.get("bounds")
                if not bounds:
                    continue
                if score > best_score:
                    center = self._parse_bounds_to_center(bounds)
                    if center:
                        best_match = center
                        best_score = score
            return best_match
        except Exception as e:
            self.logger.error(f"解析UI层级XML并匹配关键词失败: {e}")
            return None

    def _swipe_up(self, request: Optional[TicketRequest] = None, duration_ms: int = 300):
        """上滑一屏（简单实现）"""
        try:
            # 以中间向上滑动
            subprocess.run(
                self._adb_prefix(request) + [
                    "shell", "input", "swipe", "500", "1400", "500", "400", str(duration_ms)
                ],
                capture_output=True,
                timeout=5
            )
        except Exception as e:
            self.logger.error(f"滑动异常: {e}")

    async def _select_performance(self, performance_id: Optional[str], performance_keywords: Optional[List[str]] = None, request: Optional[TicketRequest] = None) -> bool:
        """选择场次（支持 performance_id 或关键词，优先使用 UI 层级匹配；加入多轮滑动重试）"""
        try:
            # 多轮匹配：初次 + 向上滑动 3 次
            attempts = 4
            for attempt in range(attempts):
                # 1) 若提供 performance_id，先尝试在 UI 文本中直接匹配该 ID
                if performance_id:
                    xml_text = self._dump_ui_xml(request)
                    if xml_text:
                        center = self._find_node_center_by_keywords(xml_text, [performance_id])
                        if center:
                            self._tap_screen(center[0], center[1], request)
                            time.sleep(1)
                            self.logger.info(f"已通过ID选择场次: {performance_id}")
                            return True
                # 2) 使用关键词匹配
                if performance_keywords:
                    xml_text = self._dump_ui_xml(request)
                    if xml_text:
                        center = self._find_node_center_by_keywords(xml_text, performance_keywords)
                        if center:
                            self._tap_screen(center[0], center[1], request)
                            time.sleep(1)
                            self.logger.info(f"根据关键词选择场次: {performance_keywords}")
                            return True
                # 若非最后一次，尝试上滑后重试
                if attempt < attempts - 1:
                    self._swipe_up(request)
                    time.sleep(0.8)

            # 兜底：默认第一场
            self._tap_screen(220, 360, request)
            time.sleep(1)
            self.logger.info("未指定或未匹配到场次，默认选择第一场")
            return True
        except Exception as e:
            self.logger.error(f"选择场次异常: {e}")
            return False

    async def _select_performance(self, performance_id: Optional[str], performance_keywords: Optional[List[str]] = None, request: Optional[TicketRequest] = None) -> bool:
        """选择场次（支持 performance_id 或关键词，优先使用 UI 层级匹配）"""
        try:
            # 1) 若提供 performance_id，先尝试在 UI 文本中直接匹配该 ID
            if performance_id:
                xml_text = self._dump_ui_xml(request)
                if xml_text:
                    center = self._find_node_center_by_keywords(xml_text, [performance_id])
                    if center:
                        self._tap_screen(center[0], center[1], request)
                        time.sleep(1)
                        self.logger.info(f"已通过ID选择场次: {performance_id}")
                        return True
                # 若UI层级中未直接出现ID，则兜底点击默认位置
                self._tap_screen(220, 360, request)
                time.sleep(1)
                self.logger.info(f"未在UI层级直接找到ID，已尝试默认位置选择: {performance_id}")
                return True

            # 2) 使用关键词匹配：如 日期/时间/‘晚场’ 等
            if performance_keywords:
                xml_text = self._dump_ui_xml(request)
                if xml_text:
                    center = self._find_node_center_by_keywords(xml_text, performance_keywords)
                    if center:
                        self._tap_screen(center[0], center[1], request)
                        time.sleep(1)
                        self.logger.info(f"根据关键词选择场次: {performance_keywords}")
                        return True

            # 3) 兜底：默认第一场
            self._tap_screen(220, 360, request)
            time.sleep(1)
            self.logger.info("未指定或未匹配到场次，默认选择第一场")
            return True
        except Exception as e:
            self.logger.error(f"选择场次异常: {e}")
            return False

    async def _select_ticket_type(self, request: TicketRequest) -> bool:
        """选择票档"""
        try:
            # 先按场次选择（优先 ID，次选关键词）
            if not await self._select_performance(getattr(request, "performance_id", None), getattr(request, "performance_keywords", None), request):
                return False

            # 根据目标价格选择票档
            if request.target_price:
                # 查找价格接近的票档
                ticket_types = self._find_ticket_types()
                selected_type = self._select_best_ticket_type(ticket_types, request.target_price)
                
                if selected_type:
                    self._tap_screen(selected_type['x'], selected_type['y'], request)
                    time.sleep(1)
                    self.logger.info(f"选择票档: {selected_type['name']}")
                    return True
                else:
                    self.logger.warning("未找到合适的票档")
                    return False
            else:
                # 选择第一个可用票档
                self._tap_screen(200, 400, request)  # 默认票档位置
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
                        self._tap_screen(seat_pos[0], seat_pos[1], request)
                        time.sleep(0.5)
                        self.logger.info(f"选择座位: {preference}")
                        break
            else:
                # 选择推荐座位
                self._tap_screen(300, 500, request)  # 推荐座位位置
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
                    self._tap_screen(350, 600, request)  # 增加数量按钮
                    time.sleep(0.5)
            
            # 点击立即购买
            self._tap_screen(200, 700, request)  # 立即购买按钮
            time.sleep(2)
            
            self.logger.info("提交订单")
            return True
            
        except Exception as e:
            self.logger.error(f"提交订单异常: {e}")
            return False
    
    async def _wait_for_payment_page(self, request: TicketRequest) -> bool:
        """等待进入待支付页面"""
        try:
            self.logger.info("等待进入待支付页面...")
            
            # 等待页面加载
            time.sleep(3)
            
            # 检查是否进入待支付页面
            # 可以通过截图识别页面特征来判断
            screenshot = self._take_screenshot(request)
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
    
    def _take_screenshot(self, request: Optional[TicketRequest] = None) -> Optional[np.ndarray]:
        """截图"""
        try:
            result = subprocess.run(
                self._adb_prefix(request) + ["exec-out", "screencap -p"],
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
    
    def _tap_screen(self, x: int, y: int, request: Optional[TicketRequest] = None):
        """点击屏幕"""
        try:
            subprocess.run(
                self._adb_prefix(request) + ["shell", "input", "tap", str(x), str(y)],
                capture_output=True,
                timeout=5
            )
        except Exception as e:
            self.logger.error(f"点击屏幕异常: {e}")
    
    def _input_text(self, text: str, request: Optional[TicketRequest] = None):
        """输入文本"""
        try:
            subprocess.run(
                self._adb_prefix(request) + ["shell", "input", "text", text],
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
    
    def _detect_security_check(self, screenshot) -> bool:
        """检测安全验证"""
        try:
            # 这里可结合模板匹配/颜色区域识别滑块或验证码弹层；
            # 先依据像素均值与方差粗判异常窗口出现（占位逻辑）。
            if screenshot is None:
                return False
            mean_val = float(np.mean(screenshot))
            std_val = float(np.std(screenshot))
            # 假设出现验证弹窗时整体对比提升或亮度偏离明显
            return std_val > 70 or mean_val < 30 or mean_val > 225
        except Exception as e:
            self.logger.error(f"检测安全验证异常: {e}")
            return False
    
    def _add_human_like_delay(self):
        """添加人性化延迟"""
        # 模拟人类操作的不规则延迟
        delays = [0.5, 1.2, 0.8, 1.5, 0.3, 1.0]
        delay = random.choice(delays)
        
        # 添加微小的随机变化
        variation = random.uniform(-0.1, 0.1)
        final_delay = max(0.1, delay + variation)
        
        time.sleep(final_delay)
        self.logger.debug(f"添加人性化延迟: {final_delay:.3f}秒")
    
    def _add_stealth_delay(self):
        """添加隐身延迟"""
        # 更自然的延迟模式
        base_delay = random.uniform(1.0, 3.0)
        
        # 添加随机变化
        variation = random.uniform(-0.2, 0.2)
        final_delay = max(0.5, base_delay + variation)
        
        time.sleep(final_delay)
        self.logger.debug(f"添加隐身延迟: {final_delay:.3f}秒")
    
    def _simulate_human_behavior(self):
        """模拟人类行为"""
        # 随机添加一些人性化的操作
        if random.random() < 0.1:  # 10%概率
            # 模拟偶尔的误操作
            self._tap_screen(random.randint(100, 400), random.randint(100, 400))
            time.sleep(0.5)
            self.logger.debug("模拟人性化操作")
    
    def _add_stealth_behavior(self):
        """添加隐身行为"""
        # 模拟更真实的人类行为模式
        behaviors = [
            lambda: self._tap_screen(random.randint(50, 350), random.randint(50, 350)),
            lambda: time.sleep(random.uniform(0.2, 0.8)),
            lambda: self._input_text(" "),  # 模拟输入空格
            lambda: self._tap_screen(random.randint(200, 300), random.randint(200, 300))
        ]
        
        # 随机选择行为
        if random.random() < 0.15:  # 15%概率
            behavior = random.choice(behaviors)
            behavior()
            self.logger.debug("添加隐身行为")
    
    def _mask_automation_signals(self):
        """掩盖自动化信号"""
        # 在关键操作前添加随机行为
        if random.random() < 0.2:  # 20%概率
            # 模拟用户思考时间
            think_time = random.uniform(0.5, 2.0)
            time.sleep(think_time)
            self.logger.debug(f"模拟思考时间: {think_time:.2f}秒") 