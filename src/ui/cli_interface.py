"""
命令行界面模块
"""
import asyncio
import sys
from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich import box

from ..core.grab_engine import GrabEngine
from ..data.config_manager import ConfigManager
from ..data.models import TicketRequest, PlatformType, UserConfig, GrabResult
from ..platforms.damai_adapter import DamaiMobileAdapter, DamaiConfig
from ..utils.logger import get_logger, set_log_level
from ..utils.task_scheduler import TaskScheduler
from ..utils.metrics import start_metrics_server
from .web_console import WebConsoleServer


class CLIInterface:
    """命令行界面"""
    
    def __init__(self):
        self.console = Console()
        self.config_manager = ConfigManager()
        self.grab_engine = GrabEngine()
        self.logger = get_logger("cli_interface")
        self.scheduler: TaskScheduler | None = None
        self.web_console: WebConsoleServer | None = None
        self.is_running = False
        
        # 初始化配置
        self._init_config()
    
    def _init_config(self):
        """初始化配置"""
        try:
            # 设置日志级别
            log_level = self.config_manager.get_setting("log_level", "INFO")
            set_log_level(log_level)
            
            # 添加大麦网适配器
            damai_config = DamaiConfig()
            damai_adapter = DamaiMobileAdapter(damai_config)
            self.grab_engine.add_strategy(damai_adapter)
            
            # 启动 Prometheus 指标端口（可在 settings 中配置端口，默认 8001）
            metrics_port = int(self.config_manager.get_setting("metrics_port", 8001))
            start_metrics_server(metrics_port)

            self.logger.info("CLI界面初始化完成")
            # 启动任务调度器（后台自动抢）
            self.scheduler = TaskScheduler(self.config_manager, self.grab_engine)
            self.scheduler.start()

            # 启动 Web 控制台（可配置开关与端口）
            web_enabled = bool(self.config_manager.get_setting("web_console_enabled", True))
            if web_enabled:
                web_port = int(self.config_manager.get_setting("web_port", 8080))
                self.web_console = WebConsoleServer(self.config_manager, self.grab_engine, port=web_port)
                self.web_console.start()
        except Exception as e:
            self.logger.error(f"CLI界面初始化失败: {e}")
    
    def show_welcome(self):
        """显示欢迎界面"""
        welcome_text = Text()
        welcome_text.append("🎫 通用抢票软件框架", style="bold blue")
        welcome_text.append("\n")
        welcome_text.append("支持大麦网等平台的自动抢票功能", style="green")
        welcome_text.append("\n")
        welcome_text.append("请选择操作:", style="yellow")
        
        panel = Panel(
            Align.center(welcome_text),
            title="欢迎使用",
            border_style="blue",
            box=box.ROUNDED
        )
        
        self.console.print(panel)
    
    def show_main_menu(self):
        """显示主菜单"""
        menu_items = [
            "1. 配置管理",
            "2. 抢票任务管理", 
            "3. 开始抢票",
            "4. 查看状态",
            "5. 设置",
            "6. 帮助",
            "0. 退出"
        ]
        
        menu_text = Text()
        for item in menu_items:
            menu_text.append(item + "\n", style="cyan")
        
        panel = Panel(
            Align.left(menu_text),
            title="主菜单",
            border_style="green",
            box=box.ROUNDED
        )
        
        self.console.print(panel)
    
    def get_user_choice(self, prompt: str = "请选择操作", choices: List[str] = None) -> str:
        """获取用户选择"""
        if choices:
            for i, choice in enumerate(choices):
                self.console.print(f"{i+1}. {choice}")
        
        return Prompt.ask(prompt)
    
    def show_config_menu(self):
        """显示配置管理菜单"""
        while True:
            self.console.clear()
            self.console.print(Panel("配置管理", style="bold blue"))
            
            config_items = [
                "1. 用户配置",
                "2. 平台配置",
                "3. 验证配置",
                "4. 备份配置",
                "5. 恢复配置",
                "0. 返回主菜单"
            ]
            
            for item in config_items:
                self.console.print(item)
            
            choice = self.get_user_choice("请选择配置操作")
            
            if choice == "1":
                self._manage_user_config()
            elif choice == "2":
                self._manage_platform_config()
            elif choice == "3":
                self._validate_config()
            elif choice == "4":
                self._backup_config()
            elif choice == "5":
                self._restore_config()
            elif choice == "0":
                break
            else:
                self.console.print("无效选择，请重试", style="red")
    
    def _manage_user_config(self):
        """管理用户配置"""
        self.console.clear()
        self.console.print(Panel("用户配置管理", style="bold blue"))
        
        user_config = self.config_manager.load_user_config()
        
        if user_config:
            self.console.print(f"当前用户: {user_config.username}")
            self.console.print(f"手机号: {user_config.phone or '未设置'}")
            self.console.print(f"邮箱: {user_config.email or '未设置'}")
            
            if Confirm.ask("是否修改用户配置?"):
                self._edit_user_config(user_config)
        else:
            self.console.print("未找到用户配置")
            if Confirm.ask("是否创建新用户配置?"):
                self._create_user_config()
    
    def _create_user_config(self):
        """创建用户配置"""
        self.console.print("创建新用户配置")
        
        username = Prompt.ask("请输入用户名")
        password = Prompt.ask("请输入密码", password=True)
        phone = Prompt.ask("请输入手机号(可选)")
        email = Prompt.ask("请输入邮箱(可选)")
        
        config = UserConfig(
            username=username,
            password=password,
            phone=phone if phone else None,
            email=email if email else None
        )
        
        if self.config_manager.save_user_config(config):
            self.console.print("用户配置创建成功", style="green")
        else:
            self.console.print("用户配置创建失败", style="red")
    
    def _edit_user_config(self, config: UserConfig):
        """编辑用户配置"""
        self.console.print("编辑用户配置")
        
        new_username = Prompt.ask("用户名", default=config.username)
        new_password = Prompt.ask("密码", password=True, default=config.password)
        new_phone = Prompt.ask("手机号", default=config.phone or "")
        new_email = Prompt.ask("邮箱", default=config.email or "")
        
        config.username = new_username
        config.password = new_password
        config.phone = new_phone if new_phone else None
        config.email = new_email if new_email else None
        
        if self.config_manager.save_user_config(config):
            self.console.print("用户配置更新成功", style="green")
        else:
            self.console.print("用户配置更新失败", style="red")
    
    def _manage_platform_config(self):
        """管理平台配置"""
        self.console.clear()
        self.console.print(Panel("平台配置管理", style="bold blue"))
        
        platforms = list(PlatformType)
        for i, platform in enumerate(platforms):
            self.console.print(f"{i+1}. {platform.value}")
        
        choice = self.get_user_choice("请选择平台")
        
        try:
            platform_index = int(choice) - 1
            if 0 <= platform_index < len(platforms):
                platform = platforms[platform_index]
                self._edit_platform_config(platform)
            else:
                self.console.print("无效选择", style="red")
        except ValueError:
            self.console.print("请输入有效数字", style="red")
    
    def _edit_platform_config(self, platform: PlatformType):
        """编辑平台配置"""
        self.console.print(f"编辑 {platform.value} 平台配置")
        
        # 这里可以添加平台特定的配置编辑逻辑
        # 目前使用默认配置
        if platform == PlatformType.DAMAI:
            config = DamaiConfig()
            if self.config_manager.save_platform_config(platform, config):
                self.console.print(f"{platform.value} 平台配置保存成功", style="green")
            else:
                self.console.print(f"{platform.value} 平台配置保存失败", style="red")
    
    def _validate_config(self):
        """验证配置"""
        self.console.print("验证配置完整性...")
        
        errors = self.config_manager.validate_config()
        
        if any(errors.values()):
            self.console.print("配置验证失败:", style="red")
            for category, error_list in errors.items():
                if error_list:
                    self.console.print(f"\n{category}:", style="yellow")
                    for error in error_list:
                        self.console.print(f"  - {error}", style="red")
        else:
            self.console.print("配置验证通过", style="green")
    
    def _backup_config(self):
        """备份配置"""
        backup_dir = Prompt.ask("请输入备份目录路径", default="config/backup")
        
        if self.config_manager.backup_configs(backup_dir):
            self.console.print("配置备份成功", style="green")
        else:
            self.console.print("配置备份失败", style="red")
    
    def _restore_config(self):
        """恢复配置"""
        backup_dir = Prompt.ask("请输入备份目录路径")
        
        if self.config_manager.restore_configs(backup_dir):
            self.console.print("配置恢复成功", style="green")
        else:
            self.console.print("配置恢复失败", style="red")
    
    def show_ticket_menu(self):
        """显示抢票任务管理菜单"""
        while True:
            self.console.clear()
            self.console.print(Panel("抢票任务管理", style="bold blue"))
            
            ticket_items = [
                "1. 查看任务列表",
                "2. 添加抢票任务",
                "3. 删除抢票任务",
                "4. 编辑抢票任务",
                "0. 返回主菜单"
            ]
            
            for item in ticket_items:
                self.console.print(item)
            
            choice = self.get_user_choice("请选择操作")
            
            if choice == "1":
                self._show_ticket_list()
            elif choice == "2":
                self._add_ticket_task()
            elif choice == "3":
                self._delete_ticket_task()
            elif choice == "4":
                self._edit_ticket_task()
            elif choice == "0":
                break
            else:
                self.console.print("无效选择，请重试", style="red")
    
    def _show_ticket_list(self):
        """显示抢票任务列表"""
        requests = self.config_manager.load_ticket_requests()
        
        if not requests:
            self.console.print("暂无抢票任务")
            return
        
        table = Table(title="抢票任务列表")
        table.add_column("序号", style="cyan")
        table.add_column("活动ID", style="green")
        table.add_column("平台", style="blue")
        table.add_column("目标价格", style="yellow")
        table.add_column("数量", style="magenta")
        table.add_column("座位偏好", style="white")
        table.add_column("场次ID", style="cyan")
        table.add_column("场次关键词", style="cyan")
        table.add_column("设备ID", style="white")
        table.add_column("自动开始", style="green")
        table.add_column("开始时间", style="yellow")
        
        for i, request in enumerate(requests, 1):
            table.add_row(
                str(i),
                request.event_id,
                request.platform.value,
                str(request.target_price) if request.target_price else "不限",
                str(request.quantity),
                ", ".join(request.seat_preference) if request.seat_preference else "推荐",
                request.performance_id or "-",
                ", ".join(request.performance_keywords) if getattr(request, "performance_keywords", None) else "-",
                getattr(request, "device_id", "-") or "-",
                "是" if getattr(request, "auto_start", False) else "否",
                getattr(request, "start_time", "-") or "-"
            )
        
        self.console.print(table)
    
    def _add_ticket_task(self):
        """添加抢票任务"""
        self.console.print("添加抢票任务")
        
        event_id = Prompt.ask("请输入活动ID")
        performance_id = Prompt.ask("请输入场次ID(可选)", default="")
        performance_keywords = Prompt.ask("请输入场次关键词(逗号分隔，可选)", default="")
        device_id = Prompt.ask("请输入设备序列号(ADB序列/IP:端口，可选)", default="")
        
        # 选择平台
        platforms = list(PlatformType)
        for i, platform in enumerate(platforms):
            self.console.print(f"{i+1}. {platform.value}")
        
        platform_choice = self.get_user_choice("请选择平台")
        try:
            platform_index = int(platform_choice) - 1
            platform = platforms[platform_index]
        except (ValueError, IndexError):
            self.console.print("无效平台选择", style="red")
            return
        
        # 其他参数
        target_price = Prompt.ask("目标价格(可选)", default="")
        target_price = float(target_price) if target_price else None
        
        quantity = IntPrompt.ask("抢票数量", default=1)
        
        seat_preference = Prompt.ask("座位偏好(用逗号分隔，可选)", default="")
        seat_preference = [s.strip() for s in seat_preference.split(",")] if seat_preference else None
        performance_keywords = [s.strip() for s in performance_keywords.split(",")] if performance_keywords else None

        auto_start = Confirm.ask("是否自动开始?", default=False)
        start_time = None
        if auto_start:
            start_time = Prompt.ask("请输入自动开始时间(YYYY-MM-DD HH:MM:SS)", default="") or None
        
        retry_times = IntPrompt.ask("重试次数", default=3)
        retry_interval = FloatPrompt.ask("重试间隔(秒)", default=1.0)
        
        request = TicketRequest(
            event_id=event_id,
            performance_id=performance_id or None,
            platform=platform,
            target_price=target_price,
            seat_preference=seat_preference,
            performance_keywords=performance_keywords,
            device_id=device_id or None,
            auto_start=auto_start,
            start_time=start_time,
            quantity=quantity,
            retry_times=retry_times,
            retry_interval=retry_interval
        )
        
        if self.config_manager.add_ticket_request(request):
            self.console.print("抢票任务添加成功", style="green")
        else:
            self.console.print("抢票任务添加失败", style="red")
    
    def _delete_ticket_task(self):
        """删除抢票任务"""
        requests = self.config_manager.load_ticket_requests()
        
        if not requests:
            self.console.print("暂无抢票任务")
            return
        
        self._show_ticket_list()
        
        choice = self.get_user_choice("请选择要删除的任务序号")
        try:
            index = int(choice) - 1
            if 0 <= index < len(requests):
                request = requests[index]
                if Confirm.ask(f"确认删除活动 {request.event_id} 的抢票任务?"):
                    if self.config_manager.remove_ticket_request(request.event_id, request.platform):
                        self.console.print("任务删除成功", style="green")
                    else:
                        self.console.print("任务删除失败", style="red")
            else:
                self.console.print("无效序号", style="red")
        except ValueError:
            self.console.print("请输入有效数字", style="red")
    
    def _edit_ticket_task(self):
        """编辑抢票任务"""
        # 类似添加任务的逻辑，先选择任务再编辑
        self.console.print("编辑功能开发中...", style="yellow")
    
    def start_grab_tickets(self):
        """开始抢票"""
        self.console.clear()
        self.console.print(Panel("开始抢票", style="bold blue"))
        
        requests = self.config_manager.load_ticket_requests()
        
        if not requests:
            self.console.print("暂无抢票任务，请先添加任务", style="red")
            return
        
        self._show_ticket_list()
        
        if Confirm.ask("确认开始抢票?"):
            self.is_running = True
            
            # 添加回调函数
            self.grab_engine.add_callback("on_start", self._on_grab_start)
            self.grab_engine.add_callback("on_success", self._on_grab_success)
            self.grab_engine.add_callback("on_failure", self._on_grab_failure)
            self.grab_engine.add_callback("on_complete", self._on_grab_complete)
            
            try:
                # 运行抢票
                asyncio.run(self._run_grab_tickets(requests))
            except KeyboardInterrupt:
                self.console.print("\n用户中断抢票", style="yellow")
            finally:
                self.is_running = False
    
    async def _run_grab_tickets(self, requests: List[TicketRequest]):
        """运行抢票任务"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("正在抢票...", total=len(requests))
            
            results = await self.grab_engine.grab_tickets_batch(requests)
            
            # 显示结果
            self._show_grab_results(results)
    
    def _on_grab_start(self, request: TicketRequest):
        """抢票开始回调"""
        self.console.print(f"开始抢票: {request.event_id}", style="blue")
    
    def _on_grab_success(self, request: TicketRequest, result: GrabResult):
        """抢票成功回调"""
        self.console.print(f"抢票成功: {request.event_id}", style="green")
        self.console.print(f"订单ID: {result.order_id}", style="green")
    
    def _on_grab_failure(self, request: TicketRequest, result: GrabResult):
        """抢票失败回调"""
        self.console.print(f"抢票失败: {request.event_id}", style="red")
        self.console.print(f"原因: {result.message}", style="red")
    
    def _on_grab_complete(self, request: TicketRequest, result: GrabResult):
        """抢票完成回调"""
        self.console.print(f"任务完成: {request.event_id}", style="cyan")
    
    def _show_grab_results(self, results: List[GrabResult]):
        """显示抢票结果"""
        self.console.print("\n抢票结果:", style="bold")
        
        success_count = sum(1 for r in results if r.success)
        total_count = len(results)
        
        self.console.print(f"成功: {success_count}/{total_count}", style="green")
        
        table = Table(title="详细结果")
        table.add_column("活动ID", style="cyan")
        table.add_column("状态", style="green")
        table.add_column("消息", style="white")
        table.add_column("订单ID", style="blue")
        
        for result in results:
            status = "成功" if result.success else "失败"
            status_style = "green" if result.success else "red"
            
            table.add_row(
                result.ticket_id or "未知",
                status,
                result.message,
                result.order_id or "无"
            )
        
        self.console.print(table)
    
    def show_status(self):
        """显示状态"""
        self.console.print(Panel("系统状态", style="bold blue"))
        
        # 显示配置状态
        errors = self.config_manager.validate_config()
        config_status = "正常" if not any(errors.values()) else "异常"
        config_style = "green" if config_status == "正常" else "red"
        
        self.console.print(f"配置状态: {config_status}", style=config_style)
        
        # 显示任务状态
        requests = self.config_manager.load_ticket_requests()
        self.console.print(f"抢票任务数: {len(requests)}")
        
        # 显示引擎状态
        engine_status = "运行中" if self.is_running else "已停止"
        engine_style = "green" if self.is_running else "yellow"
        self.console.print(f"抢票引擎: {engine_status}", style=engine_style)
    
    def show_settings(self):
        """显示设置"""
        while True:
            self.console.clear()
            self.console.print(Panel("设置", style="bold blue"))
            
            settings = self.config_manager.load_settings()
            
            for key, value in settings.items():
                self.console.print(f"{key}: {value}")
            
            setting_items = [
                "1. 修改日志级别",
                "2. 修改最大工作线程数",
                "3. 修改默认重试次数",
                "0. 返回主菜单"
            ]
            
            for item in setting_items:
                self.console.print(item)
            
            choice = self.get_user_choice("请选择操作")
            
            if choice == "1":
                self._change_log_level()
            elif choice == "2":
                self._change_max_workers()
            elif choice == "3":
                self._change_retry_times()
            elif choice == "0":
                break
            else:
                self.console.print("无效选择，请重试", style="red")
    
    def _change_log_level(self):
        """修改日志级别"""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        for i, level in enumerate(levels):
            self.console.print(f"{i+1}. {level}")
        
        choice = self.get_user_choice("请选择日志级别")
        try:
            level_index = int(choice) - 1
            if 0 <= level_index < len(levels):
                level = levels[level_index]
                if self.config_manager.update_setting("log_level", level):
                    set_log_level(level)
                    self.console.print("日志级别修改成功", style="green")
                else:
                    self.console.print("日志级别修改失败", style="red")
            else:
                self.console.print("无效选择", style="red")
        except ValueError:
            self.console.print("请输入有效数字", style="red")
    
    def _change_max_workers(self):
        """修改最大工作线程数"""
        try:
            max_workers = IntPrompt.ask("请输入最大工作线程数", default=5)
            if self.config_manager.update_setting("max_workers", max_workers):
                self.console.print("最大工作线程数修改成功", style="green")
            else:
                self.console.print("最大工作线程数修改失败", style="red")
        except ValueError:
            self.console.print("请输入有效数字", style="red")
    
    def _change_retry_times(self):
        """修改默认重试次数"""
        try:
            retry_times = IntPrompt.ask("请输入默认重试次数", default=3)
            if self.config_manager.update_setting("default_retry_times", retry_times):
                self.console.print("默认重试次数修改成功", style="green")
            else:
                self.console.print("默认重试次数修改失败", style="red")
        except ValueError:
            self.console.print("请输入有效数字", style="red")
    
    def show_help(self):
        """显示帮助"""
        help_text = """
        通用抢票软件框架使用说明
        
        1. 配置管理
           - 用户配置：设置用户名、密码等个人信息
           - 平台配置：配置各平台的参数
           - 验证配置：检查配置完整性
           - 备份/恢复：备份和恢复配置文件
        
        2. 抢票任务管理
           - 添加任务：创建新的抢票任务
           - 查看任务：查看所有抢票任务
           - 删除任务：删除不需要的任务
        
        3. 开始抢票
           - 执行所有配置的抢票任务
           - 支持并发抢票
           - 自动重试机制
        
        4. 注意事项
           - 请确保手机已连接并开启USB调试
           - 请先在大麦网APP中登录
           - 合理设置抢票频率，避免对服务器造成压力
           - 遵守相关平台的使用条款
        
        5. 技术支持
           - 查看日志文件了解详细运行信息
           - 配置文件位于config目录
        """
        
        panel = Panel(help_text, title="帮助", border_style="blue")
        self.console.print(panel)
    
    def run(self):
        """运行CLI界面"""
        try:
            self.show_welcome()
            
            while True:
                self.show_main_menu()
                choice = self.get_user_choice("请选择操作")
                
                if choice == "1":
                    self.show_config_menu()
                elif choice == "2":
                    self.show_ticket_menu()
                elif choice == "3":
                    self.start_grab_tickets()
                elif choice == "4":
                    self.show_status()
                elif choice == "5":
                    self.show_settings()
                elif choice == "6":
                    self.show_help()
                elif choice == "0":
                    self.console.print("感谢使用，再见！", style="green")
                    break
                else:
                    self.console.print("无效选择，请重试", style="red")
                
                if choice != "0":
                    input("\n按回车键继续...")
        
        except KeyboardInterrupt:
            self.console.print("\n程序被用户中断", style="yellow")
        except Exception as e:
            self.logger.error(f"CLI界面运行异常: {e}")
            self.console.print(f"程序运行异常: {e}", style="red")
        finally:
            if self.scheduler:
                self.scheduler.stop()
            # Web 控制台后台线程随进程退出
            self.grab_engine.stop()