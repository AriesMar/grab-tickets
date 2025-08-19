"""
配置管理模块
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

from .models import UserConfig, PlatformConfig, PlatformType, TicketRequest


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logger.bind(name="config_manager")
        
        # 配置文件路径
        self.user_config_file = self.config_dir / "user_config.json"
        self.platform_config_file = self.config_dir / "platform_config.json"
        self.ticket_requests_file = self.config_dir / "ticket_requests.json"
        self.settings_file = self.config_dir / "settings.json"
        
        # 默认设置
        self.default_settings = {
            "log_level": "INFO",
            "log_dir": "logs",
            "max_workers": 5,
            "default_retry_times": 3,
            "default_retry_interval": 1.0,
            "auto_save_config": True,
            "enable_notifications": True,
            # 监控与控制台
            "metrics_port": 8001,
            "web_console_enabled": True,
            "web_port": 8080,
            "web_token": "",  # 留空则不鉴权
            # 通知
            "captcha_webhook": "",  # 可选：验证码通知的Webhook地址
            # 调度周期
            "scheduler_interval_seconds": 1
        }
    
    def load_user_config(self) -> Optional[UserConfig]:
        """加载用户配置"""
        try:
            if self.user_config_file.exists():
                with open(self.user_config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return UserConfig(**data)
            else:
                self.logger.warning("用户配置文件不存在")
                return None
        except Exception as e:
            self.logger.error(f"加载用户配置失败: {e}")
            return None
    
    def save_user_config(self, config: UserConfig) -> bool:
        """保存用户配置"""
        try:
            with open(self.user_config_file, 'w', encoding='utf-8') as f:
                json.dump(config.dict(), f, ensure_ascii=False, indent=2)
            self.logger.info("用户配置保存成功")
            return True
        except Exception as e:
            self.logger.error(f"保存用户配置失败: {e}")
            return False
    
    def create_user_config(self, username: str, password: str, **kwargs) -> UserConfig:
        """创建用户配置"""
        config = UserConfig(
            username=username,
            password=password,
            **kwargs
        )
        
        if self.save_user_config(config):
            self.logger.info(f"创建用户配置成功: {username}")
        else:
            self.logger.error(f"创建用户配置失败: {username}")
        
        return config
    
    def load_platform_config(self, platform: PlatformType) -> Optional[PlatformConfig]:
        """加载平台配置"""
        try:
            if self.platform_config_file.exists():
                with open(self.platform_config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    platform_data = data.get(platform.value)
                    if platform_data:
                        return PlatformConfig(**platform_data)
                    else:
                        self.logger.warning(f"未找到平台配置: {platform}")
                        return None
            else:
                self.logger.warning("平台配置文件不存在")
                return None
        except Exception as e:
            self.logger.error(f"加载平台配置失败: {e}")
            return None
    
    def save_platform_config(self, platform: PlatformType, config: PlatformConfig) -> bool:
        """保存平台配置"""
        try:
            # 读取现有配置
            all_configs = {}
            if self.platform_config_file.exists():
                with open(self.platform_config_file, 'r', encoding='utf-8') as f:
                    all_configs = json.load(f)
            
            # 更新指定平台配置
            all_configs[platform.value] = config.dict()
            
            # 保存所有配置
            with open(self.platform_config_file, 'w', encoding='utf-8') as f:
                json.dump(all_configs, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"平台配置保存成功: {platform}")
            return True
        except Exception as e:
            self.logger.error(f"保存平台配置失败: {e}")
            return False
    
    def load_all_platform_configs(self) -> Dict[PlatformType, PlatformConfig]:
        """加载所有平台配置"""
        configs = {}
        try:
            if self.platform_config_file.exists():
                with open(self.platform_config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for platform_str, config_data in data.items():
                        try:
                            platform = PlatformType(platform_str)
                            configs[platform] = PlatformConfig(**config_data)
                        except ValueError:
                            self.logger.warning(f"无效的平台类型: {platform_str}")
                            continue
        except Exception as e:
            self.logger.error(f"加载平台配置失败: {e}")
        
        return configs
    
    def load_ticket_requests(self) -> List[TicketRequest]:
        """加载抢票请求配置"""
        try:
            if self.ticket_requests_file.exists():
                with open(self.ticket_requests_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [TicketRequest(**item) for item in data]
            else:
                self.logger.warning("抢票请求配置文件不存在")
                return []
        except Exception as e:
            self.logger.error(f"加载抢票请求配置失败: {e}")
            return []
    
    def save_ticket_requests(self, requests: List[TicketRequest]) -> bool:
        """保存抢票请求配置"""
        try:
            data = [request.dict() for request in requests]
            with open(self.ticket_requests_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info("抢票请求配置保存成功")
            return True
        except Exception as e:
            self.logger.error(f"保存抢票请求配置失败: {e}")
            return False
    
    def add_ticket_request(self, request: TicketRequest) -> bool:
        """添加抢票请求"""
        try:
            requests = self.load_ticket_requests()
            requests.append(request)
            return self.save_ticket_requests(requests)
        except Exception as e:
            self.logger.error(f"添加抢票请求失败: {e}")
            return False
    
    def remove_ticket_request(self, event_id: str, platform: PlatformType) -> bool:
        """移除抢票请求"""
        try:
            requests = self.load_ticket_requests()
            requests = [req for req in requests 
                       if not (req.event_id == event_id and req.platform == platform)]
            return self.save_ticket_requests(requests)
        except Exception as e:
            self.logger.error(f"移除抢票请求失败: {e}")
            return False
    
    def load_settings(self) -> Dict[str, Any]:
        """加载应用设置"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 合并默认设置
                    settings = self.default_settings.copy()
                    settings.update(data)
                    return settings
            else:
                self.logger.info("设置文件不存在，使用默认设置")
                return self.default_settings.copy()
        except Exception as e:
            self.logger.error(f"加载设置失败: {e}")
            return self.default_settings.copy()
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """保存应用设置"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            self.logger.info("应用设置保存成功")
            return True
        except Exception as e:
            self.logger.error(f"保存应用设置失败: {e}")
            return False
    
    def update_setting(self, key: str, value: Any) -> bool:
        """更新单个设置"""
        try:
            settings = self.load_settings()
            settings[key] = value
            return self.save_settings(settings)
        except Exception as e:
            self.logger.error(f"更新设置失败: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """获取设置值"""
        try:
            settings = self.load_settings()
            return settings.get(key, default)
        except Exception as e:
            self.logger.error(f"获取设置失败: {e}")
            return default
    
    def create_default_configs(self):
        """创建默认配置文件"""
        try:
            # 创建默认的大麦网配置
            from ..platforms.damai_adapter import DamaiConfig
            damai_config = DamaiConfig()
            self.save_platform_config(PlatformType.DAMAI, damai_config)
            
            # 创建默认设置
            self.save_settings(self.default_settings)
            
            self.logger.info("默认配置文件创建成功")
        except Exception as e:
            self.logger.error(f"创建默认配置文件失败: {e}")
    
    def validate_config(self) -> Dict[str, List[str]]:
        """验证配置完整性"""
        errors = {
            "user_config": [],
            "platform_config": [],
            "ticket_requests": [],
            "settings": []
        }
        
        try:
            # 验证用户配置
            user_config = self.load_user_config()
            if not user_config:
                errors["user_config"].append("用户配置不存在")
            elif not user_config.username or not user_config.password:
                errors["user_config"].append("用户名或密码为空")
            
            # 验证平台配置
            platform_configs = self.load_all_platform_configs()
            if not platform_configs:
                errors["platform_config"].append("平台配置不存在")
            
            # 验证抢票请求
            ticket_requests = self.load_ticket_requests()
            for i, request in enumerate(ticket_requests):
                if not request.event_id:
                    errors["ticket_requests"].append(f"第{i+1}个请求缺少活动ID")
                if not request.platform:
                    errors["ticket_requests"].append(f"第{i+1}个请求缺少平台信息")
            
            # 验证设置
            settings = self.load_settings()
            required_settings = ["log_level", "max_workers"]
            for setting in required_settings:
                if setting not in settings:
                    errors["settings"].append(f"缺少必需设置: {setting}")
            
        except Exception as e:
            errors["settings"].append(f"配置验证异常: {e}")
        
        return errors
    
    def backup_configs(self, backup_dir: str = None) -> bool:
        """备份配置文件"""
        try:
            if backup_dir is None:
                backup_dir = self.config_dir / "backup"
            
            backup_path = Path(backup_dir)
            backup_path.mkdir(exist_ok=True)
            
            # 备份所有配置文件
            config_files = [
                self.user_config_file,
                self.platform_config_file,
                self.ticket_requests_file,
                self.settings_file
            ]
            
            for config_file in config_files:
                if config_file.exists():
                    backup_file = backup_path / config_file.name
                    import shutil
                    shutil.copy2(config_file, backup_file)
            
            self.logger.info(f"配置文件备份成功: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"备份配置文件失败: {e}")
            return False
    
    def restore_configs(self, backup_dir: str) -> bool:
        """恢复配置文件"""
        try:
            backup_path = Path(backup_dir)
            if not backup_path.exists():
                self.logger.error(f"备份目录不存在: {backup_dir}")
                return False
            
            # 恢复所有配置文件
            config_files = [
                "user_config.json",
                "platform_config.json",
                "ticket_requests.json",
                "settings.json"
            ]
            
            for config_file in config_files:
                backup_file = backup_path / config_file
                if backup_file.exists():
                    target_file = self.config_dir / config_file
                    import shutil
                    shutil.copy2(backup_file, target_file)
            
            self.logger.info(f"配置文件恢复成功: {backup_dir}")
            return True
        except Exception as e:
            self.logger.error(f"恢复配置文件失败: {e}")
            return False 