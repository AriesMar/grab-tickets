"""
日志工具模块
"""
import sys
from pathlib import Path
from loguru import logger
from typing import Optional


class LoggerManager:
    """日志管理器"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志配置"""
        # 创建日志目录
        self.log_dir.mkdir(exist_ok=True)
        
        # 移除默认的日志处理器
        logger.remove()
        
        # 添加控制台输出
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=self.log_level,
            colorize=True
        )
        
        # 添加文件输出 - 按日期分割
        logger.add(
            self.log_dir / "grab_tickets_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=self.log_level,
            rotation="00:00",  # 每天轮换
            retention="30 days",  # 保留30天
            compression="zip"
        )
        
        # 添加错误日志文件
        logger.add(
            self.log_dir / "error_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="00:00",
            retention="30 days",
            compression="zip"
        )
    
    def get_logger(self, name: str = "grab_tickets") -> logger:
        """获取日志记录器"""
        return logger.bind(name=name)


# 全局日志管理器实例
log_manager = LoggerManager()
grab_logger = log_manager.get_logger()


def get_logger(name: str = "grab_tickets"):
    """获取日志记录器的便捷函数"""
    return log_manager.get_logger(name)


def set_log_level(level: str):
    """设置日志级别"""
    log_manager.log_level = level
    log_manager._setup_logger()


def set_log_dir(log_dir: str):
    """设置日志目录"""
    log_manager.log_dir = Path(log_dir)
    log_manager._setup_logger() 