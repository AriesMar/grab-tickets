#!/usr/bin/env python3
"""
通用抢票软件框架 - 主程序入口
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.ui.cli_interface import CLIInterface
from src.data.config_manager import ConfigManager
from src.utils.logger import get_logger


def main():
    """主函数"""
    logger = get_logger("main")
    
    try:
        logger.info("启动通用抢票软件框架")
        
        # 创建配置管理器并初始化默认配置
        config_manager = ConfigManager()
        config_manager.create_default_configs()
        
        # 启动CLI界面
        cli = CLIInterface()
        cli.run()
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        print("\n程序已退出")
    except Exception as e:
        logger.error(f"程序运行异常: {e}")
        print(f"程序运行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 