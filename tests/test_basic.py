"""
基本功能测试
"""
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.data.models import TicketRequest, PlatformType, UserConfig
from src.data.config_manager import ConfigManager
from src.utils.logger import get_logger


class TestDataModels:
    """数据模型测试"""
    
    def test_ticket_request_creation(self):
        """测试抢票请求创建"""
        request = TicketRequest(
            event_id="123456",
            platform=PlatformType.DAMAI,
            target_price=580.0,
            quantity=2,
            retry_times=3,
            retry_interval=1.0
        )
        
        assert request.event_id == "123456"
        assert request.platform == PlatformType.DAMAI
        assert request.target_price == 580.0
        assert request.quantity == 2
        assert request.retry_times == 3
        assert request.retry_interval == 1.0
    
    def test_user_config_creation(self):
        """测试用户配置创建"""
        config = UserConfig(
            username="test_user",
            password="test_password",
            phone="13800138000"
        )
        
        assert config.username == "test_user"
        assert config.password == "test_password"
        assert config.phone == "13800138000"


class TestConfigManager:
    """配置管理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.config_manager = ConfigManager("test_config")
        self.test_config_dir = Path("test_config")
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        if self.test_config_dir.exists():
            shutil.rmtree(self.test_config_dir)
    
    def test_create_user_config(self):
        """测试创建用户配置"""
        config = self.config_manager.create_user_config(
            username="test_user",
            password="test_password"
        )
        
        assert config.username == "test_user"
        assert config.password == "test_password"
        
        # 验证文件是否创建
        assert self.config_manager.user_config_file.exists()
    
    def test_load_settings(self):
        """测试加载设置"""
        settings = self.config_manager.load_settings()
        
        # 验证默认设置
        assert "log_level" in settings
        assert "max_workers" in settings
        assert settings["log_level"] == "INFO"
        assert settings["max_workers"] == 5
    
    def test_validate_config(self):
        """测试配置验证"""
        errors = self.config_manager.validate_config()
        
        # 新创建的配置应该有错误（缺少用户配置）
        assert "user_config" in errors
        assert len(errors["user_config"]) > 0


class TestLogger:
    """日志测试"""
    
    def test_logger_creation(self):
        """测试日志创建"""
        logger = get_logger("test_logger")
        assert logger is not None
        
        # 测试日志记录
        logger.info("测试日志信息")
        logger.warning("测试警告信息")
        logger.error("测试错误信息")


if __name__ == "__main__":
    # 运行基本测试
    print("运行基本功能测试...")
    
    # 测试数据模型
    test_models = TestDataModels()
    test_models.test_ticket_request_creation()
    test_models.test_user_config_creation()
    print("✓ 数据模型测试通过")
    
    # 测试配置管理器
    test_config = TestConfigManager()
    test_config.setup_method()
    test_config.test_create_user_config()
    test_config.test_load_settings()
    test_config.test_validate_config()
    test_config.teardown_method()
    print("✓ 配置管理器测试通过")
    
    # 测试日志
    test_logger = TestLogger()
    test_logger.test_logger_creation()
    print("✓ 日志测试通过")
    
    print("所有基本测试通过！") 