"""
安全报告生成器
"""
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger

from .security import SecurityManager, AntiDetectionManager, SecurityValidator
from ..data.security_config import SecurityConfig, SecurityMonitor


class SecurityReportGenerator:
    """安全报告生成器"""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self.security_manager = SecurityManager()
        self.anti_detection = AntiDetectionManager()
        self.security_validator = SecurityValidator()
        self.security_monitor = SecurityMonitor(security_config)
        self.logger = logger.bind(name="security_report")
        
        # 报告数据
        self.report_data = {
            "timestamp": time.time(),
            "security_level": "unknown",
            "violations": [],
            "suspicious_activities": [],
            "recommendations": [],
            "statistics": {}
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """生成安全报告"""
        try:
            # 收集安全数据
            self._collect_security_data()
            
            # 分析安全状态
            self._analyze_security_status()
            
            # 生成建议
            self._generate_recommendations()
            
            # 计算统计信息
            self._calculate_statistics()
            
            return self.report_data
            
        except Exception as e:
            self.logger.error(f"生成安全报告失败: {e}")
            return self._get_error_report()
    
    def _collect_security_data(self):
        """收集安全数据"""
        # 获取安全监控数据
        monitor_report = self.security_monitor.get_security_report()
        self.report_data.update(monitor_report)
        
        # 获取反检测数据
        suspicious_activities = self.anti_detection.detect_suspicious_activity()
        self.report_data["suspicious_activities"] = suspicious_activities
        
        # 获取请求历史
        request_history = self.security_manager.request_history
        self.report_data["request_history"] = request_history
    
    def _analyze_security_status(self):
        """分析安全状态"""
        violations = self.report_data.get("total_violations", 0)
        suspicious_count = len(self.report_data.get("suspicious_activities", []))
        
        # 计算安全等级
        if violations == 0 and suspicious_count == 0:
            security_level = "safe"
        elif violations < 3 and suspicious_count < 2:
            security_level = "warning"
        elif violations < 10 and suspicious_count < 5:
            security_level = "danger"
        else:
            security_level = "critical"
        
        self.report_data["security_level"] = security_level
    
    def _generate_recommendations(self):
        """生成安全建议"""
        recommendations = []
        security_level = self.report_data["security_level"]
        
        if security_level == "critical":
            recommendations.extend([
                "立即停止所有自动化操作",
                "检查并修复所有安全配置",
                "考虑更换IP地址或使用代理",
                "增加请求间隔时间",
                "启用更强的反检测机制"
            ])
        elif security_level == "danger":
            recommendations.extend([
                "减少请求频率",
                "增加随机延迟",
                "轮换设备指纹",
                "检查验证码处理机制"
            ])
        elif security_level == "warning":
            recommendations.extend([
                "监控请求模式",
                "优化请求间隔",
                "启用会话轮换"
            ])
        else:  # safe
            recommendations.extend([
                "继续保持当前配置",
                "定期检查安全状态",
                "监控异常活动"
            ])
        
        self.report_data["recommendations"] = recommendations
    
    def _calculate_statistics(self):
        """计算统计信息"""
        request_history = self.report_data.get("request_history", [])
        current_time = time.time()
        
        # 计算最近1小时的请求统计
        recent_requests = [
            req_time for req_time in request_history
            if current_time - req_time < 3600
        ]
        
        # 计算最近1分钟的请求统计
        minute_requests = [
            req_time for req_time in request_history
            if current_time - req_time < 60
        ]
        
        statistics = {
            "total_requests": len(request_history),
            "requests_last_hour": len(recent_requests),
            "requests_last_minute": len(minute_requests),
            "average_requests_per_minute": len(minute_requests),
            "violation_rate": self._calculate_violation_rate(),
            "suspicious_activity_rate": self._calculate_suspicious_rate()
        }
        
        self.report_data["statistics"] = statistics
    
    def _calculate_violation_rate(self) -> float:
        """计算违规率"""
        total_requests = len(self.report_data.get("request_history", []))
        violations = self.report_data.get("total_violations", 0)
        
        if total_requests == 0:
            return 0.0
        
        return (violations / total_requests) * 100
    
    def _calculate_suspicious_rate(self) -> float:
        """计算可疑活动率"""
        total_requests = len(self.report_data.get("request_history", []))
        suspicious_count = len(self.report_data.get("suspicious_activities", []))
        
        if total_requests == 0:
            return 0.0
        
        return (suspicious_count / total_requests) * 100
    
    def _get_error_report(self) -> Dict[str, Any]:
        """获取错误报告"""
        return {
            "timestamp": time.time(),
            "security_level": "error",
            "error": "报告生成失败",
            "recommendations": ["检查系统配置", "查看错误日志"]
        }
    
    def save_report(self, file_path: str = None) -> bool:
        """保存报告到文件"""
        try:
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"logs/security_report_{timestamp}.json"
            
            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存报告
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"安全报告已保存: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存安全报告失败: {e}")
            return False
    
    def print_report(self):
        """打印安全报告"""
        report = self.report_data
        
        print("\n" + "="*50)
        print("安全报告")
        print("="*50)
        
        # 基本信息
        print(f"报告时间: {datetime.fromtimestamp(report['timestamp'])}")
        print(f"安全等级: {report['security_level']}")
        print(f"违规次数: {report.get('total_violations', 0)}")
        print(f"可疑活动: {len(report.get('suspicious_activities', []))}")
        
        # 统计信息
        stats = report.get('statistics', {})
        print(f"\n统计信息:")
        print(f"  总请求数: {stats.get('total_requests', 0)}")
        print(f"  最近1小时: {stats.get('requests_last_hour', 0)}")
        print(f"  最近1分钟: {stats.get('requests_last_minute', 0)}")
        print(f"  违规率: {stats.get('violation_rate', 0):.2f}%")
        print(f"  可疑活动率: {stats.get('suspicious_activity_rate', 0):.2f}%")
        
        # 建议
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\n安全建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # 可疑活动详情
        suspicious = report.get('suspicious_activities', [])
        if suspicious:
            print(f"\n可疑活动:")
            for activity in suspicious:
                print(f"  - {activity.get('type', 'unknown')}: {activity.get('message', '')}")
        
        print("="*50)


class SecurityAlertManager:
    """安全告警管理器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logger.bind(name="security_alert")
        self.alert_history = []
    
    def check_alerts(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查是否需要告警"""
        alerts = []
        
        # 检查安全等级
        security_level = report_data.get("security_level", "unknown")
        if security_level in ["danger", "critical"]:
            alerts.append({
                "type": "security_level",
                "level": "high",
                "message": f"安全等级为 {security_level}，需要立即关注",
                "timestamp": time.time()
            })
        
        # 检查违规次数
        violations = report_data.get("total_violations", 0)
        if violations > 5:
            alerts.append({
                "type": "violation_count",
                "level": "medium",
                "message": f"违规次数过多: {violations}",
                "timestamp": time.time()
            })
        
        # 检查请求频率
        stats = report_data.get("statistics", {})
        requests_per_minute = stats.get("requests_last_minute", 0)
        if requests_per_minute > 50:
            alerts.append({
                "type": "high_frequency",
                "level": "high",
                "message": f"请求频率过高: {requests_per_minute}/分钟",
                "timestamp": time.time()
            })
        
        # 记录告警历史
        for alert in alerts:
            self.alert_history.append(alert)
        
        return alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        current_time = time.time()
        
        # 统计最近24小时的告警
        recent_alerts = [
            alert for alert in self.alert_history
            if current_time - alert.get("timestamp", 0) < 86400
        ]
        
        # 按级别统计
        high_alerts = [a for a in recent_alerts if a.get("level") == "high"]
        medium_alerts = [a for a in recent_alerts if a.get("level") == "medium"]
        low_alerts = [a for a in recent_alerts if a.get("level") == "low"]
        
        return {
            "total_alerts_24h": len(recent_alerts),
            "high_alerts": len(high_alerts),
            "medium_alerts": len(medium_alerts),
            "low_alerts": len(low_alerts),
            "latest_alerts": recent_alerts[-5:] if recent_alerts else []
        } 