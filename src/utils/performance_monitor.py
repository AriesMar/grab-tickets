#!/usr/bin/env python3
"""
性能监控和优化模块 - 实时监控系统性能并提供优化建议
"""
import time
import random
import json
import numpy as np
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from loguru import logger
import psutil
import os


class MetricType(Enum):
    """指标类型"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    CONCURRENCY = "concurrency"


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetric:
    """性能指标"""
    metric_type: MetricType
    value: float
    timestamp: float
    unit: str
    threshold: Optional[float] = None
    alert_level: AlertLevel = AlertLevel.INFO


@dataclass
class PerformanceAlert:
    """性能告警"""
    alert_id: str
    metric_type: MetricType
    current_value: float
    threshold: float
    alert_level: AlertLevel
    message: str
    timestamp: float
    resolved: bool = False


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.logger = logger.bind(name="performance_monitor")
        self.monitoring_interval = monitoring_interval
        self.metrics_history: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[PerformanceAlert] = []
        self.thresholds = self._initialize_thresholds()
        self.monitoring_active = False
        self.monitor_thread = None
        
    def _initialize_thresholds(self) -> Dict[MetricType, Dict[str, float]]:
        """初始化阈值"""
        return {
            MetricType.CPU_USAGE: {
                "warning": 70.0,
                "critical": 90.0,
                "emergency": 95.0
            },
            MetricType.MEMORY_USAGE: {
                "warning": 80.0,
                "critical": 90.0,
                "emergency": 95.0
            },
            MetricType.RESPONSE_TIME: {
                "warning": 2.0,
                "critical": 5.0,
                "emergency": 10.0
            },
            MetricType.SUCCESS_RATE: {
                "warning": 0.8,
                "critical": 0.6,
                "emergency": 0.4
            },
            MetricType.ERROR_RATE: {
                "warning": 0.1,
                "critical": 0.3,
                "emergency": 0.5
            }
        }
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("性能监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                
                # 存储指标
                for metric in system_metrics:
                    self.metrics_history[metric.metric_type].append(metric)
                
                # 检查告警
                self._check_alerts(system_metrics)
                
                # 等待下次监控
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> List[PerformanceMetric]:
        """收集系统指标"""
        metrics = []
        current_time = time.time()
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.append(PerformanceMetric(
                metric_type=MetricType.CPU_USAGE,
                value=cpu_percent,
                timestamp=current_time,
                unit="%",
                threshold=self.thresholds[MetricType.CPU_USAGE]["critical"]
            ))
            
            # 内存使用率
            memory = psutil.virtual_memory()
            metrics.append(PerformanceMetric(
                metric_type=MetricType.MEMORY_USAGE,
                value=memory.percent,
                timestamp=current_time,
                unit="%",
                threshold=self.thresholds[MetricType.MEMORY_USAGE]["critical"]
            ))
            
            # 网络IO
            network_io = psutil.net_io_counters()
            metrics.append(PerformanceMetric(
                metric_type=MetricType.NETWORK_IO,
                value=network_io.bytes_sent + network_io.bytes_recv,
                timestamp=current_time,
                unit="bytes",
                threshold=None
            ))
            
            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.append(PerformanceMetric(
                    metric_type=MetricType.DISK_IO,
                    value=disk_io.read_bytes + disk_io.write_bytes,
                    timestamp=current_time,
                    unit="bytes",
                    threshold=None
                ))
            
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {e}")
        
        return metrics
    
    def _check_alerts(self, metrics: List[PerformanceMetric]):
        """检查告警"""
        for metric in metrics:
            if metric.threshold is None:
                continue
            
            # 确定告警级别
            alert_level = self._determine_alert_level(metric)
            
            if alert_level != AlertLevel.INFO:
                # 创建告警
                alert = PerformanceAlert(
                    alert_id=f"alert_{int(time.time() * 1000)}",
                    metric_type=metric.metric_type,
                    current_value=metric.value,
                    threshold=metric.threshold,
                    alert_level=alert_level,
                    message=self._generate_alert_message(metric, alert_level),
                    timestamp=time.time()
                )
                
                self.alerts.append(alert)
                self.logger.warning(f"性能告警: {alert.message}")
    
    def _determine_alert_level(self, metric: PerformanceMetric) -> AlertLevel:
        """确定告警级别"""
        thresholds = self.thresholds.get(metric.metric_type, {})
        
        if metric.value >= thresholds.get("emergency", float('inf')):
            return AlertLevel.EMERGENCY
        elif metric.value >= thresholds.get("critical", float('inf')):
            return AlertLevel.CRITICAL
        elif metric.value >= thresholds.get("warning", float('inf')):
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def _generate_alert_message(self, metric: PerformanceMetric, alert_level: AlertLevel) -> str:
        """生成告警消息"""
        level_names = {
            AlertLevel.WARNING: "警告",
            AlertLevel.CRITICAL: "严重",
            AlertLevel.EMERGENCY: "紧急"
        }
        
        return f"{level_names[alert_level]} - {metric.metric_type.value}: {metric.value}{metric.unit} (阈值: {metric.threshold}{metric.unit})"
    
    def add_custom_metric(self, metric_type: MetricType, value: float, unit: str = "", threshold: Optional[float] = None):
        """添加自定义指标"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            unit=unit,
            threshold=threshold
        )
        
        self.metrics_history[metric_type].append(metric)
        
        # 检查告警
        if threshold is not None:
            alert_level = self._determine_alert_level(metric)
            if alert_level != AlertLevel.INFO:
                alert = PerformanceAlert(
                    alert_id=f"custom_alert_{int(time.time() * 1000)}",
                    metric_type=metric_type,
                    current_value=value,
                    threshold=threshold,
                    alert_level=alert_level,
                    message=self._generate_alert_message(metric, alert_level),
                    timestamp=time.time()
                )
                self.alerts.append(alert)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {}
        
        for metric_type, history in self.metrics_history.items():
            if not history:
                continue
            
            values = [metric.value for metric in history]
            summary[metric_type.value] = {
                "current": values[-1] if values else 0,
                "average": np.mean(values),
                "min": np.min(values),
                "max": np.max(values),
                "std": np.std(values),
                "count": len(values)
            }
        
        return summary
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        return [asdict(alert) for alert in active_alerts]
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self.logger.info(f"告警已解决: {alert_id}")
                break


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.logger = logger.bind(name="performance_optimizer")
        self.optimization_history = []
        self.optimization_strategies = self._initialize_optimization_strategies()
        
    def _initialize_optimization_strategies(self) -> Dict[str, Dict[str, Any]]:
        """初始化优化策略"""
        return {
            "cpu_optimization": {
                "condition": lambda metrics: metrics.get("cpu_usage", {}).get("current", 0) > 80,
                "action": self._optimize_cpu_usage,
                "description": "优化CPU使用率"
            },
            "memory_optimization": {
                "condition": lambda metrics: metrics.get("memory_usage", {}).get("current", 0) > 85,
                "action": self._optimize_memory_usage,
                "description": "优化内存使用率"
            },
            "response_time_optimization": {
                "condition": lambda metrics: metrics.get("response_time", {}).get("current", 0) > 3.0,
                "action": self._optimize_response_time,
                "description": "优化响应时间"
            },
            "success_rate_optimization": {
                "condition": lambda metrics: metrics.get("success_rate", {}).get("current", 1.0) < 0.8,
                "action": self._optimize_success_rate,
                "description": "优化成功率"
            },
            "concurrency_optimization": {
                "condition": lambda metrics: metrics.get("concurrency", {}).get("current", 0) < 5,
                "action": self._optimize_concurrency,
                "description": "优化并发数"
            }
        }
    
    def analyze_and_optimize(self) -> Dict[str, Any]:
        """分析并优化性能"""
        metrics_summary = self.monitor.get_metrics_summary()
        optimizations_applied = []
        
        # 检查每个优化策略
        for strategy_name, strategy in self.optimization_strategies.items():
            if strategy["condition"](metrics_summary):
                try:
                    optimization_result = strategy["action"](metrics_summary)
                    optimizations_applied.append({
                        "strategy": strategy_name,
                        "description": strategy["description"],
                        "result": optimization_result
                    })
                except Exception as e:
                    self.logger.error(f"应用优化策略失败 {strategy_name}: {e}")
        
        # 记录优化历史
        optimization_record = {
            "timestamp": time.time(),
            "metrics_summary": metrics_summary,
            "optimizations_applied": optimizations_applied
        }
        self.optimization_history.append(optimization_record)
        
        return {
            "optimizations_applied": optimizations_applied,
            "total_optimizations": len(optimizations_applied),
            "optimization_history_count": len(self.optimization_history)
        }
    
    def _optimize_cpu_usage(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """优化CPU使用率"""
        current_cpu = metrics.get("cpu_usage", {}).get("current", 0)
        
        optimizations = []
        
        if current_cpu > 90:
            optimizations.append({
                "action": "reduce_worker_threads",
                "parameter": "max_workers",
                "value": "decrease_by_50%",
                "priority": "critical"
            })
        elif current_cpu > 80:
            optimizations.append({
                "action": "optimize_algorithm",
                "parameter": "algorithm_complexity",
                "value": "use_caching",
                "priority": "high"
            })
        
        return {
            "cpu_usage_before": current_cpu,
            "optimizations": optimizations,
            "expected_improvement": max(0, (current_cpu - 70) / 100)
        }
    
    def _optimize_memory_usage(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """优化内存使用率"""
        current_memory = metrics.get("memory_usage", {}).get("current", 0)
        
        optimizations = []
        
        if current_memory > 90:
            optimizations.append({
                "action": "force_garbage_collection",
                "parameter": "gc_cycles",
                "value": 3,
                "priority": "critical"
            })
        elif current_memory > 85:
            optimizations.append({
                "action": "clear_cache",
                "parameter": "cache_size",
                "value": "reduce_by_30%",
                "priority": "high"
            })
        
        return {
            "memory_usage_before": current_memory,
            "optimizations": optimizations,
            "expected_improvement": max(0, (current_memory - 75) / 100)
        }
    
    def _optimize_response_time(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """优化响应时间"""
        current_response_time = metrics.get("response_time", {}).get("current", 0)
        
        optimizations = []
        
        if current_response_time > 5.0:
            optimizations.append({
                "action": "enable_connection_pooling",
                "parameter": "pool_size",
                "value": "increase_by_50%",
                "priority": "critical"
            })
        elif current_response_time > 3.0:
            optimizations.append({
                "action": "optimize_database_queries",
                "parameter": "query_timeout",
                "value": "reduce_by_30%",
                "priority": "high"
            })
        
        return {
            "response_time_before": current_response_time,
            "optimizations": optimizations,
            "expected_improvement": max(0, (current_response_time - 2.0) / 10.0)
        }
    
    def _optimize_success_rate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """优化成功率"""
        current_success_rate = metrics.get("success_rate", {}).get("current", 1.0)
        
        optimizations = []
        
        if current_success_rate < 0.6:
            optimizations.append({
                "action": "increase_retry_attempts",
                "parameter": "max_retries",
                "value": "increase_by_100%",
                "priority": "critical"
            })
        elif current_success_rate < 0.8:
            optimizations.append({
                "action": "improve_error_handling",
                "parameter": "error_recovery",
                "value": "enhance_graceful_degradation",
                "priority": "high"
            })
        
        return {
            "success_rate_before": current_success_rate,
            "optimizations": optimizations,
            "expected_improvement": max(0, (0.9 - current_success_rate))
        }
    
    def _optimize_concurrency(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """优化并发数"""
        current_concurrency = metrics.get("concurrency", {}).get("current", 0)
        
        optimizations = []
        
        if current_concurrency < 3:
            optimizations.append({
                "action": "increase_thread_pool",
                "parameter": "thread_pool_size",
                "value": "double_current_size",
                "priority": "high"
            })
        elif current_concurrency < 5:
            optimizations.append({
                "action": "optimize_task_scheduling",
                "parameter": "scheduling_algorithm",
                "value": "use_priority_queue",
                "priority": "medium"
            })
        
        return {
            "concurrency_before": current_concurrency,
            "optimizations": optimizations,
            "expected_improvement": max(0, (10 - current_concurrency) / 10)
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        if not self.optimization_history:
            return {"message": "暂无优化历史"}
        
        recent_optimizations = self.optimization_history[-10:]
        
        total_optimizations = sum(
            len(record["optimizations_applied"]) 
            for record in recent_optimizations
        )
        
        return {
            "total_optimizations": total_optimizations,
            "optimization_frequency": len(recent_optimizations) / 10.0,
            "recent_optimizations": [
                {
                    "timestamp": record["timestamp"],
                    "optimizations_count": len(record["optimizations_applied"])
                }
                for record in recent_optimizations
            ],
            "most_common_optimizations": self._get_most_common_optimizations(recent_optimizations)
        }
    
    def _get_most_common_optimizations(self, optimizations: List[Dict[str, Any]]) -> List[str]:
        """获取最常见的优化类型"""
        optimization_types = []
        
        for record in optimizations:
            for optimization in record["optimizations_applied"]:
                optimization_types.append(optimization["strategy"])
        
        if not optimization_types:
            return []
        
        from collections import Counter
        counter = Counter(optimization_types)
        return [strategy for strategy, count in counter.most_common(3)]


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.logger = logger.bind(name="performance_analyzer")
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5分钟缓存
        
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """分析性能趋势"""
        metrics_summary = self.monitor.get_metrics_summary()
        
        trends = {}
        
        for metric_type, data in metrics_summary.items():
            if len(self.monitor.metrics_history[MetricType(metric_type)]) < 10:
                continue
            
            # 计算趋势
            recent_values = [metric.value for metric in 
                           list(self.monitor.metrics_history[MetricType(metric_type)])[-20:]]
            
            trend = self._calculate_trend(recent_values)
            trends[metric_type] = {
                "trend": trend,
                "trend_strength": abs(trend),
                "prediction": self._predict_next_value(recent_values),
                "stability": self._calculate_stability(recent_values)
            }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        
        # 使用线性回归计算趋势
        x = np.arange(len(values))
        y = np.array(values)
        
        # 计算斜率
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    def _predict_next_value(self, values: List[float]) -> float:
        """预测下一个值"""
        if len(values) < 3:
            return values[-1] if values else 0.0
        
        # 使用简单移动平均预测
        window_size = min(5, len(values))
        recent_values = values[-window_size:]
        
        return np.mean(recent_values)
    
    def _calculate_stability(self, values: List[float]) -> float:
        """计算稳定性"""
        if len(values) < 2:
            return 1.0
        
        # 计算变异系数
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        if mean_value == 0:
            return 1.0
        
        cv = std_value / mean_value
        stability = max(0.0, 1.0 - cv)
        
        return stability
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        bottlenecks = []
        metrics_summary = self.monitor.get_metrics_summary()
        
        # 检查CPU瓶颈
        cpu_data = metrics_summary.get("cpu_usage", {})
        if cpu_data.get("current", 0) > 80:
            bottlenecks.append({
                "type": "cpu_bottleneck",
                "severity": "high" if cpu_data.get("current", 0) > 90 else "medium",
                "current_value": cpu_data.get("current", 0),
                "threshold": 80,
                "recommendation": "考虑增加CPU资源或优化算法"
            })
        
        # 检查内存瓶颈
        memory_data = metrics_summary.get("memory_usage", {})
        if memory_data.get("current", 0) > 85:
            bottlenecks.append({
                "type": "memory_bottleneck",
                "severity": "high" if memory_data.get("current", 0) > 95 else "medium",
                "current_value": memory_data.get("current", 0),
                "threshold": 85,
                "recommendation": "检查内存泄漏或增加内存"
            })
        
        # 检查响应时间瓶颈
        response_time_data = metrics_summary.get("response_time", {})
        if response_time_data.get("current", 0) > 3.0:
            bottlenecks.append({
                "type": "response_time_bottleneck",
                "severity": "high" if response_time_data.get("current", 0) > 5.0 else "medium",
                "current_value": response_time_data.get("current", 0),
                "threshold": 3.0,
                "recommendation": "优化网络连接或数据库查询"
            })
        
        return bottlenecks
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        metrics_summary = self.monitor.get_metrics_summary()
        trends = self.analyze_performance_trends()
        bottlenecks = self.identify_bottlenecks()
        active_alerts = self.monitor.get_active_alerts()
        
        return {
            "timestamp": time.time(),
            "metrics_summary": metrics_summary,
            "trends": trends,
            "bottlenecks": bottlenecks,
            "active_alerts": active_alerts,
            "recommendations": self._generate_recommendations(metrics_summary, bottlenecks)
        }
    
    def _generate_recommendations(self, metrics: Dict[str, Any], bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于瓶颈生成建议
        for bottleneck in bottlenecks:
            recommendations.append(bottleneck["recommendation"])
        
        # 基于趋势生成建议
        for metric_type, trend_data in metrics.items():
            if trend_data.get("current", 0) > 80:
                recommendations.append(f"监控{metric_type}指标，当前值较高")
        
        # 基于告警生成建议
        if len(self.monitor.get_active_alerts()) > 3:
            recommendations.append("系统告警较多，建议检查系统配置")
        
        return recommendations


class PerformanceMonitoringSystem:
    """性能监控系统主类"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.optimizer = PerformanceOptimizer(self.monitor)
        self.analyzer = PerformanceAnalyzer(self.monitor)
        self.logger = logger.bind(name="performance_monitoring_system")
        
    def start_system(self):
        """启动系统"""
        self.monitor.start_monitoring()
        self.logger.info("性能监控系统已启动")
    
    def stop_system(self):
        """停止系统"""
        self.monitor.stop_monitoring()
        self.logger.info("性能监控系统已停止")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "monitor_status": {
                "active": self.monitor.monitoring_active,
                "metrics_count": sum(len(history) for history in self.monitor.metrics_history.values()),
                "active_alerts": len(self.monitor.get_active_alerts())
            },
            "optimizer_status": self.optimizer.get_optimization_report(),
            "analyzer_status": {
                "cache_size": len(self.analyzer.analysis_cache),
                "last_analysis": time.time()
            }
        }
    
    def add_custom_metric(self, metric_type: str, value: float, unit: str = "", threshold: Optional[float] = None):
        """添加自定义指标"""
        try:
            metric_enum = MetricType(metric_type)
            self.monitor.add_custom_metric(metric_enum, value, unit, threshold)
        except ValueError:
            self.logger.error(f"未知的指标类型: {metric_type}")
    
    def run_optimization(self) -> Dict[str, Any]:
        """运行优化"""
        return self.optimizer.analyze_and_optimize()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return self.analyzer.generate_performance_report()


# 使用示例
if __name__ == "__main__":
    # 创建性能监控系统
    performance_system = PerformanceMonitoringSystem()
    
    # 启动系统
    performance_system.start_system()
    
    # 添加自定义指标
    performance_system.add_custom_metric("response_time", 2.5, "seconds", 3.0)
    performance_system.add_custom_metric("success_rate", 0.85, "", 0.8)
    performance_system.add_custom_metric("concurrency", 8, "threads", 5)
    
    # 等待一段时间收集数据
    time.sleep(5)
    
    # 运行优化
    optimization_result = performance_system.run_optimization()
    print("优化结果:", json.dumps(optimization_result, indent=2, ensure_ascii=False))
    
    # 获取性能报告
    performance_report = performance_system.get_performance_report()
    print("\n性能报告:")
    print(json.dumps(performance_report, indent=2, ensure_ascii=False, default=str))
    
    # 获取系统状态
    system_status = performance_system.get_system_status()
    print("\n系统状态:")
    print(json.dumps(system_status, indent=2, ensure_ascii=False))
    
    # 停止系统
    performance_system.stop_system() 