#!/usr/bin/env python3
"""
边缘计算优化模块
实现分布式计算、负载均衡、容错机制等功能
"""

import asyncio
import aiohttp
import json
import time
import random
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import hashlib
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


class NodeStatus(Enum):
    """节点状态"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class EdgeNode:
    """边缘节点"""
    node_id: str
    host: str
    port: int
    status: NodeStatus
    capabilities: Dict[str, Any]
    load: float
    last_heartbeat: float
    performance_metrics: Dict[str, float]


@dataclass
class ComputingTask:
    """计算任务"""
    task_id: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    requirements: Dict[str, Any]
    created_at: float
    deadline: Optional[float] = None


class DistributedComputing:
    """分布式计算管理器"""
    
    def __init__(self, max_workers: int = 10):
        self.nodes: Dict[str, EdgeNode] = {}
        self.task_queue = queue.PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_tasks: Dict[str, Any] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Any] = {}
        self.lock = threading.Lock()
        
    def add_node(self, node: EdgeNode) -> None:
        """添加节点"""
        with self.lock:
            self.nodes[node.node_id] = node
    
    def remove_node(self, node_id: str) -> None:
        """移除节点"""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
    
    def execute(self, task: ComputingTask, nodes: List[str]) -> Dict[str, Any]:
        """执行分布式计算任务"""
        
        # 验证节点可用性
        available_nodes = self._get_available_nodes(nodes)
        if not available_nodes:
            return {"error": "没有可用的节点"}
        
        # 选择最佳节点
        best_node = self._select_best_node(available_nodes, task)
        if not best_node:
            return {"error": "无法找到合适的节点"}
        
        # 提交任务
        future = self.executor.submit(self._execute_task_on_node, task, best_node)
        self.running_tasks[task.task_id] = {
            "future": future,
            "node": best_node,
            "start_time": time.time()
        }
        
        try:
            result = future.result(timeout=task.deadline or 60)
            self.completed_tasks[task.task_id] = result
            return result
        except Exception as e:
            self.failed_tasks[task.task_id] = {"error": str(e)}
            return {"error": f"任务执行失败: {e}"}
        finally:
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
    
    def _get_available_nodes(self, node_ids: List[str]) -> List[EdgeNode]:
        """获取可用节点"""
        available_nodes = []
        
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if (node.status == NodeStatus.ONLINE and 
                    time.time() - node.last_heartbeat < 30):  # 30秒心跳超时
                    available_nodes.append(node)
        
        return available_nodes
    
    def _select_best_node(self, nodes: List[EdgeNode], task: ComputingTask) -> Optional[EdgeNode]:
        """选择最佳节点"""
        if not nodes:
            return None
        
        # 根据负载和性能选择最佳节点
        best_node = None
        best_score = float('-inf')
        
        for node in nodes:
            score = self._calculate_node_score(node, task)
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    def _calculate_node_score(self, node: EdgeNode, task: ComputingTask) -> float:
        """计算节点分数"""
        # 基础分数
        score = 100.0
        
        # 负载惩罚
        load_penalty = node.load * 50
        score -= load_penalty
        
        # 性能奖励
        performance_bonus = node.performance_metrics.get("cpu_performance", 0.5) * 20
        score += performance_bonus
        
        # 网络延迟惩罚
        network_penalty = node.performance_metrics.get("network_latency", 0.0) * 10
        score -= network_penalty
        
        # 任务匹配度
        if self._check_task_compatibility(node, task):
            score += 30
        
        return max(0, score)
    
    def _check_task_compatibility(self, node: EdgeNode, task: ComputingTask) -> bool:
        """检查任务兼容性"""
        required_capabilities = task.requirements.get("capabilities", [])
        node_capabilities = node.capabilities.get("supported_features", [])
        
        return all(cap in node_capabilities for cap in required_capabilities)
    
    def _execute_task_on_node(self, task: ComputingTask, node: EdgeNode) -> Dict[str, Any]:
        """在节点上执行任务"""
        try:
            # 模拟任务执行
            execution_time = random.uniform(0.1, 2.0)
            time.sleep(execution_time)
            
            # 更新节点负载
            node.load += 0.1
            node.performance_metrics["cpu_usage"] = min(1.0, node.performance_metrics.get("cpu_usage", 0.0) + 0.1)
            
            return {
                "task_id": task.task_id,
                "node_id": node.node_id,
                "result": self._process_task_result(task, node),
                "execution_time": execution_time,
                "status": "completed"
            }
        except Exception as e:
            # 更新节点状态为错误
            node.status = NodeStatus.ERROR
            raise e
    
    def _process_task_result(self, task: ComputingTask, node: EdgeNode) -> Dict[str, Any]:
        """处理任务结果"""
        # 根据任务类型处理结果
        if task.task_type == "detection_analysis":
            return self._process_detection_analysis(task.data, node)
        elif task.task_type == "behavior_simulation":
            return self._process_behavior_simulation(task.data, node)
        elif task.task_type == "pattern_recognition":
            return self._process_pattern_recognition(task.data, node)
        else:
            return {"processed_data": task.data, "node_capabilities": node.capabilities}
    
    def _process_detection_analysis(self, data: Dict[str, Any], node: EdgeNode) -> Dict[str, Any]:
        """处理检测分析任务"""
        return {
            "detection_score": random.uniform(0.0, 1.0),
            "risk_level": random.choice(["low", "medium", "high"]),
            "recommendations": ["加强反检测措施", "优化行为模式"],
            "node_performance": node.performance_metrics
        }
    
    def _process_behavior_simulation(self, data: Dict[str, Any], node: EdgeNode) -> Dict[str, Any]:
        """处理行为模拟任务"""
        return {
            "simulated_behavior": {
                "mouse_movement": self._generate_mouse_trajectory(),
                "keyboard_input": self._generate_keyboard_pattern(),
                "naturalness_score": random.uniform(0.8, 1.0)
            },
            "node_capabilities": node.capabilities
        }
    
    def _process_pattern_recognition(self, data: Dict[str, Any], node: EdgeNode) -> Dict[str, Any]:
        """处理模式识别任务"""
        return {
            "recognized_patterns": ["user_behavior", "detection_pattern"],
            "confidence": random.uniform(0.7, 1.0),
            "pattern_features": data.get("features", {})
        }
    
    def _generate_mouse_trajectory(self) -> List[Tuple[float, float]]:
        """生成鼠标轨迹"""
        trajectory = []
        x, y = 0, 0
        
        for i in range(20):
            x += random.uniform(-10, 10)
            y += random.uniform(-10, 10)
            trajectory.append((x, y))
        
        return trajectory
    
    def _generate_keyboard_pattern(self) -> Dict[str, Any]:
        """生成键盘模式"""
        return {
            "typing_speed": random.uniform(50, 200),
            "pause_patterns": [random.uniform(0.1, 0.5) for _ in range(5)],
            "error_rate": random.uniform(0.01, 0.05)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self.lock:
            return {
                "total_nodes": len(self.nodes),
                "online_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "average_load": np.mean([n.load for n in self.nodes.values()]) if self.nodes else 0.0
            }


class IntelligentLoadBalancer:
    """智能负载均衡器"""
    
    def __init__(self):
        self.nodes: Dict[str, EdgeNode] = {}
        self.load_history: Dict[str, List[float]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.balancing_strategy = "weighted_round_robin"
        
    def balance(self, nodes: List[str], tasks: List[ComputingTask]) -> Dict[str, List[str]]:
        """负载均衡"""
        
        # 更新节点信息
        self._update_node_info(nodes)
        
        # 根据策略分配任务
        if self.balancing_strategy == "weighted_round_robin":
            return self._weighted_round_robin_balance(tasks)
        elif self.balancing_strategy == "least_connections":
            return self._least_connections_balance(tasks)
        elif self.balancing_strategy == "performance_based":
            return self._performance_based_balance(tasks)
        else:
            return self._simple_round_robin_balance(tasks)
    
    def _update_node_info(self, node_ids: List[str]) -> None:
        """更新节点信息"""
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # 更新负载历史
                if node_id not in self.load_history:
                    self.load_history[node_id] = []
                self.load_history[node_id].append(node.load)
                
                # 保持历史记录在合理范围内
                if len(self.load_history[node_id]) > 100:
                    self.load_history[node_id] = self.load_history[node_id][-50:]
                
                # 更新性能历史
                if node_id not in self.performance_history:
                    self.performance_history[node_id] = []
                performance_score = self._calculate_performance_score(node)
                self.performance_history[node_id].append(performance_score)
                
                if len(self.performance_history[node_id]) > 100:
                    self.performance_history[node_id] = self.performance_history[node_id][-50:]
    
    def _calculate_performance_score(self, node: EdgeNode) -> float:
        """计算性能分数"""
        cpu_performance = node.performance_metrics.get("cpu_performance", 0.5)
        memory_usage = node.performance_metrics.get("memory_usage", 0.5)
        network_latency = node.performance_metrics.get("network_latency", 0.0)
        
        # 性能分数 = CPU性能 * (1 - 内存使用率) * (1 - 网络延迟)
        return cpu_performance * (1 - memory_usage) * (1 - network_latency)
    
    def _weighted_round_robin_balance(self, tasks: List[ComputingTask]) -> Dict[str, List[str]]:
        """加权轮询负载均衡"""
        assignment = {}
        available_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
        
        if not available_nodes:
            return {}
        
        # 计算权重
        total_weight = sum(1.0 / (n.load + 0.1) for n in available_nodes)
        node_weights = {n.node_id: (1.0 / (n.load + 0.1)) / total_weight for n in available_nodes}
        
        # 分配任务
        for i, task in enumerate(tasks):
            # 选择权重最高的节点
            best_node = max(available_nodes, key=lambda n: node_weights[n.node_id])
            
            if best_node.node_id not in assignment:
                assignment[best_node.node_id] = []
            assignment[best_node.node_id].append(task.task_id)
            
            # 更新权重（减少已分配节点的权重）
            node_weights[best_node.node_id] *= 0.8
        
        return assignment
    
    def _least_connections_balance(self, tasks: List[ComputingTask]) -> Dict[str, List[str]]:
        """最少连接负载均衡"""
        assignment = {}
        available_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
        
        if not available_nodes:
            return {}
        
        # 按负载排序
        available_nodes.sort(key=lambda n: n.load)
        
        # 分配任务
        for i, task in enumerate(tasks):
            selected_node = available_nodes[i % len(available_nodes)]
            
            if selected_node.node_id not in assignment:
                assignment[selected_node.node_id] = []
            assignment[selected_node.node_id].append(task.task_id)
        
        return assignment
    
    def _performance_based_balance(self, tasks: List[ComputingTask]) -> Dict[str, List[str]]:
        """基于性能的负载均衡"""
        assignment = {}
        available_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
        
        if not available_nodes:
            return {}
        
        # 计算性能分数
        node_scores = {}
        for node in available_nodes:
            recent_performance = self.performance_history.get(node.node_id, [0.5])
            avg_performance = np.mean(recent_performance[-10:]) if recent_performance else 0.5
            node_scores[node.node_id] = avg_performance * (1 - node.load)
        
        # 按性能分数排序
        sorted_nodes = sorted(available_nodes, key=lambda n: node_scores[n.node_id], reverse=True)
        
        # 分配任务
        for i, task in enumerate(tasks):
            selected_node = sorted_nodes[i % len(sorted_nodes)]
            
            if selected_node.node_id not in assignment:
                assignment[selected_node.node_id] = []
            assignment[selected_node.node_id].append(task.task_id)
        
        return assignment
    
    def _simple_round_robin_balance(self, tasks: List[ComputingTask]) -> Dict[str, List[str]]:
        """简单轮询负载均衡"""
        assignment = {}
        available_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
        
        if not available_nodes:
            return {}
        
        # 分配任务
        for i, task in enumerate(tasks):
            selected_node = available_nodes[i % len(available_nodes)]
            
            if selected_node.node_id not in assignment:
                assignment[selected_node.node_id] = []
            assignment[selected_node.node_id].append(task.task_id)
        
        return assignment
    
    def set_balancing_strategy(self, strategy: str) -> None:
        """设置负载均衡策略"""
        valid_strategies = ["weighted_round_robin", "least_connections", "performance_based", "simple_round_robin"]
        if strategy in valid_strategies:
            self.balancing_strategy = strategy
    
    def get_balancing_metrics(self) -> Dict[str, Any]:
        """获取负载均衡指标"""
        return {
            "strategy": self.balancing_strategy,
            "total_nodes": len(self.nodes),
            "average_load": np.mean([n.load for n in self.nodes.values()]) if self.nodes else 0.0,
            "load_distribution": {n.node_id: n.load for n in self.nodes.values()},
            "performance_distribution": {n.node_id: self._calculate_performance_score(n) for n in self.nodes.values()}
        }


class FaultToleranceMechanism:
    """容错机制"""
    
    def __init__(self):
        self.node_health: Dict[str, Dict[str, Any]] = {}
        self.failure_history: Dict[str, List[Dict[str, Any]]] = {}
        self.recovery_strategies = self._load_recovery_strategies()
        self.max_retries = 3
        self.retry_delay = 1.0
        
    def _load_recovery_strategies(self) -> Dict[str, Dict[str, Any]]:
        """加载恢复策略"""
        return {
            "node_failure": {
                "action": "failover",
                "timeout": 30,
                "retry_count": 3
            },
            "task_failure": {
                "action": "retry",
                "timeout": 60,
                "retry_count": 5
            },
            "network_failure": {
                "action": "switch_node",
                "timeout": 15,
                "retry_count": 2
            },
            "performance_degradation": {
                "action": "load_redistribution",
                "timeout": 10,
                "retry_count": 1
            }
        }
    
    def handle_failure(self, failed_node: str, failure_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理故障"""
        
        # 记录故障
        self._record_failure(failed_node, failure_type, context)
        
        # 获取恢复策略
        strategy = self.recovery_strategies.get(failure_type, {})
        
        # 执行恢复操作
        recovery_result = self._execute_recovery_strategy(failed_node, failure_type, strategy, context)
        
        # 更新节点健康状态
        self._update_node_health(failed_node, failure_type, recovery_result)
        
        return recovery_result
    
    def _record_failure(self, node_id: str, failure_type: str, context: Dict[str, Any]) -> None:
        """记录故障"""
        failure_record = {
            "timestamp": time.time(),
            "failure_type": failure_type,
            "context": context,
            "node_id": node_id
        }
        
        if node_id not in self.failure_history:
            self.failure_history[node_id] = []
        
        self.failure_history[node_id].append(failure_record)
        
        # 保持历史记录在合理范围内
        if len(self.failure_history[node_id]) > 50:
            self.failure_history[node_id] = self.failure_history[node_id][-25:]
    
    def _execute_recovery_strategy(self, node_id: str, failure_type: str, 
                                 strategy: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """执行恢复策略"""
        
        action = strategy.get("action", "retry")
        timeout = strategy.get("timeout", 30)
        retry_count = strategy.get("retry_count", 3)
        
        if action == "failover":
            return self._execute_failover(node_id, context)
        elif action == "retry":
            return self._execute_retry(node_id, retry_count, timeout, context)
        elif action == "switch_node":
            return self._execute_switch_node(node_id, context)
        elif action == "load_redistribution":
            return self._execute_load_redistribution(node_id, context)
        else:
            return {"status": "unknown_action", "error": f"未知的恢复动作: {action}"}
    
    def _execute_failover(self, node_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行故障转移"""
        try:
            # 查找备用节点
            backup_nodes = self._find_backup_nodes(node_id)
            
            if not backup_nodes:
                return {"status": "failed", "error": "没有可用的备用节点"}
            
            # 选择最佳备用节点
            best_backup = self._select_best_backup_node(backup_nodes)
            
            # 转移任务
            transferred_tasks = self._transfer_tasks(node_id, best_backup.node_id, context)
            
            return {
                "status": "success",
                "action": "failover",
                "backup_node": best_backup.node_id,
                "transferred_tasks": len(transferred_tasks),
                "recovery_time": time.time()
            }
        except Exception as e:
            return {"status": "failed", "error": f"故障转移失败: {e}"}
    
    def _execute_retry(self, node_id: str, retry_count: int, timeout: float, 
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """执行重试"""
        for attempt in range(retry_count):
            try:
                # 模拟重试
                time.sleep(self.retry_delay)
                
                # 检查节点是否恢复
                if self._check_node_recovery(node_id):
                    return {
                        "status": "success",
                        "action": "retry",
                        "attempts": attempt + 1,
                        "recovery_time": time.time()
                    }
            except Exception as e:
                if attempt == retry_count - 1:
                    return {"status": "failed", "error": f"重试失败: {e}"}
        
        return {"status": "failed", "error": "重试次数已用完"}
    
    def _execute_switch_node(self, node_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行节点切换"""
        try:
            # 查找替代节点
            alternative_nodes = self._find_alternative_nodes(node_id)
            
            if not alternative_nodes:
                return {"status": "failed", "error": "没有可用的替代节点"}
            
            # 选择替代节点
            selected_node = random.choice(alternative_nodes)
            
            # 切换任务
            switched_tasks = self._switch_tasks(node_id, selected_node.node_id, context)
            
            return {
                "status": "success",
                "action": "switch_node",
                "new_node": selected_node.node_id,
                "switched_tasks": len(switched_tasks),
                "recovery_time": time.time()
            }
        except Exception as e:
            return {"status": "failed", "error": f"节点切换失败: {e}"}
    
    def _execute_load_redistribution(self, node_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行负载重分配"""
        try:
            # 获取其他可用节点
            available_nodes = self._find_available_nodes(exclude_node=node_id)
            
            if not available_nodes:
                return {"status": "failed", "error": "没有可用的节点进行负载重分配"}
            
            # 重新分配负载
            redistributed_load = self._redistribute_load(node_id, available_nodes, context)
            
            return {
                "status": "success",
                "action": "load_redistribution",
                "target_nodes": [n.node_id for n in available_nodes],
                "redistributed_load": redistributed_load,
                "recovery_time": time.time()
            }
        except Exception as e:
            return {"status": "failed", "error": f"负载重分配失败: {e}"}
    
    def _find_backup_nodes(self, failed_node_id: str) -> List[EdgeNode]:
        """查找备用节点"""
        # 模拟查找备用节点
        backup_nodes = []
        for node_id, node in self.nodes.items():
            if (node_id != failed_node_id and 
                node.status == NodeStatus.ONLINE and
                node.load < 0.8):  # 负载较低的节点
                backup_nodes.append(node)
        
        return backup_nodes
    
    def _select_best_backup_node(self, backup_nodes: List[EdgeNode]) -> EdgeNode:
        """选择最佳备用节点"""
        if not backup_nodes:
            raise ValueError("没有备用节点")
        
        # 选择负载最低的节点
        return min(backup_nodes, key=lambda n: n.load)
    
    def _transfer_tasks(self, from_node_id: str, to_node_id: str, context: Dict[str, Any]) -> List[str]:
        """转移任务"""
        # 模拟任务转移
        transferred_tasks = []
        tasks_to_transfer = context.get("tasks", [])
        
        for task_id in tasks_to_transfer:
            transferred_tasks.append(task_id)
        
        return transferred_tasks
    
    def _check_node_recovery(self, node_id: str) -> bool:
        """检查节点是否恢复"""
        # 模拟节点恢复检查
        return random.random() > 0.3  # 70%概率恢复
    
    def _find_alternative_nodes(self, failed_node_id: str) -> List[EdgeNode]:
        """查找替代节点"""
        # 模拟查找替代节点
        alternative_nodes = []
        for node_id, node in self.nodes.items():
            if (node_id != failed_node_id and 
                node.status == NodeStatus.ONLINE):
                alternative_nodes.append(node)
        
        return alternative_nodes
    
    def _switch_tasks(self, from_node_id: str, to_node_id: str, context: Dict[str, Any]) -> List[str]:
        """切换任务"""
        # 模拟任务切换
        switched_tasks = []
        tasks_to_switch = context.get("tasks", [])
        
        for task_id in tasks_to_switch:
            switched_tasks.append(task_id)
        
        return switched_tasks
    
    def _find_available_nodes(self, exclude_node: str = None) -> List[EdgeNode]:
        """查找可用节点"""
        available_nodes = []
        for node_id, node in self.nodes.items():
            if (node_id != exclude_node and 
                node.status == NodeStatus.ONLINE and
                node.load < 0.9):  # 负载不太高的节点
                available_nodes.append(node)
        
        return available_nodes
    
    def _redistribute_load(self, failed_node_id: str, target_nodes: List[EdgeNode], 
                          context: Dict[str, Any]) -> Dict[str, float]:
        """重新分配负载"""
        # 模拟负载重分配
        redistributed_load = {}
        load_to_redistribute = context.get("load", 1.0)
        
        for node in target_nodes:
            # 按节点容量分配负载
            capacity = 1.0 - node.load
            allocated_load = min(capacity, load_to_redistribute / len(target_nodes))
            redistributed_load[node.node_id] = allocated_load
        
        return redistributed_load
    
    def _update_node_health(self, node_id: str, failure_type: str, recovery_result: Dict[str, Any]) -> None:
        """更新节点健康状态"""
        if node_id not in self.node_health:
            self.node_health[node_id] = {}
        
        self.node_health[node_id].update({
            "last_failure": time.time(),
            "failure_type": failure_type,
            "recovery_status": recovery_result.get("status", "unknown"),
            "recovery_time": recovery_result.get("recovery_time", 0)
        })
    
    def get_fault_tolerance_metrics(self) -> Dict[str, Any]:
        """获取容错指标"""
        return {
            "total_nodes": len(self.nodes),
            "healthy_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]),
            "failed_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.ERROR]),
            "failure_history": {node_id: len(history) for node_id, history in self.failure_history.items()},
            "recovery_strategies": list(self.recovery_strategies.keys())
        }


class EdgeComputingOptimization:
    """边缘计算优化主类"""
    
    def __init__(self):
        self.distributed_computing = DistributedComputing()
        self.load_balancer = IntelligentLoadBalancer()
        self.fault_tolerance = FaultToleranceMechanism()
        self.nodes: Dict[str, EdgeNode] = {}
        
    def add_node(self, node: EdgeNode) -> None:
        """添加节点"""
        self.nodes[node.node_id] = node
        self.distributed_computing.add_node(node)
        self.load_balancer.nodes[node.node_id] = node
        self.fault_tolerance.nodes = self.nodes
    
    def remove_node(self, node_id: str) -> None:
        """移除节点"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.distributed_computing.remove_node(node_id)
            if node_id in self.load_balancer.nodes:
                del self.load_balancer.nodes[node_id]
    
    def distribute_computation(self, task: ComputingTask, nodes: List[str]) -> Dict[str, Any]:
        """分布式计算"""
        return self.distributed_computing.execute(task, nodes)
    
    def balance_load(self, nodes: List[str], tasks: List[ComputingTask]) -> Dict[str, List[str]]:
        """负载均衡"""
        return self.load_balancer.balance(nodes, tasks)
    
    def handle_fault(self, failed_node: str, failure_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理故障"""
        return self.fault_tolerance.handle_failure(failed_node, failure_type, context)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "distributed_computing": self.distributed_computing.get_system_status(),
            "load_balancing": self.load_balancer.get_balancing_metrics(),
            "fault_tolerance": self.fault_tolerance.get_fault_tolerance_metrics(),
            "total_nodes": len(self.nodes),
            "node_status": {node_id: node.status.value for node_id, node in self.nodes.items()}
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """性能优化"""
        optimization_results = {}
        
        # 负载均衡优化
        if self.nodes:
            avg_load = np.mean([n.load for n in self.nodes.values()])
            if avg_load > 0.8:
                optimization_results["load_balancing"] = "需要负载均衡优化"
                self.load_balancer.set_balancing_strategy("performance_based")
        
        # 容错机制优化
        failed_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ERROR]
        if failed_nodes:
            optimization_results["fault_tolerance"] = f"检测到{len(failed_nodes)}个故障节点"
        
        # 性能监控
        performance_metrics = {}
        for node_id, node in self.nodes.items():
            performance_metrics[node_id] = {
                "load": node.load,
                "performance_score": self._calculate_node_performance(node)
            }
        
        optimization_results["performance_metrics"] = performance_metrics
        
        return optimization_results
    
    def _calculate_node_performance(self, node: EdgeNode) -> float:
        """计算节点性能分数"""
        cpu_performance = node.performance_metrics.get("cpu_performance", 0.5)
        memory_usage = node.performance_metrics.get("memory_usage", 0.5)
        network_latency = node.performance_metrics.get("network_latency", 0.0)
        
        # 性能分数 = CPU性能 * (1 - 内存使用率) * (1 - 网络延迟) * (1 - 负载)
        return cpu_performance * (1 - memory_usage) * (1 - network_latency) * (1 - node.load)


# 使用示例
if __name__ == "__main__":
    import numpy as np
    
    # 创建边缘计算优化系统
    edge_optimizer = EdgeComputingOptimization()
    
    # 添加节点
    nodes = [
        EdgeNode(
            node_id="node1",
            host="192.168.1.10",
            port=8080,
            status=NodeStatus.ONLINE,
            capabilities={"supported_features": ["detection_analysis", "behavior_simulation"]},
            load=0.3,
            last_heartbeat=time.time(),
            performance_metrics={"cpu_performance": 0.8, "memory_usage": 0.4, "network_latency": 0.1}
        ),
        EdgeNode(
            node_id="node2",
            host="192.168.1.11",
            port=8080,
            status=NodeStatus.ONLINE,
            capabilities={"supported_features": ["pattern_recognition", "behavior_simulation"]},
            load=0.5,
            last_heartbeat=time.time(),
            performance_metrics={"cpu_performance": 0.7, "memory_usage": 0.6, "network_latency": 0.2}
        )
    ]
    
    for node in nodes:
        edge_optimizer.add_node(node)
    
    # 创建任务
    tasks = [
        ComputingTask(
            task_id="task1",
            task_type="detection_analysis",
            priority=TaskPriority.HIGH,
            data={"detection_data": "sample_data"},
            requirements={"capabilities": ["detection_analysis"]},
            created_at=time.time()
        ),
        ComputingTask(
            task_id="task2",
            task_type="behavior_simulation",
            priority=TaskPriority.NORMAL,
            data={"behavior_data": "sample_behavior"},
            requirements={"capabilities": ["behavior_simulation"]},
            created_at=time.time()
        )
    ]
    
    # 执行分布式计算
    result = edge_optimizer.distribute_computation(tasks[0], ["node1", "node2"])
    print("分布式计算结果:", result)
    
    # 负载均衡
    balance_result = edge_optimizer.balance_load(["node1", "node2"], tasks)
    print("负载均衡结果:", balance_result)
    
    # 获取系统状态
    status = edge_optimizer.get_system_status()
    print("系统状态:", json.dumps(status, indent=2, ensure_ascii=False))
    
    # 性能优化
    optimization = edge_optimizer.optimize_performance()
    print("性能优化结果:", optimization) 