#!/usr/bin/env python3
"""
自适应学习模块 - 根据环境变化自动调整策略
"""
import time
import random
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque
from loguru import logger


class LearningType(Enum):
    """学习类型"""
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    TRANSFER = "transfer"
    META = "meta"


class EnvironmentType(Enum):
    """环境类型"""
    STABLE = "stable"
    CHANGING = "changing"
    ADVERSARIAL = "adversarial"
    UNKNOWN = "unknown"


@dataclass
class LearningContext:
    """学习上下文"""
    environment_type: EnvironmentType
    success_rate: float
    detection_rate: float
    performance_metrics: Dict[str, float]
    adaptation_needed: bool
    timestamp: float


@dataclass
class AdaptationStrategy:
    """适应策略"""
    strategy_id: str
    strategy_type: str
    parameters: Dict[str, Any]
    success_rate: float
    last_used: float
    usage_count: int


class AdaptiveLearningEngine:
    """自适应学习引擎"""
    
    def __init__(self, memory_size: int = 1000):
        self.logger = logger.bind(name="adaptive_learning_engine")
        self.memory_size = memory_size
        self.experience_buffer = deque(maxlen=memory_size)
        self.adaptation_strategies: Dict[str, AdaptationStrategy] = {}
        self.current_environment = EnvironmentType.UNKNOWN
        self.learning_rate = 0.1
        self.exploration_rate = 0.3
        self.performance_threshold = 0.8
        
        # 初始化策略
        self._initialize_strategies()
        
    def _initialize_strategies(self):
        """初始化适应策略"""
        strategies = [
            AdaptationStrategy(
                strategy_id="conservative",
                strategy_type="defensive",
                parameters={"risk_tolerance": 0.1, "adaptation_speed": 0.5},
                success_rate=0.8,
                last_used=time.time(),
                usage_count=0
            ),
            AdaptationStrategy(
                strategy_id="aggressive",
                strategy_type="offensive",
                parameters={"risk_tolerance": 0.8, "adaptation_speed": 0.9},
                success_rate=0.7,
                last_used=time.time(),
                usage_count=0
            ),
            AdaptationStrategy(
                strategy_id="balanced",
                strategy_type="adaptive",
                parameters={"risk_tolerance": 0.5, "adaptation_speed": 0.7},
                success_rate=0.85,
                last_used=time.time(),
                usage_count=0
            ),
            AdaptationStrategy(
                strategy_id="reactive",
                strategy_type="responsive",
                parameters={"risk_tolerance": 0.3, "adaptation_speed": 0.8},
                success_rate=0.75,
                last_used=time.time(),
                usage_count=0
            )
        ]
        
        for strategy in strategies:
            self.adaptation_strategies[strategy.strategy_id] = strategy
    
    def learn_from_experience(self, context: LearningContext) -> Dict[str, Any]:
        """从经验中学习"""
        
        # 记录经验
        self.experience_buffer.append(context)
        
        # 分析环境变化
        environment_change = self._analyze_environment_change(context)
        
        # 选择适应策略
        selected_strategy = self._select_adaptation_strategy(context, environment_change)
        
        # 应用策略
        adaptation_result = self._apply_adaptation_strategy(selected_strategy, context)
        
        # 更新策略性能
        self._update_strategy_performance(selected_strategy, context)
        
        # 学习新策略
        new_strategies = self._learn_new_strategies(context)
        
        return {
            "environment_change": environment_change,
            "selected_strategy": selected_strategy.strategy_id,
            "adaptation_result": adaptation_result,
            "new_strategies": new_strategies,
            "learning_progress": self._get_learning_progress()
        }
    
    def _analyze_environment_change(self, context: LearningContext) -> Dict[str, Any]:
        """分析环境变化"""
        if len(self.experience_buffer) < 2:
            return {"change_detected": False, "change_magnitude": 0.0}
        
        # 获取最近的上下文
        recent_contexts = list(self.experience_buffer)[-10:]
        
        # 计算变化指标
        success_rates = [ctx.success_rate for ctx in recent_contexts]
        detection_rates = [ctx.detection_rate for ctx in recent_contexts]
        
        # 计算变化幅度
        success_rate_change = np.std(success_rates) if len(success_rates) > 1 else 0.0
        detection_rate_change = np.std(detection_rates) if len(detection_rates) > 1 else 0.0
        
        # 确定环境类型
        total_change = success_rate_change + detection_rate_change
        
        if total_change < 0.1:
            environment_type = EnvironmentType.STABLE
        elif total_change < 0.3:
            environment_type = EnvironmentType.CHANGING
        else:
            environment_type = EnvironmentType.ADVERSARIAL
        
        self.current_environment = environment_type
        
        return {
            "change_detected": total_change > 0.05,
            "change_magnitude": total_change,
            "environment_type": environment_type.value,
            "success_rate_volatility": success_rate_change,
            "detection_rate_volatility": detection_rate_change
        }
    
    def _select_adaptation_strategy(self, context: LearningContext, 
                                  environment_change: Dict[str, Any]) -> AdaptationStrategy:
        """选择适应策略"""
        
        # 根据环境类型和性能选择策略
        available_strategies = []
        
        for strategy in self.adaptation_strategies.values():
            # 计算策略适用性分数
            applicability_score = self._calculate_strategy_applicability(
                strategy, context, environment_change
            )
            
            available_strategies.append((strategy, applicability_score))
        
        # 按适用性分数排序
        available_strategies.sort(key=lambda x: x[1], reverse=True)
        
        # 探索与利用平衡
        if random.random() < self.exploration_rate:
            # 探索：随机选择策略
            selected_strategy = random.choice(available_strategies)[0]
        else:
            # 利用：选择最佳策略
            selected_strategy = available_strategies[0][0]
        
        # 更新策略使用信息
        selected_strategy.last_used = time.time()
        selected_strategy.usage_count += 1
        
        return selected_strategy
    
    def _calculate_strategy_applicability(self, strategy: AdaptationStrategy, 
                                        context: LearningContext,
                                        environment_change: Dict[str, Any]) -> float:
        """计算策略适用性分数"""
        
        base_score = strategy.success_rate
        
        # 环境匹配度
        environment_match = 0.0
        if environment_change["environment_type"] == "stable":
            if strategy.strategy_type == "defensive":
                environment_match = 0.8
        elif environment_change["environment_type"] == "changing":
            if strategy.strategy_type == "adaptive":
                environment_match = 0.9
        elif environment_change["environment_type"] == "adversarial":
            if strategy.strategy_type == "responsive":
                environment_match = 0.8
        
        # 性能匹配度
        performance_match = 0.0
        if context.success_rate < 0.5:
            # 低成功率，需要激进策略
            if strategy.parameters["risk_tolerance"] > 0.6:
                performance_match = 0.8
        elif context.success_rate > 0.8:
            # 高成功率，可以保守
            if strategy.parameters["risk_tolerance"] < 0.4:
                performance_match = 0.8
        
        # 时间衰减
        time_decay = max(0.1, 1.0 - (time.time() - strategy.last_used) / 3600)
        
        # 综合分数
        total_score = (base_score * 0.4 + 
                      environment_match * 0.3 + 
                      performance_match * 0.2 + 
                      time_decay * 0.1)
        
        return total_score
    
    def _apply_adaptation_strategy(self, strategy: AdaptationStrategy, 
                                  context: LearningContext) -> Dict[str, Any]:
        """应用适应策略"""
        
        result = {
            "strategy_applied": strategy.strategy_id,
            "parameters_used": strategy.parameters,
            "adaptation_actions": [],
            "expected_improvement": 0.0
        }
        
        # 根据策略类型执行适应动作
        if strategy.strategy_type == "defensive":
            actions = self._apply_defensive_strategy(strategy, context)
        elif strategy.strategy_type == "offensive":
            actions = self._apply_offensive_strategy(strategy, context)
        elif strategy.strategy_type == "adaptive":
            actions = self._apply_adaptive_strategy(strategy, context)
        elif strategy.strategy_type == "responsive":
            actions = self._apply_responsive_strategy(strategy, context)
        else:
            actions = []
        
        result["adaptation_actions"] = actions
        
        # 计算预期改进
        expected_improvement = self._calculate_expected_improvement(strategy, context)
        result["expected_improvement"] = expected_improvement
        
        return result
    
    def _apply_defensive_strategy(self, strategy: AdaptationStrategy, 
                                context: LearningContext) -> List[Dict[str, Any]]:
        """应用防御策略"""
        actions = []
        
        # 降低风险
        if context.detection_rate > 0.3:
            actions.append({
                "action": "reduce_aggression",
                "parameter": "risk_tolerance",
                "value": strategy.parameters["risk_tolerance"] * 0.8
            })
        
        # 增加隐蔽性
        if context.success_rate < 0.7:
            actions.append({
                "action": "enhance_stealth",
                "parameter": "stealth_level",
                "value": 0.9
            })
        
        # 降低速度
        actions.append({
            "action": "slow_down",
            "parameter": "operation_speed",
            "value": 0.6
        })
        
        return actions
    
    def _apply_offensive_strategy(self, strategy: AdaptationStrategy, 
                                context: LearningContext) -> List[Dict[str, Any]]:
        """应用进攻策略"""
        actions = []
        
        # 提高成功率
        if context.success_rate < 0.8:
            actions.append({
                "action": "increase_aggression",
                "parameter": "risk_tolerance",
                "value": min(0.9, strategy.parameters["risk_tolerance"] * 1.2)
            })
        
        # 加快速度
        actions.append({
            "action": "speed_up",
            "parameter": "operation_speed",
            "value": 1.2
        })
        
        # 增加并发
        actions.append({
            "action": "increase_concurrency",
            "parameter": "concurrent_operations",
            "value": 3
        })
        
        return actions
    
    def _apply_adaptive_strategy(self, strategy: AdaptationStrategy, 
                               context: LearningContext) -> List[Dict[str, Any]]:
        """应用自适应策略"""
        actions = []
        
        # 动态调整参数
        adaptation_speed = strategy.parameters["adaptation_speed"]
        
        if context.success_rate < 0.6:
            # 性能差，增加适应性
            actions.append({
                "action": "increase_adaptation",
                "parameter": "adaptation_speed",
                "value": min(1.0, adaptation_speed * 1.3)
            })
        elif context.success_rate > 0.9:
            # 性能好，减少变化
            actions.append({
                "action": "stabilize",
                "parameter": "adaptation_speed",
                "value": max(0.3, adaptation_speed * 0.8)
            })
        
        # 平衡风险
        actions.append({
            "action": "balance_risk",
            "parameter": "risk_tolerance",
            "value": 0.5
        })
        
        return actions
    
    def _apply_responsive_strategy(self, strategy: AdaptationStrategy, 
                                 context: LearningContext) -> List[Dict[str, Any]]:
        """应用响应策略"""
        actions = []
        
        # 快速响应环境变化
        if context.adaptation_needed:
            actions.append({
                "action": "quick_response",
                "parameter": "response_time",
                "value": 0.1
            })
        
        # 动态调整检测率
        if context.detection_rate > 0.5:
            actions.append({
                "action": "evade_detection",
                "parameter": "evasion_level",
                "value": 0.9
            })
        
        # 优化性能
        actions.append({
            "action": "optimize_performance",
            "parameter": "performance_target",
            "value": 0.95
        })
        
        return actions
    
    def _calculate_expected_improvement(self, strategy: AdaptationStrategy, 
                                      context: LearningContext) -> float:
        """计算预期改进"""
        
        base_improvement = 0.1
        
        # 根据策略类型调整
        if strategy.strategy_type == "defensive":
            if context.detection_rate > 0.3:
                base_improvement += 0.2
        elif strategy.strategy_type == "offensive":
            if context.success_rate < 0.7:
                base_improvement += 0.3
        elif strategy.strategy_type == "adaptive":
            base_improvement += 0.15
        elif strategy.strategy_type == "responsive":
            if context.adaptation_needed:
                base_improvement += 0.25
        
        # 根据历史成功率调整
        historical_improvement = strategy.success_rate - 0.5
        base_improvement += historical_improvement * 0.5
        
        return min(1.0, max(0.0, base_improvement))
    
    def _update_strategy_performance(self, strategy: AdaptationStrategy, 
                                   context: LearningContext):
        """更新策略性能"""
        
        # 计算新的成功率
        current_performance = context.success_rate
        
        # 使用指数移动平均更新成功率
        alpha = 0.1  # 学习率
        strategy.success_rate = (alpha * current_performance + 
                               (1 - alpha) * strategy.success_rate)
    
    def _learn_new_strategies(self, context: LearningContext) -> List[Dict[str, Any]]:
        """学习新策略"""
        new_strategies = []
        
        # 检查是否需要学习新策略
        if len(self.experience_buffer) > 50:
            # 分析成功模式
            successful_patterns = self._analyze_successful_patterns()
            
            for pattern in successful_patterns:
                if self._should_create_new_strategy(pattern):
                    new_strategy = self._create_strategy_from_pattern(pattern)
                    self.adaptation_strategies[new_strategy.strategy_id] = new_strategy
                    new_strategies.append(asdict(new_strategy))
        
        return new_strategies
    
    def _analyze_successful_patterns(self) -> List[Dict[str, Any]]:
        """分析成功模式"""
        patterns = []
        
        # 获取成功的经验
        successful_experiences = [
            exp for exp in self.experience_buffer 
            if exp.success_rate > 0.8
        ]
        
        if len(successful_experiences) < 5:
            return patterns
        
        # 分析模式
        for i in range(len(successful_experiences) - 1):
            exp1 = successful_experiences[i]
            exp2 = successful_experiences[i + 1]
            
            pattern = {
                "environment_type": exp1.environment_type,
                "success_rate": (exp1.success_rate + exp2.success_rate) / 2,
                "detection_rate": (exp1.detection_rate + exp2.detection_rate) / 2,
                "adaptation_speed": 0.7,
                "risk_tolerance": 0.5
            }
            
            patterns.append(pattern)
        
        return patterns
    
    def _should_create_new_strategy(self, pattern: Dict[str, Any]) -> bool:
        """判断是否应该创建新策略"""
        
        # 检查是否已有类似策略
        for strategy in self.adaptation_strategies.values():
            if (abs(strategy.parameters.get("risk_tolerance", 0.5) - pattern["risk_tolerance"]) < 0.1 and
                abs(strategy.parameters.get("adaptation_speed", 0.7) - pattern["adaptation_speed"]) < 0.1):
                return False
        
        # 检查模式是否足够好
        return pattern["success_rate"] > 0.8 and pattern["detection_rate"] < 0.3
    
    def _create_strategy_from_pattern(self, pattern: Dict[str, Any]) -> AdaptationStrategy:
        """从模式创建策略"""
        
        strategy_id = f"learned_{int(time.time())}"
        
        return AdaptationStrategy(
            strategy_id=strategy_id,
            strategy_type="learned",
            parameters={
                "risk_tolerance": pattern["risk_tolerance"],
                "adaptation_speed": pattern["adaptation_speed"],
                "environment_type": pattern["environment_type"].value
            },
            success_rate=pattern["success_rate"],
            last_used=time.time(),
            usage_count=0
        )
    
    def _get_learning_progress(self) -> Dict[str, Any]:
        """获取学习进度"""
        return {
            "total_experiences": len(self.experience_buffer),
            "current_environment": self.current_environment.value,
            "total_strategies": len(self.adaptation_strategies),
            "average_success_rate": np.mean([exp.success_rate for exp in self.experience_buffer]) if self.experience_buffer else 0.0,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate
        }


class MetaLearningOptimizer:
    """元学习优化器"""
    
    def __init__(self):
        self.logger = logger.bind(name="meta_learning_optimizer")
        self.task_performance = {}
        self.transfer_learning_data = {}
        self.meta_parameters = {
            "learning_rate": 0.01,
            "meta_batch_size": 5,
            "adaptation_steps": 3
        }
    
    def optimize_meta_learning(self, task_id: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化元学习"""
        
        # 记录任务性能
        self.task_performance[task_id] = performance_data
        
        # 分析任务相似性
        similar_tasks = self._find_similar_tasks(task_id)
        
        # 应用迁移学习
        transfer_result = self._apply_transfer_learning(task_id, similar_tasks)
        
        # 更新元参数
        meta_update = self._update_meta_parameters(task_id, performance_data)
        
        return {
            "similar_tasks": similar_tasks,
            "transfer_result": transfer_result,
            "meta_update": meta_update,
            "optimization_applied": True
        }
    
    def _find_similar_tasks(self, task_id: str) -> List[str]:
        """查找相似任务"""
        similar_tasks = []
        
        current_task = self.task_performance.get(task_id, {})
        if not current_task:
            return similar_tasks
        
        for other_task_id, other_task in self.task_performance.items():
            if other_task_id == task_id:
                continue
            
            # 计算相似度
            similarity = self._calculate_task_similarity(current_task, other_task)
            
            if similarity > 0.7:
                similar_tasks.append(other_task_id)
        
        return similar_tasks
    
    def _calculate_task_similarity(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """计算任务相似度"""
        
        # 提取特征
        features1 = self._extract_task_features(task1)
        features2 = self._extract_task_features(task2)
        
        if not features1 or not features2:
            return 0.0
        
        # 计算余弦相似度
        dot_product = sum(f1 * f2 for f1, f2 in zip(features1, features2))
        norm1 = sum(f1 * f1 for f1 in features1) ** 0.5
        norm2 = sum(f2 * f2 for f2 in features2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _extract_task_features(self, task: Dict[str, Any]) -> List[float]:
        """提取任务特征"""
        features = []
        
        # 性能特征
        features.extend([
            task.get("success_rate", 0.5),
            task.get("detection_rate", 0.5),
            task.get("response_time", 1.0),
            task.get("adaptation_speed", 0.7)
        ])
        
        # 环境特征
        environment_type = task.get("environment_type", "unknown")
        if environment_type == "stable":
            features.extend([1, 0, 0])
        elif environment_type == "changing":
            features.extend([0, 1, 0])
        elif environment_type == "adversarial":
            features.extend([0, 0, 1])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _apply_transfer_learning(self, task_id: str, similar_tasks: List[str]) -> Dict[str, Any]:
        """应用迁移学习"""
        
        if not similar_tasks:
            return {"success": False, "reason": "no_similar_tasks"}
        
        # 收集相似任务的知识
        transferred_knowledge = []
        
        for similar_task in similar_tasks:
            task_data = self.task_performance[similar_task]
            
            knowledge = {
                "task_id": similar_task,
                "successful_strategies": task_data.get("successful_strategies", []),
                "optimal_parameters": task_data.get("optimal_parameters", {}),
                "performance_metrics": task_data.get("performance_metrics", {})
            }
            
            transferred_knowledge.append(knowledge)
        
        # 应用迁移的知识
        adaptation_result = self._adapt_from_transferred_knowledge(task_id, transferred_knowledge)
        
        return {
            "success": True,
            "transferred_tasks": similar_tasks,
            "transferred_knowledge": transferred_knowledge,
            "adaptation_result": adaptation_result
        }
    
    def _adapt_from_transferred_knowledge(self, task_id: str, 
                                        transferred_knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从迁移知识中适应"""
        
        # 聚合最优参数
        optimal_parameters = {}
        
        for knowledge in transferred_knowledge:
            params = knowledge.get("optimal_parameters", {})
            for key, value in params.items():
                if key not in optimal_parameters:
                    optimal_parameters[key] = []
                optimal_parameters[key].append(value)
        
        # 计算平均参数
        adapted_parameters = {}
        for key, values in optimal_parameters.items():
            if values:
                adapted_parameters[key] = sum(values) / len(values)
        
        return {
            "adapted_parameters": adapted_parameters,
            "confidence": min(1.0, len(transferred_knowledge) * 0.2)
        }
    
    def _update_meta_parameters(self, task_id: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新元参数"""
        
        # 根据性能调整元参数
        success_rate = performance_data.get("success_rate", 0.5)
        
        if success_rate > 0.8:
            # 性能好，减少学习率
            self.meta_parameters["learning_rate"] *= 0.9
        elif success_rate < 0.5:
            # 性能差，增加学习率
            self.meta_parameters["learning_rate"] *= 1.1
        
        # 确保参数在合理范围内
        self.meta_parameters["learning_rate"] = max(0.001, min(0.1, self.meta_parameters["learning_rate"]))
        
        return {
            "updated_parameters": self.meta_parameters.copy(),
            "performance_based_adjustment": True
        }


class AdaptiveLearningSystem:
    """自适应学习系统主类"""
    
    def __init__(self):
        self.learning_engine = AdaptiveLearningEngine()
        self.meta_optimizer = MetaLearningOptimizer()
        self.logger = logger.bind(name="adaptive_learning_system")
        
    def process_learning_request(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理学习请求"""
        
        # 创建学习上下文
        context = LearningContext(
            environment_type=EnvironmentType(context_data.get("environment_type", "unknown")),
            success_rate=context_data.get("success_rate", 0.5),
            detection_rate=context_data.get("detection_rate", 0.5),
            performance_metrics=context_data.get("performance_metrics", {}),
            adaptation_needed=context_data.get("adaptation_needed", False),
            timestamp=time.time()
        )
        
        # 执行学习
        learning_result = self.learning_engine.learn_from_experience(context)
        
        # 元学习优化
        meta_result = self.meta_optimizer.optimize_meta_learning(
            context_data.get("task_id", "default"),
            context_data
        )
        
        return {
            "learning_result": learning_result,
            "meta_result": meta_result,
            "system_status": self.get_system_status()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "learning_engine_status": self.learning_engine._get_learning_progress(),
            "meta_optimizer_status": {
                "total_tasks": len(self.meta_optimizer.task_performance),
                "meta_parameters": self.meta_optimizer.meta_parameters
            },
            "total_strategies": len(self.learning_engine.adaptation_strategies),
            "current_environment": self.learning_engine.current_environment.value
        }


# 使用示例
if __name__ == "__main__":
    # 创建自适应学习系统
    adaptive_system = AdaptiveLearningSystem()
    
    # 模拟学习请求
    context_data = {
        "task_id": "ticket_grabbing_001",
        "environment_type": "changing",
        "success_rate": 0.75,
        "detection_rate": 0.25,
        "performance_metrics": {
            "response_time": 1.2,
            "throughput": 10.5,
            "error_rate": 0.05
        },
        "adaptation_needed": True
    }
    
    # 处理学习请求
    result = adaptive_system.process_learning_request(context_data)
    
    print("自适应学习结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    # 获取系统状态
    status = adaptive_system.get_system_status()
    print("\n系统状态:")
    print(json.dumps(status, indent=2, ensure_ascii=False)) 