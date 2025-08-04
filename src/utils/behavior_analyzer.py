"""
行为分析器 - 分析和优化用户行为模式
"""
import time
import random
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import numpy as np


class BehaviorAnalyzer:
    """行为分析器"""
    
    def __init__(self):
        self.logger = logger.bind(name="behavior_analyzer")
        
        # 行为模式数据库
        self.behavior_patterns = {
            "typing_speed": {
                "fast": {"mean": 0.1, "std": 0.02, "range": (0.05, 0.15)},
                "normal": {"mean": 0.3, "std": 0.05, "range": (0.2, 0.4)},
                "slow": {"mean": 0.6, "std": 0.1, "range": (0.4, 0.8)}
            },
            "click_intervals": {
                "rapid": {"mean": 0.2, "std": 0.05, "range": (0.1, 0.3)},
                "normal": {"mean": 0.8, "std": 0.15, "range": (0.5, 1.2)},
                "thoughtful": {"mean": 2.0, "std": 0.5, "range": (1.0, 3.5)}
            },
            "mouse_movements": {
                "direct": {"speed": "fast", "path": "straight"},
                "natural": {"speed": "variable", "path": "curved"},
                "hesitant": {"speed": "slow", "path": "zigzag"}
            },
            "reading_patterns": {
                "fast_reader": {"time_per_char": 0.01, "pause_probability": 0.1},
                "normal_reader": {"time_per_char": 0.02, "pause_probability": 0.2},
                "slow_reader": {"time_per_char": 0.04, "pause_probability": 0.3}
            }
        }
        
        # 用户行为历史
        self.behavior_history = []
        
        # 当前行为模式
        self.current_pattern = "normal"
        
        # 行为一致性检查
        self.consistency_checker = BehaviorConsistencyChecker()
    
    def analyze_user_behavior(self, action_type: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析用户行为"""
        analysis = {
            "action_type": action_type,
            "timestamp": time.time(),
            "pattern_score": 0.0,
            "naturalness_score": 0.0,
            "consistency_score": 0.0,
            "recommendations": []
        }
        
        # 分析行为模式
        pattern_analysis = self._analyze_pattern(action_type, action_data)
        analysis.update(pattern_analysis)
        
        # 检查自然度
        naturalness = self._check_naturalness(action_type, action_data)
        analysis["naturalness_score"] = naturalness
        
        # 检查一致性
        consistency = self.consistency_checker.check_consistency(action_type, action_data)
        analysis["consistency_score"] = consistency
        
        # 生成建议
        recommendations = self._generate_recommendations(analysis)
        analysis["recommendations"] = recommendations
        
        # 记录行为历史
        self.behavior_history.append(analysis)
        
        return analysis
    
    def _analyze_pattern(self, action_type: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析行为模式"""
        pattern_score = 0.0
        detected_pattern = "unknown"
        
        if action_type == "typing":
            pattern_score, detected_pattern = self._analyze_typing_pattern(action_data)
        elif action_type == "clicking":
            pattern_score, detected_pattern = self._analyze_clicking_pattern(action_data)
        elif action_type == "scrolling":
            pattern_score, detected_pattern = self._analyze_scrolling_pattern(action_data)
        elif action_type == "reading":
            pattern_score, detected_pattern = self._analyze_reading_pattern(action_data)
        
        return {
            "pattern_score": pattern_score,
            "detected_pattern": detected_pattern
        }
    
    def _analyze_typing_pattern(self, action_data: Dict[str, Any]) -> Tuple[float, str]:
        """分析打字模式"""
        typing_speed = action_data.get("typing_speed", 0.3)
        intervals = action_data.get("intervals", [])
        
        # 计算打字速度模式
        if typing_speed < 0.2:
            pattern = "fast"
            score = 0.9
        elif typing_speed < 0.5:
            pattern = "normal"
            score = 0.8
        else:
            pattern = "slow"
            score = 0.7
        
        # 检查间隔的一致性
        if intervals:
            interval_variance = np.var(intervals)
            if interval_variance < 0.01:  # 过于规律
                score *= 0.5
                pattern = "robotic"
        
        return score, pattern
    
    def _analyze_clicking_pattern(self, action_data: Dict[str, Any]) -> Tuple[float, str]:
        """分析点击模式"""
        click_interval = action_data.get("interval", 1.0)
        click_position = action_data.get("position", (0, 0))
        
        # 分析点击间隔
        if click_interval < 0.3:
            pattern = "rapid"
            score = 0.6
        elif click_interval < 1.5:
            pattern = "normal"
            score = 0.8
        else:
            pattern = "thoughtful"
            score = 0.9
        
        # 检查点击位置的随机性
        if hasattr(self, 'last_click_position'):
            distance = np.sqrt(
                (click_position[0] - self.last_click_position[0])**2 +
                (click_position[1] - self.last_click_position[1])**2
            )
            if distance < 10:  # 点击位置过于接近
                score *= 0.7
        
        self.last_click_position = click_position
        return score, pattern
    
    def _analyze_scrolling_pattern(self, action_data: Dict[str, Any]) -> Tuple[float, str]:
        """分析滚动模式"""
        scroll_speed = action_data.get("speed", 1.0)
        scroll_direction = action_data.get("direction", "down")
        
        # 分析滚动速度
        if scroll_speed > 5.0:
            pattern = "fast"
            score = 0.6
        elif scroll_speed > 1.0:
            pattern = "normal"
            score = 0.8
        else:
            pattern = "slow"
            score = 0.9
        
        # 检查滚动方向的自然性
        if scroll_direction not in ["up", "down"]:
            score *= 0.8
        
        return score, pattern
    
    def _analyze_reading_pattern(self, action_data: Dict[str, Any]) -> Tuple[float, str]:
        """分析阅读模式"""
        reading_time = action_data.get("reading_time", 1.0)
        text_length = action_data.get("text_length", 100)
        
        # 计算阅读速度
        chars_per_second = text_length / reading_time
        
        if chars_per_second > 20:
            pattern = "fast_reader"
            score = 0.7
        elif chars_per_second > 10:
            pattern = "normal_reader"
            score = 0.8
        else:
            pattern = "slow_reader"
            score = 0.9
        
        return score, pattern
    
    def _check_naturalness(self, action_type: str, action_data: Dict[str, Any]) -> float:
        """检查行为的自然度"""
        naturalness_score = 1.0
        
        # 检查时间间隔的随机性
        if "intervals" in action_data:
            intervals = action_data["intervals"]
            if len(intervals) > 1:
                # 计算间隔的变异系数
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                cv = std_interval / mean_interval if mean_interval > 0 else 0
                
                # 变异系数应该在合理范围内
                if cv < 0.1:  # 过于规律
                    naturalness_score *= 0.5
                elif cv > 2.0:  # 过于随机
                    naturalness_score *= 0.7
                else:
                    naturalness_score *= 1.0
        
        # 检查速度的一致性
        if "speed" in action_data:
            speed = action_data["speed"]
            if speed < 0.1 or speed > 10.0:  # 不合理的速度
                naturalness_score *= 0.6
        
        # 检查位置的合理性
        if "position" in action_data:
            position = action_data["position"]
            if position[0] < 0 or position[1] < 0:  # 不合理的位置
                naturalness_score *= 0.5
        
        return naturalness_score
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成行为优化建议"""
        recommendations = []
        
        # 基于模式分数
        if analysis["pattern_score"] < 0.6:
            recommendations.append("行为模式过于机械化，建议增加随机性")
        
        # 基于自然度分数
        if analysis["naturalness_score"] < 0.7:
            recommendations.append("行为不够自然，建议调整时间间隔")
        
        # 基于一致性分数
        if analysis["consistency_score"] < 0.8:
            recommendations.append("行为一致性较差，建议保持稳定的行为模式")
        
        # 基于具体行为类型
        if analysis["action_type"] == "typing" and analysis["pattern_score"] < 0.7:
            recommendations.append("打字速度过于规律，建议添加随机停顿")
        
        if analysis["action_type"] == "clicking" and analysis["naturalness_score"] < 0.8:
            recommendations.append("点击行为不够自然，建议调整点击间隔")
        
        if not recommendations:
            recommendations.append("行为模式正常，继续保持")
        
        return recommendations
    
    def optimize_behavior(self, action_type: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化行为数据"""
        optimized_data = current_data.copy()
        
        # 根据行为类型优化
        if action_type == "typing":
            optimized_data = self._optimize_typing_behavior(optimized_data)
        elif action_type == "clicking":
            optimized_data = self._optimize_clicking_behavior(optimized_data)
        elif action_type == "scrolling":
            optimized_data = self._optimize_scrolling_behavior(optimized_data)
        elif action_type == "reading":
            optimized_data = self._optimize_reading_behavior(optimized_data)
        
        return optimized_data
    
    def _optimize_typing_behavior(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """优化打字行为"""
        # 添加随机停顿
        if "intervals" in data:
            intervals = data["intervals"]
            if len(intervals) > 0:
                # 随机添加停顿
                if random.random() < 0.1:  # 10%概率添加停顿
                    pause_index = random.randint(0, len(intervals))
                    intervals.insert(pause_index, random.uniform(0.5, 2.0))
                    data["intervals"] = intervals
        
        # 调整打字速度
        if "typing_speed" in data:
            current_speed = data["typing_speed"]
            # 添加微小的随机变化
            variation = random.uniform(-0.1, 0.1)
            data["typing_speed"] = max(0.05, current_speed + variation)
        
        return data
    
    def _optimize_clicking_behavior(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """优化点击行为"""
        # 调整点击间隔
        if "interval" in data:
            current_interval = data["interval"]
            # 添加随机变化
            variation = random.uniform(-0.2, 0.2)
            data["interval"] = max(0.1, current_interval + variation)
        
        # 调整点击位置
        if "position" in data:
            position = data["position"]
            # 添加微小的位置偏移
            offset_x = random.uniform(-5, 5)
            offset_y = random.uniform(-5, 5)
            data["position"] = (position[0] + offset_x, position[1] + offset_y)
        
        return data
    
    def _optimize_scrolling_behavior(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """优化滚动行为"""
        # 调整滚动速度
        if "speed" in data:
            current_speed = data["speed"]
            # 添加随机变化
            variation = random.uniform(-0.5, 0.5)
            data["speed"] = max(0.1, current_speed + variation)
        
        return data
    
    def _optimize_reading_behavior(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """优化阅读行为"""
        # 调整阅读时间
        if "reading_time" in data:
            current_time = data["reading_time"]
            # 添加随机变化
            variation = random.uniform(-0.2, 0.2)
            data["reading_time"] = max(0.1, current_time + variation)
        
        return data
    
    def get_behavior_report(self) -> Dict[str, Any]:
        """获取行为分析报告"""
        if not self.behavior_history:
            return {"message": "暂无行为数据"}
        
        # 计算统计信息
        pattern_scores = [b["pattern_score"] for b in self.behavior_history]
        naturalness_scores = [b["naturalness_score"] for b in self.behavior_history]
        consistency_scores = [b["consistency_score"] for b in self.behavior_history]
        
        report = {
            "total_actions": len(self.behavior_history),
            "average_pattern_score": np.mean(pattern_scores),
            "average_naturalness_score": np.mean(naturalness_scores),
            "average_consistency_score": np.mean(consistency_scores),
            "behavior_trends": self._analyze_trends(),
            "recommendations": self._generate_global_recommendations()
        }
        
        return report
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """分析行为趋势"""
        if len(self.behavior_history) < 2:
            return {}
        
        # 分析最近的行为趋势
        recent_actions = self.behavior_history[-10:]
        
        pattern_trend = np.polyfit(range(len(recent_actions)), 
                                  [a["pattern_score"] for a in recent_actions], 1)[0]
        naturalness_trend = np.polyfit(range(len(recent_actions)), 
                                     [a["naturalness_score"] for a in recent_actions], 1)[0]
        
        return {
            "pattern_trend": "improving" if pattern_trend > 0 else "declining",
            "naturalness_trend": "improving" if naturalness_trend > 0 else "declining"
        }
    
    def _generate_global_recommendations(self) -> List[str]:
        """生成全局建议"""
        recommendations = []
        
        if len(self.behavior_history) < 5:
            recommendations.append("行为数据不足，建议收集更多数据")
            return recommendations
        
        # 基于平均分数生成建议
        avg_pattern = np.mean([b["pattern_score"] for b in self.behavior_history])
        avg_naturalness = np.mean([b["naturalness_score"] for b in self.behavior_history])
        avg_consistency = np.mean([b["consistency_score"] for b in self.behavior_history])
        
        if avg_pattern < 0.7:
            recommendations.append("整体行为模式需要优化，建议增加随机性")
        
        if avg_naturalness < 0.8:
            recommendations.append("行为自然度较低，建议调整时间间隔和速度")
        
        if avg_consistency < 0.8:
            recommendations.append("行为一致性较差，建议保持稳定的行为特征")
        
        if not recommendations:
            recommendations.append("行为模式良好，继续保持")
        
        return recommendations


class BehaviorConsistencyChecker:
    """行为一致性检查器"""
    
    def __init__(self):
        self.behavior_signatures = {}
        self.consistency_threshold = 0.8
    
    def check_consistency(self, action_type: str, action_data: Dict[str, Any]) -> float:
        """检查行为一致性"""
        # 生成行为签名
        signature = self._generate_signature(action_type, action_data)
        
        # 检查与历史行为的一致性
        if action_type in self.behavior_signatures:
            historical_signatures = self.behavior_signatures[action_type]
            consistency_score = self._calculate_consistency(signature, historical_signatures)
        else:
            consistency_score = 1.0  # 首次行为，认为一致
        
        # 更新行为签名
        if action_type not in self.behavior_signatures:
            self.behavior_signatures[action_type] = []
        self.behavior_signatures[action_type].append(signature)
        
        # 保持最近10个签名
        if len(self.behavior_signatures[action_type]) > 10:
            self.behavior_signatures[action_type] = self.behavior_signatures[action_type][-10:]
        
        return consistency_score
    
    def _generate_signature(self, action_type: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成行为签名"""
        signature = {
            "type": action_type,
            "timestamp": time.time()
        }
        
        # 根据行为类型提取特征
        if action_type == "typing":
            signature.update({
                "speed": action_data.get("typing_speed", 0.3),
                "intervals": action_data.get("intervals", []),
                "text_length": action_data.get("text_length", 0)
            })
        elif action_type == "clicking":
            signature.update({
                "interval": action_data.get("interval", 1.0),
                "position": action_data.get("position", (0, 0)),
                "button": action_data.get("button", "left")
            })
        elif action_type == "scrolling":
            signature.update({
                "speed": action_data.get("speed", 1.0),
                "direction": action_data.get("direction", "down"),
                "distance": action_data.get("distance", 0)
            })
        
        return signature
    
    def _calculate_consistency(self, current_signature: Dict[str, Any], 
                             historical_signatures: List[Dict[str, Any]]) -> float:
        """计算一致性分数"""
        if not historical_signatures:
            return 1.0
        
        consistency_scores = []
        
        for hist_signature in historical_signatures:
            score = 0.0
            
            # 比较数值特征
            for key in current_signature:
                if key in hist_signature and isinstance(current_signature[key], (int, float)):
                    current_val = current_signature[key]
                    hist_val = hist_signature[key]
                    
                    if hist_val != 0:
                        similarity = 1 - abs(current_val - hist_val) / max(abs(hist_val), 1)
                        score += similarity
            
            # 比较位置特征
            if "position" in current_signature and "position" in hist_signature:
                current_pos = current_signature["position"]
                hist_pos = hist_signature["position"]
                
                distance = np.sqrt(
                    (current_pos[0] - hist_pos[0])**2 + 
                    (current_pos[1] - hist_pos[1])**2
                )
                
                # 距离越近，相似度越高
                position_similarity = max(0, 1 - distance / 100)
                score += position_similarity
            
            consistency_scores.append(score / len(current_signature))
        
        return np.mean(consistency_scores) if consistency_scores else 1.0 