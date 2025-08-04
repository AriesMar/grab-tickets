#!/usr/bin/env python3
"""
生物特征模拟模块
模拟真实用户的生物特征，包括鼠标轨迹、键盘输入、触摸屏操作等
"""

import numpy as np
import time
import random
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json


class BiometricType(Enum):
    """生物特征类型"""
    MOUSE_TRAJECTORY = "mouse_trajectory"
    KEYBOARD_INPUT = "keyboard_input"
    TOUCH_GESTURE = "touch_gesture"
    VOICE_PATTERN = "voice_pattern"
    FACE_MOVEMENT = "face_movement"


@dataclass
class BiometricContext:
    """生物特征上下文"""
    user_age: int  # 用户年龄
    user_gender: str  # 用户性别
    device_type: str  # 设备类型
    screen_size: Tuple[int, int]  # 屏幕尺寸
    input_method: str  # 输入方式
    time_of_day: float  # 一天中的时间 (0-1)


class MouseTrajectorySimulator:
    """鼠标轨迹模拟器"""
    
    def __init__(self):
        self.bezier_curves = []
        self.human_like_patterns = self._load_human_patterns()
    
    def _load_human_patterns(self) -> Dict[str, Any]:
        """加载人类鼠标移动模式"""
        return {
            "speed_variations": [0.8, 1.0, 1.2, 1.5, 2.0],
            "acceleration_patterns": [0.1, 0.2, 0.3, 0.4, 0.5],
            "pause_probabilities": [0.05, 0.1, 0.15, 0.2],
            "curve_smoothness": [0.3, 0.5, 0.7, 0.9],
            "target_overshoot": [0.1, 0.2, 0.3, 0.4]
        }
    
    def simulate(self, start_pos: Tuple[float, float], 
                end_pos: Tuple[float, float], 
                context: BiometricContext) -> List[Tuple[float, float]]:
        """模拟鼠标移动轨迹"""
        
        # 计算距离和方向
        distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        
        # 根据用户年龄调整速度
        age_factor = self._get_age_factor(context.user_age)
        speed_factor = self._get_speed_factor(context)
        
        # 生成控制点
        control_points = self._generate_control_points(start_pos, end_pos, distance, angle, context)
        
        # 生成贝塞尔曲线轨迹
        trajectory = self._generate_bezier_curve(control_points, context)
        
        # 添加人类特征
        trajectory = self._add_human_characteristics(trajectory, context)
        
        return trajectory
    
    def _get_age_factor(self, age: int) -> float:
        """根据年龄获取速度因子"""
        if age < 20:
            return 1.3  # 年轻人更快
        elif age < 40:
            return 1.0  # 中年人正常
        elif age < 60:
            return 0.8  # 中年人稍慢
        else:
            return 0.6  # 老年人更慢
    
    def _get_speed_factor(self, context: BiometricContext) -> float:
        """获取速度因子"""
        time_factor = 0.8 + 0.4 * context.time_of_day  # 时间影响
        device_factor = self._get_device_factor(context.device_type)
        return time_factor * device_factor
    
    def _get_device_factor(self, device_type: str) -> float:
        """获取设备因子"""
        device_factors = {
            "desktop": 1.0,
            "laptop": 0.9,
            "tablet": 0.7,
            "mobile": 0.6
        }
        return device_factors.get(device_type, 1.0)
    
    def _generate_control_points(self, start_pos: Tuple[float, float], 
                               end_pos: Tuple[float, float], 
                               distance: float, angle: float, 
                               context: BiometricContext) -> List[Tuple[float, float]]:
        """生成贝塞尔曲线控制点"""
        
        # 基础控制点
        control_points = [start_pos]
        
        # 添加中间控制点
        num_control_points = max(2, int(distance / 100))  # 根据距离决定控制点数量
        
        for i in range(1, num_control_points + 1):
            # 计算中间位置
            t = i / (num_control_points + 1)
            base_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            base_y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            
            # 添加随机偏移
            offset_distance = random.uniform(10, 50) * (1 - t)  # 越接近目标偏移越小
            offset_angle = angle + random.uniform(-0.5, 0.5)  # 随机角度偏移
            
            offset_x = offset_distance * math.cos(offset_angle)
            offset_y = offset_distance * math.sin(offset_angle)
            
            control_point = (base_x + offset_x, base_y + offset_y)
            control_points.append(control_point)
        
        control_points.append(end_pos)
        return control_points
    
    def _generate_bezier_curve(self, control_points: List[Tuple[float, float]], 
                              context: BiometricContext) -> List[Tuple[float, float]]:
        """生成贝塞尔曲线"""
        
        trajectory = []
        num_points = max(50, len(control_points) * 10)  # 生成足够的点
        
        for i in range(num_points):
            t = i / (num_points - 1)
            point = self._bezier_point(control_points, t)
            trajectory.append(point)
        
        return trajectory
    
    def _bezier_point(self, control_points: List[Tuple[float, float]], t: float) -> Tuple[float, float]:
        """计算贝塞尔曲线上的点"""
        n = len(control_points) - 1
        x, y = 0, 0
        
        for i, point in enumerate(control_points):
            coefficient = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            x += coefficient * point[0]
            y += coefficient * point[1]
        
        return (x, y)
    
    def _add_human_characteristics(self, trajectory: List[Tuple[float, float]], 
                                 context: BiometricContext) -> List[Tuple[float, float]]:
        """添加人类特征"""
        
        # 添加速度变化
        trajectory = self._add_speed_variations(trajectory, context)
        
        # 添加暂停
        trajectory = self._add_pauses(trajectory, context)
        
        # 添加微抖动
        trajectory = self._add_micro_tremors(trajectory, context)
        
        # 添加目标过冲
        trajectory = self._add_target_overshoot(trajectory, context)
        
        return trajectory
    
    def _add_speed_variations(self, trajectory: List[Tuple[float, float]], 
                            context: BiometricContext) -> List[Tuple[float, float]]:
        """添加速度变化"""
        
        # 根据年龄调整速度变化
        age_factor = self._get_age_factor(context.user_age)
        speed_variations = [v * age_factor for v in self.human_like_patterns["speed_variations"]]
        
        # 在轨迹中随机应用速度变化
        for i in range(1, len(trajectory)):
            if random.random() < 0.3:  # 30%概率改变速度
                speed_factor = random.choice(speed_variations)
                # 这里可以调整点的间距来模拟速度变化
                # 简化实现：跳过一些点来模拟速度变化
                if speed_factor > 1.0 and i + 1 < len(trajectory):
                    trajectory.pop(i)
        
        return trajectory
    
    def _add_pauses(self, trajectory: List[Tuple[float, float]], 
                   context: BiometricContext) -> List[Tuple[float, float]]:
        """添加暂停"""
        
        pause_prob = random.choice(self.human_like_patterns["pause_probabilities"])
        
        # 在轨迹中随机添加暂停点
        for i in range(1, len(trajectory) - 1):
            if random.random() < pause_prob:
                # 在当前位置添加多个相同的点来模拟暂停
                pause_duration = random.randint(2, 5)
                for _ in range(pause_duration):
                    trajectory.insert(i, trajectory[i])
        
        return trajectory
    
    def _add_micro_tremors(self, trajectory: List[Tuple[float, float]], 
                          context: BiometricContext) -> List[Tuple[float, float]]:
        """添加微抖动"""
        
        # 根据年龄调整抖动幅度
        age_factor = self._get_age_factor(context.user_age)
        tremor_amplitude = 0.5 + (1.0 - age_factor) * 2.0  # 年龄越大抖动越明显
        
        for i in range(len(trajectory)):
            if random.random() < 0.1:  # 10%概率添加抖动
                dx = random.uniform(-tremor_amplitude, tremor_amplitude)
                dy = random.uniform(-tremor_amplitude, tremor_amplitude)
                trajectory[i] = (trajectory[i][0] + dx, trajectory[i][1] + dy)
        
        return trajectory
    
    def _add_target_overshoot(self, trajectory: List[Tuple[float, float]], 
                            context: BiometricContext) -> List[Tuple[float, float]]:
        """添加目标过冲"""
        
        if len(trajectory) < 10:
            return trajectory
        
        # 在接近目标时添加过冲
        overshoot_prob = random.choice(self.human_like_patterns["target_overshoot"])
        
        if random.random() < overshoot_prob:
            # 在轨迹末尾添加过冲点
            end_point = trajectory[-1]
            overshoot_distance = random.uniform(5, 15)
            overshoot_angle = random.uniform(0, 2 * math.pi)
            
            overshoot_x = end_point[0] + overshoot_distance * math.cos(overshoot_angle)
            overshoot_y = end_point[1] + overshoot_distance * math.sin(overshoot_angle)
            
            # 添加过冲点
            trajectory.append((overshoot_x, overshoot_y))
            # 回到目标点
            trajectory.append(end_point)
        
        return trajectory


class KeyboardInputSimulator:
    """键盘输入模拟器"""
    
    def __init__(self):
        self.typing_patterns = self._load_typing_patterns()
        self.error_patterns = self._load_error_patterns()
    
    def _load_typing_patterns(self) -> Dict[str, Any]:
        """加载打字模式"""
        return {
            "typing_speeds": {
                "expert": [200, 250, 300],  # 字符/分钟
                "intermediate": [100, 150, 200],
                "beginner": [50, 80, 120],
                "elderly": [30, 50, 80]
            },
            "pause_patterns": {
                "word_pause": [0.1, 0.3, 0.5],  # 词间暂停
                "sentence_pause": [0.5, 1.0, 1.5],  # 句间暂停
                "paragraph_pause": [1.0, 2.0, 3.0]  # 段落暂停
            },
            "key_press_duration": [0.05, 0.1, 0.15, 0.2]  # 按键持续时间
        }
    
    def _load_error_patterns(self) -> Dict[str, Any]:
        """加载错误模式"""
        return {
            "error_rates": {
                "expert": 0.005,  # 0.5%
                "intermediate": 0.02,  # 2%
                "beginner": 0.05,  # 5%
                "elderly": 0.08  # 8%
            },
            "common_errors": {
                "adjacent_keys": ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
                "shift_errors": ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
                "caps_lock_errors": ["z", "x", "c", "v", "b", "n", "m"]
            }
        }
    
    def simulate(self, text: str, context: BiometricContext) -> Dict[str, Any]:
        """模拟键盘输入"""
        
        # 确定用户类型
        user_type = self._determine_user_type(context)
        
        # 生成打字速度
        typing_speed = self._generate_typing_speed(user_type, context)
        
        # 生成按键序列
        key_events = self._generate_key_events(text, user_type, context)
        
        # 添加错误
        key_events = self._add_typing_errors(key_events, user_type, context)
        
        # 添加暂停
        key_events = self._add_typing_pauses(key_events, context)
        
        return {
            "type": "keyboard_input",
            "text": text,
            "typing_speed": typing_speed,
            "key_events": key_events,
            "total_duration": self._calculate_total_duration(key_events),
            "error_count": self._count_errors(key_events)
        }
    
    def _determine_user_type(self, context: BiometricContext) -> str:
        """确定用户类型"""
        if context.user_age < 25:
            return "expert"
        elif context.user_age < 45:
            return "intermediate"
        elif context.user_age < 65:
            return "beginner"
        else:
            return "elderly"
    
    def _generate_typing_speed(self, user_type: str, context: BiometricContext) -> float:
        """生成打字速度"""
        base_speeds = self.typing_patterns["typing_speeds"][user_type]
        base_speed = random.choice(base_speeds)
        
        # 根据时间调整速度
        time_factor = 0.8 + 0.4 * context.time_of_day
        # 根据性别调整速度（统计数据显示差异）
        gender_factor = 1.1 if context.user_gender == "male" else 1.0
        
        return base_speed * time_factor * gender_factor
    
    def _generate_key_events(self, text: str, user_type: str, 
                           context: BiometricContext) -> List[Dict[str, Any]]:
        """生成按键事件"""
        
        key_events = []
        current_time = 0.0
        
        for char in text:
            # 按键按下
            press_time = current_time
            key_events.append({
                "type": "key_press",
                "key": char,
                "timestamp": press_time,
                "duration": random.choice(self.typing_patterns["key_press_duration"])
            })
            
            # 计算到下一个字符的时间间隔
            interval = self._calculate_key_interval(char, user_type, context)
            current_time += interval
            
            # 按键释放
            release_time = press_time + key_events[-1]["duration"]
            key_events.append({
                "type": "key_release",
                "key": char,
                "timestamp": release_time
            })
        
        return key_events
    
    def _calculate_key_interval(self, char: str, user_type: str, 
                              context: BiometricContext) -> float:
        """计算按键间隔"""
        
        # 基础间隔（基于打字速度）
        typing_speed = self._generate_typing_speed(user_type, context)
        base_interval = 60.0 / typing_speed  # 秒/字符
        
        # 根据字符类型调整
        if char.isupper():
            base_interval *= 1.2  # 大写字母需要shift
        elif char in ".,!?;:":
            base_interval *= 1.1  # 标点符号稍慢
        elif char == " ":
            base_interval *= 0.8  # 空格稍快
        
        # 添加随机变化
        variation = random.uniform(0.8, 1.2)
        
        return base_interval * variation
    
    def _add_typing_errors(self, key_events: List[Dict[str, Any]], 
                          user_type: str, context: BiometricContext) -> List[Dict[str, Any]]:
        """添加打字错误"""
        
        error_rate = self.error_patterns["error_rates"][user_type]
        
        # 根据时间调整错误率（疲劳时错误更多）
        time_factor = 1.0 + 0.5 * (1 - context.time_of_day)
        adjusted_error_rate = error_rate * time_factor
        
        new_events = []
        
        for event in key_events:
            if event["type"] == "key_press" and random.random() < adjusted_error_rate:
                # 生成错误
                error_events = self._generate_typing_error(event, context)
                new_events.extend(error_events)
            else:
                new_events.append(event)
        
        return new_events
    
    def _generate_typing_error(self, original_event: Dict[str, Any], 
                             context: BiometricContext) -> List[Dict[str, Any]]:
        """生成打字错误"""
        
        original_key = original_event["key"]
        timestamp = original_event["timestamp"]
        
        # 选择错误类型
        error_type = random.choice(["adjacent", "shift", "caps_lock", "double_press"])
        
        if error_type == "adjacent":
            # 相邻键错误
            adjacent_keys = self._get_adjacent_keys(original_key)
            wrong_key = random.choice(adjacent_keys) if adjacent_keys else original_key
            
        elif error_type == "shift":
            # Shift错误
            wrong_key = original_key.upper() if original_key.islower() else original_key.lower()
            
        elif error_type == "caps_lock":
            # Caps Lock错误
            wrong_key = original_key.upper() if original_key.islower() else original_key.lower()
            
        else:  # double_press
            # 重复按键
            return [
                original_event,
                {
                    "type": "key_press",
                    "key": original_key,
                    "timestamp": timestamp + 0.1,
                    "duration": 0.05
                },
                {
                    "type": "key_release",
                    "key": original_key,
                    "timestamp": timestamp + 0.15
                }
            ]
        
        # 生成错误事件
        error_events = [
            {
                "type": "key_press",
                "key": wrong_key,
                "timestamp": timestamp,
                "duration": original_event["duration"],
                "is_error": True
            },
            {
                "type": "key_release",
                "key": wrong_key,
                "timestamp": timestamp + original_event["duration"],
                "is_error": True
            },
            {
                "type": "key_press",
                "key": "backspace",
                "timestamp": timestamp + original_event["duration"] + 0.2,
                "duration": 0.1
            },
            {
                "type": "key_release",
                "key": "backspace",
                "timestamp": timestamp + original_event["duration"] + 0.3
            },
            original_event
        ]
        
        return error_events
    
    def _get_adjacent_keys(self, key: str) -> List[str]:
        """获取相邻键"""
        # 简化的QWERTY键盘布局
        adjacent_map = {
            "q": ["w", "a", "1", "2"],
            "w": ["q", "e", "s", "2", "3"],
            "e": ["w", "r", "d", "3", "4"],
            # ... 可以扩展更多
        }
        return adjacent_map.get(key.lower(), [])
    
    def _add_typing_pauses(self, key_events: List[Dict[str, Any]], 
                          context: BiometricContext) -> List[Dict[str, Any]]:
        """添加打字暂停"""
        
        new_events = []
        current_time = 0.0
        
        for event in key_events:
            # 添加事件
            new_events.append(event)
            current_time = event["timestamp"]
            
            # 随机添加暂停
            if event["type"] == "key_release":
                # 词间暂停
                if random.random() < 0.1:  # 10%概率
                    pause_duration = random.choice(self.typing_patterns["pause_patterns"]["word_pause"])
                    current_time += pause_duration
                
                # 句间暂停
                if random.random() < 0.05:  # 5%概率
                    pause_duration = random.choice(self.typing_patterns["pause_patterns"]["sentence_pause"])
                    current_time += pause_duration
        
        return new_events
    
    def _calculate_total_duration(self, key_events: List[Dict[str, Any]]) -> float:
        """计算总持续时间"""
        if not key_events:
            return 0.0
        return max(event["timestamp"] for event in key_events)
    
    def _count_errors(self, key_events: List[Dict[str, Any]]) -> int:
        """计算错误数量"""
        return sum(1 for event in key_events if event.get("is_error", False))


class TouchScreenSimulator:
    """触摸屏模拟器"""
    
    def __init__(self):
        self.gesture_patterns = self._load_gesture_patterns()
    
    def _load_gesture_patterns(self) -> Dict[str, Any]:
        """加载手势模式"""
        return {
            "tap_duration": [0.05, 0.1, 0.15, 0.2],
            "swipe_speed": [100, 200, 300, 400],  # 像素/秒
            "pinch_scale": [0.5, 0.8, 1.2, 1.5, 2.0],
            "rotation_angle": [15, 30, 45, 60, 90],
            "pressure_levels": [0.3, 0.5, 0.7, 0.9, 1.0]
        }
    
    def simulate(self, gesture_type: str, coordinates: List[Tuple[float, float]], 
                context: BiometricContext) -> Dict[str, Any]:
        """模拟触摸手势"""
        
        if gesture_type == "tap":
            return self._simulate_tap(coordinates[0], context)
        elif gesture_type == "swipe":
            return self._simulate_swipe(coordinates[0], coordinates[1], context)
        elif gesture_type == "pinch":
            return self._simulate_pinch(coordinates, context)
        elif gesture_type == "rotation":
            return self._simulate_rotation(coordinates, context)
        elif gesture_type == "long_press":
            return self._simulate_long_press(coordinates[0], context)
        else:
            raise ValueError(f"不支持的手势类型: {gesture_type}")
    
    def _simulate_tap(self, position: Tuple[float, float], 
                     context: BiometricContext) -> Dict[str, Any]:
        """模拟点击"""
        
        # 根据年龄调整点击精度
        age_factor = self._get_age_factor(context.user_age)
        accuracy_radius = 5 + (1.0 - age_factor) * 10  # 年龄越大精度越低
        
        # 添加精度误差
        error_x = random.uniform(-accuracy_radius, accuracy_radius)
        error_y = random.uniform(-accuracy_radius, accuracy_radius)
        actual_position = (position[0] + error_x, position[1] + error_y)
        
        # 点击持续时间
        duration = random.choice(self.gesture_patterns["tap_duration"])
        
        return {
            "type": "tap",
            "intended_position": position,
            "actual_position": actual_position,
            "duration": duration,
            "pressure": random.choice(self.gesture_patterns["pressure_levels"]),
            "finger_size": self._get_finger_size(context)
        }
    
    def _simulate_swipe(self, start_pos: Tuple[float, float], 
                       end_pos: Tuple[float, float], 
                       context: BiometricContext) -> Dict[str, Any]:
        """模拟滑动"""
        
        # 计算滑动距离和方向
        distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        
        # 根据年龄调整滑动速度
        age_factor = self._get_age_factor(context.user_age)
        base_speed = random.choice(self.gesture_patterns["swipe_speed"])
        speed = base_speed * age_factor
        
        # 生成滑动轨迹
        trajectory = self._generate_swipe_trajectory(start_pos, end_pos, speed, context)
        
        return {
            "type": "swipe",
            "start_position": start_pos,
            "end_position": end_pos,
            "distance": distance,
            "angle": angle,
            "speed": speed,
            "trajectory": trajectory,
            "duration": distance / speed,
            "pressure": random.choice(self.gesture_patterns["pressure_levels"])
        }
    
    def _simulate_pinch(self, coordinates: List[Tuple[float, float]], 
                       context: BiometricContext) -> Dict[str, Any]:
        """模拟捏合手势"""
        
        if len(coordinates) < 2:
            raise ValueError("捏合手势需要至少两个坐标点")
        
        # 计算捏合中心
        center_x = sum(pos[0] for pos in coordinates) / len(coordinates)
        center_y = sum(pos[1] for pos in coordinates) / len(coordinates)
        center = (center_x, center_y)
        
        # 计算缩放比例
        scale = random.choice(self.gesture_patterns["pinch_scale"])
        
        # 根据年龄调整精度
        age_factor = self._get_age_factor(context.user_age)
        accuracy = 0.9 + age_factor * 0.1
        
        return {
            "type": "pinch",
            "center": center,
            "scale": scale,
            "accuracy": accuracy,
            "finger_positions": coordinates,
            "duration": random.uniform(0.5, 2.0)
        }
    
    def _simulate_rotation(self, coordinates: List[Tuple[float, float]], 
                          context: BiometricContext) -> Dict[str, Any]:
        """模拟旋转手势"""
        
        if len(coordinates) < 2:
            raise ValueError("旋转手势需要至少两个坐标点")
        
        # 计算旋转中心
        center_x = sum(pos[0] for pos in coordinates) / len(coordinates)
        center_y = sum(pos[1] for pos in coordinates) / len(coordinates)
        center = (center_x, center_y)
        
        # 计算旋转角度
        angle = random.choice(self.gesture_patterns["rotation_angle"])
        
        # 根据年龄调整精度
        age_factor = self._get_age_factor(context.user_age)
        accuracy = 0.9 + age_factor * 0.1
        
        return {
            "type": "rotation",
            "center": center,
            "angle": angle,
            "accuracy": accuracy,
            "finger_positions": coordinates,
            "duration": random.uniform(0.5, 2.0)
        }
    
    def _simulate_long_press(self, position: Tuple[float, float], 
                            context: BiometricContext) -> Dict[str, Any]:
        """模拟长按"""
        
        # 根据年龄调整长按时间
        age_factor = self._get_age_factor(context.user_age)
        base_duration = random.uniform(0.5, 1.5)
        duration = base_duration / age_factor  # 年龄越大需要更长时间
        
        # 添加精度误差
        accuracy_radius = 3 + (1.0 - age_factor) * 7
        error_x = random.uniform(-accuracy_radius, accuracy_radius)
        error_y = random.uniform(-accuracy_radius, accuracy_radius)
        actual_position = (position[0] + error_x, position[1] + error_y)
        
        return {
            "type": "long_press",
            "intended_position": position,
            "actual_position": actual_position,
            "duration": duration,
            "pressure": random.choice(self.gesture_patterns["pressure_levels"]),
            "finger_size": self._get_finger_size(context)
        }
    
    def _generate_swipe_trajectory(self, start_pos: Tuple[float, float], 
                                 end_pos: Tuple[float, float], 
                                 speed: float, context: BiometricContext) -> List[Tuple[float, float]]:
        """生成滑动轨迹"""
        
        distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        duration = distance / speed
        num_points = max(10, int(duration * 30))  # 30fps
        
        trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            
            # 添加微抖动
            if random.random() < 0.1:
                x += random.uniform(-2, 2)
                y += random.uniform(-2, 2)
            
            trajectory.append((x, y))
        
        return trajectory
    
    def _get_age_factor(self, age: int) -> float:
        """获取年龄因子"""
        if age < 25:
            return 1.2
        elif age < 45:
            return 1.0
        elif age < 65:
            return 0.8
        else:
            return 0.6
    
    def _get_finger_size(self, context: BiometricContext) -> float:
        """获取手指大小"""
        # 根据年龄和性别估算手指大小
        base_size = 8.0  # 基础手指大小（像素）
        
        if context.user_gender == "male":
            base_size *= 1.1
        else:
            base_size *= 0.9
        
        # 年龄影响
        age_factor = self._get_age_factor(context.user_age)
        base_size *= age_factor
        
        return base_size


class VoiceRecognitionEvasion:
    """语音识别规避"""
    
    def __init__(self):
        self.voice_patterns = self._load_voice_patterns()
    
    def _load_voice_patterns(self) -> Dict[str, Any]:
        """加载语音模式"""
        return {
            "speech_rates": [120, 150, 180, 210],  # 词/分钟
            "pitch_variations": [0.8, 0.9, 1.0, 1.1, 1.2],
            "volume_levels": [0.6, 0.7, 0.8, 0.9, 1.0],
            "pause_patterns": [0.2, 0.5, 1.0, 1.5, 2.0]
        }
    
    def apply_evasion(self, audio_data: bytes, context: BiometricContext) -> bytes:
        """应用语音识别规避"""
        
        # 添加背景噪声
        audio_data = self._add_background_noise(audio_data, context)
        
        # 调整语音特征
        audio_data = self._adjust_voice_characteristics(audio_data, context)
        
        # 添加语音干扰
        audio_data = self._add_voice_interference(audio_data, context)
        
        return audio_data
    
    def _add_background_noise(self, audio_data: bytes, context: BiometricContext) -> bytes:
        """添加背景噪声"""
        # 模拟实现：添加随机噪声
        noise_level = random.uniform(0.1, 0.3)
        # 这里应该实现实际的音频处理
        return audio_data
    
    def _adjust_voice_characteristics(self, audio_data: bytes, context: BiometricContext) -> bytes:
        """调整语音特征"""
        # 根据年龄调整语音特征
        age_factor = self._get_age_factor(context.user_age)
        
        # 调整语速
        speech_rate = random.choice(self.voice_patterns["speech_rates"]) * age_factor
        
        # 调整音调
        pitch_factor = random.choice(self.voice_patterns["pitch_variations"])
        
        # 调整音量
        volume_factor = random.choice(self.voice_patterns["volume_levels"])
        
        # 这里应该实现实际的音频处理
        return audio_data
    
    def _add_voice_interference(self, audio_data: bytes, context: BiometricContext) -> bytes:
        """添加语音干扰"""
        # 添加随机干扰信号
        interference_type = random.choice(["white_noise", "frequency_shift", "time_shift"])
        
        # 这里应该实现实际的音频处理
        return audio_data
    
    def _get_age_factor(self, age: int) -> float:
        """获取年龄因子"""
        if age < 25:
            return 1.1
        elif age < 45:
            return 1.0
        elif age < 65:
            return 0.9
        else:
            return 0.8


class FaceRecognitionEvasion:
    """面部识别规避"""
    
    def __init__(self):
        self.face_patterns = self._load_face_patterns()
    
    def _load_face_patterns(self) -> Dict[str, Any]:
        """加载面部模式"""
        return {
            "expression_variations": ["neutral", "slight_smile", "frown", "surprise"],
            "head_movements": ["slight_tilt", "nod", "turn", "none"],
            "eye_movements": ["blink", "look_away", "focus", "random"],
            "lighting_variations": [0.8, 0.9, 1.0, 1.1, 1.2]
        }
    
    def apply_evasion(self, face_data: bytes, context: BiometricContext) -> bytes:
        """应用面部识别规避"""
        
        # 添加面部表情变化
        face_data = self._add_expression_variations(face_data, context)
        
        # 添加头部运动
        face_data = self._add_head_movements(face_data, context)
        
        # 添加眼部运动
        face_data = self._add_eye_movements(face_data, context)
        
        # 调整光照
        face_data = self._adjust_lighting(face_data, context)
        
        return face_data
    
    def _add_expression_variations(self, face_data: bytes, context: BiometricContext) -> bytes:
        """添加面部表情变化"""
        expression = random.choice(self.face_patterns["expression_variations"])
        
        # 根据年龄调整表情强度
        age_factor = self._get_age_factor(context.user_age)
        intensity = random.uniform(0.5, 1.0) * age_factor
        
        # 这里应该实现实际的面部处理
        return face_data
    
    def _add_head_movements(self, face_data: bytes, context: BiometricContext) -> bytes:
        """添加头部运动"""
        movement = random.choice(self.face_patterns["head_movements"])
        
        # 根据年龄调整运动幅度
        age_factor = self._get_age_factor(context.user_age)
        amplitude = random.uniform(0.1, 0.3) * age_factor
        
        # 这里应该实现实际的面部处理
        return face_data
    
    def _add_eye_movements(self, face_data: bytes, context: BiometricContext) -> bytes:
        """添加眼部运动"""
        movement = random.choice(self.face_patterns["eye_movements"])
        
        # 根据年龄调整眨眼频率
        age_factor = self._get_age_factor(context.user_age)
        blink_rate = 15 + (1.0 - age_factor) * 10  # 年龄越大眨眼越频繁
        
        # 这里应该实现实际的面部处理
        return face_data
    
    def _adjust_lighting(self, face_data: bytes, context: BiometricContext) -> bytes:
        """调整光照"""
        lighting_factor = random.choice(self.face_patterns["lighting_variations"])
        
        # 这里应该实现实际的面部处理
        return face_data
    
    def _get_age_factor(self, age: int) -> float:
        """获取年龄因子"""
        if age < 25:
            return 1.0
        elif age < 45:
            return 0.95
        elif age < 65:
            return 0.9
        else:
            return 0.85


class BiometricSimulation:
    """生物特征模拟主类"""
    
    def __init__(self):
        self.mouse_tracker = MouseTrajectorySimulator()
        self.keyboard_simulator = KeyboardInputSimulator()
        self.touch_simulator = TouchScreenSimulator()
        self.voice_evasion = VoiceRecognitionEvasion()
        self.face_evasion = FaceRecognitionEvasion()
    
    def simulate_mouse_movement(self, start_pos: Tuple[float, float], 
                              end_pos: Tuple[float, float], 
                              context: BiometricContext) -> List[Tuple[float, float]]:
        """模拟鼠标移动"""
        return self.mouse_tracker.simulate(start_pos, end_pos, context)
    
    def simulate_keyboard_input(self, text: str, context: BiometricContext) -> Dict[str, Any]:
        """模拟键盘输入"""
        return self.keyboard_simulator.simulate(text, context)
    
    def simulate_touch_gesture(self, gesture_type: str, coordinates: List[Tuple[float, float]], 
                             context: BiometricContext) -> Dict[str, Any]:
        """模拟触摸手势"""
        return self.touch_simulator.simulate(gesture_type, coordinates, context)
    
    def apply_voice_evasion(self, audio_data: bytes, context: BiometricContext) -> bytes:
        """应用语音识别规避"""
        return self.voice_evasion.apply_evasion(audio_data, context)
    
    def apply_face_evasion(self, face_data: bytes, context: BiometricContext) -> bytes:
        """应用面部识别规避"""
        return self.face_evasion.apply_evasion(face_data, context)
    
    def generate_comprehensive_behavior(self, behavior_type: BiometricType, 
                                     data: Any, context: BiometricContext) -> Dict[str, Any]:
        """生成综合行为数据"""
        
        if behavior_type == BiometricType.MOUSE_TRAJECTORY:
            start_pos, end_pos = data
            trajectory = self.simulate_mouse_movement(start_pos, end_pos, context)
            return {
                "type": "mouse_trajectory",
                "trajectory": trajectory,
                "start_position": start_pos,
                "end_position": end_pos,
                "context": {
                    "user_age": context.user_age,
                    "user_gender": context.user_gender,
                    "device_type": context.device_type
                }
            }
        
        elif behavior_type == BiometricType.KEYBOARD_INPUT:
            text = data
            keyboard_data = self.simulate_keyboard_input(text, context)
            return {
                "type": "keyboard_input",
                "data": keyboard_data,
                "context": {
                    "user_age": context.user_age,
                    "user_gender": context.user_gender,
                    "input_method": context.input_method
                }
            }
        
        elif behavior_type == BiometricType.TOUCH_GESTURE:
            gesture_type, coordinates = data
            touch_data = self.simulate_touch_gesture(gesture_type, coordinates, context)
            return {
                "type": "touch_gesture",
                "data": touch_data,
                "context": {
                    "user_age": context.user_age,
                    "device_type": context.device_type,
                    "screen_size": context.screen_size
                }
            }
        
        else:
            raise ValueError(f"不支持的生物特征类型: {behavior_type}")


# 使用示例
if __name__ == "__main__":
    # 创建生物特征模拟器
    biometric_sim = BiometricSimulation()
    
    # 创建上下文
    context = BiometricContext(
        user_age=30,
        user_gender="male",
        device_type="desktop",
        screen_size=(1920, 1080),
        input_method="keyboard",
        time_of_day=0.5
    )
    
    # 模拟鼠标移动
    mouse_trajectory = biometric_sim.simulate_mouse_movement(
        (100, 100), (500, 300), context
    )
    print(f"鼠标轨迹点数: {len(mouse_trajectory)}")
    
    # 模拟键盘输入
    keyboard_data = biometric_sim.simulate_keyboard_input("Hello World!", context)
    print(f"键盘输入持续时间: {keyboard_data['total_duration']:.2f}秒")
    
    # 模拟触摸手势
    touch_data = biometric_sim.simulate_touch_gesture(
        "swipe", [(100, 100), (300, 300)], context
    )
    print(f"触摸手势类型: {touch_data['type']}") 