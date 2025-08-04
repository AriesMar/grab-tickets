#!/usr/bin/env python3
"""
深度学习行为模拟模块
使用深度学习和强化学习技术生成更自然的行为模式
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any
import random
import time
import json
from dataclasses import dataclass
from enum import Enum


class BehaviorType(Enum):
    """行为类型枚举"""
    MOUSE_MOVEMENT = "mouse_movement"
    KEYBOARD_INPUT = "keyboard_input"
    SCROLL_BEHAVIOR = "scroll_behavior"
    CLICK_PATTERN = "click_pattern"
    NAVIGATION_PATTERN = "navigation_pattern"


@dataclass
class BehaviorContext:
    """行为上下文"""
    user_type: str  # 用户类型
    time_of_day: float  # 一天中的时间 (0-1)
    session_duration: float  # 会话持续时间
    page_type: str  # 页面类型
    device_type: str  # 设备类型
    network_speed: float  # 网络速度


class BehaviorNeuralNetwork(nn.Module):
    """行为神经网络"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(BehaviorNeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.relu(self.fc1(lstm_out[:, -1, :]))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GANBehaviorGenerator:
    """GAN行为生成器"""
    
    def __init__(self):
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.criterion = nn.BCELoss()
    
    def _build_generator(self):
        """构建生成器"""
        return nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.Tanh()
        )
    
    def _build_discriminator(self):
        """构建判别器"""
        return nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def generate_behavior(self, noise: torch.Tensor) -> torch.Tensor:
        """生成行为数据"""
        return self.generator(noise)
    
    def train_step(self, real_data: torch.Tensor):
        """训练步骤"""
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # 训练判别器
        self.d_optimizer.zero_grad()
        real_output = self.discriminator(real_data)
        d_real_loss = self.criterion(real_output, real_labels)
        
        noise = torch.randn(batch_size, 100)
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data.detach())
        d_fake_loss = self.criterion(fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # 训练生成器
        self.g_optimizer.zero_grad()
        fake_output = self.discriminator(fake_data)
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()


class ReinforcementLearningOptimizer:
    """强化学习优化器"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = []
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32
    
    def _build_q_network(self):
        """构建Q网络"""
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def remember(self, state, action, reward, next_state, done):
        """记住经验"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)
    
    def act(self, state):
        """选择动作"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return np.argmax(q_values.detach().numpy())
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DeepLearningBehaviorSimulation:
    """深度学习行为模拟"""
    
    def __init__(self):
        self.behavior_models = {}
        self.gan_generator = GANBehaviorGenerator()
        self.rl_optimizer = ReinforcementLearningOptimizer(10, 5)
        self.behavior_patterns = self._load_behavior_patterns()
        self.training_data = []
        
    def _load_behavior_patterns(self) -> Dict:
        """加载行为模式"""
        return {
            BehaviorType.MOUSE_MOVEMENT: {
                "speed_patterns": [0.5, 1.0, 1.5, 2.0],
                "acceleration_patterns": [0.1, 0.2, 0.3, 0.4],
                "pause_patterns": [0.1, 0.2, 0.5, 1.0]
            },
            BehaviorType.KEYBOARD_INPUT: {
                "typing_speed": [50, 100, 150, 200],  # 字符/分钟
                "pause_patterns": [0.1, 0.2, 0.5, 1.0],
                "error_patterns": [0.01, 0.02, 0.05]
            },
            BehaviorType.SCROLL_BEHAVIOR: {
                "scroll_speed": [100, 200, 300, 400],  # 像素/秒
                "scroll_patterns": ["smooth", "jerky", "natural"],
                "pause_duration": [0.5, 1.0, 2.0, 5.0]
            }
        }
    
    def generate_natural_behavior(self, behavior_type: BehaviorType, 
                                context: BehaviorContext) -> Dict[str, Any]:
        """生成自然行为模式"""
        
        if behavior_type == BehaviorType.MOUSE_MOVEMENT:
            return self._generate_mouse_movement(context)
        elif behavior_type == BehaviorType.KEYBOARD_INPUT:
            return self._generate_keyboard_input(context)
        elif behavior_type == BehaviorType.SCROLL_BEHAVIOR:
            return self._generate_scroll_behavior(context)
        elif behavior_type == BehaviorType.CLICK_PATTERN:
            return self._generate_click_pattern(context)
        elif behavior_type == BehaviorType.NAVIGATION_PATTERN:
            return self._generate_navigation_pattern(context)
        else:
            raise ValueError(f"不支持的行为类型: {behavior_type}")
    
    def _generate_mouse_movement(self, context: BehaviorContext) -> Dict[str, Any]:
        """生成鼠标移动行为"""
        # 基于时间、用户类型等因素调整行为
        time_factor = context.time_of_day
        user_factor = self._get_user_factor(context.user_type)
        
        # 生成自然的鼠标轨迹
        trajectory = self._generate_natural_trajectory(
            speed_factor=user_factor * (0.8 + 0.4 * time_factor),
            acceleration_factor=0.2 + 0.3 * time_factor,
            pause_factor=0.1 + 0.2 * (1 - time_factor)
        )
        
        return {
            "type": "mouse_movement",
            "trajectory": trajectory,
            "speed": self._get_natural_speed(context),
            "acceleration": self._get_natural_acceleration(context),
            "pauses": self._get_natural_pauses(context)
        }
    
    def _generate_keyboard_input(self, context: BehaviorContext) -> Dict[str, Any]:
        """生成键盘输入行为"""
        typing_speed = self._get_typing_speed(context)
        pause_patterns = self._get_pause_patterns(context)
        error_rate = self._get_error_rate(context)
        
        return {
            "type": "keyboard_input",
            "typing_speed": typing_speed,
            "pause_patterns": pause_patterns,
            "error_rate": error_rate,
            "correction_patterns": self._get_correction_patterns(error_rate)
        }
    
    def _generate_scroll_behavior(self, context: BehaviorContext) -> Dict[str, Any]:
        """生成滚动行为"""
        scroll_speed = self._get_scroll_speed(context)
        scroll_pattern = self._get_scroll_pattern(context)
        
        return {
            "type": "scroll_behavior",
            "speed": scroll_speed,
            "pattern": scroll_pattern,
            "direction": self._get_scroll_direction(context),
            "pause_duration": self._get_scroll_pause_duration(context)
        }
    
    def _generate_click_pattern(self, context: BehaviorContext) -> Dict[str, Any]:
        """生成点击模式"""
        return {
            "type": "click_pattern",
            "click_speed": self._get_click_speed(context),
            "double_click_timing": self._get_double_click_timing(context),
            "click_accuracy": self._get_click_accuracy(context),
            "hover_duration": self._get_hover_duration(context)
        }
    
    def _generate_navigation_pattern(self, context: BehaviorContext) -> Dict[str, Any]:
        """生成导航模式"""
        return {
            "type": "navigation_pattern",
            "page_load_time": self._get_page_load_time(context),
            "navigation_speed": self._get_navigation_speed(context),
            "back_forward_usage": self._get_back_forward_usage(context),
            "bookmark_usage": self._get_bookmark_usage(context)
        }
    
    def _generate_natural_trajectory(self, speed_factor: float, 
                                   acceleration_factor: float, 
                                   pause_factor: float) -> List[Tuple[float, float]]:
        """生成自然轨迹"""
        trajectory = []
        x, y = 0, 0
        
        for i in range(50):
            # 添加自然的加速和减速
            speed = speed_factor * (1 + 0.2 * np.sin(i * 0.1))
            
            # 添加自然的曲线运动
            angle = i * 0.1 + 0.1 * np.sin(i * 0.05)
            dx = speed * np.cos(angle)
            dy = speed * np.sin(angle)
            
            x += dx
            y += dy
            
            trajectory.append((x, y))
            
            # 添加自然的暂停
            if random.random() < pause_factor:
                time.sleep(random.uniform(0.1, 0.3))
        
        return trajectory
    
    def _get_user_factor(self, user_type: str) -> float:
        """获取用户因子"""
        user_factors = {
            "expert": 1.2,
            "intermediate": 1.0,
            "beginner": 0.8,
            "elderly": 0.6
        }
        return user_factors.get(user_type, 1.0)
    
    def _get_natural_speed(self, context: BehaviorContext) -> float:
        """获取自然速度"""
        base_speed = 100  # 像素/秒
        time_factor = 0.8 + 0.4 * context.time_of_day
        user_factor = self._get_user_factor(context.user_type)
        return base_speed * time_factor * user_factor
    
    def _get_natural_acceleration(self, context: BehaviorContext) -> float:
        """获取自然加速度"""
        return 50 + 30 * context.time_of_day
    
    def _get_natural_pauses(self, context: BehaviorContext) -> List[float]:
        """获取自然暂停"""
        return [random.uniform(0.1, 0.5) for _ in range(3)]
    
    def _get_typing_speed(self, context: BehaviorContext) -> float:
        """获取打字速度"""
        base_speed = 100  # 字符/分钟
        time_factor = 0.8 + 0.4 * context.time_of_day
        user_factor = self._get_user_factor(context.user_type)
        return base_speed * time_factor * user_factor
    
    def _get_pause_patterns(self, context: BehaviorContext) -> List[float]:
        """获取暂停模式"""
        return [random.uniform(0.1, 2.0) for _ in range(5)]
    
    def _get_error_rate(self, context: BehaviorContext) -> float:
        """获取错误率"""
        base_error_rate = 0.02
        time_factor = 1.0 + 0.5 * (1 - context.time_of_day)  # 疲劳时错误率更高
        user_factor = 1.0 / self._get_user_factor(context.user_type)  # 新手错误率更高
        return base_error_rate * time_factor * user_factor
    
    def _get_correction_patterns(self, error_rate: float) -> Dict[str, Any]:
        """获取纠正模式"""
        return {
            "backspace_frequency": error_rate * 10,
            "correction_delay": random.uniform(0.5, 2.0),
            "retype_speed": random.uniform(0.8, 1.2)
        }
    
    def _get_scroll_speed(self, context: BehaviorContext) -> float:
        """获取滚动速度"""
        base_speed = 200  # 像素/秒
        time_factor = 0.8 + 0.4 * context.time_of_day
        user_factor = self._get_user_factor(context.user_type)
        return base_speed * time_factor * user_factor
    
    def _get_scroll_pattern(self, context: BehaviorContext) -> str:
        """获取滚动模式"""
        patterns = ["smooth", "jerky", "natural"]
        weights = [0.3, 0.2, 0.5]
        return random.choices(patterns, weights=weights)[0]
    
    def _get_scroll_direction(self, context: BehaviorContext) -> str:
        """获取滚动方向"""
        return random.choice(["up", "down", "left", "right"])
    
    def _get_scroll_pause_duration(self, context: BehaviorContext) -> float:
        """获取滚动暂停持续时间"""
        return random.uniform(0.5, 3.0)
    
    def _get_click_speed(self, context: BehaviorContext) -> float:
        """获取点击速度"""
        base_speed = 0.1  # 秒
        time_factor = 0.8 + 0.4 * context.time_of_day
        user_factor = self._get_user_factor(context.user_type)
        return base_speed / (time_factor * user_factor)
    
    def _get_double_click_timing(self, context: BehaviorContext) -> float:
        """获取双击时间间隔"""
        return random.uniform(0.2, 0.5)
    
    def _get_click_accuracy(self, context: BehaviorContext) -> float:
        """获取点击精度"""
        base_accuracy = 0.95
        time_factor = 1.0 - 0.1 * (1 - context.time_of_day)
        user_factor = self._get_user_factor(context.user_type)
        return min(0.99, base_accuracy * time_factor * user_factor)
    
    def _get_hover_duration(self, context: BehaviorContext) -> float:
        """获取悬停持续时间"""
        return random.uniform(0.5, 3.0)
    
    def _get_page_load_time(self, context: BehaviorContext) -> float:
        """获取页面加载时间"""
        base_time = 2.0  # 秒
        network_factor = 1.0 / context.network_speed
        return base_time * network_factor
    
    def _get_navigation_speed(self, context: BehaviorContext) -> float:
        """获取导航速度"""
        base_speed = 1.0  # 页面/分钟
        time_factor = 0.8 + 0.4 * context.time_of_day
        user_factor = self._get_user_factor(context.user_type)
        return base_speed * time_factor * user_factor
    
    def _get_back_forward_usage(self, context: BehaviorContext) -> float:
        """获取后退前进使用频率"""
        return random.uniform(0.1, 0.3)
    
    def _get_bookmark_usage(self, context: BehaviorContext) -> float:
        """获取书签使用频率"""
        return random.uniform(0.05, 0.15)
    
    def optimize_with_rl(self, feedback: Dict[str, Any]):
        """使用强化学习优化行为"""
        state = self._extract_state_from_feedback(feedback)
        action = self.rl_optimizer.act(state)
        reward = self._calculate_reward(feedback)
        next_state = self._extract_next_state(feedback)
        done = feedback.get("session_ended", False)
        
        self.rl_optimizer.remember(state, action, reward, next_state, done)
        self.rl_optimizer.replay()
        
        return action
    
    def _extract_state_from_feedback(self, feedback: Dict[str, Any]) -> List[float]:
        """从反馈中提取状态"""
        return [
            feedback.get("detection_score", 0.0),
            feedback.get("success_rate", 0.0),
            feedback.get("response_time", 0.0),
            feedback.get("error_rate", 0.0),
            feedback.get("naturalness_score", 0.0),
            feedback.get("session_duration", 0.0),
            feedback.get("page_count", 0.0),
            feedback.get("interaction_count", 0.0),
            feedback.get("time_of_day", 0.0),
            feedback.get("user_type_factor", 1.0)
        ]
    
    def _calculate_reward(self, feedback: Dict[str, Any]) -> float:
        """计算奖励"""
        detection_score = feedback.get("detection_score", 0.0)
        success_rate = feedback.get("success_rate", 0.0)
        naturalness_score = feedback.get("naturalness_score", 0.0)
        
        # 奖励函数：低检测分数 + 高成功率 + 高自然度
        reward = (1.0 - detection_score) + success_rate + naturalness_score
        return reward / 3.0
    
    def _extract_next_state(self, feedback: Dict[str, Any]) -> List[float]:
        """提取下一个状态"""
        return self._extract_state_from_feedback(feedback)
    
    def generate_with_gan(self, seed_data: torch.Tensor) -> torch.Tensor:
        """使用GAN生成行为数据"""
        return self.gan_generator.generate_behavior(seed_data)
    
    def train_gan(self, real_behavior_data: torch.Tensor):
        """训练GAN"""
        return self.gan_generator.train_step(real_behavior_data)
    
    def save_models(self, filepath: str):
        """保存模型"""
        torch.save({
            'behavior_models': self.behavior_models,
            'gan_generator': self.gan_generator.state_dict(),
            'rl_optimizer': self.rl_optimizer.q_network.state_dict()
        }, filepath)
    
    def load_models(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.behavior_models = checkpoint['behavior_models']
        self.gan_generator.load_state_dict(checkpoint['gan_generator'])
        self.rl_optimizer.q_network.load_state_dict(checkpoint['rl_optimizer'])


# 使用示例
if __name__ == "__main__":
    # 创建深度学习行为模拟器
    dl_simulator = DeepLearningBehaviorSimulation()
    
    # 创建行为上下文
    context = BehaviorContext(
        user_type="intermediate",
        time_of_day=0.5,  # 中午
        session_duration=1800,  # 30分钟
        page_type="ticket_booking",
        device_type="desktop",
        network_speed=10.0  # 10Mbps
    )
    
    # 生成自然行为
    mouse_behavior = dl_simulator.generate_natural_behavior(
        BehaviorType.MOUSE_MOVEMENT, context
    )
    print("鼠标行为:", json.dumps(mouse_behavior, indent=2, ensure_ascii=False))
    
    keyboard_behavior = dl_simulator.generate_natural_behavior(
        BehaviorType.KEYBOARD_INPUT, context
    )
    print("键盘行为:", json.dumps(keyboard_behavior, indent=2, ensure_ascii=False)) 