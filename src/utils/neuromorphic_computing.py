#!/usr/bin/env python3
"""
神经形态计算模块 - 提供类脑计算、脉冲神经网络、神经形态优化等功能
"""
import time
import json
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
from loguru import logger


class NeuronType(Enum):
    """神经元类型"""
    LIF = "leaky_integrate_and_fire"
    IZH = "izhikevich"
    HODGKIN_HUXLEY = "hodgkin_huxley"
    ADAPTIVE = "adaptive"


class SynapseType(Enum):
    """突触类型"""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"


@dataclass
class Neuron:
    """神经元"""
    neuron_id: str
    neuron_type: NeuronType
    membrane_potential: float
    threshold: float
    refractory_period: float
    last_spike_time: float
    spike_history: List[float]
    parameters: Dict[str, float]


@dataclass
class Synapse:
    """突触"""
    synapse_id: str
    pre_neuron_id: str
    post_neuron_id: str
    synapse_type: SynapseType
    weight: float
    delay: float
    plasticity: Dict[str, float]
    last_activity: float


@dataclass
class SpikingNeuralNetwork:
    """脉冲神经网络"""
    network_id: str
    neurons: Dict[str, Neuron]
    synapses: Dict[str, Synapse]
    topology: Dict[str, List[str]]
    learning_rule: str
    simulation_time: float


class LeakyIntegrateAndFire:
    """漏积分发放神经元模型"""
    
    def __init__(self, neuron_id: str, parameters: Dict[str, float] = None):
        self.neuron_id = neuron_id
        self.logger = logger.bind(name=f"lif_neuron_{neuron_id}")
        
        # 默认参数
        self.parameters = parameters or {
            "tau_m": 20.0,  # 膜时间常数 (ms)
            "v_rest": -65.0,  # 静息电位 (mV)
            "v_threshold": -55.0,  # 阈值电位 (mV)
            "v_reset": -65.0,  # 重置电位 (mV)
            "refractory_period": 2.0,  # 不应期 (ms)
            "r_m": 10.0  # 膜电阻 (MΩ)
        }
        
        # 神经元状态
        self.membrane_potential = self.parameters["v_rest"]
        self.last_spike_time = -np.inf
        self.spike_history = []
        self.is_refractory = False
        
    def update(self, current_input: float, dt: float = 1.0) -> bool:
        """更新神经元状态"""
        current_time = time.time()
        
        # 检查不应期
        if current_time - self.last_spike_time < self.parameters["refractory_period"]:
            self.is_refractory = True
            self.membrane_potential = self.parameters["v_reset"]
            return False
        
        self.is_refractory = False
        
        # 漏积分方程
        tau_m = self.parameters["tau_m"]
        v_rest = self.parameters["v_rest"]
        
        # 微分方程: τ_m * dv/dt = -(v - v_rest) + R * I
        dv_dt = (-(self.membrane_potential - v_rest) + 
                 self.parameters["r_m"] * current_input) / tau_m
        
        # 欧拉积分
        self.membrane_potential += dv_dt * dt
        
        # 检查是否发放
        if self.membrane_potential >= self.parameters["v_threshold"]:
            self.membrane_potential = self.parameters["v_reset"]
            self.last_spike_time = current_time
            self.spike_history.append(current_time)
            return True
        
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """获取神经元状态"""
        return {
            "neuron_id": self.neuron_id,
            "membrane_potential": self.membrane_potential,
            "is_refractory": self.is_refractory,
            "last_spike_time": self.last_spike_time,
            "spike_count": len(self.spike_history),
            "firing_rate": self._calculate_firing_rate()
        }
    
    def _calculate_firing_rate(self) -> float:
        """计算发放率"""
        if not self.spike_history:
            return 0.0
        
        current_time = time.time()
        recent_spikes = [spike for spike in self.spike_history 
                        if current_time - spike < 1000.0]  # 最近1秒
        
        return len(recent_spikes) / 1.0  # Hz


class IzhikevichNeuron:
    """Izhikevich神经元模型"""
    
    def __init__(self, neuron_id: str, parameters: Dict[str, float] = None):
        self.neuron_id = neuron_id
        self.logger = logger.bind(name=f"izhikevich_neuron_{neuron_id}")
        
        # 默认参数 (RS神经元)
        self.parameters = parameters or {
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 2.0,
            "v_threshold": 30.0
        }
        
        # 神经元状态
        self.v = self.parameters["c"]  # 膜电位
        self.u = 0.0  # 恢复变量
        self.last_spike_time = -np.inf
        self.spike_history = []
        
    def update(self, current_input: float, dt: float = 1.0) -> bool:
        """更新神经元状态"""
        current_time = time.time()
        
        # Izhikevich方程
        # dv/dt = 0.04v² + 5v + 140 - u + I
        # du/dt = a(bv - u)
        
        v = self.v
        u = self.u
        
        # 微分方程
        dv_dt = 0.04 * v * v + 5 * v + 140 - u + current_input
        du_dt = self.parameters["a"] * (self.parameters["b"] * v - u)
        
        # 欧拉积分
        self.v += dv_dt * dt
        self.u += du_dt * dt
        
        # 检查是否发放
        if self.v >= self.parameters["v_threshold"]:
            self.v = self.parameters["c"]
            self.u += self.parameters["d"]
            self.last_spike_time = current_time
            self.spike_history.append(current_time)
            return True
        
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """获取神经元状态"""
        return {
            "neuron_id": self.neuron_id,
            "membrane_potential": self.v,
            "recovery_variable": self.u,
            "last_spike_time": self.last_spike_time,
            "spike_count": len(self.spike_history),
            "firing_rate": self._calculate_firing_rate()
        }
    
    def _calculate_firing_rate(self) -> float:
        """计算发放率"""
        if not self.spike_history:
            return 0.0
        
        current_time = time.time()
        recent_spikes = [spike for spike in self.spike_history 
                        if current_time - spike < 1000.0]
        
        return len(recent_spikes) / 1.0


class SpikeTimeDependentPlasticity:
    """脉冲时间依赖可塑性 (STDP)"""
    
    def __init__(self):
        self.logger = logger.bind(name="stdp")
        self.parameters = {
            "tau_plus": 20.0,  # LTP时间常数
            "tau_minus": 20.0,  # LTD时间常数
            "a_plus": 0.1,  # LTP强度
            "a_minus": 0.1,  # LTD强度
            "w_max": 1.0,  # 最大权重
            "w_min": 0.0   # 最小权重
        }
        
    def update_weight(self, synapse: Synapse, pre_spike_time: float, 
                     post_spike_time: float) -> float:
        """更新突触权重"""
        if pre_spike_time == -1 or post_spike_time == -1:
            return synapse.weight
        
        # 计算时间差
        delta_t = pre_spike_time - post_spike_time
        
        # STDP规则
        if delta_t > 0:  # 前突触先发放 (LTP)
            delta_w = self.parameters["a_plus"] * np.exp(-delta_t / self.parameters["tau_plus"])
        else:  # 后突触先发放 (LTD)
            delta_w = -self.parameters["a_minus"] * np.exp(delta_t / self.parameters["tau_minus"])
        
        # 更新权重
        new_weight = synapse.weight + delta_w
        new_weight = max(self.parameters["w_min"], 
                        min(self.parameters["w_max"], new_weight))
        
        return new_weight


class NeuromorphicNetwork:
    """神经形态网络"""
    
    def __init__(self, network_id: str):
        self.network_id = network_id
        self.logger = logger.bind(name=f"neuromorphic_network_{network_id}")
        
        self.neurons: Dict[str, Union[LeakyIntegrateAndFire, IzhikevichNeuron]] = {}
        self.synapses: Dict[str, Synapse] = {}
        self.topology: Dict[str, List[str]] = defaultdict(list)
        self.stdp = SpikeTimeDependentPlasticity()
        
        self.simulation_time = 0.0
        self.spike_history = deque(maxlen=10000)
        
    def add_neuron(self, neuron_id: str, neuron_type: NeuronType, 
                   parameters: Dict[str, float] = None) -> bool:
        """添加神经元"""
        try:
            if neuron_type == NeuronType.LIF:
                neuron = LeakyIntegrateAndFire(neuron_id, parameters)
            elif neuron_type == NeuronType.IZH:
                neuron = IzhikevichNeuron(neuron_id, parameters)
            else:
                self.logger.error(f"不支持的神经元类型: {neuron_type}")
                return False
            
            self.neurons[neuron_id] = neuron
            self.logger.info(f"神经元已添加: {neuron_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加神经元失败: {e}")
            return False
    
    def add_synapse(self, synapse_id: str, pre_neuron_id: str, post_neuron_id: str,
                    synapse_type: SynapseType, weight: float = 0.5, delay: float = 1.0) -> bool:
        """添加突触"""
        try:
            if pre_neuron_id not in self.neurons or post_neuron_id not in self.neurons:
                self.logger.error(f"神经元不存在: {pre_neuron_id} -> {post_neuron_id}")
                return False
            
            synapse = Synapse(
                synapse_id=synapse_id,
                pre_neuron_id=pre_neuron_id,
                post_neuron_id=post_neuron_id,
                synapse_type=synapse_type,
                weight=weight,
                delay=delay,
                plasticity={"stdp_enabled": True, "learning_rate": 0.01},
                last_activity=0.0
            )
            
            self.synapses[synapse_id] = synapse
            self.topology[pre_neuron_id].append(post_neuron_id)
            
            self.logger.info(f"突触已添加: {pre_neuron_id} -> {post_neuron_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加突触失败: {e}")
            return False
    
    def simulate_step(self, external_inputs: Dict[str, float] = None, dt: float = 1.0) -> Dict[str, Any]:
        """模拟一个时间步"""
        self.simulation_time += dt
        external_inputs = external_inputs or {}
        
        # 更新神经元
        spike_events = {}
        for neuron_id, neuron in self.neurons.items():
            # 获取外部输入
            external_input = external_inputs.get(neuron_id, 0.0)
            
            # 获取突触输入
            synaptic_input = self._calculate_synaptic_input(neuron_id)
            
            # 总输入
            total_input = external_input + synaptic_input
            
            # 更新神经元
            spike_occurred = neuron.update(total_input, dt)
            
            if spike_occurred:
                spike_events[neuron_id] = self.simulation_time
                self.spike_history.append({
                    "neuron_id": neuron_id,
                    "time": self.simulation_time,
                    "membrane_potential": neuron.membrane_potential if hasattr(neuron, 'membrane_potential') else neuron.v
                })
        
        # 更新突触权重 (STDP)
        self._update_synapse_weights(spike_events)
        
        return {
            "simulation_time": self.simulation_time,
            "spike_events": spike_events,
            "network_state": self._get_network_state()
        }
    
    def _calculate_synaptic_input(self, neuron_id: str) -> float:
        """计算突触输入"""
        total_input = 0.0
        
        for synapse in self.synapses.values():
            if synapse.post_neuron_id == neuron_id:
                pre_neuron = self.neurons.get(synapse.pre_neuron_id)
                if pre_neuron and hasattr(pre_neuron, 'spike_history'):
                    # 检查是否有最近的脉冲
                    current_time = time.time()
                    recent_spikes = [spike for spike in pre_neuron.spike_history 
                                   if current_time - spike < synapse.delay]
                    
                    if recent_spikes:
                        # 根据突触类型调整输入
                        if synapse.synapse_type == SynapseType.EXCITATORY:
                            total_input += synapse.weight
                        elif synapse.synapse_type == SynapseType.INHIBITORY:
                            total_input -= synapse.weight
                        elif synapse.synapse_type == SynapseType.MODULATORY:
                            total_input += synapse.weight * 0.5
        
        return total_input
    
    def _update_synapse_weights(self, spike_events: Dict[str, float]):
        """更新突触权重"""
        for synapse in self.synapses.values():
            if not synapse.plasticity.get("stdp_enabled", False):
                continue
            
            pre_spike_time = spike_events.get(synapse.pre_neuron_id, -1)
            post_spike_time = spike_events.get(synapse.post_neuron_id, -1)
            
            if pre_spike_time != -1 or post_spike_time != -1:
                new_weight = self.stdp.update_weight(synapse, pre_spike_time, post_spike_time)
                synapse.weight = new_weight
                synapse.last_activity = self.simulation_time
    
    def _get_network_state(self) -> Dict[str, Any]:
        """获取网络状态"""
        neuron_states = {}
        for neuron_id, neuron in self.neurons.items():
            neuron_states[neuron_id] = neuron.get_state()
        
        synapse_states = {}
        for synapse_id, synapse in self.synapses.items():
            synapse_states[synapse_id] = {
                "weight": synapse.weight,
                "last_activity": synapse.last_activity,
                "plasticity": synapse.plasticity
            }
        
        return {
            "neurons": neuron_states,
            "synapses": synapse_states,
            "topology": dict(self.topology)
        }
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        total_neurons = len(self.neurons)
        total_synapses = len(self.synapses)
        
        # 计算平均发放率
        firing_rates = []
        for neuron in self.neurons.values():
            firing_rate = neuron._calculate_firing_rate()
            firing_rates.append(firing_rate)
        
        avg_firing_rate = np.mean(firing_rates) if firing_rates else 0.0
        
        # 计算突触权重统计
        weights = [synapse.weight for synapse in self.synapses.values()]
        avg_weight = np.mean(weights) if weights else 0.0
        weight_std = np.std(weights) if weights else 0.0
        
        return {
            "total_neurons": total_neurons,
            "total_synapses": total_synapses,
            "average_firing_rate": avg_firing_rate,
            "average_weight": avg_weight,
            "weight_std": weight_std,
            "spike_count": len(self.spike_history),
            "simulation_time": self.simulation_time
        }


class NeuromorphicOptimization:
    """神经形态优化"""
    
    def __init__(self):
        self.logger = logger.bind(name="neuromorphic_optimization")
        self.networks: Dict[str, NeuromorphicNetwork] = {}
        self.optimization_history = deque(maxlen=1000)
        
    def create_optimization_network(self, network_id: str, num_neurons: int = 100) -> NeuromorphicNetwork:
        """创建优化网络"""
        network = NeuromorphicNetwork(network_id)
        
        # 创建神经元
        for i in range(num_neurons):
            neuron_id = f"neuron_{i}"
            neuron_type = NeuronType.LIF if i % 2 == 0 else NeuronType.IZH
            
            # 随机参数
            parameters = {
                "tau_m": random.uniform(15.0, 25.0),
                "v_threshold": random.uniform(-60.0, -50.0),
                "refractory_period": random.uniform(1.0, 3.0)
            }
            
            network.add_neuron(neuron_id, neuron_type, parameters)
        
        # 创建突触连接
        for i in range(num_neurons):
            for j in range(random.randint(1, 5)):  # 每个神经元连接1-5个其他神经元
                target = random.randint(0, num_neurons - 1)
                if target != i:
                    synapse_id = f"synapse_{i}_{target}"
                    synapse_type = random.choice(list(SynapseType))
                    weight = random.uniform(0.1, 0.9)
                    
                    network.add_synapse(
                        synapse_id, f"neuron_{i}", f"neuron_{target}",
                        synapse_type, weight
                    )
        
        self.networks[network_id] = network
        return network
    
    def optimize_network_parameters(self, network_id: str, target_function: callable) -> Dict[str, Any]:
        """优化网络参数"""
        self.logger.info(f"优化网络参数: {network_id}")
        
        if network_id not in self.networks:
            return {"success": False, "error": "网络不存在"}
        
        network = self.networks[network_id]
        start_time = time.time()
        
        # 模拟网络运行
        best_fitness = float('-inf')
        best_parameters = {}
        
        for iteration in range(10):  # 10次迭代
            # 随机调整参数
            for neuron in network.neurons.values():
                if hasattr(neuron, 'parameters'):
                    for param_name in neuron.parameters:
                        if random.random() < 0.1:  # 10%概率调整参数
                            current_value = neuron.parameters[param_name]
                            neuron.parameters[param_name] = current_value * random.uniform(0.8, 1.2)
            
            # 运行网络
            for step in range(100):  # 100个时间步
                network.simulate_step()
            
            # 评估适应度
            fitness = target_function(network)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_parameters = {neuron_id: neuron.parameters.copy() 
                                 for neuron_id, neuron in network.neurons.items()}
        
        execution_time = time.time() - start_time
        
        result = {
            "network_id": network_id,
            "best_fitness": best_fitness,
            "best_parameters": best_parameters,
            "execution_time": execution_time,
            "iterations": 10,
            "success": True
        }
        
        self.optimization_history.append(result)
        return result
    
    def neuromorphic_learning(self, network_id: str, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """神经形态学习"""
        self.logger.info(f"神经形态学习: {network_id}")
        
        if network_id not in self.networks:
            return {"success": False, "error": "网络不存在"}
        
        network = self.networks[network_id]
        start_time = time.time()
        
        learning_results = []
        
        for data_point in training_data:
            # 准备输入
            inputs = data_point.get("inputs", {})
            target_output = data_point.get("target", {})
            
            # 运行网络
            for step in range(50):  # 50个时间步
                result = network.simulate_step(inputs)
            
            # 计算输出误差
            actual_output = self._extract_network_output(network)
            error = self._calculate_error(actual_output, target_output)
            
            learning_results.append({
                "data_point": data_point,
                "actual_output": actual_output,
                "error": error
            })
        
        execution_time = time.time() - start_time
        
        result = {
            "network_id": network_id,
            "learning_results": learning_results,
            "average_error": np.mean([r["error"] for r in learning_results]),
            "execution_time": execution_time,
            "success": True
        }
        
        self.optimization_history.append(result)
        return result
    
    def _extract_network_output(self, network: NeuromorphicNetwork) -> Dict[str, float]:
        """提取网络输出"""
        outputs = {}
        
        for neuron_id, neuron in network.neurons.items():
            # 使用发放率作为输出
            firing_rate = neuron._calculate_firing_rate()
            outputs[neuron_id] = firing_rate
        
        return outputs
    
    def _calculate_error(self, actual_output: Dict[str, float], target_output: Dict[str, float]) -> float:
        """计算误差"""
        total_error = 0.0
        count = 0
        
        for neuron_id in actual_output:
            if neuron_id in target_output:
                error = abs(actual_output[neuron_id] - target_output[neuron_id])
                total_error += error
                count += 1
        
        return total_error / count if count > 0 else 0.0


class NeuromorphicComputing:
    """神经形态计算主类"""
    
    def __init__(self):
        self.optimization = NeuromorphicOptimization()
        self.logger = logger.bind(name="neuromorphic_computing")
        
    def create_brain_inspired_network(self, network_id: str, num_neurons: int = 100) -> NeuromorphicNetwork:
        """创建类脑网络"""
        return self.optimization.create_optimization_network(network_id, num_neurons)
    
    def run_neuromorphic_optimization(self, network_id: str, optimization_target: str) -> Dict[str, Any]:
        """运行神经形态优化"""
        # 定义目标函数
        def target_function(network):
            stats = network.get_network_statistics()
            # 优化目标：最大化发放率，最小化权重方差
            firing_rate_score = stats["average_firing_rate"] / 100.0  # 归一化
            weight_stability_score = 1.0 / (1.0 + stats["weight_std"])  # 权重稳定性
            return firing_rate_score + weight_stability_score
        
        return self.optimization.optimize_network_parameters(network_id, target_function)
    
    def run_neuromorphic_learning(self, network_id: str, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行神经形态学习"""
        return self.optimization.neuromorphic_learning(network_id, training_data)
    
    def simulate_brain_activity(self, network_id: str, duration: float = 1000.0) -> Dict[str, Any]:
        """模拟大脑活动"""
        if network_id not in self.optimization.networks:
            return {"success": False, "error": "网络不存在"}
        
        network = self.optimization.networks[network_id]
        start_time = time.time()
        
        # 模拟大脑活动
        activity_data = []
        dt = 1.0
        
        for step in range(int(duration / dt)):
            # 随机外部输入
            external_inputs = {}
            for neuron_id in network.neurons:
                if random.random() < 0.1:  # 10%概率有外部输入
                    external_inputs[neuron_id] = random.uniform(0.0, 10.0)
            
            # 运行网络
            result = network.simulate_step(external_inputs, dt)
            
            # 记录活动
            activity_data.append({
                "time": result["simulation_time"],
                "spike_count": len(result["spike_events"]),
                "active_neurons": len(result["spike_events"])
            })
        
        execution_time = time.time() - start_time
        
        # 计算活动统计
        spike_counts = [data["spike_count"] for data in activity_data]
        active_neurons = [data["active_neurons"] for data in activity_data]
        
        return {
            "network_id": network_id,
            "duration": duration,
            "total_spikes": sum(spike_counts),
            "average_spikes_per_step": np.mean(spike_counts),
            "average_active_neurons": np.mean(active_neurons),
            "activity_data": activity_data,
            "execution_time": execution_time,
            "success": True
        }
    
    def get_neuromorphic_report(self) -> Dict[str, Any]:
        """获取神经形态计算报告"""
        total_networks = len(self.optimization.networks)
        total_optimizations = len(self.optimization.optimization_history)
        
        network_statistics = {}
        for network_id, network in self.optimization.networks.items():
            stats = network.get_network_statistics()
            network_statistics[network_id] = stats
        
        return {
            "total_networks": total_networks,
            "total_optimizations": total_optimizations,
            "network_statistics": network_statistics,
            "optimization_history": list(self.optimization.optimization_history)
        }


# 使用示例
if __name__ == "__main__":
    # 创建神经形态计算系统
    neuromorphic = NeuromorphicComputing()
    
    # 创建类脑网络
    network = neuromorphic.create_brain_inspired_network("brain_network", 50)
    print("类脑网络已创建")
    
    # 运行神经形态优化
    optimization_result = neuromorphic.run_neuromorphic_optimization("brain_network", "performance")
    print("神经形态优化结果:")
    print(json.dumps(optimization_result, indent=2, ensure_ascii=False, default=str))
    
    # 运行神经形态学习
    training_data = [
        {"inputs": {"neuron_0": 5.0, "neuron_1": 3.0}, "target": {"neuron_10": 0.8}},
        {"inputs": {"neuron_2": 4.0, "neuron_3": 6.0}, "target": {"neuron_15": 0.6}},
        {"inputs": {"neuron_5": 2.0, "neuron_7": 8.0}, "target": {"neuron_20": 0.9}}
    ]
    
    learning_result = neuromorphic.run_neuromorphic_learning("brain_network", training_data)
    print("\n神经形态学习结果:")
    print(json.dumps(learning_result, indent=2, ensure_ascii=False, default=str))
    
    # 模拟大脑活动
    brain_activity = neuromorphic.simulate_brain_activity("brain_network", 500.0)
    print("\n大脑活动模拟结果:")
    print(json.dumps(brain_activity, indent=2, ensure_ascii=False, default=str))
    
    # 获取神经形态计算报告
    neuromorphic_report = neuromorphic.get_neuromorphic_report()
    print(f"\n神经形态计算报告: {neuromorphic_report}") 