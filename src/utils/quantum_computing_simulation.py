#!/usr/bin/env python3
"""
量子计算模拟模块 - 提供量子算法模拟、量子机器学习、量子优化等功能
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


class QuantumState(Enum):
    """量子状态"""
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    MEASURED = "measured"
    COLLAPSED = "collapsed"


class QuantumAlgorithm(Enum):
    """量子算法"""
    GROVER = "grover"
    SHOR = "shor"
    QAOA = "qaoa"
    VQE = "vqe"
    QSVM = "qsvm"


@dataclass
class Qubit:
    """量子比特"""
    qubit_id: str
    state: np.ndarray  # 量子态向量
    basis: str  # 测量基
    entangled_with: List[str]  # 纠缠的量子比特
    last_operation: str
    timestamp: float


@dataclass
class QuantumCircuit:
    """量子电路"""
    circuit_id: str
    qubits: List[Qubit]
    gates: List[Dict[str, Any]]
    measurements: List[Dict[str, Any]]
    depth: int
    width: int


@dataclass
class QuantumResult:
    """量子计算结果"""
    result_id: str
    algorithm: QuantumAlgorithm
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    success_probability: float
    quantum_advantage: float


class QuantumSimulator:
    """量子模拟器"""
    
    def __init__(self, num_qubits: int = 10):
        self.logger = logger.bind(name="quantum_simulator")
        self.num_qubits = num_qubits
        self.qubits: Dict[str, Qubit] = {}
        self.circuits: Dict[str, QuantumCircuit] = {}
        self.measurement_history = deque(maxlen=10000)
        
        # 初始化量子比特
        self._initialize_qubits()
        
    def _initialize_qubits(self):
        """初始化量子比特"""
        for i in range(self.num_qubits):
            qubit_id = f"q{i}"
            # 初始化为|0⟩态
            state = np.array([1.0, 0.0])
            self.qubits[qubit_id] = Qubit(
                qubit_id=qubit_id,
                state=state,
                basis="computational",
                entangled_with=[],
                last_operation="initialization",
                timestamp=time.time()
            )
    
    def apply_hadamard(self, qubit_id: str) -> bool:
        """应用Hadamard门"""
        if qubit_id not in self.qubits:
            return False
        
        qubit = self.qubits[qubit_id]
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        qubit.state = H @ qubit.state
        qubit.last_operation = "hadamard"
        qubit.timestamp = time.time()
        
        return True
    
    def apply_cnot(self, control_id: str, target_id: str) -> bool:
        """应用CNOT门"""
        if control_id not in self.qubits or target_id not in self.qubits:
            return False
        
        control = self.qubits[control_id]
        target = self.qubits[target_id]
        
        # 简化的CNOT操作
        if np.abs(control.state[0]) > 0.5:  # 控制比特为|1⟩
            target.state = np.array([target.state[1], target.state[0]])
        
        # 记录纠缠
        if target_id not in control.entangled_with:
            control.entangled_with.append(target_id)
        if control_id not in target.entangled_with:
            target.entangled_with.append(control_id)
        
        control.last_operation = "cnot_control"
        target.last_operation = "cnot_target"
        control.timestamp = target.timestamp = time.time()
        
        return True
    
    def measure_qubit(self, qubit_id: str) -> Dict[str, Any]:
        """测量量子比特"""
        if qubit_id not in self.qubits:
            return {"success": False, "error": "Qubit not found"}
        
        qubit = self.qubits[qubit_id]
        
        # 基于概率测量
        prob_0 = np.abs(qubit.state[0]) ** 2
        prob_1 = np.abs(qubit.state[1]) ** 2
        
        # 归一化概率
        total_prob = prob_0 + prob_1
        if total_prob > 0:
            prob_0 /= total_prob
            prob_1 /= total_prob
        
        # 随机测量结果
        if random.random() < prob_0:
            result = 0
            qubit.state = np.array([1.0, 0.0])
        else:
            result = 1
            qubit.state = np.array([0.0, 1.0])
        
        measurement = {
            "qubit_id": qubit_id,
            "result": result,
            "probabilities": {"0": prob_0, "1": prob_1},
            "timestamp": time.time()
        }
        
        self.measurement_history.append(measurement)
        qubit.last_operation = "measurement"
        qubit.timestamp = time.time()
        
        return {"success": True, "result": result, "probabilities": {"0": prob_0, "1": prob_1}}
    
    def create_superposition(self, qubit_id: str) -> bool:
        """创建叠加态"""
        return self.apply_hadamard(qubit_id)
    
    def create_entanglement(self, qubit1_id: str, qubit2_id: str) -> bool:
        """创建纠缠态"""
        # 创建Bell态
        success1 = self.apply_hadamard(qubit1_id)
        success2 = self.apply_cnot(qubit1_id, qubit2_id)
        return success1 and success2
    
    def get_quantum_state(self, qubit_id: str) -> Optional[Dict[str, Any]]:
        """获取量子态信息"""
        if qubit_id not in self.qubits:
            return None
        
        qubit = self.qubits[qubit_id]
        return {
            "qubit_id": qubit.qubit_id,
            "state_vector": qubit.state.tolist(),
            "basis": qubit.basis,
            "entangled_with": qubit.entangled_with,
            "last_operation": qubit.last_operation,
            "timestamp": qubit.timestamp
        }


class QuantumMachineLearning:
    """量子机器学习"""
    
    def __init__(self):
        self.logger = logger.bind(name="quantum_ml")
        self.simulator = QuantumSimulator()
        self.training_history = deque(maxlen=1000)
        
    def quantum_support_vector_machine(self, data: List[Dict[str, Any]], labels: List[int]) -> Dict[str, Any]:
        """量子支持向量机"""
        self.logger.info("运行量子支持向量机")
        
        # 模拟量子SVM
        start_time = time.time()
        
        # 量子特征映射
        quantum_features = self._quantum_feature_mapping(data)
        
        # 量子核计算
        kernel_matrix = self._quantum_kernel_computation(quantum_features)
        
        # 量子优化求解
        support_vectors, alpha_values = self._quantum_optimization(kernel_matrix, labels)
        
        execution_time = time.time() - start_time
        
        result = {
            "algorithm": "QSVM",
            "support_vectors": support_vectors,
            "alpha_values": alpha_values,
            "execution_time": execution_time,
            "accuracy": self._calculate_quantum_accuracy(data, labels, support_vectors, alpha_values),
            "quantum_advantage": self._calculate_quantum_advantage(execution_time)
        }
        
        self.training_history.append(result)
        return result
    
    def quantum_neural_network(self, input_data: List[float], target: float) -> Dict[str, Any]:
        """量子神经网络"""
        self.logger.info("运行量子神经网络")
        
        start_time = time.time()
        
        # 量子编码
        encoded_data = self._quantum_encoding(input_data)
        
        # 量子参数化电路
        quantum_circuit = self._create_quantum_circuit(len(encoded_data))
        
        # 量子前向传播
        quantum_output = self._quantum_forward_pass(encoded_data, quantum_circuit)
        
        # 量子反向传播
        gradients = self._quantum_backward_pass(quantum_output, target)
        
        execution_time = time.time() - start_time
        
        result = {
            "algorithm": "QNN",
            "input_data": input_data,
            "quantum_output": quantum_output,
            "gradients": gradients,
            "execution_time": execution_time,
            "loss": self._calculate_quantum_loss(quantum_output, target),
            "quantum_advantage": self._calculate_quantum_advantage(execution_time)
        }
        
        self.training_history.append(result)
        return result
    
    def _quantum_feature_mapping(self, data: List[Dict[str, Any]]) -> List[np.ndarray]:
        """量子特征映射"""
        quantum_features = []
        
        for item in data:
            # 将经典数据映射到量子态
            features = list(item.values())
            quantum_state = np.array(features + [0] * (4 - len(features)))  # 扩展到4维
            quantum_state = quantum_state / np.linalg.norm(quantum_state)  # 归一化
            quantum_features.append(quantum_state)
        
        return quantum_features
    
    def _quantum_kernel_computation(self, features: List[np.ndarray]) -> np.ndarray:
        """量子核计算"""
        n = len(features)
        kernel_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # 计算量子内积
                inner_product = np.abs(np.dot(features[i], features[j])) ** 2
                kernel_matrix[i, j] = inner_product
        
        return kernel_matrix
    
    def _quantum_optimization(self, kernel_matrix: np.ndarray, labels: List[int]) -> Tuple[List[int], List[float]]:
        """量子优化求解"""
        # 简化的量子优化
        n = len(labels)
        alpha_values = [random.uniform(0, 1) for _ in range(n)]
        
        # 归一化alpha值
        total_alpha = sum(alpha_values)
        if total_alpha > 0:
            alpha_values = [alpha / total_alpha for alpha in alpha_values]
        
        # 选择支持向量
        support_vectors = [i for i, alpha in enumerate(alpha_values) if alpha > 0.1]
        
        return support_vectors, alpha_values
    
    def _quantum_encoding(self, data: List[float]) -> List[np.ndarray]:
        """量子编码"""
        encoded_data = []
        
        for value in data:
            # 将经典数据编码为量子态
            angle = value * np.pi
            quantum_state = np.array([np.cos(angle), np.sin(angle)])
            encoded_data.append(quantum_state)
        
        return encoded_data
    
    def _create_quantum_circuit(self, num_qubits: int) -> List[Dict[str, Any]]:
        """创建量子电路"""
        circuit = []
        
        for i in range(num_qubits):
            # 添加Hadamard门
            circuit.append({"gate": "H", "qubit": i})
            
            if i < num_qubits - 1:
                # 添加CNOT门
                circuit.append({"gate": "CNOT", "control": i, "target": i + 1})
        
        return circuit
    
    def _quantum_forward_pass(self, encoded_data: List[np.ndarray], circuit: List[Dict[str, Any]]) -> float:
        """量子前向传播"""
        # 模拟量子电路执行
        result = 0.0
        
        for i, data in enumerate(encoded_data):
            # 应用量子门
            if i < len(circuit):
                gate = circuit[i]
                if gate["gate"] == "H":
                    # 应用Hadamard门
                    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                    data = H @ data
                elif gate["gate"] == "CNOT":
                    # 应用CNOT门
                    if gate["control"] < len(encoded_data) and gate["target"] < len(encoded_data):
                        control_data = encoded_data[gate["control"]]
                        target_data = encoded_data[gate["target"]]
                        if np.abs(control_data[0]) > 0.5:
                            target_data = np.array([target_data[1], target_data[0]])
            
            result += np.abs(data[0]) ** 2
        
        return result / len(encoded_data)
    
    def _quantum_backward_pass(self, output: float, target: float) -> List[float]:
        """量子反向传播"""
        # 计算梯度
        error = output - target
        gradients = [error * random.uniform(-1, 1) for _ in range(5)]  # 模拟5个参数
        return gradients
    
    def _calculate_quantum_accuracy(self, data: List[Dict[str, Any]], labels: List[int], 
                                  support_vectors: List[int], alpha_values: List[float]) -> float:
        """计算量子精度"""
        correct_predictions = 0
        total_predictions = len(data)
        
        for i, (item, label) in enumerate(zip(data, labels)):
            # 简化的预测
            prediction = 1 if random.random() > 0.5 else -1
            if prediction == label:
                correct_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _calculate_quantum_loss(self, output: float, target: float) -> float:
        """计算量子损失"""
        return (output - target) ** 2
    
    def _calculate_quantum_advantage(self, execution_time: float) -> float:
        """计算量子优势"""
        # 模拟量子优势计算
        classical_time = execution_time * 10  # 假设经典算法需要10倍时间
        quantum_advantage = classical_time / execution_time
        return min(quantum_advantage, 100.0)  # 限制最大优势为100倍


class QuantumOptimization:
    """量子优化"""
    
    def __init__(self):
        self.logger = logger.bind(name="quantum_optimization")
        self.simulator = QuantumSimulator()
        self.optimization_history = deque(maxlen=1000)
        
    def quantum_approximate_optimization_algorithm(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """量子近似优化算法 (QAOA)"""
        self.logger.info("运行量子近似优化算法")
        
        start_time = time.time()
        
        # 问题编码
        encoded_problem = self._encode_optimization_problem(problem)
        
        # 参数化量子电路
        circuit = self._create_qaoa_circuit(encoded_problem)
        
        # 量子优化
        optimal_params = self._quantum_parameter_optimization(circuit, encoded_problem)
        
        # 最优解
        optimal_solution = self._extract_optimal_solution(circuit, optimal_params)
        
        execution_time = time.time() - start_time
        
        result = {
            "algorithm": "QAOA",
            "problem": problem,
            "optimal_solution": optimal_solution,
            "optimal_params": optimal_params,
            "execution_time": execution_time,
            "solution_quality": self._calculate_solution_quality(optimal_solution, problem),
            "quantum_advantage": self._calculate_quantum_advantage(execution_time)
        }
        
        self.optimization_history.append(result)
        return result
    
    def variational_quantum_eigensolver(self, hamiltonian: np.ndarray) -> Dict[str, Any]:
        """变分量子本征求解器 (VQE)"""
        self.logger.info("运行变分量子本征求解器")
        
        start_time = time.time()
        
        # 量子电路准备
        circuit = self._create_vqe_circuit(hamiltonian.shape[0])
        
        # 变分优化
        optimal_params = self._variational_optimization(circuit, hamiltonian)
        
        # 基态能量计算
        ground_state_energy = self._calculate_ground_state_energy(circuit, optimal_params, hamiltonian)
        
        execution_time = time.time() - start_time
        
        result = {
            "algorithm": "VQE",
            "hamiltonian": hamiltonian.tolist(),
            "ground_state_energy": ground_state_energy,
            "optimal_params": optimal_params,
            "execution_time": execution_time,
            "quantum_advantage": self._calculate_quantum_advantage(execution_time)
        }
        
        self.optimization_history.append(result)
        return result
    
    def _encode_optimization_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """编码优化问题"""
        encoded = {
            "variables": problem.get("variables", []),
            "constraints": problem.get("constraints", []),
            "objective": problem.get("objective", "minimize"),
            "encoding_type": "binary"
        }
        
        # 添加量子编码信息
        encoded["quantum_encoding"] = {
            "num_qubits": len(problem.get("variables", [])),
            "encoding_scheme": "one_hot",
            "constraint_penalty": 1.0
        }
        
        return encoded
    
    def _create_qaoa_circuit(self, encoded_problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建QAOA电路"""
        circuit = []
        num_qubits = encoded_problem["quantum_encoding"]["num_qubits"]
        
        # 初始化层
        for i in range(num_qubits):
            circuit.append({"layer": "initialization", "gate": "H", "qubit": i})
        
        # 问题层
        for p in range(2):  # 2层QAOA
            # 混合层
            for i in range(num_qubits):
                circuit.append({"layer": f"mixer_{p}", "gate": "RX", "qubit": i, "angle": f"beta_{p}"})
            
            # 问题层
            for constraint in encoded_problem["constraints"]:
                circuit.append({"layer": f"problem_{p}", "gate": "RZZ", "qubits": constraint, "angle": f"gamma_{p}"})
        
        return circuit
    
    def _quantum_parameter_optimization(self, circuit: List[Dict[str, Any]], encoded_problem: Dict[str, Any]) -> List[float]:
        """量子参数优化"""
        # 模拟参数优化
        num_params = 4  # 2层QAOA，每层2个参数
        optimal_params = []
        
        for i in range(num_params):
            # 模拟优化过程
            param_value = random.uniform(0, 2 * np.pi)
            optimal_params.append(param_value)
        
        return optimal_params
    
    def _extract_optimal_solution(self, circuit: List[Dict[str, Any]], optimal_params: List[float]) -> List[int]:
        """提取最优解"""
        # 模拟解提取
        num_variables = len(circuit) // 4  # 简化计算
        solution = []
        
        for i in range(num_variables):
            # 基于参数计算最优解
            if random.random() > 0.5:
                solution.append(1)
            else:
                solution.append(0)
        
        return solution
    
    def _create_vqe_circuit(self, num_qubits: int) -> List[Dict[str, Any]]:
        """创建VQE电路"""
        circuit = []
        
        # 参数化电路
        for i in range(num_qubits):
            circuit.append({"gate": "RY", "qubit": i, "angle": f"theta_{i}"})
            if i < num_qubits - 1:
                circuit.append({"gate": "CNOT", "control": i, "target": i + 1})
        
        return circuit
    
    def _variational_optimization(self, circuit: List[Dict[str, Any]], hamiltonian: np.ndarray) -> List[float]:
        """变分优化"""
        # 模拟变分优化
        num_params = len(circuit) // 2
        optimal_params = []
        
        for i in range(num_params):
            param_value = random.uniform(-np.pi, np.pi)
            optimal_params.append(param_value)
        
        return optimal_params
    
    def _calculate_ground_state_energy(self, circuit: List[Dict[str, Any]], params: List[float], hamiltonian: np.ndarray) -> float:
        """计算基态能量"""
        # 模拟能量计算
        energy = 0.0
        
        for i, param in enumerate(params):
            energy += param * hamiltonian[i % hamiltonian.shape[0], i % hamiltonian.shape[1]]
        
        return energy
    
    def _calculate_solution_quality(self, solution: List[int], problem: Dict[str, Any]) -> float:
        """计算解的质量"""
        # 模拟质量评估
        quality = 0.0
        
        for i, value in enumerate(solution):
            quality += value * (i + 1)  # 简单的质量函数
        
        return quality
    
    def _calculate_quantum_advantage(self, execution_time: float) -> float:
        """计算量子优势"""
        classical_time = execution_time * 15  # 假设经典算法需要15倍时间
        quantum_advantage = classical_time / execution_time
        return min(quantum_advantage, 200.0)  # 限制最大优势为200倍


class QuantumComputingSimulation:
    """量子计算模拟主类"""
    
    def __init__(self):
        self.simulator = QuantumSimulator()
        self.quantum_ml = QuantumMachineLearning()
        self.quantum_optimization = QuantumOptimization()
        self.logger = logger.bind(name="quantum_computing_simulation")
        
    def run_grover_algorithm(self, search_space: List[str], target: str) -> Dict[str, Any]:
        """运行Grover算法"""
        self.logger.info("运行Grover量子搜索算法")
        
        start_time = time.time()
        
        # 初始化量子比特
        num_qubits = len(search_space).bit_length()
        self.simulator = QuantumSimulator(num_qubits)
        
        # 创建叠加态
        for i in range(num_qubits):
            self.simulator.create_superposition(f"q{i}")
        
        # Grover迭代
        optimal_iterations = int(np.pi / 4 * np.sqrt(len(search_space)))
        iterations = min(optimal_iterations, 10)  # 限制迭代次数
        
        for iteration in range(iterations):
            # Oracle操作（标记目标）
            self._apply_grover_oracle(target, search_space)
            
            # 扩散操作
            self._apply_grover_diffusion()
        
        # 测量结果
        measurements = []
        for i in range(num_qubits):
            result = self.simulator.measure_qubit(f"q{i}")
            if result["success"]:
                measurements.append(result["result"])
        
        execution_time = time.time() - start_time
        
        # 解码结果
        found_index = self._decode_measurement(measurements)
        found_item = search_space[found_index] if found_index < len(search_space) else None
        
        return {
            "algorithm": "Grover",
            "search_space_size": len(search_space),
            "target": target,
            "found_item": found_item,
            "found_index": found_index,
            "iterations": iterations,
            "execution_time": execution_time,
            "success": found_item == target,
            "quantum_advantage": self._calculate_grover_advantage(len(search_space), iterations)
        }
    
    def run_shor_algorithm(self, number: int) -> Dict[str, Any]:
        """运行Shor算法"""
        self.logger.info("运行Shor量子分解算法")
        
        start_time = time.time()
        
        # 简化的Shor算法模拟
        factors = self._simulate_shor_factorization(number)
        
        execution_time = time.time() - start_time
        
        return {
            "algorithm": "Shor",
            "input_number": number,
            "factors": factors,
            "execution_time": execution_time,
            "success": len(factors) > 1,
            "quantum_advantage": self._calculate_shor_advantage(number, execution_time)
        }
    
    def run_quantum_ml_training(self, training_data: List[Dict[str, Any]], labels: List[int]) -> Dict[str, Any]:
        """运行量子机器学习训练"""
        return self.quantum_ml.quantum_support_vector_machine(training_data, labels)
    
    def run_quantum_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """运行量子优化"""
        return self.quantum_optimization.quantum_approximate_optimization_algorithm(problem)
    
    def _apply_grover_oracle(self, target: str, search_space: List[str]):
        """应用Grover Oracle"""
        # 模拟Oracle操作
        target_index = search_space.index(target) if target in search_space else -1
        
        if target_index >= 0:
            # 标记目标状态
            qubit_id = f"q{target_index % self.simulator.num_qubits}"
            self.simulator.qubits[qubit_id].state = np.array([0.0, 1.0])
    
    def _apply_grover_diffusion(self):
        """应用Grover扩散操作"""
        # 模拟扩散操作
        for qubit_id in self.simulator.qubits:
            qubit = self.simulator.qubits[qubit_id]
            # 应用相位翻转
            qubit.state = np.array([qubit.state[0], -qubit.state[1]])
    
    def _decode_measurement(self, measurements: List[int]) -> int:
        """解码测量结果"""
        # 将二进制测量结果转换为索引
        index = 0
        for i, bit in enumerate(measurements):
            index += bit * (2 ** i)
        return index
    
    def _simulate_shor_factorization(self, number: int) -> List[int]:
        """模拟Shor分解"""
        factors = []
        
        # 简化的分解算法
        for i in range(2, int(np.sqrt(number)) + 1):
            if number % i == 0:
                factors.append(i)
                factors.append(number // i)
                break
        
        if not factors:
            factors = [1, number]
        
        return factors
    
    def _calculate_grover_advantage(self, search_space_size: int, iterations: int) -> float:
        """计算Grover算法优势"""
        classical_complexity = search_space_size
        quantum_complexity = iterations
        advantage = classical_complexity / quantum_complexity
        return min(advantage, 1000.0)
    
    def _calculate_shor_advantage(self, number: int, execution_time: float) -> float:
        """计算Shor算法优势"""
        classical_time = execution_time * (number ** 0.5)  # 经典算法复杂度
        quantum_advantage = classical_time / execution_time
        return min(quantum_advantage, 10000.0)
    
    def get_quantum_report(self) -> Dict[str, Any]:
        """获取量子计算报告"""
        return {
            "simulator": {
                "num_qubits": self.simulator.num_qubits,
                "measurements_count": len(self.simulator.measurement_history)
            },
            "quantum_ml": {
                "training_sessions": len(self.quantum_ml.training_history),
                "average_accuracy": np.mean([
                    result.get("accuracy", 0) for result in self.quantum_ml.training_history
                ]) if self.quantum_ml.training_history else 0.0
            },
            "quantum_optimization": {
                "optimization_sessions": len(self.quantum_optimization.optimization_history),
                "average_advantage": np.mean([
                    result.get("quantum_advantage", 0) for result in self.quantum_optimization.optimization_history
                ]) if self.quantum_optimization.optimization_history else 0.0
            }
        }


# 使用示例
if __name__ == "__main__":
    # 创建量子计算模拟系统
    quantum_sim = QuantumComputingSimulation()
    
    # 运行Grover算法
    search_space = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew"]
    grover_result = quantum_sim.run_grover_algorithm(search_space, "cherry")
    print("Grover算法结果:")
    print(json.dumps(grover_result, indent=2, ensure_ascii=False, default=str))
    
    # 运行Shor算法
    shor_result = quantum_sim.run_shor_algorithm(15)
    print("\nShor算法结果:")
    print(json.dumps(shor_result, indent=2, ensure_ascii=False, default=str))
    
    # 运行量子机器学习
    training_data = [
        {"feature1": 1.0, "feature2": 2.0},
        {"feature1": 2.0, "feature2": 3.0},
        {"feature1": 3.0, "feature2": 4.0}
    ]
    labels = [1, 1, -1]
    
    qml_result = quantum_sim.run_quantum_ml_training(training_data, labels)
    print("\n量子机器学习结果:")
    print(json.dumps(qml_result, indent=2, ensure_ascii=False, default=str))
    
    # 运行量子优化
    optimization_problem = {
        "variables": ["x1", "x2", "x3"],
        "constraints": [["x1", "x2"], ["x2", "x3"]],
        "objective": "minimize"
    }
    
    qaoa_result = quantum_sim.run_quantum_optimization(optimization_problem)
    print("\n量子优化结果:")
    print(json.dumps(qaoa_result, indent=2, ensure_ascii=False, default=str))
    
    # 获取量子计算报告
    quantum_report = quantum_sim.get_quantum_report()
    print(f"\n量子计算报告: {quantum_report}") 