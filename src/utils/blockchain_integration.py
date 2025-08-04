#!/usr/bin/env python3
"""
区块链集成模块 - 提供去中心化、不可篡改的日志记录和智能合约功能
"""
import time
import json
import hashlib
import hmac
import base64
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import deque
from loguru import logger
import random


class BlockType(Enum):
    """区块类型"""
    TRANSACTION = "transaction"
    LOG = "log"
    CONTRACT = "contract"
    AUDIT = "audit"
    CONFIG = "config"


class TransactionStatus(Enum):
    """交易状态"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REVERTED = "reverted"


@dataclass
class Block:
    """区块"""
    index: int
    timestamp: float
    data: Dict[str, Any]
    previous_hash: str
    hash: str
    nonce: int
    block_type: BlockType
    difficulty: int


@dataclass
class Transaction:
    """交易"""
    tx_id: str
    from_address: str
    to_address: str
    data: Dict[str, Any]
    timestamp: float
    status: TransactionStatus
    gas_used: int
    block_hash: Optional[str] = None


@dataclass
class SmartContract:
    """智能合约"""
    contract_id: str
    name: str
    code: str
    owner: str
    created_at: float
    state: Dict[str, Any]
    functions: List[str]


class BlockchainNode:
    """区块链节点"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logger.bind(name=f"blockchain_node_{node_id}")
        
        # 区块链数据
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.contracts: Dict[str, SmartContract] = {}
        
        # 网络配置
        self.peers: List[str] = []
        self.difficulty = 4
        self.mining_reward = 10
        
        # 初始化创世区块
        self._create_genesis_block()
        
        # 线程锁
        self.lock = threading.Lock()
        
    def _create_genesis_block(self):
        """创建创世区块"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            data={"message": "Genesis Block", "creator": self.node_id},
            previous_hash="0",
            hash="",
            nonce=0,
            block_type=BlockType.CONFIG,
            difficulty=self.difficulty
        )
        
        genesis_block.hash = self._calculate_hash(genesis_block)
        self.chain.append(genesis_block)
        
        self.logger.info("创世区块已创建")
    
    def _calculate_hash(self, block: Block) -> str:
        """计算区块哈希"""
        block_string = json.dumps(asdict(block), sort_keys=True, default=str)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _mine_block(self, data: Dict[str, Any], block_type: BlockType) -> Block:
        """挖矿创建新区块"""
        previous_block = self.chain[-1]
        new_block = Block(
            index=previous_block.index + 1,
            timestamp=time.time(),
            data=data,
            previous_hash=previous_block.hash,
            hash="",
            nonce=0,
            block_type=block_type,
            difficulty=self.difficulty
        )
        
        # 工作量证明
        while True:
            new_block.hash = self._calculate_hash(new_block)
            if new_block.hash.startswith('0' * self.difficulty):
                break
            new_block.nonce += 1
        
        return new_block
    
    def add_transaction(self, from_address: str, to_address: str, data: Dict[str, Any]) -> str:
        """添加交易"""
        tx_id = hashlib.sha256(f"{from_address}{to_address}{time.time()}".encode()).hexdigest()
        
        transaction = Transaction(
            tx_id=tx_id,
            from_address=from_address,
            to_address=to_address,
            data=data,
            timestamp=time.time(),
            status=TransactionStatus.PENDING,
            gas_used=0
        )
        
        with self.lock:
            self.pending_transactions.append(transaction)
        
        self.logger.info(f"新交易已添加: {tx_id}")
        return tx_id
    
    def mine_pending_transactions(self, miner_address: str) -> Block:
        """挖掘待处理交易"""
        with self.lock:
            if not self.pending_transactions:
                raise ValueError("没有待处理的交易")
            
            # 创建区块数据
            block_data = {
                "transactions": [asdict(tx) for tx in self.pending_transactions],
                "miner": miner_address,
                "reward": self.mining_reward
            }
            
            # 挖掘新区块
            new_block = self._mine_block(block_data, BlockType.TRANSACTION)
            
            # 更新交易状态
            for transaction in self.pending_transactions:
                transaction.status = TransactionStatus.CONFIRMED
                transaction.block_hash = new_block.hash
            
            # 添加到区块链
            self.chain.append(new_block)
            
            # 清空待处理交易
            self.pending_transactions = []
            
            self.logger.info(f"新区块已挖掘: {new_block.hash}")
            return new_block
    
    def add_log_entry(self, log_data: Dict[str, Any]) -> str:
        """添加日志条目"""
        log_hash = hashlib.sha256(json.dumps(log_data, sort_keys=True).encode()).hexdigest()
        
        # 创建日志区块
        log_block = self._mine_block({
            "log_hash": log_hash,
            "log_data": log_data,
            "timestamp": time.time()
        }, BlockType.LOG)
        
        self.chain.append(log_block)
        
        self.logger.info(f"日志条目已添加: {log_hash}")
        return log_hash
    
    def deploy_contract(self, contract_name: str, contract_code: str, owner: str) -> str:
        """部署智能合约"""
        contract_id = hashlib.sha256(f"{contract_name}{owner}{time.time()}".encode()).hexdigest()
        
        contract = SmartContract(
            contract_id=contract_id,
            name=contract_name,
            code=contract_code,
            owner=owner,
            created_at=time.time(),
            state={},
            functions=self._extract_functions(contract_code)
        )
        
        with self.lock:
            self.contracts[contract_id] = contract
        
        # 记录合约部署
        self.add_log_entry({
            "action": "contract_deployed",
            "contract_id": contract_id,
            "contract_name": contract_name,
            "owner": owner
        })
        
        self.logger.info(f"智能合约已部署: {contract_id}")
        return contract_id
    
    def _extract_functions(self, contract_code: str) -> List[str]:
        """提取合约函数"""
        functions = []
        lines = contract_code.split('\n')
        
        for line in lines:
            if 'def ' in line and '(' in line:
                func_name = line.split('def ')[1].split('(')[0].strip()
                functions.append(func_name)
        
        return functions
    
    def execute_contract(self, contract_id: str, function_name: str, params: Dict[str, Any], caller: str) -> Dict[str, Any]:
        """执行智能合约"""
        if contract_id not in self.contracts:
            raise ValueError(f"合约不存在: {contract_id}")
        
        contract = self.contracts[contract_id]
        
        if function_name not in contract.functions:
            raise ValueError(f"函数不存在: {function_name}")
        
        # 模拟合约执行
        result = {
            "success": True,
            "function": function_name,
            "params": params,
            "caller": caller,
            "gas_used": random.randint(100, 1000),
            "return_value": f"executed_{function_name}",
            "timestamp": time.time()
        }
        
        # 更新合约状态
        contract.state[f"last_execution_{function_name}"] = result
        
        # 记录执行日志
        self.add_log_entry({
            "action": "contract_executed",
            "contract_id": contract_id,
            "function": function_name,
            "caller": caller,
            "result": result
        })
        
        self.logger.info(f"合约执行完成: {contract_id}.{function_name}")
        return result
    
    def get_chain_info(self) -> Dict[str, Any]:
        """获取区块链信息"""
        return {
            "node_id": self.node_id,
            "chain_length": len(self.chain),
            "pending_transactions": len(self.pending_transactions),
            "contracts_count": len(self.contracts),
            "difficulty": self.difficulty,
            "last_block": {
                "index": self.chain[-1].index,
                "hash": self.chain[-1].hash,
                "timestamp": self.chain[-1].timestamp
            } if self.chain else None
        }
    
    def verify_chain(self) -> bool:
        """验证区块链完整性"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # 验证哈希链接
            if current_block.previous_hash != previous_block.hash:
                return False
            
            # 验证当前区块哈希
            if current_block.hash != self._calculate_hash(current_block):
                return False
        
        return True
    
    def get_audit_trail(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """获取审计轨迹"""
        audit_trail = []
        
        for block in self.chain:
            if block.block_type == BlockType.AUDIT:
                if start_time and block.timestamp < start_time:
                    continue
                if end_time and block.timestamp > end_time:
                    continue
                
                audit_trail.append({
                    "block_index": block.index,
                    "timestamp": block.timestamp,
                    "data": block.data,
                    "hash": block.hash
                })
        
        return audit_trail


class BlockchainNetwork:
    """区块链网络"""
    
    def __init__(self):
        self.logger = logger.bind(name="blockchain_network")
        self.nodes: Dict[str, BlockchainNode] = {}
        self.consensus_algorithm = "proof_of_work"
        
    def add_node(self, node_id: str) -> BlockchainNode:
        """添加节点"""
        node = BlockchainNode(node_id)
        self.nodes[node_id] = node
        
        # 添加其他节点作为对等节点
        for existing_node_id in self.nodes.keys():
            if existing_node_id != node_id:
                node.peers.append(existing_node_id)
                self.nodes[existing_node_id].peers.append(node_id)
        
        self.logger.info(f"节点已添加: {node_id}")
        return node
    
    def remove_node(self, node_id: str):
        """移除节点"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            
            # 从其他节点的对等节点列表中移除
            for node in self.nodes.values():
                if node_id in node.peers:
                    node.peers.remove(node_id)
            
            self.logger.info(f"节点已移除: {node_id}")
    
    def broadcast_transaction(self, from_node: str, to_address: str, data: Dict[str, Any]) -> str:
        """广播交易"""
        if from_node not in self.nodes:
            raise ValueError(f"节点不存在: {from_node}")
        
        node = self.nodes[from_node]
        tx_id = node.add_transaction(from_node, to_address, data)
        
        # 广播到其他节点
        for peer_id in node.peers:
            if peer_id in self.nodes:
                try:
                    self.nodes[peer_id].add_transaction(from_node, to_address, data)
                except Exception as e:
                    self.logger.error(f"广播到节点 {peer_id} 失败: {e}")
        
        return tx_id
    
    def mine_on_all_nodes(self, miner_address: str) -> List[Block]:
        """在所有节点上挖矿"""
        blocks = []
        
        for node_id, node in self.nodes.items():
            try:
                if node.pending_transactions:
                    block = node.mine_pending_transactions(miner_address)
                    blocks.append(block)
            except Exception as e:
                self.logger.error(f"节点 {node_id} 挖矿失败: {e}")
        
        return blocks
    
    def sync_chain(self, source_node_id: str, target_node_id: str) -> bool:
        """同步区块链"""
        if source_node_id not in self.nodes or target_node_id not in self.nodes:
            return False
        
        source_node = self.nodes[source_node_id]
        target_node = self.nodes[target_node_id]
        
        # 同步区块链
        if len(source_node.chain) > len(target_node.chain):
            target_node.chain = source_node.chain.copy()
            target_node.pending_transactions = source_node.pending_transactions.copy()
            target_node.contracts = source_node.contracts.copy()
            
            self.logger.info(f"区块链已从 {source_node_id} 同步到 {target_node_id}")
            return True
        
        return False
    
    def get_network_status(self) -> Dict[str, Any]:
        """获取网络状态"""
        return {
            "total_nodes": len(self.nodes),
            "nodes": [node_id for node_id in self.nodes.keys()],
            "consensus_algorithm": self.consensus_algorithm,
            "network_health": self._calculate_network_health()
        }
    
    def _calculate_network_health(self) -> float:
        """计算网络健康度"""
        if not self.nodes:
            return 0.0
        
        total_blocks = sum(len(node.chain) for node in self.nodes.values())
        avg_blocks = total_blocks / len(self.nodes)
        
        # 计算一致性
        consistency_score = 0.0
        if len(self.nodes) > 1:
            chain_lengths = [len(node.chain) for node in self.nodes.values()]
            max_length = max(chain_lengths)
            min_length = min(chain_lengths)
            
            if max_length > 0:
                consistency_score = min_length / max_length
        
        return (avg_blocks / 10.0 + consistency_score) / 2.0


class BlockchainIntegration:
    """区块链集成主类"""
    
    def __init__(self):
        self.network = BlockchainNetwork()
        self.logger = logger.bind(name="blockchain_integration")
        
        # 创建默认节点
        self.main_node = self.network.add_node("main_node")
        
    def log_activity(self, activity_type: str, data: Dict[str, Any]) -> str:
        """记录活动到区块链"""
        log_data = {
            "activity_type": activity_type,
            "data": data,
            "timestamp": time.time(),
            "node_id": self.main_node.node_id
        }
        
        return self.main_node.add_log_entry(log_data)
    
    def create_smart_contract(self, contract_name: str, contract_code: str, owner: str) -> str:
        """创建智能合约"""
        return self.main_node.deploy_contract(contract_name, contract_code, owner)
    
    def execute_smart_contract(self, contract_id: str, function_name: str, params: Dict[str, Any], caller: str) -> Dict[str, Any]:
        """执行智能合约"""
        return self.main_node.execute_contract(contract_id, function_name, params, caller)
    
    def add_node(self, node_id: str) -> BlockchainNode:
        """添加节点"""
        return self.network.add_node(node_id)
    
    def get_network_status(self) -> Dict[str, Any]:
        """获取网络状态"""
        return self.network.get_network_status()
    
    def get_audit_trail(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """获取审计轨迹"""
        return self.main_node.get_audit_trail(start_time, end_time)
    
    def verify_integrity(self) -> bool:
        """验证区块链完整性"""
        return self.main_node.verify_chain()


# 使用示例
if __name__ == "__main__":
    # 创建区块链集成系统
    blockchain = BlockchainIntegration()
    
    # 记录活动
    log_hash = blockchain.log_activity("ticket_grabbing", {
        "event_id": "concert_001",
        "user_id": "user_123",
        "success": True,
        "timestamp": time.time()
    })
    print(f"活动已记录: {log_hash}")
    
    # 创建智能合约
    contract_code = """
def check_eligibility(user_id, event_id):
    return {"eligible": True, "reason": "user_verified"}

def process_ticket_purchase(user_id, event_id, ticket_count):
    return {"success": True, "tickets": ticket_count}
    """
    
    contract_id = blockchain.create_smart_contract("TicketContract", contract_code, "admin")
    print(f"智能合约已创建: {contract_id}")
    
    # 执行智能合约
    result = blockchain.execute_smart_contract(
        contract_id, 
        "check_eligibility", 
        {"user_id": "user_123", "event_id": "concert_001"}, 
        "user_123"
    )
    print(f"合约执行结果: {result}")
    
    # 获取网络状态
    network_status = blockchain.get_network_status()
    print(f"网络状态: {network_status}")
    
    # 获取审计轨迹
    audit_trail = blockchain.get_audit_trail()
    print(f"审计轨迹: {len(audit_trail)} 条记录")
    
    # 验证完整性
    integrity = blockchain.verify_integrity()
    print(f"区块链完整性: {integrity}") 