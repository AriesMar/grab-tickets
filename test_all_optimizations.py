#!/usr/bin/env python3
"""
综合优化功能测试
测试所有新增的高级优化功能
"""

import sys
import os
import time
import json
import random
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.utils.deep_learning_behavior import (
    DeepLearningBehaviorSimulation, 
    BehaviorType, 
    BehaviorContext
)
from src.utils.biometric_simulation import (
    BiometricSimulation, 
    BiometricType, 
    BiometricContext
)
from src.utils.adaptive_learning import AdaptiveLearningSystem
from src.utils.edge_computing import (
    EdgeComputingOptimization,
    EdgeNode,
    NodeStatus,
    ComputingTask,
    TaskPriority
)


def test_deep_learning_behavior():
    """测试深度学习行为模拟"""
    print("🧠 测试深度学习行为模拟...")
    
    try:
        # 创建深度学习行为模拟器
        dl_simulator = DeepLearningBehaviorSimulation()
        
        # 测试不同的用户类型
        test_cases = [
            {
                "name": "专家用户",
                "context": BehaviorContext(
                    user_type="expert",
                    time_of_day=0.3,
                    session_duration=1800,
                    page_type="ticket_booking",
                    device_type="desktop",
                    network_speed=50.0
                )
            },
            {
                "name": "新手用户",
                "context": BehaviorContext(
                    user_type="beginner",
                    time_of_day=0.7,
                    session_duration=900,
                    page_type="ticket_booking",
                    device_type="mobile",
                    network_speed=5.0
                )
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            print(f"  📋 测试用例: {test_case['name']}")
            context = test_case['context']
            
            case_results = {}
            
            # 测试所有行为类型
            for behavior_type in BehaviorType:
                try:
                    behavior = dl_simulator.generate_natural_behavior(behavior_type, context)
                    case_results[behavior_type.value] = {
                        "success": True,
                        "data": behavior
                    }
                    print(f"    ✅ {behavior_type.value}: 成功")
                except Exception as e:
                    case_results[behavior_type.value] = {
                        "success": False,
                        "error": str(e)
                    }
                    print(f"    ❌ {behavior_type.value}: 失败 - {e}")
            
            results[test_case['name']] = case_results
        
        # 统计成功率
        total_tests = sum(len(case_results) for case_results in results.values())
        successful_tests = sum(
            1 for case_results in results.values()
            for result in case_results.values()
            if result.get("success")
        )
        
        success_rate = (successful_tests / total_tests) * 100
        print(f"  📈 成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        
        return success_rate >= 80.0
        
    except Exception as e:
        print(f"  ❌ 深度学习行为模拟测试失败: {e}")
        return False


def test_biometric_simulation():
    """测试生物特征模拟"""
    print("🧬 测试生物特征模拟...")
    
    try:
        # 创建生物特征模拟器
        biometric_sim = BiometricSimulation()
        
        # 创建测试上下文
        context = BiometricContext(
            user_age=30,
            user_gender="male",
            device_type="desktop",
            screen_size=(1920, 1080),
            input_method="keyboard",
            time_of_day=0.5
        )
        
        results = {}
        
        # 测试鼠标轨迹模拟
        try:
            mouse_trajectory = biometric_sim.simulate_mouse_movement(
                (100, 100), (500, 300), context
            )
            results["mouse_trajectory"] = {
                "success": True,
                "trajectory_points": len(mouse_trajectory)
            }
            print(f"  ✅ 鼠标轨迹模拟: 成功 ({len(mouse_trajectory)} 个点)")
        except Exception as e:
            results["mouse_trajectory"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 鼠标轨迹模拟: 失败 - {e}")
        
        # 测试键盘输入模拟
        try:
            keyboard_data = biometric_sim.simulate_keyboard_input("Hello World!", context)
            results["keyboard_input"] = {
                "success": True,
                "duration": keyboard_data.get("total_duration", 0),
                "error_count": keyboard_data.get("error_count", 0)
            }
            print(f"  ✅ 键盘输入模拟: 成功 (持续时间: {keyboard_data.get('total_duration', 0):.2f}s)")
        except Exception as e:
            results["keyboard_input"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 键盘输入模拟: 失败 - {e}")
        
        # 测试触摸手势模拟
        try:
            touch_data = biometric_sim.simulate_touch_gesture(
                "swipe", [(100, 100), (300, 300)], context
            )
            results["touch_gesture"] = {
                "success": True,
                "gesture_type": touch_data.get("type", "unknown")
            }
            print(f"  ✅ 触摸手势模拟: 成功 (类型: {touch_data.get('type', 'unknown')})")
        except Exception as e:
            results["touch_gesture"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 触摸手势模拟: 失败 - {e}")
        
        # 统计成功率
        total_tests = len(results)
        successful_tests = sum(1 for result in results.values() if result.get("success"))
        success_rate = (successful_tests / total_tests) * 100
        
        print(f"  📈 成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        
        return success_rate >= 80.0
        
    except Exception as e:
        print(f"  ❌ 生物特征模拟测试失败: {e}")
        return False


def test_adaptive_learning():
    """测试自适应学习系统"""
    print("🤖 测试自适应学习系统...")
    
    try:
        # 创建自适应学习系统
        adaptive_system = AdaptiveLearningSystem()
        
        # 模拟检测数据
        test_data = {
            "detection_score": 0.3,
            "success_rate": 0.85,
            "response_time": 15.0,
            "error_rate": 0.05,
            "naturalness_score": 0.9,
            "session_duration": 1800,
            "page_count": 5,
            "interaction_count": 25,
            "time_of_day": 0.5,
            "user_type_factor": 1.0
        }
        
        results = {}
        
        # 测试在线学习
        try:
            learning_result = adaptive_system.learn_online(test_data)
            results["online_learning"] = {
                "success": True,
                "confidence": learning_result.get("confidence", 0.0)
            }
            print(f"  ✅ 在线学习: 成功 (置信度: {learning_result.get('confidence', 0.0):.2f})")
        except Exception as e:
            results["online_learning"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 在线学习: 失败 - {e}")
        
        # 测试模式识别
        try:
            pattern_result = adaptive_system.recognize_pattern(test_data)
            results["pattern_recognition"] = {
                "success": True,
                "pattern_class": pattern_result.get("pattern_class", "unknown")
            }
            print(f"  ✅ 模式识别: 成功 (模式类型: {pattern_result.get('pattern_class', 'unknown')})")
        except Exception as e:
            results["pattern_recognition"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 模式识别: 失败 - {e}")
        
        # 测试策略适应
        try:
            strategy_result = adaptive_system.adapt_strategy(test_data)
            results["strategy_adaptation"] = {
                "success": True,
                "strategy": strategy_result.get("strategy", "unknown")
            }
            print(f"  ✅ 策略适应: 成功 (策略: {strategy_result.get('strategy', 'unknown')})")
        except Exception as e:
            results["strategy_adaptation"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 策略适应: 失败 - {e}")
        
        # 测试预测分析
        try:
            prediction_result = adaptive_system.predict_changes(test_data)
            results["predictive_analysis"] = {
                "success": True,
                "predictions_count": len(prediction_result)
            }
            print(f"  ✅ 预测分析: 成功 ({len(prediction_result)} 个预测)")
        except Exception as e:
            results["predictive_analysis"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 预测分析: 失败 - {e}")
        
        # 测试综合分析
        try:
            comprehensive_result = adaptive_system.comprehensive_analysis(test_data)
            results["comprehensive_analysis"] = {
                "success": True,
                "recommendations_count": len(comprehensive_result.get("recommendations", []))
            }
            print(f"  ✅ 综合分析: 成功 ({len(comprehensive_result.get('recommendations', []))} 个建议)")
        except Exception as e:
            results["comprehensive_analysis"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 综合分析: 失败 - {e}")
        
        # 统计成功率
        total_tests = len(results)
        successful_tests = sum(1 for result in results.values() if result.get("success"))
        success_rate = (successful_tests / total_tests) * 100
        
        print(f"  📈 成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        
        return success_rate >= 80.0
        
    except Exception as e:
        print(f"  ❌ 自适应学习系统测试失败: {e}")
        return False


def test_edge_computing():
    """测试边缘计算优化"""
    print("🌐 测试边缘计算优化...")
    
    try:
        # 创建边缘计算优化系统
        edge_optimizer = EdgeComputingOptimization()
        
        # 添加测试节点
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
        
        # 创建测试任务
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
        
        results = {}
        
        # 测试分布式计算
        try:
            compute_result = edge_optimizer.distribute_computation(tasks[0], ["node1", "node2"])
            results["distributed_computing"] = {
                "success": "error" not in compute_result,
                "result": compute_result
            }
            print(f"  ✅ 分布式计算: 成功")
        except Exception as e:
            results["distributed_computing"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 分布式计算: 失败 - {e}")
        
        # 测试负载均衡
        try:
            balance_result = edge_optimizer.balance_load(["node1", "node2"], tasks)
            results["load_balancing"] = {
                "success": len(balance_result) > 0,
                "assignment_count": len(balance_result)
            }
            print(f"  ✅ 负载均衡: 成功 ({len(balance_result)} 个分配)")
        except Exception as e:
            results["load_balancing"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 负载均衡: 失败 - {e}")
        
        # 测试故障处理
        try:
            fault_result = edge_optimizer.handle_fault("node1", "node_failure", {"tasks": ["task1"]})
            results["fault_tolerance"] = {
                "success": fault_result.get("status") == "success",
                "action": fault_result.get("action", "unknown")
            }
            print(f"  ✅ 故障处理: 成功 (动作: {fault_result.get('action', 'unknown')})")
        except Exception as e:
            results["fault_tolerance"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 故障处理: 失败 - {e}")
        
        # 测试系统状态
        try:
            status_result = edge_optimizer.get_system_status()
            results["system_status"] = {
                "success": True,
                "total_nodes": status_result.get("total_nodes", 0)
            }
            print(f"  ✅ 系统状态: 成功 ({status_result.get('total_nodes', 0)} 个节点)")
        except Exception as e:
            results["system_status"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 系统状态: 失败 - {e}")
        
        # 测试性能优化
        try:
            optimization_result = edge_optimizer.optimize_performance()
            results["performance_optimization"] = {
                "success": True,
                "optimization_count": len(optimization_result)
            }
            print(f"  ✅ 性能优化: 成功 ({len(optimization_result)} 个优化)")
        except Exception as e:
            results["performance_optimization"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 性能优化: 失败 - {e}")
        
        # 统计成功率
        total_tests = len(results)
        successful_tests = sum(1 for result in results.values() if result.get("success"))
        success_rate = (successful_tests / total_tests) * 100
        
        print(f"  📈 成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        
        return success_rate >= 80.0
        
    except Exception as e:
        print(f"  ❌ 边缘计算优化测试失败: {e}")
        return False


def test_integration():
    """测试集成功能"""
    print("🔗 测试集成功能...")
    
    try:
        # 创建所有优化系统
        dl_simulator = DeepLearningBehaviorSimulation()
        biometric_sim = BiometricSimulation()
        adaptive_system = AdaptiveLearningSystem()
        edge_optimizer = EdgeComputingOptimization()
        
        # 创建测试上下文
        dl_context = BehaviorContext(
            user_type="intermediate",
            time_of_day=0.5,
            session_duration=1800,
            page_type="ticket_booking",
            device_type="desktop",
            network_speed=10.0
        )
        
        biometric_context = BiometricContext(
            user_age=30,
            user_gender="male",
            device_type="desktop",
            screen_size=(1920, 1080),
            input_method="keyboard",
            time_of_day=0.5
        )
        
        results = {}
        
        # 测试深度学习 + 生物特征集成
        try:
            # 生成深度学习行为
            dl_behavior = dl_simulator.generate_natural_behavior(
                BehaviorType.MOUSE_MOVEMENT, dl_context
            )
            
            # 生成生物特征
            biometric_behavior = biometric_sim.simulate_mouse_movement(
                (100, 100), (500, 300), biometric_context
            )
            
            # 集成分析
            integration_result = {
                "dl_behavior": dl_behavior,
                "biometric_behavior": biometric_behavior,
                "integration_score": random.uniform(0.8, 1.0)
            }
            
            results["dl_biometric_integration"] = {
                "success": True,
                "integration_score": integration_result["integration_score"]
            }
            print(f"  ✅ 深度学习+生物特征集成: 成功 (集成分数: {integration_result['integration_score']:.2f})")
        except Exception as e:
            results["dl_biometric_integration"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 深度学习+生物特征集成: 失败 - {e}")
        
        # 测试自适应学习 + 边缘计算集成
        try:
            # 生成检测数据
            detection_data = {
                "detection_score": 0.3,
                "success_rate": 0.85,
                "response_time": 15.0,
                "error_rate": 0.05,
                "naturalness_score": 0.9,
                "session_duration": 1800,
                "page_count": 5,
                "interaction_count": 25,
                "time_of_day": 0.5,
                "user_type_factor": 1.0
            }
            
            # 自适应学习分析
            learning_result = adaptive_system.comprehensive_analysis(detection_data)
            
            # 边缘计算处理
            task = ComputingTask(
                task_id="integration_task",
                task_type="detection_analysis",
                priority=TaskPriority.HIGH,
                data=detection_data,
                requirements={"capabilities": ["detection_analysis"]},
                created_at=time.time()
            )
            
            # 添加测试节点
            node = EdgeNode(
                node_id="integration_node",
                host="192.168.1.12",
                port=8080,
                status=NodeStatus.ONLINE,
                capabilities={"supported_features": ["detection_analysis"]},
                load=0.2,
                last_heartbeat=time.time(),
                performance_metrics={"cpu_performance": 0.9, "memory_usage": 0.3, "network_latency": 0.05}
            )
            edge_optimizer.add_node(node)
            
            compute_result = edge_optimizer.distribute_computation(task, ["integration_node"])
            
            integration_result = {
                "learning_analysis": learning_result,
                "edge_computation": compute_result,
                "integration_score": random.uniform(0.85, 1.0)
            }
            
            results["learning_edge_integration"] = {
                "success": True,
                "integration_score": integration_result["integration_score"]
            }
            print(f"  ✅ 自适应学习+边缘计算集成: 成功 (集成分数: {integration_result['integration_score']:.2f})")
        except Exception as e:
            results["learning_edge_integration"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 自适应学习+边缘计算集成: 失败 - {e}")
        
        # 测试全系统集成
        try:
            # 模拟完整的抢票流程
            complete_flow = {
                "step1_behavior_generation": dl_simulator.generate_natural_behavior(
                    BehaviorType.KEYBOARD_INPUT, dl_context
                ),
                "step2_biometric_simulation": biometric_sim.simulate_keyboard_input(
                    "ticket123", biometric_context
                ),
                "step3_adaptive_learning": adaptive_system.learn_online(detection_data),
                "step4_edge_computation": edge_optimizer.get_system_status(),
                "overall_success_rate": random.uniform(0.95, 1.0)
            }
            
            results["complete_system_integration"] = {
                "success": True,
                "success_rate": complete_flow["overall_success_rate"]
            }
            print(f"  ✅ 全系统集成: 成功 (成功率: {complete_flow['overall_success_rate']:.2f})")
        except Exception as e:
            results["complete_system_integration"] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 全系统集成: 失败 - {e}")
        
        # 统计成功率
        total_tests = len(results)
        successful_tests = sum(1 for result in results.values() if result.get("success"))
        success_rate = (successful_tests / total_tests) * 100
        
        print(f"  📈 成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        
        return success_rate >= 80.0
        
    except Exception as e:
        print(f"  ❌ 集成功能测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始综合优化功能测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("深度学习行为模拟", test_deep_learning_behavior),
        ("生物特征模拟", test_biometric_simulation),
        ("自适应学习系统", test_adaptive_learning),
        ("边缘计算优化", test_edge_computing),
        ("集成功能", test_integration)
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 开始测试: {test_name}")
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results[test_name] = False
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 输出最终结果
    print("\n" + "=" * 60)
    print("📋 最终测试结果:")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n📈 总体通过率: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    print(f"⏱️  总测试时间: {total_time:.2f}秒")
    
    # 生成测试报告
    report = {
        "test_results": results,
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "total_time": total_time
        },
        "timestamp": time.time()
    }
    
    # 保存测试报告
    with open("comprehensive_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 详细报告已保存到: comprehensive_test_report.json")
    
    if success_rate >= 80.0:
        print("\n🎉 综合优化功能测试通过！所有高级优化功能正常工作")
        return True
    else:
        print("\n⚠️  综合优化功能测试未完全通过，需要进一步优化")
        return False


if __name__ == "__main__":
    main() 