#!/usr/bin/env python3
"""
深度学习行为模拟测试
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import json
import time
from src.utils.deep_learning_behavior import (
    DeepLearningBehaviorSimulation, 
    BehaviorType, 
    BehaviorContext
)


def test_deep_learning_behavior():
    """测试深度学习行为模拟"""
    print("🧪 开始测试深度学习行为模拟...")
    
    # 创建深度学习行为模拟器
    dl_simulator = DeepLearningBehaviorSimulation()
    
    # 测试不同的用户类型和时间
    test_cases = [
        {
            "name": "专家用户-上午",
            "context": BehaviorContext(
                user_type="expert",
                time_of_day=0.3,  # 上午
                session_duration=1800,
                page_type="ticket_booking",
                device_type="desktop",
                network_speed=50.0
            )
        },
        {
            "name": "新手用户-下午",
            "context": BehaviorContext(
                user_type="beginner",
                time_of_day=0.7,  # 下午
                session_duration=900,
                page_type="ticket_booking",
                device_type="mobile",
                network_speed=5.0
            )
        },
        {
            "name": "老年用户-晚上",
            "context": BehaviorContext(
                user_type="elderly",
                time_of_day=0.9,  # 晚上
                session_duration=3600,
                page_type="ticket_booking",
                device_type="tablet",
                network_speed=10.0
            )
        }
    ]
    
    # 测试所有行为类型
    behavior_types = [
        BehaviorType.MOUSE_MOVEMENT,
        BehaviorType.KEYBOARD_INPUT,
        BehaviorType.SCROLL_BEHAVIOR,
        BehaviorType.CLICK_PATTERN,
        BehaviorType.NAVIGATION_PATTERN
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n📋 测试用例: {test_case['name']}")
        context = test_case['context']
        
        case_results = {}
        
        for behavior_type in behavior_types:
            print(f"  🔄 测试行为类型: {behavior_type.value}")
            
            try:
                # 生成行为
                start_time = time.time()
                behavior = dl_simulator.generate_natural_behavior(behavior_type, context)
                end_time = time.time()
                
                # 验证行为数据
                is_valid = validate_behavior(behavior, behavior_type)
                
                case_results[behavior_type.value] = {
                    "success": True,
                    "generation_time": end_time - start_time,
                    "is_valid": is_valid,
                    "behavior_data": behavior
                }
                
                print(f"    ✅ 生成成功 (耗时: {end_time - start_time:.3f}s)")
                
            except Exception as e:
                case_results[behavior_type.value] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"    ❌ 生成失败: {e}")
        
        results[test_case['name']] = case_results
    
    # 输出详细结果
    print("\n📊 测试结果总结:")
    print("=" * 60)
    
    for test_name, case_results in results.items():
        print(f"\n🔍 {test_name}:")
        for behavior_type, result in case_results.items():
            if result.get("success"):
                print(f"  ✅ {behavior_type}: 成功 (耗时: {result['generation_time']:.3f}s, 有效: {result['is_valid']})")
            else:
                print(f"  ❌ {behavior_type}: 失败 - {result.get('error', '未知错误')}")
    
    # 统计成功率
    total_tests = len(test_cases) * len(behavior_types)
    successful_tests = sum(
        1 for case_results in results.values()
        for result in case_results.values()
        if result.get("success")
    )
    
    success_rate = (successful_tests / total_tests) * 100
    print(f"\n📈 总体成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    
    # 保存详细结果到文件
    with open("test_results_deep_learning.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 详细结果已保存到: test_results_deep_learning.json")
    
    return success_rate >= 80.0


def validate_behavior(behavior: dict, behavior_type: BehaviorType) -> bool:
    """验证行为数据的有效性"""
    
    if not isinstance(behavior, dict):
        return False
    
    if behavior.get("type") != behavior_type.value:
        return False
    
    # 根据行为类型验证特定字段
    if behavior_type == BehaviorType.MOUSE_MOVEMENT:
        required_fields = ["trajectory", "speed", "acceleration", "pauses"]
        return all(field in behavior for field in required_fields)
    
    elif behavior_type == BehaviorType.KEYBOARD_INPUT:
        required_fields = ["typing_speed", "pause_patterns", "error_rate", "correction_patterns"]
        return all(field in behavior for field in required_fields)
    
    elif behavior_type == BehaviorType.SCROLL_BEHAVIOR:
        required_fields = ["speed", "pattern", "direction", "pause_duration"]
        return all(field in behavior for field in required_fields)
    
    elif behavior_type == BehaviorType.CLICK_PATTERN:
        required_fields = ["click_speed", "double_click_timing", "click_accuracy", "hover_duration"]
        return all(field in behavior for field in required_fields)
    
    elif behavior_type == BehaviorType.NAVIGATION_PATTERN:
        required_fields = ["page_load_time", "navigation_speed", "back_forward_usage", "bookmark_usage"]
        return all(field in behavior for field in required_fields)
    
    return True


def test_gan_generation():
    """测试GAN生成功能"""
    print("\n🎨 测试GAN行为生成...")
    
    try:
        import torch
        from src.utils.deep_learning_behavior import GANBehaviorGenerator
        
        gan = GANBehaviorGenerator()
        
        # 生成随机噪声
        noise = torch.randn(5, 100)
        
        # 生成行为数据
        generated_data = gan.generate_behavior(noise)
        
        print(f"  ✅ GAN生成成功，数据形状: {generated_data.shape}")
        
        # 测试训练步骤
        real_data = torch.randn(5, 2048)
        d_loss, g_loss = gan.train_step(real_data)
        
        print(f"  ✅ GAN训练成功，判别器损失: {d_loss:.4f}, 生成器损失: {g_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GAN测试失败: {e}")
        return False


def test_reinforcement_learning():
    """测试强化学习优化"""
    print("\n🤖 测试强化学习优化...")
    
    try:
        from src.utils.deep_learning_behavior import ReinforcementLearningOptimizer
        
        rl_optimizer = ReinforcementLearningOptimizer(10, 5)
        
        # 测试动作选择
        state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        action = rl_optimizer.act(state)
        
        print(f"  ✅ 强化学习动作选择成功，选择动作: {action}")
        
        # 测试经验回放
        for i in range(10):
            rl_optimizer.remember(
                state=[random.random() for _ in range(10)],
                action=random.randint(0, 4),
                reward=random.random(),
                next_state=[random.random() for _ in range(10)],
                done=random.choice([True, False])
            )
        
        rl_optimizer.replay()
        print(f"  ✅ 强化学习经验回放成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 强化学习测试失败: {e}")
        return False


def test_model_save_load():
    """测试模型保存和加载"""
    print("\n💾 测试模型保存和加载...")
    
    try:
        dl_simulator = DeepLearningBehaviorSimulation()
        
        # 保存模型
        dl_simulator.save_models("test_models.pth")
        print("  ✅ 模型保存成功")
        
        # 创建新的模拟器并加载模型
        new_simulator = DeepLearningBehaviorSimulation()
        new_simulator.load_models("test_models.pth")
        print("  ✅ 模型加载成功")
        
        # 清理测试文件
        import os
        if os.path.exists("test_models.pth"):
            os.remove("test_models.pth")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型保存/加载测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始深度学习行为模拟全面测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("深度学习行为模拟", test_deep_learning_behavior),
        ("GAN生成功能", test_gan_generation),
        ("强化学习优化", test_reinforcement_learning),
        ("模型保存加载", test_model_save_load)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results[test_name] = False
    
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
    
    if success_rate >= 75.0:
        print("🎉 测试通过！深度学习行为模拟功能正常工作")
        return True
    else:
        print("⚠️  测试未完全通过，需要进一步优化")
        return False


if __name__ == "__main__":
    import random
    main() 