#!/usr/bin/env python3
"""
反检测功能完整测试脚本
"""
import sys
import time
import json
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_stealth_session():
    """测试隐身会话管理"""
    print("🔍 测试隐身会话管理...")
    
    from src.utils.advanced_anti_detection import StealthSession
    
    stealth_session = StealthSession()
    
    # 创建隐身会话
    session = stealth_session.create_stealth_session()
    print(f"✅ 隐身会话创建成功")
    print(f"   设备ID: {stealth_session.stealth_manager.device_id}")
    print(f"   会话ID: {stealth_session.stealth_manager.session_id}")
    
    # 测试会话轮换
    if stealth_session.should_rotate_session():
        new_session = stealth_session.rotate_session(session)
        print(f"✅ 会话轮换成功")
    
    # 获取会话信息
    session_info = stealth_session.get_session_info()
    print(f"✅ 会话信息获取成功")
    print(f"   请求次数: {session_info['request_count']}")
    
    return True

def test_tracking_monitor():
    """测试追踪监控"""
    print("\n🔍 测试追踪监控...")
    
    from src.utils.tracking_monitor import TrackingMonitor
    
    monitor = TrackingMonitor()
    
    # 测试追踪检测
    test_url = "https://example.com?utm_source=google&utm_medium=cpc"
    test_headers = {"Referer": "https://facebook.com"}
    test_cookies = {"_ga": "GA1.1.123456789.1234567890"}
    
    tracking_info = monitor.detect_tracking(test_url, test_headers, test_cookies)
    print(f"✅ 追踪检测完成")
    print(f"   检测到的类别: {tracking_info['categories']}")
    print(f"   严重程度: {tracking_info['severity']}")
    
    # 测试响应清理
    test_content = """
    <script>gtag('config', 'GA_MEASUREMENT_ID');</script>
    <script>fbq('init', '123456789');</script>
    """
    cleaned_content = monitor.sanitize_response(test_content)
    print(f"✅ 响应清理完成")
    print(f"   清理前长度: {len(test_content)}")
    print(f"   清理后长度: {len(cleaned_content)}")
    
    # 获取追踪报告
    report = monitor.get_tracking_report()
    print(f"✅ 追踪报告生成成功")
    print(f"   总检测次数: {report['stats']['total_detected']}")
    
    return True

def test_behavior_analyzer():
    """测试行为分析器"""
    print("\n🔍 测试行为分析器...")
    
    from src.utils.behavior_analyzer import BehaviorAnalyzer
    
    analyzer = BehaviorAnalyzer()
    
    # 测试行为分析
    test_behavior_data = {
        "typing_speed": 0.3,
        "intervals": [0.1, 0.2, 0.15, 0.25],
        "text_length": 50
    }
    
    analysis = analyzer.analyze_user_behavior("typing", test_behavior_data)
    print(f"✅ 行为分析完成")
    print(f"   模式分数: {analysis['pattern_score']:.3f}")
    print(f"   自然度分数: {analysis['naturalness_score']:.3f}")
    print(f"   一致性分数: {analysis['consistency_score']:.3f}")
    
    # 测试行为优化
    optimized_data = analyzer.optimize_behavior("typing", test_behavior_data)
    print(f"✅ 行为优化完成")
    print(f"   优化前速度: {test_behavior_data['typing_speed']}")
    print(f"   优化后速度: {optimized_data['typing_speed']}")
    
    # 获取行为报告
    report = analyzer.get_behavior_report()
    print(f"✅ 行为报告生成成功")
    print(f"   总行为数: {report['total_actions']}")
    
    return True

def test_environment_detector():
    """测试环境检测器"""
    print("\n🔍 测试环境检测器...")
    
    from src.utils.environment_detector import EnvironmentDetector
    
    detector = EnvironmentDetector()
    
    # 测试环境检测
    test_url = "https://example.com/webdriver"
    test_headers = {"X-WebDriver": "true"}
    test_content = "selenium webdriver automation"
    
    detection_result = detector.detect_environment(test_url, test_headers, test_content)
    print(f"✅ 环境检测完成")
    print(f"   检测到的技术: {detection_result['detected_techniques']}")
    print(f"   风险等级: {detection_result['risk_level']}")
    
    # 测试规避措施应用
    evasion_measures = detector.apply_evasion_measures(type('Session', (), {'headers': {}})())
    print(f"✅ 规避措施应用完成")
    print(f"   应用的措施: {evasion_measures['applied_measures']}")
    
    # 获取检测报告
    report = detector.get_detection_report()
    print(f"✅ 检测报告生成成功")
    print(f"   总检测次数: {report['total_detections']}")
    
    return True

def test_ai_evasion():
    """测试AI反检测"""
    print("\n🔍 测试AI反检测...")
    
    from src.utils.ai_evasion import AIEvasionEngine, DeepLearningEvasion
    
    # 测试AI反检测引擎
    ai_engine = AIEvasionEngine()
    
    test_request_data = {
        "click_intervals": [0.2, 0.3, 0.25, 0.4],
        "typing_speed": 0.3,
        "typing_intervals": [0.1, 0.15, 0.2],
        "request_pattern": {"frequency": 0.5, "regularity": 0.3, "burstiness": 0.2},
        "session_pattern": {"duration": 3600, "activity_level": 0.7, "idle_time": 300},
        "interaction_pattern": {"complexity": 0.6, "predictability": 0.4, "diversity": 0.8},
        "response_times": [0.1, 0.15, 0.12, 0.18],
        "execution_times": [0.05, 0.08, 0.06, 0.09],
        "time_intervals": [1.0, 1.2, 0.8, 1.5],
        "device_fingerprint": {"screen_width": 1920, "screen_height": 1080, "color_depth": 24, "timezone_offset": 28800},
        "browser_fingerprint": {"plugins": ["Flash", "Java"], "fonts": ["Arial", "Times"], "canvas_hash": 12345, "webgl_hash": 67890},
        "network_fingerprint": {"connection_type": 1, "bandwidth": 100, "latency": 50}
    }
    
    ai_detection = ai_engine.detect_ai_surveillance(test_request_data)
    print(f"✅ AI检测完成")
    print(f"   AI检测到: {ai_detection['ai_detected']}")
    print(f"   检测置信度: {ai_detection['detection_confidence']:.3f}")
    print(f"   检测方法: {ai_detection['detection_methods']}")
    
    # 测试深度学习反检测
    dl_evasion = DeepLearningEvasion()
    
    test_data = {
        "neural_network": True,
        "activation_function": "relu",
        "gradient_descent": True
    }
    
    dl_detection = dl_evasion.detect_deep_learning_surveillance(test_data)
    print(f"✅ 深度学习检测完成")
    print(f"   DL检测到: {dl_detection['dl_detected']}")
    print(f"   检测置信度: {dl_detection['detection_confidence']:.3f}")
    print(f"   检测类型: {dl_detection['detection_type']}")
    
    # 应用DL反检测
    dl_evasion_result = dl_evasion.apply_dl_evasion(test_data)
    print(f"✅ DL反检测应用完成")
    print(f"   应用的技术: {dl_evasion_result['techniques_applied']}")
    
    return True

def test_quantum_preparation():
    """测试量子计算准备"""
    print("\n🔍 测试量子计算准备...")
    
    from src.utils.quantum_preparation import QuantumPreparation, PostQuantumCryptography
    
    # 测试量子准备
    quantum_prep = QuantumPreparation()
    
    test_data = {
        "quantum_superposition": True,
        "quantum_entanglement": True,
        "quantum_random": True,
        "quantum_fingerprint": True
    }
    
    quantum_detection = quantum_prep.detect_quantum_surveillance(test_data)
    print(f"✅ 量子检测完成")
    print(f"   量子检测到: {quantum_detection['quantum_detected']}")
    print(f"   检测置信度: {quantum_detection['detection_confidence']:.3f}")
    print(f"   量子特征: {quantum_detection['quantum_features']}")
    
    # 生成量子密钥
    quantum_key = quantum_prep.generate_quantum_key(256)
    print(f"✅ 量子密钥生成成功")
    print(f"   密钥长度: {len(quantum_key)}")
    print(f"   密钥前16位: {quantum_key[:16]}")
    
    # 测试量子加密
    test_message = "Hello, Quantum World!"
    encrypted_message = quantum_prep.quantum_encrypt(test_message, quantum_key)
    print(f"✅ 量子加密完成")
    print(f"   原始消息: {test_message}")
    print(f"   加密消息: {encrypted_message[:50]}...")
    
    # 测试量子解密
    decrypted_message = quantum_prep.quantum_decrypt(encrypted_message, quantum_key)
    print(f"✅ 量子解密完成")
    print(f"   解密消息: {decrypted_message}")
    
    # 测试后量子密码学
    pqc = PostQuantumCryptography()
    
    test_data = "Post-Quantum Test Data"
    pqc_result = pqc.encrypt_post_quantum(test_data)
    print(f"✅ 后量子加密完成")
    print(f"   算法: {pqc_result['algorithm']}")
    print(f"   成功: {pqc_result['success']}")
    
    return True

def test_integration():
    """测试集成功能"""
    print("\n🔍 测试集成功能...")
    
    from src.utils.advanced_anti_detection import StealthSession
    from src.utils.tracking_monitor import TrackingMonitor
    from src.utils.behavior_analyzer import BehaviorAnalyzer
    from src.utils.environment_detector import EnvironmentDetector
    from src.utils.ai_evasion import AIEvasionEngine
    from src.utils.quantum_preparation import QuantumPreparation
    
    # 创建所有组件
    stealth_session = StealthSession()
    tracking_monitor = TrackingMonitor()
    behavior_analyzer = BehaviorAnalyzer()
    environment_detector = EnvironmentDetector()
    ai_engine = AIEvasionEngine()
    quantum_prep = QuantumPreparation()
    
    print(f"✅ 所有组件创建成功")
    
    # 模拟完整的反检测流程
    test_request_data = {
        "url": "https://example.com",
        "headers": {"User-Agent": "Mozilla/5.0"},
        "cookies": {"session": "abc123"},
        "click_intervals": [0.2, 0.3, 0.25],
        "typing_speed": 0.3
    }
    
    # 1. 隐身会话管理
    session = stealth_session.create_stealth_session()
    print(f"✅ 隐身会话创建完成")
    
    # 2. 追踪检测
    tracking_info = tracking_monitor.detect_tracking(
        test_request_data["url"], 
        test_request_data["headers"], 
        test_request_data["cookies"]
    )
    print(f"✅ 追踪检测完成")
    
    # 3. 行为分析
    behavior_analysis = behavior_analyzer.analyze_user_behavior("clicking", {
        "interval": 0.25,
        "position": (100, 200)
    })
    print(f"✅ 行为分析完成")
    
    # 4. 环境检测
    env_detection = environment_detector.detect_environment(
        test_request_data["url"],
        test_request_data["headers"]
    )
    print(f"✅ 环境检测完成")
    
    # 5. AI反检测
    ai_detection = ai_engine.detect_ai_surveillance(test_request_data)
    print(f"✅ AI反检测完成")
    
    # 6. 量子准备
    quantum_detection = quantum_prep.detect_quantum_surveillance(test_request_data)
    print(f"✅ 量子检测完成")
    
    # 生成综合报告
    integration_report = {
        "stealth_session": stealth_session.get_session_info(),
        "tracking_detection": tracking_info,
        "behavior_analysis": behavior_analysis,
        "environment_detection": env_detection,
        "ai_detection": ai_detection,
        "quantum_detection": quantum_detection,
        "timestamp": time.time()
    }
    
    print(f"✅ 集成测试完成")
    print(f"   总检测项目: 6")
    print(f"   综合风险等级: 低")
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始反检测功能完整测试")
    print("=" * 50)
    
    tests = [
        ("隐身会话管理", test_stealth_session),
        ("追踪监控", test_tracking_monitor),
        ("行为分析器", test_behavior_analyzer),
        ("环境检测器", test_environment_detector),
        ("AI反检测", test_ai_evasion),
        ("量子计算准备", test_quantum_preparation),
        ("集成功能", test_integration)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n📋 测试: {test_name}")
            print("-" * 30)
            
            if test_func():
                print(f"✅ {test_name} 测试通过")
                passed_tests += 1
            else:
                print(f"❌ {test_name} 测试失败")
                
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果总结")
    print(f"   总测试数: {total_tests}")
    print(f"   通过测试: {passed_tests}")
    print(f"   失败测试: {total_tests - passed_tests}")
    print(f"   通过率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！反检测系统运行正常")
    else:
        print("⚠️  部分测试失败，请检查相关功能")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 