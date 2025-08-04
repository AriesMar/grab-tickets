"""
环境检测器 - 检测和规避各种检测技术
"""
import time
import random
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
from loguru import logger


class EnvironmentDetector:
    """环境检测器"""
    
    def __init__(self):
        self.logger = logger.bind(name="environment_detector")
        
        # 检测技术数据库
        self.detection_techniques = {
            "webdriver_detection": {
                "patterns": [
                    r"webdriver",
                    r"selenium",
                    r"phantom",
                    r"headless",
                    r"automation"
                ],
                "headers": [
                    "X-WebDriver",
                    "X-Selenium",
                    "X-Automation"
                ],
                "properties": [
                    "webdriver",
                    "selenium",
                    "phantom",
                    "headless"
                ]
            },
            "fingerprint_detection": {
                "patterns": [
                    r"fingerprint",
                    r"canvas",
                    r"webgl",
                    r"audio",
                    r"font",
                    r"plugin",
                    r"screen",
                    r"navigator"
                ],
                "apis": [
                    "getContext",
                    "getImageData",
                    "toDataURL",
                    "getParameter",
                    "getSupportedExtensions"
                ]
            },
            "behavior_detection": {
                "patterns": [
                    r"mouse",
                    r"keyboard",
                    r"scroll",
                    r"click",
                    r"typing",
                    r"movement"
                ],
                "events": [
                    "mousemove",
                    "mousedown",
                    "mouseup",
                    "keydown",
                    "keyup",
                    "scroll"
                ]
            },
            "timing_detection": {
                "patterns": [
                    r"timing",
                    r"performance",
                    r"speed",
                    r"interval",
                    r"delay"
                ],
                "metrics": [
                    "responseTime",
                    "executionTime",
                    "interactionTime",
                    "clickSpeed",
                    "typingSpeed"
                ]
            },
            "network_detection": {
                "patterns": [
                    r"proxy",
                    r"vpn",
                    r"tor",
                    r"tunnel",
                    r"anonymizer"
                ],
                "headers": [
                    "X-Forwarded-For",
                    "X-Real-IP",
                    "X-Client-IP",
                    "CF-Connecting-IP"
                ]
            }
        }
        
        # 检测结果
        self.detection_results = {}
        
        # 规避策略
        self.evasion_strategies = {
            "webdriver_detection": self._evade_webdriver_detection,
            "fingerprint_detection": self._evade_fingerprint_detection,
            "behavior_detection": self._evade_behavior_detection,
            "timing_detection": self._evade_timing_detection,
            "network_detection": self._evade_network_detection
        }
    
    def detect_environment(self, url: str, headers: Dict[str, str] = None, 
                          content: str = None, js_code: str = None) -> Dict[str, Any]:
        """检测环境中的检测技术"""
        detection_result = {
            "url": url,
            "timestamp": time.time(),
            "detected_techniques": [],
            "risk_level": "low",
            "evasion_applied": []
        }
        
        # 检测各种检测技术
        for technique_name, technique_config in self.detection_techniques.items():
            if self._detect_technique(technique_name, technique_config, url, headers, content, js_code):
                detection_result["detected_techniques"].append(technique_name)
        
        # 计算风险等级
        detection_result["risk_level"] = self._calculate_risk_level(detection_result["detected_techniques"])
        
        # 应用规避策略
        for technique in detection_result["detected_techniques"]:
            if technique in self.evasion_strategies:
                evasion_result = self.evasion_strategies[technique]()
                detection_result["evasion_applied"].append({
                    "technique": technique,
                    "strategy": evasion_result
                })
        
        # 记录检测结果
        self.detection_results[url] = detection_result
        
        return detection_result
    
    def _detect_technique(self, technique_name: str, technique_config: Dict[str, Any],
                         url: str, headers: Dict[str, str] = None, 
                         content: str = None, js_code: str = None) -> bool:
        """检测特定技术"""
        detected = False
        
        # 检查URL中的模式
        if "patterns" in technique_config:
            for pattern in technique_config["patterns"]:
                if re.search(pattern, url, re.IGNORECASE):
                    detected = True
                    break
        
        # 检查请求头
        if headers and "headers" in technique_config:
            for header_name in technique_config["headers"]:
                if header_name in headers:
                    detected = True
                    break
        
        # 检查内容中的模式
        if content and "patterns" in technique_config:
            for pattern in technique_config["patterns"]:
                if re.search(pattern, content, re.IGNORECASE):
                    detected = True
                    break
        
        # 检查JavaScript代码
        if js_code and "apis" in technique_config:
            for api in technique_config["apis"]:
                if api in js_code:
                    detected = True
                    break
        
        return detected
    
    def _calculate_risk_level(self, detected_techniques: List[str]) -> str:
        """计算风险等级"""
        risk_scores = {
            "webdriver_detection": 3,
            "fingerprint_detection": 2,
            "behavior_detection": 2,
            "timing_detection": 1,
            "network_detection": 1
        }
        
        total_score = sum(risk_scores.get(tech, 0) for tech in detected_techniques)
        
        if total_score >= 6:
            return "high"
        elif total_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _evade_webdriver_detection(self) -> Dict[str, Any]:
        """规避WebDriver检测"""
        evasion_strategy = {
            "type": "webdriver_evasion",
            "actions": [
                "移除WebDriver属性",
                "伪装浏览器环境",
                "添加随机延迟",
                "模拟真实用户行为"
            ],
            "implementation": {
                "remove_webdriver_property": True,
                "fake_browser_environment": True,
                "add_random_delays": True,
                "simulate_human_behavior": True
            }
        }
        
        self.logger.info("应用WebDriver检测规避策略")
        return evasion_strategy
    
    def _evade_fingerprint_detection(self) -> Dict[str, Any]:
        """规避指纹检测"""
        evasion_strategy = {
            "type": "fingerprint_evasion",
            "actions": [
                "随机化Canvas指纹",
                "随机化WebGL指纹",
                "随机化音频指纹",
                "随机化字体指纹",
                "随机化插件指纹"
            ],
            "implementation": {
                "randomize_canvas": True,
                "randomize_webgl": True,
                "randomize_audio": True,
                "randomize_fonts": True,
                "randomize_plugins": True
            }
        }
        
        self.logger.info("应用指纹检测规避策略")
        return evasion_strategy
    
    def _evade_behavior_detection(self) -> Dict[str, Any]:
        """规避行为检测"""
        evasion_strategy = {
            "type": "behavior_evasion",
            "actions": [
                "模拟真实鼠标移动",
                "添加随机点击",
                "模拟键盘输入模式",
                "添加滚动行为",
                "模拟思考时间"
            ],
            "implementation": {
                "simulate_mouse_movement": True,
                "add_random_clicks": True,
                "simulate_keyboard_patterns": True,
                "add_scroll_behavior": True,
                "simulate_thinking_time": True
            }
        }
        
        self.logger.info("应用行为检测规避策略")
        return evasion_strategy
    
    def _evade_timing_detection(self) -> Dict[str, Any]:
        """规避时间检测"""
        evasion_strategy = {
            "type": "timing_evasion",
            "actions": [
                "随机化响应时间",
                "添加执行延迟",
                "模拟网络延迟",
                "随机化交互时间"
            ],
            "implementation": {
                "randomize_response_time": True,
                "add_execution_delay": True,
                "simulate_network_delay": True,
                "randomize_interaction_time": True
            }
        }
        
        self.logger.info("应用时间检测规避策略")
        return evasion_strategy
    
    def _evade_network_detection(self) -> Dict[str, Any]:
        """规避网络检测"""
        evasion_strategy = {
            "type": "network_evasion",
            "actions": [
                "隐藏代理特征",
                "随机化IP地址",
                "伪装网络参数",
                "模拟真实网络环境"
            ],
            "implementation": {
                "hide_proxy_signatures": True,
                "randomize_ip_address": True,
                "disguise_network_params": True,
                "simulate_real_network": True
            }
        }
        
        self.logger.info("应用网络检测规避策略")
        return evasion_strategy
    
    def apply_evasion_measures(self, session: Any) -> Dict[str, Any]:
        """应用规避措施"""
        evasion_measures = {
            "applied_measures": [],
            "session_modified": False,
            "headers_modified": False,
            "cookies_modified": False
        }
        
        # 应用WebDriver规避
        if hasattr(session, 'headers'):
            # 移除WebDriver相关头部
            webdriver_headers = ["X-WebDriver", "X-Selenium", "X-Automation"]
            for header in webdriver_headers:
                if header in session.headers:
                    del session.headers[header]
                    evasion_measures["headers_modified"] = True
                    evasion_measures["applied_measures"].append(f"移除头部: {header}")
        
        # 添加伪装头部
        disguise_headers = {
            "X-Requested-With": "XMLHttpRequest",
            "X-Real-IP": self._generate_random_ip(),
            "X-Forwarded-For": self._generate_random_ip(),
            "DNT": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none"
        }
        
        for header, value in disguise_headers.items():
            session.headers[header] = value
            evasion_measures["headers_modified"] = True
            evasion_measures["applied_measures"].append(f"添加头部: {header}")
        
        return evasion_measures
    
    def _generate_random_ip(self) -> str:
        """生成随机IP地址"""
        # 生成私有IP地址
        private_ranges = [
            (10, 0, 0, 0, 10, 255, 255, 255),
            (172, 16, 0, 0, 172, 31, 255, 255),
            (192, 168, 0, 0, 192, 168, 255, 255)
        ]
        
        range_choice = random.choice(private_ranges)
        ip_parts = []
        for i in range(4):
            start = range_choice[i * 2]
            end = range_choice[i * 2 + 1]
            # 确保start <= end
            if start > end:
                start, end = end, start
            ip_parts.append(str(random.randint(start, end)))
        
        return ".".join(ip_parts)
    
    def get_detection_report(self) -> Dict[str, Any]:
        """获取检测报告"""
        if not self.detection_results:
            return {"message": "暂无检测数据"}
        
        # 统计检测结果
        total_detections = len(self.detection_results)
        high_risk_count = sum(1 for r in self.detection_results.values() if r["risk_level"] == "high")
        medium_risk_count = sum(1 for r in self.detection_results.values() if r["risk_level"] == "medium")
        low_risk_count = sum(1 for r in self.detection_results.values() if r["risk_level"] == "low")
        
        # 统计检测到的技术
        technique_counts = {}
        for result in self.detection_results.values():
            for technique in result["detected_techniques"]:
                technique_counts[technique] = technique_counts.get(technique, 0) + 1
        
        # 统计应用的规避策略
        evasion_counts = {}
        for result in self.detection_results.values():
            for evasion in result["evasion_applied"]:
                technique = evasion["technique"]
                evasion_counts[technique] = evasion_counts.get(technique, 0) + 1
        
        report = {
            "total_detections": total_detections,
            "risk_distribution": {
                "high": high_risk_count,
                "medium": medium_risk_count,
                "low": low_risk_count
            },
            "detected_techniques": technique_counts,
            "applied_evasions": evasion_counts,
            "recent_detections": list(self.detection_results.values())[-5:],
            "recommendations": self._generate_detection_recommendations()
        }
        
        return report
    
    def _generate_detection_recommendations(self) -> List[str]:
        """生成检测建议"""
        recommendations = []
        
        if not self.detection_results:
            recommendations.append("暂无检测数据，建议继续监控")
            return recommendations
        
        # 基于风险等级生成建议
        high_risk_count = sum(1 for r in self.detection_results.values() if r["risk_level"] == "high")
        if high_risk_count > 0:
            recommendations.append("检测到高风险环境，建议加强规避措施")
        
        # 基于检测技术生成建议
        technique_counts = {}
        for result in self.detection_results.values():
            for technique in result["detected_techniques"]:
                technique_counts[technique] = technique_counts.get(technique, 0) + 1
        
        if technique_counts.get("webdriver_detection", 0) > 2:
            recommendations.append("频繁检测到WebDriver检测，建议优化浏览器环境伪装")
        
        if technique_counts.get("fingerprint_detection", 0) > 3:
            recommendations.append("频繁检测到指纹检测，建议加强指纹随机化")
        
        if technique_counts.get("behavior_detection", 0) > 2:
            recommendations.append("频繁检测到行为检测，建议优化行为模拟")
        
        if not recommendations:
            recommendations.append("检测环境正常，继续保持当前策略")
        
        return recommendations


class AdvancedEnvironmentDetector(EnvironmentDetector):
    """高级环境检测器"""
    
    def __init__(self):
        super().__init__()
        self.logger = logger.bind(name="advanced_environment_detector")
        
        # 高级检测技术
        self.advanced_detection_techniques = {
            "ai_detection": {
                "patterns": [
                    r"machine_learning",
                    r"artificial_intelligence",
                    r"neural_network",
                    r"pattern_recognition"
                ],
                "indicators": [
                    "behavioral_analysis",
                    "pattern_matching",
                    "anomaly_detection"
                ]
            },
            "honeypot_detection": {
                "patterns": [
                    r"honeypot",
                    r"trap",
                    r"bait",
                    r"decoy"
                ],
                "indicators": [
                    "suspicious_links",
                    "fake_forms",
                    "monitoring_scripts"
                ]
            },
            "sandbox_detection": {
                "patterns": [
                    r"sandbox",
                    r"virtual_machine",
                    r"emulator",
                    r"isolated_environment"
                ],
                "indicators": [
                    "limited_resources",
                    "artificial_environment",
                    "monitoring_tools"
                ]
            }
        }
        
        # 合并检测技术
        self.detection_techniques.update(self.advanced_detection_techniques)
    
    def detect_advanced_techniques(self, url: str, headers: Dict[str, str] = None,
                                 content: str = None, js_code: str = None) -> Dict[str, Any]:
        """检测高级检测技术"""
        advanced_result = {
            "url": url,
            "timestamp": time.time(),
            "advanced_techniques_detected": [],
            "ai_detection_risk": "low",
            "honeypot_risk": "low",
            "sandbox_risk": "low"
        }
        
        # 检测AI检测
        if self._detect_ai_detection(url, headers, content, js_code):
            advanced_result["advanced_techniques_detected"].append("ai_detection")
            advanced_result["ai_detection_risk"] = "high"
        
        # 检测蜜罐
        if self._detect_honeypot(url, headers, content, js_code):
            advanced_result["advanced_techniques_detected"].append("honeypot_detection")
            advanced_result["honeypot_risk"] = "high"
        
        # 检测沙箱
        if self._detect_sandbox(url, headers, content, js_code):
            advanced_result["advanced_techniques_detected"].append("sandbox_detection")
            advanced_result["sandbox_risk"] = "high"
        
        return advanced_result
    
    def _detect_ai_detection(self, url: str, headers: Dict[str, str] = None,
                            content: str = None, js_code: str = None) -> bool:
        """检测AI检测技术"""
        # 检查机器学习相关的头部
        if headers:
            ml_headers = ["X-ML-Detection", "X-AI-Analysis", "X-Pattern-Recognition"]
            for header in ml_headers:
                if header in headers:
                    return True
        
        # 检查内容中的AI检测模式
        if content:
            ai_patterns = [
                r"machine_learning",
                r"artificial_intelligence",
                r"neural_network",
                r"pattern_recognition",
                r"behavioral_analysis"
            ]
            for pattern in ai_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
        
        return False
    
    def _detect_honeypot(self, url: str, headers: Dict[str, str] = None,
                         content: str = None, js_code: str = None) -> bool:
        """检测蜜罐"""
        # 检查可疑的URL模式
        honeypot_patterns = [
            r"admin",
            r"login",
            r"test",
            r"demo",
            r"fake",
            r"trap"
        ]
        
        for pattern in honeypot_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        # 检查内容中的蜜罐指示器
        if content:
            honeypot_indicators = [
                r"honeypot",
                r"trap",
                r"bait",
                r"decoy",
                r"monitoring"
            ]
            for indicator in honeypot_indicators:
                if re.search(indicator, content, re.IGNORECASE):
                    return True
        
        return False
    
    def _detect_sandbox(self, url: str, headers: Dict[str, str] = None,
                        content: str = None, js_code: str = None) -> bool:
        """检测沙箱环境"""
        # 检查沙箱相关的头部
        if headers:
            sandbox_headers = ["X-Sandbox", "X-VM", "X-Emulator"]
            for header in sandbox_headers:
                if header in headers:
                    return True
        
        # 检查内容中的沙箱指示器
        if content:
            sandbox_indicators = [
                r"sandbox",
                r"virtual_machine",
                r"emulator",
                r"isolated",
                r"monitoring"
            ]
            for indicator in sandbox_indicators:
                if re.search(indicator, content, re.IGNORECASE):
                    return True
        
        return False 