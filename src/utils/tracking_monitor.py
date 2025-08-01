"""
反追踪监控器 - 实时监控和阻止追踪行为
"""
import time
import re
import json
from typing import Dict, Any, List, Set, Optional
from urllib.parse import urlparse, parse_qs
from loguru import logger


class TrackingMonitor:
    """反追踪监控器"""
    
    def __init__(self):
        self.logger = logger.bind(name="tracking_monitor")
        
        # 追踪检测模式
        self.tracking_patterns = {
            "analytics": [
                r"google-analytics\.com",
                r"googletagmanager\.com",
                r"gtag",
                r"ga\(",
                r"analytics",
                r"tracking"
            ],
            "advertising": [
                r"doubleclick\.net",
                r"adnxs\.com",
                r"googlesyndication\.com",
                r"amazon-adsystem\.com",
                r"criteo\.com",
                r"taboola\.com",
                r"advertising",
                r"ads\."
            ],
            "social_media": [
                r"facebook\.com",
                r"twitter\.com",
                r"linkedin\.com",
                r"pinterest\.com",
                r"instagram\.com",
                r"social",
                r"share"
            ],
            "fingerprinting": [
                r"fingerprint",
                r"canvas",
                r"webgl",
                r"audio",
                r"font",
                r"plugin",
                r"screen",
                r"navigator"
            ],
            "cookies": [
                r"_ga",
                r"_gid",
                r"_fbp",
                r"_fbc",
                r"utm_",
                r"ref_",
                r"source",
                r"medium",
                r"campaign"
            ]
        }
        
        # 已阻止的追踪
        self.blocked_tracking = {
            "domains": set(),
            "scripts": set(),
            "cookies": set(),
            "requests": set()
        }
        
        # 追踪统计
        self.tracking_stats = {
            "total_detected": 0,
            "total_blocked": 0,
            "by_category": {},
            "by_domain": {},
            "recent_activity": []
        }
    
    def detect_tracking(self, url: str, headers: Dict[str, str] = None, 
                       cookies: Dict[str, str] = None, content: str = None) -> Dict[str, Any]:
        """检测追踪行为"""
        tracking_info = {
            "url": url,
            "timestamp": time.time(),
            "categories": [],
            "details": {},
            "severity": "low"
        }
        
        # 检查URL中的追踪
        url_tracking = self._check_url_tracking(url)
        if url_tracking:
            tracking_info["categories"].extend(url_tracking["categories"])
            tracking_info["details"]["url"] = url_tracking["details"]
        
        # 检查请求头中的追踪
        if headers:
            header_tracking = self._check_header_tracking(headers)
            if header_tracking:
                tracking_info["categories"].extend(header_tracking["categories"])
                tracking_info["details"]["headers"] = header_tracking["details"]
        
        # 检查Cookie中的追踪
        if cookies:
            cookie_tracking = self._check_cookie_tracking(cookies)
            if cookie_tracking:
                tracking_info["categories"].extend(cookie_tracking["categories"])
                tracking_info["details"]["cookies"] = cookie_tracking["details"]
        
        # 检查内容中的追踪
        if content:
            content_tracking = self._check_content_tracking(content)
            if content_tracking:
                tracking_info["categories"].extend(content_tracking["categories"])
                tracking_info["details"]["content"] = content_tracking["details"]
        
        # 计算严重程度
        if tracking_info["categories"]:
            tracking_info["severity"] = self._calculate_severity(tracking_info["categories"])
            self._update_stats(tracking_info)
        
        return tracking_info
    
    def _check_url_tracking(self, url: str) -> Optional[Dict[str, Any]]:
        """检查URL中的追踪"""
        tracking = {"categories": [], "details": {}}
        
        # 检查域名
        domain = urlparse(url).netloc.lower()
        for category, patterns in self.tracking_patterns.items():
            for pattern in patterns:
                if re.search(pattern, domain, re.IGNORECASE):
                    tracking["categories"].append(category)
                    tracking["details"][category] = pattern
                    break
        
        # 检查URL参数
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        tracking_params = []
        for param, values in query_params.items():
            for pattern in self.tracking_patterns["cookies"]:
                if re.search(pattern, param, re.IGNORECASE):
                    tracking_params.append(param)
                    break
        
        if tracking_params:
            tracking["categories"].append("cookies")
            tracking["details"]["tracking_params"] = tracking_params
        
        return tracking if tracking["categories"] else None
    
    def _check_header_tracking(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查请求头中的追踪"""
        tracking = {"categories": [], "details": {}}
        
        # 检查Referer头
        referer = headers.get("Referer", "")
        if referer:
            for category, patterns in self.tracking_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, referer, re.IGNORECASE):
                        tracking["categories"].append(category)
                        tracking["details"]["referer"] = referer
                        break
        
        # 检查User-Agent中的追踪信息
        user_agent = headers.get("User-Agent", "")
        if "bot" in user_agent.lower() or "crawler" in user_agent.lower():
            tracking["categories"].append("automation")
            tracking["details"]["user_agent"] = user_agent
        
        return tracking if tracking["categories"] else None
    
    def _check_cookie_tracking(self, cookies: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查Cookie中的追踪"""
        tracking = {"categories": [], "details": {}}
        
        tracking_cookies = []
        for cookie_name, cookie_value in cookies.items():
            for pattern in self.tracking_patterns["cookies"]:
                if re.search(pattern, cookie_name, re.IGNORECASE):
                    tracking_cookies.append(cookie_name)
                    break
        
        if tracking_cookies:
            tracking["categories"].append("cookies")
            tracking["details"]["tracking_cookies"] = tracking_cookies
        
        return tracking if tracking["categories"] else None
    
    def _check_content_tracking(self, content: str) -> Optional[Dict[str, Any]]:
        """检查内容中的追踪"""
        tracking = {"categories": [], "details": {}}
        
        # 检查追踪脚本
        for category, patterns in self.tracking_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    tracking["categories"].append(category)
                    tracking["details"][category] = pattern
                    break
        
        # 检查JavaScript追踪代码
        js_patterns = [
            r"gtag\(",
            r"ga\(",
            r"fbq\(",
            r"twq\(",
            r"pintrk\(",
            r"snap\(",
            r"track\(",
            r"analytics"
        ]
        
        js_tracking = []
        for pattern in js_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                js_tracking.append(pattern)
        
        if js_tracking:
            tracking["categories"].append("analytics")
            tracking["details"]["js_tracking"] = js_tracking
        
        return tracking if tracking["categories"] else None
    
    def _calculate_severity(self, categories: List[str]) -> str:
        """计算追踪严重程度"""
        severity_scores = {
            "fingerprinting": 3,
            "analytics": 2,
            "advertising": 2,
            "social_media": 1,
            "cookies": 1,
            "automation": 3
        }
        
        total_score = sum(severity_scores.get(cat, 0) for cat in categories)
        
        if total_score >= 6:
            return "high"
        elif total_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _update_stats(self, tracking_info: Dict[str, Any]):
        """更新追踪统计"""
        self.tracking_stats["total_detected"] += 1
        
        # 按类别统计
        for category in tracking_info["categories"]:
            if category not in self.tracking_stats["by_category"]:
                self.tracking_stats["by_category"][category] = 0
            self.tracking_stats["by_category"][category] += 1
        
        # 按域名统计
        domain = urlparse(tracking_info["url"]).netloc
        if domain not in self.tracking_stats["by_domain"]:
            self.tracking_stats["by_domain"][domain] = 0
        self.tracking_stats["by_domain"][domain] += 1
        
        # 记录最近活动
        self.tracking_stats["recent_activity"].append({
            "timestamp": tracking_info["timestamp"],
            "url": tracking_info["url"],
            "categories": tracking_info["categories"],
            "severity": tracking_info["severity"]
        })
        
        # 保持最近100条记录
        if len(self.tracking_stats["recent_activity"]) > 100:
            self.tracking_stats["recent_activity"] = self.tracking_stats["recent_activity"][-100:]
    
    def block_tracking(self, tracking_info: Dict[str, Any]) -> bool:
        """阻止追踪"""
        url = tracking_info["url"]
        
        # 检查是否应该阻止
        if tracking_info["severity"] in ["high", "medium"]:
            self.blocked_tracking["requests"].add(url)
            self.tracking_stats["total_blocked"] += 1
            
            self.logger.warning(f"阻止追踪请求: {url}")
            return True
        
        return False
    
    def sanitize_response(self, response_text: str) -> str:
        """清理响应中的追踪代码"""
        # 移除常见的追踪脚本
        tracking_scripts = [
            r'<script[^>]*google-analytics[^>]*>.*?</script>',
            r'<script[^>]*gtag[^>]*>.*?</script>',
            r'<script[^>]*ga\([^>]*>.*?</script>',
            r'<script[^>]*fbq[^>]*>.*?</script>',
            r'<script[^>]*twq[^>]*>.*?</script>',
            r'<script[^>]*pintrk[^>]*>.*?</script>',
            r'<script[^>]*snap[^>]*>.*?</script>'
        ]
        
        for pattern in tracking_scripts:
            response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE | re.DOTALL)
        
        # 移除追踪参数
        tracking_params = [
            r'utm_source=[^&]*',
            r'utm_medium=[^&]*',
            r'utm_campaign=[^&]*',
            r'utm_term=[^&]*',
            r'utm_content=[^&]*',
            r'fbclid=[^&]*',
            r'gclid=[^&]*',
            r'msclkid=[^&]*'
        ]
        
        for pattern in tracking_params:
            response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE)
        
        return response_text
    
    def get_tracking_report(self) -> Dict[str, Any]:
        """获取追踪报告"""
        return {
            "timestamp": time.time(),
            "stats": self.tracking_stats,
            "blocked": {
                "domains": list(self.blocked_tracking["domains"]),
                "scripts": list(self.blocked_tracking["scripts"]),
                "cookies": list(self.blocked_tracking["cookies"]),
                "requests": list(self.blocked_tracking["requests"])
            },
            "recent_activity": self.tracking_stats["recent_activity"][-10:],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if self.tracking_stats["total_detected"] > 10:
            recommendations.append("检测到大量追踪行为，建议启用更强的追踪保护")
        
        if self.tracking_stats["by_category"].get("fingerprinting", 0) > 5:
            recommendations.append("检测到指纹追踪，建议启用指纹随机化")
        
        if self.tracking_stats["by_category"].get("analytics", 0) > 10:
            recommendations.append("检测到大量分析追踪，建议阻止分析脚本")
        
        if not recommendations:
            recommendations.append("追踪保护运行正常")
        
        return recommendations 