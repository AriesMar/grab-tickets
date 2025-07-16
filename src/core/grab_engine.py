"""
抢票引擎核心模块
"""
import asyncio
import time
from typing import List, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from ..data.models import TicketRequest, GrabResult, TicketInfo, PlatformType
from ..utils.logger import get_logger


class GrabStrategy(ABC):
    """抢票策略抽象基类"""
    
    @abstractmethod
    async def execute(self, request: TicketRequest) -> GrabResult:
        """执行抢票策略"""
        pass
    
    @abstractmethod
    def can_handle(self, platform: PlatformType) -> bool:
        """判断是否可以处理该平台"""
        pass


class GrabEngine:
    """抢票引擎"""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.strategies: List[GrabStrategy] = []
        self.logger = get_logger("grab_engine")
        self.is_running = False
        self._callbacks: Dict[str, List[Callable]] = {
            "on_start": [],
            "on_success": [],
            "on_failure": [],
            "on_complete": []
        }
    
    def add_strategy(self, strategy: GrabStrategy):
        """添加抢票策略"""
        self.strategies.append(strategy)
        self.logger.info(f"添加抢票策略: {strategy.__class__.__name__}")
    
    def remove_strategy(self, strategy: GrabStrategy):
        """移除抢票策略"""
        if strategy in self.strategies:
            self.strategies.remove(strategy)
            self.logger.info(f"移除抢票策略: {strategy.__class__.__name__}")
    
    def get_strategy(self, platform: PlatformType) -> Optional[GrabStrategy]:
        """获取适合的策略"""
        for strategy in self.strategies:
            if strategy.can_handle(platform):
                return strategy
        return None
    
    def add_callback(self, event: str, callback: Callable):
        """添加回调函数"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
            self.logger.debug(f"添加回调函数: {event} -> {callback.__name__}")
    
    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """触发回调函数"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f"回调函数执行失败: {callback.__name__}, 错误: {e}")
    
    async def grab_ticket(self, request: TicketRequest) -> GrabResult:
        """抢票主方法"""
        self.logger.info(f"开始抢票: {request.event_id} on {request.platform}")
        self._trigger_callbacks("on_start", request)
        
        strategy = self.get_strategy(request.platform)
        if not strategy:
            result = GrabResult(
                success=False,
                message=f"未找到适合的策略处理平台: {request.platform}"
            )
            self._trigger_callbacks("on_failure", request, result)
            return result
        
        try:
            result = await strategy.execute(request)
            
            if result.success:
                self.logger.success(f"抢票成功: {request.event_id}")
                self._trigger_callbacks("on_success", request, result)
            else:
                self.logger.warning(f"抢票失败: {request.event_id}, 原因: {result.message}")
                self._trigger_callbacks("on_failure", request, result)
            
            self._trigger_callbacks("on_complete", request, result)
            return result
            
        except Exception as e:
            self.logger.error(f"抢票异常: {request.event_id}, 错误: {e}")
            result = GrabResult(
                success=False,
                message=f"抢票过程中发生异常: {str(e)}"
            )
            self._trigger_callbacks("on_failure", request, result)
            self._trigger_callbacks("on_complete", request, result)
            return result
    
    async def grab_tickets_batch(self, requests: List[TicketRequest]) -> List[GrabResult]:
        """批量抢票"""
        self.logger.info(f"开始批量抢票，共 {len(requests)} 个任务")
        self.is_running = True
        
        results = []
        tasks = []
        
        # 创建异步任务
        for request in requests:
            task = asyncio.create_task(self.grab_ticket(request))
            tasks.append(task)
        
        # 等待所有任务完成
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(GrabResult(
                        success=False,
                        message=f"任务执行异常: {str(result)}"
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        finally:
            self.is_running = False
    
    def grab_tickets_concurrent(self, requests: List[TicketRequest]) -> List[GrabResult]:
        """并发抢票（使用线程池）"""
        self.logger.info(f"开始并发抢票，共 {len(requests)} 个任务")
        self.is_running = True
        
        results = []
        
        try:
            # 提交任务到线程池
            future_to_request = {}
            for request in requests:
                future = self.executor.submit(
                    asyncio.run, 
                    self.grab_ticket(request)
                )
                future_to_request[future] = request
            
            # 收集结果
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"任务执行失败: {request.event_id}, 错误: {e}")
                    results.append(GrabResult(
                        success=False,
                        message=f"任务执行异常: {str(e)}"
                    ))
            
            return results
            
        finally:
            self.is_running = False
    
    async def grab_ticket_with_retry(self, request: TicketRequest) -> GrabResult:
        """带重试的抢票"""
        self.logger.info(f"开始重试抢票: {request.event_id}, 重试次数: {request.retry_times}")
        
        for attempt in range(request.retry_times + 1):
            try:
                result = await self.grab_ticket(request)
                
                if result.success:
                    self.logger.success(f"抢票成功 (尝试 {attempt + 1}/{request.retry_times + 1}): {request.event_id}")
                    return result
                else:
                    self.logger.warning(f"抢票失败 (尝试 {attempt + 1}/{request.retry_times + 1}): {request.event_id}, 原因: {result.message}")
                    
                    if attempt < request.retry_times:
                        self.logger.info(f"等待 {request.retry_interval} 秒后重试...")
                        await asyncio.sleep(request.retry_interval)
                
            except Exception as e:
                self.logger.error(f"抢票异常 (尝试 {attempt + 1}/{request.retry_times + 1}): {request.event_id}, 错误: {e}")
                
                if attempt < request.retry_times:
                    self.logger.info(f"等待 {request.retry_interval} 秒后重试...")
                    await asyncio.sleep(request.retry_interval)
        
        # 所有重试都失败
        return GrabResult(
            success=False,
            message=f"经过 {request.retry_times + 1} 次尝试后抢票失败"
        )
    
    def stop(self):
        """停止抢票引擎"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.logger.info("抢票引擎已停止")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class GrabTask:
    """抢票任务"""
    
    def __init__(self, request: TicketRequest, engine: GrabEngine):
        self.request = request
        self.engine = engine
        self.logger = get_logger("grab_task")
        self.start_time = None
        self.end_time = None
        self.result = None
    
    async def execute(self) -> GrabResult:
        """执行任务"""
        self.start_time = time.time()
        self.logger.info(f"开始执行抢票任务: {self.request.event_id}")
        
        try:
            self.result = await self.engine.grab_ticket_with_retry(self.request)
            return self.result
        finally:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.logger.info(f"抢票任务完成: {self.request.event_id}, 耗时: {duration:.2f}秒")
    
    def get_duration(self) -> float:
        """获取任务执行时长"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0 