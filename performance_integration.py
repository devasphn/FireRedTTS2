#!/usr/bin/env python3
"""
Performance Integration System
Integrates advanced performance monitoring and optimization for FireRedTTS2
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware

from advanced_performance_monitor import (
    AdvancedPerformanceMonitor, PerformanceConfig,
    GPUMetrics, PipelineMetrics, ThroughputMetrics, PerformanceAlert
)
from performance_optimization import (
    PerformanceOptimizer, OptimizationConfig,
    ModelCacheManager, GPUMemoryOptimizer, AudioBufferOptimizer
)
from data_models import APIResponse, create_success_response, create_error_response

logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE INTEGRATION CLASS
# ============================================================================

class PerformanceIntegration:
    """Integrates performance monitoring and optimization with FireRedTTS2"""
    
    def __init__(self, app: FastAPI, 
                 monitoring_config: PerformanceConfig = None,
                 optimization_config: OptimizationConfig = None):
        self.app = app
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.performance_monitor = AdvancedPerformanceMonitor(monitoring_config)
        self.performance_optimizer = PerformanceOptimizer(optimization_config)
        
        # Integration state
        self.is_running = False
        self.request_counter = 0
        
        # Setup middleware and endpoints
        self._setup_middleware()
        self._setup_performance_endpoints()
    
    def _setup_middleware(self):
        """Setup performance monitoring middleware"""
        
        @self.app.middleware("http")
        async def performance_middleware(request: Request, call_next):
            # Generate request ID
            request_id = f"req_{int(time.time() * 1000000)}_{self.request_counter}"
            self.request_counter += 1
            
            # Start request tracking
            self.performance_monitor.start_request(request_id, {
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent", "")
            })
            
            # Add request ID to request state
            request.state.request_id = request_id
            request.state.performance_monitor = self.performance_monitor
            
            try:
                # Process request
                response = await call_next(request)
                
                # End request tracking
                self.performance_monitor.end_request(request_id, True)
                
                # Add performance headers
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Processing-Time"] = str(
                    time.time() - request.state.get("start_time", time.time())
                )
                
                return response
            
            except Exception as e:
                # End request tracking with error
                self.performance_monitor.end_request(request_id, False)
                raise
    
    def _setup_performance_endpoints(self):
        """Setup performance monitoring API endpoints"""
        
        @self.app.get("/api/v1/performance/status")
        async def get_performance_status():
            """Get current performance status"""
            
            try:
                summary = self.performance_monitor.get_performance_summary()
                optimization_summary = self.performance_optimizer.get_optimization_summary()
                
                combined_summary = {
                    "monitoring": summary,
                    "optimization": optimization_summary,
                    "integration_status": {
                        "running": self.is_running,
                        "request_count": self.request_counter,
                        "uptime_seconds": time.time() - getattr(self, 'start_time', time.time())
                    }
                }
                
                return create_success_response(combined_summary, "Performance status retrieved")
            
            except Exception as e:
                logger.error(f"Performance status error: {e}")
                return create_error_response("Failed to get performance status", "PERFORMANCE_ERROR")
        
        @self.app.get("/api/v1/performance/metrics")
        async def get_performance_metrics():
            """Get detailed performance metrics"""
            
            try:
                # Get GPU metrics
                gpu_metrics = []
                for gpu_metric in self.performance_monitor.gpu_monitor.get_current_gpu_metrics():
                    gpu_metrics.append(gpu_metric.dict())
                
                # Get throughput metrics
                throughput = self.performance_monitor.latency_monitor.get_current_throughput()
                throughput_data = throughput.dict() if throughput else None
                
                # Get stage performance
                stage_performance = {}
                for stage in ["asr", "llm", "tts", "audio_processing"]:
                    stage_perf = self.performance_monitor.latency_monitor.get_stage_performance_summary(stage)
                    if stage_perf:
                        stage_performance[stage] = stage_perf
                
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "gpu_metrics": gpu_metrics,
                    "throughput_metrics": throughput_data,
                    "stage_performance": stage_performance,
                    "optimization_stats": self.performance_optimizer.get_optimization_summary()
                }
                
                return create_success_response(metrics, "Performance metrics retrieved")
            
            except Exception as e:
                logger.error(f"Performance metrics error: {e}")
                return create_error_response("Failed to get performance metrics", "METRICS_ERROR")
        
        @self.app.get("/api/v1/performance/alerts")
        async def get_performance_alerts():
            """Get active performance alerts"""
            
            try:
                # Get GPU alerts
                gpu_alerts = [
                    alert.dict() for alert in self.performance_monitor.gpu_monitor.gpu_alerts
                    if not alert.resolved
                ]
                
                # Get latency alerts
                latency_alerts = [
                    alert.dict() for alert in self.performance_monitor.latency_monitor.latency_alerts
                    if not alert.resolved
                ]
                
                alerts = {
                    "gpu_alerts": gpu_alerts,
                    "latency_alerts": latency_alerts,
                    "total_alerts": len(gpu_alerts) + len(latency_alerts),
                    "critical_alerts": len([
                        alert for alert in gpu_alerts + latency_alerts
                        if alert.get("severity") == "critical"
                    ])
                }
                
                return create_success_response(alerts, "Performance alerts retrieved")
            
            except Exception as e:
                logger.error(f"Performance alerts error: {e}")
                return create_error_response("Failed to get performance alerts", "ALERTS_ERROR")
        
        @self.app.post("/api/v1/performance/optimize")
        async def trigger_optimization():
            """Trigger performance optimization"""
            
            try:
                # Run optimization analysis
                recommendations = await self.performance_optimizer.optimization_engine.analyze_performance(
                    self.performance_monitor.gpu_monitor,
                    self.performance_monitor.latency_monitor
                )
                
                optimization_result = {
                    "optimization_triggered": True,
                    "recommendations_generated": len(recommendations),
                    "recommendations": [rec.dict() for rec in recommendations[-5:]]  # Last 5
                }
                
                return create_success_response(optimization_result, "Performance optimization triggered")
            
            except Exception as e:
                logger.error(f"Performance optimization error: {e}")
                return create_error_response("Failed to trigger optimization", "OPTIMIZATION_ERROR")
        
        @self.app.post("/api/v1/performance/clear-cache")
        async def clear_performance_cache():
            """Clear performance-related caches"""
            
            try:
                # Clear GPU cache
                if self.performance_optimizer.gpu_optimizer:
                    self.performance_optimizer.gpu_optimizer.optimize_memory_usage()
                
                # Clear audio buffers
                if self.performance_optimizer.audio_optimizer:
                    self.performance_optimizer.audio_optimizer.clear_buffers()
                
                # Clear model cache (partially)
                if self.performance_optimizer.model_cache:
                    # This would implement selective cache clearing
                    pass
                
                return create_success_response(None, "Performance caches cleared")
            
            except Exception as e:
                logger.error(f"Cache clearing error: {e}")
                return create_error_response("Failed to clear caches", "CACHE_ERROR")
    
    async def start(self):
        """Start performance integration"""
        
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start monitoring and optimization
        await self.performance_monitor.start_monitoring()
        await self.performance_optimizer.start()
        
        logger.info("Performance integration started")
    
    async def stop(self):
        """Stop performance integration"""
        
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop monitoring and optimization
        await self.performance_monitor.stop_monitoring()
        await self.performance_optimizer.stop()
        
        logger.info("Performance integration stopped")
    
    # Convenience methods for pipeline tracking
    def start_pipeline_stage(self, request_id: str, stage_name: str):
        """Start tracking a pipeline stage"""
        return self.performance_monitor.start_stage(request_id, stage_name)
    
    def end_pipeline_stage(self, request_id: str, stage_name: str, 
                          success: bool = True, error_message: str = None,
                          input_size: int = 0, output_size: int = 0,
                          metadata: Dict[str, Any] = None):
        """End tracking a pipeline stage"""
        self.performance_monitor.end_stage(
            request_id, stage_name, success, error_message, 
            input_size, output_size, metadata
        )
    
    # Model management integration
    async def get_cached_model(self, model_name: str, model_path: str, 
                              loader_func: Callable, **loader_kwargs):
        """Get model from cache with performance optimization"""
        
        if not self.performance_optimizer.model_cache:
            # Fallback to direct loading
            return loader_func(model_path, **loader_kwargs)
        
        return await self.performance_optimizer.model_cache.get_model(
            model_name, model_path, loader_func, **loader_kwargs
        )
    
    # Audio buffer management integration
    async def get_audio_buffer(self, buffer_type: str, size: int, dtype=None):
        """Get optimized audio buffer"""
        
        if not self.performance_optimizer.audio_optimizer:
            import numpy as np
            return np.zeros(size, dtype=dtype or np.float32)
        
        return await self.performance_optimizer.audio_optimizer.get_input_buffer(
            buffer_type, size, dtype or np.float32
        )
    
    async def return_audio_buffer(self, buffer_type: str, buffer):
        """Return audio buffer to pool"""
        
        if self.performance_optimizer.audio_optimizer:
            await self.performance_optimizer.audio_optimizer.return_input_buffer(
                buffer_type, buffer
            )
    
    # Batch processing integration
    async def process_batch(self, batch_type: str, item: Dict[str, Any], 
                           processor_func: Callable):
        """Process item in batch for improved throughput"""
        
        if not self.performance_optimizer.batch_processor:
            # Fallback to direct processing
            return await processor_func(item.get("input"))
        
        return await self.performance_optimizer.batch_processor.add_to_batch(
            batch_type, item, processor_func
        )
    
    # Thread pool integration
    async def execute_in_thread(self, func: Callable, *args, **kwargs):
        """Execute function in optimized thread pool"""
        
        if not self.performance_optimizer.connection_pool:
            # Fallback to direct execution
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
        
        return await self.performance_optimizer.connection_pool.execute_in_thread(
            func, *args, **kwargs
        )

# ============================================================================
# PERFORMANCE DECORATORS
# ============================================================================

def track_performance(stage_name: str, track_io_size: bool = False):
    """Decorator to track performance of functions"""
    
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Get request ID from context (if available)
            request_id = getattr(asyncio.current_task(), 'request_id', f"task_{int(time.time() * 1000000)}")
            
            # Get performance monitor from global context or create dummy
            performance_monitor = getattr(asyncio.current_task(), 'performance_monitor', None)
            
            if performance_monitor:
                performance_monitor.start_stage(request_id, stage_name)
            
            try:
                result = await func(*args, **kwargs)
                
                if performance_monitor:
                    # Calculate I/O sizes if requested
                    input_size = 0
                    output_size = 0
                    
                    if track_io_size:
                        try:
                            import sys
                            if args:
                                input_size = sys.getsizeof(args[0])
                            if result:
                                output_size = sys.getsizeof(result)
                        except:
                            pass
                    
                    performance_monitor.end_stage(
                        request_id, stage_name, True, None, input_size, output_size
                    )
                
                return result
            
            except Exception as e:
                if performance_monitor:
                    performance_monitor.end_stage(
                        request_id, stage_name, False, str(e)
                    )
                raise
        
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, create a simple wrapper
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def optimize_gpu_memory(func):
    """Decorator to optimize GPU memory usage"""
    
    def wrapper(*args, **kwargs):
        # Create memory snapshot before
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Clean up GPU memory after
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
    
    return wrapper

# ============================================================================
# INTEGRATION HELPER FUNCTIONS
# ============================================================================

def integrate_performance_with_app(app: FastAPI, 
                                  monitoring_config: PerformanceConfig = None,
                                  optimization_config: OptimizationConfig = None) -> PerformanceIntegration:
    """Integrate performance system with FastAPI application"""
    
    try:
        performance_integration = PerformanceIntegration(
            app, monitoring_config, optimization_config
        )
        
        # Add startup and shutdown events
        @app.on_event("startup")
        async def startup_performance():
            await performance_integration.start()
        
        @app.on_event("shutdown")
        async def shutdown_performance():
            await performance_integration.stop()
        
        logger.info("Performance system integrated with FastAPI application")
        return performance_integration
    
    except Exception as e:
        logger.error(f"Failed to integrate performance system: {e}")
        raise

def create_performance_config(config_file: str = None) -> Tuple[PerformanceConfig, OptimizationConfig]:
    """Create performance configuration from file"""
    
    monitoring_config = PerformanceConfig()
    optimization_config = OptimizationConfig()
    
    if config_file and Path(config_file).exists():
        try:
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update monitoring config
            monitoring_data = config_data.get('monitoring', {})
            for key, value in monitoring_data.items():
                if hasattr(monitoring_config, key):
                    setattr(monitoring_config, key, value)
            
            # Update optimization config
            optimization_data = config_data.get('optimization', {})
            for key, value in optimization_data.items():
                if hasattr(optimization_config, key):
                    setattr(optimization_config, key, value)
            
            logger.info(f"Loaded performance configuration from {config_file}")
        
        except Exception as e:
            logger.error(f"Failed to load performance config: {e}")
    
    return monitoring_config, optimization_config

def create_performance_config_template(output_path: str):
    """Create performance configuration template"""
    
    template_config = {
        "monitoring": {
            "gpu_monitoring_interval": 1.0,
            "latency_monitoring_interval": 0.1,
            "system_monitoring_interval": 5.0,
            "optimization_check_interval": 30.0,
            "gpu_utilization_target": 80.0,
            "gpu_memory_warning": 85.0,
            "gpu_memory_critical": 95.0,
            "latency_warning_ms": 500.0,
            "latency_critical_ms": 1000.0,
            "throughput_target_rps": 10.0,
            "enable_auto_optimization": True,
            "enable_predictive_scaling": True,
            "metrics_history_size": 10000,
            "performance_window_minutes": 60
        },
        "optimization": {
            "enable_model_caching": True,
            "max_cached_models": 3,
            "model_cache_size_gb": 8.0,
            "model_lazy_loading": True,
            "enable_gpu_optimization": True,
            "gpu_memory_fraction": 0.9,
            "enable_mixed_precision": True,
            "enable_audio_buffering": True,
            "audio_buffer_size_ms": 1000,
            "max_audio_buffers": 10,
            "enable_connection_pooling": True,
            "max_connections_per_pool": 20,
            "enable_batch_processing": True,
            "max_batch_size": 8,
            "batch_timeout_ms": 50,
            "max_worker_threads": 4,
            "enable_result_caching": True,
            "result_cache_size_mb": 500
        }
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(template_config, f, indent=2)
        
        logger.info(f"Created performance configuration template: {output_path}")
    
    except Exception as e:
        logger.error(f"Failed to create performance config template: {e}")

# Export main components
__all__ = [
    'PerformanceIntegration', 'track_performance', 'optimize_gpu_memory',
    'integrate_performance_with_app', 'create_performance_config',
    'create_performance_config_template'
]