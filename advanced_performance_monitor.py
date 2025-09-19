#!/usr/bin/env python3
"""
Advanced Performance Monitoring System
Comprehensive performance monitoring for FireRedTTS2 with real-time GPU monitoring,
latency tracking, system health indicators, and intelligent performance optimization
"""

import asyncio
import time
import json
import logging
import threading
import statistics
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
from pathlib import Path
import numpy as np

import psutil
import torch
import GPUtil
from pydantic import BaseModel, Field

from data_models import LatencyMetrics, ResourceUsage, QualityMetrics
from system_monitoring import SystemMonitor, MonitoringConfig

# ============================================================================
# PERFORMANCE MONITORING CONFIGURATION
# ============================================================================

@dataclass
class PerformanceConfig:
    """Advanced performance monitoring configuration"""
    
    # Monitoring intervals (seconds)
    gpu_monitoring_interval: float = 1.0
    latency_monitoring_interval: float = 0.1
    system_monitoring_interval: float = 5.0
    optimization_check_interval: float = 30.0
    
    # Performance thresholds
    gpu_utilization_target: float = 80.0
    gpu_memory_warning: float = 85.0
    gpu_memory_critical: float = 95.0
    latency_warning_ms: float = 500.0
    latency_critical_ms: float = 1000.0
    throughput_target_rps: float = 10.0
    
    # Optimization settings
    enable_auto_optimization: bool = True
    enable_predictive_scaling: bool = True
    enable_adaptive_batching: bool = True
    enable_model_switching: bool = True
    
    # Data retention
    metrics_history_size: int = 10000
    latency_samples_size: int = 1000
    performance_window_minutes: int = 60
    
    # Alerting thresholds
    performance_degradation_threshold: float = 0.2  # 20% degradation
    alert_cooldown_minutes: int = 5

# ============================================================================
# PERFORMANCE METRICS MODELS
# ============================================================================

class GPUMetrics(BaseModel):
    """Detailed GPU performance metrics"""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    gpu_id: int
    name: str
    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_percent: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    clock_speed_mhz: int = 0
    memory_clock_mhz: int = 0
    fan_speed_percent: float = 0.0
    processes: List[Dict[str, Any]] = Field(default_factory=list)

class PipelineMetrics(BaseModel):
    """Pipeline stage performance metrics"""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str
    stage_name: str
    start_time: float
    end_time: float
    duration_ms: float
    input_size: int = 0
    output_size: int = 0
    gpu_memory_used_mb: float = 0.0
    cpu_percent: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ThroughputMetrics(BaseModel):
    """System throughput metrics"""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    requests_per_second: float = 0.0
    concurrent_requests: int = 0
    queue_length: int = 0
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    total_requests: int = 0

class PerformanceAlert(BaseModel):
    """Performance alert model"""
    
    alert_id: str = Field(default_factory=lambda: f"perf_{int(time.time())}")
    timestamp: datetime = Field(default_factory=datetime.now)
    severity: str  # info, warning, critical
    category: str  # gpu, latency, throughput, system
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    recommendations: List[str] = Field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class OptimizationRecommendation(BaseModel):
    """Performance optimization recommendation"""
    
    recommendation_id: str = Field(default_factory=lambda: f"opt_{int(time.time())}")
    timestamp: datetime = Field(default_factory=datetime.now)
    category: str  # model, batch_size, memory, caching
    priority: str  # low, medium, high, critical
    title: str
    description: str
    expected_improvement: str
    implementation_effort: str  # low, medium, high
    auto_applicable: bool = False
    applied: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ============================================================================
# GPU PERFORMANCE MONITOR
# ============================================================================

class GPUPerformanceMonitor:
    """Advanced GPU performance monitoring"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # GPU metrics storage
        self.gpu_metrics_history: Dict[int, deque] = {}
        self.gpu_alerts: List[PerformanceAlert] = []
        
        # Initialize GPU monitoring
        self._initialize_gpu_monitoring()
        
        # Background monitoring task
        self.monitoring_task = None
        self.is_monitoring = False
    
    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring"""
        
        try:
            if torch.cuda.is_available():
                self.gpu_count = torch.cuda.device_count()
                self.gpu_info = []
                
                for i in range(self.gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    self.gpu_info.append({
                        'id': i,
                        'name': props.name,
                        'total_memory': props.total_memory,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                    
                    # Initialize metrics history
                    self.gpu_metrics_history[i] = deque(maxlen=self.config.metrics_history_size)
                
                self.logger.info(f"Initialized GPU monitoring for {self.gpu_count} GPUs")
            else:
                self.gpu_count = 0
                self.gpu_info = []
                self.logger.warning("No CUDA GPUs available for monitoring")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU monitoring: {e}")
            self.gpu_count = 0
            self.gpu_info = []
    
    async def start_monitoring(self):
        """Start GPU performance monitoring"""
        
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started GPU performance monitoring")
    
    async def stop_monitoring(self):
        """Stop GPU performance monitoring"""
        
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped GPU performance monitoring")
    
    async def _monitoring_loop(self):
        """Main GPU monitoring loop"""
        
        while self.is_monitoring:
            try:
                await self._collect_gpu_metrics()
                await self._check_gpu_alerts()
                await asyncio.sleep(self.config.gpu_monitoring_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"GPU monitoring error: {e}")
                await asyncio.sleep(self.config.gpu_monitoring_interval)
    
    async def _collect_gpu_metrics(self):
        """Collect GPU performance metrics"""
        
        try:
            if self.gpu_count == 0:
                return
            
            # Use GPUtil for detailed GPU metrics
            gpus = GPUtil.getGPUs()
            
            for i, gpu in enumerate(gpus):
                if i >= self.gpu_count:
                    break
                
                # Get PyTorch memory stats
                torch_memory = torch.cuda.memory_stats(i) if torch.cuda.is_available() else {}
                allocated_mb = torch_memory.get('allocated_bytes.all.current', 0) / (1024**2)
                
                # Get GPU processes
                processes = []
                try:
                    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                        # This is a simplified process detection
                        # In practice, you'd use nvidia-ml-py for accurate GPU process info
                        if 'python' in proc.info['name'].lower():
                            processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'memory_mb': proc.info['memory_info'].rss / (1024**2)
                            })
                except:
                    pass
                
                metrics = GPUMetrics(
                    gpu_id=i,
                    name=gpu.name,
                    utilization_percent=gpu.load * 100,
                    memory_used_mb=gpu.memoryUsed,
                    memory_total_mb=gpu.memoryTotal,
                    memory_percent=(gpu.memoryUsed / gpu.memoryTotal) * 100,
                    temperature_c=gpu.temperature,
                    processes=processes[:5]  # Limit to top 5 processes
                )
                
                self.gpu_metrics_history[i].append(metrics)
        
        except Exception as e:
            self.logger.error(f"Failed to collect GPU metrics: {e}")
    
    async def _check_gpu_alerts(self):
        """Check for GPU performance alerts"""
        
        try:
            for gpu_id, metrics_history in self.gpu_metrics_history.items():
                if not metrics_history:
                    continue
                
                latest_metrics = metrics_history[-1]
                
                # Check memory usage
                if latest_metrics.memory_percent > self.config.gpu_memory_critical:
                    await self._create_alert(
                        "critical", "gpu", f"gpu_{gpu_id}_memory",
                        latest_metrics.memory_percent, self.config.gpu_memory_critical,
                        f"GPU {gpu_id} memory usage critical: {latest_metrics.memory_percent:.1f}%",
                        [
                            "Clear GPU cache with torch.cuda.empty_cache()",
                            "Reduce batch size",
                            "Enable gradient checkpointing",
                            "Consider model quantization"
                        ]
                    )
                elif latest_metrics.memory_percent > self.config.gpu_memory_warning:
                    await self._create_alert(
                        "warning", "gpu", f"gpu_{gpu_id}_memory",
                        latest_metrics.memory_percent, self.config.gpu_memory_warning,
                        f"GPU {gpu_id} memory usage high: {latest_metrics.memory_percent:.1f}%",
                        [
                            "Monitor memory usage closely",
                            "Consider reducing batch size",
                            "Clear unused variables"
                        ]
                    )
                
                # Check temperature
                if latest_metrics.temperature_c > 85:
                    await self._create_alert(
                        "warning", "gpu", f"gpu_{gpu_id}_temperature",
                        latest_metrics.temperature_c, 85,
                        f"GPU {gpu_id} temperature high: {latest_metrics.temperature_c:.1f}°C",
                        [
                            "Check GPU cooling",
                            "Reduce GPU workload",
                            "Check case ventilation"
                        ]
                    )
        
        except Exception as e:
            self.logger.error(f"GPU alert checking error: {e}")
    
    async def _create_alert(self, severity: str, category: str, metric_name: str,
                          current_value: float, threshold_value: float,
                          message: str, recommendations: List[str]):
        """Create performance alert"""
        
        # Check if similar alert exists recently
        recent_alerts = [
            alert for alert in self.gpu_alerts
            if alert.metric_name == metric_name and
            not alert.resolved and
            (datetime.now() - alert.timestamp).total_seconds() < 300  # 5 minutes
        ]
        
        if recent_alerts:
            return  # Don't spam alerts
        
        alert = PerformanceAlert(
            severity=severity,
            category=category,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            recommendations=recommendations
        )
        
        self.gpu_alerts.append(alert)
        
        # Log alert
        log_level = logging.CRITICAL if severity == "critical" else logging.WARNING
        self.logger.log(log_level, f"PERFORMANCE ALERT: {message}")
    
    def get_current_gpu_metrics(self) -> List[GPUMetrics]:
        """Get current GPU metrics"""
        
        current_metrics = []
        for gpu_id, metrics_history in self.gpu_metrics_history.items():
            if metrics_history:
                current_metrics.append(metrics_history[-1])
        
        return current_metrics
    
    def get_gpu_utilization_trend(self, gpu_id: int, minutes: int = 10) -> Dict[str, float]:
        """Get GPU utilization trend"""
        
        if gpu_id not in self.gpu_metrics_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [
            m for m in self.gpu_metrics_history[gpu_id]
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        utilizations = [m.utilization_percent for m in recent_metrics]
        memory_usages = [m.memory_percent for m in recent_metrics]
        
        return {
            "avg_utilization": statistics.mean(utilizations),
            "max_utilization": max(utilizations),
            "min_utilization": min(utilizations),
            "avg_memory": statistics.mean(memory_usages),
            "max_memory": max(memory_usages),
            "trend_direction": "increasing" if utilizations[-1] > utilizations[0] else "decreasing",
            "sample_count": len(recent_metrics)
        }

# ============================================================================
# LATENCY PERFORMANCE MONITOR
# ============================================================================

class LatencyPerformanceMonitor:
    """Advanced latency performance monitoring"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Latency tracking
        self.pipeline_metrics: deque = deque(maxlen=config.metrics_history_size)
        self.stage_latencies: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.latency_samples_size)
        )
        self.latency_alerts: List[PerformanceAlert] = []
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=1000)
        self.throughput_history: deque = deque(maxlen=config.metrics_history_size)
        
        # Active request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
    
    def start_request_tracking(self, request_id: str, metadata: Dict[str, Any] = None) -> str:
        """Start tracking a request"""
        
        self.active_requests[request_id] = {
            'start_time': time.time(),
            'stages': {},
            'metadata': metadata or {}
        }
        
        return request_id
    
    def start_stage_tracking(self, request_id: str, stage_name: str) -> str:
        """Start tracking a pipeline stage"""
        
        if request_id not in self.active_requests:
            self.logger.warning(f"Request {request_id} not found for stage tracking")
            return request_id
        
        self.active_requests[request_id]['stages'][stage_name] = {
            'start_time': time.time()
        }
        
        return request_id
    
    def end_stage_tracking(self, request_id: str, stage_name: str, 
                          success: bool = True, error_message: str = None,
                          input_size: int = 0, output_size: int = 0,
                          metadata: Dict[str, Any] = None):
        """End tracking a pipeline stage"""
        
        if request_id not in self.active_requests:
            return
        
        if stage_name not in self.active_requests[request_id]['stages']:
            return
        
        end_time = time.time()
        start_time = self.active_requests[request_id]['stages'][stage_name]['start_time']
        duration_ms = (end_time - start_time) * 1000
        
        # Get current GPU memory usage
        gpu_memory_used = 0.0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024**2)
        
        # Create pipeline metrics
        pipeline_metric = PipelineMetrics(
            request_id=request_id,
            stage_name=stage_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            input_size=input_size,
            output_size=output_size,
            gpu_memory_used_mb=gpu_memory_used,
            cpu_percent=psutil.cpu_percent(),
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        self.pipeline_metrics.append(pipeline_metric)
        self.stage_latencies[stage_name].append(duration_ms)
        
        # Check for latency alerts
        asyncio.create_task(self._check_latency_alert(stage_name, duration_ms))
    
    def end_request_tracking(self, request_id: str, success: bool = True):
        """End tracking a request"""
        
        if request_id not in self.active_requests:
            return
        
        end_time = time.time()
        start_time = self.active_requests[request_id]['start_time']
        total_duration = end_time - start_time
        
        self.request_times.append(total_duration * 1000)  # Convert to ms
        
        # Update throughput metrics
        asyncio.create_task(self._update_throughput_metrics())
        
        # Clean up
        del self.active_requests[request_id]
    
    async def _check_latency_alert(self, stage_name: str, duration_ms: float):
        """Check for latency performance alerts"""
        
        try:
            if duration_ms > self.config.latency_critical_ms:
                await self._create_latency_alert(
                    "critical", stage_name, duration_ms, self.config.latency_critical_ms,
                    f"Critical latency in {stage_name}: {duration_ms:.1f}ms",
                    [
                        "Check GPU memory usage",
                        "Reduce batch size",
                        "Optimize model configuration",
                        "Check system resources"
                    ]
                )
            elif duration_ms > self.config.latency_warning_ms:
                await self._create_latency_alert(
                    "warning", stage_name, duration_ms, self.config.latency_warning_ms,
                    f"High latency in {stage_name}: {duration_ms:.1f}ms",
                    [
                        "Monitor system performance",
                        "Check for resource contention",
                        "Consider optimization"
                    ]
                )
        
        except Exception as e:
            self.logger.error(f"Latency alert checking error: {e}")
    
    async def _create_latency_alert(self, severity: str, stage_name: str,
                                  current_value: float, threshold_value: float,
                                  message: str, recommendations: List[str]):
        """Create latency alert"""
        
        alert = PerformanceAlert(
            severity=severity,
            category="latency",
            metric_name=f"latency_{stage_name}",
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            recommendations=recommendations
        )
        
        self.latency_alerts.append(alert)
        
        # Log alert
        log_level = logging.CRITICAL if severity == "critical" else logging.WARNING
        self.logger.log(log_level, f"LATENCY ALERT: {message}")
    
    async def _update_throughput_metrics(self):
        """Update throughput metrics"""
        
        try:
            current_time = datetime.now()
            
            # Calculate requests per second (last minute)
            minute_ago = time.time() - 60
            recent_requests = [
                req_time for req_time in self.request_times
                if req_time > minute_ago * 1000  # Convert to ms
            ]
            
            rps = len(recent_requests) / 60.0 if recent_requests else 0.0
            
            # Calculate response time percentiles
            if self.request_times:
                response_times = list(self.request_times)
                avg_response_time = statistics.mean(response_times)
                p95_response_time = np.percentile(response_times, 95)
                p99_response_time = np.percentile(response_times, 99)
            else:
                avg_response_time = 0.0
                p95_response_time = 0.0
                p99_response_time = 0.0
            
            # Calculate success rate
            recent_pipeline_metrics = [
                m for m in self.pipeline_metrics
                if (current_time - m.timestamp).total_seconds() < 60
            ]
            
            if recent_pipeline_metrics:
                success_count = sum(1 for m in recent_pipeline_metrics if m.success)
                success_rate = success_count / len(recent_pipeline_metrics)
                error_rate = 1.0 - success_rate
            else:
                success_rate = 1.0
                error_rate = 0.0
            
            throughput_metric = ThroughputMetrics(
                requests_per_second=rps,
                concurrent_requests=len(self.active_requests),
                queue_length=0,  # Would need to be tracked separately
                average_response_time_ms=avg_response_time,
                p95_response_time_ms=p95_response_time,
                p99_response_time_ms=p99_response_time,
                success_rate=success_rate,
                error_rate=error_rate,
                total_requests=len(self.request_times)
            )
            
            self.throughput_history.append(throughput_metric)
        
        except Exception as e:
            self.logger.error(f"Throughput metrics update error: {e}")
    
    def get_stage_performance_summary(self, stage_name: str, minutes: int = 10) -> Dict[str, Any]:
        """Get performance summary for a specific stage"""
        
        if stage_name not in self.stage_latencies:
            return {}
        
        latencies = list(self.stage_latencies[stage_name])
        if not latencies:
            return {}
        
        # Filter to recent data
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [
            m for m in self.pipeline_metrics
            if m.stage_name == stage_name and m.timestamp > cutoff_time
        ]
        
        recent_latencies = [m.duration_ms for m in recent_metrics]
        if not recent_latencies:
            recent_latencies = latencies[-100:]  # Fallback to last 100 samples
        
        return {
            "stage_name": stage_name,
            "sample_count": len(recent_latencies),
            "avg_latency_ms": statistics.mean(recent_latencies),
            "min_latency_ms": min(recent_latencies),
            "max_latency_ms": max(recent_latencies),
            "p50_latency_ms": np.percentile(recent_latencies, 50),
            "p95_latency_ms": np.percentile(recent_latencies, 95),
            "p99_latency_ms": np.percentile(recent_latencies, 99),
            "success_rate": sum(1 for m in recent_metrics if m.success) / len(recent_metrics) if recent_metrics else 1.0,
            "throughput_ops_per_sec": len(recent_metrics) / (minutes * 60) if recent_metrics else 0.0
        }
    
    def get_current_throughput(self) -> Optional[ThroughputMetrics]:
        """Get current throughput metrics"""
        
        return self.throughput_history[-1] if self.throughput_history else None

# ============================================================================
# PERFORMANCE OPTIMIZATION ENGINE
# ============================================================================

class PerformanceOptimizationEngine:
    """Intelligent performance optimization engine"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization tracking
        self.recommendations: List[OptimizationRecommendation] = []
        self.applied_optimizations: Dict[str, Any] = {}
        self.performance_baselines: Dict[str, float] = {}
        
        # Optimization strategies
        self.optimization_strategies = {
            "gpu_memory": self._optimize_gpu_memory,
            "batch_size": self._optimize_batch_size,
            "model_caching": self._optimize_model_caching,
            "audio_buffering": self._optimize_audio_buffering,
            "concurrent_processing": self._optimize_concurrent_processing
        }
    
    async def analyze_performance(self, gpu_monitor: GPUPerformanceMonitor,
                                latency_monitor: LatencyPerformanceMonitor) -> List[OptimizationRecommendation]:
        """Analyze performance and generate optimization recommendations"""
        
        recommendations = []
        
        try:
            # Analyze GPU performance
            gpu_recommendations = await self._analyze_gpu_performance(gpu_monitor)
            recommendations.extend(gpu_recommendations)
            
            # Analyze latency performance
            latency_recommendations = await self._analyze_latency_performance(latency_monitor)
            recommendations.extend(latency_recommendations)
            
            # Analyze throughput performance
            throughput_recommendations = await self._analyze_throughput_performance(latency_monitor)
            recommendations.extend(throughput_recommendations)
            
            # Store recommendations
            self.recommendations.extend(recommendations)
            
            # Auto-apply high-priority recommendations if enabled
            if self.config.enable_auto_optimization:
                await self._auto_apply_optimizations(recommendations)
        
        except Exception as e:
            self.logger.error(f"Performance analysis error: {e}")
        
        return recommendations
    
    async def _analyze_gpu_performance(self, gpu_monitor: GPUPerformanceMonitor) -> List[OptimizationRecommendation]:
        """Analyze GPU performance and generate recommendations"""
        
        recommendations = []
        
        try:
            current_metrics = gpu_monitor.get_current_gpu_metrics()
            
            for metrics in current_metrics:
                # High memory usage
                if metrics.memory_percent > 90:
                    recommendations.append(OptimizationRecommendation(
                        category="gpu_memory",
                        priority="high",
                        title=f"GPU {metrics.gpu_id} Memory Optimization",
                        description=f"GPU {metrics.gpu_id} memory usage is {metrics.memory_percent:.1f}%",
                        expected_improvement="10-30% memory reduction",
                        implementation_effort="medium",
                        auto_applicable=True
                    ))
                
                # Low utilization
                if metrics.utilization_percent < 50:
                    recommendations.append(OptimizationRecommendation(
                        category="batch_size",
                        priority="medium",
                        title=f"GPU {metrics.gpu_id} Utilization Optimization",
                        description=f"GPU {metrics.gpu_id} utilization is only {metrics.utilization_percent:.1f}%",
                        expected_improvement="20-50% throughput increase",
                        implementation_effort="low",
                        auto_applicable=True
                    ))
        
        except Exception as e:
            self.logger.error(f"GPU performance analysis error: {e}")
        
        return recommendations
    
    async def _analyze_latency_performance(self, latency_monitor: LatencyPerformanceMonitor) -> List[OptimizationRecommendation]:
        """Analyze latency performance and generate recommendations"""
        
        recommendations = []
        
        try:
            # Analyze each pipeline stage
            stages = ["asr", "llm", "tts", "audio_processing"]
            
            for stage in stages:
                summary = latency_monitor.get_stage_performance_summary(stage)
                
                if not summary:
                    continue
                
                # High latency
                if summary["avg_latency_ms"] > self.config.latency_warning_ms:
                    recommendations.append(OptimizationRecommendation(
                        category="latency",
                        priority="high" if summary["avg_latency_ms"] > self.config.latency_critical_ms else "medium",
                        title=f"{stage.upper()} Latency Optimization",
                        description=f"{stage} average latency is {summary['avg_latency_ms']:.1f}ms",
                        expected_improvement="20-40% latency reduction",
                        implementation_effort="medium",
                        auto_applicable=False
                    ))
                
                # High P99 latency
                if summary["p99_latency_ms"] > summary["avg_latency_ms"] * 3:
                    recommendations.append(OptimizationRecommendation(
                        category="consistency",
                        priority="medium",
                        title=f"{stage.upper()} Consistency Optimization",
                        description=f"{stage} P99 latency ({summary['p99_latency_ms']:.1f}ms) is much higher than average",
                        expected_improvement="Improved latency consistency",
                        implementation_effort="high",
                        auto_applicable=False
                    ))
        
        except Exception as e:
            self.logger.error(f"Latency performance analysis error: {e}")
        
        return recommendations
    
    async def _analyze_throughput_performance(self, latency_monitor: LatencyPerformanceMonitor) -> List[OptimizationRecommendation]:
        """Analyze throughput performance and generate recommendations"""
        
        recommendations = []
        
        try:
            current_throughput = latency_monitor.get_current_throughput()
            
            if not current_throughput:
                return recommendations
            
            # Low throughput
            if current_throughput.requests_per_second < self.config.throughput_target_rps:
                recommendations.append(OptimizationRecommendation(
                    category="throughput",
                    priority="high",
                    title="Throughput Optimization",
                    description=f"Current throughput ({current_throughput.requests_per_second:.1f} RPS) below target ({self.config.throughput_target_rps} RPS)",
                    expected_improvement="50-100% throughput increase",
                    implementation_effort="medium",
                    auto_applicable=True
                ))
            
            # High error rate
            if current_throughput.error_rate > 0.05:  # 5% error rate
                recommendations.append(OptimizationRecommendation(
                    category="reliability",
                    priority="critical",
                    title="Error Rate Optimization",
                    description=f"High error rate: {current_throughput.error_rate:.1%}",
                    expected_improvement="Improved system reliability",
                    implementation_effort="high",
                    auto_applicable=False
                ))
        
        except Exception as e:
            self.logger.error(f"Throughput performance analysis error: {e}")
        
        return recommendations
    
    async def _auto_apply_optimizations(self, recommendations: List[OptimizationRecommendation]):
        """Automatically apply applicable optimizations"""
        
        for recommendation in recommendations:
            if not recommendation.auto_applicable:
                continue
            
            if recommendation.priority not in ["high", "critical"]:
                continue
            
            try:
                strategy = self.optimization_strategies.get(recommendation.category)
                if strategy:
                    success = await strategy(recommendation)
                    if success:
                        recommendation.applied = True
                        self.logger.info(f"Auto-applied optimization: {recommendation.title}")
                    else:
                        self.logger.warning(f"Failed to auto-apply optimization: {recommendation.title}")
            
            except Exception as e:
                self.logger.error(f"Auto-optimization error for {recommendation.title}: {e}")
    
    async def _optimize_gpu_memory(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize GPU memory usage"""
        
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.logger.info("Applied GPU memory optimization: cleared cache")
            return True
        
        except Exception as e:
            self.logger.error(f"GPU memory optimization error: {e}")
            return False
    
    async def _optimize_batch_size(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize batch size for better GPU utilization"""
        
        try:
            # This would integrate with your model configuration
            # For now, just log the recommendation
            self.logger.info("Batch size optimization recommended - manual implementation needed")
            return True
        
        except Exception as e:
            self.logger.error(f"Batch size optimization error: {e}")
            return False
    
    async def _optimize_model_caching(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize model caching strategy"""
        
        try:
            # This would implement model caching optimizations
            self.logger.info("Model caching optimization recommended - manual implementation needed")
            return True
        
        except Exception as e:
            self.logger.error(f"Model caching optimization error: {e}")
            return False
    
    async def _optimize_audio_buffering(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize audio buffering for streaming"""
        
        try:
            # This would implement audio buffering optimizations
            self.logger.info("Audio buffering optimization recommended - manual implementation needed")
            return True
        
        except Exception as e:
            self.logger.error(f"Audio buffering optimization error: {e}")
            return False
    
    async def _optimize_concurrent_processing(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize concurrent processing"""
        
        try:
            # This would implement concurrent processing optimizations
            self.logger.info("Concurrent processing optimization recommended - manual implementation needed")
            return True
        
        except Exception as e:
            self.logger.error(f"Concurrent processing optimization error: {e}")
            return False

# ============================================================================
# MAIN PERFORMANCE MONITOR
# ============================================================================

class AdvancedPerformanceMonitor:
    """Main advanced performance monitoring system"""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.gpu_monitor = GPUPerformanceMonitor(self.config)
        self.latency_monitor = LatencyPerformanceMonitor(self.config)
        self.optimization_engine = PerformanceOptimizationEngine(self.config)
        
        # Monitoring state
        self.is_running = False
        self.monitoring_tasks = []
    
    async def start_monitoring(self):
        """Start all performance monitoring"""
        
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting advanced performance monitoring")
        
        # Start GPU monitoring
        await self.gpu_monitor.start_monitoring()
        
        # Start optimization analysis task
        self.monitoring_tasks = [
            asyncio.create_task(self._optimization_analysis_loop())
        ]
    
    async def stop_monitoring(self):
        """Stop all performance monitoring"""
        
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("Stopping advanced performance monitoring")
        
        # Stop GPU monitoring
        await self.gpu_monitor.stop_monitoring()
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
    
    async def _optimization_analysis_loop(self):
        """Optimization analysis loop"""
        
        while self.is_running:
            try:
                # Run performance analysis
                recommendations = await self.optimization_engine.analyze_performance(
                    self.gpu_monitor, self.latency_monitor
                )
                
                if recommendations:
                    self.logger.info(f"Generated {len(recommendations)} performance recommendations")
                
                await asyncio.sleep(self.config.optimization_check_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization analysis error: {e}")
                await asyncio.sleep(self.config.optimization_check_interval)
    
    # Convenience methods for request tracking
    def start_request(self, request_id: str, metadata: Dict[str, Any] = None) -> str:
        """Start tracking a request"""
        return self.latency_monitor.start_request_tracking(request_id, metadata)
    
    def start_stage(self, request_id: str, stage_name: str) -> str:
        """Start tracking a pipeline stage"""
        return self.latency_monitor.start_stage_tracking(request_id, stage_name)
    
    def end_stage(self, request_id: str, stage_name: str, success: bool = True,
                  error_message: str = None, input_size: int = 0, output_size: int = 0,
                  metadata: Dict[str, Any] = None):
        """End tracking a pipeline stage"""
        self.latency_monitor.end_stage_tracking(
            request_id, stage_name, success, error_message, input_size, output_size, metadata
        )
    
    def end_request(self, request_id: str, success: bool = True):
        """End tracking a request"""
        self.latency_monitor.end_request_tracking(request_id, success)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "gpu_metrics": [m.dict() for m in self.gpu_monitor.get_current_gpu_metrics()],
            "throughput_metrics": self.latency_monitor.get_current_throughput().dict() if self.latency_monitor.get_current_throughput() else None,
            "stage_performance": {
                stage: self.latency_monitor.get_stage_performance_summary(stage)
                for stage in ["asr", "llm", "tts", "audio_processing"]
            },
            "active_alerts": [
                alert.dict() for alert in self.gpu_monitor.gpu_alerts + self.latency_monitor.latency_alerts
                if not alert.resolved
            ],
            "optimization_recommendations": [
                rec.dict() for rec in self.optimization_engine.recommendations[-10:]  # Last 10
            ],
            "system_status": {
                "monitoring_active": self.is_running,
                "gpu_count": self.gpu_monitor.gpu_count,
                "active_requests": len(self.latency_monitor.active_requests)
            }
        }

# Export main components
__all__ = [
    'PerformanceConfig', 'AdvancedPerformanceMonitor',
    'GPUPerformanceMonitor', 'LatencyPerformanceMonitor', 'PerformanceOptimizationEngine',
    'GPUMetrics', 'PipelineMetrics', 'ThroughputMetrics', 'PerformanceAlert', 'OptimizationRecommendation'
]