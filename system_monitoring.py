#!/usr/bin/env python3

"""
System Monitoring and Health Checks
Comprehensive monitoring system for FireRedTTS2 with real-time metrics and automated recovery
"""

import asyncio
import logging
import time
import json
import threading
import subprocess
import socket
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque, defaultdict
import statistics

import psutil
import torch
import numpy as np
from pydantic import BaseModel, Field

from error_handling_system import ErrorManager, SystemHealth, ErrorSeverity, ErrorCategory
from data_models import LatencyMetrics, ResourceUsage, QualityMetrics

# ============================================================================
# MONITORING CONFIGURATION
# ============================================================================

@dataclass
class MonitoringConfig:
    """Configuration for system monitoring"""
    
    # Monitoring intervals (seconds)
    health_check_interval: float = 30.0
    metrics_collection_interval: float = 5.0
    performance_check_interval: float = 10.0
    
    # Thresholds
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 95.0
    memory_warning_threshold: float = 85.0
    memory_critical_threshold: float = 95.0
    gpu_memory_warning_threshold: float = 90.0
    gpu_memory_critical_threshold: float = 98.0
    disk_warning_threshold: float = 85.0
    disk_critical_threshold: float = 95.0
    
    # Latency thresholds (milliseconds)
    latency_warning_threshold: float = 1000.0
    latency_critical_threshold: float = 3000.0
    
    # History retention
    metrics_history_size: int = 1000
    alert_history_size: int = 500
    
    # Recovery settings
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_cooldown: float = 300.0  # 5 minutes

# ============================================================================
# MONITORING DATA MODELS
# ============================================================================

class ServiceStatus(BaseModel):
    """Status of individual service"""
    
    name: str
    status: str = "unknown"  # running, stopped, error, starting, stopping
    pid: Optional[int] = None
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    uptime_seconds: float = 0.0
    last_check: datetime = Field(default_factory=datetime.now)
    error_count: int = 0
    restart_count: int = 0

class NetworkStatus(BaseModel):
    """Network connectivity status"""
    
    external_connectivity: bool = True
    dns_resolution: bool = True
    api_endpoints_reachable: bool = True
    websocket_connections: int = 0
    active_sessions: int = 0
    bandwidth_in_mbps: float = 0.0
    bandwidth_out_mbps: float = 0.0
    last_check: datetime = Field(default_factory=datetime.now)

class ModelStatus(BaseModel):
    """AI model status"""
    
    model_name: str
    loaded: bool = False
    memory_usage_mb: float = 0.0
    inference_count: int = 0
    average_inference_time_ms: float = 0.0
    error_rate: float = 0.0
    last_inference: Optional[datetime] = None
    load_time_ms: float = 0.0

class AlertInfo(BaseModel):
    """System alert information"""
    
    alert_id: str
    severity: str  # info, warning, critical
    category: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SystemMetrics(BaseModel):
    """Comprehensive system metrics"""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Resource usage
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    
    # GPU metrics
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_utilization: List[float] = Field(default_factory=list)
    gpu_memory_used: List[float] = Field(default_factory=list)
    gpu_memory_total: List[float] = Field(default_factory=list)
    gpu_temperature: List[float] = Field(default_factory=list)
    
    # Network metrics
    network_in_mbps: float = 0.0
    network_out_mbps: float = 0.0
    connections_count: int = 0
    
    # Application metrics
    active_sessions: int = 0
    active_requests: int = 0
    total_requests: int = 0
    error_rate: float = 0.0
    
    # Performance metrics
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

# ============================================================================
# SYSTEM MONITOR
# ============================================================================

class SystemMonitor:
    """Main system monitoring class"""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.is_running = False
        self.monitoring_tasks = []
        
        # Data storage
        self.metrics_history = deque(maxlen=self.config.metrics_history_size)
        self.alert_history = deque(maxlen=self.config.alert_history_size)
        self.latency_samples = deque(maxlen=100)
        
        # Service tracking
        self.services: Dict[str, ServiceStatus] = {}
        self.models: Dict[str, ModelStatus] = {}
        
        # Network tracking
        self.network_status = NetworkStatus()
        
        # Recovery tracking
        self.recovery_attempts: Dict[str, int] = defaultdict(int)
        self.last_recovery_time: Dict[str, datetime] = {}
        
        # Error manager
        self.error_manager = ErrorManager.get_instance()
        
        # Initialize baseline metrics
        self._initialize_baseline_metrics()
    
    def _initialize_baseline_metrics(self):
        """Initialize baseline system metrics"""
        try:
            # Get initial system info
            self.system_info = {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_total_gb": psutil.disk_usage('/').total / (1024**3),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()),
                "python_version": f"{psutil.version_info}",
            }
            
            # GPU info
            if torch.cuda.is_available():
                self.system_info["gpu_count"] = torch.cuda.device_count()
                self.system_info["gpu_names"] = [
                    torch.cuda.get_device_name(i) 
                    for i in range(torch.cuda.device_count())
                ]
            else:
                self.system_info["gpu_count"] = 0
                self.system_info["gpu_names"] = []
            
            self.logger.info(f"System monitoring initialized: {self.system_info}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize baseline metrics: {e}")
    
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        if self.is_running:
            self.logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.logger.info("Starting system monitoring")
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._network_monitoring_loop()),
            asyncio.create_task(self._service_monitoring_loop()),
        ]
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.monitoring_tasks)
        except asyncio.CancelledError:
            self.logger.info("Monitoring tasks cancelled")
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("Stopping system monitoring")
        
        # Cancel all tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
    
    async def _health_check_loop(self):
        """Main health check loop"""
        while self.is_running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _metrics_collection_loop(self):
        """Metrics collection loop"""
        while self.is_running:
            try:
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)
                await self._check_thresholds(metrics)
                await asyncio.sleep(self.config.metrics_collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.config.metrics_collection_interval)
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                await self._monitor_performance()
                await asyncio.sleep(self.config.performance_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(self.config.performance_check_interval)
    
    async def _network_monitoring_loop(self):
        """Network monitoring loop"""
        while self.is_running:
            try:
                await self._monitor_network()
                await asyncio.sleep(30.0)  # Check network every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Network monitoring error: {e}")
                await asyncio.sleep(30.0)
    
    async def _service_monitoring_loop(self):
        """Service monitoring loop"""
        while self.is_running:
            try:
                await self._monitor_services()
                await asyncio.sleep(15.0)  # Check services every 15 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Service monitoring error: {e}")
                await asyncio.sleep(15.0)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        health_status = "healthy"
        issues = []
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.cpu_critical_threshold:
                health_status = "critical"
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > self.config.cpu_warning_threshold:
                if health_status == "healthy":
                    health_status = "warning"
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.config.memory_critical_threshold:
                health_status = "critical"
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent > self.config.memory_warning_threshold:
                if health_status == "healthy":
                    health_status = "warning"
                issues.append(f"Memory usage high: {memory.percent:.1f}%")
            
            # Check GPU if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.memory_stats(i)
                    allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                    total = torch.cuda.get_device_properties(i).total_memory
                    gpu_percent = (allocated / total) * 100
                    
                    if gpu_percent > self.config.gpu_memory_critical_threshold:
                        health_status = "critical"
                        issues.append(f"GPU {i} memory critical: {gpu_percent:.1f}%")
                    elif gpu_percent > self.config.gpu_memory_warning_threshold:
                        if health_status == "healthy":
                            health_status = "warning"
                        issues.append(f"GPU {i} memory high: {gpu_percent:.1f}%")
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > self.config.disk_critical_threshold:
                health_status = "critical"
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > self.config.disk_warning_threshold:
                if health_status == "healthy":
                    health_status = "warning"
                issues.append(f"Disk usage high: {disk_percent:.1f}%")
            
            # Update system health
            system_health = self.error_manager.get_system_health()
            system_health.overall_status = health_status
            system_health.last_check = datetime.now()
            
            # Log health status
            if health_status != "healthy":
                self.logger.warning(f"System health: {health_status} - Issues: {', '.join(issues)}")
            else:
                self.logger.debug("System health check: healthy")
            
            # Trigger recovery if needed
            if health_status == "critical" and self.config.enable_auto_recovery:
                await self._trigger_recovery("system_health", issues)
        
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            await self._create_alert("critical", "health_check", f"Health check failed: {e}")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        
        metrics = SystemMetrics()
        
        try:
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used_gb = memory.used / (1024**3)
            metrics.memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_percent = (disk.used / disk.total) * 100
            metrics.disk_used_gb = disk.used / (1024**3)
            metrics.disk_total_gb = disk.total / (1024**3)
            
            # GPU metrics
            if torch.cuda.is_available():
                metrics.gpu_available = True
                metrics.gpu_count = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    # GPU utilization (approximation)
                    gpu_memory = torch.cuda.memory_stats(i)
                    allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                    total = torch.cuda.get_device_properties(i).total_memory
                    utilization = (allocated / total) * 100
                    
                    metrics.gpu_utilization.append(utilization)
                    metrics.gpu_memory_used.append(allocated / (1024**3))
                    metrics.gpu_memory_total.append(total / (1024**3))
                    
                    # Temperature (if available)
                    try:
                        temp = torch.cuda.temperature(i) if hasattr(torch.cuda, 'temperature') else 0
                        metrics.gpu_temperature.append(temp)
                    except:
                        metrics.gpu_temperature.append(0)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                time_delta = time.time() - self._last_net_time
                bytes_in_delta = net_io.bytes_recv - self._last_net_io.bytes_recv
                bytes_out_delta = net_io.bytes_sent - self._last_net_io.bytes_sent
                
                metrics.network_in_mbps = (bytes_in_delta / time_delta) / (1024**2) * 8
                metrics.network_out_mbps = (bytes_out_delta / time_delta) / (1024**2) * 8
            
            self._last_net_io = net_io
            self._last_net_time = time.time()
            
            # Connection count
            metrics.connections_count = len(psutil.net_connections())
            
            # Calculate latency metrics if available
            if self.latency_samples:
                latencies = list(self.latency_samples)
                metrics.average_latency_ms = statistics.mean(latencies)
                metrics.p95_latency_ms = np.percentile(latencies, 95)
                metrics.p99_latency_ms = np.percentile(latencies, 99)
        
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    async def _check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against thresholds and create alerts"""
        
        # CPU threshold check
        if metrics.cpu_percent > self.config.cpu_critical_threshold:
            await self._create_alert("critical", "cpu", f"CPU usage critical: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent > self.config.cpu_warning_threshold:
            await self._create_alert("warning", "cpu", f"CPU usage high: {metrics.cpu_percent:.1f}%")
        
        # Memory threshold check
        if metrics.memory_percent > self.config.memory_critical_threshold:
            await self._create_alert("critical", "memory", f"Memory usage critical: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent > self.config.memory_warning_threshold:
            await self._create_alert("warning", "memory", f"Memory usage high: {metrics.memory_percent:.1f}%")
        
        # GPU threshold check
        for i, gpu_util in enumerate(metrics.gpu_utilization):
            if gpu_util > self.config.gpu_memory_critical_threshold:
                await self._create_alert("critical", "gpu", f"GPU {i} memory critical: {gpu_util:.1f}%")
            elif gpu_util > self.config.gpu_memory_warning_threshold:
                await self._create_alert("warning", "gpu", f"GPU {i} memory high: {gpu_util:.1f}%")
        
        # Latency threshold check
        if metrics.average_latency_ms > self.config.latency_critical_threshold:
            await self._create_alert("critical", "latency", f"Average latency critical: {metrics.average_latency_ms:.1f}ms")
        elif metrics.average_latency_ms > self.config.latency_warning_threshold:
            await self._create_alert("warning", "latency", f"Average latency high: {metrics.average_latency_ms:.1f}ms")
    
    async def _monitor_performance(self):
        """Monitor application performance"""
        try:
            # This would integrate with your application's performance metrics
            # For now, we'll simulate some basic checks
            
            # Check if services are responding
            for service_name, service in self.services.items():
                if service.status == "running":
                    # Simulate health check
                    response_time = await self._check_service_health(service_name)
                    if response_time > 5000:  # 5 seconds
                        await self._create_alert("warning", "performance", 
                                               f"Service {service_name} slow response: {response_time}ms")
        
        except Exception as e:
            self.logger.error(f"Performance monitoring error: {e}")
    
    async def _monitor_network(self):
        """Monitor network connectivity"""
        try:
            # Check external connectivity
            self.network_status.external_connectivity = await self._check_internet_connectivity()
            
            # Check DNS resolution
            self.network_status.dns_resolution = await self._check_dns_resolution()
            
            # Update timestamp
            self.network_status.last_check = datetime.now()
            
            # Create alerts for network issues
            if not self.network_status.external_connectivity:
                await self._create_alert("critical", "network", "External connectivity lost")
            
            if not self.network_status.dns_resolution:
                await self._create_alert("warning", "network", "DNS resolution issues")
        
        except Exception as e:
            self.logger.error(f"Network monitoring error: {e}")
    
    async def _monitor_services(self):
        """Monitor registered services"""
        try:
            for service_name, service in self.services.items():
                # Check if process is running
                if service.pid:
                    try:
                        process = psutil.Process(service.pid)
                        if process.is_running():
                            service.status = "running"
                            service.cpu_percent = process.cpu_percent()
                            service.memory_mb = process.memory_info().rss / (1024**2)
                            service.uptime_seconds = time.time() - process.create_time()
                        else:
                            service.status = "stopped"
                            await self._create_alert("critical", "service", f"Service {service_name} stopped")
                    except psutil.NoSuchProcess:
                        service.status = "stopped"
                        await self._create_alert("critical", "service", f"Service {service_name} not found")
                
                service.last_check = datetime.now()
        
        except Exception as e:
            self.logger.error(f"Service monitoring error: {e}")
    
    async def _check_service_health(self, service_name: str) -> float:
        """Check individual service health"""
        # This would implement actual health checks for your services
        # For now, return a simulated response time
        return 100.0  # milliseconds
    
    async def _check_internet_connectivity(self) -> bool:
        """Check internet connectivity"""
        try:
            # Try to connect to a reliable external service
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('8.8.8.8', 53))  # Google DNS
            sock.close()
            return result == 0
        except:
            return False
    
    async def _check_dns_resolution(self) -> bool:
        """Check DNS resolution"""
        try:
            socket.gethostbyname('google.com')
            return True
        except:
            return False
    
    async def _create_alert(self, severity: str, category: str, message: str, metadata: Dict[str, Any] = None):
        """Create system alert"""
        
        alert = AlertInfo(
            alert_id=f"{category}_{int(time.time())}",
            severity=severity,
            category=category,
            message=message,
            metadata=metadata or {}
        )
        
        self.alert_history.append(alert)
        
        # Log alert
        log_level = logging.CRITICAL if severity == "critical" else logging.WARNING
        self.logger.log(log_level, f"ALERT [{severity.upper()}] {category}: {message}")
        
        # Trigger recovery if critical
        if severity == "critical" and self.config.enable_auto_recovery:
            await self._trigger_recovery(category, [message])
    
    async def _trigger_recovery(self, category: str, issues: List[str]):
        """Trigger automated recovery actions"""
        
        # Check recovery cooldown
        if category in self.last_recovery_time:
            time_since_last = datetime.now() - self.last_recovery_time[category]
            if time_since_last.total_seconds() < self.config.recovery_cooldown:
                self.logger.info(f"Recovery for {category} in cooldown")
                return
        
        # Check max attempts
        if self.recovery_attempts[category] >= self.config.max_recovery_attempts:
            self.logger.error(f"Max recovery attempts reached for {category}")
            return
        
        self.recovery_attempts[category] += 1
        self.last_recovery_time[category] = datetime.now()
        
        self.logger.info(f"Triggering recovery for {category} (attempt {self.recovery_attempts[category]})")
        
        try:
            if category == "memory":
                await self._recover_memory_issues()
            elif category == "gpu":
                await self._recover_gpu_issues()
            elif category == "cpu":
                await self._recover_cpu_issues()
            elif category == "service":
                await self._recover_service_issues()
            else:
                await self._generic_recovery()
        
        except Exception as e:
            self.logger.error(f"Recovery failed for {category}: {e}")
    
    async def _recover_memory_issues(self):
        """Recover from memory issues"""
        self.logger.info("Attempting memory recovery")
        
        # Clear Python garbage collection
        import gc
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Could also implement:
        # - Reduce batch sizes
        # - Clear application caches
        # - Restart non-critical services
    
    async def _recover_gpu_issues(self):
        """Recover from GPU issues"""
        self.logger.info("Attempting GPU recovery")
        
        if torch.cuda.is_available():
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Reset GPU if possible
            try:
                torch.cuda.reset_peak_memory_stats()
            except:
                pass
    
    async def _recover_cpu_issues(self):
        """Recover from CPU issues"""
        self.logger.info("Attempting CPU recovery")
        
        # Could implement:
        # - Reduce thread counts
        # - Lower process priorities
        # - Pause non-critical tasks
    
    async def _recover_service_issues(self):
        """Recover from service issues"""
        self.logger.info("Attempting service recovery")
        
        # Could implement:
        # - Restart failed services
        # - Switch to backup services
        # - Graceful degradation
    
    async def _generic_recovery(self):
        """Generic recovery actions"""
        self.logger.info("Attempting generic recovery")
        
        # Clear caches
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def register_service(self, name: str, pid: int = None):
        """Register a service for monitoring"""
        self.services[name] = ServiceStatus(name=name, pid=pid)
        self.logger.info(f"Registered service for monitoring: {name}")
    
    def register_model(self, name: str, memory_usage_mb: float = 0.0):
        """Register a model for monitoring"""
        self.models[name] = ModelStatus(name=name, memory_usage_mb=memory_usage_mb, loaded=True)
        self.logger.info(f"Registered model for monitoring: {name}")
    
    def record_latency(self, latency_ms: float):
        """Record latency measurement"""
        self.latency_samples.append(latency_ms)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent system metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get metrics history for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_active_alerts(self) -> List[AlertInfo]:
        """Get active (unresolved) alerts"""
        return [alert for alert in self.alert_history if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        for alert in self.alert_history:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                self.logger.info(f"Alert resolved: {alert_id}")
                break
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_metrics = self.get_current_metrics()
        active_alerts = self.get_active_alerts()
        
        return {
            "monitoring_active": self.is_running,
            "system_info": self.system_info,
            "current_metrics": current_metrics.dict() if current_metrics else None,
            "services": {name: service.dict() for name, service in self.services.items()},
            "models": {name: model.dict() for name, model in self.models.items()},
            "network_status": self.network_status.dict(),
            "active_alerts": [alert.dict() for alert in active_alerts],
            "alert_summary": {
                "total": len(active_alerts),
                "critical": len([a for a in active_alerts if a.severity == "critical"]),
                "warning": len([a for a in active_alerts if a.severity == "warning"]),
            }
        }

# ============================================================================
# DIAGNOSTIC TOOLS
# ============================================================================

class DiagnosticTools:
    """Diagnostic and troubleshooting tools"""
    
    @staticmethod
    async def run_system_diagnostics() -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {},
            "resource_usage": {},
            "gpu_info": {},
            "network_info": {},
            "process_info": {},
            "recommendations": []
        }
        
        try:
            # System information
            diagnostics["system_info"] = {
                "platform": psutil.LINUX if hasattr(psutil, 'LINUX') else "unknown",
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                "uptime_hours": (time.time() - psutil.boot_time()) / 3600
            }
            
            # Resource usage
            diagnostics["resource_usage"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
            
            # GPU information
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_stats = torch.cuda.memory_stats(i)
                    
                    gpu_info.append({
                        "device_id": i,
                        "name": props.name,
                        "total_memory_gb": props.total_memory / (1024**3),
                        "allocated_memory_gb": memory_stats.get('allocated_bytes.all.current', 0) / (1024**3),
                        "cached_memory_gb": memory_stats.get('reserved_bytes.all.current', 0) / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}"
                    })
                
                diagnostics["gpu_info"] = gpu_info
            
            # Network information
            diagnostics["network_info"] = {
                "connections": len(psutil.net_connections()),
                "interfaces": list(psutil.net_if_addrs().keys()),
                "io_counters": psutil.net_io_counters()._asdict()
            }
            
            # Process information
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if proc.info['cpu_percent'] > 5 or proc.info['memory_percent'] > 5:
                        processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            diagnostics["process_info"] = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:10]
            
            # Generate recommendations
            recommendations = []
            
            if diagnostics["resource_usage"]["memory_percent"] > 85:
                recommendations.append("High memory usage detected. Consider reducing batch sizes or clearing caches.")
            
            if diagnostics["resource_usage"]["cpu_percent"] > 80:
                recommendations.append("High CPU usage detected. Consider reducing concurrent operations.")
            
            if torch.cuda.is_available():
                for gpu in diagnostics["gpu_info"]:
                    gpu_usage = (gpu["allocated_memory_gb"] / gpu["total_memory_gb"]) * 100
                    if gpu_usage > 90:
                        recommendations.append(f"GPU {gpu['device_id']} memory usage high ({gpu_usage:.1f}%). Consider reducing model size or batch size.")
            
            diagnostics["recommendations"] = recommendations
        
        except Exception as e:
            diagnostics["error"] = str(e)
        
        return diagnostics
    
    @staticmethod
    async def test_model_loading(model_name: str) -> Dict[str, Any]:
        """Test model loading performance"""
        
        result = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "load_time_ms": 0,
            "memory_usage_mb": 0,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            # This would implement actual model loading test
            # For now, simulate the test
            await asyncio.sleep(0.1)  # Simulate loading time
            
            load_time = (time.time() - start_time) * 1000
            result["load_time_ms"] = load_time
            result["success"] = True
            
            # Simulate memory usage
            result["memory_usage_mb"] = 500.0
            
        except Exception as e:
            result["error"] = str(e)
        
        return result

# ============================================================================
# INITIALIZATION
# ============================================================================

# Global monitor instance
_monitor_instance = None

def get_system_monitor(config: MonitoringConfig = None) -> SystemMonitor:
    """Get global system monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor(config)
    return _monitor_instance

# Export main components
__all__ = [
    'MonitoringConfig', 'SystemMonitor', 'DiagnosticTools',
    'ServiceStatus', 'NetworkStatus', 'ModelStatus', 'AlertInfo', 'SystemMetrics',
    'get_system_monitor'
]