#!/usr/bin/env python3
"""
Performance Monitoring System for FireRedTTS2
Real-time monitoring of system resources, model performance, and audio quality
"""

import time
import json
import logging
import threading
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

import numpy as np
import torch
import psutil

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_mb: float
    network_recv_mb: float

@dataclass
class GPUMetrics:
    """GPU-specific metrics"""
    timestamp: float
    gpu_id: int
    gpu_name: str
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature_c: float
    power_draw_w: float
    power_limit_w: float

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    timestamp: float
    model_name: str
    inference_time_ms: float
    tokens_per_second: float
    memory_usage_mb: float
    batch_size: int
    sequence_length: int
    temperature: float
    top_k: int

@dataclass
class AudioMetrics:
    """Audio quality and processing metrics"""
    timestamp: float
    sample_rate: int
    duration_seconds: float
    rms_energy: float
    peak_amplitude: float
    snr_db: float
    thd_percent: float
    processing_time_ms: float
    quality_score: float

class SystemMonitor:
    """Monitors system resources"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.network_baseline = None
    
    def start_monitoring(self):
        """Start system monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(5.0)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network = psutil.net_io_counters()
        if self.network_baseline is None:
            self.network_baseline = (network.bytes_sent, network.bytes_recv)
            network_sent_mb = 0.0
            network_recv_mb = 0.0
        else:
            network_sent_mb = (network.bytes_sent - self.network_baseline[0]) / (1024 * 1024)
            network_recv_mb = (network.bytes_recv - self.network_baseline[1]) / (1024 * 1024)
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_percent=disk.percent,
            disk_used_gb=disk.used / (1024**3),
            disk_total_gb=disk.total / (1024**3),
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the latest system metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, duration_minutes: int = 10) -> List[SystemMetrics]:
        """Get metrics history for the specified duration"""
        cutoff_time = time.time() - (duration_minutes * 60)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

class GPUMonitor:
    """Monitors GPU resources and performance"""
    
    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=500)
        self.nvidia_smi_available = self._check_nvidia_smi()
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            subprocess.run(['nvidia-smi', '--version'], 
                         capture_output=True, check=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def start_monitoring(self):
        """Start GPU monitoring"""
        if torch.cuda.is_available() and not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Main GPU monitoring loop"""
        while self.is_monitoring:
            try:
                for gpu_id in range(torch.cuda.device_count()):
                    metrics = self._collect_gpu_metrics(gpu_id)
                    if metrics:
                        self.metrics_history.append(metrics)
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                time.sleep(5.0)
    
    def _collect_gpu_metrics(self, gpu_id: int) -> Optional[GPUMetrics]:
        """Collect metrics for a specific GPU"""
        
        try:
            # Basic PyTorch metrics
            device = torch.device(f'cuda:{gpu_id}')
            props = torch.cuda.get_device_properties(gpu_id)
            
            memory_allocated = torch.cuda.memory_allocated(gpu_id)
            memory_reserved = torch.cuda.memory_reserved(gpu_id)
            memory_total = props.total_memory
            
            # Try to get detailed metrics from nvidia-smi
            utilization = 0.0
            temperature = 0.0
            power_draw = 0.0
            power_limit = 0.0
            
            if self.nvidia_smi_available:
                try:
                    cmd = [
                        'nvidia-smi',
                        '--query-gpu=utilization.gpu,temperature.gpu,power.draw,power.limit',
                        '--format=csv,noheader,nounits',
                        f'--id={gpu_id}'
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        values = result.stdout.strip().split(', ')
                        if len(values) >= 4:
                            utilization = float(values[0])
                            temperature = float(values[1])
                            power_draw = float(values[2]) if values[2] != '[N/A]' else 0.0
                            power_limit = float(values[3]) if values[3] != '[N/A]' else 0.0
                
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
                    pass
            
            return GPUMetrics(
                timestamp=time.time(),
                gpu_id=gpu_id,
                gpu_name=props.name,
                utilization_percent=utilization,
                memory_used_mb=memory_allocated / (1024**2),
                memory_total_mb=memory_total / (1024**2),
                memory_percent=(memory_allocated / memory_total) * 100,
                temperature_c=temperature,
                power_draw_w=power_draw,
                power_limit_w=power_limit
            )
            
        except Exception as e:
            logger.error(f"Failed to collect GPU {gpu_id} metrics: {e}")
            return None
    
    def get_latest_metrics(self, gpu_id: int = 0) -> Optional[GPUMetrics]:
        """Get the latest GPU metrics for specified GPU"""
        gpu_metrics = [m for m in self.metrics_history if m.gpu_id == gpu_id]
        return gpu_metrics[-1] if gpu_metrics else None
    
    def get_all_latest_metrics(self) -> List[GPUMetrics]:
        """Get latest metrics for all GPUs"""
        latest_metrics = {}
        for metrics in reversed(self.metrics_history):
            if metrics.gpu_id not in latest_metrics:
                latest_metrics[metrics.gpu_id] = metrics
        return list(latest_metrics.values())

class ModelPerformanceTracker:
    """Tracks model inference performance"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.current_inference = None
        self.inference_start_time = None
    
    def start_inference(self, model_name: str, batch_size: int = 1, 
                       sequence_length: int = 0, temperature: float = 1.0, 
                       top_k: int = 50):
        """Start tracking an inference operation"""
        self.current_inference = {
            "model_name": model_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "temperature": temperature,
            "top_k": top_k
        }
        self.inference_start_time = time.time()
    
    def end_inference(self, tokens_generated: int = 0) -> Optional[ModelMetrics]:
        """End tracking and record metrics"""
        if self.current_inference is None or self.inference_start_time is None:
            return None
        
        end_time = time.time()
        inference_time_ms = (end_time - self.inference_start_time) * 1000
        
        # Calculate tokens per second
        tokens_per_second = tokens_generated / (inference_time_ms / 1000) if inference_time_ms > 0 else 0
        
        # Get current GPU memory usage
        memory_usage_mb = 0.0
        if torch.cuda.is_available():
            memory_usage_mb = torch.cuda.memory_allocated() / (1024**2)
        
        metrics = ModelMetrics(
            timestamp=end_time,
            model_name=self.current_inference["model_name"],
            inference_time_ms=inference_time_ms,
            tokens_per_second=tokens_per_second,
            memory_usage_mb=memory_usage_mb,
            batch_size=self.current_inference["batch_size"],
            sequence_length=self.current_inference["sequence_length"],
            temperature=self.current_inference["temperature"],
            top_k=self.current_inference["top_k"]
        )
        
        self.metrics_history.append(metrics)
        
        # Reset current inference
        self.current_inference = None
        self.inference_start_time = None
        
        return metrics
    
    def get_latest_metrics(self) -> Optional[ModelMetrics]:
        """Get the latest model metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, duration_minutes: int = 10) -> Dict[str, float]:
        """Get average metrics over specified duration"""
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        return {
            "avg_inference_time_ms": np.mean([m.inference_time_ms for m in recent_metrics]),
            "avg_tokens_per_second": np.mean([m.tokens_per_second for m in recent_metrics]),
            "avg_memory_usage_mb": np.mean([m.memory_usage_mb for m in recent_metrics]),
            "total_inferences": len(recent_metrics)
        }

class AudioQualityAnalyzer:
    """Analyzes audio quality metrics"""
    
    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
    
    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int, 
                     processing_time_ms: float = 0.0) -> AudioMetrics:
        """Analyze audio quality and record metrics"""
        
        # Ensure audio is 1D
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Basic audio metrics
        duration_seconds = len(audio_data) / sample_rate
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        peak_amplitude = np.max(np.abs(audio_data))
        
        # Signal-to-noise ratio (simplified)
        signal_power = np.mean(audio_data ** 2)
        noise_floor = np.percentile(np.abs(audio_data), 10) ** 2
        snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-10)) if noise_floor > 0 else 60.0
        
        # Total harmonic distortion (simplified estimate)
        # This is a very basic approximation
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft)
        fundamental_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        
        # Estimate THD as ratio of harmonics to fundamental
        harmonics_power = 0.0
        fundamental_power = magnitude[fundamental_freq_idx] ** 2
        
        for harmonic in range(2, 6):  # Check first few harmonics
            harmonic_idx = fundamental_freq_idx * harmonic
            if harmonic_idx < len(magnitude):
                harmonics_power += magnitude[harmonic_idx] ** 2
        
        thd_percent = (harmonics_power / (fundamental_power + 1e-10)) * 100 if fundamental_power > 0 else 0.0
        
        # Overall quality score (0-1, higher is better)
        quality_score = self._calculate_quality_score(
            rms_energy, peak_amplitude, snr_db, thd_percent, duration_seconds
        )
        
        metrics = AudioMetrics(
            timestamp=time.time(),
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            rms_energy=float(rms_energy),
            peak_amplitude=float(peak_amplitude),
            snr_db=float(snr_db),
            thd_percent=float(thd_percent),
            processing_time_ms=processing_time_ms,
            quality_score=float(quality_score)
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_quality_score(self, rms_energy: float, peak_amplitude: float, 
                                snr_db: float, thd_percent: float, 
                                duration_seconds: float) -> float:
        """Calculate overall audio quality score"""
        
        # Normalize individual metrics to 0-1 scale
        
        # RMS energy (good range: 0.01 - 0.3)
        rms_score = np.clip((rms_energy - 0.001) / 0.299, 0, 1)
        
        # Peak amplitude (should be close to but not exceed 1.0)
        peak_score = 1.0 - np.clip(abs(peak_amplitude - 0.8) / 0.2, 0, 1)
        
        # SNR (good range: 20-60 dB)
        snr_score = np.clip((snr_db - 20) / 40, 0, 1)
        
        # THD (lower is better, good range: 0-5%)
        thd_score = 1.0 - np.clip(thd_percent / 10, 0, 1)
        
        # Duration (penalize very short audio)
        duration_score = np.clip(duration_seconds / 2.0, 0, 1)
        
        # Weighted average
        quality_score = (
            0.2 * rms_score +
            0.2 * peak_score +
            0.3 * snr_score +
            0.2 * thd_score +
            0.1 * duration_score
        )
        
        return quality_score
    
    def get_latest_metrics(self) -> Optional[AudioMetrics]:
        """Get the latest audio metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_quality_score(self, duration_minutes: int = 10) -> float:
        """Get average quality score over specified duration"""
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return 0.0
        
        return np.mean([m.quality_score for m in recent_metrics])

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, 
                 system_update_interval: float = 1.0,
                 gpu_update_interval: float = 2.0,
                 storage_dir: str = "/workspace/logs"):
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize monitors
        self.system_monitor = SystemMonitor(system_update_interval)
        self.gpu_monitor = GPUMonitor(gpu_update_interval)
        self.model_tracker = ModelPerformanceTracker()
        self.audio_analyzer = AudioQualityAnalyzer()
        
        # Overall status
        self.is_monitoring = False
        self.start_time = None
    
    def start_monitoring(self):
        """Start all monitoring systems"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.start_time = time.time()
            
            self.system_monitor.start_monitoring()
            self.gpu_monitor.start_monitoring()
            
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems"""
        if self.is_monitoring:
            self.is_monitoring = False
            
            self.system_monitor.stop_monitoring()
            self.gpu_monitor.stop_monitoring()
            
            logger.info("Performance monitoring stopped")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            "monitoring_active": self.is_monitoring,
            "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
            "timestamp": time.time()
        }
        
        # System metrics
        system_metrics = self.system_monitor.get_latest_metrics()
        if system_metrics:
            status["system"] = {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "memory_used_gb": system_metrics.memory_used_gb,
                "memory_total_gb": system_metrics.memory_total_gb,
                "disk_percent": system_metrics.disk_percent,
                "disk_used_gb": system_metrics.disk_used_gb,
                "disk_total_gb": system_metrics.disk_total_gb
            }
        
        # GPU metrics
        gpu_metrics = self.gpu_monitor.get_all_latest_metrics()
        if gpu_metrics:
            status["gpus"] = []
            for gpu in gpu_metrics:
                status["gpus"].append({
                    "gpu_id": gpu.gpu_id,
                    "name": gpu.gpu_name,
                    "utilization_percent": gpu.utilization_percent,
                    "memory_percent": gpu.memory_percent,
                    "memory_used_mb": gpu.memory_used_mb,
                    "memory_total_mb": gpu.memory_total_mb,
                    "temperature_c": gpu.temperature_c,
                    "power_draw_w": gpu.power_draw_w
                })
        
        # Model performance
        model_metrics = self.model_tracker.get_latest_metrics()
        if model_metrics:
            status["model"] = {
                "last_inference_time_ms": model_metrics.inference_time_ms,
                "tokens_per_second": model_metrics.tokens_per_second,
                "memory_usage_mb": model_metrics.memory_usage_mb
            }
        
        # Add average metrics
        avg_model_metrics = self.model_tracker.get_average_metrics(10)
        if avg_model_metrics:
            status["model_averages"] = avg_model_metrics
        
        # Audio quality
        audio_metrics = self.audio_analyzer.get_latest_metrics()
        if audio_metrics:
            status["audio"] = {
                "last_quality_score": audio_metrics.quality_score,
                "last_snr_db": audio_metrics.snr_db,
                "last_processing_time_ms": audio_metrics.processing_time_ms
            }
        
        avg_quality = self.audio_analyzer.get_average_quality_score(10)
        status["audio_quality_avg"] = avg_quality
        
        return status
    
    def save_metrics_snapshot(self) -> str:
        """Save current metrics to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_snapshot_{timestamp}.json"
        filepath = self.storage_dir / filename
        
        try:
            status = self.get_comprehensive_status()
            
            with open(filepath, 'w') as f:
                json.dump(status, f, indent=2, default=str)
            
            logger.info(f"Metrics snapshot saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save metrics snapshot: {e}")
            return ""

# Global performance monitor instance
performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global performance_monitor
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor()
    return performance_monitor