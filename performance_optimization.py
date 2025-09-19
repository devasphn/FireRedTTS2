#!/usr/bin/env python3
"""
Performance Optimization System
Advanced performance optimization for FireRedTTS2 including model caching,
GPU memory optimization, audio buffering, and connection pooling
"""

import asyncio
import time
import json
import logging
import threading
import weakref
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from pathlib import Path
import gc
import os

import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil

# ============================================================================
# OPTIMIZATION CONFIGURATION
# ============================================================================

@dataclass
class OptimizationConfig:
    """Performance optimization configuration"""
    
    # Model caching settings
    enable_model_caching: bool = True
    max_cached_models: int = 3
    model_cache_size_gb: float = 8.0
    model_lazy_loading: bool = True
    model_preload_popular: bool = True
    
    # GPU memory optimization
    enable_gpu_optimization: bool = True
    gpu_memory_fraction: float = 0.9
    enable_memory_growth: bool = True
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    
    # Audio buffering optimization
    enable_audio_buffering: bool = True
    audio_buffer_size_ms: int = 1000
    audio_chunk_size_ms: int = 100
    max_audio_buffers: int = 10
    enable_adaptive_buffering: bool = True
    
    # Connection pooling
    enable_connection_pooling: bool = True
    max_connections_per_pool: int = 20
    connection_timeout_seconds: int = 30
    pool_cleanup_interval_seconds: int = 300
    
    # Batch processing optimization
    enable_batch_processing: bool = True
    max_batch_size: int = 8
    batch_timeout_ms: int = 50
    adaptive_batch_sizing: bool = True
    
    # Concurrent processing
    max_worker_threads: int = 4
    max_worker_processes: int = 2
    enable_async_processing: bool = True
    
    # Caching strategies
    enable_result_caching: bool = True
    result_cache_size_mb: int = 500
    cache_ttl_seconds: int = 3600

# ============================================================================
# MODEL CACHE MANAGER
# ============================================================================

class ModelCacheManager:
    """Advanced model caching and lazy loading system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model cache
        self.cached_models: OrderedDict = OrderedDict()
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.model_usage_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'load_count': 0,
            'last_used': datetime.now(),
            'total_inference_time': 0.0,
            'memory_usage_mb': 0.0
        })
        
        # Loading state
        self.loading_models: Dict[str, asyncio.Event] = {}
        self.model_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_memory_used = 0.0
        
        # Background cleanup task
        self.cleanup_task = None
        self.is_running = False
    
    async def start(self):
        """Start the model cache manager"""
        
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Model cache manager started")
    
    async def stop(self):
        """Stop the model cache manager"""
        
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all cached models
        await self._clear_all_models()
        self.logger.info("Model cache manager stopped")
    
    async def get_model(self, model_name: str, model_path: str, 
                       loader_func: Callable, **loader_kwargs) -> Any:
        """Get model from cache or load it"""
        
        async with self.model_locks[model_name]:
            # Check if model is in cache
            if model_name in self.cached_models:
                self.cache_hits += 1
                self._update_usage_stats(model_name)
                
                # Move to end (most recently used)
                model = self.cached_models[model_name]
                del self.cached_models[model_name]
                self.cached_models[model_name] = model
                
                self.logger.debug(f"Model cache hit: {model_name}")
                return model
            
            # Check if model is currently being loaded
            if model_name in self.loading_models:
                self.logger.debug(f"Waiting for model to load: {model_name}")
                await self.loading_models[model_name].wait()
                
                if model_name in self.cached_models:
                    return self.cached_models[model_name]
            
            # Load the model
            self.cache_misses += 1
            self.loading_models[model_name] = asyncio.Event()
            
            try:
                self.logger.info(f"Loading model: {model_name}")
                start_time = time.time()
                
                # Load model in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                model = await loop.run_in_executor(None, loader_func, model_path, **loader_kwargs)
                
                load_time = time.time() - start_time
                
                # Calculate model memory usage
                model_memory = self._calculate_model_memory(model)
                
                # Check if we need to free space
                await self._ensure_cache_space(model_memory)
                
                # Cache the model
                self.cached_models[model_name] = model
                self.total_memory_used += model_memory
                
                # Update metadata
                self.model_metadata[model_name] = {
                    'path': model_path,
                    'load_time': load_time,
                    'memory_mb': model_memory,
                    'loaded_at': datetime.now()
                }
                
                self._update_usage_stats(model_name)
                
                self.logger.info(f"Model loaded: {model_name} ({model_memory:.1f}MB, {load_time:.2f}s)")
                
                return model
            
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                raise
            
            finally:
                # Signal that loading is complete
                if model_name in self.loading_models:
                    self.loading_models[model_name].set()
                    del self.loading_models[model_name]
    
    async def preload_models(self, model_configs: List[Dict[str, Any]]):
        """Preload popular models"""
        
        if not self.config.model_preload_popular:
            return
        
        self.logger.info(f"Preloading {len(model_configs)} models")
        
        for config in model_configs:
            try:
                await self.get_model(
                    config['name'],
                    config['path'],
                    config['loader_func'],
                    **config.get('loader_kwargs', {})
                )
            except Exception as e:
                self.logger.error(f"Failed to preload model {config['name']}: {e}")
    
    async def _ensure_cache_space(self, required_memory_mb: float):
        """Ensure there's enough cache space for a new model"""
        
        max_memory_mb = self.config.model_cache_size_gb * 1024
        
        while (self.total_memory_used + required_memory_mb > max_memory_mb and 
               len(self.cached_models) > 0):
            
            # Remove least recently used model
            lru_model_name = next(iter(self.cached_models))
            await self._remove_model(lru_model_name)
    
    async def _remove_model(self, model_name: str):
        """Remove model from cache"""
        
        if model_name not in self.cached_models:
            return
        
        model = self.cached_models[model_name]
        memory_freed = self.model_metadata.get(model_name, {}).get('memory_mb', 0)
        
        # Remove from cache
        del self.cached_models[model_name]
        self.total_memory_used -= memory_freed
        
        # Clean up GPU memory if it's a torch model
        if hasattr(model, 'cpu'):
            model.cpu()
        
        del model
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info(f"Removed model from cache: {model_name} (freed {memory_freed:.1f}MB)")
    
    async def _clear_all_models(self):
        """Clear all cached models"""
        
        model_names = list(self.cached_models.keys())
        for model_name in model_names:
            await self._remove_model(model_name)
    
    def _calculate_model_memory(self, model) -> float:
        """Calculate model memory usage in MB"""
        
        try:
            if hasattr(model, 'parameters'):
                # PyTorch model
                total_params = sum(p.numel() for p in model.parameters())
                # Assume 4 bytes per parameter (float32)
                memory_mb = (total_params * 4) / (1024 * 1024)
                return memory_mb
            else:
                # Fallback: estimate based on object size
                import sys
                return sys.getsizeof(model) / (1024 * 1024)
        
        except Exception as e:
            self.logger.warning(f"Could not calculate model memory: {e}")
            return 100.0  # Default estimate
    
    def _update_usage_stats(self, model_name: str):
        """Update model usage statistics"""
        
        stats = self.model_usage_stats[model_name]
        stats['load_count'] += 1
        stats['last_used'] = datetime.now()
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        
        while self.is_running:
            try:
                await self._cleanup_unused_models()
                await asyncio.sleep(300)  # Run every 5 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Model cache cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_unused_models(self):
        """Clean up unused models"""
        
        current_time = datetime.now()
        unused_threshold = timedelta(hours=1)  # Remove models unused for 1 hour
        
        models_to_remove = []
        
        for model_name, stats in self.model_usage_stats.items():
            if (current_time - stats['last_used']) > unused_threshold:
                models_to_remove.append(model_name)
        
        for model_name in models_to_remove:
            if model_name in self.cached_models:
                await self._remove_model(model_name)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cached_models": len(self.cached_models),
            "total_memory_used_mb": self.total_memory_used,
            "max_memory_mb": self.config.model_cache_size_gb * 1024,
            "memory_utilization": self.total_memory_used / (self.config.model_cache_size_gb * 1024),
            "model_stats": dict(self.model_usage_stats)
        }

# ============================================================================
# GPU MEMORY OPTIMIZER
# ============================================================================

class GPUMemoryOptimizer:
    """Advanced GPU memory optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Memory tracking
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.peak_memory_usage = 0.0
        self.optimization_applied = False
        
        # Initialize GPU optimization
        self._initialize_gpu_optimization()
    
    def _initialize_gpu_optimization(self):
        """Initialize GPU optimization settings"""
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available - GPU optimization disabled")
            return
        
        try:
            # Set memory fraction
            if self.config.gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
                self.logger.info(f"Set GPU memory fraction to {self.config.gpu_memory_fraction}")
            
            # Enable memory growth (PyTorch equivalent)
            if self.config.enable_memory_growth:
                # PyTorch manages memory growth automatically
                pass
            
            # Configure mixed precision
            if self.config.enable_mixed_precision:
                # This would be configured per model/training loop
                self.logger.info("Mixed precision enabled")
            
            self.optimization_applied = True
            
        except Exception as e:
            self.logger.error(f"GPU optimization initialization failed: {e}")
    
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Optimize model for memory efficiency"""
        
        try:
            # Enable gradient checkpointing if supported
            if self.config.enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                self.logger.info("Enabled gradient checkpointing")
            
            # Move model to GPU with optimal settings
            if torch.cuda.is_available():
                model = model.cuda()
                
                # Enable mixed precision if configured
                if self.config.enable_mixed_precision:
                    model = model.half()  # Convert to FP16
                    self.logger.info("Converted model to FP16")
            
            return model
        
        except Exception as e:
            self.logger.error(f"Model memory optimization failed: {e}")
            return model
    
    def create_memory_snapshot(self, label: str = ""):
        """Create memory usage snapshot"""
        
        if not torch.cuda.is_available():
            return
        
        try:
            snapshot = {
                "timestamp": datetime.now(),
                "label": label,
                "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2)
            }
            
            self.memory_snapshots.append(snapshot)
            
            # Update peak usage
            current_peak = snapshot["max_allocated_mb"]
            if current_peak > self.peak_memory_usage:
                self.peak_memory_usage = current_peak
            
            # Keep only recent snapshots
            if len(self.memory_snapshots) > 100:
                self.memory_snapshots.pop(0)
        
        except Exception as e:
            self.logger.error(f"Memory snapshot failed: {e}")
    
    def optimize_memory_usage(self):
        """Optimize current memory usage"""
        
        try:
            if torch.cuda.is_available():
                # Clear cache
                torch.cuda.empty_cache()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                
                self.logger.info("GPU memory optimized")
        
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        try:
            stats = {
                "gpu_available": True,
                "current_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "current_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
                "total_memory_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2),
                "memory_utilization": torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory,
                "optimization_applied": self.optimization_applied,
                "snapshot_count": len(self.memory_snapshots)
            }
            
            return stats
        
        except Exception as e:
            self.logger.error(f"Memory stats collection failed: {e}")
            return {"gpu_available": True, "error": str(e)}

# ============================================================================
# AUDIO BUFFER OPTIMIZER
# ============================================================================

class AudioBufferOptimizer:
    """Advanced audio buffering optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Buffer pools
        self.input_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.output_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.buffer_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Buffer statistics
        self.buffer_stats = {
            "allocations": 0,
            "deallocations": 0,
            "reuses": 0,
            "peak_usage": 0,
            "current_usage": 0
        }
        
        # Adaptive buffering
        self.buffer_usage_history: deque = deque(maxlen=100)
        self.optimal_buffer_sizes: Dict[str, int] = {}
    
    async def get_input_buffer(self, buffer_type: str, size: int, dtype: np.dtype = np.float32) -> np.ndarray:
        """Get optimized input buffer"""
        
        async with self.buffer_locks[buffer_type]:
            # Try to reuse existing buffer
            available_buffers = self.input_buffers[buffer_type]
            
            for i, buffer in enumerate(available_buffers):
                if buffer.size >= size and buffer.dtype == dtype:
                    # Reuse buffer
                    reused_buffer = available_buffers.pop(i)
                    self.buffer_stats["reuses"] += 1
                    
                    # Resize if necessary
                    if reused_buffer.size > size:
                        return reused_buffer[:size]
                    else:
                        return reused_buffer
            
            # Create new buffer
            new_buffer = np.zeros(size, dtype=dtype)
            self.buffer_stats["allocations"] += 1
            self.buffer_stats["current_usage"] += 1
            
            if self.buffer_stats["current_usage"] > self.buffer_stats["peak_usage"]:
                self.buffer_stats["peak_usage"] = self.buffer_stats["current_usage"]
            
            return new_buffer
    
    async def return_input_buffer(self, buffer_type: str, buffer: np.ndarray):
        """Return input buffer to pool"""
        
        async with self.buffer_locks[buffer_type]:
            # Only keep buffers if we haven't exceeded the limit
            if len(self.input_buffers[buffer_type]) < self.config.max_audio_buffers:
                self.input_buffers[buffer_type].append(buffer)
            else:
                self.buffer_stats["deallocations"] += 1
                self.buffer_stats["current_usage"] -= 1
    
    async def get_output_buffer(self, buffer_type: str, size: int, dtype: np.dtype = np.float32) -> np.ndarray:
        """Get optimized output buffer"""
        
        return await self.get_input_buffer(f"output_{buffer_type}", size, dtype)
    
    async def return_output_buffer(self, buffer_type: str, buffer: np.ndarray):
        """Return output buffer to pool"""
        
        await self.return_input_buffer(f"output_{buffer_type}", buffer)
    
    def optimize_buffer_sizes(self):
        """Optimize buffer sizes based on usage patterns"""
        
        if not self.config.enable_adaptive_buffering:
            return
        
        try:
            # Analyze usage patterns
            for buffer_type, buffers in self.input_buffers.items():
                if buffers:
                    sizes = [buf.size for buf in buffers]
                    optimal_size = int(np.percentile(sizes, 75))  # 75th percentile
                    self.optimal_buffer_sizes[buffer_type] = optimal_size
            
            self.logger.debug(f"Optimized buffer sizes: {self.optimal_buffer_sizes}")
        
        except Exception as e:
            self.logger.error(f"Buffer size optimization failed: {e}")
    
    def clear_buffers(self):
        """Clear all buffers"""
        
        for buffer_type in list(self.input_buffers.keys()):
            self.input_buffers[buffer_type].clear()
        
        for buffer_type in list(self.output_buffers.keys()):
            self.output_buffers[buffer_type].clear()
        
        self.buffer_stats["current_usage"] = 0
        self.logger.info("Cleared all audio buffers")
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        
        total_buffers = sum(len(buffers) for buffers in self.input_buffers.values())
        total_buffers += sum(len(buffers) for buffers in self.output_buffers.values())
        
        return {
            **self.buffer_stats,
            "total_buffers": total_buffers,
            "buffer_types": len(self.input_buffers) + len(self.output_buffers),
            "optimal_sizes": self.optimal_buffer_sizes,
            "reuse_rate": self.buffer_stats["reuses"] / max(self.buffer_stats["allocations"], 1)
        }

# ============================================================================
# CONNECTION POOL MANAGER
# ============================================================================

class ConnectionPoolManager:
    """Advanced connection pooling for concurrent requests"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Connection pools
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_worker_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_worker_processes) if config.max_worker_processes > 0 else None
        
        # Connection tracking
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "completed_connections": 0,
            "failed_connections": 0,
            "average_duration": 0.0
        }
        
        # Cleanup task
        self.cleanup_task = None
        self.is_running = False
    
    async def start(self):
        """Start connection pool manager"""
        
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Connection pool manager started")
    
    async def stop(self):
        """Stop connection pool manager"""
        
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown pools
        self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        self.logger.info("Connection pool manager stopped")
    
    async def execute_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in thread pool"""
        
        connection_id = f"thread_{int(time.time() * 1000000)}"
        
        try:
            # Track connection
            self.active_connections[connection_id] = {
                "type": "thread",
                "start_time": time.time(),
                "function": func.__name__
            }
            
            self.connection_stats["total_connections"] += 1
            self.connection_stats["active_connections"] += 1
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
            
            # Update stats
            duration = time.time() - self.active_connections[connection_id]["start_time"]
            self._update_connection_stats(duration, True)
            
            return result
        
        except Exception as e:
            self._update_connection_stats(0, False)
            self.logger.error(f"Thread execution failed: {e}")
            raise
        
        finally:
            # Clean up connection tracking
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
                self.connection_stats["active_connections"] -= 1
    
    async def execute_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in process pool"""
        
        if not self.process_pool:
            raise RuntimeError("Process pool not available")
        
        connection_id = f"process_{int(time.time() * 1000000)}"
        
        try:
            # Track connection
            self.active_connections[connection_id] = {
                "type": "process",
                "start_time": time.time(),
                "function": func.__name__
            }
            
            self.connection_stats["total_connections"] += 1
            self.connection_stats["active_connections"] += 1
            
            # Execute in process pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.process_pool, func, *args, **kwargs)
            
            # Update stats
            duration = time.time() - self.active_connections[connection_id]["start_time"]
            self._update_connection_stats(duration, True)
            
            return result
        
        except Exception as e:
            self._update_connection_stats(0, False)
            self.logger.error(f"Process execution failed: {e}")
            raise
        
        finally:
            # Clean up connection tracking
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
                self.connection_stats["active_connections"] -= 1
    
    def _update_connection_stats(self, duration: float, success: bool):
        """Update connection statistics"""
        
        if success:
            self.connection_stats["completed_connections"] += 1
            
            # Update average duration
            total_completed = self.connection_stats["completed_connections"]
            current_avg = self.connection_stats["average_duration"]
            self.connection_stats["average_duration"] = (
                (current_avg * (total_completed - 1) + duration) / total_completed
            )
        else:
            self.connection_stats["failed_connections"] += 1
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        
        while self.is_running:
            try:
                await self._cleanup_stale_connections()
                await asyncio.sleep(self.config.pool_cleanup_interval_seconds)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Connection cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_stale_connections(self):
        """Clean up stale connections"""
        
        current_time = time.time()
        timeout = self.config.connection_timeout_seconds
        
        stale_connections = []
        
        for conn_id, conn_info in self.active_connections.items():
            if current_time - conn_info["start_time"] > timeout:
                stale_connections.append(conn_id)
        
        for conn_id in stale_connections:
            self.logger.warning(f"Cleaning up stale connection: {conn_id}")
            if conn_id in self.active_connections:
                del self.active_connections[conn_id]
                self.connection_stats["active_connections"] -= 1
                self.connection_stats["failed_connections"] += 1
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        
        return {
            **self.connection_stats,
            "thread_pool_size": self.config.max_worker_threads,
            "process_pool_size": self.config.max_worker_processes,
            "success_rate": (
                self.connection_stats["completed_connections"] / 
                max(self.connection_stats["total_connections"], 1)
            )
        }

# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchProcessor:
    """Intelligent batch processing for improved throughput"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Batch queues
        self.batch_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.batch_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.batch_events: Dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        
        # Processing tasks
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Batch statistics
        self.batch_stats = {
            "total_batches": 0,
            "total_items": 0,
            "average_batch_size": 0.0,
            "processing_time": 0.0
        }
        
        # Adaptive batch sizing
        self.optimal_batch_sizes: Dict[str, int] = {}
        self.batch_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
    
    async def add_to_batch(self, batch_type: str, item: Dict[str, Any], 
                          processor_func: Callable) -> Any:
        """Add item to batch for processing"""
        
        # Create future for result
        result_future = asyncio.Future()
        item["result_future"] = result_future
        item["processor_func"] = processor_func
        
        async with self.batch_locks[batch_type]:
            self.batch_queues[batch_type].append(item)
            
            # Start processing task if not running
            if batch_type not in self.processing_tasks:
                self.processing_tasks[batch_type] = asyncio.create_task(
                    self._process_batch_queue(batch_type)
                )
            
            # Signal that new item is available
            self.batch_events[batch_type].set()
        
        # Wait for result
        return await result_future
    
    async def _process_batch_queue(self, batch_type: str):
        """Process batch queue for specific type"""
        
        while True:
            try:
                # Wait for items or timeout
                try:
                    await asyncio.wait_for(
                        self.batch_events[batch_type].wait(),
                        timeout=self.config.batch_timeout_ms / 1000.0
                    )
                except asyncio.TimeoutError:
                    pass
                
                # Get batch to process
                async with self.batch_locks[batch_type]:
                    if not self.batch_queues[batch_type]:
                        self.batch_events[batch_type].clear()
                        continue
                    
                    # Determine batch size
                    max_batch_size = self._get_optimal_batch_size(batch_type)
                    batch_items = self.batch_queues[batch_type][:max_batch_size]
                    self.batch_queues[batch_type] = self.batch_queues[batch_type][max_batch_size:]
                    
                    # Clear event if queue is empty
                    if not self.batch_queues[batch_type]:
                        self.batch_events[batch_type].clear()
                
                if batch_items:
                    await self._process_batch(batch_type, batch_items)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Batch processing error for {batch_type}: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, batch_type: str, batch_items: List[Dict[str, Any]]):
        """Process a batch of items"""
        
        start_time = time.time()
        
        try:
            # Group items by processor function
            processor_groups = defaultdict(list)
            for item in batch_items:
                processor_func = item["processor_func"]
                processor_groups[processor_func].append(item)
            
            # Process each group
            for processor_func, items in processor_groups.items():
                try:
                    # Extract inputs
                    inputs = [item.get("input") for item in items]
                    
                    # Process batch
                    results = await self._call_batch_processor(processor_func, inputs)
                    
                    # Set results
                    for item, result in zip(items, results):
                        if not item["result_future"].done():
                            item["result_future"].set_result(result)
                
                except Exception as e:
                    # Set error for all items in this group
                    for item in items:
                        if not item["result_future"].done():
                            item["result_future"].set_exception(e)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_batch_stats(batch_type, len(batch_items), processing_time)
        
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            
            # Set error for all items
            for item in batch_items:
                if not item["result_future"].done():
                    item["result_future"].set_exception(e)
    
    async def _call_batch_processor(self, processor_func: Callable, inputs: List[Any]) -> List[Any]:
        """Call batch processor function"""
        
        try:
            # Check if function supports batch processing
            if hasattr(processor_func, 'process_batch'):
                return await processor_func.process_batch(inputs)
            else:
                # Process individually
                results = []
                for input_item in inputs:
                    result = await processor_func(input_item)
                    results.append(result)
                return results
        
        except Exception as e:
            self.logger.error(f"Batch processor call failed: {e}")
            raise
    
    def _get_optimal_batch_size(self, batch_type: str) -> int:
        """Get optimal batch size for batch type"""
        
        if not self.config.adaptive_batch_sizing:
            return self.config.max_batch_size
        
        # Use learned optimal size or default
        return self.optimal_batch_sizes.get(batch_type, self.config.max_batch_size)
    
    def _update_batch_stats(self, batch_type: str, batch_size: int, processing_time: float):
        """Update batch processing statistics"""
        
        # Global stats
        self.batch_stats["total_batches"] += 1
        self.batch_stats["total_items"] += batch_size
        self.batch_stats["processing_time"] += processing_time
        
        total_batches = self.batch_stats["total_batches"]
        self.batch_stats["average_batch_size"] = (
            self.batch_stats["total_items"] / total_batches
        )
        
        # Per-type performance tracking
        performance_record = {
            "batch_size": batch_size,
            "processing_time": processing_time,
            "throughput": batch_size / processing_time if processing_time > 0 else 0
        }
        
        self.batch_performance_history[batch_type].append(performance_record)
        
        # Update optimal batch size
        if self.config.adaptive_batch_sizing:
            self._update_optimal_batch_size(batch_type)
    
    def _update_optimal_batch_size(self, batch_type: str):
        """Update optimal batch size based on performance history"""
        
        history = self.batch_performance_history[batch_type]
        
        if len(history) < 10:  # Need enough data
            return
        
        # Find batch size with best throughput
        best_throughput = 0
        best_batch_size = self.config.max_batch_size
        
        for record in history:
            if record["throughput"] > best_throughput:
                best_throughput = record["throughput"]
                best_batch_size = record["batch_size"]
        
        self.optimal_batch_sizes[batch_type] = min(best_batch_size, self.config.max_batch_size)
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        
        return {
            **self.batch_stats,
            "optimal_batch_sizes": self.optimal_batch_sizes,
            "active_queues": len(self.batch_queues),
            "queue_lengths": {
                batch_type: len(queue) 
                for batch_type, queue in self.batch_queues.items()
            }
        }

# ============================================================================
# MAIN PERFORMANCE OPTIMIZER
# ============================================================================

class PerformanceOptimizer:
    """Main performance optimization system"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_cache = ModelCacheManager(self.config) if self.config.enable_model_caching else None
        self.gpu_optimizer = GPUMemoryOptimizer(self.config) if self.config.enable_gpu_optimization else None
        self.audio_optimizer = AudioBufferOptimizer(self.config) if self.config.enable_audio_buffering else None
        self.connection_pool = ConnectionPoolManager(self.config) if self.config.enable_connection_pooling else None
        self.batch_processor = BatchProcessor(self.config) if self.config.enable_batch_processing else None
        
        # Optimization state
        self.is_running = False
        self.optimization_tasks = []
    
    async def start(self):
        """Start performance optimization system"""
        
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting performance optimization system")
        
        # Start components
        if self.model_cache:
            await self.model_cache.start()
        
        if self.connection_pool:
            await self.connection_pool.start()
        
        # Start optimization tasks
        self.optimization_tasks = [
            asyncio.create_task(self._optimization_loop())
        ]
    
    async def stop(self):
        """Stop performance optimization system"""
        
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("Stopping performance optimization system")
        
        # Stop optimization tasks
        for task in self.optimization_tasks:
            task.cancel()
        
        await asyncio.gather(*self.optimization_tasks, return_exceptions=True)
        self.optimization_tasks.clear()
        
        # Stop components
        if self.model_cache:
            await self.model_cache.stop()
        
        if self.connection_pool:
            await self.connection_pool.stop()
    
    async def _optimization_loop(self):
        """Background optimization loop"""
        
        while self.is_running:
            try:
                # Run periodic optimizations
                await self._run_periodic_optimizations()
                await asyncio.sleep(300)  # Run every 5 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)
    
    async def _run_periodic_optimizations(self):
        """Run periodic optimization tasks"""
        
        try:
            # GPU memory optimization
            if self.gpu_optimizer:
                self.gpu_optimizer.optimize_memory_usage()
            
            # Audio buffer optimization
            if self.audio_optimizer:
                self.audio_optimizer.optimize_buffer_sizes()
            
            # System garbage collection
            gc.collect()
            
            self.logger.debug("Periodic optimizations completed")
        
        except Exception as e:
            self.logger.error(f"Periodic optimization error: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "optimization_active": self.is_running,
            "config": {
                "model_caching": self.config.enable_model_caching,
                "gpu_optimization": self.config.enable_gpu_optimization,
                "audio_buffering": self.config.enable_audio_buffering,
                "connection_pooling": self.config.enable_connection_pooling,
                "batch_processing": self.config.enable_batch_processing
            }
        }
        
        # Add component statistics
        if self.model_cache:
            summary["model_cache"] = self.model_cache.get_cache_stats()
        
        if self.gpu_optimizer:
            summary["gpu_memory"] = self.gpu_optimizer.get_memory_stats()
        
        if self.audio_optimizer:
            summary["audio_buffers"] = self.audio_optimizer.get_buffer_stats()
        
        if self.connection_pool:
            summary["connection_pool"] = self.connection_pool.get_pool_stats()
        
        if self.batch_processor:
            summary["batch_processing"] = self.batch_processor.get_batch_stats()
        
        return summary

# Export main components
__all__ = [
    'OptimizationConfig', 'PerformanceOptimizer',
    'ModelCacheManager', 'GPUMemoryOptimizer', 'AudioBufferOptimizer',
    'ConnectionPoolManager', 'BatchProcessor'
]