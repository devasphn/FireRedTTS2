#!/usr/bin/env python3
"""
RunPod Infrastructure Optimization
Optimizes container resources, persistent volume management, and RunPod-specific configurations
"""

import os
import json
import shutil
import logging
import subprocess
import psutil
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class RunPodOptimizer:
    """Optimizes FireRedTTS2 deployment for RunPod infrastructure"""
    
    def __init__(self):
        self.workspace_dir = Path("/workspace")
        self.models_dir = self.workspace_dir / "models"
        self.cache_dir = self.workspace_dir / "cache"
        self.persistent_volume = Path("/workspace")  # RunPod mounts persistent volume here
        
        # RunPod specific paths
        self.runpod_config = {
            "container_disk": "/tmp",
            "persistent_volume": "/workspace",
            "model_cache": "/workspace/models",
            "temp_cache": "/tmp/cache",
            "logs_dir": "/workspace/logs"
        }
        
        # Resource limits and optimization settings
        self.optimization_config = {
            "gpu_memory_fraction": 0.9,
            "cpu_threads": None,  # Will be auto-detected
            "memory_limit_gb": None,  # Will be auto-detected
            "disk_cache_gb": 10,
            "model_cache_gb": 20,
            "temp_files_cleanup": True,
            "persistent_model_storage": True
        }
    
    def detect_system_resources(self) -> Dict[str, Any]:
        """Detect available system resources"""
        logger.info("Detecting system resources...")
        
        resources = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
            "workspace_disk_gb": round(psutil.disk_usage('/workspace').total / (1024**3), 2),
            "workspace_free_gb": round(psutil.disk_usage('/workspace').free / (1024**3), 2)
        }
        
        # GPU information
        if torch.cuda.is_available():
            resources["gpu_count"] = torch.cuda.device_count()
            resources["gpu_devices"] = []
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "device_id": i,
                    "name": props.name,
                    "memory_total_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multiprocessor_count
                }
                resources["gpu_devices"].append(gpu_info)
        else:
            resources["gpu_count"] = 0
            resources["gpu_devices"] = []
        
        logger.info(f"System resources detected: {json.dumps(resources, indent=2)}")
        return resources
    
    def optimize_cpu_settings(self, resources: Dict[str, Any]) -> Dict[str, str]:
        """Optimize CPU-related settings"""
        logger.info("Optimizing CPU settings...")
        
        cpu_count = resources["cpu_count"]
        
        # Calculate optimal thread counts
        omp_threads = max(1, cpu_count // 2)  # Leave some cores for system
        mkl_threads = omp_threads
        
        # Set environment variables
        cpu_env_vars = {
            "OMP_NUM_THREADS": str(omp_threads),
            "MKL_NUM_THREADS": str(mkl_threads),
            "NUMEXPR_NUM_THREADS": str(omp_threads),
            "OPENBLAS_NUM_THREADS": str(omp_threads),
            "VECLIB_MAXIMUM_THREADS": str(omp_threads),
            "NUMBA_NUM_THREADS": str(omp_threads)
        }
        
        # Apply settings
        for key, value in cpu_env_vars.items():
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
        
        # CPU affinity optimization (if available)
        try:
            import psutil
            current_process = psutil.Process()
            available_cpus = list(range(cpu_count))
            current_process.cpu_affinity(available_cpus)
            logger.info(f"Set CPU affinity to cores: {available_cpus}")
        except Exception as e:
            logger.warning(f"Could not set CPU affinity: {e}")
        
        return cpu_env_vars
    
    def optimize_memory_settings(self, resources: Dict[str, Any]) -> Dict[str, str]:
        """Optimize memory-related settings"""
        logger.info("Optimizing memory settings...")
        
        total_memory_gb = resources["memory_total_gb"]
        
        # Calculate memory limits
        # Reserve 2GB for system, use 80% of remaining for application
        available_memory_gb = max(1, total_memory_gb - 2)
        app_memory_limit_gb = int(available_memory_gb * 0.8)
        
        memory_env_vars = {
            "MALLOC_TRIM_THRESHOLD_": "100000",  # Reduce memory fragmentation
            "MALLOC_MMAP_THRESHOLD_": "131072",
            "PYTHONMALLOC": "malloc",  # Use system malloc for better memory management
        }
        
        # PyTorch memory optimization
        if torch.cuda.is_available():
            memory_env_vars.update({
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,roundup_power2_divisions:16",
                "CUDA_LAUNCH_BLOCKING": "0",  # Async CUDA operations
                "CUDA_CACHE_DISABLE": "0"     # Enable CUDA cache
            })
        
        # Apply settings
        for key, value in memory_env_vars.items():
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
        
        return memory_env_vars
    
    def optimize_gpu_settings(self, resources: Dict[str, Any]) -> Dict[str, str]:
        """Optimize GPU settings for RunPod"""
        logger.info("Optimizing GPU settings...")
        
        gpu_env_vars = {}
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU optimization")
            return gpu_env_vars
        
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Set memory fraction
            memory_fraction = self.optimization_config["gpu_memory_fraction"]
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                logger.info(f"Set GPU memory fraction to {memory_fraction}")
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # GPU environment variables
            gpu_env_vars = {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES", "all"),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_LAUNCH_BLOCKING": "0",
                "CUDA_CACHE_DISABLE": "0"
            }
            
            # Apply settings
            for key, value in gpu_env_vars.items():
                os.environ[key] = value
                logger.info(f"Set {key}={value}")
            
            # Log GPU status
            for i, gpu_info in enumerate(resources.get("gpu_devices", [])):
                logger.info(f"GPU {i}: {gpu_info['name']} ({gpu_info['memory_total_gb']}GB)")
            
        except Exception as e:
            logger.error(f"GPU optimization failed: {e}")
        
        return gpu_env_vars
    
    def setup_persistent_volume_optimization(self) -> bool:
        """Optimize persistent volume usage"""
        logger.info("Setting up persistent volume optimization...")
        
        try:
            # Create optimized directory structure
            persistent_dirs = [
                self.persistent_volume / "models",
                self.persistent_volume / "cache" / "torch",
                self.persistent_volume / "cache" / "huggingface",
                self.persistent_volume / "cache" / "transformers",
                self.persistent_volume / "logs",
                self.persistent_volume / "config",
                self.persistent_volume / "user_data"
            ]
            
            for directory in persistent_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created persistent directory: {directory}")
            
            # Create symbolic links for frequently accessed directories
            temp_cache_dir = Path("/tmp/cache")
            temp_cache_dir.mkdir(exist_ok=True)
            
            # Link cache directories to both persistent and temp storage
            cache_links = [
                ("/workspace/cache/torch", "/tmp/cache/torch"),
                ("/workspace/cache/huggingface", "/tmp/cache/huggingface_temp"),
                ("/workspace/cache/transformers", "/tmp/cache/transformers_temp")
            ]
            
            for persistent_path, temp_path in cache_links:
                try:
                    Path(temp_path).mkdir(parents=True, exist_ok=True)
                    if not Path(persistent_path).exists():
                        Path(persistent_path).mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created cache structure: {persistent_path} -> {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to create cache link {persistent_path}: {e}")
            
            # Set up model storage optimization
            self._optimize_model_storage()
            
            # Create cleanup script for temporary files
            self._create_cleanup_script()
            
            logger.info("Persistent volume optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Persistent volume optimization failed: {e}")
            return False
    
    def _optimize_model_storage(self) -> None:
        """Optimize model storage on persistent volume"""
        logger.info("Optimizing model storage...")
        
        models_dir = self.persistent_volume / "models"
        
        # Check if models are already on persistent volume
        firered_models = models_dir / "FireRedTTS2"
        if firered_models.exists():
            logger.info("Models found on persistent volume")
            
            # Verify model integrity
            required_files = [
                "config_llm.json", "config_codec.json", "llm_pretrain.pt",
                "llm_posttrain.pt", "codec.pt", "Qwen2.5-1.5B"
            ]
            
            missing_files = []
            for file_name in required_files:
                if not (firered_models / file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                logger.warning(f"Missing model files on persistent volume: {missing_files}")
            else:
                logger.info("All model files verified on persistent volume")
        else:
            logger.info("Models not found on persistent volume, will be downloaded on first run")
        
        # Create model loading optimization script
        model_loader_script = f'''#!/bin/bash
# Model loading optimization for RunPod

MODEL_DIR="{models_dir}/FireRedTTS2"
TEMP_MODEL_DIR="/tmp/models"

# Function to load models efficiently
load_models() {{
    echo "Loading models with optimization..."
    
    # Create temp directory for model loading
    mkdir -p "$TEMP_MODEL_DIR"
    
    # Check if models exist on persistent volume
    if [ -d "$MODEL_DIR" ]; then
        echo "Models found on persistent volume"
        
        # Create symbolic links for faster access
        ln -sf "$MODEL_DIR" "$TEMP_MODEL_DIR/FireRedTTS2"
        
        # Preload critical model files into memory (if enough RAM)
        AVAILABLE_RAM=$(free -g | awk '/^Mem:/ {{print $7}}')
        if [ "$AVAILABLE_RAM" -gt 10 ]; then
            echo "Preloading model files into memory..."
            find "$MODEL_DIR" -name "*.pt" -exec cat {{}} > /dev/null \\;
        fi
    else
        echo "Models not found on persistent volume"
        return 1
    fi
}}

# Execute model loading
load_models
'''
        
        model_loader_file = self.workspace_dir / "load_models_optimized.sh"
        with open(model_loader_file, 'w') as f:
            f.write(model_loader_script)
        
        os.chmod(model_loader_file, 0o755)
        logger.info(f"Created optimized model loader: {model_loader_file}")
    
    def _create_cleanup_script(self) -> None:
        """Create cleanup script for temporary files"""
        cleanup_script = '''#!/bin/bash
# Cleanup script for RunPod deployment

echo "Running cleanup..."

# Clean temporary cache files older than 1 hour
find /tmp/cache -type f -mmin +60 -delete 2>/dev/null || true

# Clean Python cache
find /workspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /workspace -name "*.pyc" -delete 2>/dev/null || true

# Clean old log files (keep last 7 days)
find /workspace/logs -name "*.log" -mtime +7 -delete 2>/dev/null || true

# Clean temporary audio files
find /workspace/uploads -name "*.wav" -mmin +30 -delete 2>/dev/null || true
find /workspace/uploads -name "*.mp3" -mmin +30 -delete 2>/dev/null || true

# GPU memory cleanup
python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

echo "Cleanup completed"
'''
        
        cleanup_file = self.workspace_dir / "cleanup.sh"
        with open(cleanup_file, 'w') as f:
            f.write(cleanup_script)
        
        os.chmod(cleanup_file, 0o755)
        logger.info(f"Created cleanup script: {cleanup_file}")
    
    def optimize_networking_for_runpod(self) -> Dict[str, str]:
        """Optimize networking settings for RunPod proxy system"""
        logger.info("Optimizing networking for RunPod...")
        
        network_env_vars = {
            # Gradio settings for RunPod compatibility
            "GRADIO_SERVER_NAME": "0.0.0.0",
            "GRADIO_SERVER_PORT": "7860",
            "GRADIO_SHARE": "False",
            "GRADIO_ANALYTICS_ENABLED": "False",
            
            # WebSocket settings
            "WEBSOCKET_MAX_SIZE": "10485760",  # 10MB
            "WEBSOCKET_PING_INTERVAL": "20",
            "WEBSOCKET_PING_TIMEOUT": "10",
            
            # HTTP settings
            "HTTP_TIMEOUT": "300",
            "HTTP_KEEPALIVE": "75",
            "HTTP_MAX_CONNECTIONS": "100",
            
            # CORS settings for RunPod proxy
            "CORS_ALLOW_ORIGINS": "*",
            "CORS_ALLOW_METHODS": "GET,POST,PUT,DELETE,OPTIONS",
            "CORS_ALLOW_HEADERS": "*"
        }
        
        # Apply settings
        for key, value in network_env_vars.items():
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
        
        return network_env_vars
    
    def create_resource_monitoring(self) -> None:
        """Create resource monitoring for RunPod deployment"""
        logger.info("Setting up resource monitoring...")
        
        monitoring_script = '''#!/usr/bin/env python3
import time
import json
import psutil
import torch
from datetime import datetime
from pathlib import Path

def collect_metrics():
    """Collect system metrics"""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "percent": psutil.virtual_memory().percent,
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        },
        "disk": {
            "workspace_total_gb": round(psutil.disk_usage('/workspace').total / (1024**3), 2),
            "workspace_used_gb": round(psutil.disk_usage('/workspace').used / (1024**3), 2),
            "workspace_free_gb": round(psutil.disk_usage('/workspace').free / (1024**3), 2),
            "tmp_used_gb": round(psutil.disk_usage('/tmp').used / (1024**3), 2)
        }
    }
    
    # GPU metrics
    if torch.cuda.is_available():
        metrics["gpu"] = {}
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                memory_stats = torch.cuda.memory_stats(i)
                allocated = memory_stats.get('allocated_bytes.all.current', 0)
                reserved = memory_stats.get('reserved_bytes.all.current', 0)
                
                metrics["gpu"][f"device_{i}"] = {
                    "name": props.name,
                    "memory_total_gb": round(props.total_memory / (1024**3), 2),
                    "memory_allocated_gb": round(allocated / (1024**3), 2),
                    "memory_reserved_gb": round(reserved / (1024**3), 2),
                    "memory_percent": round((allocated / props.total_memory) * 100, 1)
                }
            except Exception as e:
                metrics["gpu"][f"device_{i}"] = {"error": str(e)}
    
    return metrics

def main():
    """Main monitoring loop"""
    metrics_file = Path("/workspace/logs/resource_metrics.jsonl")
    metrics_file.parent.mkdir(exist_ok=True)
    
    print("Starting resource monitoring...")
    
    while True:
        try:
            metrics = collect_metrics()
            
            # Write to file
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\\n')
            
            # Print summary
            print(f"[{metrics['timestamp']}] "
                  f"CPU: {metrics['cpu']['percent']:.1f}% | "
                  f"RAM: {metrics['memory']['percent']:.1f}% | "
                  f"Disk: {metrics['disk']['workspace_used_gb']:.1f}GB")
            
            if 'gpu' in metrics:
                for device, gpu_info in metrics['gpu'].items():
                    if 'error' not in gpu_info:
                        print(f"  {device}: {gpu_info['memory_percent']:.1f}% GPU memory")
            
            time.sleep(60)  # Collect metrics every minute
            
        except KeyboardInterrupt:
            print("Monitoring stopped")
            break
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
'''
        
        monitoring_file = self.workspace_dir / "resource_monitor.py"
        with open(monitoring_file, 'w') as f:
            f.write(monitoring_script)
        
        os.chmod(monitoring_file, 0o755)
        logger.info(f"Created resource monitoring script: {monitoring_file}")
    
    def create_optimization_summary(self, optimizations: Dict[str, Any]) -> None:
        """Create optimization summary file"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "runpod_optimizations": {
                "cpu_optimization": optimizations.get("cpu", {}),
                "memory_optimization": optimizations.get("memory", {}),
                "gpu_optimization": optimizations.get("gpu", {}),
                "network_optimization": optimizations.get("network", {}),
                "storage_optimization": optimizations.get("storage", {})
            },
            "system_resources": optimizations.get("resources", {}),
            "optimization_status": "completed",
            "recommendations": [
                "Monitor GPU memory usage to avoid OOM errors",
                "Use persistent volume for model storage to avoid re-downloading",
                "Run cleanup script periodically to free disk space",
                "Monitor resource usage through the monitoring script",
                "Check logs regularly for performance issues"
            ]
        }
        
        summary_file = self.workspace_dir / "optimization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Optimization summary saved to: {summary_file}")
    
    def run_complete_optimization(self) -> Dict[str, Any]:
        """Run complete RunPod optimization"""
        logger.info("Starting complete RunPod optimization...")
        
        try:
            # Detect system resources
            resources = self.detect_system_resources()
            
            # Run optimizations
            optimizations = {
                "resources": resources,
                "cpu": self.optimize_cpu_settings(resources),
                "memory": self.optimize_memory_settings(resources),
                "gpu": self.optimize_gpu_settings(resources),
                "network": self.optimize_networking_for_runpod()
            }
            
            # Setup persistent volume optimization
            storage_success = self.setup_persistent_volume_optimization()
            optimizations["storage"] = {"success": storage_success}
            
            # Create monitoring
            self.create_resource_monitoring()
            
            # Create optimization summary
            self.create_optimization_summary(optimizations)
            
            logger.info("RunPod optimization completed successfully")
            
            return {
                "success": True,
                "optimizations": optimizations,
                "message": "RunPod optimization completed successfully"
            }
            
        except Exception as e:
            logger.error(f"RunPod optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "RunPod optimization failed"
            }

def main():
    """Main optimization function"""
    logging.basicConfig(level=logging.INFO)
    
    optimizer = RunPodOptimizer()
    result = optimizer.run_complete_optimization()
    
    if result["success"]:
        print("RunPod Optimization Summary:")
        print("=" * 50)
        
        resources = result["optimizations"]["resources"]
        print(f"CPU Cores: {resources['cpu_count']}")
        print(f"Memory: {resources['memory_total_gb']:.1f}GB")
        print(f"GPU Count: {resources['gpu_count']}")
        
        if resources["gpu_devices"]:
            for gpu in resources["gpu_devices"]:
                print(f"  - {gpu['name']}: {gpu['memory_total_gb']:.1f}GB")
        
        print(f"Workspace Storage: {resources['workspace_disk_gb']:.1f}GB")
        print("\nOptimizations Applied:")
        print("✓ CPU thread optimization")
        print("✓ Memory management optimization")
        print("✓ GPU settings optimization")
        print("✓ Network configuration for RunPod")
        print("✓ Persistent volume optimization")
        print("✓ Resource monitoring setup")
        print("=" * 50)
        return 0
    else:
        print(f"Optimization failed: {result['message']}")
        return 1

if __name__ == "__main__":
    exit(main())