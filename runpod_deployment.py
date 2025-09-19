#!/usr/bin/env python3
"""
RunPod Deployment Script for FireRedTTS2
Handles model loading, GPU configuration, and service initialization
"""

import os
import sys
import json
import time
import torch
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RunPodDeployment:
    """Handles RunPod-specific deployment configuration and initialization"""
    
    def __init__(self):
        self.workspace_dir = Path("/workspace")
        self.models_dir = self.workspace_dir / "models"
        self.cache_dir = self.workspace_dir / "cache"
        self.logs_dir = self.workspace_dir / "logs"
        
        # Ensure directories exist
        for dir_path in [self.models_dir, self.cache_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements and GPU availability"""
        logger.info("Checking system requirements...")
        
        system_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_space_gb": round(psutil.disk_usage('/workspace').free / (1024**3), 2)
        }
        
        if system_info["cuda_available"]:
            for i in range(system_info["gpu_count"]):
                gpu_props = torch.cuda.get_device_properties(i)
                system_info[f"gpu_{i}"] = {
                    "name": gpu_props.name,
                    "memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                }
        
        logger.info(f"System info: {json.dumps(system_info, indent=2)}")
        return system_info
    
    def configure_gpu_environment(self) -> None:
        """Configure GPU environment for optimal performance"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, running on CPU")
            return
        
        # Set CUDA device
        device_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        logger.info(f"Using CUDA device: {device_id}")
        
        # Configure memory allocation
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        
        # Enable memory optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        logger.info("GPU environment configured successfully")
    
    def check_model_availability(self) -> Dict[str, bool]:
        """Check if required models are available"""
        logger.info("Checking model availability...")
        
        model_base_dir = self.models_dir / "FireRedTTS2"
        required_files = [
            "config_llm.json",
            "config_codec.json", 
            "llm_pretrain.pt",
            "llm_posttrain.pt",
            "codec.pt",
            "Qwen2.5-1.5B"
        ]
        
        availability = {}
        for file_name in required_files:
            file_path = model_base_dir / file_name
            availability[file_name] = file_path.exists()
            logger.info(f"{file_name}: {'✓' if availability[file_name] else '✗'}")
        
        all_available = all(availability.values())
        logger.info(f"All models available: {'✓' if all_available else '✗'}")
        
        return availability
    
    def download_models_if_needed(self) -> bool:
        """Download models if not present in persistent volume"""
        model_availability = self.check_model_availability()
        
        if all(model_availability.values()):
            logger.info("All models already available, skipping download")
            return True
        
        logger.info("Downloading missing models...")
        try:
            import subprocess
            
            # Change to models directory
            os.chdir(self.models_dir)
            
            # Initialize git lfs
            subprocess.run(["git", "lfs", "install"], check=True)
            
            # Clone the model repository
            if not (self.models_dir / "FireRedTTS2").exists():
                logger.info("Cloning FireRedTTS2 model repository...")
                subprocess.run([
                    "git", "clone", 
                    "https://huggingface.co/FireRedTeam/FireRedTTS2"
                ], check=True)
            
            # Verify download
            model_availability = self.check_model_availability()
            success = all(model_availability.values())
            
            if success:
                logger.info("Models downloaded successfully")
            else:
                logger.error("Model download incomplete")
            
            return success
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Model download failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during model download: {e}")
            return False
    
    def setup_environment_variables(self) -> None:
        """Setup environment variables for RunPod deployment"""
        env_vars = {
            "PYTHONPATH": "/workspace",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "TORCH_HOME": "/workspace/cache/torch",
            "HF_HOME": "/workspace/cache/huggingface",
            "TRANSFORMERS_CACHE": "/workspace/cache/transformers",
            "GRADIO_SERVER_NAME": "0.0.0.0",
            "GRADIO_SERVER_PORT": "7860",
            "GRADIO_SHARE": "False"
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
    
    def create_health_check_endpoint(self) -> None:
        """Create a simple health check endpoint"""
        health_check_script = '''
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "service": "FireRedTTS2"
            }
            
            self.wfile.write(json.dumps(health_status).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress default logging

def start_health_server():
    server = HTTPServer(('0.0.0.0', 8080), HealthCheckHandler)
    server.serve_forever()

# Start health check server in background
health_thread = threading.Thread(target=start_health_server, daemon=True)
health_thread.start()
'''
        
        with open('/workspace/health_check.py', 'w') as f:
            f.write(health_check_script)
    
    def optimize_for_runpod(self) -> None:
        """Apply RunPod-specific optimizations"""
        logger.info("Applying RunPod optimizations...")
        
        # Set optimal number of workers based on CPU count
        cpu_count = psutil.cpu_count()
        os.environ["OMP_NUM_THREADS"] = str(max(1, cpu_count // 2))
        os.environ["MKL_NUM_THREADS"] = str(max(1, cpu_count // 2))
        
        # Configure memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Set up proper file permissions
        os.system("chmod -R 755 /workspace")
        
        logger.info("RunPod optimizations applied")
    
    def initialize_deployment(self) -> bool:
        """Initialize the complete deployment"""
        logger.info("Initializing RunPod deployment for FireRedTTS2...")
        
        try:
            # Check system requirements
            system_info = self.check_system_requirements()
            
            # Verify minimum requirements
            if system_info["memory_gb"] < 16:
                logger.warning(f"Low memory: {system_info['memory_gb']}GB (16GB+ recommended)")
            
            if system_info["disk_space_gb"] < 50:
                logger.error(f"Insufficient disk space: {system_info['disk_space_gb']}GB (50GB+ required)")
                return False
            
            # Configure GPU environment
            self.configure_gpu_environment()
            
            # Setup environment variables
            self.setup_environment_variables()
            
            # Download models if needed
            if not self.download_models_if_needed():
                logger.error("Failed to download required models")
                return False
            
            # Apply RunPod optimizations
            self.optimize_for_runpod()
            
            # Create health check endpoint
            self.create_health_check_endpoint()
            
            logger.info("RunPod deployment initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Deployment initialization failed: {e}")
            return False

def main():
    """Main deployment function"""
    deployment = RunPodDeployment()
    
    if deployment.initialize_deployment():
        logger.info("Deployment ready - starting application...")
        return 0
    else:
        logger.error("Deployment failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())