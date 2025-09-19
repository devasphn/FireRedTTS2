#!/usr/bin/env python3
"""
Complete RunPod Deployment Script for FireRedTTS2
Comprehensive deployment with sequential terminal commands, configuration, and monitoring
"""

import os
import sys
import json
import time
import subprocess
import logging
import shutil
import socket
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import torch
import psutil
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RunPodDeploymentManager:
    """Complete RunPod deployment management system"""
    
    def __init__(self):
        self.workspace_dir = Path("/workspace")
        self.models_dir = self.workspace_dir / "models"
        self.cache_dir = self.workspace_dir / "cache"
        self.logs_dir = self.workspace_dir / "logs"
        self.config_dir = self.workspace_dir / "config"
        self.scripts_dir = self.workspace_dir / "scripts"
        
        # Deployment configuration
        self.config = self._load_deployment_config()
        
        # Ensure all directories exist
        self._create_directory_structure()
        
        # Deployment state
        self.deployment_state = {
            "started_at": datetime.now(),
            "current_step": 0,
            "total_steps": 12,
            "completed_steps": [],
            "failed_steps": [],
            "warnings": [],
            "system_info": {}
        }
    
    def _load_deployment_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        default_config = {
            "container": {
                "python_version": "3.11",
                "cuda_version": "11.8",
                "base_image": "nvidia/cuda:11.8-devel-ubuntu22.04"
            },
            "hardware": {
                "min_gpu_memory_gb": 10,
                "recommended_gpu_memory_gb": 24,
                "min_system_memory_gb": 16,
                "min_disk_space_gb": 50
            },
            "network": {
                "primary_port": 7860,
                "health_port": 8080,
                "api_port": 8000
            },
            "models": {
                "base_path": "/workspace/models/FireRedTTS2",
                "download_url": "https://huggingface.co/FireRedTeam/FireRedTTS2",
                "required_files": [
                    "config_llm.json",
                    "config_codec.json",
                    "llm_pretrain.pt",
                    "llm_posttrain.pt",
                    "codec.pt",
                    "Qwen2.5-1.5B"
                ]
            },
            "environment": {
                "PYTHONPATH": "/workspace",
                "CUDA_VISIBLE_DEVICES": "0",
                "TORCH_HOME": "/workspace/cache/torch",
                "HF_HOME": "/workspace/cache/huggingface",
                "TRANSFORMERS_CACHE": "/workspace/cache/transformers",
                "GRADIO_SERVER_NAME": "0.0.0.0",
                "GRADIO_SERVER_PORT": "7860",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
            }
        }
        
        # Try to load from config file if exists
        config_file = self.workspace_dir / "runpod_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    if "runpod_deployment" in loaded_config:
                        return loaded_config["runpod_deployment"]
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        return default_config
    
    def _create_directory_structure(self):
        """Create necessary directory structure"""
        directories = [
            self.models_dir,
            self.cache_dir,
            self.logs_dir,
            self.config_dir,
            self.scripts_dir,
            self.workspace_dir / "uploads",
            self.workspace_dir / "sessions",
            self.cache_dir / "torch",
            self.cache_dir / "huggingface",
            self.cache_dir / "transformers"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def _run_command(self, command: str, shell: bool = True, check: bool = True, 
                    timeout: Optional[int] = None) -> Tuple[bool, str, str]:
        """Run shell command with logging and error handling"""
        logger.info(f"Executing: {command}")
        
        try:
            if shell:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout,
                    check=check
                )
            else:
                result = subprocess.run(
                    command.split(), 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout,
                    check=check
                )
            
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.debug(f"STDERR: {result.stderr}")
            
            return True, result.stdout, result.stderr
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}: {command}")
            logger.error(f"STDERR: {e.stderr}")
            return False, e.stdout or "", e.stderr or ""
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out: {command}")
            return False, "", "Command timed out"
        except Exception as e:
            logger.error(f"Unexpected error running command: {e}")
            return False, "", str(e)
    
    def step_1_system_setup(self) -> bool:
        """Step 1: System setup and package installation"""
        logger.info("=== Step 1: System Setup ===")
        self.deployment_state["current_step"] = 1
        
        commands = [
            "apt-get update",
            "apt-get upgrade -y",
            "apt-get install -y git git-lfs wget curl htop nvtop build-essential cmake",
            "apt-get install -y ffmpeg libsndfile1 libsox-dev",
            "apt-get autoremove -y",
            "apt-get autoclean"
        ]
        
        for cmd in commands:
            success, stdout, stderr = self._run_command(cmd, timeout=300)
            if not success:
                logger.error(f"System setup failed at command: {cmd}")
                self.deployment_state["failed_steps"].append(1)
                return False
        
        logger.info("System setup completed successfully")
        self.deployment_state["completed_steps"].append(1)
        return True
    
    def step_2_python_environment(self) -> bool:
        """Step 2: Python environment setup"""
        logger.info("=== Step 2: Python Environment Setup ===")
        self.deployment_state["current_step"] = 2
        
        commands = [
            "add-apt-repository ppa:deadsnakes/ppa -y",
            "apt-get update",
            "apt-get install -y python3.11 python3.11-dev python3.11-venv python3-pip",
            "ln -sf /usr/bin/python3.11 /usr/bin/python3",
            "ln -sf /usr/bin/python3.11 /usr/bin/python",
            "python3 -m pip install --upgrade pip setuptools wheel"
        ]
        
        for cmd in commands:
            success, stdout, stderr = self._run_command(cmd, timeout=300)
            if not success:
                logger.error(f"Python setup failed at command: {cmd}")
                self.deployment_state["failed_steps"].append(2)
                return False
        
        # Verify Python installation
        success, stdout, stderr = self._run_command("python3 --version")
        if success and "3.11" in stdout:
            logger.info(f"Python installation verified: {stdout.strip()}")
        else:
            logger.error("Python 3.11 installation verification failed")
            return False
        
        logger.info("Python environment setup completed successfully")
        self.deployment_state["completed_steps"].append(2)
        return True
    
    def step_3_repository_setup(self) -> bool:
        """Step 3: Repository cloning and setup"""
        logger.info("=== Step 3: Repository Setup ===")
        self.deployment_state["current_step"] = 3
        
        # Change to workspace directory
        os.chdir(self.workspace_dir)
        
        # Clone repository if not exists
        repo_dir = self.workspace_dir / "FireRedTTS2"
        if not repo_dir.exists():
            success, stdout, stderr = self._run_command(
                "git clone https://github.com/FireRedTeam/FireRedTTS2.git",
                timeout=600
            )
            if not success:
                logger.error("Failed to clone FireRedTTS2 repository")
                self.deployment_state["failed_steps"].append(3)
                return False
        else:
            logger.info("Repository already exists, updating...")
            os.chdir(repo_dir)
            success, stdout, stderr = self._run_command("git pull")
            if not success:
                logger.warning("Failed to update repository, continuing with existing version")
        
        # Set up repository
        os.chdir(repo_dir)
        
        logger.info("Repository setup completed successfully")
        self.deployment_state["completed_steps"].append(3)
        return True
    
    def step_4_pytorch_installation(self) -> bool:
        """Step 4: PyTorch and CUDA setup"""
        logger.info("=== Step 4: PyTorch Installation ===")
        self.deployment_state["current_step"] = 4
        
        # Install PyTorch with CUDA support
        pytorch_cmd = (
            "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 "
            "--index-url https://download.pytorch.org/whl/cu118"
        )
        
        success, stdout, stderr = self._run_command(pytorch_cmd, timeout=1200)
        if not success:
            logger.error("PyTorch installation failed")
            self.deployment_state["failed_steps"].append(4)
            return False
        
        # Verify CUDA installation
        success, stdout, stderr = self._run_command(
            'python3 -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\'); '
            'print(f\'CUDA version: {torch.version.cuda}\'); '
            'print(f\'GPU count: {torch.cuda.device_count()}\'); '
            'print(f\'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}\');"'
        )
        
        if success:
            logger.info(f"CUDA verification: {stdout}")
            if "CUDA available: True" not in stdout:
                logger.warning("CUDA not available, will run on CPU")
                self.deployment_state["warnings"].append("CUDA not available")
        else:
            logger.error("CUDA verification failed")
            return False
        
        logger.info("PyTorch installation completed successfully")
        self.deployment_state["completed_steps"].append(4)
        return True
    
    def step_5_dependencies_installation(self) -> bool:
        """Step 5: Install application dependencies"""
        logger.info("=== Step 5: Dependencies Installation ===")
        self.deployment_state["current_step"] = 5
        
        repo_dir = self.workspace_dir / "FireRedTTS2"
        os.chdir(repo_dir)
        
        # Install requirements
        if (repo_dir / "requirements.txt").exists():
            success, stdout, stderr = self._run_command("pip install -r requirements.txt", timeout=600)
            if not success:
                logger.error("Failed to install requirements.txt")
                self.deployment_state["failed_steps"].append(5)
                return False
        
        # Install package in development mode
        success, stdout, stderr = self._run_command("pip install -e .", timeout=300)
        if not success:
            logger.error("Failed to install FireRedTTS2 package")
            self.deployment_state["failed_steps"].append(5)
            return False
        
        # Install additional dependencies for enhanced features
        additional_deps = [
            "websockets", "aiohttp", "fastapi", "uvicorn", "python-multipart",
            "openai-whisper", "webrtcvad", "redis", "psutil", "gradio"
        ]
        
        for dep in additional_deps:
            success, stdout, stderr = self._run_command(f"pip install {dep}", timeout=300)
            if not success:
                logger.warning(f"Failed to install {dep}, continuing...")
                self.deployment_state["warnings"].append(f"Failed to install {dep}")
        
        logger.info("Dependencies installation completed successfully")
        self.deployment_state["completed_steps"].append(5)
        return True
    
    def step_6_model_download(self) -> bool:
        """Step 6: Download and verify models"""
        logger.info("=== Step 6: Model Download ===")
        self.deployment_state["current_step"] = 6
        
        # Initialize git LFS
        success, stdout, stderr = self._run_command("git lfs install")
        if not success:
            logger.error("Failed to initialize git LFS")
            self.deployment_state["failed_steps"].append(6)
            return False
        
        # Change to models directory
        os.chdir(self.models_dir)
        
        # Download models if not present
        model_repo_dir = self.models_dir / "FireRedTTS2"
        if not model_repo_dir.exists():
            logger.info("Downloading FireRedTTS2 models...")
            success, stdout, stderr = self._run_command(
                f"git clone {self.config['models']['download_url']}",
                timeout=1800  # 30 minutes timeout for large model download
            )
            if not success:
                logger.error("Failed to download models")
                self.deployment_state["failed_steps"].append(6)
                return False
        else:
            logger.info("Models directory exists, verifying...")
        
        # Verify model files
        missing_files = []
        for required_file in self.config['models']['required_files']:
            file_path = model_repo_dir / required_file
            if not file_path.exists():
                missing_files.append(required_file)
        
        if missing_files:
            logger.error(f"Missing model files: {missing_files}")
            # Try to pull missing files
            os.chdir(model_repo_dir)
            success, stdout, stderr = self._run_command("git lfs pull", timeout=1800)
            if not success:
                logger.error("Failed to pull missing model files")
                self.deployment_state["failed_steps"].append(6)
                return False
            
            # Re-verify
            missing_files = []
            for required_file in self.config['models']['required_files']:
                file_path = model_repo_dir / required_file
                if not file_path.exists():
                    missing_files.append(required_file)
            
            if missing_files:
                logger.error(f"Still missing model files after git lfs pull: {missing_files}")
                self.deployment_state["failed_steps"].append(6)
                return False
        
        logger.info("Model download and verification completed successfully")
        self.deployment_state["completed_steps"].append(6)
        return True
    
    def step_7_environment_configuration(self) -> bool:
        """Step 7: Environment configuration"""
        logger.info("=== Step 7: Environment Configuration ===")
        self.deployment_state["current_step"] = 7
        
        # Set environment variables
        for key, value in self.config['environment'].items():
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
        
        # Create environment file for persistence
        env_file = self.workspace_dir / ".env"
        with open(env_file, 'w') as f:
            for key, value in self.config['environment'].items():
                f.write(f"{key}={value}\n")
        
        # Set optimal thread counts based on CPU
        cpu_count = psutil.cpu_count()
        optimal_threads = max(1, cpu_count // 2)
        os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
        os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
        
        # Set file permissions
        success, stdout, stderr = self._run_command("chmod -R 755 /workspace")
        if not success:
            logger.warning("Failed to set file permissions")
            self.deployment_state["warnings"].append("Failed to set file permissions")
        
        logger.info("Environment configuration completed successfully")
        self.deployment_state["completed_steps"].append(7)
        return True
    
    def step_8_gpu_configuration(self) -> bool:
        """Step 8: GPU configuration and optimization"""
        logger.info("=== Step 8: GPU Configuration ===")
        self.deployment_state["current_step"] = 8
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU configuration")
            self.deployment_state["warnings"].append("CUDA not available")
            self.deployment_state["completed_steps"].append(8)
            return True
        
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Log GPU information
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f}GB")
            
            logger.info("GPU configuration completed successfully")
            self.deployment_state["completed_steps"].append(8)
            return True
            
        except Exception as e:
            logger.error(f"GPU configuration failed: {e}")
            self.deployment_state["failed_steps"].append(8)
            return False
    
    def step_9_model_verification(self) -> bool:
        """Step 9: Model loading verification"""
        logger.info("=== Step 9: Model Verification ===")
        self.deployment_state["current_step"] = 9
        
        # Test model loading
        test_script = f'''
import sys
sys.path.append("/workspace/FireRedTTS2")
try:
    from fireredtts2.fireredtts2 import FireRedTTS2
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {{device}}")
    
    model = FireRedTTS2(
        pretrained_dir="{self.config['models']['base_path']}",
        gen_type='dialogue',
        device=device
    )
    print("Model loaded successfully!")
    
    # Test basic generation
    test_text = "Hello, this is a test."
    result = model.generate(test_text)
    print(f"Test generation completed, output length: {{len(result) if result else 0}}")
    
except Exception as e:
    print(f"Model verification failed: {{e}}")
    sys.exit(1)
'''
        
        # Write test script
        test_file = self.workspace_dir / "test_model.py"
        with open(test_file, 'w') as f:
            f.write(test_script)
        
        # Run test
        success, stdout, stderr = self._run_command(f"python3 {test_file}", timeout=300)
        
        if success and "Model loaded successfully!" in stdout:
            logger.info("Model verification completed successfully")
            logger.info(stdout)
            self.deployment_state["completed_steps"].append(9)
            return True
        else:
            logger.error(f"Model verification failed: {stderr}")
            self.deployment_state["failed_steps"].append(9)
            return False
    
    def step_10_service_scripts(self) -> bool:
        """Step 10: Create service startup scripts"""
        logger.info("=== Step 10: Service Scripts Creation ===")
        self.deployment_state["current_step"] = 10
        
        # Create main startup script
        startup_script = f'''#!/bin/bash
set -e

echo "Starting FireRedTTS2 RunPod Deployment..."
echo "Timestamp: $(date)"

# Load environment variables
if [ -f "/workspace/.env" ]; then
    source /workspace/.env
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}}

# Change to application directory
cd /workspace/FireRedTTS2

# Check if models exist
if [ ! -d "{self.config['models']['base_path']}" ]; then
    echo "ERROR: Models not found at {self.config['models']['base_path']}"
    echo "Please run the deployment script to download models"
    exit 1
fi

# Start health check service in background
python3 /workspace/health_check.py &
HEALTH_PID=$!
echo "Health check service started with PID: $HEALTH_PID"

# Start the enhanced web interface
echo "Starting enhanced web interface on port {self.config['network']['primary_port']}..."
python3 enhanced_gradio_demo.py \\
    --pretrained-dir "{self.config['models']['base_path']}" \\
    --host 0.0.0.0 \\
    --port {self.config['network']['primary_port']} \\
    --enable-streaming \\
    --enable-conversation \\
    --enable-voice-cloning

# If the main process exits, kill the health check service
kill $HEALTH_PID 2>/dev/null || true
'''
        
        startup_file = self.workspace_dir / "start_fireredtts2.sh"
        with open(startup_file, 'w') as f:
            f.write(startup_script)
        
        success, stdout, stderr = self._run_command(f"chmod +x {startup_file}")
        if not success:
            logger.error("Failed to make startup script executable")
            return False
        
        # Create health check script
        health_script = '''#!/usr/bin/env python3
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
import psutil

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Collect health information
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "service": "FireRedTTS2",
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/workspace').percent
            }
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.memory_stats(i)
                    allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                    total = torch.cuda.get_device_properties(i).total_memory
                    health_status[f'gpu_{i}_memory_percent'] = (allocated / total) * 100
            
            self.wfile.write(json.dumps(health_status).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress default logging

def start_health_server():
    server = HTTPServer(('0.0.0.0', 8080), HealthCheckHandler)
    print("Health check server started on port 8080")
    server.serve_forever()

if __name__ == "__main__":
    start_health_server()
'''
        
        health_file = self.workspace_dir / "health_check.py"
        with open(health_file, 'w') as f:
            f.write(health_script)
        
        success, stdout, stderr = self._run_command(f"chmod +x {health_file}")
        if not success:
            logger.error("Failed to make health check script executable")
            return False
        
        # Create monitoring script
        monitor_script = '''#!/bin/bash
echo "FireRedTTS2 System Monitor"
echo "=========================="

while true; do
    echo "=== $(date) ==="
    
    echo "GPU Status:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    else
        echo "nvidia-smi not available"
    fi
    
    echo "System Resources:"
    echo "CPU: $(cat /proc/loadavg | cut -d' ' -f1-3)"
    echo "Memory: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
    echo "Disk: $(df -h /workspace | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"
    
    echo "Network Connections:"
    ss -tlnp | grep -E ':(7860|8080|8000)'
    
    echo "Process Status:"
    ps aux | grep -E '(python|gradio)' | grep -v grep | head -5
    
    echo "===================="
    sleep 30
done
'''
        
        monitor_file = self.workspace_dir / "monitor.sh"
        with open(monitor_file, 'w') as f:
            f.write(monitor_script)
        
        success, stdout, stderr = self._run_command(f"chmod +x {monitor_file}")
        if not success:
            logger.error("Failed to make monitor script executable")
            return False
        
        logger.info("Service scripts created successfully")
        self.deployment_state["completed_steps"].append(10)
        return True
    
    def step_11_port_configuration(self) -> bool:
        """Step 11: Port configuration and networking"""
        logger.info("=== Step 11: Port Configuration ===")
        self.deployment_state["current_step"] = 11
        
        # Check if ports are available
        ports_to_check = [
            self.config['network']['primary_port'],
            self.config['network']['health_port']
        ]
        
        for port in ports_to_check:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                result = sock.bind(('0.0.0.0', port))
                sock.close()
                logger.info(f"Port {port} is available")
            except OSError:
                logger.warning(f"Port {port} is already in use")
                self.deployment_state["warnings"].append(f"Port {port} already in use")
        
        # Create port configuration file
        port_config = {
            "primary_port": self.config['network']['primary_port'],
            "health_port": self.config['network']['health_port'],
            "api_port": self.config['network'].get('api_port', 8000),
            "websocket_enabled": True,
            "cors_enabled": True,
            "runpod_proxy_compatible": True
        }
        
        config_file = self.config_dir / "network_config.json"
        with open(config_file, 'w') as f:
            json.dump(port_config, f, indent=2)
        
        logger.info("Port configuration completed successfully")
        self.deployment_state["completed_steps"].append(11)
        return True
    
    def step_12_final_verification(self) -> bool:
        """Step 12: Final deployment verification"""
        logger.info("=== Step 12: Final Verification ===")
        self.deployment_state["current_step"] = 12
        
        # Collect system information
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_space_gb": round(psutil.disk_usage('/workspace').free / (1024**3), 2),
            "python_version": sys.version,
            "pytorch_version": torch.__version__
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                system_info[f"gpu_{i}"] = {
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}"
                }
        
        self.deployment_state["system_info"] = system_info
        
        # Verify all required files exist
        verification_items = [
            (self.workspace_dir / "start_fireredtts2.sh", "Startup script"),
            (self.workspace_dir / "health_check.py", "Health check script"),
            (self.workspace_dir / "monitor.sh", "Monitor script"),
            (self.config['models']['base_path'], "Models directory"),
            (self.workspace_dir / ".env", "Environment file")
        ]
        
        missing_items = []
        for item_path, item_name in verification_items:
            if not item_path.exists():
                missing_items.append(item_name)
        
        if missing_items:
            logger.error(f"Missing required items: {missing_items}")
            self.deployment_state["failed_steps"].append(12)
            return False
        
        # Create deployment summary
        summary = {
            "deployment_completed": True,
            "completion_time": datetime.now().isoformat(),
            "duration_minutes": (datetime.now() - self.deployment_state["started_at"]).total_seconds() / 60,
            "completed_steps": len(self.deployment_state["completed_steps"]),
            "failed_steps": len(self.deployment_state["failed_steps"]),
            "warnings": len(self.deployment_state["warnings"]),
            "system_info": system_info,
            "next_steps": [
                "Run '/workspace/start_fireredtts2.sh' to start the application",
                "Access the web interface at the RunPod provided URL",
                "Monitor system health at /health endpoint",
                "Check logs in /workspace/logs/ directory"
            ]
        }
        
        summary_file = self.workspace_dir / "deployment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Final verification completed successfully")
        self.deployment_state["completed_steps"].append(12)
        return True
    
    def run_complete_deployment(self) -> bool:
        """Run the complete deployment process"""
        logger.info("Starting complete RunPod deployment for FireRedTTS2")
        logger.info(f"Deployment started at: {self.deployment_state['started_at']}")
        
        deployment_steps = [
            ("System Setup", self.step_1_system_setup),
            ("Python Environment", self.step_2_python_environment),
            ("Repository Setup", self.step_3_repository_setup),
            ("PyTorch Installation", self.step_4_pytorch_installation),
            ("Dependencies Installation", self.step_5_dependencies_installation),
            ("Model Download", self.step_6_model_download),
            ("Environment Configuration", self.step_7_environment_configuration),
            ("GPU Configuration", self.step_8_gpu_configuration),
            ("Model Verification", self.step_9_model_verification),
            ("Service Scripts", self.step_10_service_scripts),
            ("Port Configuration", self.step_11_port_configuration),
            ("Final Verification", self.step_12_final_verification)
        ]
        
        for step_name, step_function in deployment_steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting: {step_name}")
            logger.info(f"{'='*60}")
            
            try:
                success = step_function()
                if not success:
                    logger.error(f"Deployment failed at step: {step_name}")
                    return False
                    
                logger.info(f"✓ Completed: {step_name}")
                
            except Exception as e:
                logger.error(f"Unexpected error in step '{step_name}': {e}")
                self.deployment_state["failed_steps"].append(self.deployment_state["current_step"])
                return False
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("DEPLOYMENT COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*60}")
        logger.info(f"Total time: {(datetime.now() - self.deployment_state['started_at']).total_seconds() / 60:.1f} minutes")
        logger.info(f"Completed steps: {len(self.deployment_state['completed_steps'])}/{self.deployment_state['total_steps']}")
        logger.info(f"Warnings: {len(self.deployment_state['warnings'])}")
        
        if self.deployment_state["warnings"]:
            logger.info("Warnings encountered:")
            for warning in self.deployment_state["warnings"]:
                logger.info(f"  - {warning}")
        
        logger.info("\nNext steps:")
        logger.info("1. Run: /workspace/start_fireredtts2.sh")
        logger.info("2. Access web interface via RunPod URL")
        logger.info("3. Monitor health at /health endpoint")
        
        return True

def main():
    """Main deployment function"""
    try:
        deployment_manager = RunPodDeploymentManager()
        success = deployment_manager.run_complete_deployment()
        
        if success:
            logger.info("Deployment completed successfully!")
            return 0
        else:
            logger.error("Deployment failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during deployment: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())