#!/bin/bash
set -e

# Container Startup Script for FireRedTTS2 on RunPod
# This script handles model loading, service initialization, and system configuration

echo "=============================================="
echo "FireRedTTS2 RunPod Container Startup"
echo "Timestamp: $(date)"
echo "=============================================="

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for a service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local timeout=${3:-30}
    local count=0
    
    log "Waiting for service at $host:$port..."
    while ! nc -z $host $port; do
        sleep 1
        count=$((count + 1))
        if [ $count -ge $timeout ]; then
            log "ERROR: Service at $host:$port not ready after $timeout seconds"
            return 1
        fi
    done
    log "Service at $host:$port is ready"
    return 0
}

# Set up error handling
trap 'log "ERROR: Script failed at line $LINENO"' ERR

# ============================================
# ENVIRONMENT SETUP
# ============================================

log "Setting up environment variables..."

# Core environment variables
export PYTHONPATH="/workspace:/workspace/FireRedTTS2"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_HOME="/workspace/cache/torch"
export HF_HOME="/workspace/cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/cache/transformers"
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT="7860"
export GRADIO_SHARE="False"

# Performance optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export CUDA_LAUNCH_BLOCKING="0"

# Create necessary directories
log "Creating directory structure..."
mkdir -p /workspace/models
mkdir -p /workspace/cache/{torch,huggingface,transformers}
mkdir -p /workspace/logs
mkdir -p /workspace/uploads
mkdir -p /workspace/sessions
mkdir -p /workspace/config

# Set permissions
chmod -R 755 /workspace

# ============================================
# SYSTEM CHECKS
# ============================================

log "Performing system checks..."

# Check CUDA availability
if command_exists nvidia-smi; then
    log "CUDA Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
else
    log "WARNING: nvidia-smi not available"
fi

# Check Python and PyTorch
log "Python version: $(python3 --version)"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Check disk space
log "Disk space:"
df -h /workspace

# Check memory
log "Memory usage:"
free -h

# ============================================
# MODEL MANAGEMENT
# ============================================

log "Checking model availability..."

MODEL_DIR="/workspace/models/FireRedTTS2"
REQUIRED_FILES=(
    "config_llm.json"
    "config_codec.json"
    "llm_pretrain.pt"
    "llm_posttrain.pt"
    "codec.pt"
    "Qwen2.5-1.5B"
)

# Function to check if all models are present
check_models() {
    local all_present=true
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -e "$MODEL_DIR/$file" ]; then
            log "Missing model file: $file"
            all_present=false
        fi
    done
    $all_present
}

# Download models if not present
if [ ! -d "$MODEL_DIR" ] || ! check_models; then
    log "Models not found or incomplete, downloading..."
    
    cd /workspace/models
    
    # Initialize git LFS
    git lfs install
    
    # Clone model repository
    if [ ! -d "FireRedTTS2" ]; then
        log "Cloning FireRedTTS2 model repository..."
        git clone https://huggingface.co/FireRedTeam/FireRedTTS2
    else
        log "Model directory exists, pulling latest..."
        cd FireRedTTS2
        git pull
        git lfs pull
        cd ..
    fi
    
    # Verify download
    if check_models; then
        log "✓ All model files are present"
    else
        log "ERROR: Model download incomplete"
        exit 1
    fi
else
    log "✓ All model files are present"
fi

# ============================================
# SERVICE INITIALIZATION
# ============================================

log "Initializing services..."

# Start health check service
log "Starting health check service..."
cat > /workspace/health_service.py << 'EOF'
#!/usr/bin/env python3
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
import psutil
import os

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == '/health':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                # Collect comprehensive health information
                health_status = {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "service": "FireRedTTS2",
                    "version": "1.0.0",
                    "uptime": time.time() - start_time,
                    "system": {
                        "cuda_available": torch.cuda.is_available(),
                        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                        "cpu_percent": psutil.cpu_percent(interval=1),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage('/workspace').percent
                    }
                }
                
                # GPU information
                if torch.cuda.is_available():
                    health_status["gpu"] = {}
                    for i in range(torch.cuda.device_count()):
                        try:
                            props = torch.cuda.get_device_properties(i)
                            memory_stats = torch.cuda.memory_stats(i)
                            allocated = memory_stats.get('allocated_bytes.all.current', 0)
                            total = props.total_memory
                            
                            health_status["gpu"][f"device_{i}"] = {
                                "name": props.name,
                                "memory_total_gb": round(total / (1024**3), 2),
                                "memory_used_gb": round(allocated / (1024**3), 2),
                                "memory_percent": round((allocated / total) * 100, 1),
                                "compute_capability": f"{props.major}.{props.minor}"
                            }
                        except Exception as e:
                            health_status["gpu"][f"device_{i}"] = {"error": str(e)}
                
                # Model status
                model_dir = "/workspace/models/FireRedTTS2"
                required_files = [
                    "config_llm.json", "config_codec.json", "llm_pretrain.pt",
                    "llm_posttrain.pt", "codec.pt", "Qwen2.5-1.5B"
                ]
                
                health_status["models"] = {
                    "base_path": model_dir,
                    "files_present": sum(1 for f in required_files if os.path.exists(os.path.join(model_dir, f))),
                    "files_required": len(required_files),
                    "status": "ready" if all(os.path.exists(os.path.join(model_dir, f)) for f in required_files) else "incomplete"
                }
                
                self.wfile.write(json.dumps(health_status, indent=2).encode())
                
            elif self.path == '/metrics':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                # Detailed metrics
                metrics = {
                    "timestamp": time.time(),
                    "cpu": {
                        "percent": psutil.cpu_percent(interval=1),
                        "count": psutil.cpu_count(),
                        "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
                    },
                    "memory": {
                        "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                        "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                        "percent": psutil.virtual_memory().percent
                    },
                    "disk": {
                        "total_gb": round(psutil.disk_usage('/workspace').total / (1024**3), 2),
                        "used_gb": round(psutil.disk_usage('/workspace').used / (1024**3), 2),
                        "percent": round((psutil.disk_usage('/workspace').used / psutil.disk_usage('/workspace').total) * 100, 1)
                    }
                }
                
                if torch.cuda.is_available():
                    metrics["gpu"] = {}
                    for i in range(torch.cuda.device_count()):
                        try:
                            memory_stats = torch.cuda.memory_stats(i)
                            props = torch.cuda.get_device_properties(i)
                            metrics["gpu"][f"device_{i}"] = {
                                "utilization": torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0,
                                "memory_allocated": memory_stats.get('allocated_bytes.all.current', 0),
                                "memory_reserved": memory_stats.get('reserved_bytes.all.current', 0),
                                "memory_total": props.total_memory
                            }
                        except Exception as e:
                            metrics["gpu"][f"device_{i}"] = {"error": str(e)}
                
                self.wfile.write(json.dumps(metrics, indent=2).encode())
                
            else:
                self.send_response(404)
                self.end_headers()
                
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {"error": str(e), "timestamp": time.time()}
            self.wfile.write(json.dumps(error_response).encode())
    
    def log_message(self, format, *args):
        pass  # Suppress default logging

def start_health_server():
    server = HTTPServer(('0.0.0.0', 8080), HealthCheckHandler)
    print(f"Health check server started on port 8080")
    server.serve_forever()

if __name__ == "__main__":
    start_time = time.time()
    start_health_server()
EOF

chmod +x /workspace/health_service.py

# Start health service in background
python3 /workspace/health_service.py &
HEALTH_PID=$!
log "Health check service started with PID: $HEALTH_PID"

# Wait for health service to be ready
sleep 2
if ! wait_for_service localhost 8080 10; then
    log "WARNING: Health service not responding"
fi

# ============================================
# APPLICATION STARTUP
# ============================================

log "Starting FireRedTTS2 application..."

# Change to application directory
cd /workspace/FireRedTTS2

# Check if enhanced demo exists, otherwise use standard demo
if [ -f "enhanced_gradio_demo.py" ]; then
    DEMO_SCRIPT="enhanced_gradio_demo.py"
    log "Using enhanced Gradio demo"
else
    DEMO_SCRIPT="gradio_demo.py"
    log "Using standard Gradio demo"
fi

# Create application startup command
APP_CMD="python3 $DEMO_SCRIPT \
    --pretrained-dir '$MODEL_DIR' \
    --host 0.0.0.0 \
    --port 7860"

# Add enhanced features if available
if [ "$DEMO_SCRIPT" = "enhanced_gradio_demo.py" ]; then
    APP_CMD="$APP_CMD \
        --enable-streaming \
        --enable-conversation \
        --enable-voice-cloning \
        --enable-monitoring"
fi

log "Application command: $APP_CMD"

# Function to cleanup on exit
cleanup() {
    log "Shutting down services..."
    if [ ! -z "$HEALTH_PID" ]; then
        kill $HEALTH_PID 2>/dev/null || true
        log "Health service stopped"
    fi
    log "Cleanup completed"
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Start the application
log "Starting application on port 7860..."
log "Application will be accessible via RunPod's provided URL"

# Execute the application
exec $APP_CMD