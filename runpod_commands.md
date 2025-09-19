# RunPod Deployment Commands

This document provides the complete sequence of commands to deploy FireRedTTS2 on RunPod.

## Prerequisites

1. **RunPod Account**: Sign up at https://runpod.io
2. **GPU Selection**: Choose RTX 4090 (24GB) or A100 (40GB+) 
3. **Storage**: Configure 100GB+ persistent volume
4. **Network**: Ensure port 7860 is exposed

## Step 1: Container Setup

### Launch RunPod Instance
```bash
# In RunPod web interface:
# 1. Select "Deploy" → "Pods"
# 2. Choose GPU: RTX 4090 or A100
# 3. Select Container Disk: 50GB
# 4. Add Persistent Volume: 100GB mounted at /workspace/models
# 5. Expose Port: 7860 (HTTP)
# 6. Use Custom Docker Image or start with PyTorch base
```

### Initial System Update
```bash
# Update system packages
apt-get update && apt-get upgrade -y

# Install essential tools
apt-get install -y git git-lfs wget curl htop nvtop
```

## Step 2: Python Environment Setup

### Install Python 3.11
```bash
# Add Python 3.11 repository
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update

# Install Python 3.11
apt-get install -y python3.11 python3.11-dev python3.11-venv python3-pip

# Create symbolic links
ln -sf /usr/bin/python3.11 /usr/bin/python3
ln -sf /usr/bin/python3.11 /usr/bin/python
```

### Upgrade pip and install wheel
```bash
python3 -m pip install --upgrade pip setuptools wheel
```

## Step 3: Clone Repository

### Clone FireRedTTS2 Repository
```bash
cd /workspace
git clone https://github.com/FireRedTeam/FireRedTTS2.git
cd FireRedTTS2
```

### Copy deployment files
```bash
# Copy the Dockerfile and deployment scripts we created
# (These should be uploaded to the container or created manually)
```

## Step 4: Install Dependencies

### Install PyTorch with CUDA Support
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### Install Core Requirements
```bash
pip install -r requirements.txt
pip install -e .
```

### Install Additional Dependencies for Enhanced Features
```bash
pip install websockets aiohttp asyncio numpy scipy librosa soundfile webrtcvad openai-whisper fastapi uvicorn python-multipart redis psutil
```

## Step 5: Model Download and Setup

### Initialize Git LFS
```bash
git lfs install
```

### Download FireRedTTS2 Models
```bash
cd /workspace/models
git clone https://huggingface.co/FireRedTeam/FireRedTTS2
```

### Verify Model Files
```bash
ls -la /workspace/models/FireRedTTS2/
# Should contain:
# - config_llm.json
# - config_codec.json  
# - llm_pretrain.pt
# - llm_posttrain.pt
# - codec.pt
# - Qwen2.5-1.5B/
```

## Step 6: Environment Configuration

### Set Environment Variables
```bash
export PYTHONPATH="/workspace/FireRedTTS2"
export CUDA_VISIBLE_DEVICES="0"
export TORCH_HOME="/workspace/cache/torch"
export HF_HOME="/workspace/cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/cache/transformers"
```

### Create Cache Directories
```bash
mkdir -p /workspace/cache/torch
mkdir -p /workspace/cache/huggingface
mkdir -p /workspace/cache/transformers
mkdir -p /workspace/logs
mkdir -p /workspace/uploads
mkdir -p /workspace/sessions
```

### Set Permissions
```bash
chmod -R 755 /workspace
```

## Step 7: GPU Configuration

### Check GPU Status
```bash
nvidia-smi
nvtop
```

### Test CUDA Installation
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Configure GPU Memory
```bash
# Set GPU memory fraction (optional)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

## Step 8: Test Basic Functionality

### Test Model Loading
```bash
cd /workspace/FireRedTTS2
python3 -c "
from fireredtts2.fireredtts2 import FireRedTTS2
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
try:
    model = FireRedTTS2(
        pretrained_dir='/workspace/models/FireRedTTS2',
        gen_type='dialogue',
        device=device
    )
    print('Model loaded successfully!')
except Exception as e:
    print(f'Model loading failed: {e}')
"
```

## Step 9: Launch Application

### Start Basic Gradio Interface
```bash
cd /workspace/FireRedTTS2
python3 gradio_demo.py --pretrained-dir "/workspace/models/FireRedTTS2" --host 0.0.0.0 --port 7860
```

### Alternative: Use Deployment Script
```bash
python3 runpod_deployment.py
```

## Step 10: Verify Deployment

### Check Service Status
```bash
# Check if service is running
ps aux | grep python
netstat -tlnp | grep 7860
```

### Test Web Interface
```bash
# Access via RunPod's provided URL (typically https://xxxxx-7860.proxy.runpod.net)
curl -f http://localhost:7860/health || echo "Health check failed"
```

### Monitor Resources
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Check logs
tail -f /workspace/logs/deployment.log
```

## Step 11: Optimization Commands

### Memory Optimization
```bash
# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Set memory management
echo 'vm.swappiness=10' >> /etc/sysctl.conf
sysctl -p
```

### Performance Tuning
```bash
# Set CPU affinity for optimal performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Enable CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
```

## Step 12: Create Startup Script

### Create Persistent Startup Script
```bash
cat > /workspace/start_fireredtts2.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting FireRedTTS2 on RunPod..."

# Set environment variables
export PYTHONPATH="/workspace/FireRedTTS2"
export CUDA_VISIBLE_DEVICES="0"
export TORCH_HOME="/workspace/cache/torch"

# Change to application directory
cd /workspace/FireRedTTS2

# Check if models exist
if [ ! -d "/workspace/models/FireRedTTS2" ]; then
    echo "Models not found, downloading..."
    cd /workspace/models
    git lfs install
    git clone https://huggingface.co/FireRedTeam/FireRedTTS2
fi

# Start the application
echo "Starting Gradio interface..."
python3 gradio_demo.py \
    --pretrained-dir "/workspace/models/FireRedTTS2" \
    --host 0.0.0.0 \
    --port 7860

EOF

chmod +x /workspace/start_fireredtts2.sh
```

### Test Startup Script
```bash
/workspace/start_fireredtts2.sh
```

## Troubleshooting Commands

### Check CUDA Issues
```bash
# Verify CUDA installation
nvcc --version
python3 -c "import torch; print(torch.version.cuda)"

# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

### Debug Model Loading
```bash
# Check model files
find /workspace/models -name "*.pt" -exec ls -lh {} \;
find /workspace/models -name "*.json" -exec cat {} \;
```

### Network Debugging
```bash
# Check port binding
ss -tlnp | grep 7860
lsof -i :7860

# Test local connection
curl -I http://localhost:7860
```

### Log Analysis
```bash
# Application logs
tail -f /workspace/logs/*.log

# System logs
journalctl -f

# GPU logs
nvidia-smi dmon
```

## Performance Monitoring

### Continuous Monitoring Setup
```bash
# Create monitoring script
cat > /workspace/monitor.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    echo "Memory Usage:"
    free -h
    echo "Disk Usage:"
    df -h /workspace
    echo "Network:"
    ss -tlnp | grep 7860
    echo "===================="
    sleep 30
done
EOF

chmod +x /workspace/monitor.sh
```

### Run Monitoring
```bash
# Run in background
nohup /workspace/monitor.sh > /workspace/logs/monitor.log 2>&1 &
```

This completes the RunPod deployment setup. The application should now be accessible via the RunPod-provided URL on port 7860.