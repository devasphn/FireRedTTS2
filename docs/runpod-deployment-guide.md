# FireRedTTS2 RunPod Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the FireRedTTS2 speech-to-speech conversation system on RunPod cloud platform. The deployment includes a complete web interface with real-time audio processing, voice cloning, and conversational AI capabilities.

## Prerequisites

### RunPod Account Setup
- Active RunPod account with sufficient credits
- Access to GPU instances (RTX 4090, A100, or H100 recommended)
- Basic familiarity with RunPod's web interface

### System Requirements
- **GPU**: RTX 4090 (24GB VRAM) minimum, A100 (40GB) or H100 (80GB) recommended
- **RAM**: 32GB minimum, 64GB recommended
- **Storage**: 100GB persistent volume for models
- **Container Disk**: 50GB for application files
- **Network**: HTTP/TCP protocols only (RunPod limitation)

## Step-by-Step Deployment

### Step 1: Create RunPod Instance

1. **Log into RunPod Console**
   - Navigate to https://www.runpod.io/
   - Sign in to your account

2. **Create New Pod**
   - Click "Deploy" → "GPU Pods"
   - Select GPU type (RTX 4090 minimum recommended)
   - Choose template: "RunPod PyTorch 2.1" or "NVIDIA PyTorch"

3. **Configure Pod Settings**
   ```
   Container Disk: 50 GB
   Volume Disk: 100 GB (persistent)
   Volume Mount Path: /workspace
   Expose HTTP Ports: 7860, 8000
   Expose TCP Ports: 7860, 8000
   ```

4. **Deploy Pod**
   - Click "Deploy On-Demand" or "Deploy Spot"
   - Wait for pod to initialize (2-5 minutes)

### Step 2: Access Pod Terminal

1. **Connect to Pod**
   - Click "Connect" on your running pod
   - Select "Start Web Terminal"
   - Wait for terminal to load

2. **Verify Environment**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Check Python version
   python --version
   
   # Check CUDA version
   nvcc --version
   ```

### Step 3: Download and Setup Application

1. **Clone Repository**
   ```bash
   cd /workspace
   git clone https://github.com/your-repo/fireredtts2-runpod.git
   cd fireredtts2-runpod
   ```

2. **Install Dependencies**
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Install additional audio dependencies
   apt-get update
   apt-get install -y ffmpeg libsndfile1 portaudio19-dev
   
   # Install PyTorch with CUDA support (if not already installed)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Download Required Models**
   ```bash
   # Create models directory
   mkdir -p /workspace/models
   
   # Download Whisper models
   python -c "import whisper; whisper.load_model('base')"
   python -c "import whisper; whisper.load_model('small')"
   
   # Download FireRedTTS2 models (this may take 10-15 minutes)
   python download_models.py
   ```

### Step 4: Configure Application

1. **Update Configuration Files**
   ```bash
   # Copy and modify configuration
   cp runpod_config.json config.json
   
   # Edit configuration for your setup
   nano config.json
   ```

2. **Key Configuration Settings**
   ```json
   {
     "model_path": "/workspace/models",
     "cache_path": "/tmp/audio_cache",
     "gpu_device": "cuda:0",
     "max_concurrent_users": 4,
     "audio_sample_rate": 22050,
     "websocket_port": 7860,
     "http_port": 7860,
     "enable_voice_cloning": true,
     "enable_conversation": true,
     "whisper_model": "base",
     "llm_provider": "local"
   }
   ```

3. **Set Environment Variables**
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   export GRADIO_SERVER_NAME=0.0.0.0
   export GRADIO_SERVER_PORT=7860
   ```

### Step 5: Start the Application

1. **Run Startup Script**
   ```bash
   # Make startup script executable
   chmod +x container_startup.sh
   
   # Start the application
   ./container_startup.sh
   ```

2. **Alternative Manual Start**
   ```bash
   # Start the enhanced Gradio interface
   python enhanced_gradio_demo.py
   ```

3. **Verify Startup**
   - Look for "Running on local URL: http://0.0.0.0:7860" message
   - Check for model loading completion messages
   - Verify no CUDA errors in output

### Step 6: Access Web Interface

1. **Get Pod URL**
   - In RunPod console, click on your pod
   - Copy the "Connect" URL (usually ends with :7860)
   - Example: `https://abc123-7860.proxy.runpod.net`

2. **Test Connection**
   - Open the URL in your browser
   - You should see the FireRedTTS2 interface
   - Test microphone permissions when prompted

## Configuration Options

### GPU Memory Optimization

For different GPU configurations:

**RTX 4090 (24GB)**
```json
{
  "max_concurrent_users": 2,
  "model_precision": "fp16",
  "enable_model_offloading": true,
  "chunk_size": 1024
}
```

**A100 (40GB)**
```json
{
  "max_concurrent_users": 4,
  "model_precision": "fp16",
  "enable_model_offloading": false,
  "chunk_size": 2048
}
```

**H100 (80GB)**
```json
{
  "max_concurrent_users": 8,
  "model_precision": "fp16",
  "enable_model_offloading": false,
  "chunk_size": 4096
}
```

### Network Configuration

RunPod requires specific network settings:

```json
{
  "server_config": {
    "host": "0.0.0.0",
    "port": 7860,
    "protocol": "http",
    "enable_websockets": true,
    "websocket_upgrade": true,
    "proxy_compatible": true
  }
}
```

### Storage Configuration

```json
{
  "storage_config": {
    "model_cache_path": "/workspace/models",
    "audio_cache_path": "/tmp/audio_cache",
    "session_storage": "memory",
    "persistent_volume": "/workspace",
    "cleanup_interval": 3600
  }
}
```

## Verification Steps

### 1. System Health Check

```bash
# Check system status
python system_monitoring.py --health-check

# Verify GPU utilization
nvidia-smi

# Check disk space
df -h /workspace
df -h /tmp
```

### 2. Audio Pipeline Test

```bash
# Run comprehensive test suite
python comprehensive_test_suite.py --quick

# Test specific components
python -m pytest tests/test_audio_pipeline.py -v
python -m pytest tests/test_websocket_server.py -v
```

### 3. Web Interface Verification

1. **Access Interface**: Open pod URL in browser
2. **Test Microphone**: Click microphone button, grant permissions
3. **Test TTS**: Enter text, select voice, generate audio
4. **Test Voice Cloning**: Upload reference audio, test cloning
5. **Test Conversation**: Enable speech-to-speech mode, test conversation

## Performance Optimization

### Model Loading Optimization

```bash
# Pre-load models on startup
export PRELOAD_MODELS=true
export MODEL_CACHE_SIZE=4

# Enable model quantization for memory efficiency
export ENABLE_QUANTIZATION=true
export QUANTIZATION_BITS=8
```

### Audio Processing Optimization

```json
{
  "audio_optimization": {
    "buffer_size": 1024,
    "sample_rate": 22050,
    "channels": 1,
    "format": "float32",
    "enable_vad": true,
    "vad_threshold": 0.5
  }
}
```

### Concurrent User Management

```json
{
  "concurrency": {
    "max_users": 4,
    "queue_size": 10,
    "timeout_seconds": 30,
    "enable_load_balancing": true
  }
}
```

## Security Configuration

### Basic Security Settings

```json
{
  "security": {
    "enable_rate_limiting": true,
    "max_requests_per_minute": 60,
    "max_upload_size_mb": 50,
    "allowed_audio_formats": ["wav", "mp3", "flac", "webm"],
    "enable_input_validation": true
  }
}
```

### Access Control (Optional)

```json
{
  "access_control": {
    "enable_authentication": false,
    "api_key_required": false,
    "allowed_origins": ["*"],
    "cors_enabled": true
  }
}
```

## Monitoring and Logging

### Enable Monitoring

```bash
# Start monitoring dashboard
python system_monitoring.py --dashboard &

# Enable detailed logging
export LOG_LEVEL=INFO
export ENABLE_PERFORMANCE_LOGGING=true
```

### Log Configuration

```json
{
  "logging": {
    "level": "INFO",
    "file": "/workspace/logs/fireredtts2.log",
    "max_size_mb": 100,
    "backup_count": 5,
    "enable_performance_logs": true
  }
}
```

## Backup and Persistence

### Model Backup

```bash
# Backup models to persistent volume
cp -r /root/.cache/whisper /workspace/models/whisper_cache
cp -r /root/.cache/huggingface /workspace/models/hf_cache

# Create backup script
cat > /workspace/backup_models.sh << 'EOF'
#!/bin/bash
rsync -av /root/.cache/ /workspace/models/cache_backup/
echo "Model backup completed: $(date)"
EOF
chmod +x /workspace/backup_models.sh
```

### Configuration Backup

```bash
# Backup configuration files
cp config.json /workspace/config_backup.json
cp runpod_config.json /workspace/runpod_config_backup.json
```

## Troubleshooting Common Issues

See the separate troubleshooting guide for detailed solutions to common deployment issues.

## Next Steps

After successful deployment:

1. **Read the User Guide**: Familiarize yourself with web interface features
2. **Review Admin Guide**: Learn about monitoring and maintenance procedures
3. **Test All Features**: Verify voice cloning, conversation, and multi-speaker capabilities
4. **Monitor Performance**: Use built-in monitoring tools to optimize performance
5. **Set Up Backups**: Implement regular backup procedures for models and configurations

## Support and Resources

- **Documentation**: See additional guides in the `docs/` directory
- **Troubleshooting**: Refer to `docs/troubleshooting-guide.md`
- **System Administration**: See `docs/admin-guide.md`
- **API Reference**: Available at `/docs` endpoint when running

For additional support, check the project repository issues or contact the development team.