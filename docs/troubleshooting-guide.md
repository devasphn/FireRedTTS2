# FireRedTTS2 Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when deploying and using the FireRedTTS2 system on RunPod. Issues are organized by category with step-by-step resolution procedures.

## Quick Diagnostic Checklist

Before diving into specific issues, run this quick diagnostic:

```bash
# System health check
python system_monitoring.py --health-check

# GPU status
nvidia-smi

# Disk space
df -h

# Memory usage
free -h

# Process status
ps aux | grep python

# Network connectivity
curl -I http://localhost:7860
```

## Deployment Issues

### Pod Won't Start

**Symptoms:**
- Pod shows "Starting" status indefinitely
- Container fails to initialize
- No response from web interface

**Causes & Solutions:**

1. **Insufficient GPU Memory**
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # If memory is full, restart pod or reduce model size
   # Edit config.json to use smaller models
   {
     "whisper_model": "tiny",  # Instead of "base" or "small"
     "model_precision": "fp16",
     "enable_model_offloading": true
   }
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   
   # Install system dependencies
   apt-get update
   apt-get install -y ffmpeg libsndfile1 portaudio19-dev
   ```

3. **Port Conflicts**
   ```bash
   # Check if port 7860 is in use
   netstat -tulpn | grep 7860
   
   # Kill conflicting processes
   pkill -f gradio
   pkill -f python
   
   # Restart application
   python enhanced_gradio_demo.py
   ```

### Model Download Failures

**Symptoms:**
- Models fail to download
- "Model not found" errors
- Incomplete model files

**Solutions:**

1. **Network Issues**
   ```bash
   # Test internet connectivity
   ping google.com
   
   # Check DNS resolution
   nslookup huggingface.co
   
   # Use alternative download method
   wget https://huggingface.co/model-url -O /workspace/models/model.bin
   ```

2. **Storage Space Issues**
   ```bash
   # Check available space
   df -h /workspace
   
   # Clean up temporary files
   rm -rf /tmp/*
   rm -rf /root/.cache/pip
   
   # Move models to persistent volume
   mkdir -p /workspace/models
   mv /root/.cache/huggingface /workspace/models/
   ```

3. **Authentication Issues**
   ```bash
   # Set Hugging Face token if required
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   
   # Login to Hugging Face CLI
   huggingface-cli login
   ```

### Configuration Errors

**Symptoms:**
- Application starts but features don't work
- Error messages about missing configuration
- Incorrect model paths

**Solutions:**

1. **Fix Configuration Paths**
   ```bash
   # Verify config.json exists and is valid
   cat config.json | python -m json.tool
   
   # Update paths for RunPod environment
   sed -i 's|/local/path|/workspace|g' config.json
   
   # Set correct model paths
   {
     "model_path": "/workspace/models",
     "cache_path": "/tmp/audio_cache",
     "whisper_model_path": "/workspace/models/whisper"
   }
   ```

2. **Environment Variables**
   ```bash
   # Set required environment variables
   export CUDA_VISIBLE_DEVICES=0
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   export GRADIO_SERVER_NAME=0.0.0.0
   export GRADIO_SERVER_PORT=7860
   
   # Add to startup script
   echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
   ```

## Runtime Issues

### High Memory Usage

**Symptoms:**
- Out of memory errors
- Slow performance
- System freezing

**Solutions:**

1. **GPU Memory Optimization**
   ```bash
   # Monitor GPU memory
   watch -n 1 nvidia-smi
   
   # Clear GPU cache
   python -c "import torch; torch.cuda.empty_cache()"
   
   # Reduce batch size in config
   {
     "batch_size": 1,
     "max_concurrent_users": 2,
     "enable_model_offloading": true
   }
   ```

2. **System Memory Optimization**
   ```bash
   # Check memory usage
   free -h
   
   # Clear system cache
   sync && echo 3 > /proc/sys/vm/drop_caches
   
   # Restart application with memory limits
   python -Xmx4g enhanced_gradio_demo.py
   ```

3. **Model Optimization**
   ```json
   {
     "model_optimization": {
       "use_quantization": true,
       "quantization_bits": 8,
       "enable_cpu_offload": true,
       "max_model_cache": 2
     }
   }
   ```

### Audio Processing Issues

**Symptoms:**
- No audio output
- Distorted audio
- Audio playback errors

**Solutions:**

1. **Audio Format Issues**
   ```bash
   # Check supported formats
   python -c "import soundfile as sf; print(sf.available_formats())"
   
   # Convert audio format
   ffmpeg -i input.webm -ar 22050 -ac 1 output.wav
   
   # Update audio configuration
   {
     "audio_config": {
       "sample_rate": 22050,
       "channels": 1,
       "format": "wav",
       "bit_depth": 16
     }
   }
   ```

2. **WebSocket Audio Streaming**
   ```bash
   # Test WebSocket connection
   python -c "
   import websocket
   ws = websocket.create_connection('ws://localhost:7860/audio')
   print('WebSocket connected')
   ws.close()
   "
   
   # Check audio streaming logs
   tail -f /workspace/logs/audio_streaming.log
   ```

3. **Browser Audio Issues**
   - Clear browser cache and cookies
   - Try different browser (Chrome recommended)
   - Check browser audio permissions
   - Disable browser extensions
   - Test with incognito/private mode

### Performance Issues

**Symptoms:**
- Slow response times
- High latency
- Timeouts

**Solutions:**

1. **Optimize Processing Pipeline**
   ```json
   {
     "performance_optimization": {
       "enable_caching": true,
       "preload_models": true,
       "use_gpu_acceleration": true,
       "optimize_for_latency": true
     }
   }
   ```

2. **Network Optimization**
   ```bash
   # Test network latency
   ping -c 10 google.com
   
   # Optimize WebSocket settings
   {
     "websocket_config": {
       "buffer_size": 4096,
       "compression": true,
       "heartbeat_interval": 30
     }
   }
   ```

3. **Resource Monitoring**
   ```bash
   # Monitor system resources
   htop
   
   # Check I/O wait
   iostat -x 1
   
   # Monitor network usage
   iftop
   ```

## Web Interface Issues

### Interface Won't Load

**Symptoms:**
- Blank page or loading spinner
- JavaScript errors in browser console
- 404 or 500 errors

**Solutions:**

1. **Check Server Status**
   ```bash
   # Verify server is running
   ps aux | grep gradio
   
   # Check server logs
   tail -f /workspace/logs/gradio.log
   
   # Test local connection
   curl http://localhost:7860
   ```

2. **Browser Issues**
   - Clear browser cache (Ctrl+Shift+Delete)
   - Disable ad blockers and extensions
   - Try different browser
   - Check browser console for JavaScript errors

3. **Proxy Issues**
   ```bash
   # Check RunPod proxy configuration
   curl -I https://your-pod-url.proxy.runpod.net
   
   # Verify port forwarding
   netstat -tulpn | grep 7860
   ```

### Microphone Access Issues

**Symptoms:**
- Microphone permission denied
- No audio input detected
- Voice activity detection not working

**Solutions:**

1. **Browser Permissions**
   - Click lock icon in address bar
   - Allow microphone access
   - Refresh page after granting permissions
   - Check browser microphone settings

2. **Audio Input Configuration**
   ```javascript
   // Test microphone access in browser console
   navigator.mediaDevices.getUserMedia({ audio: true })
     .then(stream => console.log('Microphone access granted'))
     .catch(err => console.error('Microphone access denied:', err));
   ```

3. **WebRTC Issues**
   - Ensure HTTPS connection (required for microphone)
   - Check WebRTC compatibility
   - Try different audio input device
   - Test with different browser

### Voice Cloning Problems

**Symptoms:**
- Voice cloning fails
- Poor quality cloned voices
- Reference audio upload errors

**Solutions:**

1. **Audio Quality Issues**
   ```bash
   # Check audio file properties
   ffprobe reference_audio.wav
   
   # Convert to optimal format
   ffmpeg -i input.mp3 -ar 22050 -ac 1 -sample_fmt s16 output.wav
   
   # Validate audio quality
   python -c "
   import librosa
   y, sr = librosa.load('reference_audio.wav')
   print(f'Duration: {len(y)/sr:.2f}s, Sample rate: {sr}')
   "
   ```

2. **Reference Text Mismatch**
   - Ensure reference text exactly matches spoken audio
   - Check for punctuation and capitalization
   - Remove background noise from audio
   - Use clear, natural speech

3. **Model Issues**
   ```bash
   # Check voice cloning model status
   python -c "
   from voice_cloning_interface import VoiceCloningInterface
   vci = VoiceCloningInterface()
   print(vci.check_model_status())
   "
   ```

## Network and Connectivity Issues

### WebSocket Connection Failures

**Symptoms:**
- Real-time features not working
- Connection drops frequently
- WebSocket errors in browser console

**Solutions:**

1. **Connection Diagnostics**
   ```bash
   # Test WebSocket server
   python websocket_server.py --test
   
   # Check WebSocket logs
   tail -f /workspace/logs/websocket.log
   
   # Monitor connections
   netstat -an | grep 7860
   ```

2. **Proxy Configuration**
   ```json
   {
     "websocket_config": {
       "proxy_compatible": true,
       "upgrade_insecure": true,
       "heartbeat_interval": 30,
       "reconnect_attempts": 5
     }
   }
   ```

3. **Firewall Issues**
   ```bash
   # Check iptables rules
   iptables -L
   
   # Allow WebSocket traffic
   iptables -A INPUT -p tcp --dport 7860 -j ACCEPT
   ```

### API Connection Issues

**Symptoms:**
- API requests failing
- Timeout errors
- Authentication failures

**Solutions:**

1. **API Endpoint Testing**
   ```bash
   # Test API endpoints
   curl -X GET http://localhost:7860/api/health
   curl -X POST http://localhost:7860/api/tts -d '{"text":"test"}'
   
   # Check API logs
   tail -f /workspace/logs/api.log
   ```

2. **Authentication Issues**
   ```bash
   # Check API key configuration
   grep -r "api_key" config.json
   
   # Test with authentication
   curl -H "Authorization: Bearer your-api-key" http://localhost:7860/api/tts
   ```

## Model-Specific Issues

### Whisper ASR Problems

**Symptoms:**
- Speech recognition not working
- Poor transcription accuracy
- ASR timeouts

**Solutions:**

1. **Model Configuration**
   ```json
   {
     "whisper_config": {
       "model_size": "base",
       "language": "auto",
       "task": "transcribe",
       "temperature": 0.0,
       "best_of": 1
     }
   }
   ```

2. **Audio Preprocessing**
   ```python
   # Test Whisper directly
   import whisper
   model = whisper.load_model("base")
   result = model.transcribe("test_audio.wav")
   print(result["text"])
   ```

3. **Performance Optimization**
   ```bash
   # Use faster Whisper implementation
   pip install faster-whisper
   
   # Update configuration
   {
     "asr_backend": "faster-whisper",
     "compute_type": "float16"
   }
   ```

### TTS Model Issues

**Symptoms:**
- TTS generation fails
- Poor audio quality
- Model loading errors

**Solutions:**

1. **Model Validation**
   ```python
   # Test FireRedTTS2 directly
   from enhanced_fireredtts2 import EnhancedFireRedTTS2
   tts = EnhancedFireRedTTS2()
   audio = tts.generate("Hello world", voice_profile="default")
   ```

2. **GPU Memory Issues**
   ```json
   {
     "tts_config": {
       "model_precision": "fp16",
       "enable_cpu_offload": true,
       "max_batch_size": 1,
       "chunk_size": 1024
     }
   }
   ```

### LLM Integration Issues

**Symptoms:**
- Conversation responses fail
- LLM timeouts
- Poor response quality

**Solutions:**

1. **Local LLM Issues**
   ```bash
   # Check LLM model status
   python -c "
   from conversation_llm import ConversationLLM
   llm = ConversationLLM()
   response = llm.generate('Hello')
   print(response)
   "
   ```

2. **API LLM Issues**
   ```bash
   # Test API connectivity
   curl -X POST https://api.openai.com/v1/chat/completions \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}]}'
   ```

## System Monitoring and Diagnostics

### Health Check Script

Create a comprehensive health check script:

```bash
#!/bin/bash
# health_check.sh

echo "=== FireRedTTS2 Health Check ==="
echo "Timestamp: $(date)"
echo

# System resources
echo "=== System Resources ==="
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo
echo "Memory Usage:"
free -h
echo
echo "Disk Usage:"
df -h /workspace /tmp
echo

# Process status
echo "=== Process Status ==="
ps aux | grep -E "(python|gradio)" | grep -v grep
echo

# Network status
echo "=== Network Status ==="
netstat -tulpn | grep -E "(7860|8000)"
echo

# Model status
echo "=== Model Status ==="
ls -la /workspace/models/
echo

# Log errors
echo "=== Recent Errors ==="
tail -n 20 /workspace/logs/*.log | grep -i error
echo

# API test
echo "=== API Test ==="
curl -s -o /dev/null -w "%{http_code}" http://localhost:7860/api/health
echo
```

### Performance Monitoring

```python
# performance_monitor.py
import psutil
import GPUtil
import time
import json

def monitor_system():
    while True:
        # GPU metrics
        gpus = GPUtil.getGPUs()
        gpu_data = {
            'utilization': gpus[0].load * 100,
            'memory_used': gpus[0].memoryUsed,
            'memory_total': gpus[0].memoryTotal,
            'temperature': gpus[0].temperature
        }
        
        # System metrics
        system_data = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        # Combine and log
        metrics = {
            'timestamp': time.time(),
            'gpu': gpu_data,
            'system': system_data
        }
        
        print(json.dumps(metrics, indent=2))
        time.sleep(10)

if __name__ == "__main__":
    monitor_system()
```

## Emergency Recovery Procedures

### Complete System Reset

If the system is completely unresponsive:

```bash
# 1. Stop all processes
pkill -f python
pkill -f gradio

# 2. Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# 3. Clear system cache
sync && echo 3 > /proc/sys/vm/drop_caches

# 4. Remove temporary files
rm -rf /tmp/*
rm -rf /root/.cache/pip

# 5. Restart application
cd /workspace/fireredtts2-runpod
./container_startup.sh
```

### Model Recovery

If models are corrupted or missing:

```bash
# 1. Backup current models
mv /workspace/models /workspace/models_backup

# 2. Create fresh model directory
mkdir -p /workspace/models

# 3. Re-download models
python download_models.py

# 4. Verify model integrity
python -c "
import torch
model = torch.load('/workspace/models/model.pt')
print('Model loaded successfully')
"
```

### Configuration Recovery

If configuration is corrupted:

```bash
# 1. Backup current config
cp config.json config_backup.json

# 2. Restore default configuration
cp runpod_config.json config.json

# 3. Update paths for current environment
sed -i 's|/default/path|/workspace|g' config.json

# 4. Restart application
python enhanced_gradio_demo.py
```

## Getting Additional Help

### Log Collection

When reporting issues, collect these logs:

```bash
# Create support bundle
mkdir -p /workspace/support_bundle
cp config.json /workspace/support_bundle/
cp /workspace/logs/*.log /workspace/support_bundle/
nvidia-smi > /workspace/support_bundle/gpu_status.txt
free -h > /workspace/support_bundle/memory_status.txt
df -h > /workspace/support_bundle/disk_status.txt
ps aux > /workspace/support_bundle/process_status.txt

# Create archive
tar -czf support_bundle.tar.gz /workspace/support_bundle/
```

### Diagnostic Information

Include this information when seeking help:

- RunPod pod type and GPU model
- Error messages (exact text)
- Steps to reproduce the issue
- Browser type and version
- Audio file formats being used
- Configuration settings
- System resource usage

### Contact Information

- **System Administrator**: Contact your RunPod deployment administrator
- **Technical Documentation**: Refer to other guides in `/docs` directory
- **Community Support**: Check project repository for known issues
- **Emergency Support**: Follow emergency recovery procedures above

---

This troubleshooting guide covers the most common issues. For complex problems, combine multiple solutions and always check system logs for additional clues.