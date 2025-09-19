# FireRedTTS2 Quick Reference

## Emergency Commands

### System Health Check
```bash
# Quick health check
python system_monitoring.py --health-check

# GPU status
nvidia-smi

# Application status
ps aux | grep gradio
```

### Restart Services
```bash
# Stop application
pkill -f gradio

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Restart application
./container_startup.sh
```

### Clear Cache and Temporary Files
```bash
# Clear audio cache
rm -rf /tmp/audio_*

# Clear model cache
python -c "import torch; torch.cuda.empty_cache()"

# Clear system cache
sync && echo 3 > /proc/sys/vm/drop_caches
```

## Common Configuration Changes

### Reduce Memory Usage
```json
{
  "max_concurrent_users": 2,
  "model_precision": "fp16",
  "enable_model_offloading": true,
  "whisper_model": "tiny"
}
```

### Optimize for Speed
```json
{
  "enable_caching": true,
  "preload_models": true,
  "optimize_for_latency": true,
  "batch_size": 1
}
```

### Enable Debug Logging
```bash
export LOG_LEVEL=DEBUG
export ENABLE_PERFORMANCE_LOGGING=true
```

## Troubleshooting Quick Fixes

### Audio Not Working
1. Check browser permissions
2. Refresh page
3. Try different browser
4. Check system volume

### High Memory Usage
1. Reduce concurrent users
2. Enable model offloading
3. Use smaller models
4. Clear GPU cache

### Slow Performance
1. Check GPU utilization
2. Reduce text length
3. Enable caching
4. Use faster models

### Connection Issues
1. Check port 7860 is open
2. Verify RunPod proxy
3. Test local connection
4. Check firewall settings

## Useful URLs

- **Main Interface**: `https://your-pod-id.proxy.runpod.net`
- **API Documentation**: `https://your-pod-id.proxy.runpod.net/docs`
- **Health Check**: `https://your-pod-id.proxy.runpod.net/api/health`
- **Monitoring**: `https://your-pod-id.proxy.runpod.net/monitor`

## Log Locations

- **Application Logs**: `/workspace/logs/application.log`
- **Error Logs**: `/workspace/logs/error.log`
- **Security Logs**: `/workspace/logs/security.log`
- **Performance Logs**: `/workspace/logs/performance.log`

## Key File Locations

- **Configuration**: `/workspace/config.json`
- **Models**: `/workspace/models/`
- **Cache**: `/tmp/audio_cache/`
- **Backups**: `/workspace/backups/`
- **Scripts**: `/workspace/scripts/`

## Performance Monitoring

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Monitor System Resources
```bash
htop
```

### Check Disk Space
```bash
df -h /workspace
```

### Monitor Network
```bash
netstat -tulpn | grep 7860
```

## API Quick Reference

### Text-to-Speech
```bash
curl -X POST http://localhost:7860/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "default"}'
```

### Health Check
```bash
curl http://localhost:7860/api/health
```

### System Metrics
```bash
curl http://localhost:7860/api/metrics
```

## Common Error Messages

| Error | Quick Fix |
|-------|-----------|
| "CUDA out of memory" | Reduce concurrent users, clear GPU cache |
| "Model not found" | Check model paths, re-download models |
| "Connection refused" | Check if application is running, verify ports |
| "Permission denied" | Check file permissions, browser permissions |
| "WebSocket connection failed" | Check proxy settings, try different browser |

## Maintenance Schedule

### Daily
- Check system health
- Monitor resource usage
- Clear temporary files

### Weekly
- Analyze performance logs
- Update system metrics
- Clean old cache files

### Monthly
- Full system backup
- Security audit
- Performance optimization

## Contact Information

- **System Administrator**: [Your admin contact]
- **Technical Support**: [Your support contact]
- **Emergency Contact**: [Your emergency contact]

---

For detailed information, refer to the complete documentation guides.