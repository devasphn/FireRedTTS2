#!/usr/bin/env python3
"""
Audio Streaming Protocols
Comprehensive audio streaming protocols with buffering, quality control, and error recovery
"""

import asyncio
import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

import numpy as np
import torch

logger = logging.getLogger(__name__)

class AudioFormat(Enum):
    """Supported audio formats"""
    PCM_16 = "pcm_16"
    PCM_32 = "pcm_32"
    FLOAT32 = "float32"
    OPUS = "opus"
    MP3 = "mp3"
    WEBM = "webm"

class StreamingMode(Enum):
    """Audio streaming modes"""
    REAL_TIME = "real_time"
    BUFFERED = "buffered"
    ADAPTIVE = "adaptive"

class QualityLevel(Enum):
    """Audio quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LOSSLESS = "lossless"

@dataclass
class AudioConfig:
    """Audio configuration"""
    sample_rate: int = 16000
    channels: int = 1
    format: AudioFormat = AudioFormat.PCM_16
    bit_depth: int = 16
    chunk_duration_ms: int = 100
    buffer_size_ms: int = 1000
    quality_level: QualityLevel = QualityLevel.MEDIUM
    enable_compression: bool = True
    enable_noise_reduction: bool = False
    enable_echo_cancellation: bool = False

@dataclass
class AudioChunk:
    """Audio chunk with metadata"""
    chunk_id: int
    timestamp: float
    sample_rate: int
    channels: int
    format: AudioFormat
    data: np.ndarray
    duration_ms: float
    sequence_number: int
    is_silence: bool = False
    quality_score: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class StreamingStats:
    """Streaming statistics"""
    total_chunks: int = 0
    bytes_transmitted: int = 0
    packets_lost: int = 0
    packets_duplicated: int = 0
    average_latency_ms: float = 0.0
    jitter_ms: float = 0.0
    quality_score: float = 1.0
    buffer_underruns: int = 0
    buffer_overruns: int = 0
    start_time: float = 0.0
    last_update: float = 0.0

class AudioBuffer:
    """Circular audio buffer with adaptive sizing"""
    
    def __init__(self, 
                 config: AudioConfig,
                 max_size_ms: int = 5000,
                 min_size_ms: int = 100):
        
        self.config = config
        self.max_size_ms = max_size_ms
        self.min_size_ms = min_size_ms
        
        # Calculate buffer parameters
        self.chunk_size_samples = int(config.chunk_duration_ms * config.sample_rate / 1000)
        self.max_chunks = int(max_size_ms / config.chunk_duration_ms)
        self.min_chunks = int(min_size_ms / config.chunk_duration_ms)
        
        # Buffer storage
        self.buffer = deque(maxlen=self.max_chunks)
        self.sequence_numbers = deque(maxlen=self.max_chunks)
        
        # State tracking
        self.total_chunks_added = 0
        self.total_chunks_consumed = 0
        self.underrun_count = 0
        self.overrun_count = 0
        
        # Adaptive parameters
        self.target_buffer_size = self.min_chunks * 2
        self.adaptation_rate = 0.1
        
        # Thread safety
        self.lock = threading.Lock()
    
    def add_chunk(self, chunk: AudioChunk) -> bool:
        """Add audio chunk to buffer"""
        
        with self.lock:
            # Check for buffer overrun
            if len(self.buffer) >= self.max_chunks:
                self.overrun_count += 1
                # Remove oldest chunk
                self.buffer.popleft()
                self.sequence_numbers.popleft()
            
            # Add new chunk
            self.buffer.append(chunk)
            self.sequence_numbers.append(chunk.sequence_number)
            self.total_chunks_added += 1
            
            return True
    
    def get_chunk(self) -> Optional[AudioChunk]:
        """Get next audio chunk from buffer"""
        
        with self.lock:
            if not self.buffer:
                self.underrun_count += 1
                return None
            
            chunk = self.buffer.popleft()
            self.sequence_numbers.popleft()
            self.total_chunks_consumed += 1
            
            return chunk
    
    def peek_chunk(self) -> Optional[AudioChunk]:
        """Peek at next chunk without removing it"""
        
        with self.lock:
            return self.buffer[0] if self.buffer else None
    
    def get_buffer_level(self) -> float:
        """Get buffer level as percentage of target size"""
        
        with self.lock:
            return len(self.buffer) / max(self.target_buffer_size, 1)
    
    def get_buffer_duration_ms(self) -> float:
        """Get current buffer duration in milliseconds"""
        
        with self.lock:
            return len(self.buffer) * self.config.chunk_duration_ms
    
    def adapt_buffer_size(self, network_conditions: Dict[str, float]):
        """Adapt buffer size based on network conditions"""
        
        latency = network_conditions.get("latency_ms", 0)
        jitter = network_conditions.get("jitter_ms", 0)
        packet_loss = network_conditions.get("packet_loss_rate", 0)
        
        # Calculate optimal buffer size
        base_buffer_ms = self.min_size_ms
        latency_buffer_ms = latency * 2  # 2x latency for safety
        jitter_buffer_ms = jitter * 3    # 3x jitter for stability
        loss_buffer_ms = packet_loss * 1000  # Additional buffer for packet loss
        
        optimal_buffer_ms = base_buffer_ms + latency_buffer_ms + jitter_buffer_ms + loss_buffer_ms
        optimal_buffer_ms = min(optimal_buffer_ms, self.max_size_ms)
        
        # Adapt gradually
        new_target_chunks = int(optimal_buffer_ms / self.config.chunk_duration_ms)
        self.target_buffer_size = int(
            (1 - self.adaptation_rate) * self.target_buffer_size +
            self.adaptation_rate * new_target_chunks
        )
        
        self.target_buffer_size = max(self.min_chunks, min(self.target_buffer_size, self.max_chunks))
    
    def clear(self):
        """Clear buffer"""
        
        with self.lock:
            self.buffer.clear()
            self.sequence_numbers.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        
        with self.lock:
            return {
                "current_size": len(self.buffer),
                "target_size": self.target_buffer_size,
                "max_size": self.max_chunks,
                "buffer_level": self.get_buffer_level(),
                "duration_ms": self.get_buffer_duration_ms(),
                "total_added": self.total_chunks_added,
                "total_consumed": self.total_chunks_consumed,
                "underruns": self.underrun_count,
                "overruns": self.overrun_count
            }

class AudioEncoder:
    """Audio encoder for different formats"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def encode_chunk(self, audio_data: np.ndarray) -> bytes:
        """Encode audio chunk to specified format"""
        
        if self.config.format == AudioFormat.PCM_16:
            return self._encode_pcm_16(audio_data)
        elif self.config.format == AudioFormat.PCM_32:
            return self._encode_pcm_32(audio_data)
        elif self.config.format == AudioFormat.FLOAT32:
            return self._encode_float32(audio_data)
        elif self.config.format == AudioFormat.OPUS:
            return self._encode_opus(audio_data)
        else:
            # Default to PCM 16
            return self._encode_pcm_16(audio_data)
    
    def _encode_pcm_16(self, audio_data: np.ndarray) -> bytes:
        """Encode as 16-bit PCM"""
        
        # Normalize to [-1, 1] range
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Clip to prevent overflow
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to 16-bit integers
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        return audio_int16.tobytes()
    
    def _encode_pcm_32(self, audio_data: np.ndarray) -> bytes:
        """Encode as 32-bit PCM"""
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int32 = (audio_data * 2147483647).astype(np.int32)
        
        return audio_int32.tobytes()
    
    def _encode_float32(self, audio_data: np.ndarray) -> bytes:
        """Encode as 32-bit float"""
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        return audio_data.tobytes()
    
    def _encode_opus(self, audio_data: np.ndarray) -> bytes:
        """Encode using Opus codec (placeholder)"""
        
        # Placeholder for Opus encoding
        # In production, use opuslib or similar
        logger.warning("Opus encoding not implemented, falling back to PCM 16")
        return self._encode_pcm_16(audio_data)

class AudioDecoder:
    """Audio decoder for different formats"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def decode_chunk(self, encoded_data: bytes) -> np.ndarray:
        """Decode audio chunk from specified format"""
        
        if self.config.format == AudioFormat.PCM_16:
            return self._decode_pcm_16(encoded_data)
        elif self.config.format == AudioFormat.PCM_32:
            return self._decode_pcm_32(encoded_data)
        elif self.config.format == AudioFormat.FLOAT32:
            return self._decode_float32(encoded_data)
        elif self.config.format == AudioFormat.OPUS:
            return self._decode_opus(encoded_data)
        else:
            # Default to PCM 16
            return self._decode_pcm_16(encoded_data)
    
    def _decode_pcm_16(self, encoded_data: bytes) -> np.ndarray:
        """Decode 16-bit PCM"""
        
        audio_int16 = np.frombuffer(encoded_data, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32767.0
        
        return audio_float
    
    def _decode_pcm_32(self, encoded_data: bytes) -> np.ndarray:
        """Decode 32-bit PCM"""
        
        audio_int32 = np.frombuffer(encoded_data, dtype=np.int32)
        audio_float = audio_int32.astype(np.float32) / 2147483647.0
        
        return audio_float
    
    def _decode_float32(self, encoded_data: bytes) -> np.ndarray:
        """Decode 32-bit float"""
        
        return np.frombuffer(encoded_data, dtype=np.float32)
    
    def _decode_opus(self, encoded_data: bytes) -> np.ndarray:
        """Decode Opus codec (placeholder)"""
        
        # Placeholder for Opus decoding
        logger.warning("Opus decoding not implemented, assuming PCM 16")
        return self._decode_pcm_16(encoded_data)

class QualityController:
    """Controls audio quality based on network conditions"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.current_quality = config.quality_level
        
        # Quality parameters
        self.quality_configs = {
            QualityLevel.LOW: {
                "sample_rate": 8000,
                "bit_depth": 8,
                "compression_ratio": 0.3
            },
            QualityLevel.MEDIUM: {
                "sample_rate": 16000,
                "bit_depth": 16,
                "compression_ratio": 0.6
            },
            QualityLevel.HIGH: {
                "sample_rate": 24000,
                "bit_depth": 16,
                "compression_ratio": 0.8
            },
            QualityLevel.LOSSLESS: {
                "sample_rate": 48000,
                "bit_depth": 24,
                "compression_ratio": 1.0
            }
        }
    
    def adapt_quality(self, network_conditions: Dict[str, float]) -> QualityLevel:
        """Adapt quality based on network conditions"""
        
        bandwidth_kbps = network_conditions.get("bandwidth_kbps", 1000)
        latency_ms = network_conditions.get("latency_ms", 100)
        packet_loss_rate = network_conditions.get("packet_loss_rate", 0.0)
        
        # Calculate quality score
        bandwidth_score = min(bandwidth_kbps / 500, 1.0)  # 500 kbps for high quality
        latency_score = max(0, 1.0 - latency_ms / 500)    # Penalty for high latency
        loss_score = max(0, 1.0 - packet_loss_rate * 10)  # Penalty for packet loss
        
        overall_score = (bandwidth_score + latency_score + loss_score) / 3
        
        # Select quality level
        if overall_score >= 0.8:
            new_quality = QualityLevel.HIGH
        elif overall_score >= 0.6:
            new_quality = QualityLevel.MEDIUM
        else:
            new_quality = QualityLevel.LOW
        
        # Update current quality if changed
        if new_quality != self.current_quality:
            logger.info(f"Quality adapted from {self.current_quality.value} to {new_quality.value}")
            self.current_quality = new_quality
        
        return self.current_quality
    
    def get_quality_config(self) -> Dict[str, Any]:
        """Get current quality configuration"""
        return self.quality_configs[self.current_quality]

class AudioStreamingProtocol:
    """Main audio streaming protocol handler"""
    
    def __init__(self, 
                 config: AudioConfig,
                 streaming_mode: StreamingMode = StreamingMode.ADAPTIVE):
        
        self.config = config
        self.streaming_mode = streaming_mode
        
        # Components
        self.encoder = AudioEncoder(config)
        self.decoder = AudioDecoder(config)
        self.quality_controller = QualityController(config)
        
        # Buffers
        self.input_buffer = AudioBuffer(config)
        self.output_buffer = AudioBuffer(config)
        
        # State
        self.is_streaming = False
        self.sequence_number = 0
        self.stats = StreamingStats()
        
        # Network monitoring
        self.network_conditions = {
            "bandwidth_kbps": 1000,
            "latency_ms": 100,
            "jitter_ms": 10,
            "packet_loss_rate": 0.0
        }
        
        # Callbacks
        self.chunk_received_callback = None
        self.chunk_sent_callback = None
        self.quality_changed_callback = None
    
    def start_streaming(self):
        """Start audio streaming"""
        
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.stats.start_time = time.time()
        self.sequence_number = 0
        
        logger.info(f"Audio streaming started in {self.streaming_mode.value} mode")
    
    def stop_streaming(self):
        """Stop audio streaming"""
        
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Clear buffers
        self.input_buffer.clear()
        self.output_buffer.clear()
        
        logger.info("Audio streaming stopped")
    
    def send_audio_chunk(self, audio_data: np.ndarray) -> Optional[bytes]:
        """Process and encode audio chunk for transmission"""
        
        if not self.is_streaming:
            return None
        
        try:
            # Create audio chunk
            chunk = AudioChunk(
                chunk_id=self.sequence_number,
                timestamp=time.time(),
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                format=self.config.format,
                data=audio_data,
                duration_ms=len(audio_data) / self.config.sample_rate * 1000,
                sequence_number=self.sequence_number,
                is_silence=self._detect_silence(audio_data),
                quality_score=self._calculate_quality_score(audio_data)
            )
            
            # Add to input buffer
            self.input_buffer.add_chunk(chunk)
            
            # Encode chunk
            encoded_data = self.encoder.encode_chunk(audio_data)
            
            # Update statistics
            self.stats.total_chunks += 1
            self.stats.bytes_transmitted += len(encoded_data)
            self.sequence_number += 1
            
            # Trigger callback
            if self.chunk_sent_callback:
                self.chunk_sent_callback(chunk, encoded_data)
            
            return encoded_data
            
        except Exception as e:
            logger.error(f"Failed to send audio chunk: {e}")
            return None
    
    def receive_audio_chunk(self, encoded_data: bytes, metadata: Dict[str, Any] = None) -> Optional[AudioChunk]:
        """Decode and process received audio chunk"""
        
        if not self.is_streaming:
            return None
        
        try:
            # Decode audio data
            audio_data = self.decoder.decode_chunk(encoded_data)
            
            # Extract metadata
            metadata = metadata or {}
            chunk_id = metadata.get("chunk_id", 0)
            timestamp = metadata.get("timestamp", time.time())
            sequence_number = metadata.get("sequence_number", 0)
            
            # Create audio chunk
            chunk = AudioChunk(
                chunk_id=chunk_id,
                timestamp=timestamp,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                format=self.config.format,
                data=audio_data,
                duration_ms=len(audio_data) / self.config.sample_rate * 1000,
                sequence_number=sequence_number,
                is_silence=self._detect_silence(audio_data),
                quality_score=self._calculate_quality_score(audio_data),
                metadata=metadata
            )
            
            # Add to output buffer
            self.output_buffer.add_chunk(chunk)
            
            # Update statistics
            self._update_receive_stats(chunk)
            
            # Trigger callback
            if self.chunk_received_callback:
                self.chunk_received_callback(chunk)
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to receive audio chunk: {e}")
            return None
    
    def get_next_output_chunk(self) -> Optional[AudioChunk]:
        """Get next audio chunk for playback"""
        
        return self.output_buffer.get_chunk()
    
    def update_network_conditions(self, conditions: Dict[str, float]):
        """Update network conditions for adaptive streaming"""
        
        self.network_conditions.update(conditions)
        
        # Adapt quality
        old_quality = self.quality_controller.current_quality
        new_quality = self.quality_controller.adapt_quality(conditions)
        
        if new_quality != old_quality and self.quality_changed_callback:
            self.quality_changed_callback(old_quality, new_quality)
        
        # Adapt buffer sizes
        self.input_buffer.adapt_buffer_size(conditions)
        self.output_buffer.adapt_buffer_size(conditions)
    
    def _detect_silence(self, audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """Detect if audio chunk is silence"""
        
        energy = np.sqrt(np.mean(audio_data ** 2))
        return energy < threshold
    
    def _calculate_quality_score(self, audio_data: np.ndarray) -> float:
        """Calculate quality score for audio chunk"""
        
        # Simple quality metrics
        energy = np.sqrt(np.mean(audio_data ** 2))
        dynamic_range = np.max(audio_data) - np.min(audio_data)
        
        # Normalize scores
        energy_score = min(energy / 0.5, 1.0)
        range_score = min(dynamic_range / 2.0, 1.0)
        
        return (energy_score + range_score) / 2
    
    def _update_receive_stats(self, chunk: AudioChunk):
        """Update receive statistics"""
        
        current_time = time.time()
        
        # Calculate latency
        if chunk.timestamp > 0:
            latency = (current_time - chunk.timestamp) * 1000
            
            # Update average latency
            if self.stats.average_latency_ms == 0:
                self.stats.average_latency_ms = latency
            else:
                self.stats.average_latency_ms = (
                    0.9 * self.stats.average_latency_ms + 0.1 * latency
                )
        
        # Update quality score
        if self.stats.quality_score == 0:
            self.stats.quality_score = chunk.quality_score
        else:
            self.stats.quality_score = (
                0.9 * self.stats.quality_score + 0.1 * chunk.quality_score
            )
        
        self.stats.last_update = current_time
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics"""
        
        return {
            "protocol_stats": asdict(self.stats),
            "input_buffer_stats": self.input_buffer.get_stats(),
            "output_buffer_stats": self.output_buffer.get_stats(),
            "network_conditions": self.network_conditions,
            "current_quality": self.quality_controller.current_quality.value,
            "streaming_mode": self.streaming_mode.value,
            "is_streaming": self.is_streaming
        }
    
    def set_callbacks(self,
                     chunk_received: Optional[callable] = None,
                     chunk_sent: Optional[callable] = None,
                     quality_changed: Optional[callable] = None):
        """Set callback functions"""
        
        self.chunk_received_callback = chunk_received
        self.chunk_sent_callback = chunk_sent
        self.quality_changed_callback = quality_changed

class AdaptiveStreamingManager:
    """Manages adaptive streaming based on real-time conditions"""
    
    def __init__(self, protocol: AudioStreamingProtocol):
        self.protocol = protocol
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Monitoring parameters
        self.monitor_interval = 1.0  # seconds
        self.bandwidth_samples = deque(maxlen=10)
        self.latency_samples = deque(maxlen=10)
        self.loss_samples = deque(maxlen=10)
    
    def start_monitoring(self):
        """Start adaptive monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Adaptive streaming monitoring started")
    
    def stop_monitoring(self):
        """Stop adaptive monitoring"""
        
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Adaptive streaming monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect network metrics
                self._collect_network_metrics()
                
                # Update protocol conditions
                conditions = self._calculate_network_conditions()
                self.protocol.update_network_conditions(conditions)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)
    
    def _collect_network_metrics(self):
        """Collect network performance metrics"""
        
        # Placeholder for actual network monitoring
        # In production, implement actual bandwidth/latency measurement
        
        # Simulate network conditions
        import random
        
        bandwidth = 500 + random.uniform(-100, 200)  # 400-700 kbps
        latency = 50 + random.uniform(-20, 50)       # 30-100 ms
        loss_rate = random.uniform(0, 0.05)          # 0-5% loss
        
        self.bandwidth_samples.append(bandwidth)
        self.latency_samples.append(latency)
        self.loss_samples.append(loss_rate)
    
    def _calculate_network_conditions(self) -> Dict[str, float]:
        """Calculate average network conditions"""
        
        conditions = {}
        
        if self.bandwidth_samples:
            conditions["bandwidth_kbps"] = np.mean(self.bandwidth_samples)
        
        if self.latency_samples:
            conditions["latency_ms"] = np.mean(self.latency_samples)
            conditions["jitter_ms"] = np.std(self.latency_samples)
        
        if self.loss_samples:
            conditions["packet_loss_rate"] = np.mean(self.loss_samples)
        
        return conditions

# Global streaming protocol instance
streaming_protocol = None

def get_streaming_protocol(config: AudioConfig = None) -> AudioStreamingProtocol:
    """Get the global streaming protocol instance"""
    global streaming_protocol
    if streaming_protocol is None:
        streaming_protocol = AudioStreamingProtocol(config or AudioConfig())
    return streaming_protocol

def create_streaming_interface_components(protocol: AudioStreamingProtocol) -> Dict[str, Any]:
    """Create interface components for audio streaming"""
    
    def start_streaming():
        """Start audio streaming"""
        protocol.start_streaming()
        return "✅ Audio streaming started"
    
    def stop_streaming():
        """Stop audio streaming"""
        protocol.stop_streaming()
        return "✅ Audio streaming stopped"
    
    def get_streaming_stats():
        """Get streaming statistics"""
        return protocol.get_streaming_stats()
    
    def update_quality(quality_level):
        """Update quality level"""
        try:
            quality_enum = QualityLevel(quality_level)
            protocol.quality_controller.current_quality = quality_enum
            return f"✅ Quality set to {quality_level}"
        except ValueError:
            return f"❌ Invalid quality level: {quality_level}"
    
    def simulate_network_conditions(bandwidth, latency, packet_loss):
        """Simulate network conditions for testing"""
        conditions = {
            "bandwidth_kbps": float(bandwidth),
            "latency_ms": float(latency),
            "packet_loss_rate": float(packet_loss) / 100  # Convert percentage
        }
        protocol.update_network_conditions(conditions)
        return f"✅ Network conditions updated: {bandwidth} kbps, {latency} ms, {packet_loss}% loss"
    
    # Available options
    quality_choices = [q.value for q in QualityLevel]
    format_choices = [f.value for f in AudioFormat]
    mode_choices = [m.value for m in StreamingMode]
    
    return {
        "start_streaming": start_streaming,
        "stop_streaming": stop_streaming,
        "get_stats": get_streaming_stats,
        "update_quality": update_quality,
        "simulate_network": simulate_network_conditions,
        "quality_choices": quality_choices,
        "format_choices": format_choices,
        "mode_choices": mode_choices,
        "protocol": protocol
    }