#!/usr/bin/env python3
"""
Enhanced FireRedTTS2 Integration
Streaming support, WebSocket integration, and performance optimizations
"""

import os
import time
import json
import asyncio
import logging
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Iterator, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from fireredtts2.fireredtts2 import FireRedTTS2
from performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    """Configuration for streaming TTS"""
    chunk_size_ms: int = 500  # Audio chunk size in milliseconds
    buffer_size: int = 5  # Number of chunks to buffer
    websocket_port: int = 8766  # WebSocket port for streaming
    enable_streaming: bool = True
    quality_mode: str = "balanced"  # "fast", "balanced", "quality"
    sample_rate: int = 24000
    enable_voice_cloning: bool = True
    max_concurrent_streams: int = 10

@dataclass
class GenerationRequest:
    """TTS generation request"""
    request_id: str
    text: str
    voice_mode: str = "random"  # "random", "clone", "multi-speaker"
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None
    speaker_profiles: Optional[Dict[str, str]] = None
    temperature: float = 0.9
    top_k: int = 30
    streaming: bool = False
    websocket: Optional[WebSocketServerProtocol] = None
    metadata: Dict[str, Any] = None

@dataclass
class GenerationResult:
    """TTS generation result"""
    request_id: str
    success: bool
    audio_data: Optional[torch.Tensor] = None
    sample_rate: int = 24000
    generation_time_ms: float = 0.0
    audio_duration_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class StreamingAudioBuffer:
    """Buffer for streaming audio chunks"""
    
    def __init__(self, chunk_size_ms: int = 500, sample_rate: int = 24000):
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate
        self.chunk_size_samples = int(chunk_size_ms * sample_rate / 1000)
        
        self.buffer = []
        self.total_samples = 0
        self.is_complete = False
        
    def add_audio(self, audio_tensor: torch.Tensor):
        """Add audio to buffer"""
        if isinstance(audio_tensor, torch.Tensor):
            audio_array = audio_tensor.cpu().numpy()
        else:
            audio_array = audio_tensor
        
        # Ensure 1D array
        if len(audio_array.shape) > 1:
            audio_array = audio_array.squeeze()
        
        self.buffer.extend(audio_array)
        self.total_samples = len(self.buffer)
    
    def get_chunk(self) -> Optional[np.ndarray]:
        """Get next audio chunk"""
        if len(self.buffer) >= self.chunk_size_samples:
            chunk = np.array(self.buffer[:self.chunk_size_samples])
            self.buffer = self.buffer[self.chunk_size_samples:]
            return chunk
        elif self.is_complete and self.buffer:
            # Return remaining audio as final chunk
            chunk = np.array(self.buffer)
            self.buffer = []
            return chunk
        else:
            return None
    
    def has_chunks(self) -> bool:
        """Check if buffer has available chunks"""
        return len(self.buffer) >= self.chunk_size_samples or (self.is_complete and self.buffer)
    
    def mark_complete(self):
        """Mark audio generation as complete"""
        self.is_complete = True
    
    def get_remaining_audio(self) -> Optional[np.ndarray]:
        """Get all remaining audio"""
        if self.buffer:
            remaining = np.array(self.buffer)
            self.buffer = []
            return remaining
        return None

class EnhancedFireRedTTS2:
    """Enhanced FireRedTTS2 with streaming and WebSocket support"""
    
    def __init__(self, 
                 pretrained_dir: str,
                 device: str = "cuda",
                 config: StreamingConfig = None):
        
        self.pretrained_dir = pretrained_dir
        self.device = device
        self.config = config or StreamingConfig()
        
        # Initialize base model
        self.model = None
        self.is_loaded = False
        
        # Streaming components
        self.active_streams: Dict[str, StreamingAudioBuffer] = {}
        self.generation_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # WebSocket server
        self.websocket_server = None
        self.websocket_clients: Dict[str, WebSocketServerProtocol] = {}
        
        # Performance monitoring
        self.performance_monitor = get_performance_monitor()
        self.generation_stats = {
            "total_generations": 0,
            "total_generation_time": 0.0,
            "average_generation_time": 0.0,
            "streaming_sessions": 0,
            "concurrent_streams": 0
        }
        
        # Worker threads
        self.generation_worker = None
        self.streaming_worker = None
        self.is_running = False
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the FireRedTTS2 model"""
        try:
            logger.info("Initializing Enhanced FireRedTTS2...")
            start_time = time.time()
            
            # Initialize for dialogue generation (supports both monologue and dialogue)
            self.model = FireRedTTS2(
                pretrained_dir=self.pretrained_dir,
                gen_type="dialogue",
                device=self.device
            )
            
            load_time = time.time() - start_time
            logger.info(f"Enhanced FireRedTTS2 loaded in {load_time:.2f}s")
            
            self.is_loaded = True
            
            # Start worker threads
            self._start_workers()
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced FireRedTTS2: {e}")
            self.is_loaded = False
            raise
    
    def _start_workers(self):
        """Start background worker threads"""
        self.is_running = True
        
        # Generation worker
        self.generation_worker = threading.Thread(
            target=self._generation_worker_loop, 
            daemon=True
        )
        self.generation_worker.start()
        
        # Streaming worker
        self.streaming_worker = threading.Thread(
            target=self._streaming_worker_loop,
            daemon=True
        )
        self.streaming_worker.start()
        
        logger.info("Enhanced FireRedTTS2 workers started")
    
    def _stop_workers(self):
        """Stop background worker threads"""
        self.is_running = False
        
        if self.generation_worker:
            self.generation_worker.join(timeout=2.0)
        
        if self.streaming_worker:
            self.streaming_worker.join(timeout=2.0)
        
        logger.info("Enhanced FireRedTTS2 workers stopped")
    
    def generate_speech(self, request: GenerationRequest) -> GenerationResult:
        """Generate speech from text (non-streaming)"""
        
        if not self.is_loaded:
            return GenerationResult(
                request_id=request.request_id,
                success=False,
                error_message="Model not loaded"
            )
        
        start_time = time.time()
        
        try:
            # Track performance
            if self.performance_monitor:
                self.performance_monitor.model_tracker.start_inference(
                    model_name="FireRedTTS2",
                    batch_size=1,
                    sequence_length=len(request.text),
                    temperature=request.temperature,
                    top_k=request.top_k
                )
            
            # Generate audio based on voice mode
            if request.voice_mode == "clone" and request.reference_audio_path:
                audio_tensor = self.model.generate_monologue(
                    text=request.text,
                    prompt_wav=request.reference_audio_path,
                    prompt_text=request.reference_text or request.text[:50],
                    temperature=request.temperature,
                    topk=request.top_k
                )
            elif request.voice_mode == "multi-speaker" and request.speaker_profiles:
                # Multi-speaker dialogue
                audio_tensor = self._generate_multi_speaker_dialogue(request)
            else:
                # Random voice
                audio_tensor = self.model.generate_monologue(
                    text=request.text,
                    temperature=request.temperature,
                    topk=request.top_k
                )
            
            generation_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.generation_stats["total_generations"] += 1
            self.generation_stats["total_generation_time"] += generation_time
            self.generation_stats["average_generation_time"] = (
                self.generation_stats["total_generation_time"] / 
                self.generation_stats["total_generations"]
            )
            
            # Calculate audio duration
            if isinstance(audio_tensor, torch.Tensor):
                audio_duration = (audio_tensor.shape[-1] / self.config.sample_rate) * 1000
            else:
                audio_duration = 0.0
            
            # Track performance end
            if self.performance_monitor:
                tokens_generated = len(request.text.split())  # Approximate
                self.performance_monitor.model_tracker.end_inference(tokens_generated)
            
            return GenerationResult(
                request_id=request.request_id,
                success=True,
                audio_data=audio_tensor,
                sample_rate=self.config.sample_rate,
                generation_time_ms=generation_time,
                audio_duration_ms=audio_duration,
                metadata={
                    "voice_mode": request.voice_mode,
                    "text_length": len(request.text),
                    "model_device": self.device
                }
            )
            
        except Exception as e:
            logger.error(f"Speech generation failed for request {request.request_id}: {e}")
            
            generation_time = (time.time() - start_time) * 1000
            
            return GenerationResult(
                request_id=request.request_id,
                success=False,
                generation_time_ms=generation_time,
                error_message=str(e)
            )
    
    def generate_speech_streaming(self, request: GenerationRequest) -> str:
        """Start streaming speech generation"""
        
        if not self.is_loaded:
            return None
        
        # Add to generation queue
        request.streaming = True
        self.generation_queue.put(request)
        
        # Create streaming buffer
        buffer = StreamingAudioBuffer(
            chunk_size_ms=self.config.chunk_size_ms,
            sample_rate=self.config.sample_rate
        )
        self.active_streams[request.request_id] = buffer
        
        logger.info(f"Started streaming generation for request {request.request_id}")
        return request.request_id
    
    def get_audio_chunk(self, request_id: str) -> Optional[Tuple[np.ndarray, bool]]:
        """Get next audio chunk from streaming generation"""
        
        buffer = self.active_streams.get(request_id)
        if not buffer:
            return None
        
        chunk = buffer.get_chunk()
        is_complete = buffer.is_complete and not buffer.has_chunks()
        
        # Clean up completed streams
        if is_complete:
            del self.active_streams[request_id]
        
        return (chunk, is_complete) if chunk is not None else None
    
    def _generate_multi_speaker_dialogue(self, request: GenerationRequest) -> torch.Tensor:
        """Generate multi-speaker dialogue"""
        
        # Parse dialogue text into speaker turns
        import re
        turns = re.findall(r'(\[S\d+\][^[]*)', request.text)
        
        if not turns:
            # Fallback to single speaker
            return self.model.generate_monologue(
                text=request.text,
                temperature=request.temperature,
                topk=request.top_k
            )
        
        # Prepare speaker configurations
        prompt_wav_list = []
        prompt_text_list = []
        
        if request.speaker_profiles:
            for turn in turns:
                speaker_tag = turn[:4]  # e.g., "[S1]"
                
                if speaker_tag in request.speaker_profiles:
                    profile_path = request.speaker_profiles[speaker_tag]
                    # Assume profile_path contains both audio path and text
                    # Format: "audio_path|reference_text"
                    if "|" in profile_path:
                        audio_path, ref_text = profile_path.split("|", 1)
                        prompt_wav_list.append(audio_path)
                        prompt_text_list.append(ref_text)
                    else:
                        prompt_wav_list.append(None)
                        prompt_text_list.append(None)
                else:
                    prompt_wav_list.append(None)
                    prompt_text_list.append(None)
        
        # Generate dialogue
        if any(prompt_wav_list):
            audio_tensor = self.model.generate_dialogue(
                text_list=turns,
                prompt_wav_list=prompt_wav_list,
                prompt_text_list=prompt_text_list,
                temperature=request.temperature,
                topk=request.top_k
            )
        else:
            audio_tensor = self.model.generate_dialogue(
                text_list=turns,
                temperature=request.temperature,
                topk=request.top_k
            )
        
        return audio_tensor
    
    def _generation_worker_loop(self):
        """Background worker for processing generation requests"""
        
        while self.is_running:
            try:
                # Get request from queue
                try:
                    request = self.generation_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Generate audio
                result = self.generate_speech(request)
                
                if result.success and request.streaming:
                    # Add to streaming buffer
                    buffer = self.active_streams.get(request.request_id)
                    if buffer:
                        buffer.add_audio(result.audio_data)
                        buffer.mark_complete()
                        
                        # Send WebSocket notification if available
                        if request.websocket:
                            asyncio.create_task(
                                self._send_websocket_message(
                                    request.websocket,
                                    {
                                        "type": "generation_complete",
                                        "request_id": request.request_id,
                                        "audio_duration_ms": result.audio_duration_ms
                                    }
                                )
                            )
                
                # Put result in result queue
                self.result_queue.put(result)
                
            except Exception as e:
                logger.error(f"Generation worker error: {e}")
                time.sleep(1.0)
    
    def _streaming_worker_loop(self):
        """Background worker for streaming audio chunks"""
        
        while self.is_running:
            try:
                # Process active streams
                for request_id, buffer in list(self.active_streams.items()):
                    if buffer.has_chunks():
                        chunk = buffer.get_chunk()
                        if chunk is not None:
                            # Send chunk via WebSocket if client is connected
                            client_ws = self.websocket_clients.get(request_id)
                            if client_ws:
                                asyncio.create_task(
                                    self._send_audio_chunk(client_ws, request_id, chunk)
                                )
                
                time.sleep(0.1)  # 100ms polling interval
                
            except Exception as e:
                logger.error(f"Streaming worker error: {e}")
                time.sleep(1.0)
    
    async def start_websocket_server(self, host: str = "0.0.0.0", port: int = None):
        """Start WebSocket server for streaming"""
        
        port = port or self.config.websocket_port
        
        logger.info(f"Starting TTS WebSocket server on {host}:{port}")
        
        self.websocket_server = await websockets.serve(
            self.handle_websocket_connection,
            host,
            port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"TTS WebSocket server started on ws://{host}:{port}")
    
    async def stop_websocket_server(self):
        """Stop WebSocket server"""
        
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Close all client connections
        for client_ws in list(self.websocket_clients.values()):
            await client_ws.close()
        
        self.websocket_clients.clear()
        logger.info("TTS WebSocket server stopped")
    
    async def handle_websocket_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket connection for streaming TTS"""
        
        client_id = f"client_{len(self.websocket_clients)}"
        self.websocket_clients[client_id] = websocket
        
        logger.info(f"New TTS WebSocket client: {client_id}")
        
        try:
            # Send welcome message
            await self._send_websocket_message(websocket, {
                "type": "connection_established",
                "client_id": client_id,
                "server_info": {
                    "model_loaded": self.is_loaded,
                    "streaming_enabled": self.config.enable_streaming,
                    "sample_rate": self.config.sample_rate,
                    "chunk_size_ms": self.config.chunk_size_ms
                }
            })
            
            # Handle messages
            async for message in websocket:
                await self.handle_websocket_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"TTS WebSocket client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"TTS WebSocket error for {client_id}: {e}")
        finally:
            # Clean up
            if client_id in self.websocket_clients:
                del self.websocket_clients[client_id]
    
    async def handle_websocket_message(self, client_id: str, message):
        """Handle WebSocket message"""
        
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "generate_speech":
                await self._handle_generate_speech_request(client_id, data)
            elif message_type == "get_status":
                await self._handle_status_request(client_id)
            else:
                await self._send_websocket_error(
                    self.websocket_clients[client_id],
                    f"Unknown message type: {message_type}"
                )
                
        except json.JSONDecodeError:
            await self._send_websocket_error(
                self.websocket_clients[client_id],
                "Invalid JSON message"
            )
        except Exception as e:
            logger.error(f"WebSocket message handling error: {e}")
            await self._send_websocket_error(
                self.websocket_clients[client_id],
                f"Message processing failed: {e}"
            )
    
    async def _handle_generate_speech_request(self, client_id: str, data: Dict[str, Any]):
        """Handle speech generation request via WebSocket"""
        
        try:
            request = GenerationRequest(
                request_id=data.get("request_id", str(time.time())),
                text=data.get("text", ""),
                voice_mode=data.get("voice_mode", "random"),
                reference_audio_path=data.get("reference_audio_path"),
                reference_text=data.get("reference_text"),
                speaker_profiles=data.get("speaker_profiles"),
                temperature=data.get("temperature", 0.9),
                top_k=data.get("top_k", 30),
                streaming=data.get("streaming", False),
                websocket=self.websocket_clients[client_id],
                metadata={"client_id": client_id}
            )
            
            if request.streaming:
                # Start streaming generation
                stream_id = self.generate_speech_streaming(request)
                
                await self._send_websocket_message(
                    self.websocket_clients[client_id],
                    {
                        "type": "streaming_started",
                        "request_id": request.request_id,
                        "stream_id": stream_id
                    }
                )
            else:
                # Non-streaming generation
                result = self.generate_speech(request)
                
                if result.success:
                    # Convert audio to base64 for transmission
                    audio_array = result.audio_data.cpu().numpy()
                    audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
                    
                    await self._send_websocket_message(
                        self.websocket_clients[client_id],
                        {
                            "type": "generation_complete",
                            "request_id": request.request_id,
                            "success": True,
                            "audio_data": audio_bytes.hex(),  # Send as hex string
                            "sample_rate": result.sample_rate,
                            "generation_time_ms": result.generation_time_ms,
                            "audio_duration_ms": result.audio_duration_ms
                        }
                    )
                else:
                    await self._send_websocket_message(
                        self.websocket_clients[client_id],
                        {
                            "type": "generation_complete",
                            "request_id": request.request_id,
                            "success": False,
                            "error": result.error_message
                        }
                    )
            
        except Exception as e:
            logger.error(f"Generate speech request handling failed: {e}")
            await self._send_websocket_error(
                self.websocket_clients[client_id],
                f"Generation request failed: {e}"
            )
    
    async def _handle_status_request(self, client_id: str):
        """Handle status request"""
        
        status = {
            "type": "status_response",
            "model_loaded": self.is_loaded,
            "device": self.device,
            "active_streams": len(self.active_streams),
            "generation_stats": self.generation_stats,
            "websocket_clients": len(self.websocket_clients)
        }
        
        await self._send_websocket_message(self.websocket_clients[client_id], status)
    
    async def _send_audio_chunk(self, websocket: WebSocketServerProtocol, 
                               request_id: str, audio_chunk: np.ndarray):
        """Send audio chunk via WebSocket"""
        
        try:
            # Convert to bytes
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            
            message = {
                "type": "audio_chunk",
                "request_id": request_id,
                "audio_data": audio_bytes.hex(),
                "chunk_size": len(audio_chunk),
                "timestamp": time.time()
            }
            
            await self._send_websocket_message(websocket, message)
            
        except Exception as e:
            logger.error(f"Failed to send audio chunk: {e}")
    
    async def _send_websocket_message(self, websocket: WebSocketServerProtocol, 
                                    message: Dict[str, Any]):
        """Send JSON message via WebSocket"""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
    
    async def _send_websocket_error(self, websocket: WebSocketServerProtocol, 
                                  error_message: str):
        """Send error message via WebSocket"""
        await self._send_websocket_message(websocket, {
            "type": "error",
            "message": error_message,
            "timestamp": time.time()
        })
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            **self.generation_stats,
            "model_loaded": self.is_loaded,
            "device": self.device,
            "active_streams": len(self.active_streams),
            "websocket_clients": len(self.websocket_clients),
            "queue_size": self.generation_queue.qsize(),
            "config": {
                "chunk_size_ms": self.config.chunk_size_ms,
                "sample_rate": self.config.sample_rate,
                "streaming_enabled": self.config.enable_streaming
            }
        }
    
    def cleanup(self):
        """Clean up resources"""
        self._stop_workers()
        
        # Clear active streams
        self.active_streams.clear()
        
        # Clear queues
        while not self.generation_queue.empty():
            try:
                self.generation_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Enhanced FireRedTTS2 cleanup completed")

# Global enhanced TTS instance
enhanced_tts = None

def get_enhanced_tts(pretrained_dir: str = None, device: str = "cuda") -> EnhancedFireRedTTS2:
    """Get the global enhanced TTS instance"""
    global enhanced_tts
    if enhanced_tts is None and pretrained_dir:
        enhanced_tts = EnhancedFireRedTTS2(pretrained_dir, device)
    return enhanced_tts

def create_tts_interface_components(enhanced_tts: EnhancedFireRedTTS2) -> Dict[str, Any]:
    """Create interface components for enhanced TTS"""
    
    def generate_speech_sync(text, voice_mode="random", reference_audio=None, reference_text=""):
        """Synchronous speech generation for Gradio interface"""
        
        if not text.strip():
            return None, "❌ Text is required"
        
        request = GenerationRequest(
            request_id=str(time.time()),
            text=text.strip(),
            voice_mode=voice_mode,
            reference_audio_path=reference_audio,
            reference_text=reference_text
        )
        
        result = enhanced_tts.generate_speech(request)
        
        if result.success:
            # Convert to Gradio audio format
            audio_array = result.audio_data.cpu().numpy()
            if len(audio_array.shape) > 1:
                audio_array = audio_array.squeeze()
            
            return (
                (result.sample_rate, audio_array),
                f"✅ Generated in {result.generation_time_ms:.0f}ms"
            )
        else:
            return None, f"❌ Generation failed: {result.error_message}"
    
    def get_tts_stats():
        """Get TTS statistics for display"""
        return enhanced_tts.get_generation_stats()
    
    return {
        "generate_speech": generate_speech_sync,
        "get_stats": get_tts_stats,
        "enhanced_tts": enhanced_tts
    }