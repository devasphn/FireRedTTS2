#!/usr/bin/env python3
"""
Real-time ASR Streaming Pipeline
WebSocket-based real-time speech recognition with VAD and streaming optimization
"""

import asyncio
import json
import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque

import numpy as np
import torch
import websockets
from websockets.server import WebSocketServerProtocol

from whisper_asr import WhisperASR, StreamingWhisperASR, ASRConfig, TranscriptionResult
from conversation_manager import VoiceActivityDetector

logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    """Configuration for real-time streaming"""
    chunk_duration_ms: int = 1000  # Audio chunk duration in milliseconds
    vad_threshold: float = 0.01  # Voice activity detection threshold
    silence_timeout_ms: int = 2000  # Silence timeout for end-of-speech
    max_audio_length_ms: int = 30000  # Maximum audio length for single transcription
    sample_rate: int = 16000  # Target sample rate
    channels: int = 1  # Mono audio
    websocket_port: int = 8765  # WebSocket server port
    enable_partial_results: bool = True  # Send partial transcription results
    enable_vad: bool = True  # Enable voice activity detection
    buffer_size: int = 10  # Audio buffer size in chunks

@dataclass
class AudioChunk:
    """Audio chunk with metadata"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    chunk_id: int
    is_speech: bool = False
    energy: float = 0.0

@dataclass
class StreamingSession:
    """Streaming session state"""
    session_id: str
    websocket: WebSocketServerProtocol
    audio_buffer: deque
    vad: VoiceActivityDetector
    is_active: bool
    language: Optional[str]
    created_at: datetime
    last_activity: datetime
    total_chunks: int
    total_transcriptions: int

class AudioProcessor:
    """Processes incoming audio chunks"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.chunk_size = int(config.chunk_duration_ms * config.sample_rate / 1000)
    
    def process_audio_data(self, audio_data: bytes, sample_rate: int = None) -> Optional[AudioChunk]:
        """Process raw audio data into AudioChunk"""
        
        try:
            # Convert bytes to numpy array
            if isinstance(audio_data, bytes):
                # Assume 16-bit PCM audio
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
            else:
                audio_array = np.array(audio_data, dtype=np.float32)
            
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample if needed
            if sample_rate and sample_rate != self.config.sample_rate:
                # Simple resampling (for production, use librosa or torchaudio)
                duration = len(audio_array) / sample_rate
                target_length = int(duration * self.config.sample_rate)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), target_length),
                    np.arange(len(audio_array)),
                    audio_array
                )
            
            # Calculate energy
            energy = np.sqrt(np.mean(audio_array ** 2))
            
            return AudioChunk(
                data=audio_array,
                timestamp=time.time(),
                sample_rate=self.config.sample_rate,
                chunk_id=0,  # Will be set by caller
                energy=energy
            )
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return None
    
    def combine_chunks(self, chunks: List[AudioChunk]) -> np.ndarray:
        """Combine multiple audio chunks into single array"""
        if not chunks:
            return np.array([], dtype=np.float32)
        
        combined = np.concatenate([chunk.data for chunk in chunks])
        return combined
    
    def detect_speech_boundaries(self, chunks: List[AudioChunk], vad: VoiceActivityDetector) -> Dict[str, Any]:
        """Detect speech start and end boundaries"""
        
        if not chunks:
            return {"has_speech": False, "start_idx": 0, "end_idx": 0}
        
        speech_chunks = []
        for i, chunk in enumerate(chunks):
            vad_result = vad.detect_activity(chunk.data, chunk.sample_rate)
            chunk.is_speech = vad_result["activity_detected"]
            speech_chunks.append(vad_result["activity_detected"])
        
        # Find speech boundaries
        has_speech = any(speech_chunks)
        start_idx = 0
        end_idx = len(chunks)
        
        if has_speech:
            # Find first speech chunk
            for i, is_speech in enumerate(speech_chunks):
                if is_speech:
                    start_idx = i
                    break
            
            # Find last speech chunk
            for i in range(len(speech_chunks) - 1, -1, -1):
                if speech_chunks[i]:
                    end_idx = i + 1
                    break
        
        return {
            "has_speech": has_speech,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "speech_ratio": sum(speech_chunks) / len(speech_chunks) if speech_chunks else 0.0
        }

class RealtimeASRPipeline:
    """Real-time ASR pipeline with WebSocket support"""
    
    def __init__(self, 
                 config: StreamingConfig = None,
                 asr_config: ASRConfig = None,
                 device: str = "cuda"):
        
        self.config = config or StreamingConfig()
        self.device = device
        
        # Initialize ASR
        self.asr = WhisperASR(asr_config or ASRConfig(), device)
        self.streaming_asr = StreamingWhisperASR(asr_config or ASRConfig(), device)
        
        # Audio processing
        self.audio_processor = AudioProcessor(self.config)
        
        # Session management
        self.active_sessions: Dict[str, StreamingSession] = {}
        self.session_counter = 0
        
        # WebSocket server
        self.websocket_server = None
        self.is_running = False
        
        # Performance tracking
        self.total_sessions = 0
        self.total_audio_processed = 0.0  # seconds
        self.total_transcriptions = 0
    
    async def start_server(self, host: str = "0.0.0.0", port: int = None):
        """Start the WebSocket server"""
        
        port = port or self.config.websocket_port
        
        logger.info(f"Starting real-time ASR server on {host}:{port}")
        
        self.is_running = True
        
        # Start WebSocket server
        self.websocket_server = await websockets.serve(
            self.handle_websocket_connection,
            host,
            port,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10
        )
        
        logger.info(f"Real-time ASR server started on ws://{host}:{port}")
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        
        self.is_running = False
        
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Close all active sessions
        for session in list(self.active_sessions.values()):
            await self.close_session(session.session_id)
        
        logger.info("Real-time ASR server stopped")
    
    async def handle_websocket_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        
        session_id = f"session_{self.session_counter}"
        self.session_counter += 1
        
        logger.info(f"New WebSocket connection: {session_id}")
        
        try:
            # Create session
            session = StreamingSession(
                session_id=session_id,
                websocket=websocket,
                audio_buffer=deque(maxlen=self.config.buffer_size),
                vad=VoiceActivityDetector(),
                is_active=True,
                language=None,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                total_chunks=0,
                total_transcriptions=0
            )
            
            self.active_sessions[session_id] = session
            self.total_sessions += 1
            
            # Send welcome message
            await self.send_message(websocket, {
                "type": "session_started",
                "session_id": session_id,
                "config": {
                    "sample_rate": self.config.sample_rate,
                    "chunk_duration_ms": self.config.chunk_duration_ms,
                    "vad_enabled": self.config.enable_vad
                }
            })
            
            # Handle messages
            async for message in websocket:
                await self.handle_message(session_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {session_id}: {e}")
        finally:
            # Clean up session
            await self.close_session(session_id)
    
    async def handle_message(self, session_id: str, message):
        """Handle incoming WebSocket message"""
        
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        try:
            if isinstance(message, bytes):
                # Audio data
                await self.handle_audio_data(session_id, message)
            else:
                # JSON message
                data = json.loads(message)
                await self.handle_control_message(session_id, data)
                
        except Exception as e:
            logger.error(f"Message handling error for {session_id}: {e}")
            await self.send_error(session.websocket, f"Message processing failed: {e}")
    
    async def handle_audio_data(self, session_id: str, audio_data: bytes):
        """Handle incoming audio data"""
        
        session = self.active_sessions.get(session_id)
        if not session or not session.is_active:
            return
        
        # Process audio
        chunk = self.audio_processor.process_audio_data(audio_data)
        if not chunk:
            return
        
        # Set chunk ID
        chunk.chunk_id = session.total_chunks
        session.total_chunks += 1
        session.last_activity = datetime.now()
        
        # Add to buffer
        session.audio_buffer.append(chunk)
        
        # Voice activity detection
        if self.config.enable_vad:
            vad_result = session.vad.detect_activity(chunk.data, chunk.sample_rate)
            chunk.is_speech = vad_result["activity_detected"]
            
            # Send VAD result
            await self.send_message(session.websocket, {
                "type": "vad_result",
                "chunk_id": chunk.chunk_id,
                "is_speech": chunk.is_speech,
                "energy": float(chunk.energy),
                "turn_complete": vad_result.get("turn_complete", False)
            })
            
            # If turn is complete, transcribe accumulated audio
            if vad_result.get("turn_complete", False):
                await self.transcribe_buffered_audio(session_id)
        
        # Check if buffer is full (fallback transcription)
        elif len(session.audio_buffer) >= self.config.buffer_size:
            await self.transcribe_buffered_audio(session_id)
    
    async def handle_control_message(self, session_id: str, data: Dict[str, Any]):
        """Handle control messages"""
        
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        message_type = data.get("type")
        
        if message_type == "set_language":
            session.language = data.get("language")
            await self.send_message(session.websocket, {
                "type": "language_set",
                "language": session.language
            })
        
        elif message_type == "transcribe_now":
            await self.transcribe_buffered_audio(session_id)
        
        elif message_type == "clear_buffer":
            session.audio_buffer.clear()
            await self.send_message(session.websocket, {
                "type": "buffer_cleared"
            })
        
        elif message_type == "get_status":
            await self.send_session_status(session_id)
        
        else:
            await self.send_error(session.websocket, f"Unknown message type: {message_type}")
    
    async def transcribe_buffered_audio(self, session_id: str):
        """Transcribe accumulated audio in buffer"""
        
        session = self.active_sessions.get(session_id)
        if not session or not session.audio_buffer:
            return
        
        try:
            # Get audio chunks
            chunks = list(session.audio_buffer)
            
            # Detect speech boundaries
            if self.config.enable_vad:
                boundaries = self.audio_processor.detect_speech_boundaries(chunks, session.vad)
                
                if not boundaries["has_speech"]:
                    # No speech detected, clear buffer
                    session.audio_buffer.clear()
                    return
                
                # Use only speech portions
                speech_chunks = chunks[boundaries["start_idx"]:boundaries["end_idx"]]
            else:
                speech_chunks = chunks
            
            if not speech_chunks:
                return
            
            # Combine audio chunks
            combined_audio = self.audio_processor.combine_chunks(speech_chunks)
            
            # Skip if audio is too short
            min_duration = 0.5  # 500ms minimum
            if len(combined_audio) < min_duration * self.config.sample_rate:
                return
            
            # Transcribe
            start_time = time.time()
            result = self.asr.transcribe(
                combined_audio, 
                self.config.sample_rate, 
                session.language
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            session.total_transcriptions += 1
            self.total_transcriptions += 1
            self.total_audio_processed += len(combined_audio) / self.config.sample_rate
            
            # Send result
            if result.text.strip():
                await self.send_message(session.websocket, {
                    "type": "transcription_result",
                    "text": result.text,
                    "confidence": result.confidence,
                    "language": result.language,
                    "processing_time_ms": processing_time,
                    "audio_duration_ms": len(combined_audio) / self.config.sample_rate * 1000,
                    "chunks_used": len(speech_chunks),
                    "is_final": True
                })
            
            # Clear processed audio from buffer
            session.audio_buffer.clear()
            
        except Exception as e:
            logger.error(f"Transcription failed for {session_id}: {e}")
            await self.send_error(session.websocket, f"Transcription failed: {e}")
    
    async def send_message(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]):
        """Send JSON message to WebSocket client"""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def send_error(self, websocket: WebSocketServerProtocol, error_message: str):
        """Send error message to client"""
        await self.send_message(websocket, {
            "type": "error",
            "message": error_message,
            "timestamp": time.time()
        })
    
    async def send_session_status(self, session_id: str):
        """Send session status to client"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        status = {
            "type": "session_status",
            "session_id": session_id,
            "is_active": session.is_active,
            "language": session.language,
            "buffer_size": len(session.audio_buffer),
            "total_chunks": session.total_chunks,
            "total_transcriptions": session.total_transcriptions,
            "uptime_seconds": (datetime.now() - session.created_at).total_seconds(),
            "last_activity": session.last_activity.isoformat()
        }
        
        await self.send_message(session.websocket, status)
    
    async def close_session(self, session_id: str):
        """Close and clean up session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.is_active = False
            
            try:
                await self.send_message(session.websocket, {
                    "type": "session_ended",
                    "session_id": session_id
                })
            except:
                pass  # Connection might already be closed
            
            del self.active_sessions[session_id]
            logger.info(f"Session closed: {session_id}")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server performance statistics"""
        
        active_sessions = len(self.active_sessions)
        
        return {
            "is_running": self.is_running,
            "active_sessions": active_sessions,
            "total_sessions": self.total_sessions,
            "total_transcriptions": self.total_transcriptions,
            "total_audio_processed_seconds": self.total_audio_processed,
            "average_transcriptions_per_session": (
                self.total_transcriptions / max(self.total_sessions, 1)
            ),
            "websocket_port": self.config.websocket_port,
            "device": self.device,
            "model_name": self.asr.model_name
        }

class WebSocketClient:
    """WebSocket client for testing the real-time ASR pipeline"""
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        self.message_handlers = {}
    
    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            logger.info(f"Connected to {self.server_url}")
            
            # Start message handler
            asyncio.create_task(self.handle_messages())
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.is_connected = False
    
    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from server")
    
    async def send_audio(self, audio_data: np.ndarray):
        """Send audio data to server"""
        if not self.is_connected or not self.websocket:
            return
        
        # Convert to bytes (16-bit PCM)
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        await self.websocket.send(audio_bytes)
    
    async def send_control_message(self, message: Dict[str, Any]):
        """Send control message to server"""
        if not self.is_connected or not self.websocket:
            return
        
        await self.websocket.send(json.dumps(message))
    
    async def handle_messages(self):
        """Handle incoming messages from server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                message_type = data.get("type")
                
                # Call registered handler
                if message_type in self.message_handlers:
                    self.message_handlers[message_type](data)
                else:
                    logger.info(f"Received: {data}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Server connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Message handling error: {e}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler

# Global pipeline instance
realtime_pipeline = None

def get_realtime_pipeline() -> RealtimeASRPipeline:
    """Get the global real-time ASR pipeline instance"""
    global realtime_pipeline
    if realtime_pipeline is None:
        realtime_pipeline = RealtimeASRPipeline()
    return realtime_pipeline

async def main():
    """Main function for testing"""
    
    # Create and start pipeline
    pipeline = RealtimeASRPipeline()
    
    try:
        await pipeline.start_server()
        
        # Keep server running
        while pipeline.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await pipeline.stop_server()

if __name__ == "__main__":
    asyncio.run(main())