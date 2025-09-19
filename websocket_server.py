#!/usr/bin/env python3
"""
WebSocket Communication Server
Comprehensive WebSocket server for real-time audio streaming and communication
"""

import asyncio
import json
import time
import logging
import uuid
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException
import numpy as np

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types"""
    # Connection management
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_CLOSED = "connection_closed"
    HEARTBEAT = "heartbeat"
    
    # Audio streaming
    AUDIO_CHUNK = "audio_chunk"
    AUDIO_START = "audio_start"
    AUDIO_END = "audio_end"
    AUDIO_CONFIG = "audio_config"
    
    # Speech recognition
    TRANSCRIPTION_RESULT = "transcription_result"
    TRANSCRIPTION_PARTIAL = "transcription_partial"
    
    # TTS generation
    TTS_REQUEST = "tts_request"
    TTS_RESPONSE = "tts_response"
    TTS_STREAMING_START = "tts_streaming_start"
    TTS_STREAMING_CHUNK = "tts_streaming_chunk"
    TTS_STREAMING_END = "tts_streaming_end"
    
    # Voice activity detection
    VAD_RESULT = "vad_result"
    VAD_CONFIG = "vad_config"
    
    # Session management
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_STATUS = "session_status"
    
    # Error handling
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ClientInfo:
    """Information about connected client"""
    client_id: str
    websocket: WebSocketServerProtocol
    connected_at: datetime
    last_activity: datetime
    client_type: str  # "web", "mobile", "api"
    user_agent: Optional[str]
    ip_address: str
    session_id: Optional[str]
    capabilities: Set[str]  # Supported features
    metadata: Dict[str, Any]

@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: MessageType
    data: Dict[str, Any]
    timestamp: float
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None

@dataclass
class ServerConfig:
    """WebSocket server configuration"""
    host: str = "0.0.0.0"
    port: int = 8765
    max_connections: int = 100
    heartbeat_interval: int = 30  # seconds
    connection_timeout: int = 300  # seconds
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    enable_compression: bool = True
    enable_ssl: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    cors_origins: List[str] = None
    enable_logging: bool = True
    log_level: str = "INFO"

class MessageHandler:
    """Base class for message handlers"""
    
    def __init__(self, server: 'WebSocketServer'):
        self.server = server
    
    async def handle_message(self, client: ClientInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle incoming message and return response if needed"""
        raise NotImplementedError

class AudioStreamHandler(MessageHandler):
    """Handler for audio streaming messages"""
    
    def __init__(self, server: 'WebSocketServer'):
        super().__init__(server)
        self.active_streams: Dict[str, Dict[str, Any]] = {}
    
    async def handle_message(self, client: ClientInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle audio streaming messages"""
        
        if message.type == MessageType.AUDIO_START:
            return await self._handle_audio_start(client, message)
        elif message.type == MessageType.AUDIO_CHUNK:
            return await self._handle_audio_chunk(client, message)
        elif message.type == MessageType.AUDIO_END:
            return await self._handle_audio_end(client, message)
        elif message.type == MessageType.AUDIO_CONFIG:
            return await self._handle_audio_config(client, message)
        
        return None
    
    async def _handle_audio_start(self, client: ClientInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle audio stream start"""
        
        stream_id = message.data.get("stream_id", str(uuid.uuid4()))
        
        # Create stream info
        stream_info = {
            "stream_id": stream_id,
            "client_id": client.client_id,
            "started_at": time.time(),
            "sample_rate": message.data.get("sample_rate", 16000),
            "channels": message.data.get("channels", 1),
            "format": message.data.get("format", "pcm"),
            "chunk_count": 0,
            "total_bytes": 0
        }
        
        self.active_streams[stream_id] = stream_info
        
        logger.info(f"Audio stream started: {stream_id} for client {client.client_id}")
        
        # Notify other handlers about stream start
        await self.server.broadcast_to_handlers("audio_stream_started", {
            "stream_id": stream_id,
            "client_id": client.client_id,
            "config": stream_info
        })
        
        return WebSocketMessage(
            type=MessageType.AUDIO_START,
            data={
                "stream_id": stream_id,
                "status": "started",
                "config": stream_info
            },
            timestamp=time.time()
        )
    
    async def _handle_audio_chunk(self, client: ClientInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle audio chunk"""
        
        stream_id = message.data.get("stream_id")
        audio_data = message.data.get("audio_data")
        
        if not stream_id or not audio_data:
            return WebSocketMessage(
                type=MessageType.ERROR,
                data={"message": "Missing stream_id or audio_data"},
                timestamp=time.time()
            )
        
        if stream_id not in self.active_streams:
            return WebSocketMessage(
                type=MessageType.ERROR,
                data={"message": f"Stream {stream_id} not found"},
                timestamp=time.time()
            )
        
        # Update stream info
        stream_info = self.active_streams[stream_id]
        stream_info["chunk_count"] += 1
        stream_info["total_bytes"] += len(audio_data)
        stream_info["last_chunk_at"] = time.time()
        
        # Process audio data
        try:
            # Convert hex string to bytes if needed
            if isinstance(audio_data, str):
                audio_bytes = bytes.fromhex(audio_data)
            else:
                audio_bytes = audio_data
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Notify handlers about new audio chunk
            await self.server.broadcast_to_handlers("audio_chunk_received", {
                "stream_id": stream_id,
                "client_id": client.client_id,
                "audio_data": audio_array,
                "sample_rate": stream_info["sample_rate"],
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"Failed to process audio chunk: {e}")
            return WebSocketMessage(
                type=MessageType.ERROR,
                data={"message": f"Audio processing failed: {e}"},
                timestamp=time.time()
            )
        
        return None  # No response needed for chunks
    
    async def _handle_audio_end(self, client: ClientInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle audio stream end"""
        
        stream_id = message.data.get("stream_id")
        
        if stream_id not in self.active_streams:
            return WebSocketMessage(
                type=MessageType.ERROR,
                data={"message": f"Stream {stream_id} not found"},
                timestamp=time.time()
            )
        
        stream_info = self.active_streams[stream_id]
        stream_info["ended_at"] = time.time()
        stream_info["duration"] = stream_info["ended_at"] - stream_info["started_at"]
        
        logger.info(f"Audio stream ended: {stream_id}, duration: {stream_info['duration']:.2f}s, chunks: {stream_info['chunk_count']}")
        
        # Notify handlers about stream end
        await self.server.broadcast_to_handlers("audio_stream_ended", {
            "stream_id": stream_id,
            "client_id": client.client_id,
            "stats": stream_info
        })
        
        # Clean up
        del self.active_streams[stream_id]
        
        return WebSocketMessage(
            type=MessageType.AUDIO_END,
            data={
                "stream_id": stream_id,
                "status": "ended",
                "stats": {
                    "duration": stream_info["duration"],
                    "chunks": stream_info["chunk_count"],
                    "total_bytes": stream_info["total_bytes"]
                }
            },
            timestamp=time.time()
        )
    
    async def _handle_audio_config(self, client: ClientInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle audio configuration"""
        
        # Store client audio capabilities
        client.capabilities.update(message.data.get("capabilities", []))
        
        return WebSocketMessage(
            type=MessageType.AUDIO_CONFIG,
            data={
                "status": "configured",
                "supported_formats": ["pcm", "opus", "mp3"],
                "supported_sample_rates": [8000, 16000, 22050, 44100, 48000],
                "max_chunk_size": 8192
            },
            timestamp=time.time()
        )

class SessionHandler(MessageHandler):
    """Handler for session management messages"""
    
    def __init__(self, server: 'WebSocketServer'):
        super().__init__(server)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def handle_message(self, client: ClientInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle session management messages"""
        
        if message.type == MessageType.SESSION_START:
            return await self._handle_session_start(client, message)
        elif message.type == MessageType.SESSION_END:
            return await self._handle_session_end(client, message)
        elif message.type == MessageType.SESSION_STATUS:
            return await self._handle_session_status(client, message)
        
        return None
    
    async def _handle_session_start(self, client: ClientInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle session start"""
        
        session_id = message.data.get("session_id", str(uuid.uuid4()))
        
        session_info = {
            "session_id": session_id,
            "client_id": client.client_id,
            "started_at": time.time(),
            "language": message.data.get("language", "English"),
            "voice_mode": message.data.get("voice_mode", "consistent"),
            "features": message.data.get("features", []),
            "metadata": message.data.get("metadata", {})
        }
        
        self.active_sessions[session_id] = session_info
        client.session_id = session_id
        
        logger.info(f"Session started: {session_id} for client {client.client_id}")
        
        return WebSocketMessage(
            type=MessageType.SESSION_START,
            data={
                "session_id": session_id,
                "status": "started",
                "config": session_info
            },
            timestamp=time.time()
        )
    
    async def _handle_session_end(self, client: ClientInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle session end"""
        
        session_id = message.data.get("session_id") or client.session_id
        
        if session_id and session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            session_info["ended_at"] = time.time()
            session_info["duration"] = session_info["ended_at"] - session_info["started_at"]
            
            logger.info(f"Session ended: {session_id}, duration: {session_info['duration']:.2f}s")
            
            del self.active_sessions[session_id]
            client.session_id = None
            
            return WebSocketMessage(
                type=MessageType.SESSION_END,
                data={
                    "session_id": session_id,
                    "status": "ended",
                    "duration": session_info["duration"]
                },
                timestamp=time.time()
            )
        
        return WebSocketMessage(
            type=MessageType.ERROR,
            data={"message": "Session not found"},
            timestamp=time.time()
        )
    
    async def _handle_session_status(self, client: ClientInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle session status request"""
        
        session_id = message.data.get("session_id") or client.session_id
        
        if session_id and session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            
            return WebSocketMessage(
                type=MessageType.SESSION_STATUS,
                data={
                    "session_id": session_id,
                    "status": "active",
                    "uptime": time.time() - session_info["started_at"],
                    "config": session_info
                },
                timestamp=time.time()
            )
        
        return WebSocketMessage(
            type=MessageType.SESSION_STATUS,
            data={
                "session_id": session_id,
                "status": "not_found"
            },
            timestamp=time.time()
        )

class WebSocketServer:
    """Comprehensive WebSocket server for real-time communication"""
    
    def __init__(self, config: ServerConfig = None):
        self.config = config or ServerConfig()
        
        # Server state
        self.server = None
        self.is_running = False
        self.clients: Dict[str, ClientInfo] = {}
        
        # Message handlers
        self.handlers: Dict[MessageType, MessageHandler] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Statistics
        self.stats = {
            "connections_total": 0,
            "connections_active": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "errors": 0,
            "start_time": None
        }
        
        # Initialize default handlers
        self._initialize_handlers()
        
        # Heartbeat task
        self.heartbeat_task = None
    
    def _initialize_handlers(self):
        """Initialize default message handlers"""
        
        # Audio streaming handler
        audio_handler = AudioStreamHandler(self)
        self.handlers[MessageType.AUDIO_START] = audio_handler
        self.handlers[MessageType.AUDIO_CHUNK] = audio_handler
        self.handlers[MessageType.AUDIO_END] = audio_handler
        self.handlers[MessageType.AUDIO_CONFIG] = audio_handler
        
        # Session handler
        session_handler = SessionHandler(self)
        self.handlers[MessageType.SESSION_START] = session_handler
        self.handlers[MessageType.SESSION_END] = session_handler
        self.handlers[MessageType.SESSION_STATUS] = session_handler
    
    async def start_server(self):
        """Start the WebSocket server"""
        
        if self.is_running:
            logger.warning("Server is already running")
            return
        
        logger.info(f"Starting WebSocket server on {self.config.host}:{self.config.port}")
        
        # Configure server options
        server_options = {
            "ping_interval": self.config.heartbeat_interval,
            "ping_timeout": self.config.connection_timeout,
            "max_size": self.config.max_message_size,
            "compression": "deflate" if self.config.enable_compression else None
        }
        
        # SSL configuration
        if self.config.enable_ssl and self.config.ssl_cert_path and self.config.ssl_key_path:
            import ssl
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.config.ssl_cert_path, self.config.ssl_key_path)
            server_options["ssl"] = ssl_context
        
        # Start server
        self.server = await websockets.serve(
            self.handle_client_connection,
            self.config.host,
            self.config.port,
            **server_options
        )
        
        self.is_running = True
        self.stats["start_time"] = time.time()
        
        # Start heartbeat task
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info(f"WebSocket server started on ws{'s' if self.config.enable_ssl else ''}://{self.config.host}:{self.config.port}")
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        
        if not self.is_running:
            return
        
        logger.info("Stopping WebSocket server...")
        
        self.is_running = False
        
        # Cancel heartbeat task
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close all client connections
        for client in list(self.clients.values()):
            try:
                await client.websocket.close()
            except:
                pass
        
        self.clients.clear()
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("WebSocket server stopped")
    
    async def handle_client_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new client connection"""
        
        client_id = str(uuid.uuid4())
        
        # Check connection limit
        if len(self.clients) >= self.config.max_connections:
            await websocket.close(code=1013, reason="Server at capacity")
            return
        
        # Create client info
        client = ClientInfo(
            client_id=client_id,
            websocket=websocket,
            connected_at=datetime.now(),
            last_activity=datetime.now(),
            client_type="unknown",
            user_agent=websocket.request_headers.get("User-Agent"),
            ip_address=websocket.remote_address[0] if websocket.remote_address else "unknown",
            session_id=None,
            capabilities=set(),
            metadata={}
        )
        
        self.clients[client_id] = client
        self.stats["connections_total"] += 1
        self.stats["connections_active"] += 1
        
        logger.info(f"New client connected: {client_id} from {client.ip_address}")
        
        try:
            # Send welcome message
            await self.send_message(client, WebSocketMessage(
                type=MessageType.CONNECTION_ESTABLISHED,
                data={
                    "client_id": client_id,
                    "server_info": {
                        "version": "1.0.0",
                        "capabilities": ["audio_streaming", "tts", "asr", "vad"],
                        "max_message_size": self.config.max_message_size,
                        "heartbeat_interval": self.config.heartbeat_interval
                    }
                },
                timestamp=time.time()
            ))
            
            # Handle messages
            async for message in websocket:
                await self.handle_client_message(client, message)
                
        except ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except WebSocketException as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for client {client_id}: {e}")
        finally:
            # Clean up client
            await self.cleanup_client(client_id)
    
    async def handle_client_message(self, client: ClientInfo, raw_message):
        """Handle message from client"""
        
        try:
            # Parse message
            if isinstance(raw_message, bytes):
                message_data = json.loads(raw_message.decode('utf-8'))
            else:
                message_data = json.loads(raw_message)
            
            # Create message object
            message = WebSocketMessage(
                type=MessageType(message_data.get("type")),
                data=message_data.get("data", {}),
                timestamp=message_data.get("timestamp", time.time()),
                client_id=client.client_id,
                session_id=client.session_id,
                message_id=message_data.get("message_id")
            )
            
            # Update client activity
            client.last_activity = datetime.now()
            self.stats["messages_received"] += 1
            self.stats["bytes_received"] += len(raw_message)
            
            # Handle heartbeat
            if message.type == MessageType.HEARTBEAT:
                await self.send_message(client, WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={"status": "alive"},
                    timestamp=time.time()
                ))
                return
            
            # Route to appropriate handler
            handler = self.handlers.get(message.type)
            if handler:
                response = await handler.handle_message(client, message)
                if response:
                    await self.send_message(client, response)
            else:
                # Unknown message type
                await self.send_message(client, WebSocketMessage(
                    type=MessageType.ERROR,
                    data={"message": f"Unknown message type: {message.type.value}"},
                    timestamp=time.time()
                ))
            
        except json.JSONDecodeError:
            await self.send_message(client, WebSocketMessage(
                type=MessageType.ERROR,
                data={"message": "Invalid JSON message"},
                timestamp=time.time()
            ))
        except ValueError as e:
            await self.send_message(client, WebSocketMessage(
                type=MessageType.ERROR,
                data={"message": f"Invalid message format: {e}"},
                timestamp=time.time()
            ))
        except Exception as e:
            logger.error(f"Message handling error for client {client.client_id}: {e}")
            self.stats["errors"] += 1
            await self.send_message(client, WebSocketMessage(
                type=MessageType.ERROR,
                data={"message": "Internal server error"},
                timestamp=time.time()
            ))
    
    async def send_message(self, client: ClientInfo, message: WebSocketMessage):
        """Send message to client"""
        
        try:
            # Prepare message data
            message_data = {
                "type": message.type.value,
                "data": message.data,
                "timestamp": message.timestamp
            }
            
            if message.message_id:
                message_data["message_id"] = message.message_id
            
            # Send message
            message_json = json.dumps(message_data)
            await client.websocket.send(message_json)
            
            # Update statistics
            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += len(message_json)
            
        except ConnectionClosed:
            logger.debug(f"Cannot send message to disconnected client: {client.client_id}")
        except Exception as e:
            logger.error(f"Failed to send message to client {client.client_id}: {e}")
            self.stats["errors"] += 1
    
    async def broadcast_message(self, message: WebSocketMessage, exclude_clients: Set[str] = None):
        """Broadcast message to all connected clients"""
        
        exclude_clients = exclude_clients or set()
        
        tasks = []
        for client in self.clients.values():
            if client.client_id not in exclude_clients:
                tasks.append(self.send_message(client, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def broadcast_to_handlers(self, event: str, data: Dict[str, Any]):
        """Broadcast event to registered handlers"""
        
        handlers = self.event_handlers.get(event, [])
        
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(data))
            else:
                # Run sync handler in thread pool
                tasks.append(asyncio.get_event_loop().run_in_executor(None, handler, data))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def cleanup_client(self, client_id: str):
        """Clean up disconnected client"""
        
        if client_id in self.clients:
            client = self.clients[client_id]
            
            # Notify handlers about disconnection
            await self.broadcast_to_handlers("client_disconnected", {
                "client_id": client_id,
                "session_id": client.session_id,
                "connected_duration": (datetime.now() - client.connected_at).total_seconds()
            })
            
            del self.clients[client_id]
            self.stats["connections_active"] -= 1
            
            logger.info(f"Client cleaned up: {client_id}")
    
    async def _heartbeat_loop(self):
        """Heartbeat loop to check client connections"""
        
        while self.is_running:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.config.connection_timeout)
                
                # Check for timed out clients
                timed_out_clients = []
                for client_id, client in self.clients.items():
                    if client.last_activity < timeout_threshold:
                        timed_out_clients.append(client_id)
                
                # Clean up timed out clients
                for client_id in timed_out_clients:
                    client = self.clients.get(client_id)
                    if client:
                        try:
                            await client.websocket.close(code=1000, reason="Connection timeout")
                        except:
                            pass
                        await self.cleanup_client(client_id)
                
                # Wait for next heartbeat interval
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(5)
    
    def register_handler(self, message_type: MessageType, handler: MessageHandler):
        """Register message handler"""
        self.handlers[message_type] = handler
    
    def register_event_handler(self, event: str, handler: Callable):
        """Register event handler"""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    def get_client(self, client_id: str) -> Optional[ClientInfo]:
        """Get client by ID"""
        return self.clients.get(client_id)
    
    def get_clients_by_session(self, session_id: str) -> List[ClientInfo]:
        """Get all clients for a session"""
        return [client for client in self.clients.values() if client.session_id == session_id]
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        
        stats = self.stats.copy()
        
        if stats["start_time"]:
            stats["uptime_seconds"] = time.time() - stats["start_time"]
        else:
            stats["uptime_seconds"] = 0
        
        stats["config"] = {
            "host": self.config.host,
            "port": self.config.port,
            "max_connections": self.config.max_connections,
            "heartbeat_interval": self.config.heartbeat_interval,
            "max_message_size": self.config.max_message_size
        }
        
        return stats

# Global WebSocket server instance
websocket_server = None

def get_websocket_server(config: ServerConfig = None) -> WebSocketServer:
    """Get the global WebSocket server instance"""
    global websocket_server
    if websocket_server is None:
        websocket_server = WebSocketServer(config)
    return websocket_server

async def main():
    """Main function for testing"""
    
    # Create and start server
    server = WebSocketServer()
    
    try:
        await server.start_server()
        
        # Keep server running
        while server.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await server.stop_server()

if __name__ == "__main__":
    asyncio.run(main())