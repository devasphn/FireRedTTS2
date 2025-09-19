#!/usr/bin/env python3

"""
API Interfaces and Handlers
Comprehensive REST API and WebSocket handlers for FireRedTTS2 system
"""

import asyncio
import json
import time
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError
import uvicorn

from data_models import (
    AudioChunk, AudioStream, VoiceProfile, ConversationTurn, ConversationSession,
    GenerationRequest, GenerationResult, LatencyMetrics, ResourceUsage, QualityMetrics,
    APIResponse, PaginatedResponse, WebSocketMessage, WebSocketEvent,
    AudioFormat, VoiceMode, GenerationStatus, SessionStatus, Priority,
    create_error_response, create_success_response, serialize_model, ModelRegistry
)

logger = logging.getLogger(__name__)

# ============================================================================
# API CONFIGURATION
# ============================================================================

class APIConfig:
    """API configuration settings"""
    
    def __init__(self):
        self.title = "FireRedTTS2 API"
        self.description = "Advanced Text-to-Speech API with real-time capabilities"
        self.version = "1.0.0"
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = False
        self.enable_cors = True
        self.cors_origins = ["*"]
        self.max_upload_size = 100 * 1024 * 1024  # 100MB
        self.rate_limit_requests = 100
        self.rate_limit_window = 60  # seconds
        self.enable_auth = False
        self.auth_token = None
        self.enable_docs = True
        self.docs_url = "/docs"
        self.redoc_url = "/redoc"

# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

class ServiceContainer:
    """Dependency injection container"""
    
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register(self, service_name: str, service_factory: Callable, singleton: bool = True):
        """Register a service"""
        self._services[service_name] = {
            'factory': service_factory,
            'singleton': singleton
        }
    
    def get(self, service_name: str):
        """Get service instance"""
        if service_name not in self._services:
            raise ValueError(f"Service '{service_name}' not registered")
        
        service_config = self._services[service_name]
        
        if service_config['singleton']:
            if service_name not in self._singletons:
                self._singletons[service_name] = service_config['factory']()
            return self._singletons[service_name]
        else:
            return service_config['factory']()

# Global service container
services = ServiceContainer()

# Dependency functions
def get_tts_service():
    """Get TTS service dependency"""
    return services.get('tts_service')

def get_asr_service():
    """Get ASR service dependency"""
    return services.get('asr_service')

def get_conversation_service():
    """Get conversation service dependency"""
    return services.get('conversation_service')

def get_voice_service():
    """Get voice cloning service dependency"""
    return services.get('voice_service')

def get_performance_service():
    """Get performance monitoring service dependency"""
    return services.get('performance_service')

# ============================================================================
# AUTHENTICATION
# ============================================================================

security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token"""
    config = services.get('config')
    
    if not config.enable_auth:
        return True
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if credentials.credentials != config.auth_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    return True

# ============================================================================
# ERROR HANDLERS
# ============================================================================

class APIException(Exception):
    """Custom API exception"""
    
    def __init__(self, message: str, status_code: int = 400, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

def create_error_handler(app: FastAPI):
    """Create error handlers for the FastAPI app"""
    
    @app.exception_handler(APIException)
    async def api_exception_handler(request, exc: APIException):
        return JSONResponse(
            status_code=exc.status_code,
            content=serialize_model(create_error_response(
                message=exc.message,
                error_code=exc.error_code,
                details=exc.details
            ))
        )
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request, exc: ValidationError):
        return JSONResponse(
            status_code=422,
            content=serialize_model(create_error_response(
                message="Validation error",
                error_code="VALIDATION_ERROR",
                details={"errors": exc.errors()}
            ))
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=serialize_model(create_error_response(
                message=exc.detail,
                error_code="HTTP_ERROR"
            ))
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content=serialize_model(create_error_response(
                message="Internal server error",
                error_code="INTERNAL_ERROR",
                details={"type": type(exc).__name__}
            ))
        )

# ============================================================================
# API ROUTES
# ============================================================================

def create_tts_routes(app: FastAPI):
    """Create TTS-related API routes"""
    
    @app.post("/api/v1/tts/generate", response_model=APIResponse)
    async def generate_speech(
        request: GenerationRequest,
        auth: bool = Depends(verify_token),
        tts_service = Depends(get_tts_service)
    ):
        """Generate speech from text"""
        
        try:
            start_time = time.time()
            
            # Validate request
            if not request.text.strip():
                raise APIException("Text cannot be empty", 400, "EMPTY_TEXT")
            
            # Generate speech
            result = await tts_service.generate_speech(request)
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000
            
            response = create_success_response(
                data=serialize_model(result),
                message="Speech generated successfully"
            )
            response.execution_time_ms = execution_time
            response.request_id = request.request_id
            
            return response
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise APIException(f"Generation failed: {str(e)}", 500, "GENERATION_ERROR")
    
    @app.post("/api/v1/tts/generate-streaming")
    async def generate_speech_streaming(
        request: GenerationRequest,
        auth: bool = Depends(verify_token),
        tts_service = Depends(get_tts_service)
    ):
        """Generate speech with streaming response"""
        
        async def stream_generator():
            try:
                async for chunk in tts_service.generate_speech_streaming(request):
                    yield chunk
            except Exception as e:
                logger.error(f"Streaming generation failed: {e}")
                yield json.dumps({"error": str(e)}).encode()
        
        return StreamingResponse(
            stream_generator(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=speech_{request.request_id}.wav"}
        )
    
    @app.get("/api/v1/tts/status/{request_id}", response_model=APIResponse)
    async def get_generation_status(
        request_id: str,
        auth: bool = Depends(verify_token),
        tts_service = Depends(get_tts_service)
    ):
        """Get generation request status"""
        
        try:
            status = await tts_service.get_generation_status(request_id)
            
            if not status:
                raise APIException("Request not found", 404, "REQUEST_NOT_FOUND")
            
            return create_success_response(
                data=serialize_model(status),
                message="Status retrieved successfully"
            )
            
        except APIException:
            raise
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            raise APIException(f"Status retrieval failed: {str(e)}", 500, "STATUS_ERROR")

def create_voice_routes(app: FastAPI):
    """Create voice cloning API routes"""
    
    @app.post("/api/v1/voices/create", response_model=APIResponse)
    async def create_voice_profile(
        name: str = Form(...),
        description: str = Form(""),
        reference_text: str = Form(...),
        language: str = Form("English"),
        gender: str = Form("Unknown"),
        age_range: str = Form("Unknown"),
        audio_file: UploadFile = File(...),
        auth: bool = Depends(verify_token),
        voice_service = Depends(get_voice_service)
    ):
        """Create new voice profile"""
        
        try:
            # Validate audio file
            if not audio_file.content_type.startswith('audio/'):
                raise APIException("Invalid audio file type", 400, "INVALID_FILE_TYPE")
            
            # Read audio data
            audio_data = await audio_file.read()
            
            # Create voice profile
            profile = await voice_service.create_voice_profile(
                name=name,
                description=description,
                reference_text=reference_text,
                language=language,
                gender=gender,
                age_range=age_range,
                audio_data=audio_data,
                filename=audio_file.filename
            )
            
            return create_success_response(
                data=serialize_model(profile),
                message="Voice profile created successfully"
            )
            
        except APIException:
            raise
        except Exception as e:
            logger.error(f"Voice profile creation failed: {e}")
            raise APIException(f"Profile creation failed: {str(e)}", 500, "PROFILE_ERROR")

def create_conversation_routes(app: FastAPI):
    """Create conversation API routes"""
    
    @app.post("/api/v1/conversations/start", response_model=APIResponse)
    async def start_conversation(
        language: str = "English",
        response_style: str = "conversational",
        voice_mode: VoiceMode = VoiceMode.CONSISTENT,
        voice_profile_id: Optional[str] = None,
        auth: bool = Depends(verify_token),
        conversation_service = Depends(get_conversation_service)
    ):
        """Start new conversation session"""
        
        try:
            session = await conversation_service.start_conversation(
                language=language,
                response_style=response_style,
                voice_mode=voice_mode,
                voice_profile_id=voice_profile_id
            )
            
            return create_success_response(
                data=serialize_model(session),
                message="Conversation started successfully"
            )
            
        except Exception as e:
            logger.error(f"Conversation start failed: {e}")
            raise APIException(f"Start failed: {str(e)}", 500, "START_ERROR")

def create_monitoring_routes(app: FastAPI):
    """Create monitoring and health check routes"""
    
    @app.get("/health", response_model=APIResponse)
    async def health_check():
        """Health check endpoint"""
        
        try:
            # Check system health
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "uptime_seconds": time.time() - getattr(health_check, 'start_time', time.time())
            }
            
            return create_success_response(
                data=health_status,
                message="System is healthy"
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise APIException(f"Health check failed: {str(e)}", 500, "HEALTH_ERROR")

# ============================================================================
# WEBSOCKET MANAGER
# ============================================================================

class WebSocketManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0
        }
        logger.info(f"WebSocket client connected: {client_id}")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_metadata:
            del self.connection_metadata[client_id]
        logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
                self.connection_metadata[client_id]["last_activity"] = datetime.now()
                self.connection_metadata[client_id]["message_count"] += 1
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)

# Global WebSocket manager
ws_manager = WebSocketManager()

def create_websocket_routes(app: FastAPI):
    """Create WebSocket routes"""
    
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(
        websocket: WebSocket,
        client_id: str
    ):
        """Main WebSocket endpoint"""
        
        await ws_manager.connect(websocket, client_id)
        
        try:
            # Send welcome message
            await ws_manager.send_message(client_id, {
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "message": "WebSocket connection established"
            })
            
            while True:
                # Receive message
                data = await websocket.receive_json()
                
                # Process message
                await handle_websocket_message(client_id, data)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket client {client_id} disconnected")
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
        finally:
            ws_manager.disconnect(client_id)

async def handle_websocket_message(client_id: str, data: Dict[str, Any]):
    """Handle incoming WebSocket message"""
    
    try:
        message_type = data.get("type")
        
        if message_type == "ping":
            await ws_manager.send_message(client_id, {
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            })
        else:
            await ws_manager.send_message(client_id, {
                "type": "error",
                "message": f"Unknown message type: {message_type}",
                "timestamp": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"WebSocket message handling error: {e}")
        await ws_manager.send_message(client_id, {
            "type": "error",
            "message": "Message processing failed",
            "timestamp": datetime.now().isoformat()
        })

# ============================================================================
# MAIN API APPLICATION
# ============================================================================

def create_api_app(config: APIConfig = None) -> FastAPI:
    """Create and configure FastAPI application"""
    
    config = config or APIConfig()
    
    # Create FastAPI app
    app = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        docs_url=config.docs_url if config.enable_docs else None,
        redoc_url=config.redoc_url if config.enable_docs else None
    )
    
    # Add CORS middleware
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Register services in container
    services.register('config', lambda: config)
    
    # Create error handlers
    create_error_handler(app)
    
    # Create routes
    create_tts_routes(app)
    create_voice_routes(app)
    create_conversation_routes(app)
    create_monitoring_routes(app)
    create_websocket_routes(app)
    
    # Set health check start time
    health_check.start_time = time.time()
    
    return app

def run_api_server(config: APIConfig = None):
    """Run the API server"""
    config = config or APIConfig()
    
    app = create_api_app(config)
    
    # Run server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        debug=config.debug,
        log_level="info"
    )

# Export main components
__all__ = [
    'APIConfig', 'ServiceContainer', 'APIException', 'WebSocketManager',
    'create_api_app', 'run_api_server', 'ws_manager', 'services'
]

if __name__ == "__main__":
    # Run API server with default configuration
    run_api_server()