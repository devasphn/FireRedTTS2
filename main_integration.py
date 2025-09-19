#!/usr/bin/env python3
"""
Main Integration Script for FireRedTTS2 Speech-to-Speech System
Integrates all components into a complete speech-to-speech conversation system
"""

import os
import sys
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import all system components
from enhanced_gradio_demo import create_enhanced_interface, initialize_interface
from speech_to_speech_interface import SpeechToSpeechInterface, SpeechToSpeechConfig
from websocket_server import WebSocketManager, create_websocket_app
from performance_monitor import PerformanceMonitor
from system_monitoring import SystemMonitor
from security_system import SecurityManager
from error_handling_system import ErrorManager
from data_models import SystemStatus, DeploymentConfig
from fireredtts2.fireredtts2 import FireRedTTS2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for the integrated system"""
    
    # Model configuration
    pretrained_dir: str = "/workspace/models"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 7860
    websocket_port: int = 8765
    
    # System configuration
    enable_performance_monitoring: bool = True
    enable_security: bool = True
    enable_error_handling: bool = True
    
    # Speech-to-speech configuration
    enable_asr: bool = True
    enable_llm: bool = True
    enable_tts: bool = True
    
    # RunPod specific settings
    runpod_mode: bool = False
    persistent_volume_path: str = "/workspace"

class IntegratedSpeechToSpeechSystem:
    """Main integrated speech-to-speech system"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.status = SystemStatus.INITIALIZING
        
        # Core components
        self.tts_model: Optional[FireRedTTS2] = None
        self.speech_interface: Optional[SpeechToSpeechInterface] = None
        self.gradio_app: Optional[gr.Blocks] = None
        self.websocket_manager: Optional[WebSocketManager] = None
        
        # System components
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.security_manager: Optional[SecurityManager] = None
        self.error_manager: Optional[ErrorManager] = None
        
        # Server components
        self.fastapi_app: Optional[FastAPI] = None
        self.gradio_server = None
        self.websocket_server = None
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        
        logger.info("Initializing integrated speech-to-speech system...")
        
        try:
            # Initialize error handling first
            if self.config.enable_error_handling:
                self.error_manager = ErrorManager()
                logger.info("Error handling system initialized")
            
            # Initialize security
            if self.config.enable_security:
                self.security_manager = SecurityManager()
                logger.info("Security system initialized")
            
            # Initialize monitoring
            if self.config.enable_performance_monitoring:
                self.performance_monitor = PerformanceMonitor()
                self.system_monitor = SystemMonitor()
                self.performance_monitor.start_monitoring()
                logger.info("Monitoring systems initialized")
            
            # Initialize TTS model
            self._initialize_tts_model()
            
            # Initialize speech-to-speech interface
            self._initialize_speech_interface()
            
            # Initialize web interface
            self._initialize_web_interface()
            
            # Initialize WebSocket server
            self._initialize_websocket_server()
            
            # Initialize FastAPI app
            self._initialize_fastapi_app()
            
            self.status = SystemStatus.READY
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            logger.error(f"System initialization failed: {e}")
            if self.error_manager:
                self.error_manager.handle_error(e, "system_initialization")
            raise
    
    def _initialize_tts_model(self):
        """Initialize the TTS model"""
        
        logger.info("Initializing TTS model...")
        
        try:
            # Check if model directory exists
            model_path = Path(self.config.pretrained_dir)
            if not model_path.exists():
                logger.warning(f"Model directory not found: {model_path}")
                # Create directory and download models if needed
                model_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize FireRedTTS2 model
            self.tts_model = FireRedTTS2(
                pretrained_dir=str(model_path),
                gen_type="dialogue",
                device=self.config.device
            )
            
            logger.info(f"TTS model initialized on device: {self.config.device}")
            
        except Exception as e:
            logger.error(f"TTS model initialization failed: {e}")
            raise
    
    def _initialize_speech_interface(self):
        """Initialize the speech-to-speech interface"""
        
        logger.info("Initializing speech-to-speech interface...")
        
        try:
            # Create speech-to-speech configuration
            s2s_config = SpeechToSpeechConfig(
                enable_asr=self.config.enable_asr,
                enable_llm=self.config.enable_llm,
                enable_tts=self.config.enable_tts,
                asr_model="whisper-base",
                llm_model="local",
                voice_mode="consistent",
                language="English"
            )
            
            # Initialize speech interface
            self.speech_interface = SpeechToSpeechInterface(
                tts_model=self.tts_model,
                config=s2s_config
            )
            
            logger.info("Speech-to-speech interface initialized")
            
        except Exception as e:
            logger.error(f"Speech interface initialization failed: {e}")
            raise
    
    def _initialize_web_interface(self):
        """Initialize the web interface"""
        
        logger.info("Initializing web interface...")
        
        try:
            # Initialize the enhanced Gradio interface
            self.gradio_app = create_enhanced_interface(self.config.pretrained_dir)
            
            logger.info("Web interface initialized")
            
        except Exception as e:
            logger.error(f"Web interface initialization failed: {e}")
            raise
    
    def _initialize_websocket_server(self):
        """Initialize the WebSocket server"""
        
        logger.info("Initializing WebSocket server...")
        
        try:
            # Initialize WebSocket manager
            self.websocket_manager = WebSocketManager()
            
            # Create WebSocket app
            self.websocket_server = create_websocket_app(
                self.websocket_manager,
                self.speech_interface
            )
            
            logger.info("WebSocket server initialized")
            
        except Exception as e:
            logger.error(f"WebSocket server initialization failed: {e}")
            raise
    
    def _initialize_fastapi_app(self):
        """Initialize the FastAPI application"""
        
        logger.info("Initializing FastAPI application...")
        
        try:
            # Create FastAPI app
            self.fastapi_app = FastAPI(
                title="FireRedTTS2 Speech-to-Speech API",
                description="Advanced speech-to-speech conversation system",
                version="1.0.0"
            )
            
            # Add CORS middleware
            self.fastapi_app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Add API routes
            self._setup_api_routes()
            
            logger.info("FastAPI application initialized")
            
        except Exception as e:
            logger.error(f"FastAPI initialization failed: {e}")
            raise
    
    def _setup_api_routes(self):
        """Setup API routes"""
        
        @self.fastapi_app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "system_status": self.status.value,
                "timestamp": time.time(),
                "components": {
                    "tts_model": self.tts_model is not None,
                    "speech_interface": self.speech_interface is not None,
                    "websocket_manager": self.websocket_manager is not None
                }
            }
        
        @self.fastapi_app.get("/api/v1/system/status")
        async def get_system_status():
            """Get detailed system status"""
            
            status_info = {
                "system_status": self.status.value,
                "timestamp": time.time(),
                "device": self.config.device,
                "cuda_available": torch.cuda.is_available()
            }
            
            # Add performance metrics if available
            if self.performance_monitor:
                status_info["performance"] = self.performance_monitor.get_current_metrics()
            
            # Add system metrics if available
            if self.system_monitor:
                status_info["system"] = self.system_monitor.get_system_info()
            
            return status_info
        
        @self.fastapi_app.post("/api/v1/tts/generate")
        async def generate_speech(request: dict):
            """Generate speech from text"""
            
            try:
                text = request.get("text", "")
                voice_mode = request.get("voice_mode", "random")
                
                if not text.strip():
                    return {"success": False, "error": "Empty text input"}
                
                # Generate speech using TTS model
                if self.tts_model:
                    audio_tensor = self.tts_model.generate_monologue(
                        text=text,
                        temperature=0.9,
                        topk=30
                    )
                    
                    return {
                        "success": True,
                        "message": "Speech generated successfully",
                        "audio_length": len(audio_tensor) if hasattr(audio_tensor, '__len__') else 0
                    }
                else:
                    return {"success": False, "error": "TTS model not available"}
                    
            except Exception as e:
                logger.error(f"Speech generation error: {e}")
                return {"success": False, "error": str(e)}
        
        @self.fastapi_app.post("/api/v1/conversation/process")
        async def process_conversation(request: dict):
            """Process conversation input"""
            
            try:
                user_input = request.get("user_input", "")
                input_type = request.get("input_type", "text")  # text or audio
                
                if not user_input.strip():
                    return {"success": False, "error": "Empty input"}
                
                # Process through speech interface
                if self.speech_interface:
                    if input_type == "text":
                        result = self.speech_interface.process_text_input(user_input)
                    else:
                        # For audio input, would need to decode audio data
                        result = {"success": False, "error": "Audio input not implemented in API"}
                    
                    return result
                else:
                    return {"success": False, "error": "Speech interface not available"}
                    
            except Exception as e:
                logger.error(f"Conversation processing error: {e}")
                return {"success": False, "error": str(e)}
    
    def start_servers(self):
        """Start all servers"""
        
        logger.info("Starting integrated system servers...")
        
        try:
            # Start Gradio server in a separate thread
            def start_gradio():
                if self.gradio_app:
                    self.gradio_app.launch(
                        server_name=self.config.host,
                        server_port=self.config.port,
                        share=False,
                        show_error=True,
                        quiet=False
                    )
            
            gradio_thread = threading.Thread(target=start_gradio, daemon=True)
            gradio_thread.start()
            
            logger.info(f"Gradio server started on {self.config.host}:{self.config.port}")
            
            # Start WebSocket server if needed
            if self.websocket_server and self.config.websocket_port != self.config.port:
                def start_websocket():
                    import websockets
                    asyncio.run(
                        websockets.serve(
                            self.websocket_server,
                            self.config.host,
                            self.config.websocket_port
                        )
                    )
                
                websocket_thread = threading.Thread(target=start_websocket, daemon=True)
                websocket_thread.start()
                
                logger.info(f"WebSocket server started on {self.config.host}:{self.config.websocket_port}")
            
            # Keep main thread alive
            logger.info("All servers started successfully")
            logger.info(f"Access the web interface at: http://{self.config.host}:{self.config.port}")
            
            # Wait for servers
            try:
                while True:
                    time.sleep(1)
                    
                    # Check system health
                    if self.system_monitor:
                        health = self.system_monitor.check_system_health()
                        if not health.get("healthy", True):
                            logger.warning("System health check failed")
                    
            except KeyboardInterrupt:
                logger.info("Shutting down servers...")
                self.shutdown()
                
        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            raise
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        
        logger.info("Shutting down integrated system...")
        
        try:
            # Stop monitoring
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            # Close connections
            if self.websocket_manager:
                self.websocket_manager.close_all_connections()
            
            self.status = SystemStatus.SHUTDOWN
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        
        info = {
            "status": self.status.value,
            "config": {
                "device": self.config.device,
                "host": self.config.host,
                "port": self.config.port,
                "pretrained_dir": self.config.pretrained_dir
            },
            "components": {
                "tts_model": self.tts_model is not None,
                "speech_interface": self.speech_interface is not None,
                "gradio_app": self.gradio_app is not None,
                "websocket_manager": self.websocket_manager is not None,
                "performance_monitor": self.performance_monitor is not None,
                "security_manager": self.security_manager is not None
            },
            "timestamp": time.time()
        }
        
        # Add performance metrics
        if self.performance_monitor:
            info["performance"] = self.performance_monitor.get_current_metrics()
        
        # Add system metrics
        if self.system_monitor:
            info["system"] = self.system_monitor.get_system_info()
        
        return info

def create_deployment_package():
    """Create final deployment package with all necessary files"""
    
    logger.info("Creating deployment package...")
    
    deployment_files = [
        # Core system files
        "main_integration.py",
        "enhanced_gradio_demo.py",
        "speech_to_speech_interface.py",
        "websocket_server.py",
        
        # Component files
        "enhanced_fireredtts2.py",
        "whisper_asr.py",
        "conversation_llm.py",
        "voice_cloning_interface.py",
        
        # System files
        "performance_monitor.py",
        "system_monitoring.py",
        "security_system.py",
        "error_handling_system.py",
        "data_models.py",
        "api_interfaces.py",
        
        # Configuration files
        "requirements.txt",
        "Dockerfile",
        "docker-compose.runpod.yml",
        "container_startup.sh",
        "runpod_config.json",
        
        # Deployment files
        "runpod_deployment_complete.py",
        "runpod_optimization.py",
        "runpod_network_config.py",
        
        # Testing files
        "comprehensive_test_suite.py",
        "run_tests.py",
        "test_config.json",
        
        # Documentation
        "README.md",
        "docs/",
        "codebase_analysis.md"
    ]
    
    # Create deployment directory
    deployment_dir = Path("deployment_package")
    deployment_dir.mkdir(exist_ok=True)
    
    # Copy files to deployment package
    import shutil
    
    copied_files = []
    missing_files = []
    
    for file_path in deployment_files:
        source_path = Path(file_path)
        
        if source_path.exists():
            if source_path.is_dir():
                # Copy directory
                dest_path = deployment_dir / source_path.name
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
                copied_files.append(str(source_path))
            else:
                # Copy file
                dest_path = deployment_dir / source_path.name
                shutil.copy2(source_path, dest_path)
                copied_files.append(str(source_path))
        else:
            missing_files.append(str(source_path))
    
    # Create deployment manifest
    manifest = {
        "package_name": "FireRedTTS2_Speech_to_Speech_System",
        "version": "1.0.0",
        "created_at": time.time(),
        "files_included": copied_files,
        "files_missing": missing_files,
        "total_files": len(copied_files),
        "deployment_instructions": [
            "1. Upload all files to RunPod container",
            "2. Run: pip install -r requirements.txt",
            "3. Run: python runpod_deployment_complete.py",
            "4. Run: python main_integration.py --pretrained-dir /workspace/models",
            "5. Access web interface at http://localhost:7860"
        ]
    }
    
    # Save manifest
    manifest_path = deployment_dir / "deployment_manifest.json"
    with open(manifest_path, 'w') as f:
        import json
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Deployment package created at: {deployment_dir}")
    logger.info(f"Files included: {len(copied_files)}")
    logger.info(f"Files missing: {len(missing_files)}")
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
    
    return deployment_dir, manifest

def main():
    """Main function to run the integrated system"""
    
    parser = ArgumentParser(description="Integrated FireRedTTS2 Speech-to-Speech System")
    parser.add_argument("--pretrained-dir", type=str, default="/workspace/models",
                       help="Path to pretrained models directory")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind servers to")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port for web interface")
    parser.add_argument("--websocket-port", type=int, default=8765,
                       help="Port for WebSocket server")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run models on (auto, cuda, cpu)")
    parser.add_argument("--runpod-mode", action="store_true",
                       help="Enable RunPod specific optimizations")
    parser.add_argument("--create-package", action="store_true",
                       help="Create deployment package and exit")
    
    args = parser.parse_args()
    
    # Create deployment package if requested
    if args.create_package:
        create_deployment_package()
        return
    
    # Auto-detect device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create integration configuration
    config = IntegrationConfig(
        pretrained_dir=args.pretrained_dir,
        device=args.device,
        host=args.host,
        port=args.port,
        websocket_port=args.websocket_port,
        runpod_mode=args.runpod_mode
    )
    
    logger.info("Starting FireRedTTS2 Integrated Speech-to-Speech System")
    logger.info(f"Configuration: {config}")
    
    try:
        # Initialize and start the integrated system
        system = IntegratedSpeechToSpeechSystem(config)
        
        # Print system information
        system_info = system.get_system_info()
        logger.info(f"System initialized: {system_info}")
        
        # Start servers
        system.start_servers()
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()