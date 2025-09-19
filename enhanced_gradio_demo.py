#!/usr/bin/env python3
"""
Enhanced Gradio Demo for FireRedTTS2 with Real-time Audio I/O
Supports speech-to-speech conversation, voice cloning, and performance monitoring
"""

import re
import json
import time
import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from argparse import ArgumentParser

import gradio as gr
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from fireredtts2.fireredtts2 import FireRedTTS2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioChunk:
    """Represents a chunk of audio data with metadata"""
    data: np.ndarray
    sample_rate: int
    timestamp: float
    format: str = "wav"
    channels: int = 1

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    turn_id: str
    user_input: str
    system_response: str
    audio_input: Optional[AudioChunk] = None
    audio_output: Optional[AudioChunk] = None
    timestamp: float = 0.0
    processing_time: float = 0.0

@dataclass
class PerformanceMetrics:
    """Performance monitoring data"""
    gpu_utilization: float = 0.0
    gpu_memory_used: int = 0
    gpu_memory_total: int = 0
    cpu_utilization: float = 0.0
    memory_used: int = 0
    generation_latency: float = 0.0
    audio_quality_score: float = 0.0
    active_sessions: int = 0

class AudioProcessor:
    """Handles audio processing and format conversion"""
    
    def __init__(self):
        self.supported_formats = ['wav', 'mp3', 'flac', 'webm', 'ogg']
        self.target_sample_rate = 16000
    
    def validate_audio(self, audio_data: Any) -> bool:
        """Validate audio input"""
        if audio_data is None:
            return False
        
        if isinstance(audio_data, tuple):
            sample_rate, audio_array = audio_data
            return isinstance(sample_rate, int) and isinstance(audio_array, np.ndarray)
        
        return False
    
    def convert_to_target_format(self, audio_data: Tuple[int, np.ndarray]) -> AudioChunk:
        """Convert audio to target format for processing"""
        sample_rate, audio_array = audio_data
        
        # Ensure mono audio
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Convert to float32 and normalize
        if audio_array.dtype != np.float32:
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            else:
                audio_array = audio_array.astype(np.float32)
        
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            # Simple resampling (for production, use librosa or torchaudio)
            duration = len(audio_array) / sample_rate
            target_length = int(duration * self.target_sample_rate)
            audio_array = np.interp(
                np.linspace(0, len(audio_array), target_length),
                np.arange(len(audio_array)),
                audio_array
            )
            sample_rate = self.target_sample_rate
        
        return AudioChunk(
            data=audio_array,
            sample_rate=sample_rate,
            timestamp=time.time(),
            format="wav",
            channels=1
        )
    
    def prepare_for_output(self, audio_tensor: torch.Tensor, sample_rate: int = 24000) -> Tuple[int, np.ndarray]:
        """Prepare audio tensor for Gradio output"""
        if isinstance(audio_tensor, torch.Tensor):
            audio_array = audio_tensor.cpu().numpy()
        else:
            audio_array = audio_tensor
        
        # Ensure proper shape
        if len(audio_array.shape) > 1:
            audio_array = audio_array.squeeze()
        
        # Normalize to prevent clipping
        max_val = np.abs(audio_array).max()
        if max_val > 1.0:
            audio_array = audio_array / max_val
        
        return sample_rate, audio_array

class PerformanceMonitor:
    """Monitors system performance and provides real-time metrics"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        import psutil
        
        while self.monitoring_active:
            try:
                # CPU and memory metrics
                self.metrics.cpu_utilization = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                self.metrics.memory_used = memory.used // (1024**3)  # GB
                
                # GPU metrics (if available)
                if torch.cuda.is_available():
                    self.metrics.gpu_memory_used = torch.cuda.memory_allocated() // (1024**3)
                    self.metrics.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                    
                    # GPU utilization (simplified)
                    try:
                        import subprocess
                        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            self.metrics.gpu_utilization = float(result.stdout.strip())
                    except:
                        pass
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(5)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "GPU Utilization": f"{self.metrics.gpu_utilization:.1f}%",
            "GPU Memory": f"{self.metrics.gpu_memory_used}/{self.metrics.gpu_memory_total} GB",
            "CPU Utilization": f"{self.metrics.cpu_utilization:.1f}%",
            "System Memory": f"{self.metrics.memory_used} GB",
            "Generation Latency": f"{self.metrics.generation_latency:.0f}ms",
            "Audio Quality": f"{self.metrics.audio_quality_score:.2f}",
            "Active Sessions": str(self.metrics.active_sessions)
        }

class EnhancedFireRedTTS2Interface:
    """Enhanced interface for FireRedTTS2 with real-time capabilities"""
    
    def __init__(self, pretrained_dir: str, device: str = "cuda"):
        self.pretrained_dir = pretrained_dir
        self.device = device
        self.model = None
        self.audio_processor = AudioProcessor()
        self.performance_monitor = PerformanceMonitor()
        self.conversation_history: List[ConversationTurn] = []
        self.current_session_id = None
        
        # Initialize model
        self._initialize_model()
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
    
    def _initialize_model(self):
        """Initialize the FireRedTTS2 model"""
        try:
            logger.info("Initializing FireRedTTS2 model...")
            self.model = FireRedTTS2(
                pretrained_dir=self.pretrained_dir,
                gen_type="dialogue",
                device=self.device,
            )
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def process_audio_input(self, audio_data: Tuple[int, np.ndarray]) -> Optional[AudioChunk]:
        """Process incoming audio input"""
        if not self.audio_processor.validate_audio(audio_data):
            return None
        
        try:
            audio_chunk = self.audio_processor.convert_to_target_format(audio_data)
            logger.info(f"Processed audio input: {len(audio_chunk.data)} samples at {audio_chunk.sample_rate}Hz")
            return audio_chunk
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None
    
    def generate_tts_audio(self, text: str, voice_mode: str = "random", 
                          reference_audio: Optional[Tuple] = None,
                          reference_text: str = "") -> Optional[Tuple[int, np.ndarray]]:
        """Generate TTS audio from text"""
        if not text.strip():
            return None
        
        try:
            start_time = time.time()
            
            # Prepare text with speaker tag
            if not text.startswith("[S"):
                text = f"[S1]{text}"
            
            if voice_mode == "clone" and reference_audio is not None:
                # Voice cloning mode
                audio_chunk = self.process_audio_input(reference_audio)
                if audio_chunk is None:
                    return None
                
                # Save temporary audio file
                temp_audio_path = "/tmp/reference_audio.wav"
                torchaudio.save(
                    temp_audio_path,
                    torch.from_numpy(audio_chunk.data).unsqueeze(0),
                    audio_chunk.sample_rate
                )
                
                # Generate with voice cloning
                audio_tensor = self.model.generate_monologue(
                    text=text[4:],  # Remove speaker tag
                    prompt_wav=temp_audio_path,
                    prompt_text=reference_text or f"[S1]{text[:50]}...",
                    temperature=0.9,
                    topk=30
                )
            else:
                # Random voice mode
                audio_tensor = self.model.generate_monologue(
                    text=text[4:],  # Remove speaker tag
                    temperature=0.9,
                    topk=30
                )
            
            # Update performance metrics
            generation_time = (time.time() - start_time) * 1000
            self.performance_monitor.metrics.generation_latency = generation_time
            
            # Prepare output
            output_audio = self.audio_processor.prepare_for_output(audio_tensor, 24000)
            
            logger.info(f"Generated audio in {generation_time:.0f}ms")
            return output_audio
            
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            return None
    
    def create_audio_visualizer(self) -> str:
        """Create JavaScript for audio visualization"""
        return """
        <div id="audio-visualizer" style="width: 100%; height: 100px; background: #1f2937; border-radius: 8px; position: relative; overflow: hidden;">
            <canvas id="visualizer-canvas" width="800" height="100" style="width: 100%; height: 100%;"></canvas>
        </div>
        
        <script>
        class AudioVisualizer {
            constructor() {
                this.canvas = document.getElementById('visualizer-canvas');
                this.ctx = this.canvas.getContext('2d');
                this.isRecording = false;
                this.animationId = null;
                this.audioData = new Array(128).fill(0);
            }
            
            startVisualization() {
                this.isRecording = true;
                this.animate();
            }
            
            stopVisualization() {
                this.isRecording = false;
                if (this.animationId) {
                    cancelAnimationFrame(this.animationId);
                }
                this.clearCanvas();
            }
            
            animate() {
                if (!this.isRecording) return;
                
                this.drawWaveform();
                this.animationId = requestAnimationFrame(() => this.animate());
            }
            
            drawWaveform() {
                const width = this.canvas.width;
                const height = this.canvas.height;
                
                this.ctx.fillStyle = '#1f2937';
                this.ctx.fillRect(0, 0, width, height);
                
                this.ctx.strokeStyle = '#10b981';
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                
                const sliceWidth = width / this.audioData.length;
                let x = 0;
                
                for (let i = 0; i < this.audioData.length; i++) {
                    const v = this.audioData[i] * height / 2;
                    const y = height / 2 + v;
                    
                    if (i === 0) {
                        this.ctx.moveTo(x, y);
                    } else {
                        this.ctx.lineTo(x, y);
                    }
                    
                    x += sliceWidth;
                }
                
                this.ctx.stroke();
                
                // Simulate audio data (replace with real audio analysis)
                for (let i = 0; i < this.audioData.length; i++) {
                    this.audioData[i] = (Math.random() - 0.5) * (this.isRecording ? 1 : 0.1);
                }
            }
            
            clearCanvas() {
                this.ctx.fillStyle = '#1f2937';
                this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            }
        }
        
        window.audioVisualizer = new AudioVisualizer();
        </script>
        """

# Global interface instance
interface = None

def initialize_interface(pretrained_dir: str, device: str = "cuda") -> EnhancedFireRedTTS2Interface:
    """Initialize the global interface instance"""
    global interface
    if interface is None:
        interface = EnhancedFireRedTTS2Interface(pretrained_dir, device)
    return interface

def create_enhanced_interface(pretrained_dir: str) -> gr.Blocks:
    """Create the enhanced Gradio interface"""
    
    # Initialize the interface
    tts_interface = initialize_interface(pretrained_dir)
    
    # Custom CSS for enhanced styling
    custom_css = """
    .performance-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 10px;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    
    .audio-controls {
        background: #f8fafc;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .conversation-turn {
        background: #ffffff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-ready { background-color: #10b981; }
    .status-processing { background-color: #f59e0b; }
    .status-error { background-color: #ef4444; }
    """
    
    with gr.Blocks(
        title="FireRedTTS2 - Enhanced Speech-to-Speech System",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🔥 FireRedTTS2 - Enhanced Speech-to-Speech System
        
        Advanced text-to-speech system with real-time audio processing, voice cloning, and multi-speaker dialogue generation.
        """)
        
        # Status indicator
        with gr.Row():
            status_display = gr.HTML(
                '<div><span class="status-indicator status-ready"></span>System Ready</div>'
            )
        
        # Main interface tabs
        with gr.Tabs():
            
            # Real-time Audio Tab
            with gr.TabItem("🎤 Real-time Audio"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Audio Input")
                        
                        # Audio input with real-time processing
                        audio_input = gr.Audio(
                            label="Microphone Input",
                            source="microphone",
                            type="numpy",
                            streaming=False,
                            interactive=True
                        )
                        
                        # Audio visualization
                        audio_viz = gr.HTML(tts_interface.create_audio_visualizer())
                        
                        # Recording controls
                        with gr.Row():
                            record_btn = gr.Button("🎙️ Start Recording", variant="primary")
                            stop_btn = gr.Button("⏹️ Stop Recording", variant="secondary")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Audio Output")
                        
                        # Audio output
                        audio_output = gr.Audio(
                            label="Generated Speech",
                            type="numpy",
                            interactive=False
                        )
                        
                        # Generation controls
                        with gr.Row():
                            generate_btn = gr.Button("🔊 Generate Speech", variant="primary")
                            play_btn = gr.Button("▶️ Play", variant="secondary")
                
                # Text input for TTS
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter text to convert to speech...",
                        lines=3,
                        max_lines=5
                    )
                
                # Voice mode selection
                with gr.Row():
                    voice_mode = gr.Radio(
                        choices=["random", "clone"],
                        value="random",
                        label="Voice Mode",
                        info="Choose between random voice generation or voice cloning"
                    )
                
                # Voice cloning controls (shown when clone mode is selected)
                with gr.Group(visible=False) as clone_controls:
                    gr.Markdown("### Voice Cloning")
                    with gr.Row():
                        reference_audio = gr.Audio(
                            label="Reference Audio",
                            source="upload",
                            type="numpy"
                        )
                        reference_text = gr.Textbox(
                            label="Reference Text",
                            placeholder="Text spoken in the reference audio...",
                            lines=2
                        )
            
            # Speech-to-Speech Tab
            with gr.TabItem("💬 Speech-to-Speech Conversation"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Conversation Interface")
                        
                        # Conversation history display
                        conversation_display = gr.HTML(
                            '<div class="conversation-turn">Start a conversation by speaking or typing...</div>'
                        )
                        
                        # Input methods
                        with gr.Row():
                            conversation_audio_input = gr.Audio(
                                label="Speak your message",
                                source="microphone",
                                type="numpy"
                            )
                            conversation_text_input = gr.Textbox(
                                label="Or type your message",
                                placeholder="Type your message here...",
                                lines=2
                            )
                        
                        # Conversation controls
                        with gr.Row():
                            send_btn = gr.Button("📤 Send Message", variant="primary")
                            new_conversation_btn = gr.Button("🆕 New Conversation", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Conversation Settings")
                        
                        # Conversation configuration
                        conversation_voice_mode = gr.Radio(
                            choices=["random", "consistent", "multi-speaker"],
                            value="consistent",
                            label="Voice Mode"
                        )
                        
                        conversation_language = gr.Dropdown(
                            choices=["English", "Chinese", "Japanese", "Korean", "French", "German", "Russian"],
                            value="English",
                            label="Language"
                        )
                        
                        response_style = gr.Dropdown(
                            choices=["conversational", "formal", "casual", "professional"],
                            value="conversational",
                            label="Response Style"
                        )
                        
                        # Conversation history
                        gr.Markdown("### Recent Turns")
                        conversation_history_display = gr.JSON(
                            label="Conversation History",
                            value=[]
                        )
            
            # Performance Monitoring Tab
            with gr.TabItem("📊 Performance Monitor"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### System Metrics")
                        
                        # Performance metrics display
                        metrics_display = gr.JSON(
                            label="Real-time Metrics",
                            value=tts_interface.performance_monitor.get_metrics()
                        )
                        
                        # Refresh button
                        refresh_metrics_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
                    
                    with gr.Column():
                        gr.Markdown("### System Status")
                        
                        # System information
                        system_info = gr.JSON(
                            label="System Information",
                            value={
                                "CUDA Available": torch.cuda.is_available(),
                                "GPU Count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                                "Device": str(tts_interface.device),
                                "Model Loaded": tts_interface.model is not None
                            }
                        )
                        
                        # Log display
                        gr.Markdown("### Recent Logs")
                        log_display = gr.Textbox(
                            label="Application Logs",
                            lines=10,
                            max_lines=20,
                            interactive=False,
                            value="System initialized successfully..."
                        )
        
        # Event handlers
        def toggle_clone_controls(voice_mode_value):
            return gr.update(visible=(voice_mode_value == "clone"))
        
        def generate_speech(text, voice_mode_value, ref_audio, ref_text):
            if not text.strip():
                return None
            
            result = tts_interface.generate_tts_audio(
                text=text,
                voice_mode=voice_mode_value,
                reference_audio=ref_audio,
                reference_text=ref_text
            )
            
            return result
        
        def update_metrics():
            return tts_interface.performance_monitor.get_metrics()
        
        def start_recording():
            return gr.update(value="🔴 Recording...")
        
        def stop_recording():
            return gr.update(value="🎙️ Start Recording")
        
        # Connect event handlers
        voice_mode.change(
            fn=toggle_clone_controls,
            inputs=[voice_mode],
            outputs=[clone_controls]
        )
        
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, voice_mode, reference_audio, reference_text],
            outputs=[audio_output]
        )
        
        refresh_metrics_btn.click(
            fn=update_metrics,
            outputs=[metrics_display]
        )
        
        record_btn.click(
            fn=start_recording,
            outputs=[record_btn]
        )
        
        stop_btn.click(
            fn=stop_recording,
            outputs=[record_btn]
        )
        
        # Auto-refresh metrics every 5 seconds
        demo.load(
            fn=update_metrics,
            outputs=[metrics_display],
            every=5
        )
    
    return demo

def main():
    """Main function to run the enhanced Gradio demo"""
    parser = ArgumentParser(description="Enhanced FireRedTTS2 Gradio Demo")
    parser.add_argument("--pretrained-dir", type=str, required=True,
                       help="Path to pretrained models directory")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the server on")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run the model on")
    parser.add_argument("--share", action="store_true",
                       help="Create a public link")
    
    args = parser.parse_args()
    
    # Create and launch the interface
    demo = create_enhanced_interface(args.pretrained_dir)
    
    logger.info(f"Starting enhanced FireRedTTS2 interface on {args.host}:{args.port}")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()