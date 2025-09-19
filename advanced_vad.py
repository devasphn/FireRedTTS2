#!/usr/bin/env python3
"""
Advanced Voice Activity Detection (VAD)
Comprehensive VAD system with multiple detection methods and conversation flow management
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

import numpy as np
import torch

logger = logging.getLogger(__name__)

class VADMethod(Enum):
    """VAD detection methods"""
    ENERGY_BASED = "energy_based"
    SPECTRAL = "spectral"
    NEURAL = "neural"
    WEBRTC = "webrtc"
    HYBRID = "hybrid"

class SpeechState(Enum):
    """Speech activity states"""
    SILENCE = "silence"
    SPEECH = "speech"
    TRANSITION_TO_SPEECH = "transition_to_speech"
    TRANSITION_TO_SILENCE = "transition_to_silence"
    UNCERTAIN = "uncertain"

@dataclass
class VADConfig:
    """Configuration for VAD system"""
    method: VADMethod = VADMethod.HYBRID
    sample_rate: int = 16000
    frame_duration_ms: int = 30  # Frame duration in milliseconds
    energy_threshold: float = 0.01
    spectral_threshold: float = 0.5
    min_speech_duration_ms: int = 250  # Minimum speech duration
    min_silence_duration_ms: int = 500  # Minimum silence for end-of-speech
    max_speech_duration_ms: int = 30000  # Maximum continuous speech
    smoothing_window: int = 5  # Number of frames for smoothing
    aggressiveness: int = 2  # WebRTC VAD aggressiveness (0-3)
    enable_noise_suppression: bool = True
    enable_echo_cancellation: bool = False

@dataclass
class VADResult:
    """Result from VAD processing"""
    timestamp: float
    frame_id: int
    is_speech: bool
    confidence: float
    energy: float
    spectral_features: Dict[str, float]
    speech_state: SpeechState
    speech_duration_ms: float
    silence_duration_ms: float
    turn_complete: bool
    metadata: Dict[str, Any]

class EnergyBasedVAD:
    """Energy-based voice activity detection"""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.energy_threshold = config.energy_threshold
        self.adaptive_threshold = config.energy_threshold
        self.noise_floor = 0.001
        self.adaptation_rate = 0.01
        
        # Energy history for adaptive thresholding
        self.energy_history = deque(maxlen=100)
    
    def detect(self, audio_frame: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """Detect voice activity using energy-based method"""
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_frame ** 2))
        
        # Update energy history
        self.energy_history.append(energy)
        
        # Adaptive threshold adjustment
        if len(self.energy_history) >= 10:
            recent_energies = list(self.energy_history)[-10:]
            noise_estimate = np.percentile(recent_energies, 20)  # 20th percentile as noise
            
            # Update noise floor
            self.noise_floor = (1 - self.adaptation_rate) * self.noise_floor + \
                              self.adaptation_rate * noise_estimate
            
            # Adaptive threshold
            self.adaptive_threshold = max(
                self.config.energy_threshold,
                self.noise_floor * 3  # 3x noise floor
            )
        
        # Voice activity detection
        is_speech = energy > self.adaptive_threshold
        confidence = min(energy / (self.adaptive_threshold + 1e-10), 1.0)
        
        features = {
            "energy": float(energy),
            "adaptive_threshold": float(self.adaptive_threshold),
            "noise_floor": float(self.noise_floor)
        }
        
        return is_speech, confidence, features

class SpectralVAD:
    """Spectral-based voice activity detection"""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.spectral_threshold = config.spectral_threshold
        
        # Frequency bands for analysis
        self.speech_bands = [
            (300, 1000),   # Fundamental frequency range
            (1000, 3000),  # Formant range
            (3000, 8000)   # High frequency content
        ]
    
    def detect(self, audio_frame: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """Detect voice activity using spectral analysis"""
        
        # Compute FFT
        fft = np.fft.fft(audio_frame)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(audio_frame), 1/self.sample_rate)[:len(fft)//2]
        
        # Analyze speech-relevant frequency bands
        band_energies = []
        for low_freq, high_freq in self.speech_bands:
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_energy = np.sum(magnitude[band_mask] ** 2)
            band_energies.append(band_energy)
        
        # Calculate spectral features
        total_energy = np.sum(magnitude ** 2)
        speech_energy = sum(band_energies)
        
        # Spectral centroid (brightness measure)
        if total_energy > 0:
            spectral_centroid = np.sum(freqs * magnitude ** 2) / total_energy
        else:
            spectral_centroid = 0
        
        # Spectral rolloff (90% of energy)
        cumulative_energy = np.cumsum(magnitude ** 2)
        if total_energy > 0:
            rolloff_idx = np.where(cumulative_energy >= 0.9 * total_energy)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        else:
            spectral_rolloff = 0
        
        # Voice activity decision
        speech_ratio = speech_energy / (total_energy + 1e-10)
        is_speech = (
            speech_ratio > self.spectral_threshold and
            spectral_centroid > 500 and  # Minimum brightness for speech
            spectral_rolloff > 2000      # Sufficient high-frequency content
        )
        
        confidence = min(speech_ratio * 2, 1.0)
        
        features = {
            "speech_ratio": float(speech_ratio),
            "spectral_centroid": float(spectral_centroid),
            "spectral_rolloff": float(spectral_rolloff),
            "total_energy": float(total_energy)
        }
        
        return is_speech, confidence, features

class WebRTCVAD:
    """WebRTC-based voice activity detection"""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.vad = None
        self.aggressiveness = config.aggressiveness
        
        # Initialize WebRTC VAD
        self._initialize_webrtc_vad()
    
    def _initialize_webrtc_vad(self):
        """Initialize WebRTC VAD"""
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(self.aggressiveness)
            logger.info(f"WebRTC VAD initialized with aggressiveness {self.aggressiveness}")
        except ImportError:
            logger.warning("webrtcvad not installed. Install with: pip install webrtcvad")
            self.vad = None
        except Exception as e:
            logger.error(f"Failed to initialize WebRTC VAD: {e}")
            self.vad = None
    
    def detect(self, audio_frame: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """Detect voice activity using WebRTC VAD"""
        
        if self.vad is None:
            # Fallback to simple energy detection
            energy = np.sqrt(np.mean(audio_frame ** 2))
            is_speech = energy > 0.01
            confidence = min(energy / 0.01, 1.0)
            features = {"energy": float(energy), "method": "fallback"}
            return is_speech, confidence, features
        
        try:
            # Convert to 16-bit PCM
            audio_int16 = (audio_frame * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # WebRTC VAD expects specific frame sizes
            # Ensure frame size is compatible (10ms, 20ms, or 30ms at 8kHz, 16kHz, or 32kHz)
            expected_frame_size = int(self.config.frame_duration_ms * self.config.sample_rate / 1000)
            
            if len(audio_bytes) != expected_frame_size * 2:  # 2 bytes per sample
                # Pad or truncate to expected size
                if len(audio_bytes) < expected_frame_size * 2:
                    audio_bytes += b'\x00' * (expected_frame_size * 2 - len(audio_bytes))
                else:
                    audio_bytes = audio_bytes[:expected_frame_size * 2]
            
            # Detect voice activity
            is_speech = self.vad.is_speech(audio_bytes, self.config.sample_rate)
            
            # WebRTC VAD doesn't provide confidence, so estimate it
            energy = np.sqrt(np.mean(audio_frame ** 2))
            confidence = 0.8 if is_speech else 0.2
            
            features = {
                "webrtc_result": is_speech,
                "energy": float(energy),
                "aggressiveness": self.aggressiveness
            }
            
            return is_speech, confidence, features
            
        except Exception as e:
            logger.error(f"WebRTC VAD detection failed: {e}")
            # Fallback to energy detection
            energy = np.sqrt(np.mean(audio_frame ** 2))
            is_speech = energy > 0.01
            confidence = min(energy / 0.01, 1.0)
            features = {"energy": float(energy), "method": "fallback", "error": str(e)}
            return is_speech, confidence, features

class NeuralVAD:
    """Neural network-based voice activity detection"""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize neural VAD model
        self._initialize_neural_vad()
    
    def _initialize_neural_vad(self):
        """Initialize neural VAD model"""
        try:
            # Try to load a pre-trained VAD model (placeholder)
            # In production, use models like Silero VAD or similar
            logger.info("Neural VAD model initialization (placeholder)")
            # self.model = torch.jit.load('path/to/vad_model.pt')
            # self.model.eval()
            self.model = None  # Placeholder
        except Exception as e:
            logger.warning(f"Neural VAD model not available: {e}")
            self.model = None
    
    def detect(self, audio_frame: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """Detect voice activity using neural network"""
        
        if self.model is None:
            # Fallback to energy detection
            energy = np.sqrt(np.mean(audio_frame ** 2))
            is_speech = energy > 0.01
            confidence = min(energy / 0.01, 1.0)
            features = {"energy": float(energy), "method": "neural_fallback"}
            return is_speech, confidence, features
        
        try:
            # Prepare input for neural model
            audio_tensor = torch.from_numpy(audio_frame).float().unsqueeze(0)
            if self.device == "cuda":
                audio_tensor = audio_tensor.cuda()
            
            # Run inference
            with torch.no_grad():
                output = self.model(audio_tensor)
                speech_prob = torch.sigmoid(output).item()
            
            is_speech = speech_prob > 0.5
            confidence = speech_prob
            
            features = {
                "speech_probability": float(speech_prob),
                "method": "neural"
            }
            
            return is_speech, confidence, features
            
        except Exception as e:
            logger.error(f"Neural VAD detection failed: {e}")
            # Fallback to energy detection
            energy = np.sqrt(np.mean(audio_frame ** 2))
            is_speech = energy > 0.01
            confidence = min(energy / 0.01, 1.0)
            features = {"energy": float(energy), "method": "neural_fallback", "error": str(e)}
            return is_speech, confidence, features

class HybridVAD:
    """Hybrid VAD combining multiple detection methods"""
    
    def __init__(self, config: VADConfig):
        self.config = config
        
        # Initialize individual VAD methods
        self.energy_vad = EnergyBasedVAD(config)
        self.spectral_vad = SpectralVAD(config)
        self.webrtc_vad = WebRTCVAD(config)
        self.neural_vad = NeuralVAD(config)
        
        # Weights for combining methods
        self.method_weights = {
            "energy": 0.3,
            "spectral": 0.3,
            "webrtc": 0.3,
            "neural": 0.1  # Lower weight due to placeholder implementation
        }
    
    def detect(self, audio_frame: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """Detect voice activity using hybrid approach"""
        
        # Run all detection methods
        energy_result = self.energy_vad.detect(audio_frame)
        spectral_result = self.spectral_vad.detect(audio_frame)
        webrtc_result = self.webrtc_vad.detect(audio_frame)
        neural_result = self.neural_vad.detect(audio_frame)
        
        # Combine results
        weighted_confidence = (
            self.method_weights["energy"] * energy_result[1] +
            self.method_weights["spectral"] * spectral_result[1] +
            self.method_weights["webrtc"] * webrtc_result[1] +
            self.method_weights["neural"] * neural_result[1]
        )
        
        # Majority voting with confidence weighting
        speech_votes = [
            energy_result[0],
            spectral_result[0],
            webrtc_result[0],
            neural_result[0]
        ]
        
        speech_count = sum(speech_votes)
        is_speech = speech_count >= 2 or weighted_confidence > 0.6
        
        # Combine features
        combined_features = {
            "hybrid_confidence": float(weighted_confidence),
            "speech_votes": speech_count,
            "energy_confidence": energy_result[1],
            "spectral_confidence": spectral_result[1],
            "webrtc_confidence": webrtc_result[1],
            "neural_confidence": neural_result[1],
            **energy_result[2],
            **spectral_result[2]
        }
        
        return is_speech, weighted_confidence, combined_features

class AdvancedVAD:
    """Advanced Voice Activity Detection system with conversation flow management"""
    
    def __init__(self, config: VADConfig = None):
        self.config = config or VADConfig()
        
        # Initialize VAD detector based on method
        if self.config.method == VADMethod.ENERGY_BASED:
            self.detector = EnergyBasedVAD(self.config)
        elif self.config.method == VADMethod.SPECTRAL:
            self.detector = SpectralVAD(self.config)
        elif self.config.method == VADMethod.WEBRTC:
            self.detector = WebRTCVAD(self.config)
        elif self.config.method == VADMethod.NEURAL:
            self.detector = NeuralVAD(self.config)
        else:  # HYBRID
            self.detector = HybridVAD(self.config)
        
        # State tracking
        self.current_state = SpeechState.SILENCE
        self.speech_start_time = None
        self.silence_start_time = None
        self.last_speech_time = None
        
        # Frame processing
        self.frame_id = 0
        self.frame_duration_s = self.config.frame_duration_ms / 1000.0
        
        # Smoothing buffer
        self.detection_buffer = deque(maxlen=self.config.smoothing_window)
        
        # Statistics
        self.stats = {
            "total_frames": 0,
            "speech_frames": 0,
            "silence_frames": 0,
            "turns_detected": 0,
            "average_turn_duration": 0.0,
            "false_positives": 0,
            "false_negatives": 0
        }
        
        # Callbacks
        self.speech_start_callback: Optional[Callable] = None
        self.speech_end_callback: Optional[Callable] = None
        self.turn_complete_callback: Optional[Callable] = None
    
    def process_frame(self, audio_frame: np.ndarray) -> VADResult:
        """Process a single audio frame"""
        
        current_time = time.time()
        
        # Detect voice activity
        is_speech, confidence, features = self.detector.detect(audio_frame)
        
        # Add to smoothing buffer
        self.detection_buffer.append(is_speech)
        
        # Apply smoothing
        if len(self.detection_buffer) >= self.config.smoothing_window:
            smoothed_speech = sum(self.detection_buffer) >= (self.config.smoothing_window // 2 + 1)
        else:
            smoothed_speech = is_speech
        
        # Update state machine
        new_state, speech_duration, silence_duration, turn_complete = self._update_state(
            smoothed_speech, current_time
        )
        
        # Update statistics
        self._update_stats(smoothed_speech, turn_complete)
        
        # Create result
        result = VADResult(
            timestamp=current_time,
            frame_id=self.frame_id,
            is_speech=smoothed_speech,
            confidence=confidence,
            energy=features.get("energy", 0.0),
            spectral_features=features,
            speech_state=new_state,
            speech_duration_ms=speech_duration,
            silence_duration_ms=silence_duration,
            turn_complete=turn_complete,
            metadata={
                "raw_detection": is_speech,
                "smoothed_detection": smoothed_speech,
                "detection_method": self.config.method.value
            }
        )
        
        # Trigger callbacks
        self._trigger_callbacks(result)
        
        self.frame_id += 1
        return result
    
    def _update_state(self, is_speech: bool, current_time: float) -> Tuple[SpeechState, float, float, bool]:
        """Update VAD state machine"""
        
        speech_duration = 0.0
        silence_duration = 0.0
        turn_complete = False
        
        if is_speech:
            if self.current_state == SpeechState.SILENCE:
                # Transition from silence to speech
                self.speech_start_time = current_time
                self.silence_start_time = None
                self.current_state = SpeechState.TRANSITION_TO_SPEECH
                
            elif self.current_state == SpeechState.TRANSITION_TO_SPEECH:
                # Check if we've been in speech long enough
                if current_time - self.speech_start_time >= self.config.min_speech_duration_ms / 1000:
                    self.current_state = SpeechState.SPEECH
                    
            elif self.current_state == SpeechState.SPEECH:
                # Continue speech
                pass
                
            elif self.current_state == SpeechState.TRANSITION_TO_SILENCE:
                # Back to speech before silence was confirmed
                self.silence_start_time = None
                self.current_state = SpeechState.SPEECH
            
            # Update last speech time
            self.last_speech_time = current_time
            
            # Calculate speech duration
            if self.speech_start_time:
                speech_duration = (current_time - self.speech_start_time) * 1000
        
        else:  # Silence
            if self.current_state == SpeechState.SPEECH:
                # Transition from speech to silence
                self.silence_start_time = current_time
                self.current_state = SpeechState.TRANSITION_TO_SILENCE
                
            elif self.current_state == SpeechState.TRANSITION_TO_SILENCE:
                # Check if we've been in silence long enough
                if current_time - self.silence_start_time >= self.config.min_silence_duration_ms / 1000:
                    self.current_state = SpeechState.SILENCE
                    turn_complete = True
                    self.speech_start_time = None
                    
            elif self.current_state == SpeechState.SILENCE:
                # Continue silence
                pass
                
            elif self.current_state == SpeechState.TRANSITION_TO_SPEECH:
                # Back to silence before speech was confirmed
                self.speech_start_time = None
                self.current_state = SpeechState.SILENCE
            
            # Calculate silence duration
            if self.silence_start_time:
                silence_duration = (current_time - self.silence_start_time) * 1000
        
        # Check for maximum speech duration
        if (self.current_state == SpeechState.SPEECH and 
            self.speech_start_time and 
            current_time - self.speech_start_time >= self.config.max_speech_duration_ms / 1000):
            
            # Force end of turn
            self.current_state = SpeechState.SILENCE
            turn_complete = True
            self.speech_start_time = None
        
        return self.current_state, speech_duration, silence_duration, turn_complete
    
    def _update_stats(self, is_speech: bool, turn_complete: bool):
        """Update VAD statistics"""
        
        self.stats["total_frames"] += 1
        
        if is_speech:
            self.stats["speech_frames"] += 1
        else:
            self.stats["silence_frames"] += 1
        
        if turn_complete:
            self.stats["turns_detected"] += 1
    
    def _trigger_callbacks(self, result: VADResult):
        """Trigger registered callbacks"""
        
        # Speech start callback
        if (result.speech_state == SpeechState.SPEECH and 
            self.speech_start_callback and
            result.speech_duration_ms <= self.frame_duration_s * 1000 * 2):  # Just started
            
            self.speech_start_callback(result)
        
        # Speech end callback
        if (result.speech_state == SpeechState.SILENCE and 
            self.speech_end_callback and
            result.turn_complete):
            
            self.speech_end_callback(result)
        
        # Turn complete callback
        if result.turn_complete and self.turn_complete_callback:
            self.turn_complete_callback(result)
    
    def set_callbacks(self, 
                     speech_start: Optional[Callable] = None,
                     speech_end: Optional[Callable] = None,
                     turn_complete: Optional[Callable] = None):
        """Set callback functions for VAD events"""
        
        self.speech_start_callback = speech_start
        self.speech_end_callback = speech_end
        self.turn_complete_callback = turn_complete
    
    def reset_state(self):
        """Reset VAD state"""
        
        self.current_state = SpeechState.SILENCE
        self.speech_start_time = None
        self.silence_start_time = None
        self.last_speech_time = None
        self.frame_id = 0
        self.detection_buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get VAD statistics"""
        
        stats = self.stats.copy()
        
        # Calculate additional metrics
        if stats["total_frames"] > 0:
            stats["speech_ratio"] = stats["speech_frames"] / stats["total_frames"]
            stats["silence_ratio"] = stats["silence_frames"] / stats["total_frames"]
        else:
            stats["speech_ratio"] = 0.0
            stats["silence_ratio"] = 0.0
        
        stats["current_state"] = self.current_state.value
        stats["detection_method"] = self.config.method.value
        stats["config"] = {
            "frame_duration_ms": self.config.frame_duration_ms,
            "min_speech_duration_ms": self.config.min_speech_duration_ms,
            "min_silence_duration_ms": self.config.min_silence_duration_ms,
            "smoothing_window": self.config.smoothing_window
        }
        
        return stats
    
    def update_config(self, **kwargs):
        """Update VAD configuration"""
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Reinitialize detector if method changed
        if "method" in kwargs:
            if kwargs["method"] == VADMethod.ENERGY_BASED:
                self.detector = EnergyBasedVAD(self.config)
            elif kwargs["method"] == VADMethod.SPECTRAL:
                self.detector = SpectralVAD(self.config)
            elif kwargs["method"] == VADMethod.WEBRTC:
                self.detector = WebRTCVAD(self.config)
            elif kwargs["method"] == VADMethod.NEURAL:
                self.detector = NeuralVAD(self.config)
            else:  # HYBRID
                self.detector = HybridVAD(self.config)

# Global VAD instance
advanced_vad = None

def get_advanced_vad(config: VADConfig = None) -> AdvancedVAD:
    """Get the global advanced VAD instance"""
    global advanced_vad
    if advanced_vad is None:
        advanced_vad = AdvancedVAD(config)
    return advanced_vad

def create_vad_interface_components(vad: AdvancedVAD) -> Dict[str, Any]:
    """Create interface components for VAD"""
    
    def process_audio_for_vad(audio_data):
        """Process audio through VAD for Gradio interface"""
        
        if audio_data is None:
            return "❌ No audio provided", {}
        
        try:
            sample_rate, audio_array = audio_data
            
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Convert to float32
            if audio_array.dtype != np.float32:
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                else:
                    audio_array = audio_array.astype(np.float32)
            
            # Process in frames
            frame_size = int(vad.config.frame_duration_ms * sample_rate / 1000)
            results = []
            
            for i in range(0, len(audio_array), frame_size):
                frame = audio_array[i:i+frame_size]
                if len(frame) < frame_size:
                    # Pad last frame
                    frame = np.pad(frame, (0, frame_size - len(frame)))
                
                result = vad.process_frame(frame)
                results.append(result)
            
            # Analyze results
            speech_frames = sum(1 for r in results if r.is_speech)
            total_frames = len(results)
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
            
            turns_detected = sum(1 for r in results if r.turn_complete)
            
            status = f"✅ Processed {total_frames} frames, {speech_ratio:.1%} speech, {turns_detected} turns detected"
            
            # Summary statistics
            summary = {
                "Total Frames": total_frames,
                "Speech Frames": speech_frames,
                "Speech Ratio": f"{speech_ratio:.1%}",
                "Turns Detected": turns_detected,
                "Detection Method": vad.config.method.value,
                "Average Confidence": f"{np.mean([r.confidence for r in results]):.2f}"
            }
            
            return status, summary
            
        except Exception as e:
            logger.error(f"VAD processing failed: {e}")
            return f"❌ Processing failed: {e}", {}
    
    def get_vad_stats():
        """Get VAD statistics"""
        return vad.get_stats()
    
    def update_vad_config(method, energy_threshold, min_speech_ms, min_silence_ms):
        """Update VAD configuration"""
        
        try:
            method_enum = VADMethod(method)
            vad.update_config(
                method=method_enum,
                energy_threshold=float(energy_threshold),
                min_speech_duration_ms=int(min_speech_ms),
                min_silence_duration_ms=int(min_silence_ms)
            )
            return f"✅ VAD configuration updated"
        except Exception as e:
            return f"❌ Configuration update failed: {e}"
    
    # Available options
    method_choices = [method.value for method in VADMethod]
    
    return {
        "process_audio": process_audio_for_vad,
        "get_stats": get_vad_stats,
        "update_config": update_vad_config,
        "method_choices": method_choices,
        "vad": vad
    }