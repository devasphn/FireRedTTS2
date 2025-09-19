#!/usr/bin/env python3
"""
Whisper ASR Integration for FireRedTTS2
Real-time speech recognition with streaming support and multi-language detection
"""

import os
import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """Result from speech recognition"""
    text: str
    confidence: float
    language: str
    start_time: float
    end_time: float
    segments: List[Dict[str, Any]]
    processing_time_ms: float
    model_name: str
    is_final: bool = True

@dataclass
class ASRConfig:
    """Configuration for ASR system"""
    model_size: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None  # Auto-detect if None
    task: str = "transcribe"  # transcribe or translate
    temperature: float = 0.0
    best_of: int = 5
    beam_size: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    suppress_tokens: str = "-1"
    initial_prompt: Optional[str] = None
    condition_on_previous_text: bool = True
    fp16: bool = True
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    word_timestamps: bool = False

class WhisperASR:
    """Whisper-based Automatic Speech Recognition"""
    
    def __init__(self, config: ASRConfig = None, device: str = "cuda"):
        self.config = config or ASRConfig()
        self.device = device
        self.model = None
        self.model_name = f"whisper-{self.config.model_size}"
        
        # Performance tracking
        self.total_transcriptions = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Whisper model"""
        try:
            import whisper
            
            logger.info(f"Loading Whisper model: {self.config.model_size}")
            start_time = time.time()
            
            self.model = whisper.load_model(
                self.config.model_size, 
                device=self.device
            )
            
            load_time = time.time() - start_time
            logger.info(f"Whisper model loaded in {load_time:.2f}s")
            
            # Test transcription to warm up the model
            self._warmup_model()
            
        except ImportError:
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _warmup_model(self):
        """Warm up the model with a dummy transcription"""
        try:
            # Create dummy audio (1 second of silence)
            dummy_audio = np.zeros(16000, dtype=np.float32)
            
            logger.info("Warming up Whisper model...")
            start_time = time.time()
            
            self.model.transcribe(
                dummy_audio,
                language=self.config.language,
                task=self.config.task,
                temperature=self.config.temperature,
                best_of=1,  # Use minimal settings for warmup
                beam_size=1,
                fp16=self.config.fp16
            )
            
            warmup_time = time.time() - start_time
            logger.info(f"Model warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def transcribe(self, 
                  audio_data: np.ndarray, 
                  sample_rate: int = 16000,
                  language: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Audio array (mono, float32)
            sample_rate: Sample rate of audio
            language: Language code (optional, auto-detect if None)
        
        Returns:
            TranscriptionResult with transcription and metadata
        """
        if self.model is None:
            raise RuntimeError("Whisper model not initialized")
        
        start_time = time.time()
        
        try:
            # Preprocess audio
            audio_processed = self._preprocess_audio(audio_data, sample_rate)
            
            # Prepare transcription options
            options = {
                "language": language or self.config.language,
                "task": self.config.task,
                "temperature": self.config.temperature,
                "best_of": self.config.best_of,
                "beam_size": self.config.beam_size,
                "patience": self.config.patience,
                "length_penalty": self.config.length_penalty,
                "suppress_tokens": self.config.suppress_tokens,
                "initial_prompt": self.config.initial_prompt,
                "condition_on_previous_text": self.config.condition_on_previous_text,
                "fp16": self.config.fp16,
                "compression_ratio_threshold": self.config.compression_ratio_threshold,
                "logprob_threshold": self.config.logprob_threshold,
                "no_speech_threshold": self.config.no_speech_threshold,
                "word_timestamps": self.config.word_timestamps
            }
            
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            
            # Transcribe
            result = self.model.transcribe(audio_processed, **options)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.total_transcriptions += 1
            self.total_processing_time += processing_time
            self.average_processing_time = self.total_processing_time / self.total_transcriptions
            
            # Extract segments information
            segments = []
            if "segments" in result:
                for segment in result["segments"]:
                    segments.append({
                        "id": segment.get("id", 0),
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                        "text": segment.get("text", ""),
                        "tokens": segment.get("tokens", []),
                        "temperature": segment.get("temperature", 0.0),
                        "avg_logprob": segment.get("avg_logprob", 0.0),
                        "compression_ratio": segment.get("compression_ratio", 0.0),
                        "no_speech_prob": segment.get("no_speech_prob", 0.0)
                    })
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(result, segments)
            
            return TranscriptionResult(
                text=result["text"].strip(),
                confidence=confidence,
                language=result.get("language", "unknown"),
                start_time=0.0,
                end_time=len(audio_processed) / 16000,
                segments=segments,
                processing_time_ms=processing_time,
                model_name=self.model_name,
                is_final=True
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            
            # Return empty result on failure
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language="unknown",
                start_time=0.0,
                end_time=0.0,
                segments=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                model_name=self.model_name,
                is_final=True
            )
    
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess audio for Whisper"""
        
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Convert to float32
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            # Use torchaudio for high-quality resampling
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_tensor = resampler(audio_tensor)
            audio_data = audio_tensor.squeeze(0).numpy()
        
        # Normalize audio
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Pad or trim to reasonable length (Whisper works best with 30s chunks)
        max_length = 30 * 16000  # 30 seconds
        if len(audio_data) > max_length:
            audio_data = audio_data[:max_length]
        
        return audio_data
    
    def _calculate_confidence(self, result: Dict[str, Any], segments: List[Dict[str, Any]]) -> float:
        """Calculate confidence score from Whisper result"""
        
        # If no segments, use a default low confidence
        if not segments:
            return 0.5
        
        # Calculate average log probability
        total_logprob = 0.0
        total_tokens = 0
        
        for segment in segments:
            avg_logprob = segment.get("avg_logprob", -1.0)
            tokens = segment.get("tokens", [])
            
            if tokens:
                total_logprob += avg_logprob * len(tokens)
                total_tokens += len(tokens)
        
        if total_tokens == 0:
            return 0.5
        
        avg_logprob = total_logprob / total_tokens
        
        # Convert log probability to confidence (0-1)
        # This is a heuristic mapping
        confidence = np.exp(avg_logprob)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return float(confidence)
    
    def detect_language(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Detect the language of the audio"""
        
        if self.model is None:
            raise RuntimeError("Whisper model not initialized")
        
        try:
            # Preprocess audio
            audio_processed = self._preprocess_audio(audio_data, sample_rate)
            
            # Use Whisper's language detection
            # Load audio into Whisper format
            audio_tensor = torch.from_numpy(audio_processed)
            if self.device == "cuda" and torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
            
            # Detect language
            _, probs = self.model.detect_language(audio_tensor)
            
            # Get top languages
            top_languages = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "detected_language": top_languages[0][0],
                "confidence": top_languages[0][1],
                "top_languages": top_languages
            }
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                "detected_language": "en",
                "confidence": 0.0,
                "top_languages": [("en", 0.0)]
            }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        try:
            import whisper
            return list(whisper.tokenizer.LANGUAGES.keys())
        except:
            # Fallback list of common languages
            return [
                "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
                "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
                "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
                "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
                "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
                "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
                "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
                "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
            ]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "model_name": self.model_name,
            "total_transcriptions": self.total_transcriptions,
            "average_processing_time_ms": self.average_processing_time,
            "total_processing_time_ms": self.total_processing_time,
            "device": self.device,
            "model_loaded": self.model is not None
        }

class StreamingWhisperASR:
    """Streaming version of Whisper ASR for real-time transcription"""
    
    def __init__(self, 
                 config: ASRConfig = None,
                 device: str = "cuda",
                 chunk_duration: float = 2.0,
                 overlap_duration: float = 0.5):
        
        self.config = config or ASRConfig()
        self.device = device
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        
        # Audio buffer for streaming
        self.audio_buffer = deque()
        self.sample_rate = 16000
        self.chunk_size = int(chunk_duration * self.sample_rate)
        self.overlap_size = int(overlap_duration * self.sample_rate)
        
        # Streaming state
        self.is_streaming = False
        self.stream_thread = None
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Initialize base ASR
        self.asr = WhisperASR(config, device)
        
        # Previous context for better continuity
        self.previous_text = ""
        self.context_window = 5  # Keep last 5 transcriptions for context
        self.context_history = deque(maxlen=self.context_window)
    
    def start_streaming(self):
        """Start streaming transcription"""
        if not self.is_streaming:
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self._streaming_loop, daemon=True)
            self.stream_thread.start()
            logger.info("Started streaming ASR")
    
    def stop_streaming(self):
        """Stop streaming transcription"""
        if self.is_streaming:
            self.is_streaming = False
            if self.stream_thread:
                self.stream_thread.join(timeout=2.0)
            logger.info("Stopped streaming ASR")
    
    def add_audio_chunk(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Add audio chunk to the streaming buffer"""
        if not self.is_streaming:
            return
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            audio_data = resampler(audio_tensor).squeeze(0).numpy()
        
        # Add to queue
        try:
            self.audio_queue.put(audio_data, timeout=0.1)
        except queue.Full:
            logger.warning("Audio queue full, dropping chunk")
    
    def get_transcription_result(self, timeout: float = 0.1) -> Optional[TranscriptionResult]:
        """Get the latest transcription result"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _streaming_loop(self):
        """Main streaming transcription loop"""
        
        while self.is_streaming:
            try:
                # Get audio chunk from queue
                try:
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Add to buffer
                self.audio_buffer.extend(audio_chunk)
                
                # Check if we have enough audio for transcription
                if len(self.audio_buffer) >= self.chunk_size:
                    # Extract chunk with overlap
                    chunk_data = np.array(list(self.audio_buffer)[:self.chunk_size])
                    
                    # Remove processed audio (keeping overlap)
                    for _ in range(self.chunk_size - self.overlap_size):
                        if self.audio_buffer:
                            self.audio_buffer.popleft()
                    
                    # Transcribe chunk
                    result = self._transcribe_chunk(chunk_data)
                    
                    if result and result.text.strip():
                        # Add to result queue
                        try:
                            self.result_queue.put(result, timeout=0.1)
                        except queue.Full:
                            # Remove old result and add new one
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put(result, timeout=0.1)
                            except queue.Empty:
                                pass
                
            except Exception as e:
                logger.error(f"Streaming transcription error: {e}")
                time.sleep(0.1)
    
    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> Optional[TranscriptionResult]:
        """Transcribe a single audio chunk"""
        
        try:
            # Use context from previous transcriptions
            initial_prompt = None
            if self.context_history:
                # Use last few transcriptions as context
                context_texts = [ctx.text for ctx in self.context_history if ctx.text.strip()]
                if context_texts:
                    initial_prompt = " ".join(context_texts[-2:])  # Last 2 transcriptions
            
            # Temporarily set initial prompt
            original_prompt = self.asr.config.initial_prompt
            if initial_prompt:
                self.asr.config.initial_prompt = initial_prompt
            
            # Transcribe
            result = self.asr.transcribe(audio_chunk, self.sample_rate)
            
            # Restore original prompt
            self.asr.config.initial_prompt = original_prompt
            
            # Mark as non-final for streaming
            result.is_final = False
            
            # Add to context history
            if result.text.strip():
                self.context_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Chunk transcription failed: {e}")
            return None

def create_asr_interface(device: str = "cuda") -> Tuple[WhisperASR, StreamingWhisperASR, Dict[str, Any]]:
    """Create ASR interface components"""
    
    # Initialize ASR systems
    config = ASRConfig(model_size="base", fp16=True)
    asr = WhisperASR(config, device)
    streaming_asr = StreamingWhisperASR(config, device)
    
    def transcribe_audio(audio_data, language=None):
        """Transcribe audio file"""
        if audio_data is None:
            return "❌ No audio provided", "", 0.0, ""
        
        try:
            sample_rate, audio_array = audio_data
            
            # Transcribe
            result = asr.transcribe(audio_array, sample_rate, language)
            
            # Format result
            status = f"✅ Transcribed in {result.processing_time_ms:.0f}ms"
            confidence_text = f"Confidence: {result.confidence:.2f}"
            language_text = f"Language: {result.language}"
            
            return status, result.text, result.confidence, f"{confidence_text} | {language_text}"
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return f"❌ Transcription failed: {e}", "", 0.0, ""
    
    def detect_language(audio_data):
        """Detect language of audio"""
        if audio_data is None:
            return "❌ No audio provided", []
        
        try:
            sample_rate, audio_array = audio_data
            
            # Detect language
            result = asr.detect_language(audio_array, sample_rate)
            
            # Format results
            status = f"✅ Detected language: {result['detected_language']} (confidence: {result['confidence']:.2f})"
            
            # Format top languages for display
            top_langs = []
            for lang, conf in result['top_languages'][:5]:
                top_langs.append([lang, f"{conf:.3f}"])
            
            return status, top_langs
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return f"❌ Language detection failed: {e}", []
    
    def start_streaming():
        """Start streaming transcription"""
        streaming_asr.start_streaming()
        return "✅ Streaming started"
    
    def stop_streaming():
        """Stop streaming transcription"""
        streaming_asr.stop_streaming()
        return "✅ Streaming stopped"
    
    def get_performance_stats():
        """Get ASR performance statistics"""
        stats = asr.get_performance_stats()
        return {
            "Model": stats["model_name"],
            "Total Transcriptions": stats["total_transcriptions"],
            "Avg Processing Time (ms)": f"{stats['average_processing_time_ms']:.1f}",
            "Device": stats["device"],
            "Model Loaded": "✅" if stats["model_loaded"] else "❌"
        }
    
    # Get supported languages
    supported_languages = asr.get_supported_languages()
    language_choices = ["Auto-detect"] + [f"{lang} ({lang})" for lang in supported_languages[:20]]
    
    return asr, streaming_asr, {
        "transcribe": transcribe_audio,
        "detect_language": detect_language,
        "start_streaming": start_streaming,
        "stop_streaming": stop_streaming,
        "get_stats": get_performance_stats,
        "language_choices": language_choices,
        "supported_languages": supported_languages
    }