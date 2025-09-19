#!/usr/bin/env python3
"""
Speech-to-Speech Interface Components
Handles real-time conversation flow with audio input/output
"""

import time
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
import gradio as gr

from conversation_manager import get_conversation_manager, ConversationTurn
from fireredtts2.fireredtts2 import FireRedTTS2

logger = logging.getLogger(__name__)

@dataclass
class SpeechToSpeechConfig:
    """Configuration for speech-to-speech interface"""
    enable_asr: bool = True
    enable_llm: bool = True
    enable_tts: bool = True
    asr_model: str = "whisper-base"
    llm_model: str = "local"  # or "openai", "anthropic"
    voice_mode: str = "consistent"
    language: str = "English"
    response_style: str = "conversational"
    max_audio_length: float = 30.0  # seconds
    vad_threshold: float = 0.01
    silence_timeout: float = 2.0

class ASRProcessor:
    """Automatic Speech Recognition processor"""
    
    def __init__(self, model_name: str = "whisper-base"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ASR model"""
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("ASR model loaded successfully")
        except ImportError:
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            self.model = None
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Returns:
            Dict with transcription results
        """
        if self.model is None:
            return {
                "text": "",
                "confidence": 0.0,
                "language": "en",
                "error": "ASR model not available"
            }
        
        try:
            # Ensure audio is in the right format
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                # Simple resampling (for production, use librosa)
                duration = len(audio_data) / sample_rate
                target_length = int(duration * 16000)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), target_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Transcribe
            result = self.model.transcribe(audio_data, language=None)
            
            return {
                "text": result["text"].strip(),
                "confidence": 1.0,  # Whisper doesn't provide confidence scores
                "language": result.get("language", "en"),
                "segments": result.get("segments", []),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": "en",
                "error": str(e)
            }

class LLMProcessor:
    """Large Language Model processor for conversation"""
    
    def __init__(self, model_type: str = "local"):
        self.model_type = model_type
        self.model = None
        self.conversation_history = []
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LLM model"""
        if self.model_type == "local":
            # For now, use a simple rule-based response
            # In production, integrate with local LLM like Llama
            logger.info("Using simple rule-based conversation model")
            self.model = "rule_based"
        elif self.model_type == "openai":
            try:
                import openai
                # Initialize OpenAI client
                logger.info("Using OpenAI GPT model")
                self.model = "openai"
            except ImportError:
                logger.error("OpenAI package not installed")
                self.model = None
        else:
            logger.warning(f"Unknown model type: {self.model_type}")
            self.model = None
    
    def generate_response(self, 
                         user_input: str, 
                         conversation_context: List[Dict[str, Any]] = None,
                         response_style: str = "conversational") -> Dict[str, Any]:
        """
        Generate a conversational response
        
        Returns:
            Dict with response and metadata
        """
        if not user_input.strip():
            return {
                "response": "I didn't catch that. Could you please repeat?",
                "confidence": 1.0,
                "error": None
            }
        
        try:
            if self.model == "rule_based":
                response = self._generate_rule_based_response(user_input, response_style)
            elif self.model == "openai":
                response = self._generate_openai_response(user_input, conversation_context, response_style)
            else:
                response = "I'm sorry, I'm having trouble processing your request right now."
            
            return {
                "response": response,
                "confidence": 1.0,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return {
                "response": "I apologize, but I'm experiencing some technical difficulties.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _generate_rule_based_response(self, user_input: str, style: str) -> str:
        """Generate a simple rule-based response"""
        
        user_input_lower = user_input.lower()
        
        # Greeting responses
        if any(word in user_input_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return "Hello! How can I help you today?"
        
        # Question responses
        if user_input_lower.startswith(("what", "how", "why", "when", "where", "who")):
            return f"That's an interesting question about {user_input[4:]}. Let me think about that."
        
        # Goodbye responses
        if any(word in user_input_lower for word in ["goodbye", "bye", "see you", "farewell"]):
            return "Goodbye! It was nice talking with you."
        
        # Thank you responses
        if any(word in user_input_lower for word in ["thank", "thanks"]):
            return "You're welcome! Is there anything else I can help you with?"
        
        # Default responses based on style
        if style == "formal":
            return f"I understand you mentioned: '{user_input}'. How may I assist you further?"
        elif style == "casual":
            return f"Got it! You said '{user_input}'. What's next?"
        elif style == "professional":
            return f"Thank you for sharing that information. Regarding '{user_input}', how can I provide assistance?"
        else:  # conversational
            return f"Interesting! You mentioned '{user_input}'. Tell me more about that."
    
    def _generate_openai_response(self, 
                                 user_input: str, 
                                 context: List[Dict[str, Any]], 
                                 style: str) -> str:
        """Generate response using OpenAI API (placeholder)"""
        # This would integrate with OpenAI API in production
        return f"I heard you say: '{user_input}'. This is a placeholder for OpenAI integration."

class SpeechToSpeechInterface:
    """Main interface for speech-to-speech conversation"""
    
    def __init__(self, 
                 tts_model: FireRedTTS2,
                 config: SpeechToSpeechConfig = None):
        
        self.tts_model = tts_model
        self.config = config or SpeechToSpeechConfig()
        
        # Initialize processors
        self.asr_processor = ASRProcessor(self.config.asr_model) if self.config.enable_asr else None
        self.llm_processor = LLMProcessor(self.config.llm_model) if self.config.enable_llm else None
        
        # Conversation manager
        self.conversation_manager = get_conversation_manager()
        
        # Current session
        self.current_session_id = None
        
        # Audio processing
        self.audio_buffer = []
        self.is_recording = False
        self.processing_lock = threading.Lock()
    
    def start_new_conversation(self, 
                             language: str = None,
                             response_style: str = None,
                             voice_mode: str = None) -> str:
        """Start a new conversation session"""
        
        session_id = self.conversation_manager.create_session(
            language=language or self.config.language,
            response_style=response_style or self.config.response_style,
            voice_mode=voice_mode or self.config.voice_mode
        )
        
        self.current_session_id = session_id
        logger.info(f"Started new conversation session: {session_id}")
        
        return session_id
    
    def process_audio_input(self, audio_data: Tuple[int, np.ndarray]) -> Dict[str, Any]:
        """Process audio input through the speech-to-speech pipeline"""
        
        if not self.current_session_id:
            self.start_new_conversation()
        
        start_time = time.time()
        
        try:
            sample_rate, audio_array = audio_data
            
            # Step 1: ASR - Convert speech to text
            if self.asr_processor:
                asr_result = self.asr_processor.transcribe(audio_array, sample_rate)
                user_text = asr_result["text"]
                
                if not user_text.strip():
                    return {
                        "success": False,
                        "error": "No speech detected",
                        "user_text": "",
                        "system_response": "",
                        "audio_output": None
                    }
            else:
                return {
                    "success": False,
                    "error": "ASR not available",
                    "user_text": "",
                    "system_response": "",
                    "audio_output": None
                }
            
            # Step 2: LLM - Generate response
            if self.llm_processor:
                conversation_context = self.conversation_manager.get_conversation_context(
                    self.current_session_id, max_turns=5
                )
                
                session = self.conversation_manager.get_session(self.current_session_id)
                response_style = session.response_style if session else "conversational"
                
                llm_result = self.llm_processor.generate_response(
                    user_text, conversation_context, response_style
                )
                system_response = llm_result["response"]
            else:
                system_response = f"I heard you say: {user_text}"
            
            # Step 3: TTS - Convert response to speech
            if self.config.enable_tts and self.tts_model:
                # Generate audio response
                audio_tensor = self.tts_model.generate_monologue(
                    text=system_response,
                    temperature=0.9,
                    topk=30
                )
                
                # Convert to output format
                if isinstance(audio_tensor, torch.Tensor):
                    audio_output = (24000, audio_tensor.squeeze().cpu().numpy())
                else:
                    audio_output = (24000, audio_tensor)
            else:
                audio_output = None
            
            # Record the conversation turn
            processing_time = (time.time() - start_time) * 1000
            
            turn = self.conversation_manager.add_turn(
                session_id=self.current_session_id,
                user_input=user_text,
                system_response=system_response,
                user_type="audio",
                response_type="audio",
                processing_time_ms=processing_time
            )
            
            return {
                "success": True,
                "error": None,
                "user_text": user_text,
                "system_response": system_response,
                "audio_output": audio_output,
                "processing_time_ms": processing_time,
                "turn_id": turn.turn_id if turn else None
            }
            
        except Exception as e:
            logger.error(f"Speech-to-speech processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_text": "",
                "system_response": "",
                "audio_output": None
            }
    
    def process_text_input(self, text_input: str) -> Dict[str, Any]:
        """Process text input through the conversation pipeline"""
        
        if not self.current_session_id:
            self.start_new_conversation()
        
        start_time = time.time()
        
        try:
            if not text_input.strip():
                return {
                    "success": False,
                    "error": "Empty text input",
                    "user_text": "",
                    "system_response": "",
                    "audio_output": None
                }
            
            # Step 1: LLM - Generate response
            if self.llm_processor:
                conversation_context = self.conversation_manager.get_conversation_context(
                    self.current_session_id, max_turns=5
                )
                
                session = self.conversation_manager.get_session(self.current_session_id)
                response_style = session.response_style if session else "conversational"
                
                llm_result = self.llm_processor.generate_response(
                    text_input, conversation_context, response_style
                )
                system_response = llm_result["response"]
            else:
                system_response = f"You said: {text_input}"
            
            # Step 2: TTS - Convert response to speech
            if self.config.enable_tts and self.tts_model:
                audio_tensor = self.tts_model.generate_monologue(
                    text=system_response,
                    temperature=0.9,
                    topk=30
                )
                
                if isinstance(audio_tensor, torch.Tensor):
                    audio_output = (24000, audio_tensor.squeeze().cpu().numpy())
                else:
                    audio_output = (24000, audio_tensor)
            else:
                audio_output = None
            
            # Record the conversation turn
            processing_time = (time.time() - start_time) * 1000
            
            turn = self.conversation_manager.add_turn(
                session_id=self.current_session_id,
                user_input=text_input,
                system_response=system_response,
                user_type="text",
                response_type="audio",
                processing_time_ms=processing_time
            )
            
            return {
                "success": True,
                "error": None,
                "user_text": text_input,
                "system_response": system_response,
                "audio_output": audio_output,
                "processing_time_ms": processing_time,
                "turn_id": turn.turn_id if turn else None
            }
            
        except Exception as e:
            logger.error(f"Text-to-speech processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_text": text_input,
                "system_response": "",
                "audio_output": None
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history"""
        
        if not self.current_session_id:
            return []
        
        return self.conversation_manager.get_conversation_context(
            self.current_session_id, max_turns=20
        )
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session"""
        
        if not self.current_session_id:
            return {"session_active": False}
        
        summary = self.conversation_manager.get_session_summary(self.current_session_id)
        if summary:
            summary["session_active"] = True
            return summary
        else:
            return {"session_active": False}
    
    def end_conversation(self) -> bool:
        """End the current conversation"""
        
        if self.current_session_id:
            success = self.conversation_manager.end_session(self.current_session_id)
            if success:
                self.current_session_id = None
            return success
        
        return True

def create_speech_to_speech_components(tts_model: FireRedTTS2) -> Tuple[gr.Interface, SpeechToSpeechInterface]:
    """Create Gradio components for speech-to-speech interface"""
    
    # Initialize the speech-to-speech interface
    s2s_interface = SpeechToSpeechInterface(tts_model)
    
    def process_conversation_audio(audio_input):
        """Process audio input for conversation"""
        if audio_input is None:
            return None, "No audio input received", ""
        
        result = s2s_interface.process_audio_input(audio_input)
        
        if result["success"]:
            return (
                result["audio_output"],
                f"You: {result['user_text']}\nAssistant: {result['system_response']}",
                result["user_text"]
            )
        else:
            return (
                None,
                f"Error: {result['error']}",
                ""
            )
    
    def process_conversation_text(text_input):
        """Process text input for conversation"""
        if not text_input.strip():
            return None, "No text input provided", ""
        
        result = s2s_interface.process_text_input(text_input)
        
        if result["success"]:
            return (
                result["audio_output"],
                f"You: {result['user_text']}\nAssistant: {result['system_response']}",
                ""  # Clear text input
            )
        else:
            return (
                None,
                f"Error: {result['error']}",
                text_input
            )
    
    def start_new_conversation():
        """Start a new conversation session"""
        session_id = s2s_interface.start_new_conversation()
        return f"Started new conversation: {session_id}", []
    
    def get_conversation_history():
        """Get conversation history for display"""
        history = s2s_interface.get_conversation_history()
        return history
    
    return s2s_interface, {
        "process_audio": process_conversation_audio,
        "process_text": process_conversation_text,
        "start_new": start_new_conversation,
        "get_history": get_conversation_history
    }