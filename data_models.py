#!/usr/bin/env python3
"""
Core Data Models for FireRedTTS2 RunPod Deployment
Comprehensive data models with validation and serialization support
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import torch

# Audio Data Models

@dataclass
class AudioChunk:
    """Represents a chunk of audio data with comprehensive metadata"""
    data: Union[np.ndarray, bytes]
    sample_rate: int
    channels: int = 1
    timestamp: float = field(default_factory=time.time)
    format: str = "wav"
    chunk_id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    duration_ms: Optional[float] = None
    quality_score: float = 1.0
    is_silence: bool = False
    energy_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.duration_ms is None and isinstance(self.data, np.ndarray):
            self.duration_ms = len(self.data) / self.sample_rate * 1000
        
        if isinstance(self.data, np.ndarray) and self.energy_level == 0.0:
            self.energy_level = float(np.sqrt(np.mean(self.data ** 2)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert numpy array to list for JSON serialization
        if isinstance(self.data, np.ndarray):
            result['data'] = self.data.tolist()
        return result
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate audio chunk data"""
        errors = []
        
        if self.sample_rate <= 0:
            errors.append("Sample rate must be positive")
        
        if self.channels <= 0:
            errors.append("Channels must be positive")
        
        if isinstance(self.data, np.ndarray):
            if len(self.data) == 0:
                errors.append("Audio data cannot be empty")
            if self.channels > 1 and len(self.data.shape) != 2:
                errors.append("Multi-channel audio must be 2D array")
        
        return len(errors) == 0, errors@dataclas
s
class AudioStream:
    """Represents a continuous audio stream"""
    stream_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunks: List[AudioChunk] = field(default_factory=list)
    total_duration_ms: float = 0.0
    is_complete: bool = False
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_chunk(self, chunk: AudioChunk):
        """Add audio chunk to stream"""
        self.chunks.append(chunk)
        self.total_duration_ms += chunk.duration_ms or 0
        
        # Update stream properties from chunk
        if not self.chunks or len(self.chunks) == 1:
            self.sample_rate = chunk.sample_rate
            self.channels = chunk.channels
            self.format = chunk.format
    
    def get_combined_audio(self) -> Optional[np.ndarray]:
        """Get combined audio data from all chunks"""
        if not self.chunks:
            return None
        
        audio_arrays = []
        for chunk in self.chunks:
            if isinstance(chunk.data, np.ndarray):
                audio_arrays.append(chunk.data)
        
        if audio_arrays:
            return np.concatenate(audio_arrays)
        return None
    
    def mark_complete(self):
        """Mark stream as complete"""
        self.is_complete = True
        self.end_time = time.time()

@dataclass
class VoiceProfile:
    """Voice profile for cloning with comprehensive metadata"""
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    reference_audio_path: str = ""
    reference_text: str = ""
    language: str = "English"
    gender: str = "Unknown"
    age_range: str = "Unknown"
    voice_characteristics: Dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    is_active: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        result['last_used'] = self.last_used.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceProfile':
        """Create from dictionary"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate voice profile"""
        errors = []
        
        if not self.name.strip():
            errors.append("Name is required")
        
        if not self.reference_audio_path or not Path(self.reference_audio_path).exists():
            errors.append("Valid reference audio path is required")
        
        if not self.reference_text.strip():
            errors.append("Reference text is required")
        
        if self.quality_score < 0 or self.quality_score > 1:
            errors.append("Quality score must be between 0 and 1")
        
        return len(errors) == 0, errors#
 Conversation Data Models

@dataclass
class ConversationTurn:
    """Enhanced conversation turn with comprehensive tracking"""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    user_input: str = ""
    system_response: str = ""
    audio_input: Optional[AudioChunk] = None
    audio_output: Optional[AudioChunk] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0
    user_type: str = "text"  # "text" or "audio"
    response_type: str = "audio"  # "text" or "audio"
    confidence_score: float = 1.0
    emotion_detected: str = "neutral"
    prosody_style: str = "conversational"
    language_detected: str = "English"
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        
        # Handle audio chunks
        if self.audio_input:
            result['audio_input'] = self.audio_input.to_dict()
        if self.audio_output:
            result['audio_output'] = self.audio_output.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Handle audio chunks
        if data.get('audio_input'):
            # Simplified - in production, properly reconstruct AudioChunk
            data['audio_input'] = None
        if data.get('audio_output'):
            data['audio_output'] = None
        
        return cls(**data)

@dataclass
class ConversationSession:
    """Enhanced conversation session with advanced features"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    turns: List[ConversationTurn] = field(default_factory=list)
    voice_profile: Optional[VoiceProfile] = None
    language: str = "English"
    response_style: str = "conversational"
    voice_mode: str = "consistent"
    personality: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    session_config: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_turn(self, turn: ConversationTurn):
        """Add a turn to the session"""
        turn.session_id = self.session_id
        self.turns.append(turn)
        self.last_activity = datetime.now()
        
        # Update performance metrics
        if turn.processing_time_ms > 0:
            if 'avg_processing_time_ms' not in self.performance_metrics:
                self.performance_metrics['avg_processing_time_ms'] = turn.processing_time_ms
            else:
                current_avg = self.performance_metrics['avg_processing_time_ms']
                turn_count = len(self.turns)
                self.performance_metrics['avg_processing_time_ms'] = (
                    (current_avg * (turn_count - 1) + turn.processing_time_ms) / turn_count
                )
    
    def get_recent_context(self, max_turns: int = 10) -> List[ConversationTurn]:
        """Get recent conversation context"""
        return self.turns[-max_turns:] if self.turns else []
    
    def get_session_duration(self) -> timedelta:
        """Get total session duration"""
        return self.last_activity - self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        result['last_activity'] = self.last_activity.isoformat()
        
        # Convert turns
        result['turns'] = [turn.to_dict() for turn in self.turns]
        
        # Convert voice profile
        if self.voice_profile:
            result['voice_profile'] = self.voice_profile.to_dict()
        
        return result