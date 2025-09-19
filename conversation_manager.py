#!/usr/bin/env python3
"""
Conversation Manager for Speech-to-Speech Interactions
Handles conversation flow, turn management, and context preservation
"""

import uuid
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    turn_id: str
    session_id: str
    user_input: str
    system_response: str
    audio_input_path: Optional[str] = None
    audio_output_path: Optional[str] = None
    timestamp: datetime = None
    processing_time_ms: float = 0.0
    user_type: str = "text"  # "text" or "audio"
    response_type: str = "audio"  # "text" or "audio"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ConversationSession:
    """Represents a complete conversation session"""
    session_id: str
    user_id: Optional[str]
    turns: List[ConversationTurn]
    voice_profile: Optional[Dict[str, Any]]
    language: str
    response_style: str
    voice_mode: str
    created_at: datetime
    last_activity: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class VoiceActivityDetector:
    """Simple Voice Activity Detection for conversation flow"""
    
    def __init__(self, 
                 energy_threshold: float = 0.01,
                 silence_duration: float = 1.0,
                 min_speech_duration: float = 0.5):
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.last_activity_time = 0
        self.speech_start_time = 0
        self.is_speaking = False
    
    def detect_activity(self, audio_chunk: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Detect voice activity in audio chunk
        
        Returns:
            Dict with activity status and timing information
        """
        current_time = time.time()
        
        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        activity_detected = energy > self.energy_threshold
        
        result = {
            "activity_detected": activity_detected,
            "energy": float(energy),
            "timestamp": current_time,
            "is_speaking": self.is_speaking,
            "speech_duration": 0.0,
            "silence_duration": 0.0,
            "turn_complete": False
        }
        
        if activity_detected:
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start_time = current_time
                logger.debug("Speech started")
            
            self.last_activity_time = current_time
            result["speech_duration"] = current_time - self.speech_start_time
            
        else:
            if self.is_speaking:
                # Check if silence duration exceeds threshold
                silence_time = current_time - self.last_activity_time
                result["silence_duration"] = silence_time
                
                if silence_time > self.silence_duration:
                    # Speech ended
                    speech_duration = self.last_activity_time - self.speech_start_time
                    
                    if speech_duration >= self.min_speech_duration:
                        result["turn_complete"] = True
                        logger.debug(f"Turn complete after {speech_duration:.2f}s of speech")
                    
                    self.is_speaking = False
                    self.speech_start_time = 0
        
        return result

class ConversationManager:
    """Manages conversation sessions and turn-based interactions"""
    
    def __init__(self, 
                 session_timeout_minutes: int = 30,
                 max_turns_per_session: int = 100,
                 storage_dir: str = "/workspace/sessions"):
        
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_turns_per_session = max_turns_per_session
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Active sessions in memory
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        # Voice activity detector
        self.vad = VoiceActivityDetector()
        
        # Load existing sessions
        self._load_sessions()
    
    def create_session(self, 
                      user_id: Optional[str] = None,
                      language: str = "English",
                      response_style: str = "conversational",
                      voice_mode: str = "consistent",
                      voice_profile: Optional[Dict[str, Any]] = None) -> str:
        """Create a new conversation session"""
        
        session_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            turns=[],
            voice_profile=voice_profile,
            language=language,
            response_style=response_style,
            voice_mode=voice_mode,
            created_at=current_time,
            last_activity=current_time
        )
        
        self.active_sessions[session_id] = session
        self._save_session(session)
        
        logger.info(f"Created new conversation session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session by ID"""
        
        # Check active sessions first
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Check if session has expired
            if datetime.now() - session.last_activity > self.session_timeout:
                self._expire_session(session_id)
                return None
            
            return session
        
        # Try to load from storage
        return self._load_session(session_id)
    
    def add_turn(self, 
                session_id: str,
                user_input: str,
                system_response: str,
                user_type: str = "text",
                response_type: str = "audio",
                audio_input_path: Optional[str] = None,
                audio_output_path: Optional[str] = None,
                processing_time_ms: float = 0.0,
                metadata: Optional[Dict[str, Any]] = None) -> Optional[ConversationTurn]:
        """Add a new turn to the conversation"""
        
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            return None
        
        # Check turn limit
        if len(session.turns) >= self.max_turns_per_session:
            logger.warning(f"Session {session_id} has reached maximum turns")
            return None
        
        # Create new turn
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            session_id=session_id,
            user_input=user_input,
            system_response=system_response,
            user_type=user_type,
            response_type=response_type,
            audio_input_path=audio_input_path,
            audio_output_path=audio_output_path,
            processing_time_ms=processing_time_ms,
            metadata=metadata or {}
        )
        
        # Add to session
        session.turns.append(turn)
        session.last_activity = datetime.now()
        
        # Update active session
        self.active_sessions[session_id] = session
        
        # Save to storage
        self._save_session(session)
        
        logger.info(f"Added turn {turn.turn_id} to session {session_id}")
        return turn
    
    def get_conversation_context(self, 
                               session_id: str, 
                               max_turns: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context for the session"""
        
        session = self.get_session(session_id)
        if not session:
            return []
        
        # Get recent turns
        recent_turns = session.turns[-max_turns:] if session.turns else []
        
        context = []
        for turn in recent_turns:
            context.append({
                "user_input": turn.user_input,
                "system_response": turn.system_response,
                "timestamp": turn.timestamp.isoformat(),
                "user_type": turn.user_type,
                "response_type": turn.response_type
            })
        
        return context
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the conversation session"""
        
        session = self.get_session(session_id)
        if not session:
            return None
        
        total_turns = len(session.turns)
        avg_processing_time = 0.0
        
        if total_turns > 0:
            total_processing_time = sum(turn.processing_time_ms for turn in session.turns)
            avg_processing_time = total_processing_time / total_turns
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "language": session.language,
            "response_style": session.response_style,
            "voice_mode": session.voice_mode,
            "total_turns": total_turns,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "duration_minutes": (session.last_activity - session.created_at).total_seconds() / 60,
            "avg_processing_time_ms": avg_processing_time,
            "is_active": session.is_active
        }
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active conversation sessions"""
        
        active_sessions = []
        current_time = datetime.now()
        
        for session_id, session in self.active_sessions.items():
            # Check if session is still active
            if current_time - session.last_activity <= self.session_timeout:
                summary = self.get_session_summary(session_id)
                if summary:
                    active_sessions.append(summary)
            else:
                # Mark as expired
                self._expire_session(session_id)
        
        return active_sessions
    
    def end_session(self, session_id: str) -> bool:
        """End a conversation session"""
        
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.is_active = False
        session.last_activity = datetime.now()
        
        # Update storage
        self._save_session(session)
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        logger.info(f"Ended conversation session: {session_id}")
        return True
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions"""
        
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self._expire_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)
    
    def detect_voice_activity(self, 
                            audio_chunk: np.ndarray, 
                            sample_rate: int) -> Dict[str, Any]:
        """Detect voice activity for conversation flow management"""
        return self.vad.detect_activity(audio_chunk, sample_rate)
    
    def _expire_session(self, session_id: str):
        """Mark a session as expired"""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.is_active = False
            self._save_session(session)
            del self.active_sessions[session_id]
            
            logger.info(f"Expired session: {session_id}")
    
    def _save_session(self, session: ConversationSession):
        """Save session to persistent storage"""
        
        try:
            session_file = self.storage_dir / f"{session.session_id}.json"
            
            # Convert to serializable format
            session_data = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "voice_profile": session.voice_profile,
                "language": session.language,
                "response_style": session.response_style,
                "voice_mode": session.voice_mode,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "is_active": session.is_active,
                "metadata": session.metadata,
                "turns": []
            }
            
            # Convert turns
            for turn in session.turns:
                turn_data = {
                    "turn_id": turn.turn_id,
                    "session_id": turn.session_id,
                    "user_input": turn.user_input,
                    "system_response": turn.system_response,
                    "audio_input_path": turn.audio_input_path,
                    "audio_output_path": turn.audio_output_path,
                    "timestamp": turn.timestamp.isoformat(),
                    "processing_time_ms": turn.processing_time_ms,
                    "user_type": turn.user_type,
                    "response_type": turn.response_type,
                    "metadata": turn.metadata
                }
                session_data["turns"].append(turn_data)
            
            # Save to file
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
    
    def _load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from persistent storage"""
        
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            
            if not session_file.exists():
                return None
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Convert back to objects
            turns = []
            for turn_data in session_data.get("turns", []):
                turn = ConversationTurn(
                    turn_id=turn_data["turn_id"],
                    session_id=turn_data["session_id"],
                    user_input=turn_data["user_input"],
                    system_response=turn_data["system_response"],
                    audio_input_path=turn_data.get("audio_input_path"),
                    audio_output_path=turn_data.get("audio_output_path"),
                    timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                    processing_time_ms=turn_data.get("processing_time_ms", 0.0),
                    user_type=turn_data.get("user_type", "text"),
                    response_type=turn_data.get("response_type", "audio"),
                    metadata=turn_data.get("metadata", {})
                )
                turns.append(turn)
            
            session = ConversationSession(
                session_id=session_data["session_id"],
                user_id=session_data.get("user_id"),
                turns=turns,
                voice_profile=session_data.get("voice_profile"),
                language=session_data.get("language", "English"),
                response_style=session_data.get("response_style", "conversational"),
                voice_mode=session_data.get("voice_mode", "consistent"),
                created_at=datetime.fromisoformat(session_data["created_at"]),
                last_activity=datetime.fromisoformat(session_data["last_activity"]),
                is_active=session_data.get("is_active", True),
                metadata=session_data.get("metadata", {})
            )
            
            # Add to active sessions if still valid
            if session.is_active and datetime.now() - session.last_activity <= self.session_timeout:
                self.active_sessions[session_id] = session
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def _load_sessions(self):
        """Load all existing sessions from storage"""
        
        try:
            session_files = list(self.storage_dir.glob("*.json"))
            loaded_count = 0
            
            for session_file in session_files:
                session_id = session_file.stem
                session = self._load_session(session_id)
                
                if session and session.is_active:
                    # Check if session is still within timeout
                    if datetime.now() - session.last_activity <= self.session_timeout:
                        loaded_count += 1
                    else:
                        # Mark as expired
                        session.is_active = False
                        self._save_session(session)
            
            logger.info(f"Loaded {loaded_count} active sessions from storage")
            
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")

# Global conversation manager instance
conversation_manager = None

def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager instance"""
    global conversation_manager
    if conversation_manager is None:
        conversation_manager = ConversationManager()
    return conversation_manager