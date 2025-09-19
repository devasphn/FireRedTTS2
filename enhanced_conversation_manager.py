#!/usr/bin/env python3
"""
Enhanced Conversation Manager
Advanced conversation session management with LLM integration and context optimization
"""

import uuid
import time
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

import numpy as np

from conversation_llm import ConversationLLMManager, ConversationContext, LLMResponse, get_llm_manager
from conversation_manager import ConversationTurn, ConversationSession, VoiceActivityDetector

logger = logging.getLogger(__name__)

@dataclass
class PersonalityProfile:
    """AI personality configuration"""
    name: str
    description: str
    system_prompt: str
    response_style: str
    personality_traits: Dict[str, float]  # e.g., {"friendliness": 0.8, "formality": 0.3}
    preferred_topics: List[str]
    conversation_starters: List[str]
    created_at: datetime
    usage_count: int = 0

@dataclass
class ConversationMetrics:
    """Metrics for conversation quality and engagement"""
    session_id: str
    total_turns: int
    avg_response_time_ms: float
    avg_response_length: int
    user_satisfaction_score: Optional[float]
    engagement_score: float
    topic_coherence_score: float
    conversation_flow_score: float
    total_duration_minutes: float
    last_updated: datetime

@dataclass
class ContextWindow:
    """Sliding context window for conversation"""
    max_turns: int
    max_tokens: int
    current_turns: List[ConversationTurn]
    current_tokens: int
    summary: Optional[str] = None
    key_topics: List[str] = None

class ConversationAnalyzer:
    """Analyzes conversation quality and engagement"""
    
    def __init__(self):
        self.sentiment_keywords = {
            "positive": ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like", "happy", "pleased"],
            "negative": ["bad", "terrible", "awful", "hate", "dislike", "angry", "frustrated", "disappointed", "sad"],
            "neutral": ["okay", "fine", "alright", "maybe", "perhaps", "possibly"]
        }
    
    def analyze_conversation(self, session: ConversationSession) -> ConversationMetrics:
        """Analyze conversation quality and engagement"""
        
        if not session.turns:
            return ConversationMetrics(
                session_id=session.session_id,
                total_turns=0,
                avg_response_time_ms=0.0,
                avg_response_length=0,
                user_satisfaction_score=None,
                engagement_score=0.0,
                topic_coherence_score=0.0,
                conversation_flow_score=0.0,
                total_duration_minutes=0.0,
                last_updated=datetime.now()
            )
        
        # Calculate basic metrics
        total_turns = len(session.turns)
        response_times = [turn.processing_time_ms for turn in session.turns if turn.processing_time_ms > 0]
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        response_lengths = [len(turn.system_response.split()) for turn in session.turns]
        avg_response_length = int(np.mean(response_lengths)) if response_lengths else 0
        
        # Calculate duration
        if session.turns:
            start_time = session.turns[0].timestamp
            end_time = session.turns[-1].timestamp
            duration_minutes = (end_time - start_time).total_seconds() / 60
        else:
            duration_minutes = 0.0
        
        # Analyze engagement
        engagement_score = self._calculate_engagement_score(session)
        
        # Analyze topic coherence
        topic_coherence_score = self._calculate_topic_coherence(session)
        
        # Analyze conversation flow
        flow_score = self._calculate_conversation_flow(session)
        
        # Estimate user satisfaction
        satisfaction_score = self._estimate_user_satisfaction(session)
        
        return ConversationMetrics(
            session_id=session.session_id,
            total_turns=total_turns,
            avg_response_time_ms=avg_response_time,
            avg_response_length=avg_response_length,
            user_satisfaction_score=satisfaction_score,
            engagement_score=engagement_score,
            topic_coherence_score=topic_coherence_score,
            conversation_flow_score=flow_score,
            total_duration_minutes=duration_minutes,
            last_updated=datetime.now()
        )
    
    def _calculate_engagement_score(self, session: ConversationSession) -> float:
        """Calculate engagement score based on conversation patterns"""
        
        if not session.turns:
            return 0.0
        
        factors = []
        
        # Turn frequency (more turns = higher engagement)
        duration_hours = (session.last_activity - session.created_at).total_seconds() / 3600
        if duration_hours > 0:
            turns_per_hour = len(session.turns) / duration_hours
            turn_frequency_score = min(turns_per_hour / 10, 1.0)  # Normalize to 0-1
            factors.append(turn_frequency_score)
        
        # Response length variation (varied responses = higher engagement)
        response_lengths = [len(turn.user_input.split()) for turn in session.turns]
        if len(response_lengths) > 1:
            length_variation = np.std(response_lengths) / (np.mean(response_lengths) + 1)
            variation_score = min(length_variation, 1.0)
            factors.append(variation_score)
        
        # Question asking (questions indicate engagement)
        questions = sum(1 for turn in session.turns if '?' in turn.user_input)
        question_ratio = questions / len(session.turns)
        factors.append(question_ratio)
        
        # Sentiment analysis (positive sentiment = higher engagement)
        sentiment_score = self._analyze_sentiment(session)
        factors.append(sentiment_score)
        
        return np.mean(factors) if factors else 0.0
    
    def _calculate_topic_coherence(self, session: ConversationSession) -> float:
        """Calculate topic coherence score"""
        
        if len(session.turns) < 2:
            return 1.0
        
        # Simple keyword-based coherence
        all_text = " ".join([turn.user_input + " " + turn.system_response for turn in session.turns])
        words = all_text.lower().split()
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only consider meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate coherence based on repeated keywords
        if not word_freq:
            return 0.5
        
        total_words = len(words)
        repeated_words = sum(1 for freq in word_freq.values() if freq > 1)
        coherence_score = repeated_words / len(word_freq) if word_freq else 0.0
        
        return min(coherence_score * 2, 1.0)  # Scale and cap at 1.0
    
    def _calculate_conversation_flow(self, session: ConversationSession) -> float:
        """Calculate conversation flow score"""
        
        if len(session.turns) < 2:
            return 1.0
        
        factors = []
        
        # Response time consistency
        response_times = [turn.processing_time_ms for turn in session.turns if turn.processing_time_ms > 0]
        if len(response_times) > 1:
            time_consistency = 1.0 - (np.std(response_times) / (np.mean(response_times) + 1))
            factors.append(max(time_consistency, 0.0))
        
        # Turn length balance
        user_lengths = [len(turn.user_input.split()) for turn in session.turns]
        system_lengths = [len(turn.system_response.split()) for turn in session.turns]
        
        if user_lengths and system_lengths:
            avg_user_length = np.mean(user_lengths)
            avg_system_length = np.mean(system_lengths)
            
            # Ideal ratio is around 1:1 to 1:2 (system slightly longer)
            ratio = avg_system_length / (avg_user_length + 1)
            balance_score = 1.0 - abs(ratio - 1.5) / 1.5
            factors.append(max(balance_score, 0.0))
        
        return np.mean(factors) if factors else 0.5
    
    def _analyze_sentiment(self, session: ConversationSession) -> float:
        """Analyze overall sentiment of the conversation"""
        
        sentiment_scores = []
        
        for turn in session.turns:
            text = turn.user_input.lower()
            
            positive_count = sum(1 for word in self.sentiment_keywords["positive"] if word in text)
            negative_count = sum(1 for word in self.sentiment_keywords["negative"] if word in text)
            
            if positive_count + negative_count > 0:
                sentiment = positive_count / (positive_count + negative_count)
                sentiment_scores.append(sentiment)
        
        return np.mean(sentiment_scores) if sentiment_scores else 0.5
    
    def _estimate_user_satisfaction(self, session: ConversationSession) -> Optional[float]:
        """Estimate user satisfaction based on conversation patterns"""
        
        if not session.turns:
            return None
        
        # Look for explicit satisfaction indicators
        satisfaction_indicators = {
            "positive": ["thank", "thanks", "great", "perfect", "excellent", "helpful", "good job"],
            "negative": ["bad", "wrong", "unhelpful", "frustrated", "disappointed"]
        }
        
        positive_signals = 0
        negative_signals = 0
        
        for turn in session.turns:
            text = turn.user_input.lower()
            
            for word in satisfaction_indicators["positive"]:
                if word in text:
                    positive_signals += 1
            
            for word in satisfaction_indicators["negative"]:
                if word in text:
                    negative_signals += 1
        
        if positive_signals + negative_signals == 0:
            return None
        
        satisfaction = positive_signals / (positive_signals + negative_signals)
        return satisfaction

class ContextOptimizer:
    """Optimizes conversation context for LLM processing"""
    
    def __init__(self, max_context_tokens: int = 4000):
        self.max_context_tokens = max_context_tokens
    
    def optimize_context(self, 
                        session: ConversationSession,
                        current_input: str) -> ConversationContext:
        """Optimize conversation context for current input"""
        
        # Start with recent turns
        context_turns = []
        total_tokens = self._estimate_tokens(current_input)
        
        # Add turns in reverse order (most recent first)
        for turn in reversed(session.turns):
            turn_tokens = self._estimate_tokens(turn.user_input + turn.system_response)
            
            if total_tokens + turn_tokens <= self.max_context_tokens:
                context_turns.insert(0, {
                    "user_input": turn.user_input,
                    "system_response": turn.system_response
                })
                total_tokens += turn_tokens
            else:
                break
        
        # Create conversation context
        context = ConversationContext(
            conversation_history=context_turns,
            language=session.language,
            response_style=session.response_style,
            max_context_length=self.max_context_tokens,
            temperature=0.7,
            max_tokens=150
        )
        
        # Add system persona based on session metadata
        if session.metadata and "personality" in session.metadata:
            context.system_persona = session.metadata["personality"]
        
        return context
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def create_context_summary(self, turns: List[ConversationTurn]) -> str:
        """Create a summary of conversation turns"""
        
        if not turns:
            return ""
        
        # Extract key topics and themes
        all_text = " ".join([turn.user_input + " " + turn.system_response for turn in turns])
        
        # Simple extractive summary (in production, use proper summarization)
        sentences = all_text.split('.')
        important_sentences = []
        
        # Look for sentences with questions or key information
        for sentence in sentences[:10]:  # Limit to first 10 sentences
            sentence = sentence.strip()
            if len(sentence) > 20 and ('?' in sentence or any(word in sentence.lower() for word in ['important', 'key', 'main', 'primary'])):
                important_sentences.append(sentence)
        
        summary = '. '.join(important_sentences[:3])  # Top 3 sentences
        return summary if summary else "General conversation"

class EnhancedConversationManager:
    """Enhanced conversation manager with LLM integration"""
    
    def __init__(self, 
                 storage_dir: str = "/workspace/sessions",
                 llm_manager: Optional[ConversationLLMManager] = None,
                 session_timeout_minutes: int = 30,
                 max_turns_per_session: int = 100):
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.llm_manager = llm_manager or get_llm_manager()
        self.analyzer = ConversationAnalyzer()
        self.context_optimizer = ContextOptimizer()
        
        # Session management
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_turns_per_session = max_turns_per_session
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        # Personality profiles
        self.personality_profiles: Dict[str, PersonalityProfile] = {}
        self._load_default_personalities()
        
        # Metrics storage
        self.session_metrics: Dict[str, ConversationMetrics] = {}
        
        # Load existing sessions
        self._load_sessions()
    
    def _load_default_personalities(self):
        """Load default personality profiles"""
        
        default_personalities = [
            PersonalityProfile(
                name="Friendly Assistant",
                description="A warm, helpful, and conversational AI assistant",
                system_prompt="You are a friendly and helpful assistant. Respond in a warm, conversational tone. Be supportive and encouraging while providing helpful information.",
                response_style="conversational",
                personality_traits={"friendliness": 0.9, "formality": 0.3, "enthusiasm": 0.7},
                preferred_topics=["general help", "learning", "problem solving"],
                conversation_starters=["How can I help you today?", "What would you like to talk about?"],
                created_at=datetime.now()
            ),
            PersonalityProfile(
                name="Professional Expert",
                description="A knowledgeable and professional AI consultant",
                system_prompt="You are a professional expert assistant. Provide authoritative, well-structured responses with detailed explanations. Maintain a professional but approachable tone.",
                response_style="professional",
                personality_traits={"friendliness": 0.6, "formality": 0.8, "expertise": 0.9},
                preferred_topics=["business", "technology", "analysis", "consulting"],
                conversation_starters=["How may I assist you today?", "What expertise do you need?"],
                created_at=datetime.now()
            ),
            PersonalityProfile(
                name="Casual Friend",
                description="A relaxed, casual, and fun conversational partner",
                system_prompt="You are a casual, friendly conversational partner. Keep things light and fun. Use informal language and be relatable. Show interest in the user's life and experiences.",
                response_style="casual",
                personality_traits={"friendliness": 0.9, "formality": 0.1, "playfulness": 0.8},
                preferred_topics=["hobbies", "entertainment", "daily life", "fun topics"],
                conversation_starters=["Hey! What's up?", "What's going on today?"],
                created_at=datetime.now()
            )
        ]
        
        for personality in default_personalities:
            self.personality_profiles[personality.name] = personality
    
    def create_session(self, 
                      user_id: Optional[str] = None,
                      language: str = "English",
                      response_style: str = "conversational",
                      voice_mode: str = "consistent",
                      personality: Optional[str] = None,
                      voice_profile: Optional[Dict[str, Any]] = None) -> str:
        """Create a new enhanced conversation session"""
        
        session_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        # Prepare metadata
        metadata = {}
        if personality and personality in self.personality_profiles:
            profile = self.personality_profiles[personality]
            metadata["personality"] = profile.system_prompt
            metadata["personality_name"] = personality
            response_style = profile.response_style
            profile.usage_count += 1
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            turns=[],
            voice_profile=voice_profile,
            language=language,
            response_style=response_style,
            voice_mode=voice_mode,
            created_at=current_time,
            last_activity=current_time,
            metadata=metadata
        )
        
        self.active_sessions[session_id] = session
        self._save_session(session)
        
        logger.info(f"Created enhanced conversation session: {session_id}")
        return session_id
    
    def process_conversation_turn(self, 
                                session_id: str,
                                user_input: str,
                                user_type: str = "text",
                                audio_input_path: Optional[str] = None,
                                preferred_model: Optional[str] = None) -> Tuple[bool, str, Optional[ConversationTurn]]:
        """Process a conversation turn with LLM integration"""
        
        session = self.get_session(session_id)
        if not session:
            return False, "Session not found", None
        
        start_time = time.time()
        
        try:
            # Optimize context for LLM
            context = self.context_optimizer.optimize_context(session, user_input)
            
            # Generate response using LLM
            llm_response = self.llm_manager.generate_response(
                user_input=user_input,
                context=context,
                preferred_model=preferred_model
            )
            
            if llm_response.error:
                logger.error(f"LLM generation failed: {llm_response.error}")
                return False, f"Response generation failed: {llm_response.error}", None
            
            # Create conversation turn
            processing_time = (time.time() - start_time) * 1000
            
            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                session_id=session_id,
                user_input=user_input,
                system_response=llm_response.text,
                user_type=user_type,
                response_type="text",
                audio_input_path=audio_input_path,
                processing_time_ms=processing_time,
                metadata={
                    "llm_model": llm_response.model_name,
                    "llm_confidence": llm_response.confidence,
                    "llm_tokens": llm_response.token_count,
                    "llm_processing_time_ms": llm_response.processing_time_ms
                }
            )
            
            # Add turn to session
            session.turns.append(turn)
            session.last_activity = datetime.now()
            
            # Update session
            self.active_sessions[session_id] = session
            self._save_session(session)
            
            # Update metrics
            self._update_session_metrics(session)
            
            logger.info(f"Processed conversation turn for session {session_id}")
            return True, "Turn processed successfully", turn
            
        except Exception as e:
            logger.error(f"Failed to process conversation turn: {e}")
            return False, f"Processing failed: {e}", None
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session with timeout check"""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Check timeout
            if datetime.now() - session.last_activity > self.session_timeout:
                self._expire_session(session_id)
                return None
            
            return session
        
        # Try to load from storage
        return self._load_session(session_id)
    
    def get_session_metrics(self, session_id: str) -> Optional[ConversationMetrics]:
        """Get metrics for a session"""
        return self.session_metrics.get(session_id)
    
    def get_personality_profiles(self) -> List[PersonalityProfile]:
        """Get available personality profiles"""
        return list(self.personality_profiles.values())
    
    def set_session_personality(self, session_id: str, personality_name: str) -> bool:
        """Set personality for a session"""
        
        session = self.get_session(session_id)
        if not session or personality_name not in self.personality_profiles:
            return False
        
        profile = self.personality_profiles[personality_name]
        
        # Update session metadata
        if not session.metadata:
            session.metadata = {}
        
        session.metadata["personality"] = profile.system_prompt
        session.metadata["personality_name"] = personality_name
        session.response_style = profile.response_style
        
        # Update usage count
        profile.usage_count += 1
        
        # Save session
        self._save_session(session)
        
        return True
    
    def _update_session_metrics(self, session: ConversationSession):
        """Update metrics for a session"""
        
        metrics = self.analyzer.analyze_conversation(session)
        self.session_metrics[session.session_id] = metrics
        
        # Save metrics to file
        try:
            metrics_file = self.storage_dir / f"{session.session_id}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metrics for {session.session_id}: {e}")
    
    def _expire_session(self, session_id: str):
        """Mark session as expired"""
        
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
                    response_type=turn_data.get("response_type", "text"),
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
                if session_file.name.endswith("_metrics.json"):
                    continue  # Skip metrics files
                
                session_id = session_file.stem
                session = self._load_session(session_id)
                
                if session and session.is_active:
                    if datetime.now() - session.last_activity <= self.session_timeout:
                        loaded_count += 1
                    else:
                        session.is_active = False
                        self._save_session(session)
            
            logger.info(f"Loaded {loaded_count} active enhanced sessions")
            
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        
        active_sessions = len(self.active_sessions)
        total_turns = sum(len(session.turns) for session in self.active_sessions.values())
        
        # LLM stats
        llm_info = self.llm_manager.get_model_info()
        
        return {
            "active_sessions": active_sessions,
            "total_turns": total_turns,
            "available_personalities": len(self.personality_profiles),
            "llm_models": list(self.llm_manager.get_available_models()),
            "llm_info": llm_info,
            "session_timeout_minutes": self.session_timeout.total_seconds() / 60,
            "max_turns_per_session": self.max_turns_per_session
        }

# Global enhanced conversation manager
enhanced_conversation_manager = None

def get_enhanced_conversation_manager() -> EnhancedConversationManager:
    """Get the global enhanced conversation manager instance"""
    global enhanced_conversation_manager
    if enhanced_conversation_manager is None:
        enhanced_conversation_manager = EnhancedConversationManager()
    return enhanced_conversation_manager