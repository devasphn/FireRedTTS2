#!/usr/bin/env python3
"""
Context-Aware TTS Integration
Integrates conversation context, emotion, and prosody into speech synthesis
"""

import re
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import torch
import numpy as np

from enhanced_fireredtts2 import EnhancedFireRedTTS2, GenerationRequest, GenerationResult
from conversation_manager import ConversationTurn, ConversationSession

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Emotion types for TTS"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    CALM = "calm"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CONFUSED = "confused"
    THOUGHTFUL = "thoughtful"
    EMPATHETIC = "empathetic"

class ProsodyStyle(Enum):
    """Prosody styles for speech"""
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"
    CASUAL = "casual"
    DRAMATIC = "dramatic"
    MONOTONE = "monotone"
    EXPRESSIVE = "expressive"

@dataclass
class ConversationContext:
    """Context information for TTS generation"""
    session_id: str
    conversation_history: List[ConversationTurn]
    current_emotion: EmotionType
    prosody_style: ProsodyStyle
    speaker_consistency: bool
    topic_context: List[str]
    user_engagement_level: float
    conversation_flow_state: str
    previous_audio_characteristics: Optional[Dict[str, Any]] = None

@dataclass
class TTSContextualRequest:
    """Enhanced TTS request with contextual information"""
    base_request: GenerationRequest
    conversation_context: ConversationContext
    emotion_override: Optional[EmotionType] = None
    prosody_override: Optional[ProsodyStyle] = None
    emphasis_words: List[str] = None
    pause_locations: List[int] = None
    speaking_rate: float = 1.0
    pitch_adjustment: float = 0.0

class EmotionAnalyzer:
    """Analyzes text and conversation context to determine appropriate emotion"""
    
    def __init__(self):
        # Emotion keywords mapping
        self.emotion_keywords = {
            EmotionType.HAPPY: [
                "happy", "joy", "excited", "great", "wonderful", "amazing", "fantastic", 
                "love", "delighted", "pleased", "cheerful", "glad", "thrilled"
            ],
            EmotionType.SAD: [
                "sad", "sorry", "disappointed", "upset", "hurt", "depressed", 
                "unfortunate", "tragic", "heartbroken", "grief", "sorrow"
            ],
            EmotionType.ANGRY: [
                "angry", "mad", "furious", "annoyed", "frustrated", "irritated", 
                "outraged", "livid", "enraged", "infuriated"
            ],
            EmotionType.SURPRISED: [
                "surprised", "shocked", "amazed", "astonished", "stunned", 
                "incredible", "unbelievable", "wow", "omg", "really"
            ],
            EmotionType.CONFUSED: [
                "confused", "puzzled", "unclear", "don't understand", "what", 
                "how", "why", "huh", "perplexed", "bewildered"
            ],
            EmotionType.THOUGHTFUL: [
                "think", "consider", "ponder", "reflect", "analyze", "contemplate", 
                "interesting", "hmm", "let me see", "perhaps"
            ],
            EmotionType.EMPATHETIC: [
                "understand", "feel", "sorry to hear", "sympathize", "empathize", 
                "care", "support", "here for you"
            ]
        }
        
        # Punctuation-based emotion indicators
        self.punctuation_emotions = {
            "!": EmotionType.EXCITED,
            "?": EmotionType.CONFUSED,
            "...": EmotionType.THOUGHTFUL,
            "!!": EmotionType.EXCITED,
            "???": EmotionType.CONFUSED
        }
    
    def analyze_emotion(self, text: str, context: ConversationContext) -> EmotionType:
        """Analyze text and context to determine appropriate emotion"""
        
        text_lower = text.lower()
        
        # Check for explicit emotion keywords
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Check punctuation patterns
        for punct, emotion in self.punctuation_emotions.items():
            if punct in text:
                emotion_scores[emotion] = emotion_scores.get(emotion, 0) + 1
        
        # Consider conversation context
        if context.conversation_history:
            last_turn = context.conversation_history[-1]
            
            # Mirror user emotion if appropriate
            if last_turn.user_type == "audio":
                user_emotion = self.analyze_emotion(last_turn.user_input, context)
                if user_emotion in [EmotionType.SAD, EmotionType.CONFUSED]:
                    emotion_scores[EmotionType.EMPATHETIC] = emotion_scores.get(EmotionType.EMPATHETIC, 0) + 2
        
        # Consider engagement level
        if context.user_engagement_level < 0.3:
            emotion_scores[EmotionType.EXCITED] = emotion_scores.get(EmotionType.EXCITED, 0) + 1
        elif context.user_engagement_level > 0.8:
            emotion_scores[EmotionType.HAPPY] = emotion_scores.get(EmotionType.HAPPY, 0) + 1
        
        # Return highest scoring emotion or neutral
        if emotion_scores:
            return max(emotion_scores.items(), key=lambda x: x[1])[0]
        else:
            return EmotionType.NEUTRAL

class ProsodyAnalyzer:
    """Analyzes conversation context to determine appropriate prosody"""
    
    def __init__(self):
        self.style_indicators = {
            ProsodyStyle.FORMAL: [
                "please", "thank you", "sir", "madam", "certainly", "indeed", 
                "furthermore", "however", "therefore"
            ],
            ProsodyStyle.CASUAL: [
                "hey", "yeah", "nah", "cool", "awesome", "dude", "guys", 
                "stuff", "thing", "kinda", "sorta"
            ],
            ProsodyStyle.DRAMATIC: [
                "incredible", "amazing", "unbelievable", "shocking", "dramatic", 
                "intense", "powerful", "overwhelming"
            ],
            ProsodyStyle.EXPRESSIVE: [
                "wow", "oh", "ah", "hmm", "well", "you know", "like", 
                "really", "totally", "absolutely"
            ]
        }
    
    def analyze_prosody(self, text: str, context: ConversationContext) -> ProsodyStyle:
        """Analyze text and context to determine appropriate prosody style"""
        
        text_lower = text.lower()
        
        # Check for style indicators
        style_scores = {}
        for style, indicators in self.style_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                style_scores[style] = score
        
        # Consider conversation context
        if context.prosody_style != ProsodyStyle.CONVERSATIONAL:
            # Maintain consistency with established style
            return context.prosody_style
        
        # Consider text characteristics
        if len(text.split()) > 50:  # Long text
            style_scores[ProsodyStyle.FORMAL] = style_scores.get(ProsodyStyle.FORMAL, 0) + 1
        
        if "?" in text:  # Questions
            style_scores[ProsodyStyle.EXPRESSIVE] = style_scores.get(ProsodyStyle.EXPRESSIVE, 0) + 1
        
        if "!" in text:  # Exclamations
            style_scores[ProsodyStyle.DRAMATIC] = style_scores.get(ProsodyStyle.DRAMATIC, 0) + 1
        
        # Return highest scoring style or conversational
        if style_scores:
            return max(style_scores.items(), key=lambda x: x[1])[0]
        else:
            return ProsodyStyle.CONVERSATIONAL

class TextProcessor:
    """Processes text for contextual TTS generation"""
    
    def __init__(self):
        self.emphasis_patterns = [
            r'\*([^*]+)\*',  # *emphasis*
            r'_([^_]+)_',    # _emphasis_
            r'\b(very|really|extremely|incredibly|absolutely)\s+(\w+)',  # intensifiers
            r'\b(NOT|NEVER|ALWAYS|MUST)\b'  # caps words
        ]
        
        self.pause_patterns = [
            r'[.!?]',  # End of sentences
            r'[,;:]',  # Commas and colons
            r'\s-\s',  # Dashes
            r'\.\.\.',  # Ellipsis
        ]
    
    def process_text_for_context(self, text: str, context: ConversationContext) -> Dict[str, Any]:
        """Process text to extract contextual information"""
        
        # Find emphasis words
        emphasis_words = []
        for pattern in self.emphasis_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if isinstance(matches[0], tuple) if matches else False:
                emphasis_words.extend([match[1] for match in matches])
            else:
                emphasis_words.extend(matches)
        
        # Find pause locations (character positions)
        pause_locations = []
        for pattern in self.pause_patterns:
            for match in re.finditer(pattern, text):
                pause_locations.append(match.start())
        
        # Determine speaking rate based on context
        speaking_rate = 1.0
        
        if context.current_emotion == EmotionType.EXCITED:
            speaking_rate = 1.2
        elif context.current_emotion == EmotionType.SAD:
            speaking_rate = 0.8
        elif context.current_emotion == EmotionType.THOUGHTFUL:
            speaking_rate = 0.9
        
        # Adjust for conversation flow
        if context.conversation_flow_state == "rapid_exchange":
            speaking_rate *= 1.1
        elif context.conversation_flow_state == "deep_discussion":
            speaking_rate *= 0.9
        
        # Determine pitch adjustment
        pitch_adjustment = 0.0
        
        if context.current_emotion == EmotionType.HAPPY:
            pitch_adjustment = 0.1
        elif context.current_emotion == EmotionType.SAD:
            pitch_adjustment = -0.1
        elif context.current_emotion == EmotionType.SURPRISED:
            pitch_adjustment = 0.2
        
        return {
            "emphasis_words": emphasis_words,
            "pause_locations": sorted(pause_locations),
            "speaking_rate": speaking_rate,
            "pitch_adjustment": pitch_adjustment,
            "processed_text": self._clean_text_for_tts(text)
        }
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS processing"""
        
        # Remove emphasis markers
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper sentence endings
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text.strip()

class ContextAwareTTS:
    """Context-aware TTS system that integrates conversation context"""
    
    def __init__(self, enhanced_tts: EnhancedFireRedTTS2):
        self.enhanced_tts = enhanced_tts
        self.emotion_analyzer = EmotionAnalyzer()
        self.prosody_analyzer = ProsodyAnalyzer()
        self.text_processor = TextProcessor()
        
        # Context tracking
        self.session_contexts: Dict[str, ConversationContext] = {}
        
        # Performance tracking
        self.generation_stats = {
            "total_contextual_generations": 0,
            "emotion_distributions": {},
            "prosody_distributions": {},
            "average_context_processing_time": 0.0
        }
    
    def create_conversation_context(self, 
                                 session: ConversationSession,
                                 current_emotion: EmotionType = EmotionType.NEUTRAL,
                                 prosody_style: ProsodyStyle = ProsodyStyle.CONVERSATIONAL) -> ConversationContext:
        """Create conversation context from session"""
        
        # Analyze conversation for topic context
        topic_context = self._extract_topic_context(session.turns)
        
        # Calculate user engagement level
        engagement_level = self._calculate_engagement_level(session.turns)
        
        # Determine conversation flow state
        flow_state = self._analyze_conversation_flow(session.turns)
        
        context = ConversationContext(
            session_id=session.session_id,
            conversation_history=session.turns[-10:],  # Last 10 turns
            current_emotion=current_emotion,
            prosody_style=prosody_style,
            speaker_consistency=True,
            topic_context=topic_context,
            user_engagement_level=engagement_level,
            conversation_flow_state=flow_state
        )
        
        self.session_contexts[session.session_id] = context
        return context
    
    def generate_contextual_speech(self, 
                                 text: str,
                                 session: ConversationSession,
                                 voice_mode: str = "consistent",
                                 reference_audio_path: Optional[str] = None,
                                 reference_text: Optional[str] = None,
                                 emotion_override: Optional[EmotionType] = None,
                                 prosody_override: Optional[ProsodyStyle] = None) -> GenerationResult:
        """Generate speech with conversation context awareness"""
        
        start_time = time.time()
        
        try:
            # Get or create conversation context
            context = self.session_contexts.get(session.session_id)
            if not context:
                context = self.create_conversation_context(session)
            
            # Analyze emotion and prosody
            detected_emotion = emotion_override or self.emotion_analyzer.analyze_emotion(text, context)
            detected_prosody = prosody_override or self.prosody_analyzer.analyze_prosody(text, context)
            
            # Update context
            context.current_emotion = detected_emotion
            context.prosody_style = detected_prosody
            
            # Process text for contextual features
            text_features = self.text_processor.process_text_for_context(text, context)
            
            # Create enhanced TTS request
            base_request = GenerationRequest(
                request_id=f"contextual_{int(time.time())}",
                text=text_features["processed_text"],
                voice_mode=voice_mode,
                reference_audio_path=reference_audio_path,
                reference_text=reference_text,
                temperature=self._get_contextual_temperature(detected_emotion, detected_prosody),
                top_k=self._get_contextual_top_k(detected_emotion, detected_prosody),
                metadata={
                    "emotion": detected_emotion.value,
                    "prosody": detected_prosody.value,
                    "emphasis_words": text_features["emphasis_words"],
                    "speaking_rate": text_features["speaking_rate"],
                    "pitch_adjustment": text_features["pitch_adjustment"],
                    "context_processing_time_ms": (time.time() - start_time) * 1000
                }
            )
            
            # Generate speech
            result = self.enhanced_tts.generate_speech(base_request)
            
            # Update statistics
            self._update_generation_stats(detected_emotion, detected_prosody, time.time() - start_time)
            
            # Update context with generation results
            if result.success:
                context.previous_audio_characteristics = {
                    "duration_ms": result.audio_duration_ms,
                    "generation_time_ms": result.generation_time_ms,
                    "emotion": detected_emotion.value,
                    "prosody": detected_prosody.value
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Contextual speech generation failed: {e}")
            
            # Fallback to basic generation
            fallback_request = GenerationRequest(
                request_id=f"fallback_{int(time.time())}",
                text=text,
                voice_mode=voice_mode,
                reference_audio_path=reference_audio_path,
                reference_text=reference_text
            )
            
            return self.enhanced_tts.generate_speech(fallback_request)
    
    def _extract_topic_context(self, turns: List[ConversationTurn]) -> List[str]:
        """Extract topic context from conversation turns"""
        
        if not turns:
            return []
        
        # Simple keyword extraction (in production, use NLP libraries)
        all_text = " ".join([turn.user_input + " " + turn.system_response for turn in turns[-5:]])
        words = all_text.lower().split()
        
        # Filter meaningful words (length > 3, not common words)
        common_words = {"the", "and", "that", "have", "for", "not", "with", "you", "this", "but", "his", "from", "they"}
        meaningful_words = [word for word in words if len(word) > 3 and word not in common_words]
        
        # Count word frequencies
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [topic[0] for topic in topics]
    
    def _calculate_engagement_level(self, turns: List[ConversationTurn]) -> float:
        """Calculate user engagement level from conversation history"""
        
        if not turns:
            return 0.5
        
        factors = []
        
        # Recent activity
        if len(turns) >= 3:
            recent_turns = turns[-3:]
            avg_response_length = np.mean([len(turn.user_input.split()) for turn in recent_turns])
            # Normalize to 0-1 scale (assuming 1-20 words is normal range)
            length_factor = min(avg_response_length / 20, 1.0)
            factors.append(length_factor)
        
        # Question asking (indicates engagement)
        questions = sum(1 for turn in turns[-5:] if '?' in turn.user_input)
        question_factor = min(questions / 5, 1.0)
        factors.append(question_factor)
        
        # Response time consistency (faster responses = higher engagement)
        response_times = [turn.processing_time_ms for turn in turns[-5:] if turn.processing_time_ms > 0]
        if response_times:
            avg_response_time = np.mean(response_times)
            # Lower response time = higher engagement (inverse relationship)
            time_factor = max(0, 1.0 - (avg_response_time / 5000))  # 5 seconds as baseline
            factors.append(time_factor)
        
        return np.mean(factors) if factors else 0.5
    
    def _analyze_conversation_flow(self, turns: List[ConversationTurn]) -> str:
        """Analyze conversation flow state"""
        
        if len(turns) < 2:
            return "starting"
        
        recent_turns = turns[-3:]
        
        # Calculate average turn length
        avg_length = np.mean([len(turn.user_input.split()) + len(turn.system_response.split()) 
                             for turn in recent_turns])
        
        # Calculate turn frequency
        if len(recent_turns) >= 2:
            time_diff = (recent_turns[-1].timestamp - recent_turns[0].timestamp).total_seconds()
            turn_frequency = len(recent_turns) / max(time_diff, 1)
        else:
            turn_frequency = 0.5
        
        # Determine flow state
        if turn_frequency > 0.5 and avg_length < 20:
            return "rapid_exchange"
        elif avg_length > 50:
            return "deep_discussion"
        elif turn_frequency < 0.1:
            return "slow_conversation"
        else:
            return "normal_flow"
    
    def _get_contextual_temperature(self, emotion: EmotionType, prosody: ProsodyStyle) -> float:
        """Get temperature parameter based on emotion and prosody"""
        
        base_temperature = 0.9
        
        # Emotion adjustments
        emotion_adjustments = {
            EmotionType.EXCITED: 0.1,
            EmotionType.HAPPY: 0.05,
            EmotionType.CALM: -0.1,
            EmotionType.SAD: -0.05,
            EmotionType.THOUGHTFUL: -0.15
        }
        
        # Prosody adjustments
        prosody_adjustments = {
            ProsodyStyle.DRAMATIC: 0.1,
            ProsodyStyle.EXPRESSIVE: 0.05,
            ProsodyStyle.FORMAL: -0.1,
            ProsodyStyle.MONOTONE: -0.2
        }
        
        temperature = base_temperature
        temperature += emotion_adjustments.get(emotion, 0)
        temperature += prosody_adjustments.get(prosody, 0)
        
        return max(0.1, min(1.5, temperature))
    
    def _get_contextual_top_k(self, emotion: EmotionType, prosody: ProsodyStyle) -> int:
        """Get top_k parameter based on emotion and prosody"""
        
        base_top_k = 30
        
        # Emotion adjustments
        if emotion in [EmotionType.EXCITED, EmotionType.HAPPY]:
            return base_top_k + 10
        elif emotion in [EmotionType.CALM, EmotionType.THOUGHTFUL]:
            return base_top_k - 10
        
        # Prosody adjustments
        if prosody == ProsodyStyle.DRAMATIC:
            return base_top_k + 15
        elif prosody == ProsodyStyle.MONOTONE:
            return base_top_k - 15
        
        return base_top_k
    
    def _update_generation_stats(self, emotion: EmotionType, prosody: ProsodyStyle, processing_time: float):
        """Update generation statistics"""
        
        self.generation_stats["total_contextual_generations"] += 1
        
        # Update emotion distribution
        emotion_key = emotion.value
        self.generation_stats["emotion_distributions"][emotion_key] = (
            self.generation_stats["emotion_distributions"].get(emotion_key, 0) + 1
        )
        
        # Update prosody distribution
        prosody_key = prosody.value
        self.generation_stats["prosody_distributions"][prosody_key] = (
            self.generation_stats["prosody_distributions"].get(prosody_key, 0) + 1
        )
        
        # Update average processing time
        total_time = self.generation_stats["average_context_processing_time"] * (
            self.generation_stats["total_contextual_generations"] - 1
        )
        total_time += processing_time * 1000  # Convert to ms
        self.generation_stats["average_context_processing_time"] = (
            total_time / self.generation_stats["total_contextual_generations"]
        )
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context-aware TTS statistics"""
        
        return {
            **self.generation_stats,
            "active_contexts": len(self.session_contexts),
            "supported_emotions": [e.value for e in EmotionType],
            "supported_prosody_styles": [p.value for p in ProsodyStyle]
        }
    
    def clear_session_context(self, session_id: str):
        """Clear context for a session"""
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]

# Global context-aware TTS instance
context_aware_tts = None

def get_context_aware_tts(enhanced_tts: EnhancedFireRedTTS2) -> ContextAwareTTS:
    """Get the global context-aware TTS instance"""
    global context_aware_tts
    if context_aware_tts is None:
        context_aware_tts = ContextAwareTTS(enhanced_tts)
    return context_aware_tts

def create_contextual_tts_interface(context_tts: ContextAwareTTS) -> Dict[str, Any]:
    """Create interface components for contextual TTS"""
    
    def generate_contextual_speech(text, session_data, emotion="neutral", prosody="conversational"):
        """Generate contextual speech for Gradio interface"""
        
        if not text.strip():
            return None, "❌ Text is required"
        
        # Create mock session for demo (in production, use real session)
        from conversation_manager import ConversationSession
        from datetime import datetime
        
        session = ConversationSession(
            session_id="demo_session",
            user_id=None,
            turns=[],
            voice_profile=None,
            language="English",
            response_style="conversational",
            voice_mode="consistent",
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        # Convert string enums
        emotion_enum = EmotionType(emotion) if emotion != "auto" else None
        prosody_enum = ProsodyStyle(prosody) if prosody != "auto" else None
        
        result = context_tts.generate_contextual_speech(
            text=text.strip(),
            session=session,
            emotion_override=emotion_enum,
            prosody_override=prosody_enum
        )
        
        if result.success:
            audio_array = result.audio_data.cpu().numpy()
            if len(audio_array.shape) > 1:
                audio_array = audio_array.squeeze()
            
            metadata = result.metadata or {}
            status = f"✅ Generated with {metadata.get('emotion', 'neutral')} emotion and {metadata.get('prosody', 'conversational')} prosody"
            
            return (result.sample_rate, audio_array), status
        else:
            return None, f"❌ Generation failed: {result.error_message}"
    
    def get_contextual_stats():
        """Get contextual TTS statistics"""
        return context_tts.get_context_stats()
    
    # Available options for interface
    emotion_choices = ["auto"] + [e.value for e in EmotionType]
    prosody_choices = ["auto"] + [p.value for p in ProsodyStyle]
    
    return {
        "generate_contextual": generate_contextual_speech,
        "get_stats": get_contextual_stats,
        "emotion_choices": emotion_choices,
        "prosody_choices": prosody_choices,
        "context_tts": context_tts
    }