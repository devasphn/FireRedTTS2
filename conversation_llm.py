#!/usr/bin/env python3
"""
Conversation LLM Integration
Handles conversational AI with local and API-based language models
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Context for conversation generation"""
    conversation_history: List[Dict[str, str]]
    user_profile: Optional[Dict[str, Any]] = None
    system_persona: Optional[str] = None
    language: str = "English"
    response_style: str = "conversational"
    max_context_length: int = 4000
    temperature: float = 0.7
    max_tokens: int = 150

@dataclass
class LLMResponse:
    """Response from language model"""
    text: str
    confidence: float
    processing_time_ms: float
    token_count: int
    model_name: str
    finish_reason: str
    usage_stats: Dict[str, Any]
    error: Optional[str] = None

class BaseLLM(ABC):
    """Abstract base class for language models"""
    
    @abstractmethod
    def generate_response(self, 
                         user_input: str, 
                         context: ConversationContext) -> LLMResponse:
        """Generate a conversational response"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass

class LocalLLM(BaseLLM):
    """Local language model implementation"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: str = "cuda",
                 max_length: int = 1000):
        
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
        # Performance tracking
        self.total_generations = 0
        self.total_processing_time = 0.0
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the local language model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading local LLM: {self.model_name}")
            start_time = time.time()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"Local LLM loaded in {load_time:.2f}s")
            
        except ImportError:
            logger.error("Transformers library not installed. Install with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            raise
    
    def generate_response(self, 
                         user_input: str, 
                         context: ConversationContext) -> LLMResponse:
        """Generate response using local model"""
        
        if self.model is None or self.tokenizer is None:
            return LLMResponse(
                text="Model not available",
                confidence=0.0,
                processing_time_ms=0.0,
                token_count=0,
                model_name=self.model_name,
                finish_reason="error",
                usage_stats={},
                error="Model not initialized"
            )
        
        start_time = time.time()
        
        try:
            # Prepare conversation prompt
            prompt = self._build_conversation_prompt(user_input, context)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=min(len(inputs[0]) + context.max_tokens, self.max_length),
                    temperature=context.temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response_tokens = outputs[0][len(inputs[0]):]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Clean up response
            response_text = self._clean_response(response_text)
            
            # Calculate metrics
            processing_time = (time.time() - start_time) * 1000
            token_count = len(response_tokens)
            
            # Update statistics
            self.total_generations += 1
            self.total_processing_time += processing_time
            
            return LLMResponse(
                text=response_text,
                confidence=0.8,  # Simplified confidence
                processing_time_ms=processing_time,
                token_count=token_count,
                model_name=self.model_name,
                finish_reason="stop",
                usage_stats={
                    "prompt_tokens": len(inputs[0]),
                    "completion_tokens": token_count,
                    "total_tokens": len(inputs[0]) + token_count
                }
            )
            
        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            return LLMResponse(
                text="I apologize, but I'm having trouble generating a response right now.",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                token_count=0,
                model_name=self.model_name,
                finish_reason="error",
                usage_stats={},
                error=str(e)
            )
    
    def _build_conversation_prompt(self, user_input: str, context: ConversationContext) -> str:
        """Build conversation prompt from context"""
        
        prompt_parts = []
        
        # Add system persona if provided
        if context.system_persona:
            prompt_parts.append(f"System: {context.system_persona}")
        
        # Add conversation history
        for turn in context.conversation_history[-5:]:  # Last 5 turns
            if turn.get("user_input"):
                prompt_parts.append(f"Human: {turn['user_input']}")
            if turn.get("system_response"):
                prompt_parts.append(f"Assistant: {turn['system_response']}")
        
        # Add current user input
        prompt_parts.append(f"Human: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        
        # Remove common artifacts
        response = response.strip()
        
        # Remove repetitive patterns
        lines = response.split('\n')
        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            if line and line != prev_line:
                cleaned_lines.append(line)
                prev_line = line
        
        response = ' '.join(cleaned_lines)
        
        # Limit response length
        if len(response) > 500:
            # Find last complete sentence
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
            else:
                response = response[:500] + "..."
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_type": "local",
            "device": self.device,
            "max_length": self.max_length,
            "total_generations": self.total_generations,
            "avg_processing_time_ms": (
                self.total_processing_time / max(self.total_generations, 1)
            ),
            "model_loaded": self.model is not None
        }

class OpenAILLM(BaseLLM):
    """OpenAI API-based language model"""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.client = None
        
        # Performance tracking
        self.total_generations = 0
        self.total_processing_time = 0.0
        self.total_tokens_used = 0
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            import openai
            
            if not self.api_key:
                logger.warning("OpenAI API key not provided")
                return
            
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            logger.info(f"OpenAI client initialized for model: {self.model_name}")
            
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def generate_response(self, 
                         user_input: str, 
                         context: ConversationContext) -> LLMResponse:
        """Generate response using OpenAI API"""
        
        if self.client is None:
            return LLMResponse(
                text="OpenAI client not available",
                confidence=0.0,
                processing_time_ms=0.0,
                token_count=0,
                model_name=self.model_name,
                finish_reason="error",
                usage_stats={},
                error="Client not initialized"
            )
        
        start_time = time.time()
        
        try:
            # Build messages
            messages = self._build_messages(user_input, context)
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=context.temperature,
                max_tokens=context.max_tokens,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            # Extract response
            response_text = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Calculate metrics
            processing_time = (time.time() - start_time) * 1000
            usage = response.usage
            
            # Update statistics
            self.total_generations += 1
            self.total_processing_time += processing_time
            self.total_tokens_used += usage.total_tokens
            
            return LLMResponse(
                text=response_text,
                confidence=0.9,  # OpenAI generally high quality
                processing_time_ms=processing_time,
                token_count=usage.completion_tokens,
                model_name=self.model_name,
                finish_reason=finish_reason,
                usage_stats={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return LLMResponse(
                text="I apologize, but I'm experiencing some technical difficulties.",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                token_count=0,
                model_name=self.model_name,
                finish_reason="error",
                usage_stats={},
                error=str(e)
            )
    
    def _build_messages(self, user_input: str, context: ConversationContext) -> List[Dict[str, str]]:
        """Build messages for OpenAI API"""
        
        messages = []
        
        # System message
        system_message = context.system_persona or self._get_default_system_message(context)
        messages.append({"role": "system", "content": system_message})
        
        # Conversation history
        for turn in context.conversation_history[-10:]:  # Last 10 turns
            if turn.get("user_input"):
                messages.append({"role": "user", "content": turn["user_input"]})
            if turn.get("system_response"):
                messages.append({"role": "assistant", "content": turn["system_response"]})
        
        # Current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _get_default_system_message(self, context: ConversationContext) -> str:
        """Get default system message based on context"""
        
        style_prompts = {
            "conversational": "You are a helpful, friendly assistant. Respond in a natural, conversational way.",
            "formal": "You are a professional assistant. Provide formal, well-structured responses.",
            "casual": "You are a casual, friendly assistant. Keep responses relaxed and informal.",
            "professional": "You are a professional expert assistant. Provide authoritative, detailed responses."
        }
        
        base_prompt = style_prompts.get(context.response_style, style_prompts["conversational"])
        
        if context.language != "English":
            base_prompt += f" Respond in {context.language}."
        
        return base_prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_type": "openai_api",
            "total_generations": self.total_generations,
            "avg_processing_time_ms": (
                self.total_processing_time / max(self.total_generations, 1)
            ),
            "total_tokens_used": self.total_tokens_used,
            "client_available": self.client is not None
        }

class RuleBasedLLM(BaseLLM):
    """Simple rule-based conversation system as fallback"""
    
    def __init__(self):
        self.model_name = "rule_based"
        self.total_generations = 0
        
        # Response templates
        self.response_templates = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What would you like to talk about?",
                "Greetings! I'm here to assist you."
            ],
            "question": [
                "That's an interesting question about {topic}. Let me think about that.",
                "You're asking about {topic}. That's a great topic to explore.",
                "I'd be happy to discuss {topic} with you."
            ],
            "goodbye": [
                "Goodbye! It was nice talking with you.",
                "Take care! Feel free to come back anytime.",
                "See you later! Have a great day!"
            ],
            "thanks": [
                "You're welcome! Is there anything else I can help with?",
                "Happy to help! Let me know if you need anything else.",
                "My pleasure! What else would you like to know?"
            ],
            "default": [
                "That's interesting. Tell me more about that.",
                "I see. Can you elaborate on that?",
                "Interesting point. What do you think about it?",
                "I understand. How does that make you feel?",
                "That's a good observation. What else comes to mind?"
            ]
        }
    
    def generate_response(self, 
                         user_input: str, 
                         context: ConversationContext) -> LLMResponse:
        """Generate rule-based response"""
        
        start_time = time.time()
        
        try:
            user_input_lower = user_input.lower()
            
            # Determine response category
            if any(word in user_input_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
                category = "greeting"
            elif user_input_lower.startswith(("what", "how", "why", "when", "where", "who")):
                category = "question"
            elif any(word in user_input_lower for word in ["goodbye", "bye", "see you", "farewell"]):
                category = "goodbye"
            elif any(word in user_input_lower for word in ["thank", "thanks"]):
                category = "thanks"
            else:
                category = "default"
            
            # Select response template
            import random
            templates = self.response_templates[category]
            response_template = random.choice(templates)
            
            # Fill in template
            if "{topic}" in response_template:
                # Extract topic (simplified)
                words = user_input.split()
                topic = " ".join(words[:5]) if len(words) > 3 else user_input
                response = response_template.format(topic=topic)
            else:
                response = response_template
            
            # Adapt to response style
            if context.response_style == "formal":
                response = response.replace("Hi there!", "Good day.")
                response = response.replace("Hey", "Hello")
            elif context.response_style == "casual":
                response = response.replace("Good day", "Hey")
                response = response.replace("I would be happy", "I'd love")
            
            processing_time = (time.time() - start_time) * 1000
            self.total_generations += 1
            
            return LLMResponse(
                text=response,
                confidence=0.6,  # Rule-based has moderate confidence
                processing_time_ms=processing_time,
                token_count=len(response.split()),
                model_name=self.model_name,
                finish_reason="stop",
                usage_stats={
                    "prompt_tokens": len(user_input.split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(user_input.split()) + len(response.split())
                }
            )
            
        except Exception as e:
            logger.error(f"Rule-based generation failed: {e}")
            return LLMResponse(
                text="I'm here to help. What would you like to talk about?",
                confidence=0.5,
                processing_time_ms=(time.time() - start_time) * 1000,
                token_count=10,
                model_name=self.model_name,
                finish_reason="error",
                usage_stats={},
                error=str(e)
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_type": "rule_based",
            "total_generations": self.total_generations,
            "avg_processing_time_ms": 1.0,  # Very fast
            "model_loaded": True
        }

class ConversationLLMManager:
    """Manages multiple LLM backends with fallback support"""
    
    def __init__(self, 
                 primary_model: str = "local",
                 device: str = "cuda",
                 openai_api_key: Optional[str] = None):
        
        self.primary_model = primary_model
        self.device = device
        self.models: Dict[str, BaseLLM] = {}
        
        # Initialize models
        self._initialize_models(openai_api_key)
        
        # Fallback order
        self.fallback_order = ["local", "openai", "rule_based"]
        if primary_model in self.fallback_order:
            self.fallback_order.remove(primary_model)
            self.fallback_order.insert(0, primary_model)
    
    def _initialize_models(self, openai_api_key: Optional[str]):
        """Initialize available models"""
        
        # Rule-based (always available)
        self.models["rule_based"] = RuleBasedLLM()
        
        # Local model
        try:
            self.models["local"] = LocalLLM(device=self.device)
            logger.info("Local LLM initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize local LLM: {e}")
        
        # OpenAI model
        if openai_api_key:
            try:
                self.models["openai"] = OpenAILLM(api_key=openai_api_key)
                logger.info("OpenAI LLM initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI LLM: {e}")
    
    def generate_response(self, 
                         user_input: str, 
                         context: ConversationContext,
                         preferred_model: Optional[str] = None) -> LLMResponse:
        """Generate response with fallback support"""
        
        # Determine model order
        model_order = self.fallback_order.copy()
        if preferred_model and preferred_model in self.models:
            model_order.remove(preferred_model)
            model_order.insert(0, preferred_model)
        
        # Try models in order
        last_error = None
        
        for model_name in model_order:
            if model_name not in self.models:
                continue
            
            try:
                logger.debug(f"Trying model: {model_name}")
                response = self.models[model_name].generate_response(user_input, context)
                
                # Check if response is valid
                if response.text.strip() and not response.error:
                    logger.debug(f"Successfully generated response with {model_name}")
                    return response
                else:
                    last_error = response.error or "Empty response"
                    
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                last_error = str(e)
                continue
        
        # All models failed, return error response
        logger.error(f"All models failed. Last error: {last_error}")
        return LLMResponse(
            text="I apologize, but I'm having trouble generating a response right now.",
            confidence=0.0,
            processing_time_ms=0.0,
            token_count=0,
            model_name="none",
            finish_reason="error",
            usage_stats={},
            error=f"All models failed: {last_error}"
        )
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about models"""
        
        if model_name:
            if model_name in self.models:
                return self.models[model_name].get_model_info()
            else:
                return {"error": f"Model {model_name} not found"}
        else:
            # Return info for all models
            info = {}
            for name, model in self.models.items():
                info[name] = model.get_model_info()
            return info
    
    def set_primary_model(self, model_name: str) -> bool:
        """Set primary model"""
        if model_name in self.models:
            self.primary_model = model_name
            # Update fallback order
            self.fallback_order.remove(model_name)
            self.fallback_order.insert(0, model_name)
            return True
        return False

# Global LLM manager instance
llm_manager = None

def get_llm_manager(device: str = "cuda", openai_api_key: Optional[str] = None) -> ConversationLLMManager:
    """Get the global LLM manager instance"""
    global llm_manager
    if llm_manager is None:
        llm_manager = ConversationLLMManager(
            primary_model="local",
            device=device,
            openai_api_key=openai_api_key
        )
    return llm_manager