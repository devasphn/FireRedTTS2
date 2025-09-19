#!/usr/bin/env python3

"""
Comprehensive Error Handling and Recovery System
Advanced error handling framework for FireRedTTS2 system with graceful degradation
"""

import asyncio
import logging
import traceback
import time
import json
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import functools
import inspect

import torch
import numpy as np
from pydantic import BaseModel, Field

# ============================================================================
# ERROR TYPES AND SEVERITY LEVELS
# ============================================================================

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(str, Enum):
    """Error categories"""
    MODEL_LOADING = "model_loading"
    AUDIO_PROCESSING = "audio_processing"
    NETWORK = "network"
    RESOURCE_MANAGEMENT = "resource_management"
    VALIDATION = "validation"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    CONFIGURATION = "configuration"

class RecoveryAction(str, Enum):
    """Recovery action types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    RESTART_SERVICE = "restart_service"
    ALERT_ADMIN = "alert_admin"
    IGNORE = "ignore"

# ============================================================================
# ERROR DATA MODELS
# ============================================================================

@dataclass
class ErrorContext:
    """Context information for error handling"""
    
    function_name: str
    module_name: str
    line_number: int
    local_variables: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

@dataclass
class ErrorInfo:
    """Comprehensive error information"""
    
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception] = None
    context: Optional[ErrorContext] = None
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class SystemHealth(BaseModel):
    """System health status"""
    
    overall_status: str = "healthy"
    gpu_available: bool = True
    gpu_memory_percent: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    active_errors: int = 0
    critical_errors: int = 0
    last_check: datetime = Field(default_factory=datetime.now)
    services_status: Dict[str, str] = Field(default_factory=dict)

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class FireRedTTSError(Exception):
    """Base exception for FireRedTTS system"""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 recovery_actions: List[RecoveryAction] = None,
                 metadata: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.recovery_actions = recovery_actions or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

class ModelLoadingError(FireRedTTSError):
    """Model loading related errors"""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.MODEL_LOADING,
            severity=ErrorSeverity.HIGH,
            recovery_actions=[RecoveryAction.FALLBACK, RecoveryAction.RETRY],
            **kwargs
        )
        self.model_name = model_name

class AudioProcessingError(FireRedTTSError):
    """Audio processing related errors"""
    
    def __init__(self, message: str, audio_format: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUDIO_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            recovery_actions=[RecoveryAction.RETRY, RecoveryAction.GRACEFUL_DEGRADATION],
            **kwargs
        )
        self.audio_format = audio_format

class NetworkError(FireRedTTSError):
    """Network related errors"""
    
    def __init__(self, message: str, connection_type: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            recovery_actions=[RecoveryAction.RETRY],
            **kwargs
        )
        self.connection_type = connection_type

class ResourceError(FireRedTTSError):
    """Resource management errors"""
    
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE_MANAGEMENT,
            severity=ErrorSeverity.HIGH,
            recovery_actions=[RecoveryAction.GRACEFUL_DEGRADATION, RecoveryAction.ALERT_ADMIN],
            **kwargs
        )
        self.resource_type = resource_type

# ============================================================================
# ERROR HANDLER REGISTRY
# ============================================================================

class ErrorHandlerRegistry:
    """Registry for error handlers"""
    
    def __init__(self):
        self._handlers: Dict[ErrorCategory, List[Callable]] = {}
        self._fallback_handlers: List[Callable] = []
        self._recovery_strategies: Dict[RecoveryAction, Callable] = {}
        
    def register_handler(self, category: ErrorCategory, handler: Callable):
        """Register error handler for specific category"""
        if category not in self._handlers:
            self._handlers[category] = []
        self._handlers[category].append(handler)
    
    def register_fallback_handler(self, handler: Callable):
        """Register fallback error handler"""
        self._fallback_handlers.append(handler)
    
    def register_recovery_strategy(self, action: RecoveryAction, strategy: Callable):
        """Register recovery strategy"""
        self._recovery_strategies[action] = strategy
    
    def get_handlers(self, category: ErrorCategory) -> List[Callable]:
        """Get handlers for error category"""
        return self._handlers.get(category, []) + self._fallback_handlers
    
    def get_recovery_strategy(self, action: RecoveryAction) -> Optional[Callable]:
        """Get recovery strategy"""
        return self._recovery_strategies.get(action)

# Global error handler registry
error_registry = ErrorHandlerRegistry()

# ============================================================================
# ERROR HANDLING DECORATORS
# ============================================================================

def handle_errors(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    recovery_actions: List[RecoveryAction] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fallback_value: Any = None
):
    """Decorator for automatic error handling"""
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_manager = ErrorManager.get_instance()
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except Exception as e:
                    context = ErrorContext(
                        function_name=func.__name__,
                        module_name=func.__module__,
                        line_number=inspect.currentframe().f_lineno,
                        stack_trace=traceback.format_exc()
                    )
                    
                    error_info = ErrorInfo(
                        error_id=f"{func.__name__}_{int(time.time())}",
                        category=category,
                        severity=severity,
                        message=str(e),
                        original_exception=e,
                        context=context,
                        recovery_actions=recovery_actions or [],
                        retry_count=attempt,
                        max_retries=max_retries
                    )
                    
                    # Handle the error
                    should_retry = await error_manager.handle_error(error_info)
                    
                    if attempt < max_retries and should_retry:
                        await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        if fallback_value is not None:
                            return fallback_value
                        raise
            
            return fallback_value
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def gpu_safe(fallback_cpu: bool = True):
    """Decorator for GPU-safe operations with CPU fallback"""
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "CUDA" in str(e) and fallback_cpu:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"GPU operation failed, falling back to CPU: {e}")
                    
                    # Move tensors to CPU if present in kwargs
                    cpu_kwargs = {}
                    for key, value in kwargs.items():
                        if isinstance(value, torch.Tensor):
                            cpu_kwargs[key] = value.cpu()
                        else:
                            cpu_kwargs[key] = value
                    
                    return await func(*args, **cpu_kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **cpu_kwargs)
                else:
                    raise ResourceError(f"GPU operation failed: {e}", resource_type="gpu")
        
        return wrapper
    return decorator

# ============================================================================
# ERROR MANAGER
# ============================================================================

class ErrorManager:
    """Central error management system"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_errors: Dict[str, ErrorInfo] = {}
        self.error_history: List[ErrorInfo] = []
        self.max_history_size = 1000
        self.system_health = SystemHealth()
        self.health_check_interval = 30  # seconds
        self.last_health_check = datetime.now()
        
        # Setup logging
        self._setup_logging()
        
        # Start health monitoring
        self._start_health_monitoring()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _setup_logging(self):
        """Setup error logging"""
        handler = logging.FileHandler('errors.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        def monitor_health():
            while True:
                try:
                    self._update_system_health()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(self.health_check_interval)
        
        thread = threading.Thread(target=monitor_health, daemon=True)
        thread.start()
    
    def _update_system_health(self):
        """Update system health metrics"""
        try:
            # CPU and memory metrics
            self.system_health.cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self.system_health.memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_health.disk_percent = (disk.used / disk.total) * 100
            
            # GPU metrics
            if torch.cuda.is_available():
                self.system_health.gpu_available = True
                gpu_memory = torch.cuda.memory_stats()
                allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                reserved = gpu_memory.get('reserved_bytes.all.current', 0)
                total_memory = torch.cuda.get_device_properties(0).total_memory
                self.system_health.gpu_memory_percent = (allocated / total_memory) * 100
            else:
                self.system_health.gpu_available = False
                self.system_health.gpu_memory_percent = 0.0
            
            # Error counts
            self.system_health.active_errors = len(self.active_errors)
            self.system_health.critical_errors = sum(
                1 for error in self.active_errors.values()
                if error.severity == ErrorSeverity.CRITICAL
            )
            
            # Overall status
            if self.system_health.critical_errors > 0:
                self.system_health.overall_status = "critical"
            elif self.system_health.active_errors > 5:
                self.system_health.overall_status = "degraded"
            elif (self.system_health.gpu_memory_percent > 90 or 
                  self.system_health.memory_percent > 90):
                self.system_health.overall_status = "warning"
            else:
                self.system_health.overall_status = "healthy"
            
            self.system_health.last_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update system health: {e}")
    
    async def handle_error(self, error_info: ErrorInfo) -> bool:
        """Handle error and return whether to retry"""
        
        # Log the error
        self.logger.error(
            f"Error {error_info.error_id}: {error_info.message} "
            f"(Category: {error_info.category}, Severity: {error_info.severity})"
        )
        
        # Add to active errors
        self.active_errors[error_info.error_id] = error_info
        
        # Add to history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
        
        # Get handlers for this error category
        handlers = error_registry.get_handlers(error_info.category)
        
        # Execute handlers
        for handler in handlers:
            try:
                await handler(error_info)
            except Exception as e:
                self.logger.error(f"Error handler failed: {e}")
        
        # Execute recovery actions
        should_retry = False
        for action in error_info.recovery_actions:
            try:
                strategy = error_registry.get_recovery_strategy(action)
                if strategy:
                    result = await strategy(error_info)
                    if action == RecoveryAction.RETRY and result:
                        should_retry = True
            except Exception as e:
                self.logger.error(f"Recovery action {action} failed: {e}")
        
        return should_retry and error_info.retry_count < error_info.max_retries
    
    def resolve_error(self, error_id: str):
        """Mark error as resolved"""
        if error_id in self.active_errors:
            error_info = self.active_errors[error_id]
            error_info.resolved = True
            error_info.resolution_time = datetime.now()
            del self.active_errors[error_id]
            
            self.logger.info(f"Error {error_id} resolved")
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health"""
        return self.system_health
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        recent_errors = [
            error for error in self.error_history
            if error.context and error.context.timestamp > last_hour
        ]
        
        daily_errors = [
            error for error in self.error_history
            if error.context and error.context.timestamp > last_day
        ]
        
        return {
            "active_errors": len(self.active_errors),
            "recent_errors_1h": len(recent_errors),
            "daily_errors": len(daily_errors),
            "error_categories": {
                category.value: sum(
                    1 for error in daily_errors
                    if error.category == category
                )
                for category in ErrorCategory
            },
            "severity_distribution": {
                severity.value: sum(
                    1 for error in daily_errors
                    if error.severity == severity
                )
                for severity in ErrorSeverity
            }
        }

# ============================================================================
# RECOVERY STRATEGIES
# ============================================================================

async def retry_strategy(error_info: ErrorInfo) -> bool:
    """Retry recovery strategy"""
    logger = logging.getLogger(__name__)
    
    if error_info.retry_count < error_info.max_retries:
        logger.info(f"Retrying operation for error {error_info.error_id} (attempt {error_info.retry_count + 1})")
        return True
    else:
        logger.warning(f"Max retries exceeded for error {error_info.error_id}")
        return False

async def fallback_strategy(error_info: ErrorInfo) -> bool:
    """Fallback recovery strategy"""
    logger = logging.getLogger(__name__)
    logger.info(f"Executing fallback strategy for error {error_info.error_id}")
    
    # Implement specific fallback logic based on error category
    if error_info.category == ErrorCategory.MODEL_LOADING:
        # Try to load a simpler model or use CPU
        logger.info("Attempting to load fallback model")
        return True
    elif error_info.category == ErrorCategory.AUDIO_PROCESSING:
        # Use simpler audio processing
        logger.info("Using simplified audio processing")
        return True
    
    return False

async def graceful_degradation_strategy(error_info: ErrorInfo) -> bool:
    """Graceful degradation recovery strategy"""
    logger = logging.getLogger(__name__)
    logger.info(f"Applying graceful degradation for error {error_info.error_id}")
    
    # Reduce system capabilities to maintain basic functionality
    if error_info.category == ErrorCategory.RESOURCE_MANAGEMENT:
        # Reduce batch sizes, lower quality settings, etc.
        logger.info("Reducing system load for graceful degradation")
        return True
    
    return False

async def restart_service_strategy(error_info: ErrorInfo) -> bool:
    """Service restart recovery strategy"""
    logger = logging.getLogger(__name__)
    logger.warning(f"Restarting service due to error {error_info.error_id}")
    
    # This would typically restart specific services
    # Implementation depends on service architecture
    return False

async def alert_admin_strategy(error_info: ErrorInfo) -> bool:
    """Admin alert recovery strategy"""
    logger = logging.getLogger(__name__)
    logger.critical(f"Alerting administrator for critical error {error_info.error_id}")
    
    # Send alert to administrators
    # This could be email, Slack, webhook, etc.
    return False

# ============================================================================
# SPECIFIC ERROR HANDLERS
# ============================================================================

async def model_loading_error_handler(error_info: ErrorInfo):
    """Handle model loading errors"""
    logger = logging.getLogger(__name__)
    
    if isinstance(error_info.original_exception, torch.cuda.OutOfMemoryError):
        logger.warning("GPU out of memory during model loading, clearing cache")
        torch.cuda.empty_cache()
    
    logger.info(f"Handling model loading error: {error_info.message}")

async def audio_processing_error_handler(error_info: ErrorInfo):
    """Handle audio processing errors"""
    logger = logging.getLogger(__name__)
    logger.info(f"Handling audio processing error: {error_info.message}")
    
    # Could implement audio format conversion, resampling, etc.

async def network_error_handler(error_info: ErrorInfo):
    """Handle network errors"""
    logger = logging.getLogger(__name__)
    logger.info(f"Handling network error: {error_info.message}")
    
    # Could implement connection retry logic, fallback endpoints, etc.

async def resource_error_handler(error_info: ErrorInfo):
    """Handle resource management errors"""
    logger = logging.getLogger(__name__)
    logger.warning(f"Handling resource error: {error_info.message}")
    
    # Free up resources, reduce load, etc.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_error_handling():
    """Initialize error handling system"""
    
    # Register error handlers
    error_registry.register_handler(ErrorCategory.MODEL_LOADING, model_loading_error_handler)
    error_registry.register_handler(ErrorCategory.AUDIO_PROCESSING, audio_processing_error_handler)
    error_registry.register_handler(ErrorCategory.NETWORK, network_error_handler)
    error_registry.register_handler(ErrorCategory.RESOURCE_MANAGEMENT, resource_error_handler)
    
    # Register recovery strategies
    error_registry.register_recovery_strategy(RecoveryAction.RETRY, retry_strategy)
    error_registry.register_recovery_strategy(RecoveryAction.FALLBACK, fallback_strategy)
    error_registry.register_recovery_strategy(RecoveryAction.GRACEFUL_DEGRADATION, graceful_degradation_strategy)
    error_registry.register_recovery_strategy(RecoveryAction.RESTART_SERVICE, restart_service_strategy)
    error_registry.register_recovery_strategy(RecoveryAction.ALERT_ADMIN, alert_admin_strategy)
    
    # Initialize error manager
    ErrorManager.get_instance()

# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

class ErrorContext:
    """Context manager for error handling"""
    
    def __init__(self, operation_name: str, category: ErrorCategory = ErrorCategory.SYSTEM):
        self.operation_name = operation_name
        self.category = category
        self.start_time = None
        self.error_manager = ErrorManager.get_instance()
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            context = ErrorContext(
                function_name=self.operation_name,
                module_name=__name__,
                line_number=0,
                stack_trace=traceback.format_exc()
            )
            
            error_info = ErrorInfo(
                error_id=f"{self.operation_name}_{int(time.time())}",
                category=self.category,
                severity=ErrorSeverity.MEDIUM,
                message=str(exc_val),
                original_exception=exc_val,
                context=context
            )
            
            await self.error_manager.handle_error(error_info)
        
        return False  # Don't suppress exceptions

# Export main components
__all__ = [
    'ErrorSeverity', 'ErrorCategory', 'RecoveryAction',
    'ErrorInfo', 'SystemHealth', 'ErrorManager',
    'FireRedTTSError', 'ModelLoadingError', 'AudioProcessingError', 
    'NetworkError', 'ResourceError',
    'handle_errors', 'gpu_safe', 'initialize_error_handling',
    'error_registry'
]

# Initialize on import
initialize_error_handling()