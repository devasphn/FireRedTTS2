#!/usr/bin/env python3
"""
Security and Access Control System
Comprehensive security implementation for FireRedTTS2 deployment with input validation,
rate limiting, session management, and audit logging
"""

import os
import json
import time
import hashlib
import secrets
import logging
import asyncio
import mimetypes
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
import re

import jwt
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, validator
import aiofiles
import magic

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    
    # Authentication settings
    enable_authentication: bool = False
    jwt_secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # Rate limiting settings
    enable_rate_limiting: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    concurrent_sessions_per_ip: int = 5
    
    # File upload security
    max_file_size_mb: int = 100
    allowed_audio_extensions: Set[str] = field(default_factory=lambda: {'.wav', '.mp3', '.flac', '.m4a', '.ogg'})
    allowed_mime_types: Set[str] = field(default_factory=lambda: {
        'audio/wav', 'audio/mpeg', 'audio/flac', 'audio/mp4', 'audio/ogg'
    })
    enable_virus_scanning: bool = False
    quarantine_suspicious_files: bool = True
    
    # Input validation
    max_text_length: int = 10000
    max_session_duration_hours: int = 24
    max_conversation_turns: int = 1000
    
    # Audit logging
    enable_audit_logging: bool = True
    log_file_path: str = "/workspace/logs/security_audit.log"
    log_retention_days: int = 30
    
    # Data privacy
    anonymize_logs: bool = True
    encrypt_sensitive_data: bool = True
    data_retention_days: int = 7

# ============================================================================
# SECURITY MODELS
# ============================================================================

class UserSession(BaseModel):
    """User session model"""
    
    session_id: str
    ip_address: str
    user_agent: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    expires_at: datetime
    is_authenticated: bool = False
    user_id: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    request_count: int = 0
    data: Dict[str, Any] = Field(default_factory=dict)

class SecurityEvent(BaseModel):
    """Security event for audit logging"""
    
    event_id: str = Field(default_factory=lambda: secrets.token_hex(16))
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str
    severity: str  # low, medium, high, critical
    source_ip: str
    user_agent: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    action_taken: Optional[str] = None

class RateLimitEntry(BaseModel):
    """Rate limiting entry"""
    
    identifier: str  # IP address or user ID
    requests: deque = Field(default_factory=lambda: deque(maxlen=1000))
    blocked_until: Optional[datetime] = None
    total_requests: int = 0
    violations: int = 0

# ============================================================================
# INPUT VALIDATION
# ============================================================================

class InputValidator:
    """Comprehensive input validation system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Compile regex patterns for efficiency
        self.patterns = {
            'sql_injection': re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)', re.IGNORECASE),
            'xss': re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            'path_traversal': re.compile(r'\.\.[\\/]'),
            'command_injection': re.compile(r'[;&|`$(){}[\]<>]'),
            'suspicious_chars': re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
        }
    
    def validate_text_input(self, text: str, field_name: str = "text") -> Tuple[bool, Optional[str]]:
        """Validate text input for security threats"""
        
        if not isinstance(text, str):
            return False, f"{field_name} must be a string"
        
        # Length check
        if len(text) > self.config.max_text_length:
            return False, f"{field_name} exceeds maximum length of {self.config.max_text_length}"
        
        # Check for suspicious patterns
        for pattern_name, pattern in self.patterns.items():
            if pattern.search(text):
                self.logger.warning(f"Suspicious pattern detected in {field_name}: {pattern_name}")
                return False, f"{field_name} contains potentially malicious content"
        
        # Check for control characters
        if self.patterns['suspicious_chars'].search(text):
            return False, f"{field_name} contains invalid characters"
        
        return True, None
    
    def validate_file_upload(self, file_path: Path, original_filename: str) -> Tuple[bool, Optional[str]]:
        """Validate uploaded file for security"""
        
        try:
            # Check file size
            file_size = file_path.stat().st_size
            max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                return False, f"File size exceeds maximum of {self.config.max_file_size_mb}MB"
            
            # Check file extension
            file_extension = Path(original_filename).suffix.lower()
            if file_extension not in self.config.allowed_audio_extensions:
                return False, f"File extension {file_extension} not allowed"
            
            # Check MIME type using python-magic
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
                if mime_type not in self.config.allowed_mime_types:
                    return False, f"File type {mime_type} not allowed"
            except Exception as e:
                self.logger.warning(f"Could not determine MIME type: {e}")
                # Fallback to extension-based validation
                pass
            
            # Check for embedded executables or suspicious content
            if self._scan_file_content(file_path):
                return False, "File contains suspicious content"
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"File validation error: {e}")
            return False, "File validation failed"
    
    def _scan_file_content(self, file_path: Path) -> bool:
        """Scan file content for suspicious patterns"""
        
        try:
            # Read first 1KB to check for executable signatures
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            # Check for common executable signatures
            executable_signatures = [
                b'MZ',      # Windows PE
                b'\x7fELF', # Linux ELF
                b'\xfe\xed\xfa', # macOS Mach-O
                b'#!/bin/',  # Shell script
                b'#!/usr/',  # Shell script
                b'<script',  # HTML/JS
                b'<?php',    # PHP
            ]
            
            for signature in executable_signatures:
                if header.startswith(signature) or signature in header:
                    self.logger.warning(f"Suspicious file signature detected: {signature}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"File content scan error: {e}")
            return True  # Err on the side of caution

# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Advanced rate limiting system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.entries: Dict[str, RateLimitEntry] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def check_rate_limit(self, identifier: str, request_type: str = "general") -> Tuple[bool, Optional[str]]:
        """Check if request is within rate limits"""
        
        current_time = datetime.now()
        
        # Cleanup old entries periodically
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries()
        
        # Get or create rate limit entry
        if identifier not in self.entries:
            self.entries[identifier] = RateLimitEntry(identifier=identifier)
        
        entry = self.entries[identifier]
        
        # Check if currently blocked
        if entry.blocked_until and current_time < entry.blocked_until:
            remaining_seconds = (entry.blocked_until - current_time).total_seconds()
            return False, f"Rate limit exceeded. Try again in {int(remaining_seconds)} seconds"
        
        # Clear block if expired
        if entry.blocked_until and current_time >= entry.blocked_until:
            entry.blocked_until = None
        
        # Add current request
        entry.requests.append(current_time)
        entry.total_requests += 1
        
        # Check per-minute limit
        minute_ago = current_time - timedelta(minutes=1)
        recent_requests = [req for req in entry.requests if req > minute_ago]
        
        if len(recent_requests) > self.config.requests_per_minute:
            # Block for 1 minute
            entry.blocked_until = current_time + timedelta(minutes=1)
            entry.violations += 1
            
            self.logger.warning(f"Rate limit exceeded for {identifier}: {len(recent_requests)} requests/minute")
            return False, "Rate limit exceeded: too many requests per minute"
        
        # Check per-hour limit
        hour_ago = current_time - timedelta(hours=1)
        hourly_requests = [req for req in entry.requests if req > hour_ago]
        
        if len(hourly_requests) > self.config.requests_per_hour:
            # Block for 1 hour
            entry.blocked_until = current_time + timedelta(hours=1)
            entry.violations += 1
            
            self.logger.warning(f"Rate limit exceeded for {identifier}: {len(hourly_requests)} requests/hour")
            return False, "Rate limit exceeded: too many requests per hour"
        
        return True, None
    
    def _cleanup_old_entries(self):
        """Clean up old rate limiting entries"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)
        
        to_remove = []
        for identifier, entry in self.entries.items():
            # Remove entries with no recent activity
            if entry.requests and entry.requests[-1] < cutoff_time:
                to_remove.append(identifier)
        
        for identifier in to_remove:
            del self.entries[identifier]
        
        self.last_cleanup = time.time()
        self.logger.debug(f"Cleaned up {len(to_remove)} old rate limit entries")

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class SessionManager:
    """Secure session management system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sessions: Dict[str, UserSession] = {}
        self.ip_session_count: Dict[str, int] = defaultdict(int)
        
        # Initialize encryption for sensitive data
        if config.encrypt_sensitive_data:
            self.cipher = Fernet(Fernet.generate_key())
        else:
            self.cipher = None
    
    def create_session(self, ip_address: str, user_agent: str, user_id: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Create a new user session"""
        
        # Check concurrent session limit per IP
        if self.ip_session_count[ip_address] >= self.config.concurrent_sessions_per_ip:
            self.logger.warning(f"Too many concurrent sessions from IP: {ip_address}")
            return None, "Too many concurrent sessions from this IP address"
        
        # Generate secure session ID
        session_id = secrets.token_urlsafe(32)
        
        # Create session
        expires_at = datetime.now() + timedelta(hours=self.config.max_session_duration_hours)
        
        session = UserSession(
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
            user_id=user_id
        )
        
        self.sessions[session_id] = session
        self.ip_session_count[ip_address] += 1
        
        self.logger.info(f"Created session {session_id} for IP {ip_address}")
        return session_id, None
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session expired
        if datetime.now() > session.expires_at:
            self.destroy_session(session_id)
            return None
        
        # Update last activity
        session.last_activity = datetime.now()
        return session
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy a session"""
        
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        self.ip_session_count[session.ip_address] -= 1
        
        if self.ip_session_count[session.ip_address] <= 0:
            del self.ip_session_count[session.ip_address]
        
        del self.sessions[session_id]
        
        self.logger.info(f"Destroyed session {session_id}")
        return True
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time > session.expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.destroy_session(session_id)
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# ============================================================================
# AUDIT LOGGING
# ============================================================================

class AuditLogger:
    """Security audit logging system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup audit log file
        self.audit_log_path = Path(config.log_file_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup file handler for audit logs
        self.audit_handler = logging.FileHandler(self.audit_log_path)
        self.audit_handler.setFormatter(
            logging.Formatter('%(asctime)s - AUDIT - %(message)s')
        )
        
        self.audit_logger = logging.getLogger('security_audit')
        self.audit_logger.addHandler(self.audit_handler)
        self.audit_logger.setLevel(logging.INFO)
    
    def log_security_event(self, event: SecurityEvent):
        """Log a security event"""
        
        # Anonymize sensitive data if configured
        if self.config.anonymize_logs:
            event = self._anonymize_event(event)
        
        # Log to audit file
        event_data = event.dict()
        self.audit_logger.info(json.dumps(event_data))
        
        # Log to main logger based on severity
        if event.severity == "critical":
            self.logger.critical(f"SECURITY: {event.description}")
        elif event.severity == "high":
            self.logger.error(f"SECURITY: {event.description}")
        elif event.severity == "medium":
            self.logger.warning(f"SECURITY: {event.description}")
        else:
            self.logger.info(f"SECURITY: {event.description}")
    
    def _anonymize_event(self, event: SecurityEvent) -> SecurityEvent:
        """Anonymize sensitive data in security event"""
        
        # Hash IP address
        if event.source_ip:
            event.source_ip = hashlib.sha256(event.source_ip.encode()).hexdigest()[:16]
        
        # Truncate user agent
        if event.user_agent and len(event.user_agent) > 50:
            event.user_agent = event.user_agent[:50] + "..."
        
        # Remove sensitive details
        if event.details:
            sensitive_keys = ['password', 'token', 'key', 'secret']
            for key in sensitive_keys:
                if key in event.details:
                    event.details[key] = "[REDACTED]"
        
        return event
    
    def cleanup_old_logs(self):
        """Clean up old audit logs"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.log_retention_days)
            
            # This is a simplified cleanup - in production, you might want
            # to implement log rotation with proper archiving
            if self.audit_log_path.exists():
                stat = self.audit_log_path.stat()
                if datetime.fromtimestamp(stat.st_mtime) < cutoff_date:
                    # Archive old log
                    archive_path = self.audit_log_path.with_suffix('.old')
                    self.audit_log_path.rename(archive_path)
                    self.logger.info("Archived old audit log")
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")

# ============================================================================
# MAIN SECURITY SYSTEM
# ============================================================================

class SecuritySystem:
    """Main security system coordinator"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.input_validator = InputValidator(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.session_manager = SessionManager(self.config)
        self.audit_logger = AuditLogger(self.config)
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        async def cleanup_task():
            while True:
                try:
                    # Cleanup expired sessions
                    self.session_manager.cleanup_expired_sessions()
                    
                    # Cleanup old audit logs
                    self.audit_logger.cleanup_old_logs()
                    
                    await asyncio.sleep(300)  # Run every 5 minutes
                    
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
                    await asyncio.sleep(60)
        
        # Start cleanup task
        asyncio.create_task(cleanup_task())
    
    def validate_request(self, ip_address: str, user_agent: str, session_id: str = None) -> Tuple[bool, Optional[str], Optional[UserSession]]:
        """Validate incoming request"""
        
        # Check rate limiting
        if self.config.enable_rate_limiting:
            rate_ok, rate_msg = self.rate_limiter.check_rate_limit(ip_address)
            if not rate_ok:
                self._log_security_event(
                    "rate_limit_exceeded",
                    "medium",
                    ip_address,
                    user_agent,
                    f"Rate limit exceeded: {rate_msg}"
                )
                return False, rate_msg, None
        
        # Validate session if provided
        session = None
        if session_id:
            session = self.session_manager.get_session(session_id)
            if not session:
                self._log_security_event(
                    "invalid_session",
                    "medium",
                    ip_address,
                    user_agent,
                    f"Invalid session ID: {session_id}"
                )
                return False, "Invalid session", None
            
            # Update session activity
            session.request_count += 1
        
        return True, None, session
    
    def validate_text_input(self, text: str, field_name: str = "text") -> Tuple[bool, Optional[str]]:
        """Validate text input"""
        return self.input_validator.validate_text_input(text, field_name)
    
    def validate_file_upload(self, file_path: Path, original_filename: str, ip_address: str, user_agent: str) -> Tuple[bool, Optional[str]]:
        """Validate file upload"""
        
        is_valid, error_msg = self.input_validator.validate_file_upload(file_path, original_filename)
        
        if not is_valid:
            self._log_security_event(
                "malicious_file_upload",
                "high",
                ip_address,
                user_agent,
                f"Malicious file upload attempt: {original_filename} - {error_msg}"
            )
        
        return is_valid, error_msg
    
    def create_session(self, ip_address: str, user_agent: str, user_id: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Create new session"""
        
        session_id, error_msg = self.session_manager.create_session(ip_address, user_agent, user_id)
        
        if session_id:
            self._log_security_event(
                "session_created",
                "low",
                ip_address,
                user_agent,
                f"New session created: {session_id}"
            )
        
        return session_id, error_msg
    
    def destroy_session(self, session_id: str, ip_address: str, user_agent: str) -> bool:
        """Destroy session"""
        
        success = self.session_manager.destroy_session(session_id)
        
        if success:
            self._log_security_event(
                "session_destroyed",
                "low",
                ip_address,
                user_agent,
                f"Session destroyed: {session_id}"
            )
        
        return success
    
    def _log_security_event(self, event_type: str, severity: str, ip_address: str, 
                          user_agent: str, description: str, details: Dict[str, Any] = None):
        """Log security event"""
        
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            source_ip=ip_address,
            user_agent=user_agent,
            description=description,
            details=details or {}
        )
        
        self.audit_logger.log_security_event(event)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security system status"""
        
        return {
            "active_sessions": len(self.session_manager.sessions),
            "rate_limit_entries": len(self.rate_limiter.entries),
            "config": {
                "authentication_enabled": self.config.enable_authentication,
                "rate_limiting_enabled": self.config.enable_rate_limiting,
                "audit_logging_enabled": self.config.enable_audit_logging,
                "max_file_size_mb": self.config.max_file_size_mb,
                "max_text_length": self.config.max_text_length
            }
        }

# ============================================================================
# SECURITY MIDDLEWARE
# ============================================================================

class SecurityMiddleware:
    """Security middleware for web applications"""
    
    def __init__(self, security_system: SecuritySystem):
        self.security_system = security_system
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self, request, call_next):
        """Process request through security middleware"""
        
        # Extract request information
        ip_address = request.client.host
        user_agent = request.headers.get("user-agent", "")
        session_id = request.cookies.get("session_id")
        
        # Validate request
        is_valid, error_msg, session = self.security_system.validate_request(
            ip_address, user_agent, session_id
        )
        
        if not is_valid:
            from fastapi import HTTPException
            raise HTTPException(status_code=429, detail=error_msg)
        
        # Add session to request state
        request.state.session = session
        request.state.security_validated = True
        
        # Process request
        response = await call_next(request)
        
        return response

# ============================================================================
# INITIALIZATION
# ============================================================================

def create_security_system(config_file: str = None) -> SecuritySystem:
    """Create and configure security system"""
    
    config = SecurityConfig()
    
    # Load configuration from file if provided
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                # Update config with loaded data
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load security config: {e}")
    
    return SecuritySystem(config)

# Export main components
__all__ = [
    'SecurityConfig', 'SecuritySystem', 'SecurityMiddleware',
    'UserSession', 'SecurityEvent', 'InputValidator', 'RateLimiter',
    'SessionManager', 'AuditLogger', 'create_security_system'
]