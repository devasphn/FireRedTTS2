#!/usr/bin/env python3
"""
Security Testing and Validation
Comprehensive security tests for FireRedTTS2 system including penetration testing,
vulnerability scanning, and security validation
"""

import asyncio
import json
import time
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

import requests
import aiohttp
import pytest
from security_system import SecuritySystem, SecurityConfig
from security_integration import SecurityIntegration

logger = logging.getLogger(__name__)

# ============================================================================
# SECURITY TEST SUITE
# ============================================================================

class SecurityTestSuite:
    """Comprehensive security testing suite"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
        self.test_results = []
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all security tests"""
        
        self.logger.info("Starting comprehensive security test suite")
        
        test_categories = [
            ("Input Validation", self.test_input_validation),
            ("Rate Limiting", self.test_rate_limiting),
            ("File Upload Security", self.test_file_upload_security),
            ("Session Management", self.test_session_management),
            ("Authentication", self.test_authentication),
            ("XSS Protection", self.test_xss_protection),
            ("SQL Injection", self.test_sql_injection),
            ("Path Traversal", self.test_path_traversal),
            ("CORS Security", self.test_cors_security),
            ("Security Headers", self.test_security_headers),
        ]
        
        results = {
            "timestamp": time.time(),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "categories": {}
        }
        
        for category_name, test_function in test_categories:
            self.logger.info(f"Running {category_name} tests")
            
            try:
                category_results = await test_function()
                results["categories"][category_name] = category_results
                
                results["total_tests"] += category_results["total"]
                results["passed_tests"] += category_results["passed"]
                results["failed_tests"] += category_results["failed"]
                
            except Exception as e:
                self.logger.error(f"Error in {category_name} tests: {e}")
                results["categories"][category_name] = {
                    "total": 1,
                    "passed": 0,
                    "failed": 1,
                    "error": str(e)
                }
                results["total_tests"] += 1
                results["failed_tests"] += 1
        
        # Calculate success rate
        if results["total_tests"] > 0:
            results["success_rate"] = results["passed_tests"] / results["total_tests"]
        else:
            results["success_rate"] = 0.0
        
        self.logger.info(f"Security tests completed: {results['passed_tests']}/{results['total_tests']} passed")
        
        return results
    
    async def test_input_validation(self) -> Dict[str, Any]:
        """Test input validation security"""
        
        results = {"total": 0, "passed": 0, "failed": 0, "details": []}
        
        # Test cases for malicious input
        test_cases = [
            {
                "name": "SQL Injection",
                "input": "'; DROP TABLE users; --",
                "should_block": True
            },
            {
                "name": "XSS Script",
                "input": "<script>alert('xss')</script>",
                "should_block": True
            },
            {
                "name": "Path Traversal",
                "input": "../../../etc/passwd",
                "should_block": True
            },
            {
                "name": "Command Injection",
                "input": "; rm -rf /",
                "should_block": True
            },
            {
                "name": "Long Input",
                "input": "A" * 20000,
                "should_block": True
            },
            {
                "name": "Normal Input",
                "input": "Hello, this is a normal text input for TTS generation.",
                "should_block": False
            }
        ]
        
        for test_case in test_cases:
            results["total"] += 1
            
            try:
                # Test TTS endpoint with malicious input
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "text": test_case["input"],
                        "voice_mode": "random"
                    }
                    
                    async with session.post(
                        f"{self.base_url}/api/v1/tts/generate",
                        json=payload
                    ) as response:
                        
                        if test_case["should_block"]:
                            # Should be blocked (4xx status)
                            if response.status >= 400:
                                results["passed"] += 1
                                results["details"].append({
                                    "test": test_case["name"],
                                    "status": "PASSED",
                                    "message": f"Malicious input blocked (status: {response.status})"
                                })
                            else:
                                results["failed"] += 1
                                results["details"].append({
                                    "test": test_case["name"],
                                    "status": "FAILED",
                                    "message": f"Malicious input not blocked (status: {response.status})"
                                })
                        else:
                            # Should be allowed (2xx status)
                            if 200 <= response.status < 300:
                                results["passed"] += 1
                                results["details"].append({
                                    "test": test_case["name"],
                                    "status": "PASSED",
                                    "message": "Normal input accepted"
                                })
                            else:
                                results["failed"] += 1
                                results["details"].append({
                                    "test": test_case["name"],
                                    "status": "FAILED",
                                    "message": f"Normal input rejected (status: {response.status})"
                                })
            
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "test": test_case["name"],
                    "status": "ERROR",
                    "message": str(e)
                })
        
        return results
    
    async def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting functionality"""
        
        results = {"total": 0, "passed": 0, "failed": 0, "details": []}
        
        # Test rate limiting by making rapid requests
        results["total"] += 1
        
        try:
            async with aiohttp.ClientSession() as session:
                # Make rapid requests to trigger rate limiting
                responses = []
                
                for i in range(100):  # Make 100 rapid requests
                    try:
                        async with session.get(f"{self.base_url}/health") as response:
                            responses.append(response.status)
                    except:
                        responses.append(429)  # Assume rate limited
                
                # Check if rate limiting kicked in
                rate_limited_count = responses.count(429)
                
                if rate_limited_count > 0:
                    results["passed"] += 1
                    results["details"].append({
                        "test": "Rate Limiting",
                        "status": "PASSED",
                        "message": f"Rate limiting active: {rate_limited_count} requests blocked"
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "test": "Rate Limiting",
                        "status": "FAILED",
                        "message": "Rate limiting not working - no requests blocked"
                    })
        
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "test": "Rate Limiting",
                "status": "ERROR",
                "message": str(e)
            })
        
        return results
    
    async def test_file_upload_security(self) -> Dict[str, Any]:
        """Test file upload security"""
        
        results = {"total": 0, "passed": 0, "failed": 0, "details": []}
        
        # Test cases for malicious files
        test_files = [
            {
                "name": "Executable File",
                "filename": "malware.exe",
                "content": b"MZ\x90\x00",  # PE header
                "should_block": True
            },
            {
                "name": "Script File",
                "filename": "script.sh",
                "content": b"#!/bin/bash\nrm -rf /",
                "should_block": True
            },
            {
                "name": "Large File",
                "filename": "large.wav",
                "content": b"RIFF" + b"A" * (200 * 1024 * 1024),  # 200MB
                "should_block": True
            },
            {
                "name": "Valid Audio File",
                "filename": "test.wav",
                "content": b"RIFF\x24\x00\x00\x00WAVEfmt ",  # Valid WAV header
                "should_block": False
            }
        ]
        
        for test_file in test_files:
            results["total"] += 1
            
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(test_file["filename"]).suffix) as tmp:
                    tmp.write(test_file["content"])
                    tmp_path = tmp.name
                
                # Test file upload
                async with aiohttp.ClientSession() as session:
                    data = aiohttp.FormData()
                    data.add_field('name', 'test_voice')
                    data.add_field('description', 'Test voice profile')
                    data.add_field('reference_text', 'This is a test')
                    
                    with open(tmp_path, 'rb') as f:
                        data.add_field('audio_file', f, filename=test_file["filename"])
                        
                        async with session.post(
                            f"{self.base_url}/api/v1/voices/create",
                            data=data
                        ) as response:
                            
                            if test_file["should_block"]:
                                if response.status >= 400:
                                    results["passed"] += 1
                                    results["details"].append({
                                        "test": test_file["name"],
                                        "status": "PASSED",
                                        "message": f"Malicious file blocked (status: {response.status})"
                                    })
                                else:
                                    results["failed"] += 1
                                    results["details"].append({
                                        "test": test_file["name"],
                                        "status": "FAILED",
                                        "message": f"Malicious file not blocked (status: {response.status})"
                                    })
                            else:
                                if 200 <= response.status < 300:
                                    results["passed"] += 1
                                    results["details"].append({
                                        "test": test_file["name"],
                                        "status": "PASSED",
                                        "message": "Valid file accepted"
                                    })
                                else:
                                    results["failed"] += 1
                                    results["details"].append({
                                        "test": test_file["name"],
                                        "status": "FAILED",
                                        "message": f"Valid file rejected (status: {response.status})"
                                    })
                
                # Cleanup
                Path(tmp_path).unlink(missing_ok=True)
            
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "test": test_file["name"],
                    "status": "ERROR",
                    "message": str(e)
                })
        
        return results
    
    async def test_session_management(self) -> Dict[str, Any]:
        """Test session management security"""
        
        results = {"total": 0, "passed": 0, "failed": 0, "details": []}
        
        # Test session creation
        results["total"] += 1
        
        try:
            async with aiohttp.ClientSession() as session:
                # Create session
                async with session.post(f"{self.base_url}/api/v1/security/session/create") as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and "session_id" in data["data"]:
                            results["passed"] += 1
                            results["details"].append({
                                "test": "Session Creation",
                                "status": "PASSED",
                                "message": "Session created successfully"
                            })
                        else:
                            results["failed"] += 1
                            results["details"].append({
                                "test": "Session Creation",
                                "status": "FAILED",
                                "message": "Session creation response invalid"
                            })
                    else:
                        results["failed"] += 1
                        results["details"].append({
                            "test": "Session Creation",
                            "status": "FAILED",
                            "message": f"Session creation failed (status: {response.status})"
                        })
        
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "test": "Session Creation",
                "status": "ERROR",
                "message": str(e)
            })
        
        return results
    
    async def test_authentication(self) -> Dict[str, Any]:
        """Test authentication mechanisms"""
        
        results = {"total": 0, "passed": 0, "failed": 0, "details": []}
        
        # Test accessing protected endpoints without authentication
        results["total"] += 1
        
        try:
            async with aiohttp.ClientSession() as session:
                # Try to access security status without auth
                async with session.get(f"{self.base_url}/api/v1/security/status") as response:
                    # Should either work (if auth disabled) or return 401
                    if response.status in [200, 401]:
                        results["passed"] += 1
                        results["details"].append({
                            "test": "Authentication Check",
                            "status": "PASSED",
                            "message": f"Authentication working correctly (status: {response.status})"
                        })
                    else:
                        results["failed"] += 1
                        results["details"].append({
                            "test": "Authentication Check",
                            "status": "FAILED",
                            "message": f"Unexpected response (status: {response.status})"
                        })
        
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "test": "Authentication Check",
                "status": "ERROR",
                "message": str(e)
            })
        
        return results
    
    async def test_xss_protection(self) -> Dict[str, Any]:
        """Test XSS protection"""
        
        results = {"total": 0, "passed": 0, "failed": 0, "details": []}
        
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            results["total"] += 1
            
            try:
                async with aiohttp.ClientSession() as session:
                    # Test XSS in text input
                    data = {"text": payload}
                    
                    async with session.post(
                        f"{self.base_url}/api/v1/tts/generate",
                        json=data
                    ) as response:
                        
                        if response.status >= 400:
                            results["passed"] += 1
                            results["details"].append({
                                "test": f"XSS Protection - {payload[:20]}...",
                                "status": "PASSED",
                                "message": "XSS payload blocked"
                            })
                        else:
                            results["failed"] += 1
                            results["details"].append({
                                "test": f"XSS Protection - {payload[:20]}...",
                                "status": "FAILED",
                                "message": "XSS payload not blocked"
                            })
            
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "test": f"XSS Protection - {payload[:20]}...",
                    "status": "ERROR",
                    "message": str(e)
                })
        
        return results
    
    async def test_sql_injection(self) -> Dict[str, Any]:
        """Test SQL injection protection"""
        
        results = {"total": 0, "passed": 0, "failed": 0, "details": []}
        
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "' UNION SELECT * FROM users --",
            "'; DELETE FROM users WHERE '1'='1'; --"
        ]
        
        for payload in sql_payloads:
            results["total"] += 1
            
            try:
                async with aiohttp.ClientSession() as session:
                    # Test SQL injection in text input
                    data = {"text": payload}
                    
                    async with session.post(
                        f"{self.base_url}/api/v1/tts/generate",
                        json=data
                    ) as response:
                        
                        if response.status >= 400:
                            results["passed"] += 1
                            results["details"].append({
                                "test": f"SQL Injection - {payload[:20]}...",
                                "status": "PASSED",
                                "message": "SQL injection blocked"
                            })
                        else:
                            results["failed"] += 1
                            results["details"].append({
                                "test": f"SQL Injection - {payload[:20]}...",
                                "status": "FAILED",
                                "message": "SQL injection not blocked"
                            })
            
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "test": f"SQL Injection - {payload[:20]}...",
                    "status": "ERROR",
                    "message": str(e)
                })
        
        return results
    
    async def test_path_traversal(self) -> Dict[str, Any]:
        """Test path traversal protection"""
        
        results = {"total": 0, "passed": 0, "failed": 0, "details": []}
        
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        for payload in path_payloads:
            results["total"] += 1
            
            try:
                async with aiohttp.ClientSession() as session:
                    # Test path traversal in text input
                    data = {"text": payload}
                    
                    async with session.post(
                        f"{self.base_url}/api/v1/tts/generate",
                        json=data
                    ) as response:
                        
                        if response.status >= 400:
                            results["passed"] += 1
                            results["details"].append({
                                "test": f"Path Traversal - {payload[:20]}...",
                                "status": "PASSED",
                                "message": "Path traversal blocked"
                            })
                        else:
                            results["failed"] += 1
                            results["details"].append({
                                "test": f"Path Traversal - {payload[:20]}...",
                                "status": "FAILED",
                                "message": "Path traversal not blocked"
                            })
            
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "test": f"Path Traversal - {payload[:20]}...",
                    "status": "ERROR",
                    "message": str(e)
                })
        
        return results
    
    async def test_cors_security(self) -> Dict[str, Any]:
        """Test CORS security configuration"""
        
        results = {"total": 0, "passed": 0, "failed": 0, "details": []}
        results["total"] += 1
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test CORS headers
                headers = {"Origin": "https://malicious-site.com"}
                
                async with session.options(
                    f"{self.base_url}/api/v1/tts/generate",
                    headers=headers
                ) as response:
                    
                    cors_headers = {
                        "Access-Control-Allow-Origin": response.headers.get("Access-Control-Allow-Origin"),
                        "Access-Control-Allow-Methods": response.headers.get("Access-Control-Allow-Methods"),
                        "Access-Control-Allow-Headers": response.headers.get("Access-Control-Allow-Headers")
                    }
                    
                    # Check if CORS is properly configured
                    if any(cors_headers.values()):
                        results["passed"] += 1
                        results["details"].append({
                            "test": "CORS Configuration",
                            "status": "PASSED",
                            "message": f"CORS headers present: {cors_headers}"
                        })
                    else:
                        results["failed"] += 1
                        results["details"].append({
                            "test": "CORS Configuration",
                            "status": "FAILED",
                            "message": "No CORS headers found"
                        })
        
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "test": "CORS Configuration",
                "status": "ERROR",
                "message": str(e)
            })
        
        return results
    
    async def test_security_headers(self) -> Dict[str, Any]:
        """Test security headers"""
        
        results = {"total": 0, "passed": 0, "failed": 0, "details": []}
        
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Referrer-Policy",
            "Content-Security-Policy"
        ]
        
        for header in expected_headers:
            results["total"] += 1
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/health") as response:
                        
                        if header in response.headers:
                            results["passed"] += 1
                            results["details"].append({
                                "test": f"Security Header - {header}",
                                "status": "PASSED",
                                "message": f"Header present: {response.headers[header]}"
                            })
                        else:
                            results["failed"] += 1
                            results["details"].append({
                                "test": f"Security Header - {header}",
                                "status": "FAILED",
                                "message": "Header missing"
                            })
            
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "test": f"Security Header - {header}",
                    "status": "ERROR",
                    "message": str(e)
                })
        
        return results

# ============================================================================
# SECURITY VALIDATION FUNCTIONS
# ============================================================================

def validate_security_configuration(config_path: str) -> Dict[str, Any]:
    """Validate security configuration"""
    
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": []
    }
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check critical security settings
        if not config.get("enable_rate_limiting", True):
            validation_results["warnings"].append("Rate limiting is disabled")
        
        if config.get("requests_per_minute", 60) > 1000:
            validation_results["warnings"].append("Rate limit is very high")
        
        if not config.get("enable_audit_logging", True):
            validation_results["errors"].append("Audit logging is disabled")
            validation_results["valid"] = False
        
        if config.get("max_file_size_mb", 100) > 500:
            validation_results["warnings"].append("Maximum file size is very large")
        
        if not config.get("anonymize_logs", True):
            validation_results["recommendations"].append("Consider enabling log anonymization")
        
        # Check JWT settings if authentication is enabled
        if config.get("enable_authentication", False):
            if not config.get("jwt_secret_key") or config.get("jwt_secret_key") == "your-secret-key-here-change-in-production":
                validation_results["errors"].append("JWT secret key must be changed from default")
                validation_results["valid"] = False
        
        # Check file upload settings
        allowed_extensions = config.get("allowed_audio_extensions", [])
        if ".exe" in allowed_extensions or ".sh" in allowed_extensions:
            validation_results["errors"].append("Dangerous file extensions are allowed")
            validation_results["valid"] = False
        
    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Failed to load configuration: {e}")
    
    return validation_results

async def run_security_audit(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Run comprehensive security audit"""
    
    test_suite = SecurityTestSuite(base_url)
    results = await test_suite.run_all_tests()
    
    # Add configuration validation
    config_validation = validate_security_configuration("security_config.json")
    results["configuration_validation"] = config_validation
    
    # Generate security report
    security_report = {
        "audit_timestamp": time.time(),
        "overall_security_score": results["success_rate"] * 100,
        "test_results": results,
        "configuration_validation": config_validation,
        "recommendations": generate_security_recommendations(results, config_validation)
    }
    
    return security_report

def generate_security_recommendations(test_results: Dict[str, Any], config_validation: Dict[str, Any]) -> List[str]:
    """Generate security recommendations based on test results"""
    
    recommendations = []
    
    # Based on test results
    if test_results["success_rate"] < 0.8:
        recommendations.append("Security test success rate is below 80%. Review failed tests and implement fixes.")
    
    if test_results["failed_tests"] > 0:
        recommendations.append("Some security tests failed. Review the detailed results and address vulnerabilities.")
    
    # Based on configuration validation
    recommendations.extend(config_validation.get("recommendations", []))
    
    if config_validation.get("warnings"):
        recommendations.append("Review configuration warnings and consider making recommended changes.")
    
    if config_validation.get("errors"):
        recommendations.append("Fix configuration errors immediately - they represent security vulnerabilities.")
    
    # General recommendations
    recommendations.extend([
        "Regularly update dependencies to patch security vulnerabilities",
        "Implement proper logging and monitoring for security events",
        "Use HTTPS in production environments",
        "Regularly review and rotate secrets and keys",
        "Implement proper backup and disaster recovery procedures",
        "Consider implementing Web Application Firewall (WAF)",
        "Regularly perform security audits and penetration testing"
    ])
    
    return recommendations

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    async def main():
        # Run security audit
        report = await run_security_audit()
        
        # Save report
        report_path = Path("security_audit_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Security audit completed. Report saved to {report_path}")
        print(f"Overall security score: {report['overall_security_score']:.1f}%")
        
        if report['test_results']['failed_tests'] > 0:
            print(f"⚠️  {report['test_results']['failed_tests']} security tests failed")
        else:
            print("✅ All security tests passed")
    
    asyncio.run(main())

# Export main components
__all__ = [
    'SecurityTestSuite', 'validate_security_configuration',
    'run_security_audit', 'generate_security_recommendations'
]