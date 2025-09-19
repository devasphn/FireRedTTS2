#!/usr/bin/env python3
"""
End-to-End System Validation for FireRedTTS2 Speech-to-Speech System
Comprehensive validation of the complete integrated system with real user scenarios
"""

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch
import aiohttp
import websockets

# Import system components
from main_integration import IntegratedSpeechToSpeechSystem, IntegrationConfig
from data_models import SystemStatus
from quality_assurance_system import QualityAssuranceSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for end-to-end validation"""
    
    # System configuration
    base_url: str = "http://localhost:7860"
    websocket_url: str = "ws://localhost:7860/ws"
    timeout_seconds: float = 30.0
    
    # Test scenarios configuration
    test_scenarios: List[str] = None
    concurrent_users: int = 5
    test_duration_minutes: int = 10
    
    # Performance thresholds
    max_response_time_ms: float = 2000.0
    min_success_rate: float = 0.85
    max_error_rate: float = 0.05
    
    # Audio testing
    test_audio_duration: float = 5.0
    sample_rate: int = 24000
    
    def __post_init__(self):
        if self.test_scenarios is None:
            self.test_scenarios = [
                "basic_tts_generation",
                "speech_to_speech_conversation",
                "voice_cloning",
                "multi_speaker_dialogue",
                "real_time_streaming",
                "concurrent_user_handling",
                "error_recovery",
                "performance_under_load"
            ]

@dataclass
class ValidationResult:
    """Result of a validation test"""
    
    scenario_name: str
    success: bool
    execution_time_seconds: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class TestScenario:
    """Test scenario definition"""
    
    name: str
    description: str
    test_function: str
    expected_duration_seconds: float
    critical: bool = False
    prerequisites: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []

class EndToEndValidator:
    """End-to-end system validator with real user scenarios"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validation state
        self.validation_results: List[ValidationResult] = []
        self.system_under_test: Optional[IntegratedSpeechToSpeechSystem] = None
        self.qa_system: Optional[QualityAssuranceSystem] = None
        
        # Test scenarios
        self.test_scenarios = self._define_test_scenarios()
    
    def _define_test_scenarios(self) -> List[TestScenario]:
        """Define all test scenarios"""
        
        return [
            TestScenario(
                name="basic_tts_generation",
                description="Basic text-to-speech generation functionality",
                test_function="test_basic_tts_generation",
                expected_duration_seconds=5.0,
                critical=True
            ),
            TestScenario(
                name="speech_to_speech_conversation",
                description="Complete speech-to-speech conversation pipeline",
                test_function="test_speech_to_speech_conversation",
                expected_duration_seconds=10.0,
                critical=True,
                prerequisites=["basic_tts_generation"]
            ),
            TestScenario(
                name="voice_cloning",
                description="Voice cloning with reference audio",
                test_function="test_voice_cloning",
                expected_duration_seconds=15.0,
                critical=False
            ),
            TestScenario(
                name="multi_speaker_dialogue",
                description="Multi-speaker dialogue generation",
                test_function="test_multi_speaker_dialogue",
                expected_duration_seconds=12.0,
                critical=False,
                prerequisites=["basic_tts_generation"]
            ),
            TestScenario(
                name="real_time_streaming",
                description="Real-time audio streaming capabilities",
                test_function="test_real_time_streaming",
                expected_duration_seconds=20.0,
                critical=True
            ),
            TestScenario(
                name="concurrent_user_handling",
                description="Concurrent user request handling",
                test_function="test_concurrent_user_handling",
                expected_duration_seconds=30.0,
                critical=True
            ),
            TestScenario(
                name="error_recovery",
                description="System error recovery and resilience",
                test_function="test_error_recovery",
                expected_duration_seconds=15.0,
                critical=True
            ),
            TestScenario(
                name="performance_under_load",
                description="Performance validation under load",
                test_function="test_performance_under_load",
                expected_duration_seconds=60.0,
                critical=True
            )
        ]
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete end-to-end validation"""
        
        self.logger.info("Starting end-to-end system validation")
        start_time = time.time()
        
        try:
            # Initialize system under test
            await self._initialize_system_under_test()
            
            # Run validation scenarios
            for scenario in self.test_scenarios:
                if scenario.name in self.config.test_scenarios:
                    self.logger.info(f"Running scenario: {scenario.name}")
                    
                    # Check prerequisites
                    if not self._check_prerequisites(scenario):
                        result = ValidationResult(
                            scenario_name=scenario.name,
                            success=False,
                            execution_time_seconds=0.0,
                            details={},
                            error_message="Prerequisites not met"
                        )
                        self.validation_results.append(result)
                        continue
                    
                    # Run scenario
                    result = await self._run_scenario(scenario)
                    self.validation_results.append(result)
                    
                    # Stop if critical scenario fails
                    if scenario.critical and not result.success:
                        self.logger.error(f"Critical scenario {scenario.name} failed, stopping validation")
                        break
            
            # Generate validation report
            validation_report = self._generate_validation_report()
            validation_report["total_execution_time_seconds"] = time.time() - start_time
            
            self.logger.info(f"Validation completed in {validation_report['total_execution_time_seconds']:.2f} seconds")
            
            return validation_report
        
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "total_execution_time_seconds": time.time() - start_time
            }
    
    async def _initialize_system_under_test(self):
        """Initialize the system under test"""
        
        try:
            # Create integration configuration
            integration_config = IntegrationConfig(
                pretrained_dir="/workspace/models",
                host="0.0.0.0",
                port=7860,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Initialize system (if not already running)
            # In practice, this would connect to an already running system
            self.logger.info("Connecting to system under test")
            
            # Test basic connectivity
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{self.config.base_url}/health", timeout=10) as response:
                        if response.status == 200:
                            self.logger.info("System connectivity verified")
                        else:
                            self.logger.warning(f"System health check returned status {response.status}")
                except Exception as e:
                    self.logger.warning(f"System connectivity test failed: {e}")
        
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    def _check_prerequisites(self, scenario: TestScenario) -> bool:
        """Check if scenario prerequisites are met"""
        
        if not scenario.prerequisites:
            return True
        
        for prereq in scenario.prerequisites:
            prereq_results = [r for r in self.validation_results if r.scenario_name == prereq]
            if not prereq_results or not prereq_results[-1].success:
                return False
        
        return True
    
    async def _run_scenario(self, scenario: TestScenario) -> ValidationResult:
        """Run a single validation scenario"""
        
        start_time = time.time()
        
        try:
            # Get test function
            test_function = getattr(self, scenario.test_function, None)
            if not test_function:
                raise ValueError(f"Test function {scenario.test_function} not found")
            
            # Run test with timeout
            result_details = await asyncio.wait_for(
                test_function(),
                timeout=scenario.expected_duration_seconds * 2  # 2x expected duration as timeout
            )
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                scenario_name=scenario.name,
                success=result_details.get("success", False),
                execution_time_seconds=execution_time,
                details=result_details
            )
        
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return ValidationResult(
                scenario_name=scenario.name,
                success=False,
                execution_time_seconds=execution_time,
                details={},
                error_message=f"Test timed out after {execution_time:.2f} seconds"
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                scenario_name=scenario.name,
                success=False,
                execution_time_seconds=execution_time,
                details={},
                error_message=str(e)
            )
    
    async def test_basic_tts_generation(self) -> Dict[str, Any]:
        """Test basic TTS generation functionality"""
        
        try:
            test_text = "Hello, this is a test of the FireRedTTS2 text-to-speech system."
            
            # Test via API
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "text": test_text,
                    "voice_mode": "random"
                }
                
                start_time = time.time()
                async with session.post(
                    f"{self.config.base_url}/api/v1/tts/generate",
                    json=request_data,
                    timeout=self.config.timeout_seconds
                ) as response:
                    
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        return {
                            "success": result.get("success", False),
                            "response_time_ms": response_time * 1000,
                            "text_length": len(test_text),
                            "api_response": result
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"API returned status {response.status}",
                            "response_time_ms": response_time * 1000
                        }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_speech_to_speech_conversation(self) -> Dict[str, Any]:
        """Test complete speech-to-speech conversation pipeline"""
        
        try:
            # Test conversation processing
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "user_input": "Hello, how are you today?",
                    "input_type": "text"
                }
                
                start_time = time.time()
                async with session.post(
                    f"{self.config.base_url}/api/v1/conversation/process",
                    json=request_data,
                    timeout=self.config.timeout_seconds
                ) as response:
                    
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        return {
                            "success": result.get("success", False),
                            "response_time_ms": response_time * 1000,
                            "conversation_result": result
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"API returned status {response.status}",
                            "response_time_ms": response_time * 1000
                        }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_voice_cloning(self) -> Dict[str, Any]:
        """Test voice cloning functionality"""
        
        try:
            # Voice cloning test would require reference audio
            # For now, we test the voice profile management API
            
            async with aiohttp.ClientSession() as session:
                # Test voice profile listing
                async with session.get(
                    f"{self.config.base_url}/api/v1/voices",
                    timeout=self.config.timeout_seconds
                ) as response:
                    
                    if response.status == 200:
                        voices = await response.json()
                        
                        return {
                            "success": True,
                            "voices_available": len(voices.get("items", [])),
                            "voice_profiles": voices
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Voice API returned status {response.status}"
                        }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_multi_speaker_dialogue(self) -> Dict[str, Any]:
        """Test multi-speaker dialogue generation"""
        
        try:
            dialogue_text = "Speaker 1: Hello there! Speaker 2: Hi, how are you doing today?"
            
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "text": dialogue_text,
                    "voice_mode": "multi_speaker"
                }
                
                start_time = time.time()
                async with session.post(
                    f"{self.config.base_url}/api/v1/tts/generate",
                    json=request_data,
                    timeout=self.config.timeout_seconds
                ) as response:
                    
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        return {
                            "success": result.get("success", False),
                            "response_time_ms": response_time * 1000,
                            "dialogue_length": len(dialogue_text),
                            "generation_result": result
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"API returned status {response.status}",
                            "response_time_ms": response_time * 1000
                        }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_real_time_streaming(self) -> Dict[str, Any]:
        """Test real-time audio streaming capabilities"""
        
        try:
            # Test WebSocket connection
            websocket_url = f"{self.config.websocket_url}/test_client"
            
            async with websockets.connect(websocket_url, timeout=self.config.timeout_seconds) as websocket:
                # Send test message
                test_message = {
                    "type": "ping",
                    "timestamp": time.time()
                }
                
                start_time = time.time()
                await websocket.send(json.dumps(test_message))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_time = time.time() - start_time
                
                response_data = json.loads(response)
                
                return {
                    "success": response_data.get("type") == "pong",
                    "response_time_ms": response_time * 1000,
                    "websocket_response": response_data
                }
        
        except (ConnectionRefusedError, OSError) as e:
            return {
                "success": False,
                "error": f"WebSocket connection failed: {e}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_concurrent_user_handling(self) -> Dict[str, Any]:
        """Test concurrent user request handling"""
        
        try:
            # Create multiple concurrent requests
            async def make_request():
                async with aiohttp.ClientSession() as session:
                    request_data = {
                        "text": "Concurrent test request",
                        "voice_mode": "random"
                    }
                    
                    start_time = time.time()
                    try:
                        async with session.post(
                            f"{self.config.base_url}/api/v1/tts/generate",
                            json=request_data,
                            timeout=self.config.timeout_seconds
                        ) as response:
                            
                            response_time = time.time() - start_time
                            
                            return {
                                "success": response.status == 200,
                                "response_time_ms": response_time * 1000,
                                "status_code": response.status
                            }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "response_time_ms": (time.time() - start_time) * 1000
                        }
            
            # Run concurrent requests
            tasks = [make_request() for _ in range(self.config.concurrent_users)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful_requests = [r for r in results if isinstance(r, dict) and r.get("success", False)]
            response_times = [r.get("response_time_ms", 0) for r in results if isinstance(r, dict)]
            
            success_rate = len(successful_requests) / len(results) if results else 0
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            return {
                "success": success_rate >= self.config.min_success_rate,
                "concurrent_users": self.config.concurrent_users,
                "successful_requests": len(successful_requests),
                "success_rate": success_rate,
                "average_response_time_ms": avg_response_time,
                "max_response_time_ms": max(response_times) if response_times else 0
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_error_recovery(self) -> Dict[str, Any]:
        """Test system error recovery and resilience"""
        
        try:
            error_tests = []
            
            # Test invalid input handling
            async with aiohttp.ClientSession() as session:
                # Test empty text
                request_data = {"text": "", "voice_mode": "random"}
                async with session.post(
                    f"{self.config.base_url}/api/v1/tts/generate",
                    json=request_data,
                    timeout=self.config.timeout_seconds
                ) as response:
                    
                    error_tests.append({
                        "test": "empty_text",
                        "success": response.status == 400,  # Should return bad request
                        "status_code": response.status
                    })
                
                # Test invalid voice mode
                request_data = {"text": "Test", "voice_mode": "invalid_mode"}
                async with session.post(
                    f"{self.config.base_url}/api/v1/tts/generate",
                    json=request_data,
                    timeout=self.config.timeout_seconds
                ) as response:
                    
                    error_tests.append({
                        "test": "invalid_voice_mode",
                        "success": response.status in [400, 422],  # Should return error
                        "status_code": response.status
                    })
                
                # Test very long text
                long_text = "This is a very long text. " * 1000  # Very long text
                request_data = {"text": long_text, "voice_mode": "random"}
                async with session.post(
                    f"{self.config.base_url}/api/v1/tts/generate",
                    json=request_data,
                    timeout=self.config.timeout_seconds
                ) as response:
                    
                    error_tests.append({
                        "test": "very_long_text",
                        "success": response.status in [200, 413, 422],  # Should handle gracefully
                        "status_code": response.status
                    })
            
            # Calculate overall error handling success
            successful_error_tests = sum(1 for test in error_tests if test["success"])
            error_handling_success_rate = successful_error_tests / len(error_tests) if error_tests else 0
            
            return {
                "success": error_handling_success_rate >= 0.8,  # 80% of error tests should pass
                "error_tests": error_tests,
                "error_handling_success_rate": error_handling_success_rate
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_performance_under_load(self) -> Dict[str, Any]:
        """Test performance validation under load"""
        
        try:
            # Run sustained load test
            load_duration = min(self.config.test_duration_minutes * 60, 60)  # Max 1 minute for validation
            end_time = time.time() + load_duration
            
            request_count = 0
            successful_requests = 0
            response_times = []
            
            async def load_request():
                nonlocal request_count, successful_requests
                
                async with aiohttp.ClientSession() as session:
                    request_data = {
                        "text": f"Load test request {request_count}",
                        "voice_mode": "random"
                    }
                    
                    start_time = time.time()
                    try:
                        async with session.post(
                            f"{self.config.base_url}/api/v1/tts/generate",
                            json=request_data,
                            timeout=self.config.timeout_seconds
                        ) as response:
                            
                            response_time = time.time() - start_time
                            response_times.append(response_time * 1000)
                            
                            if response.status == 200:
                                successful_requests += 1
                            
                            request_count += 1
                    
                    except Exception:
                        request_count += 1
            
            # Run load test
            while time.time() < end_time:
                # Create batch of concurrent requests
                batch_size = min(self.config.concurrent_users, 5)
                tasks = [load_request() for _ in range(batch_size)]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            # Calculate metrics
            success_rate = successful_requests / request_count if request_count > 0 else 0
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            throughput = request_count / load_duration if load_duration > 0 else 0
            
            return {
                "success": (success_rate >= self.config.min_success_rate and 
                          avg_response_time <= self.config.max_response_time_ms),
                "load_duration_seconds": load_duration,
                "total_requests": request_count,
                "successful_requests": successful_requests,
                "success_rate": success_rate,
                "average_response_time_ms": avg_response_time,
                "throughput_rps": throughput
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        try:
            # Calculate overall metrics
            total_scenarios = len(self.validation_results)
            successful_scenarios = sum(1 for r in self.validation_results if r.success)
            failed_scenarios = total_scenarios - successful_scenarios
            
            overall_success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0
            
            # Calculate average execution time
            total_execution_time = sum(r.execution_time_seconds for r in self.validation_results)
            avg_execution_time = total_execution_time / total_scenarios if total_scenarios > 0 else 0
            
            # Identify critical failures
            critical_failures = [
                r for r in self.validation_results 
                if not r.success and any(
                    s.critical for s in self.test_scenarios 
                    if s.name == r.scenario_name
                )
            ]
            
            # Generate recommendations
            recommendations = []
            
            if critical_failures:
                recommendations.append("Address critical scenario failures before deployment")
            
            if overall_success_rate < 0.9:
                recommendations.append("Improve system reliability - success rate below 90%")
            
            if avg_execution_time > 30:
                recommendations.append("Optimize performance - average scenario execution time is high")
            
            # Performance analysis
            performance_scenarios = [
                r for r in self.validation_results 
                if r.scenario_name in ["concurrent_user_handling", "performance_under_load", "real_time_streaming"]
            ]
            
            performance_passed = all(r.success for r in performance_scenarios)
            
            if not performance_passed:
                recommendations.append("Address performance issues identified in load testing")
            
            # Generate detailed report
            report = {
                "validation_summary": {
                    "overall_success": overall_success_rate >= self.config.min_success_rate,
                    "success_rate": overall_success_rate,
                    "total_scenarios": total_scenarios,
                    "successful_scenarios": successful_scenarios,
                    "failed_scenarios": failed_scenarios,
                    "critical_failures": len(critical_failures),
                    "average_execution_time_seconds": avg_execution_time,
                    "total_execution_time_seconds": total_execution_time
                },
                "scenario_results": [asdict(r) for r in self.validation_results],
                "critical_failures": [asdict(r) for r in critical_failures],
                "performance_analysis": {
                    "performance_scenarios_passed": performance_passed,
                    "performance_scenarios": [asdict(r) for r in performance_scenarios]
                },
                "recommendations": recommendations,
                "configuration": asdict(self.config),
                "timestamp": datetime.now().isoformat()
            }
            
            return report
        
        except Exception as e:
            self.logger.error(f"Failed to generate validation report: {e}")
            return {
                "validation_summary": {
                    "overall_success": False,
                    "error": str(e)
                },
                "timestamp": datetime.now().isoformat()
            }

async def main():
    """Main function for running end-to-end validation"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="FireRedTTS2 End-to-End System Validation")
    parser.add_argument("--base-url", default="http://localhost:7860", help="Base URL for testing")
    parser.add_argument("--websocket-url", default="ws://localhost:7860/ws", help="WebSocket URL for testing")
    parser.add_argument("--concurrent-users", type=int, default=5, help="Number of concurrent users for testing")
    parser.add_argument("--test-duration", type=int, default=10, help="Test duration in minutes")
    parser.add_argument("--scenarios", nargs="+", help="Specific scenarios to run")
    parser.add_argument("--output-file", help="Output file for validation results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = ValidationConfig(
        base_url=args.base_url,
        websocket_url=args.websocket_url,
        concurrent_users=args.concurrent_users,
        test_duration_minutes=args.test_duration,
        test_scenarios=args.scenarios
    )
    
    logger.info("Starting FireRedTTS2 End-to-End System Validation")
    logger.info(f"Configuration: {asdict(config)}")
    
    try:
        # Run validation
        validator = EndToEndValidator(config)
        results = await validator.run_validation()
        
        # Save results if output file specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output_file}")
        
        # Print summary
        summary = results.get("validation_summary", {})
        print("\n" + "="*80)
        print("END-TO-END VALIDATION SUMMARY")
        print("="*80)
        print(f"Overall Success: {summary.get('overall_success', False)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"Total Scenarios: {summary.get('total_scenarios', 0)}")
        print(f"Successful: {summary.get('successful_scenarios', 0)}")
        print(f"Failed: {summary.get('failed_scenarios', 0)}")
        print(f"Critical Failures: {summary.get('critical_failures', 0)}")
        print(f"Execution Time: {summary.get('total_execution_time_seconds', 0):.2f} seconds")
        print("="*80)
        
        # Print recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        # Exit with appropriate code
        exit_code = 0 if summary.get('overall_success', False) else 1
        return exit_code
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"\nVALIDATION FAILED: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)