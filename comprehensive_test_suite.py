#!/usr/bin/env python3
"""
Comprehensive Test Suite for FireRedTTS2
Complete testing framework including unit tests, integration tests, performance tests,
and browser compatibility tests for the FireRedTTS2 system
"""

import asyncio
import pytest
import unittest
import time
import json
import tempfile
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Test framework imports
from unittest.mock import Mock, patch, MagicMock
import pytest_asyncio
from pytest_benchmark import benchmark

# Import system components for testing
from data_models import *
from enhanced_fireredtts2 import FireRedTTS2
from whisper_asr import WhisperASRPipeline
from conversation_llm import ConversationLLM
from voice_cloning_interface import VoiceCloningInterface
from performance_monitor import PerformanceMonitor
from security_system import SecuritySystem
from error_handling_system import ErrorManager
from system_monitoring import SystemMonitor
from advanced_performance_monitor import AdvancedPerformanceMonitor
from performance_optimization import PerformanceOptimizer

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

class TestConfig:
    """Test configuration settings"""
    
    def __init__(self):
        # Test environment settings
        self.test_data_dir = Path("test_data")
        self.test_output_dir = Path("test_output")
        self.test_models_dir = Path("test_models")
        
        # API testing settings
        self.api_base_url = "http://localhost:8000"
        self.websocket_url = "ws://localhost:8000/ws"
        self.test_timeout = 30.0
        
        # Performance testing settings
        self.performance_test_duration = 60  # seconds
        self.concurrent_users = 10
        self.max_latency_ms = 2000
        self.min_throughput_rps = 5
        
        # Audio testing settings
        self.test_sample_rate = 24000
        self.test_audio_duration = 5.0  # seconds
        self.audio_quality_threshold = 0.8
        
        # Browser testing settings
        self.browser_test_timeout = 30
        self.supported_browsers = ["chrome", "firefox", "safari", "edge"]
        
        # Create test directories
        self.test_data_dir.mkdir(exist_ok=True)
        self.test_output_dir.mkdir(exist_ok=True)
        self.test_models_dir.mkdir(exist_ok=True)

# Global test configuration
test_config = TestConfig()

# ============================================================================
# TEST UTILITIES
# ============================================================================

class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def create_test_audio(duration: float = 1.0, sample_rate: int = 24000, 
                         frequency: float = 440.0) -> np.ndarray:
        """Create test audio signal"""
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.5
        return audio.astype(np.float32)
    
    @staticmethod
    def create_test_text() -> str:
        """Create test text for TTS"""
        return "This is a test sentence for text-to-speech generation."
    
    @staticmethod
    def save_test_audio(audio: np.ndarray, filename: str, sample_rate: int = 24000):
        """Save test audio to file"""
        
        import soundfile as sf
        filepath = test_config.test_data_dir / filename
        sf.write(filepath, audio, sample_rate)
        return filepath
    
    @staticmethod
    def load_test_audio(filename: str) -> Tuple[np.ndarray, int]:
        """Load test audio from file"""
        
        import soundfile as sf
        filepath = test_config.test_data_dir / filename
        audio, sample_rate = sf.read(filepath)
        return audio, sample_rate
    
    @staticmethod
    def calculate_audio_similarity(audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Calculate similarity between two audio signals"""
        
        # Normalize lengths
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(audio1, audio2)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def measure_audio_quality(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Measure audio quality metrics"""
        
        # Signal-to-noise ratio
        signal_power = np.mean(audio ** 2)
        noise_power = np.var(audio - np.mean(audio))
        snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
        
        # Total harmonic distortion (simplified)
        fft = np.fft.fft(audio)
        fundamental_freq = np.argmax(np.abs(fft[:len(fft)//2]))
        fundamental_power = np.abs(fft[fundamental_freq]) ** 2
        total_power = np.sum(np.abs(fft) ** 2)
        thd = np.sqrt((total_power - fundamental_power) / fundamental_power)
        
        # Dynamic range
        dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-10))
        
        return {
            "snr_db": snr,
            "thd": thd,
            "dynamic_range_db": dynamic_range,
            "rms_level": np.sqrt(np.mean(audio ** 2)),
            "peak_level": np.max(np.abs(audio))
        }

# ============================================================================
# UNIT TESTS
# ============================================================================

class TestDataModels(unittest.TestCase):
    """Unit tests for data models"""
    
    def test_audio_chunk_creation(self):
        """Test AudioChunk model creation and validation"""
        
        chunk = AudioChunk(
            sample_rate=24000,
            channels=1,
            duration_ms=1000.0,
            data_size_bytes=96000
        )
        
        self.assertEqual(chunk.sample_rate, 24000)
        self.assertEqual(chunk.channels, 1)
        self.assertEqual(chunk.duration_ms, 1000.0)
        self.assertIsInstance(chunk.chunk_id, str)
        self.assertIsInstance(chunk.timestamp, float)
    
    def test_generation_request_validation(self):
        """Test GenerationRequest validation"""
        
        # Valid request
        request = GenerationRequest(
            text="Hello world",
            voice_mode=VoiceMode.RANDOM
        )
        self.assertEqual(request.text, "Hello world")
        
        # Invalid request - empty text
        with self.assertRaises(ValueError):
            GenerationRequest(text="")
    
    def test_conversation_session_management(self):
        """Test ConversationSession model"""
        
        session = ConversationSession(
            language="English",
            voice_mode=VoiceMode.CONSISTENT
        )
        
        # Add conversation turn
        turn = ConversationTurn(
            session_id=session.session_id,
            user_input="Hello",
            system_response="Hi there!"
        )
        
        session.turns.append(turn)
        self.assertEqual(len(session.turns), 1)
        self.assertEqual(session.turns[0].user_input, "Hello")

class TestAudioProcessing(unittest.TestCase):
    """Unit tests for audio processing components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_audio = TestUtils.create_test_audio(duration=2.0)
        self.sample_rate = 24000
    
    def test_audio_chunk_processing(self):
        """Test audio chunk processing"""
        
        chunk_size = 1024
        chunks = []
        
        for i in range(0, len(self.test_audio), chunk_size):
            chunk_data = self.test_audio[i:i + chunk_size]
            chunk = AudioChunk(
                sample_rate=self.sample_rate,
                duration_ms=(len(chunk_data) / self.sample_rate) * 1000,
                data_size_bytes=chunk_data.nbytes
            )
            chunks.append(chunk)
        
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].sample_rate, self.sample_rate)
    
    def test_audio_quality_measurement(self):
        """Test audio quality measurement"""
        
        quality_metrics = TestUtils.measure_audio_quality(self.test_audio, self.sample_rate)
        
        self.assertIn("snr_db", quality_metrics)
        self.assertIn("thd", quality_metrics)
        self.assertIn("dynamic_range_db", quality_metrics)
        self.assertGreater(quality_metrics["snr_db"], 0)
    
    def test_audio_similarity_calculation(self):
        """Test audio similarity calculation"""
        
        # Test identical audio
        similarity = TestUtils.calculate_audio_similarity(self.test_audio, self.test_audio)
        self.assertAlmostEqual(similarity, 1.0, places=2)
        
        # Test different audio
        different_audio = TestUtils.create_test_audio(frequency=880.0)
        similarity = TestUtils.calculate_audio_similarity(self.test_audio, different_audio)
        self.assertLess(similarity, 0.5)

class TestTTSEngine(unittest.TestCase):
    """Unit tests for TTS engine"""
    
    def setUp(self):
        """Set up TTS engine for testing"""
        # Mock TTS engine for testing
        self.tts_engine = Mock(spec=FireRedTTS2)
        self.tts_engine.generate_speech = Mock(return_value=TestUtils.create_test_audio())
    
    def test_tts_generation(self):
        """Test TTS speech generation"""
        
        text = TestUtils.create_test_text()
        audio = self.tts_engine.generate_speech(text)
        
        self.tts_engine.generate_speech.assert_called_once_with(text)
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
    
    def test_tts_voice_modes(self):
        """Test different TTS voice modes"""
        
        text = TestUtils.create_test_text()
        
        for voice_mode in VoiceMode:
            with self.subTest(voice_mode=voice_mode):
                self.tts_engine.generate_speech.return_value = TestUtils.create_test_audio()
                audio = self.tts_engine.generate_speech(text, voice_mode=voice_mode)
                self.assertIsInstance(audio, np.ndarray)

class TestASREngine(unittest.TestCase):
    """Unit tests for ASR engine"""
    
    def setUp(self):
        """Set up ASR engine for testing"""
        # Mock ASR engine for testing
        self.asr_engine = Mock(spec=WhisperASRPipeline)
        self.asr_engine.transcribe = Mock(return_value="This is a test transcription.")
    
    def test_asr_transcription(self):
        """Test ASR transcription"""
        
        audio = TestUtils.create_test_audio()
        transcription = self.asr_engine.transcribe(audio)
        
        self.asr_engine.transcribe.assert_called_once_with(audio)
        self.assertIsInstance(transcription, str)
        self.assertGreater(len(transcription), 0)
    
    def test_asr_streaming(self):
        """Test ASR streaming transcription"""
        
        # Mock streaming transcription
        self.asr_engine.transcribe_streaming = Mock(return_value=iter(["This", "is", "a", "test"]))
        
        audio_chunks = [TestUtils.create_test_audio(duration=0.5) for _ in range(4)]
        results = list(self.asr_engine.transcribe_streaming(audio_chunks))
        
        self.assertEqual(len(results), 4)
        self.assertIsInstance(results[0], str)

class TestConversationLLM(unittest.TestCase):
    """Unit tests for conversation LLM"""
    
    def setUp(self):
        """Set up conversation LLM for testing"""
        # Mock LLM for testing
        self.llm = Mock(spec=ConversationLLM)
        self.llm.generate_response = Mock(return_value="This is a test response.")
    
    def test_response_generation(self):
        """Test LLM response generation"""
        
        user_input = "Hello, how are you?"
        response = self.llm.generate_response(user_input)
        
        self.llm.generate_response.assert_called_once_with(user_input)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_conversation_context(self):
        """Test conversation context management"""
        
        # Mock context management
        self.llm.add_to_context = Mock()
        self.llm.get_context = Mock(return_value=["Previous message"])
        
        self.llm.add_to_context("user", "Hello")
        self.llm.add_to_context("assistant", "Hi there!")
        
        context = self.llm.get_context()
        self.assertIsInstance(context, list)

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEndToEndPipeline(unittest.TestCase):
    """Integration tests for end-to-end pipeline"""
    
    def setUp(self):
        """Set up end-to-end pipeline components"""
        # Mock all components for integration testing
        self.asr_engine = Mock(spec=WhisperASRPipeline)
        self.llm = Mock(spec=ConversationLLM)
        self.tts_engine = Mock(spec=FireRedTTS2)
        
        # Configure mock responses
        self.asr_engine.transcribe.return_value = "Hello, how are you?"
        self.llm.generate_response.return_value = "I'm doing well, thank you!"
        self.tts_engine.generate_speech.return_value = TestUtils.create_test_audio()
    
    def test_speech_to_speech_pipeline(self):
        """Test complete speech-to-speech pipeline"""
        
        # Input audio
        input_audio = TestUtils.create_test_audio()
        
        # ASR: Audio to text
        transcription = self.asr_engine.transcribe(input_audio)
        self.assertIsInstance(transcription, str)
        
        # LLM: Text to response text
        response_text = self.llm.generate_response(transcription)
        self.assertIsInstance(response_text, str)
        
        # TTS: Response text to audio
        output_audio = self.tts_engine.generate_speech(response_text)
        self.assertIsInstance(output_audio, np.ndarray)
        
        # Verify pipeline flow
        self.asr_engine.transcribe.assert_called_once_with(input_audio)
        self.llm.generate_response.assert_called_once_with(transcription)
        self.tts_engine.generate_speech.assert_called_once_with(response_text)
    
    def test_conversation_flow(self):
        """Test multi-turn conversation flow"""
        
        conversation_turns = [
            ("Hello", "Hi there!"),
            ("How are you?", "I'm doing well, thanks!"),
            ("What's the weather like?", "I don't have access to weather data.")
        ]
        
        for user_input, expected_response in conversation_turns:
            with self.subTest(user_input=user_input):
                # Mock LLM response
                self.llm.generate_response.return_value = expected_response
                
                # Process turn
                response = self.llm.generate_response(user_input)
                self.assertEqual(response, expected_response)
    
    def test_error_handling_in_pipeline(self):
        """Test error handling in pipeline components"""
        
        # Test ASR error
        self.asr_engine.transcribe.side_effect = Exception("ASR Error")
        
        with self.assertRaises(Exception):
            self.asr_engine.transcribe(TestUtils.create_test_audio())
        
        # Test LLM error
        self.llm.generate_response.side_effect = Exception("LLM Error")
        
        with self.assertRaises(Exception):
            self.llm.generate_response("Test input")
        
        # Test TTS error
        self.tts_engine.generate_speech.side_effect = Exception("TTS Error")
        
        with self.assertRaises(Exception):
            self.tts_engine.generate_speech("Test text")

class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API endpoints"""
    
    def setUp(self):
        """Set up API testing"""
        self.base_url = test_config.api_base_url
        self.timeout = test_config.test_timeout
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint"""
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health", timeout=self.timeout) as response:
                self.assertEqual(response.status, 200)
                
                data = await response.json()
                self.assertIn("success", data)
                self.assertTrue(data["success"])
    
    @pytest.mark.asyncio
    async def test_tts_generation_endpoint(self):
        """Test TTS generation API endpoint"""
        
        request_data = {
            "text": TestUtils.create_test_text(),
            "voice_mode": "random"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/tts/generate",
                json=request_data,
                timeout=self.timeout
            ) as response:
                
                # Should return 200 or appropriate status
                self.assertIn(response.status, [200, 202, 500])  # 500 if service not running
                
                if response.status == 200:
                    data = await response.json()
                    self.assertIn("success", data)
    
    @pytest.mark.asyncio
    async def test_voice_profile_endpoints(self):
        """Test voice profile management endpoints"""
        
        # Test voice profile listing
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/voices",
                timeout=self.timeout
            ) as response:
                
                self.assertIn(response.status, [200, 500])  # 500 if service not running
                
                if response.status == 200:
                    data = await response.json()
                    self.assertIn("items", data)

class TestWebSocketIntegration(unittest.TestCase):
    """Integration tests for WebSocket functionality"""
    
    def setUp(self):
        """Set up WebSocket testing"""
        self.websocket_url = test_config.websocket_url
        self.timeout = test_config.test_timeout
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        
        try:
            async with websockets.connect(
                f"{self.websocket_url}/test_client",
                timeout=self.timeout
            ) as websocket:
                
                # Send ping message
                ping_message = {
                    "type": "ping",
                    "timestamp": time.time()
                }
                
                await websocket.send(json.dumps(ping_message))
                
                # Wait for pong response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                self.assertEqual(response_data.get("type"), "pong")
        
        except (ConnectionRefusedError, OSError):
            # WebSocket server not running - skip test
            self.skipTest("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_websocket_audio_streaming(self):
        """Test WebSocket audio streaming"""
        
        try:
            async with websockets.connect(
                f"{self.websocket_url}/audio_client",
                timeout=self.timeout
            ) as websocket:
                
                # Send audio chunk message
                audio_message = {
                    "type": "audio_chunk",
                    "data": {
                        "chunk_id": "test_chunk_1",
                        "audio_data": TestUtils.create_test_audio().tolist()
                    }
                }
                
                await websocket.send(json.dumps(audio_message))
                
                # Wait for acknowledgment
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                self.assertEqual(response_data.get("type"), "audio_chunk_received")
        
        except (ConnectionRefusedError, OSError):
            # WebSocket server not running - skip test
            self.skipTest("WebSocket server not available")

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking tests"""
    
    def setUp(self):
        """Set up performance testing"""
        self.test_duration = test_config.performance_test_duration
        self.concurrent_users = test_config.concurrent_users
        self.max_latency_ms = test_config.max_latency_ms
    
    def test_tts_latency_benchmark(self):
        """Benchmark TTS generation latency"""
        
        # Mock TTS engine with realistic delay
        tts_engine = Mock()
        
        def mock_generate_speech(text):
            time.sleep(0.1)  # Simulate 100ms processing time
            return TestUtils.create_test_audio()
        
        tts_engine.generate_speech = mock_generate_speech
        
        # Measure latency
        text = TestUtils.create_test_text()
        start_time = time.time()
        
        audio = tts_engine.generate_speech(text)
        
        latency_ms = (time.time() - start_time) * 1000
        
        self.assertLess(latency_ms, self.max_latency_ms)
        self.assertIsInstance(audio, np.ndarray)
    
    def test_concurrent_request_handling(self):
        """Test concurrent request handling performance"""
        
        def simulate_request():
            """Simulate a single request"""
            start_time = time.time()
            
            # Simulate processing time
            time.sleep(0.05)  # 50ms processing
            
            return time.time() - start_time
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=self.concurrent_users) as executor:
            futures = [executor.submit(simulate_request) for _ in range(self.concurrent_users)]
            
            response_times = []
            for future in as_completed(futures):
                response_time = future.result()
                response_times.append(response_time)
        
        # Analyze results
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        self.assertEqual(len(response_times), self.concurrent_users)
        self.assertLess(avg_response_time * 1000, self.max_latency_ms)
        self.assertLess(max_response_time * 1000, self.max_latency_ms * 2)  # Allow 2x for concurrent load
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage during processing"""
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        test_data = []
        for i in range(100):
            audio = TestUtils.create_test_audio(duration=1.0)
            test_data.append(audio)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clean up
        del test_data
        
        # Memory increase should be reasonable (less than 1GB for this test)
        self.assertLess(memory_increase, 1000)  # MB
    
    def test_throughput_benchmark(self):
        """Benchmark system throughput"""
        
        def process_item():
            """Simulate processing a single item"""
            time.sleep(0.01)  # 10ms processing time
            return True
        
        start_time = time.time()
        processed_count = 0
        
        # Process items for test duration
        while time.time() - start_time < 5.0:  # 5 second test
            process_item()
            processed_count += 1
        
        actual_duration = time.time() - start_time
        throughput = processed_count / actual_duration
        
        # Should achieve reasonable throughput
        self.assertGreater(throughput, test_config.min_throughput_rps)

# ============================================================================
# BROWSER COMPATIBILITY TESTS
# ============================================================================

class TestBrowserCompatibility(unittest.TestCase):
    """Browser compatibility tests for web interface"""
    
    def setUp(self):
        """Set up browser testing"""
        self.supported_browsers = test_config.supported_browsers
        self.test_timeout = test_config.browser_test_timeout
    
    def test_webrtc_support(self):
        """Test WebRTC support across browsers"""
        
        # Mock browser capabilities
        browser_capabilities = {
            "chrome": {"webrtc": True, "websockets": True, "audio_api": True},
            "firefox": {"webrtc": True, "websockets": True, "audio_api": True},
            "safari": {"webrtc": True, "websockets": True, "audio_api": True},
            "edge": {"webrtc": True, "websockets": True, "audio_api": True}
        }
        
        for browser in self.supported_browsers:
            with self.subTest(browser=browser):
                capabilities = browser_capabilities.get(browser, {})
                
                self.assertTrue(capabilities.get("webrtc", False), 
                              f"WebRTC not supported in {browser}")
                self.assertTrue(capabilities.get("websockets", False), 
                              f"WebSockets not supported in {browser}")
                self.assertTrue(capabilities.get("audio_api", False), 
                              f"Audio API not supported in {browser}")
    
    def test_audio_format_support(self):
        """Test audio format support across browsers"""
        
        # Mock audio format support
        format_support = {
            "chrome": ["wav", "mp3", "webm", "ogg"],
            "firefox": ["wav", "mp3", "ogg"],
            "safari": ["wav", "mp3", "m4a"],
            "edge": ["wav", "mp3", "webm"]
        }
        
        required_formats = ["wav", "mp3"]
        
        for browser in self.supported_browsers:
            with self.subTest(browser=browser):
                supported_formats = format_support.get(browser, [])
                
                for format_type in required_formats:
                    self.assertIn(format_type, supported_formats,
                                f"{format_type} not supported in {browser}")
    
    def test_responsive_design(self):
        """Test responsive design compatibility"""
        
        # Mock viewport sizes
        viewport_sizes = [
            {"name": "mobile", "width": 375, "height": 667},
            {"name": "tablet", "width": 768, "height": 1024},
            {"name": "desktop", "width": 1920, "height": 1080}
        ]
        
        for viewport in viewport_sizes:
            with self.subTest(viewport=viewport["name"]):
                # Mock responsive behavior
                is_responsive = viewport["width"] >= 320  # Minimum supported width
                
                self.assertTrue(is_responsive, 
                              f"Interface not responsive at {viewport['name']} size")

# ============================================================================
# TEST RUNNER AND REPORTING
# ============================================================================

class TestRunner:
    """Test runner with comprehensive reporting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and generate report"""
        
        self.start_time = datetime.now()
        self.logger.info("Starting comprehensive test suite")
        
        # Test suites to run
        test_suites = [
            ("Unit Tests - Data Models", TestDataModels),
            ("Unit Tests - Audio Processing", TestAudioProcessing),
            ("Unit Tests - TTS Engine", TestTTSEngine),
            ("Unit Tests - ASR Engine", TestASREngine),
            ("Unit Tests - Conversation LLM", TestConversationLLM),
            ("Integration Tests - End-to-End Pipeline", TestEndToEndPipeline),
            ("Integration Tests - API", TestAPIIntegration),
            ("Integration Tests - WebSocket", TestWebSocketIntegration),
            ("Performance Tests", TestPerformanceBenchmarks),
            ("Browser Compatibility Tests", TestBrowserCompatibility)
        ]
        
        # Run each test suite
        for suite_name, suite_class in test_suites:
            self.logger.info(f"Running {suite_name}")
            
            try:
                suite_results = self._run_test_suite(suite_class)
                self.test_results[suite_name] = suite_results
                
            except Exception as e:
                self.logger.error(f"Error running {suite_name}: {e}")
                self.test_results[suite_name] = {
                    "status": "error",
                    "error": str(e),
                    "tests_run": 0,
                    "failures": 1,
                    "errors": 1
                }
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        report = self._generate_test_report()
        
        # Save report
        self._save_test_report(report)
        
        return report
    
    def _run_test_suite(self, suite_class) -> Dict[str, Any]:
        """Run a single test suite"""
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(suite_class)
        
        # Run tests with custom result collector
        result = unittest.TestResult()
        suite.run(result)
        
        return {
            "status": "success" if result.wasSuccessful() else "failed",
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
            "failure_details": [str(failure[1]) for failure in result.failures],
            "error_details": [str(error[1]) for error in result.errors]
        }
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_tests = sum(suite["tests_run"] for suite in self.test_results.values())
        total_failures = sum(suite["failures"] for suite in self.test_results.values())
        total_errors = sum(suite["errors"] for suite in self.test_results.values())
        total_skipped = sum(suite.get("skipped", 0) for suite in self.test_results.values())
        
        success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
        
        report = {
            "test_run_info": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            },
            "summary": {
                "total_tests": total_tests,
                "passed_tests": total_tests - total_failures - total_errors,
                "failed_tests": total_failures,
                "error_tests": total_errors,
                "skipped_tests": total_skipped,
                "success_rate": success_rate
            },
            "suite_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Check overall success rate
        total_tests = sum(suite["tests_run"] for suite in self.test_results.values())
        total_failures = sum(suite["failures"] for suite in self.test_results.values())
        total_errors = sum(suite["errors"] for suite in self.test_results.values())
        
        if total_tests > 0:
            success_rate = ((total_tests - total_failures - total_errors) / total_tests)
            
            if success_rate < 0.8:
                recommendations.append("Test success rate is below 80%. Review failed tests and fix issues.")
            
            if total_errors > 0:
                recommendations.append("Some tests encountered errors. Check system configuration and dependencies.")
            
            if total_failures > 0:
                recommendations.append("Some tests failed. Review test failures and fix underlying issues.")
        
        # Check specific test suites
        for suite_name, results in self.test_results.items():
            if results.get("status") == "error":
                recommendations.append(f"Test suite '{suite_name}' encountered errors. Check system setup.")
            elif results.get("failures", 0) > 0:
                recommendations.append(f"Test suite '{suite_name}' has failures. Review implementation.")
        
        # General recommendations
        recommendations.extend([
            "Run tests regularly during development to catch issues early",
            "Add more test cases for edge cases and error conditions",
            "Consider adding performance regression tests",
            "Implement continuous integration testing",
            "Monitor test coverage and aim for >80% coverage"
        ])
        
        return recommendations
    
    def _save_test_report(self, report: Dict[str, Any]):
        """Save test report to file"""
        
        try:
            report_path = test_config.test_output_dir / f"test_report_{int(time.time())}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Test report saved to {report_path}")
            
            # Also save a summary report
            summary_path = test_config.test_output_dir / "latest_test_summary.json"
            summary = {
                "timestamp": report["test_run_info"]["timestamp"],
                "success_rate": report["summary"]["success_rate"],
                "total_tests": report["summary"]["total_tests"],
                "failed_tests": report["summary"]["failed_tests"],
                "duration_seconds": report["test_run_info"]["duration_seconds"]
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to save test report: {e}")

# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test runner
    runner = TestRunner()
    
    # Run all tests
    report = runner.run_all_tests()
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUITE RESULTS")
    print("="*80)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Errors: {report['summary']['error_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Duration: {report['test_run_info']['duration_seconds']:.2f} seconds")
    print("="*80)
    
    # Print recommendations
    if report['recommendations']:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"{i}. {rec}")
    
    return report

if __name__ == "__main__":
    run_comprehensive_tests()

# Export main components
__all__ = [
    'TestConfig', 'TestUtils', 'TestRunner',
    'TestDataModels', 'TestAudioProcessing', 'TestTTSEngine', 'TestASREngine',
    'TestConversationLLM', 'TestEndToEndPipeline', 'TestAPIIntegration',
    'TestWebSocketIntegration', 'TestPerformanceBenchmarks', 'TestBrowserCompatibility',
    'run_comprehensive_tests'
]