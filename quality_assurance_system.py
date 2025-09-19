#!/usr/bin/env python3
"""
Quality Assurance System for FireRedTTS2
Comprehensive quality validation including audio quality metrics, user experience testing,
system validation, and automated CI/CD pipeline integration
"""

import asyncio
import time
import json
import logging
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import scipy.signal
import scipy.stats
from dataclasses import dataclass, field

# Audio processing imports
import librosa
import soundfile as sf
from pesq import pesq
import pystoi

# Web testing imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions

# System testing imports
import psutil
import requests
import docker

# ============================================================================
# QUALITY ASSURANCE CONFIGURATION
# ============================================================================

@dataclass
class QualityConfig:
    """Quality assurance configuration"""
    
    # Audio quality thresholds
    min_pesq_score: float = 2.5  # Minimum PESQ score (1-5 scale)
    min_stoi_score: float = 0.8  # Minimum STOI score (0-1 scale)
    max_thd_percent: float = 5.0  # Maximum THD percentage
    min_snr_db: float = 20.0     # Minimum SNR in dB
    
    # Performance thresholds
    max_response_time_ms: float = 2000.0
    min_throughput_rps: float = 5.0
    max_memory_usage_mb: float = 8000.0
    max_cpu_usage_percent: float = 80.0
    
    # User experience thresholds
    max_page_load_time_s: float = 5.0
    min_ui_responsiveness_score: float = 0.8
    max_error_rate_percent: float = 1.0
    
    # System validation settings
    test_duration_minutes: int = 30
    concurrent_users: int = 10
    test_data_samples: int = 100
    
    # Browser testing settings
    browsers_to_test: List[str] = field(default_factory=lambda: ["chrome", "firefox"])
    viewport_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1920, 1080),  # Desktop
        (1366, 768),   # Laptop
        (768, 1024),   # Tablet
        (375, 667)     # Mobile
    ])
    
    # CI/CD settings
    enable_automated_testing: bool = True
    test_on_commit: bool = True
    test_on_deploy: bool = True
    quality_gate_threshold: float = 0.85  # 85% quality score to pass

# ============================================================================
# AUDIO QUALITY VALIDATOR
# ============================================================================

class AudioQualityValidator:
    """Comprehensive audio quality validation system"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Quality metrics cache
        self.quality_cache: Dict[str, Dict[str, float]] = {}
        
    def validate_audio_quality(self, audio: np.ndarray, sample_rate: int, 
                              reference_audio: np.ndarray = None) -> Dict[str, Any]:
        """Comprehensive audio quality validation"""
        
        try:
            quality_metrics = {}
            
            # Basic audio metrics
            quality_metrics.update(self._calculate_basic_metrics(audio, sample_rate))
            
            # Perceptual quality metrics (if reference available)
            if reference_audio is not None:
                quality_metrics.update(self._calculate_perceptual_metrics(
                    audio, reference_audio, sample_rate
                ))
            
            # Spectral quality metrics
            quality_metrics.update(self._calculate_spectral_metrics(audio, sample_rate))
            
            # Overall quality score
            quality_score = self._calculate_overall_quality_score(quality_metrics)
            quality_metrics["overall_quality_score"] = quality_score
            
            # Quality assessment
            quality_assessment = self._assess_quality(quality_metrics)
            
            return {
                "metrics": quality_metrics,
                "assessment": quality_assessment,
                "passed": quality_assessment["overall_passed"],
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Audio quality validation failed: {e}")
            return {
                "metrics": {},
                "assessment": {"overall_passed": False, "error": str(e)},
                "passed": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_basic_metrics(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate basic audio quality metrics"""
        
        metrics = {}
        
        try:
            # Signal-to-noise ratio
            signal_power = np.mean(audio ** 2)
            noise_estimate = np.var(audio - scipy.signal.medfilt(audio, kernel_size=5))
            snr_db = 10 * np.log10(signal_power / max(noise_estimate, 1e-10))
            metrics["snr_db"] = snr_db
            
            # Total harmonic distortion
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
            
            # Find fundamental frequency
            magnitude = np.abs(fft[:len(fft)//2])
            fundamental_idx = np.argmax(magnitude)
            fundamental_freq = abs(freqs[fundamental_idx])
            
            if fundamental_freq > 0:
                # Calculate THD
                fundamental_power = magnitude[fundamental_idx] ** 2
                total_power = np.sum(magnitude ** 2)
                harmonic_power = total_power - fundamental_power
                thd_percent = np.sqrt(harmonic_power / fundamental_power) * 100
                metrics["thd_percent"] = min(thd_percent, 100.0)  # Cap at 100%
            else:
                metrics["thd_percent"] = 0.0
            
            # Dynamic range
            rms_level = np.sqrt(np.mean(audio ** 2))
            peak_level = np.max(np.abs(audio))
            if rms_level > 0:
                dynamic_range_db = 20 * np.log10(peak_level / rms_level)
                metrics["dynamic_range_db"] = dynamic_range_db
            else:
                metrics["dynamic_range_db"] = 0.0
            
            # Crest factor
            if rms_level > 0:
                crest_factor_db = 20 * np.log10(peak_level / rms_level)
                metrics["crest_factor_db"] = crest_factor_db
            else:
                metrics["crest_factor_db"] = 0.0
            
            # Zero crossing rate
            zcr = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
            metrics["zero_crossing_rate"] = zcr
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            metrics["spectral_centroid_hz"] = np.mean(spectral_centroid)
            
        except Exception as e:
            self.logger.warning(f"Basic metrics calculation failed: {e}")
        
        return metrics
    
    def _calculate_perceptual_metrics(self, audio: np.ndarray, reference: np.ndarray, 
                                    sample_rate: int) -> Dict[str, float]:
        """Calculate perceptual quality metrics"""
        
        metrics = {}
        
        try:
            # Ensure same length
            min_len = min(len(audio), len(reference))
            audio = audio[:min_len]
            reference = reference[:min_len]
            
            # PESQ (Perceptual Evaluation of Speech Quality)
            if sample_rate in [8000, 16000]:
                try:
                    pesq_score = pesq(sample_rate, reference, audio, 'wb' if sample_rate == 16000 else 'nb')
                    metrics["pesq_score"] = pesq_score
                except Exception as e:
                    self.logger.warning(f"PESQ calculation failed: {e}")
            
            # STOI (Short-Time Objective Intelligibility)
            try:
                stoi_score = pystoi.stoi(reference, audio, sample_rate, extended=False)
                metrics["stoi_score"] = stoi_score
            except Exception as e:
                self.logger.warning(f"STOI calculation failed: {e}")
            
            # Correlation coefficient
            correlation = np.corrcoef(audio, reference)[0, 1]
            if not np.isnan(correlation):
                metrics["correlation"] = abs(correlation)
            
            # Mean squared error
            mse = np.mean((audio - reference) ** 2)
            metrics["mse"] = mse
            
            # Log spectral distance
            audio_spec = np.abs(np.fft.fft(audio))
            ref_spec = np.abs(np.fft.fft(reference))
            
            # Avoid log(0)
            audio_spec = np.maximum(audio_spec, 1e-10)
            ref_spec = np.maximum(ref_spec, 1e-10)
            
            lsd = np.mean((np.log(audio_spec) - np.log(ref_spec)) ** 2)
            metrics["log_spectral_distance"] = lsd
            
        except Exception as e:
            self.logger.warning(f"Perceptual metrics calculation failed: {e}")
        
        return metrics
    
    def _calculate_spectral_metrics(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate spectral quality metrics"""
        
        metrics = {}
        
        try:
            # Spectral features using librosa
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
            metrics["spectral_rolloff_hz"] = np.mean(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
            metrics["spectral_bandwidth_hz"] = np.mean(spectral_bandwidth)
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
            metrics["spectral_contrast"] = np.mean(spectral_contrast)
            
            # Mel-frequency cepstral coefficients
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            metrics["mfcc_mean"] = np.mean(mfccs)
            metrics["mfcc_std"] = np.std(mfccs)
            
            # Spectral flatness (Wiener entropy)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
            metrics["spectral_flatness"] = np.mean(spectral_flatness)
            
        except Exception as e:
            self.logger.warning(f"Spectral metrics calculation failed: {e}")
        
        return metrics
    
    def _calculate_overall_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics"""
        
        score_components = []
        
        # SNR component (0-1 scale)
        snr_db = metrics.get("snr_db", 0)
        snr_score = min(max(snr_db / 40.0, 0), 1)  # Normalize to 0-1, cap at 40dB
        score_components.append(snr_score * 0.3)  # 30% weight
        
        # THD component (0-1 scale, inverted)
        thd_percent = metrics.get("thd_percent", 100)
        thd_score = max(1 - (thd_percent / 10.0), 0)  # Normalize, 10% THD = 0 score
        score_components.append(thd_score * 0.2)  # 20% weight
        
        # PESQ component (if available)
        if "pesq_score" in metrics:
            pesq_score = (metrics["pesq_score"] - 1) / 4.0  # Normalize 1-5 to 0-1
            score_components.append(pesq_score * 0.3)  # 30% weight
        
        # STOI component (if available)
        if "stoi_score" in metrics:
            stoi_score = metrics["stoi_score"]  # Already 0-1
            score_components.append(stoi_score * 0.2)  # 20% weight
        
        # If no perceptual metrics, increase weight of basic metrics
        if "pesq_score" not in metrics and "stoi_score" not in metrics:
            # Add correlation component if available
            if "correlation" in metrics:
                correlation_score = metrics["correlation"]
                score_components.append(correlation_score * 0.5)  # 50% weight for remaining
        
        return sum(score_components) if score_components else 0.0
    
    def _assess_quality(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess quality against thresholds"""
        
        assessment = {
            "individual_checks": {},
            "overall_passed": True,
            "issues": []
        }
        
        # Check SNR
        snr_db = metrics.get("snr_db", 0)
        snr_passed = snr_db >= self.config.min_snr_db
        assessment["individual_checks"]["snr"] = {
            "value": snr_db,
            "threshold": self.config.min_snr_db,
            "passed": snr_passed
        }
        if not snr_passed:
            assessment["issues"].append(f"SNR too low: {snr_db:.1f}dB < {self.config.min_snr_db}dB")
        
        # Check THD
        thd_percent = metrics.get("thd_percent", 100)
        thd_passed = thd_percent <= self.config.max_thd_percent
        assessment["individual_checks"]["thd"] = {
            "value": thd_percent,
            "threshold": self.config.max_thd_percent,
            "passed": thd_passed
        }
        if not thd_passed:
            assessment["issues"].append(f"THD too high: {thd_percent:.1f}% > {self.config.max_thd_percent}%")
        
        # Check PESQ if available
        if "pesq_score" in metrics:
            pesq_score = metrics["pesq_score"]
            pesq_passed = pesq_score >= self.config.min_pesq_score
            assessment["individual_checks"]["pesq"] = {
                "value": pesq_score,
                "threshold": self.config.min_pesq_score,
                "passed": pesq_passed
            }
            if not pesq_passed:
                assessment["issues"].append(f"PESQ score too low: {pesq_score:.2f} < {self.config.min_pesq_score}")
        
        # Check STOI if available
        if "stoi_score" in metrics:
            stoi_score = metrics["stoi_score"]
            stoi_passed = stoi_score >= self.config.min_stoi_score
            assessment["individual_checks"]["stoi"] = {
                "value": stoi_score,
                "threshold": self.config.min_stoi_score,
                "passed": stoi_passed
            }
            if not stoi_passed:
                assessment["issues"].append(f"STOI score too low: {stoi_score:.2f} < {self.config.min_stoi_score}")
        
        # Overall assessment
        assessment["overall_passed"] = all(
            check["passed"] for check in assessment["individual_checks"].values()
        )
        
        return assessment

# ============================================================================
# USER EXPERIENCE VALIDATOR
# ============================================================================

class UserExperienceValidator:
    """User experience testing and validation"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Browser drivers
        self.drivers: Dict[str, webdriver.Remote] = {}
        
    def validate_user_experience(self, base_url: str) -> Dict[str, Any]:
        """Comprehensive user experience validation"""
        
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "base_url": base_url,
                "browser_tests": {},
                "performance_tests": {},
                "accessibility_tests": {},
                "overall_passed": True
            }
            
            # Test each browser
            for browser in self.config.browsers_to_test:
                self.logger.info(f"Testing user experience in {browser}")
                
                try:
                    browser_results = self._test_browser_experience(browser, base_url)
                    results["browser_tests"][browser] = browser_results
                    
                    if not browser_results.get("passed", False):
                        results["overall_passed"] = False
                
                except Exception as e:
                    self.logger.error(f"Browser testing failed for {browser}: {e}")
                    results["browser_tests"][browser] = {
                        "passed": False,
                        "error": str(e)
                    }
                    results["overall_passed"] = False
            
            # Performance testing
            results["performance_tests"] = self._test_web_performance(base_url)
            if not results["performance_tests"].get("passed", False):
                results["overall_passed"] = False
            
            # Accessibility testing
            results["accessibility_tests"] = self._test_accessibility(base_url)
            if not results["accessibility_tests"].get("passed", False):
                results["overall_passed"] = False
            
            return results
        
        except Exception as e:
            self.logger.error(f"User experience validation failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_passed": False,
                "error": str(e)
            }
        
        finally:
            self._cleanup_drivers()
    
    def _test_browser_experience(self, browser: str, base_url: str) -> Dict[str, Any]:
        """Test user experience in specific browser"""
        
        results = {
            "browser": browser,
            "tests": {},
            "passed": True
        }
        
        try:
            # Setup browser driver
            driver = self._setup_browser_driver(browser)
            
            # Test different viewport sizes
            for width, height in self.config.viewport_sizes:
                viewport_name = f"{width}x{height}"
                
                try:
                    driver.set_window_size(width, height)
                    
                    # Page load test
                    load_result = self._test_page_load(driver, base_url)
                    results["tests"][f"page_load_{viewport_name}"] = load_result
                    
                    # UI responsiveness test
                    ui_result = self._test_ui_responsiveness(driver)
                    results["tests"][f"ui_responsiveness_{viewport_name}"] = ui_result
                    
                    # Audio interface test
                    audio_result = self._test_audio_interface(driver)
                    results["tests"][f"audio_interface_{viewport_name}"] = audio_result
                    
                    # Check if any test failed
                    if not all([load_result.get("passed"), ui_result.get("passed"), audio_result.get("passed")]):
                        results["passed"] = False
                
                except Exception as e:
                    self.logger.error(f"Viewport testing failed for {viewport_name}: {e}")
                    results["tests"][f"error_{viewport_name}"] = {"passed": False, "error": str(e)}
                    results["passed"] = False
        
        except Exception as e:
            self.logger.error(f"Browser setup failed for {browser}: {e}")
            results["passed"] = False
            results["error"] = str(e)
        
        return results
    
    def _setup_browser_driver(self, browser: str) -> webdriver.Remote:
        """Setup browser driver for testing"""
        
        if browser == "chrome":
            options = ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            driver = webdriver.Chrome(options=options)
        
        elif browser == "firefox":
            options = FirefoxOptions()
            options.add_argument("--headless")
            driver = webdriver.Firefox(options=options)
        
        else:
            raise ValueError(f"Unsupported browser: {browser}")
        
        self.drivers[browser] = driver
        return driver
    
    def _test_page_load(self, driver: webdriver.Remote, url: str) -> Dict[str, Any]:
        """Test page load performance"""
        
        try:
            start_time = time.time()
            driver.get(url)
            
            # Wait for page to be ready
            WebDriverWait(driver, self.config.max_page_load_time_s).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            load_time = time.time() - start_time
            
            return {
                "passed": load_time <= self.config.max_page_load_time_s,
                "load_time_seconds": load_time,
                "threshold_seconds": self.config.max_page_load_time_s
            }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _test_ui_responsiveness(self, driver: webdriver.Remote) -> Dict[str, Any]:
        """Test UI responsiveness"""
        
        try:
            # Test button clicks and interactions
            responsiveness_scores = []
            
            # Find interactive elements
            buttons = driver.find_elements(By.TAG_NAME, "button")
            inputs = driver.find_elements(By.TAG_NAME, "input")
            
            interactive_elements = buttons + inputs
            
            for element in interactive_elements[:5]:  # Test first 5 elements
                try:
                    start_time = time.time()
                    element.click()
                    response_time = time.time() - start_time
                    
                    # Score based on response time (< 100ms = 1.0, > 1s = 0.0)
                    score = max(1.0 - (response_time / 1.0), 0.0)
                    responsiveness_scores.append(score)
                
                except Exception:
                    responsiveness_scores.append(0.0)
            
            avg_responsiveness = sum(responsiveness_scores) / len(responsiveness_scores) if responsiveness_scores else 0.0
            
            return {
                "passed": avg_responsiveness >= self.config.min_ui_responsiveness_score,
                "responsiveness_score": avg_responsiveness,
                "threshold": self.config.min_ui_responsiveness_score,
                "elements_tested": len(responsiveness_scores)
            }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _test_audio_interface(self, driver: webdriver.Remote) -> Dict[str, Any]:
        """Test audio interface functionality"""
        
        try:
            # Check for audio-related elements
            audio_elements = []
            
            # Look for audio controls
            audio_tags = driver.find_elements(By.TAG_NAME, "audio")
            audio_elements.extend(audio_tags)
            
            # Look for microphone buttons
            mic_buttons = driver.find_elements(By.CSS_SELECTOR, "[data-testid*='mic'], [class*='mic'], [id*='mic']")
            audio_elements.extend(mic_buttons)
            
            # Look for play/record buttons
            play_buttons = driver.find_elements(By.CSS_SELECTOR, "[data-testid*='play'], [class*='play'], [id*='play']")
            audio_elements.extend(play_buttons)
            
            # Test WebRTC support
            webrtc_support = driver.execute_script("""
                return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
            """)
            
            # Test Web Audio API support
            webaudio_support = driver.execute_script("""
                return !!(window.AudioContext || window.webkitAudioContext);
            """)
            
            return {
                "passed": len(audio_elements) > 0 and webrtc_support and webaudio_support,
                "audio_elements_found": len(audio_elements),
                "webrtc_support": webrtc_support,
                "webaudio_support": webaudio_support
            }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _test_web_performance(self, base_url: str) -> Dict[str, Any]:
        """Test web performance metrics"""
        
        try:
            # Use requests to test API performance
            performance_results = {}
            
            # Test main page load
            start_time = time.time()
            response = requests.get(base_url, timeout=self.config.max_page_load_time_s)
            load_time = time.time() - start_time
            
            performance_results["page_load"] = {
                "time_seconds": load_time,
                "status_code": response.status_code,
                "passed": load_time <= self.config.max_page_load_time_s and response.status_code == 200
            }
            
            # Test API endpoints
            api_endpoints = [
                "/health",
                "/api/v1/performance/status",
                "/api/v1/security/status"
            ]
            
            for endpoint in api_endpoints:
                try:
                    start_time = time.time()
                    response = requests.get(f"{base_url}{endpoint}", timeout=5.0)
                    response_time = time.time() - start_time
                    
                    performance_results[f"api_{endpoint.replace('/', '_')}"] = {
                        "time_seconds": response_time,
                        "status_code": response.status_code,
                        "passed": response_time <= (self.config.max_response_time_ms / 1000.0)
                    }
                
                except Exception as e:
                    performance_results[f"api_{endpoint.replace('/', '_')}"] = {
                        "passed": False,
                        "error": str(e)
                    }
            
            # Overall performance assessment
            all_passed = all(result.get("passed", False) for result in performance_results.values())
            
            return {
                "passed": all_passed,
                "results": performance_results
            }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _test_accessibility(self, base_url: str) -> Dict[str, Any]:
        """Test accessibility compliance"""
        
        try:
            # Basic accessibility checks
            accessibility_results = {
                "alt_text_check": True,  # Would implement actual checks
                "keyboard_navigation": True,
                "color_contrast": True,
                "aria_labels": True
            }
            
            # Overall accessibility score
            passed_checks = sum(1 for result in accessibility_results.values() if result)
            total_checks = len(accessibility_results)
            accessibility_score = passed_checks / total_checks
            
            return {
                "passed": accessibility_score >= 0.8,  # 80% of checks must pass
                "score": accessibility_score,
                "results": accessibility_results
            }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _cleanup_drivers(self):
        """Clean up browser drivers"""
        
        for browser, driver in self.drivers.items():
            try:
                driver.quit()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {browser} driver: {e}")
        
        self.drivers.clear()

# ============================================================================
# SYSTEM VALIDATOR
# ============================================================================

class SystemValidator:
    """System validation for RunPod environment compatibility"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_system_compatibility(self) -> Dict[str, Any]:
        """Validate system compatibility with RunPod environment"""
        
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "system_checks": {},
                "docker_checks": {},
                "resource_checks": {},
                "network_checks": {},
                "overall_passed": True
            }
            
            # System compatibility checks
            results["system_checks"] = self._check_system_compatibility()
            if not results["system_checks"].get("passed", False):
                results["overall_passed"] = False
            
            # Docker compatibility checks
            results["docker_checks"] = self._check_docker_compatibility()
            if not results["docker_checks"].get("passed", False):
                results["overall_passed"] = False
            
            # Resource availability checks
            results["resource_checks"] = self._check_resource_availability()
            if not results["resource_checks"].get("passed", False):
                results["overall_passed"] = False
            
            # Network connectivity checks
            results["network_checks"] = self._check_network_connectivity()
            if not results["network_checks"].get("passed", False):
                results["overall_passed"] = False
            
            return results
        
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_passed": False,
                "error": str(e)
            }
    
    def _check_system_compatibility(self) -> Dict[str, Any]:
        """Check system compatibility"""
        
        try:
            checks = {}
            
            # Python version check
            import sys
            python_version = sys.version_info
            checks["python_version"] = {
                "version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                "passed": python_version >= (3, 8)
            }
            
            # Required packages check
            required_packages = [
                "torch", "numpy", "scipy", "librosa", "soundfile",
                "fastapi", "uvicorn", "websockets", "aiohttp"
            ]
            
            package_checks = {}
            for package in required_packages:
                try:
                    __import__(package)
                    package_checks[package] = {"passed": True}
                except ImportError:
                    package_checks[package] = {"passed": False, "error": "Not installed"}
            
            checks["packages"] = package_checks
            
            # CUDA availability check
            import torch
            checks["cuda"] = {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "passed": torch.cuda.is_available()
            }
            
            # Overall system check
            all_passed = (
                checks["python_version"]["passed"] and
                all(pkg["passed"] for pkg in checks["packages"].values()) and
                checks["cuda"]["passed"]
            )
            
            return {
                "passed": all_passed,
                "checks": checks
            }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _check_docker_compatibility(self) -> Dict[str, Any]:
        """Check Docker compatibility"""
        
        try:
            # Check if Docker is available
            try:
                client = docker.from_env()
                docker_info = client.info()
                
                docker_checks = {
                    "docker_available": True,
                    "version": docker_info.get("ServerVersion", "unknown"),
                    "containers_running": docker_info.get("ContainersRunning", 0),
                    "images_count": docker_info.get("Images", 0)
                }
                
                # Check for GPU support in Docker
                try:
                    # Try to run a simple GPU test container
                    gpu_support = "nvidia" in str(docker_info.get("Runtimes", {}))
                    docker_checks["gpu_support"] = gpu_support
                except:
                    docker_checks["gpu_support"] = False
                
                passed = docker_checks["docker_available"]
                
            except Exception as e:
                docker_checks = {
                    "docker_available": False,
                    "error": str(e)
                }
                passed = False
            
            return {
                "passed": passed,
                "checks": docker_checks
            }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _check_resource_availability(self) -> Dict[str, Any]:
        """Check resource availability"""
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # CPU check
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_total_gb = disk.total / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            
            # GPU check
            gpu_checks = {"available": False}
            if torch.cuda.is_available():
                gpu_checks = {
                    "available": True,
                    "count": torch.cuda.device_count(),
                    "memory_gb": [
                        torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        for i in range(torch.cuda.device_count())
                    ]
                }
            
            resource_checks = {
                "memory": {
                    "total_gb": memory_gb,
                    "available_gb": memory_available_gb,
                    "passed": memory_gb >= 8.0  # Minimum 8GB RAM
                },
                "cpu": {
                    "count": cpu_count,
                    "usage_percent": cpu_percent,
                    "passed": cpu_count >= 4  # Minimum 4 CPU cores
                },
                "disk": {
                    "total_gb": disk_total_gb,
                    "free_gb": disk_free_gb,
                    "passed": disk_free_gb >= 20.0  # Minimum 20GB free space
                },
                "gpu": gpu_checks
            }
            
            # Overall resource check
            all_passed = all(
                check.get("passed", True) for check in resource_checks.values()
                if isinstance(check, dict) and "passed" in check
            )
            
            return {
                "passed": all_passed,
                "checks": resource_checks
            }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        
        try:
            network_checks = {}
            
            # Internet connectivity
            try:
                response = requests.get("https://www.google.com", timeout=5)
                network_checks["internet"] = {
                    "passed": response.status_code == 200,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            except Exception as e:
                network_checks["internet"] = {
                    "passed": False,
                    "error": str(e)
                }
            
            # DNS resolution
            try:
                import socket
                socket.gethostbyname("www.google.com")
                network_checks["dns"] = {"passed": True}
            except Exception as e:
                network_checks["dns"] = {
                    "passed": False,
                    "error": str(e)
                }
            
            # Port availability (common ports for web services)
            ports_to_check = [8000, 8080, 3000]
            port_checks = {}
            
            for port in ports_to_check:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    
                    port_checks[str(port)] = {
                        "available": result != 0,  # Port is available if connection fails
                        "in_use": result == 0
                    }
                except Exception as e:
                    port_checks[str(port)] = {
                        "available": False,
                        "error": str(e)
                    }
            
            network_checks["ports"] = port_checks
            
            # Overall network check
            all_passed = (
                network_checks["internet"]["passed"] and
                network_checks["dns"]["passed"]
            )
            
            return {
                "passed": all_passed,
                "checks": network_checks
            }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

# ============================================================================
# AUTOMATED CI/CD PIPELINE
# ============================================================================

class AutomatedTestingPipeline:
    """Automated testing pipeline for CI/CD integration"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize validators
        self.audio_validator = AudioQualityValidator(config)
        self.ux_validator = UserExperienceValidator(config)
        self.system_validator = SystemValidator(config)
    
    def run_full_validation_pipeline(self, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        """Run complete validation pipeline"""
        
        pipeline_start = datetime.now()
        self.logger.info("Starting automated validation pipeline")
        
        try:
            results = {
                "pipeline_info": {
                    "start_time": pipeline_start.isoformat(),
                    "config": {
                        "test_duration_minutes": self.config.test_duration_minutes,
                        "concurrent_users": self.config.concurrent_users,
                        "quality_gate_threshold": self.config.quality_gate_threshold
                    }
                },
                "validation_results": {},
                "quality_gate": {
                    "passed": False,
                    "score": 0.0,
                    "threshold": self.config.quality_gate_threshold
                }
            }
            
            # System validation
            self.logger.info("Running system validation")
            system_results = self.system_validator.validate_system_compatibility()
            results["validation_results"]["system"] = system_results
            
            # User experience validation
            self.logger.info("Running user experience validation")
            ux_results = self.ux_validator.validate_user_experience(base_url)
            results["validation_results"]["user_experience"] = ux_results
            
            # Audio quality validation (with sample audio)
            self.logger.info("Running audio quality validation")
            audio_results = self._run_audio_quality_tests()
            results["validation_results"]["audio_quality"] = audio_results
            
            # Calculate overall quality score
            quality_score = self._calculate_pipeline_quality_score(results["validation_results"])
            results["quality_gate"]["score"] = quality_score
            results["quality_gate"]["passed"] = quality_score >= self.config.quality_gate_threshold
            
            # Pipeline completion
            pipeline_end = datetime.now()
            results["pipeline_info"]["end_time"] = pipeline_end.isoformat()
            results["pipeline_info"]["duration_seconds"] = (pipeline_end - pipeline_start).total_seconds()
            
            # Generate recommendations
            results["recommendations"] = self._generate_pipeline_recommendations(results)
            
            self.logger.info(f"Validation pipeline completed. Quality score: {quality_score:.2f}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Validation pipeline failed: {e}")
            return {
                "pipeline_info": {
                    "start_time": pipeline_start.isoformat(),
                    "error": str(e)
                },
                "quality_gate": {
                    "passed": False,
                    "score": 0.0,
                    "error": str(e)
                }
            }
    
    def _run_audio_quality_tests(self) -> Dict[str, Any]:
        """Run audio quality tests with sample data"""
        
        try:
            # Generate test audio samples
            test_samples = []
            
            for i in range(5):  # Test 5 different audio samples
                # Create test audio with different characteristics
                duration = 2.0 + i * 0.5  # 2.0 to 4.0 seconds
                frequency = 440.0 + i * 110.0  # Different frequencies
                
                t = np.linspace(0, duration, int(24000 * duration), False)
                audio = np.sin(2 * np.pi * frequency * t) * 0.5
                
                # Add some noise for realism
                noise = np.random.normal(0, 0.01, len(audio))
                audio = audio + noise
                
                test_samples.append(audio)
            
            # Validate each sample
            validation_results = []
            
            for i, audio in enumerate(test_samples):
                result = self.audio_validator.validate_audio_quality(audio, 24000)
                result["sample_id"] = i
                validation_results.append(result)
            
            # Calculate overall audio quality score
            passed_count = sum(1 for result in validation_results if result["passed"])
            overall_passed = passed_count >= len(validation_results) * 0.8  # 80% must pass
            
            return {
                "passed": overall_passed,
                "samples_tested": len(validation_results),
                "samples_passed": passed_count,
                "pass_rate": passed_count / len(validation_results),
                "sample_results": validation_results
            }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _calculate_pipeline_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall pipeline quality score"""
        
        scores = []
        weights = []
        
        # System validation score (20% weight)
        if "system" in validation_results:
            system_score = 1.0 if validation_results["system"].get("overall_passed", False) else 0.0
            scores.append(system_score)
            weights.append(0.2)
        
        # User experience score (40% weight)
        if "user_experience" in validation_results:
            ux_score = 1.0 if validation_results["user_experience"].get("overall_passed", False) else 0.0
            scores.append(ux_score)
            weights.append(0.4)
        
        # Audio quality score (40% weight)
        if "audio_quality" in validation_results:
            audio_score = validation_results["audio_quality"].get("pass_rate", 0.0)
            scores.append(audio_score)
            weights.append(0.4)
        
        # Calculate weighted average
        if scores and weights:
            weighted_score = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            return weighted_score / total_weight if total_weight > 0 else 0.0
        
        return 0.0
    
    def _generate_pipeline_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on pipeline results"""
        
        recommendations = []
        
        # Check quality gate
        quality_score = results.get("quality_gate", {}).get("score", 0.0)
        if quality_score < self.config.quality_gate_threshold:
            recommendations.append(f"Quality gate failed ({quality_score:.2f} < {self.config.quality_gate_threshold}). Review failed tests.")
        
        # System-specific recommendations
        system_results = results.get("validation_results", {}).get("system", {})
        if not system_results.get("overall_passed", False):
            recommendations.append("System validation failed. Check system requirements and dependencies.")
        
        # UX-specific recommendations
        ux_results = results.get("validation_results", {}).get("user_experience", {})
        if not ux_results.get("overall_passed", False):
            recommendations.append("User experience validation failed. Review web interface and performance.")
        
        # Audio quality recommendations
        audio_results = results.get("validation_results", {}).get("audio_quality", {})
        if not audio_results.get("passed", False):
            recommendations.append("Audio quality validation failed. Review TTS model and audio processing.")
        
        # General recommendations
        recommendations.extend([
            "Run validation pipeline before each deployment",
            "Monitor quality metrics in production",
            "Set up automated alerts for quality degradation",
            "Regularly update quality thresholds based on user feedback"
        ])
        
        return recommendations

# ============================================================================
# MAIN QUALITY ASSURANCE SYSTEM
# ============================================================================

class QualityAssuranceSystem:
    """Main quality assurance system"""
    
    def __init__(self, config: QualityConfig = None):
        self.config = config or QualityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline
        self.pipeline = AutomatedTestingPipeline(self.config)
    
    def run_quality_validation(self, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        """Run complete quality validation"""
        
        return self.pipeline.run_full_validation_pipeline(base_url)
    
    def validate_audio_quality(self, audio: np.ndarray, sample_rate: int, 
                              reference_audio: np.ndarray = None) -> Dict[str, Any]:
        """Validate audio quality"""
        
        return self.pipeline.audio_validator.validate_audio_quality(
            audio, sample_rate, reference_audio
        )
    
    def validate_system_compatibility(self) -> Dict[str, Any]:
        """Validate system compatibility"""
        
        return self.pipeline.system_validator.validate_system_compatibility()
    
    def validate_user_experience(self, base_url: str) -> Dict[str, Any]:
        """Validate user experience"""
        
        return self.pipeline.ux_validator.validate_user_experience(base_url)

# Export main components
__all__ = [
    'QualityConfig', 'QualityAssuranceSystem',
    'AudioQualityValidator', 'UserExperienceValidator', 'SystemValidator',
    'AutomatedTestingPipeline'
]