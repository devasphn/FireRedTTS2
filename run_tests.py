#!/usr/bin/env python3
"""
Test Runner for FireRedTTS2
Main test execution script that runs comprehensive tests and quality assurance
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from comprehensive_test_suite import TestRunner, run_comprehensive_tests
from quality_assurance_system import QualityAssuranceSystem, QualityConfig

# ============================================================================
# TEST RUNNER CONFIGURATION
# ============================================================================

class TestRunnerConfig:
    """Configuration for test runner"""
    
    def __init__(self, config_file: str = "test_config.json"):
        self.config_file = config_file
        self.config_data = self._load_config()
        
        # Extract configuration sections
        self.testing_config = self.config_data.get("testing", {})
        self.qa_config = self.config_data.get("quality_assurance", {})
        self.ci_cd_config = self.config_data.get("ci_cd", {})
        self.reporting_config = self.config_data.get("reporting", {})
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                logging.warning(f"Config file {self.config_file} not found, using defaults")
                return {}
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return {}
    
    def get_quality_config(self) -> QualityConfig:
        """Get quality assurance configuration"""
        
        qa_config = QualityConfig()
        
        # Update with loaded configuration
        audio_quality = self.qa_config.get("audio_quality", {})
        if audio_quality:
            qa_config.min_pesq_score = audio_quality.get("min_pesq_score", qa_config.min_pesq_score)
            qa_config.min_stoi_score = audio_quality.get("min_stoi_score", qa_config.min_stoi_score)
            qa_config.max_thd_percent = audio_quality.get("max_thd_percent", qa_config.max_thd_percent)
            qa_config.min_snr_db = audio_quality.get("min_snr_db", qa_config.min_snr_db)
        
        performance = self.qa_config.get("performance_thresholds", {})
        if performance:
            qa_config.max_response_time_ms = performance.get("max_response_time_ms", qa_config.max_response_time_ms)
            qa_config.min_throughput_rps = performance.get("min_throughput_rps", qa_config.min_throughput_rps)
            qa_config.max_memory_usage_mb = performance.get("max_memory_usage_mb", qa_config.max_memory_usage_mb)
            qa_config.max_cpu_usage_percent = performance.get("max_cpu_usage_percent", qa_config.max_cpu_usage_percent)
        
        ux = self.qa_config.get("user_experience", {})
        if ux:
            qa_config.max_page_load_time_s = ux.get("max_page_load_time_seconds", qa_config.max_page_load_time_s)
            qa_config.min_ui_responsiveness_score = ux.get("min_ui_responsiveness_score", qa_config.min_ui_responsiveness_score)
            qa_config.max_error_rate_percent = ux.get("max_error_rate_percent", qa_config.max_error_rate_percent)
        
        quality_gates = self.qa_config.get("quality_gates", {})
        if quality_gates:
            qa_config.quality_gate_threshold = quality_gates.get("overall_threshold", qa_config.quality_gate_threshold)
        
        return qa_config

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

class MainTestRunner:
    """Main test runner that orchestrates all testing"""
    
    def __init__(self, config: TestRunnerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize test components
        self.test_runner = TestRunner()
        self.qa_system = QualityAssuranceSystem(config.get_quality_config())
        
        # Test results
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    async def run_all_tests(self, test_types: List[str] = None, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        """Run all specified tests"""
        
        self.start_time = time.time()
        self.logger.info("Starting comprehensive test execution")
        
        # Default test types
        if test_types is None:
            test_types = ["unit", "integration", "performance", "quality", "system"]
        
        try:
            # Initialize results structure
            self.results = {
                "execution_info": {
                    "start_time": time.time(),
                    "test_types": test_types,
                    "base_url": base_url,
                    "config_file": self.config.config_file
                },
                "test_results": {},
                "overall_summary": {}
            }
            
            # Run unit and integration tests
            if "unit" in test_types or "integration" in test_types:
                self.logger.info("Running unit and integration tests")
                unit_results = self.test_runner.run_all_tests()
                self.results["test_results"]["unit_integration"] = unit_results
            
            # Run performance tests
            if "performance" in test_types:
                self.logger.info("Running performance tests")
                performance_results = await self._run_performance_tests(base_url)
                self.results["test_results"]["performance"] = performance_results
            
            # Run quality assurance tests
            if "quality" in test_types:
                self.logger.info("Running quality assurance tests")
                qa_results = self.qa_system.run_quality_validation(base_url)
                self.results["test_results"]["quality_assurance"] = qa_results
            
            # Run system validation tests
            if "system" in test_types:
                self.logger.info("Running system validation tests")
                system_results = self.qa_system.validate_system_compatibility()
                self.results["test_results"]["system_validation"] = system_results
            
            # Generate overall summary
            self.results["overall_summary"] = self._generate_overall_summary()
            
            # Save results
            await self._save_results()
            
            # Generate reports
            await self._generate_reports()
            
            self.end_time = time.time()
            self.results["execution_info"]["end_time"] = self.end_time
            self.results["execution_info"]["duration_seconds"] = self.end_time - self.start_time
            
            self.logger.info(f"Test execution completed in {self.end_time - self.start_time:.2f} seconds")
            
            return self.results
        
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            self.results["execution_info"]["error"] = str(e)
            return self.results
    
    async def _run_performance_tests(self, base_url: str) -> Dict[str, Any]:
        """Run performance-specific tests"""
        
        try:
            import aiohttp
            import asyncio
            
            performance_results = {
                "api_performance": {},
                "load_testing": {},
                "memory_testing": {},
                "concurrent_testing": {}
            }
            
            # API performance testing
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                start_time = time.time()
                try:
                    async with session.get(f"{base_url}/health", timeout=10) as response:
                        response_time = (time.time() - start_time) * 1000
                        performance_results["api_performance"]["health_endpoint"] = {
                            "response_time_ms": response_time,
                            "status_code": response.status,
                            "passed": response_time < self.config.qa_config.get("performance_thresholds", {}).get("max_response_time_ms", 2000)
                        }
                except Exception as e:
                    performance_results["api_performance"]["health_endpoint"] = {
                        "error": str(e),
                        "passed": False
                    }
                
                # Test other endpoints
                endpoints = ["/api/v1/performance/status", "/api/v1/security/status"]
                for endpoint in endpoints:
                    start_time = time.time()
                    try:
                        async with session.get(f"{base_url}{endpoint}", timeout=10) as response:
                            response_time = (time.time() - start_time) * 1000
                            endpoint_name = endpoint.replace("/", "_").replace("api_v1_", "")
                            performance_results["api_performance"][endpoint_name] = {
                                "response_time_ms": response_time,
                                "status_code": response.status,
                                "passed": response_time < 5000  # 5 second timeout
                            }
                    except Exception as e:
                        endpoint_name = endpoint.replace("/", "_").replace("api_v1_", "")
                        performance_results["api_performance"][endpoint_name] = {
                            "error": str(e),
                            "passed": False
                        }
            
            # Concurrent request testing
            concurrent_users = self.config.testing_config.get("performance_testing", {}).get("concurrent_users", 5)
            performance_results["concurrent_testing"] = await self._test_concurrent_requests(base_url, concurrent_users)
            
            # Memory usage testing
            performance_results["memory_testing"] = self._test_memory_usage()
            
            return performance_results
        
        except Exception as e:
            return {
                "error": str(e),
                "passed": False
            }
    
    async def _test_concurrent_requests(self, base_url: str, concurrent_users: int) -> Dict[str, Any]:
        """Test concurrent request handling"""
        
        try:
            async def make_request(session, url):
                start_time = time.time()
                try:
                    async with session.get(url, timeout=10) as response:
                        return {
                            "response_time_ms": (time.time() - start_time) * 1000,
                            "status_code": response.status,
                            "success": response.status == 200
                        }
                except Exception as e:
                    return {
                        "response_time_ms": (time.time() - start_time) * 1000,
                        "error": str(e),
                        "success": False
                    }
            
            # Run concurrent requests
            async with aiohttp.ClientSession() as session:
                tasks = [
                    make_request(session, f"{base_url}/health")
                    for _ in range(concurrent_users)
                ]
                
                results = await asyncio.gather(*tasks)
            
            # Analyze results
            successful_requests = [r for r in results if r.get("success", False)]
            response_times = [r["response_time_ms"] for r in results]
            
            return {
                "total_requests": len(results),
                "successful_requests": len(successful_requests),
                "success_rate": len(successful_requests) / len(results),
                "average_response_time_ms": sum(response_times) / len(response_times),
                "max_response_time_ms": max(response_times),
                "min_response_time_ms": min(response_times),
                "passed": len(successful_requests) / len(results) >= 0.95  # 95% success rate
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "passed": False
            }
    
    def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage"""
        
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "memory_usage_mb": memory_info.rss / (1024 * 1024),
                "memory_percent": process.memory_percent(),
                "passed": memory_info.rss / (1024 * 1024) < self.config.qa_config.get("performance_thresholds", {}).get("max_memory_usage_mb", 8000)
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "passed": False
            }
    
    def _generate_overall_summary(self) -> Dict[str, Any]:
        """Generate overall test summary"""
        
        summary = {
            "total_test_suites": 0,
            "passed_test_suites": 0,
            "failed_test_suites": 0,
            "overall_success_rate": 0.0,
            "quality_gate_passed": False,
            "recommendations": []
        }
        
        try:
            # Count test suites and their results
            for test_type, results in self.results.get("test_results", {}).items():
                summary["total_test_suites"] += 1
                
                # Determine if test suite passed
                suite_passed = False
                if isinstance(results, dict):
                    if "overall_passed" in results:
                        suite_passed = results["overall_passed"]
                    elif "passed" in results:
                        suite_passed = results["passed"]
                    elif "success_rate" in results:
                        suite_passed = results["success_rate"] >= 0.8
                    elif "summary" in results:
                        suite_passed = results["summary"].get("success_rate", 0) >= 0.8
                
                if suite_passed:
                    summary["passed_test_suites"] += 1
                else:
                    summary["failed_test_suites"] += 1
            
            # Calculate overall success rate
            if summary["total_test_suites"] > 0:
                summary["overall_success_rate"] = summary["passed_test_suites"] / summary["total_test_suites"]
            
            # Check quality gate
            quality_threshold = self.config.qa_config.get("quality_gates", {}).get("overall_threshold", 0.85)
            summary["quality_gate_passed"] = summary["overall_success_rate"] >= quality_threshold
            
            # Generate recommendations
            if not summary["quality_gate_passed"]:
                summary["recommendations"].append("Quality gate failed. Review failed test suites and fix issues.")
            
            if summary["failed_test_suites"] > 0:
                summary["recommendations"].append(f"{summary['failed_test_suites']} test suite(s) failed. Check detailed results.")
            
            if summary["overall_success_rate"] < 0.9:
                summary["recommendations"].append("Overall success rate below 90%. Consider improving test coverage and fixing issues.")
            
            # Add general recommendations
            summary["recommendations"].extend([
                "Run tests regularly during development",
                "Monitor test results and trends over time",
                "Update test cases as system evolves",
                "Investigate and fix any failing tests promptly"
            ])
        
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            summary["error"] = str(e)
        
        return summary
    
    async def _save_results(self):
        """Save test results to files"""
        
        try:
            # Create output directory
            output_dir = Path(self.config.testing_config.get("test_output_directory", "test_output"))
            output_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            timestamp = int(time.time())
            results_file = output_dir / f"test_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            # Save summary
            summary_file = output_dir / "latest_test_summary.json"
            summary = {
                "timestamp": timestamp,
                "overall_summary": self.results.get("overall_summary", {}),
                "execution_info": self.results.get("execution_info", {})
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Test results saved to {results_file}")
        
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    async def _generate_reports(self):
        """Generate test reports"""
        
        try:
            if not self.config.reporting_config.get("generate_html_reports", False):
                return
            
            # Generate HTML report
            html_report = self._generate_html_report()
            
            output_dir = Path(self.config.testing_config.get("test_output_directory", "test_output"))
            report_file = output_dir / "test_report.html"
            
            with open(report_file, 'w') as f:
                f.write(html_report)
            
            self.logger.info(f"HTML report generated: {report_file}")
        
        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML test report"""
        
        summary = self.results.get("overall_summary", {})
        execution_info = self.results.get("execution_info", {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FireRedTTS2 Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .test-suite {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .recommendations {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>FireRedTTS2 Test Report</h1>
                <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Duration: {execution_info.get('duration_seconds', 0):.2f} seconds</p>
            </div>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <p>Total Test Suites: {summary.get('total_test_suites', 0)}</p>
                <p class="passed">Passed: {summary.get('passed_test_suites', 0)}</p>
                <p class="failed">Failed: {summary.get('failed_test_suites', 0)}</p>
                <p>Success Rate: {summary.get('overall_success_rate', 0):.1%}</p>
                <p>Quality Gate: <span class="{'passed' if summary.get('quality_gate_passed') else 'failed'}">
                    {'PASSED' if summary.get('quality_gate_passed') else 'FAILED'}
                </span></p>
            </div>
            
            <div class="recommendations">
                <h3>Recommendations</h3>
                <ul>
        """
        
        for rec in summary.get("recommendations", []):
            html += f"<li>{rec}</li>"
        
        html += """
                </ul>
            </div>
            
            <div class="test-results">
                <h2>Detailed Results</h2>
        """
        
        for test_type, results in self.results.get("test_results", {}).items():
            status = "passed" if results.get("overall_passed", results.get("passed", False)) else "failed"
            html += f"""
                <div class="test-suite">
                    <h3 class="{status}">{test_type.replace('_', ' ').title()}</h3>
                    <p>Status: <span class="{status}">{status.upper()}</span></p>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for test runner"""
    
    parser = argparse.ArgumentParser(description="FireRedTTS2 Test Runner")
    parser.add_argument("--config", default="test_config.json", help="Test configuration file")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for API testing")
    parser.add_argument("--test-types", nargs="+", 
                       choices=["unit", "integration", "performance", "quality", "system"],
                       default=["unit", "integration", "performance", "quality", "system"],
                       help="Types of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--output-dir", help="Output directory for test results")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = TestRunnerConfig(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            config.testing_config["test_output_directory"] = args.output_dir
        
        # Create test runner
        runner = MainTestRunner(config)
        
        # Run tests
        logger.info(f"Starting test execution with types: {args.test_types}")
        results = asyncio.run(runner.run_all_tests(args.test_types, args.base_url))
        
        # Print summary
        summary = results.get("overall_summary", {})
        print("\n" + "="*80)
        print("TEST EXECUTION SUMMARY")
        print("="*80)
        print(f"Total Test Suites: {summary.get('total_test_suites', 0)}")
        print(f"Passed: {summary.get('passed_test_suites', 0)}")
        print(f"Failed: {summary.get('failed_test_suites', 0)}")
        print(f"Success Rate: {summary.get('overall_success_rate', 0):.1%}")
        print(f"Quality Gate: {'PASSED' if summary.get('quality_gate_passed') else 'FAILED'}")
        print(f"Duration: {results.get('execution_info', {}).get('duration_seconds', 0):.2f} seconds")
        print("="*80)
        
        # Exit with appropriate code
        exit_code = 0 if summary.get('quality_gate_passed', False) else 1
        sys.exit(exit_code)
    
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Export main components
__all__ = [
    'TestRunnerConfig', 'MainTestRunner', 'main'
]