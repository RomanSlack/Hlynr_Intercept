"""
Test Runners and Automation Infrastructure

This module provides automated test runners for different testing scenarios:
- Quick validation test runner
- Full test suite runner
- Performance benchmark runner
- Regression test runner
- CI/CD integration runner
- Custom test selection and filtering

Author: Tester Agent
Date: Phase 3 Testing Framework
"""

import pytest
import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
from dataclasses import dataclass, asdict
import psutil


@dataclass
class TestRunConfig:
    """Configuration for test runs"""
    test_categories: List[str]
    include_slow: bool = False
    include_performance: bool = False
    max_duration: Optional[int] = None  # seconds
    output_format: str = 'detailed'  # 'detailed', 'summary', 'json'
    save_results: bool = True
    results_file: Optional[str] = None
    parallel_workers: int = 1
    verbose: bool = False
    fail_fast: bool = False


class TestRunner:
    """Main test runner class"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def run_quick_validation(self) -> Dict[str, Any]:
        """Run quick validation tests for basic functionality"""
        print("Running Quick Validation Tests...")
        print("=" * 50)
        
        config = TestRunConfig(
            test_categories=['basic', 'validation'],
            include_slow=False,
            include_performance=False,
            max_duration=300,  # 5 minutes
            output_format='summary'
        )
        
        # Select specific fast tests
        test_patterns = [
            'test_environment_6dof_validation.py::TestActionSpaceValidation',
            'test_environment_6dof_validation.py::TestObservationSpaceValidation::test_3dof_observation_space',
            'test_physics_6dof_validation.py::TestQuaternionUtils',
            'test_curriculum_validation.py::TestCurriculumPhaseProgression::test_phase_sequence_ordering',
            'test_integration_6dof.py::TestBackwardCompatibility::test_3dof_environment_compatibility'
        ]
        
        return self._run_tests(test_patterns, config)
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation test suite"""
        print("Running Full Validation Test Suite...")
        print("=" * 50)
        
        config = TestRunConfig(
            test_categories=['all'],
            include_slow=True,
            include_performance=False,
            output_format='detailed',
            verbose=True
        )
        
        # Run all validation tests except performance
        test_patterns = [
            'test_physics_6dof_validation.py',
            'test_environment_6dof_validation.py',
            'test_curriculum_validation.py',
            'test_integration_6dof.py',
            'test_adversary_validation.py',
            'test_realworld_validation.py::TestPhysicsAccuracy',
            'test_realworld_validation.py::TestTrajectoryRealism',
            'test_realworld_validation.py::TestCurriculumEffectiveness'
        ]
        
        return self._run_tests(test_patterns, config)
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmark tests"""
        print("Running Performance Benchmarks...")
        print("=" * 50)
        
        config = TestRunConfig(
            test_categories=['performance'],
            include_slow=True,
            include_performance=True,
            output_format='detailed',
            verbose=True
        )
        
        test_patterns = [
            'test_performance_regression.py::TestComputationalPerformance',
            'test_performance_regression.py::TestMemoryUsage',
            'test_performance_regression.py::TestScalabilityValidation'
        ]
        
        return self._run_tests(test_patterns, config)
    
    def run_regression_tests(self) -> Dict[str, Any]:
        """Run regression prevention tests"""
        print("Running Regression Tests...")
        print("=" * 50)
        
        config = TestRunConfig(
            test_categories=['regression'],
            include_slow=False,
            include_performance=True,
            output_format='detailed'
        )
        
        test_patterns = [
            'test_performance_regression.py::TestRegressionPrevention',
            'test_integration_6dof.py::TestBackwardCompatibility',
            'test_environment_6dof_validation.py::TestEnvironmentModeConsistency'
        ]
        
        return self._run_tests(test_patterns, config)
    
    def run_physics_validation(self) -> Dict[str, Any]:
        """Run physics validation tests"""
        print("Running Physics Validation Tests...")
        print("=" * 50)
        
        config = TestRunConfig(
            test_categories=['physics'],
            include_slow=True,
            output_format='detailed'
        )
        
        test_patterns = [
            'test_physics_6dof_validation.py',
            'test_realworld_validation.py::TestPhysicsAccuracy'
        ]
        
        return self._run_tests(test_patterns, config)
    
    def run_custom_tests(self, test_patterns: List[str], config: TestRunConfig) -> Dict[str, Any]:
        """Run custom selection of tests"""
        print(f"Running Custom Tests: {test_patterns}")
        print("=" * 50)
        
        return self._run_tests(test_patterns, config)
    
    def _run_tests(self, test_patterns: List[str], config: TestRunConfig) -> Dict[str, Any]:
        """Internal method to run tests with given patterns and config"""
        start_time = time.time()
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest']
        
        # Add test patterns
        for pattern in test_patterns:
            cmd.append(str(self.base_dir / pattern))
        
        # Add pytest options based on config
        if config.verbose:
            cmd.append('-v')
        
        if config.fail_fast:
            cmd.append('-x')
        
        if not config.include_slow:
            cmd.extend(['-m', 'not slow'])
        elif not config.include_performance:
            cmd.extend(['-m', 'not performance'])
        
        if config.parallel_workers > 1:
            cmd.extend(['-n', str(config.parallel_workers)])
        
        # Output format
        if config.output_format == 'json':
            cmd.append('--json-report')
            cmd.append('--json-report-file=test_results.json')
        elif config.output_format == 'summary':
            cmd.append('--tb=short')
        else:
            cmd.append('--tb=long')
        
        # Add timeout if specified
        if config.max_duration:
            cmd.extend(['--timeout', str(config.max_duration)])
        
        # Run tests
        print(f"Executing: {' '.join(cmd)}")
        print()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=config.max_duration
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse results
            test_results = self._parse_test_results(result, duration, config)
            
            # Save results if requested
            if config.save_results:
                self._save_results(test_results, config)
            
            return test_results
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'duration': config.max_duration,
                'error': f'Tests timed out after {config.max_duration} seconds'
            }
        except Exception as e:
            return {
                'status': 'error',
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def _parse_test_results(self, result: subprocess.CompletedProcess, 
                          duration: float, config: TestRunConfig) -> Dict[str, Any]:
        """Parse test results from subprocess output"""
        test_results = {
            'status': 'passed' if result.returncode == 0 else 'failed',
            'return_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'config': asdict(config),
            'timestamp': time.time(),
            'system_info': self._get_system_info()
        }
        
        # Try to extract test counts from pytest output
        stdout_lines = result.stdout.split('\n')
        
        for line in stdout_lines:
            if 'passed' in line and 'failed' in line:
                # Parse pytest summary line
                test_results['summary_line'] = line.strip()
                break
        
        # Extract any performance metrics from stdout
        performance_metrics = self._extract_performance_metrics(result.stdout)
        if performance_metrics:
            test_results['performance_metrics'] = performance_metrics
        
        return test_results
    
    def _extract_performance_metrics(self, stdout: str) -> Dict[str, Any]:
        """Extract performance metrics from test output"""
        metrics = {}
        
        lines = stdout.split('\n')
        for line in lines:
            # Look for performance output patterns
            if 'steps/second' in line.lower():
                # Extract performance numbers
                words = line.split()
                for i, word in enumerate(words):
                    if 'steps/second' in word.lower() and i > 0:
                        try:
                            value = float(words[i-1].replace(',', ''))
                            metrics['steps_per_second'] = value
                        except ValueError:
                            pass
            
            elif 'memory' in line.lower() and ('mb' in line.lower() or 'growth' in line.lower()):
                # Extract memory metrics
                words = line.split()
                for i, word in enumerate(words):
                    if ('mb' in word.lower() or 'memory' in word.lower()) and i > 0:
                        try:
                            value = float(words[i-1].replace(',', ''))
                            if 'growth' in line.lower():
                                metrics['memory_growth_mb'] = value
                            else:
                                metrics['memory_usage_mb'] = value
                        except ValueError:
                            pass
        
        return metrics
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for test results"""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': os.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available
        }
    
    def _save_results(self, results: Dict[str, Any], config: TestRunConfig):
        """Save test results to file"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        if config.results_file:
            filename = config.results_file
        else:
            test_type = '_'.join(config.test_categories)
            filename = f"test_results_{test_type}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTest results saved to: {filepath}")
    
    def generate_test_report(self, results_files: List[str]) -> str:
        """Generate a comprehensive test report from multiple result files"""
        print("Generating Test Report...")
        
        all_results = []
        for file in results_files:
            filepath = self.results_dir / file
            if filepath.exists():
                with open(filepath, 'r') as f:
                    results = json.load(f)
                    results['source_file'] = file
                    all_results.append(results)
        
        # Generate report
        report_lines = [
            "AegisIntercept 6DOF Test Report",
            "=" * 50,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Test Runs: {len(all_results)}",
            ""
        ]
        
        # Summary statistics
        total_duration = sum(r.get('duration', 0) for r in all_results)
        passed_runs = sum(1 for r in all_results if r.get('status') == 'passed')
        failed_runs = len(all_results) - passed_runs
        
        report_lines.extend([
            "Summary:",
            f"  Passed Runs: {passed_runs}",
            f"  Failed Runs: {failed_runs}",
            f"  Total Duration: {total_duration:.1f} seconds",
            ""
        ])
        
        # Performance metrics
        performance_metrics = []
        for result in all_results:
            if 'performance_metrics' in result:
                performance_metrics.append(result['performance_metrics'])
        
        if performance_metrics:
            report_lines.extend([
                "Performance Metrics:",
                "-------------------"
            ])
            
            # Average performance
            if any('steps_per_second' in pm for pm in performance_metrics):
                steps_per_sec = [pm['steps_per_second'] for pm in performance_metrics if 'steps_per_second' in pm]
                avg_steps = sum(steps_per_sec) / len(steps_per_sec)
                report_lines.append(f"  Average Steps/Second: {avg_steps:.0f}")
            
            if any('memory_growth_mb' in pm for pm in performance_metrics):
                memory_growth = [pm['memory_growth_mb'] for pm in performance_metrics if 'memory_growth_mb' in pm]
                avg_memory = sum(memory_growth) / len(memory_growth)
                report_lines.append(f"  Average Memory Growth: {avg_memory:.1f} MB")
            
            report_lines.append("")
        
        # Individual run details
        report_lines.extend([
            "Individual Runs:",
            "----------------"
        ])
        
        for i, result in enumerate(all_results):
            status = result.get('status', 'unknown')
            duration = result.get('duration', 0)
            source = result.get('source_file', 'unknown')
            
            report_lines.append(f"Run {i+1}: {status.upper()} ({duration:.1f}s) - {source}")
            
            if status == 'failed' and 'stderr' in result:
                # Include error summary
                stderr_lines = result['stderr'].split('\n')
                error_summary = next((line for line in stderr_lines if 'FAILED' in line), 'Unknown error')
                report_lines.append(f"  Error: {error_summary}")
        
        report_content = '\n'.join(report_lines)
        
        # Save report
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"test_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"Test report saved to: {report_file}")
        print("\n" + report_content)
        
        return report_content


class CITestRunner(TestRunner):
    """Specialized test runner for CI/CD environments"""
    
    def run_ci_validation(self) -> Dict[str, Any]:
        """Run tests suitable for CI/CD pipeline"""
        print("Running CI/CD Validation...")
        print("=" * 50)
        
        config = TestRunConfig(
            test_categories=['ci'],
            include_slow=False,
            include_performance=False,
            max_duration=600,  # 10 minutes max
            output_format='summary',
            fail_fast=True,
            verbose=True
        )
        
        # Essential tests for CI
        test_patterns = [
            'test_physics_6dof_validation.py::TestQuaternionUtils',
            'test_physics_6dof_validation.py::TestNumericalIntegration::test_integration_stability',
            'test_environment_6dof_validation.py::TestActionSpaceValidation',
            'test_environment_6dof_validation.py::TestObservationSpaceValidation',
            'test_curriculum_validation.py::TestCurriculumPhaseProgression',
            'test_integration_6dof.py::TestBackwardCompatibility',
            'test_performance_regression.py::TestRegressionPrevention'
        ]
        
        return self._run_tests(test_patterns, config)
    
    def run_nightly_tests(self) -> Dict[str, Any]:
        """Run comprehensive nightly test suite"""
        print("Running Nightly Test Suite...")
        print("=" * 50)
        
        config = TestRunConfig(
            test_categories=['nightly'],
            include_slow=True,
            include_performance=True,
            max_duration=3600,  # 1 hour max
            output_format='detailed',
            verbose=True
        )
        
        # Comprehensive test suite
        test_patterns = [
            'test_physics_6dof_validation.py',
            'test_environment_6dof_validation.py',
            'test_curriculum_validation.py',
            'test_integration_6dof.py',
            'test_adversary_validation.py',
            'test_performance_regression.py',
            'test_realworld_validation.py'
        ]
        
        return self._run_tests(test_patterns, config)


def main():
    """Main CLI interface for test runners"""
    parser = argparse.ArgumentParser(description='AegisIntercept 6DOF Test Runner')
    parser.add_argument('command', choices=[
        'quick', 'full', 'performance', 'regression', 'physics', 'ci', 'nightly', 'custom'
    ], help='Test suite to run')
    
    parser.add_argument('--slow', action='store_true', help='Include slow tests')
    parser.add_argument('--performance', action='store_true', help='Include performance tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fail-fast', '-x', action='store_true', help='Stop on first failure')
    parser.add_argument('--timeout', type=int, help='Timeout in seconds')
    parser.add_argument('--output', choices=['detailed', 'summary', 'json'], 
                       default='detailed', help='Output format')
    parser.add_argument('--results-file', help='Custom results filename')
    parser.add_argument('--workers', '-n', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--patterns', nargs='+', help='Custom test patterns for custom command')
    
    args = parser.parse_args()
    
    # Create test runner
    if args.command in ['ci', 'nightly']:
        runner = CITestRunner()
    else:
        runner = TestRunner()
    
    # Build config
    config = TestRunConfig(
        test_categories=[args.command],
        include_slow=args.slow,
        include_performance=args.performance,
        max_duration=args.timeout,
        output_format=args.output,
        results_file=args.results_file,
        parallel_workers=args.workers,
        verbose=args.verbose,
        fail_fast=args.fail_fast
    )
    
    # Run tests
    start_time = time.time()
    
    try:
        if args.command == 'quick':
            results = runner.run_quick_validation()
        elif args.command == 'full':
            results = runner.run_full_validation()
        elif args.command == 'performance':
            results = runner.run_performance_benchmarks()
        elif args.command == 'regression':
            results = runner.run_regression_tests()
        elif args.command == 'physics':
            results = runner.run_physics_validation()
        elif args.command == 'ci':
            results = runner.run_ci_validation()
        elif args.command == 'nightly':
            results = runner.run_nightly_tests()
        elif args.command == 'custom':
            if not args.patterns:
                print("Error: Custom command requires --patterns")
                sys.exit(1)
            results = runner.run_custom_tests(args.patterns, config)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
        
        # Print results summary
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"Status: {results['status'].upper()}")
        print(f"Duration: {results['duration']:.1f} seconds")
        print(f"Return Code: {results['return_code']}")
        
        if 'summary_line' in results:
            print(f"Summary: {results['summary_line']}")
        
        if 'performance_metrics' in results:
            print("\nPerformance Metrics:")
            for key, value in results['performance_metrics'].items():
                print(f"  {key}: {value}")
        
        # Exit with appropriate code
        sys.exit(results['return_code'])
        
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running tests: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()