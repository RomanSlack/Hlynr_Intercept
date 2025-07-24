#!/usr/bin/env python3
"""
Stress test for Unity bridge server with SLA monitoring.

This script performs comprehensive stress testing of the bridge server
with configurable load patterns and strict SLA requirements:
- ≥20 req/s sustained throughput
- ≤1% error rate
- <60ms p50 latency

Includes CI-friendly 1-minute abbreviated version.
"""

import argparse
import json
import time
import sys
import threading
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

import requests
import numpy as np

try:
    from .client_stub import BridgeClient, generate_dummy_observation
except ImportError:
    from client_stub import BridgeClient, generate_dummy_observation


@dataclass
class StressTestConfig:
    """Configuration for stress test."""
    duration_seconds: int = 300  # 5 minutes default
    target_rps: float = 25.0     # Target requests per second
    max_error_rate: float = 0.01  # 1% max error rate
    max_p50_latency: float = 0.06  # 60ms p50 latency
    warmup_seconds: int = 10      # Warmup period
    cooldown_seconds: int = 5     # Cooldown period
    workers: int = 4              # Number of worker threads
    observation_dim: int = 34     # Observation dimension


@dataclass
class RequestResult:
    """Result of a single request."""
    timestamp: float
    success: bool
    latency: float
    error_msg: str = ""


class StressTestRunner:
    """High-performance stress test runner with SLA monitoring."""
    
    def __init__(self, host: str = "localhost", port: int = 5000, config: StressTestConfig = None):
        """
        Initialize stress test runner.
        
        Args:
            host: Server host
            port: Server port
            config: Test configuration
        """
        self.host = host
        self.port = port
        self.config = config or StressTestConfig()
        self.base_url = f"http://{host}:{port}"
        
        # Test state
        self.results: List[RequestResult] = []
        self.results_lock = threading.Lock()
        self.running = False
        self.start_time = 0
        self.workers: List[threading.Thread] = []
        
        # Pre-generate observations for better performance
        self.observations = [
            generate_dummy_observation(self.config.observation_dim) 
            for _ in range(100)
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def health_check(self) -> bool:
        """
        Verify server is healthy before starting stress test.
        
        Returns:
            True if server is ready
        """
        self.logger.info("Performing pre-test health check...")
        
        try:
            client = BridgeClient(self.host, self.port)
            health = client.health_check()
            
            if health.get('status') != 'healthy' or not health.get('model_loaded'):
                self.logger.error(f"Server not ready: {health}")
                return False
            
            # Test basic inference
            obs = generate_dummy_observation(self.config.observation_dim)
            result = client.get_action(obs, deterministic=True)
            
            if not result.get('success'):
                self.logger.error(f"Inference test failed: {result.get('error')}")
                return False
            
            self.logger.info("✅ Server health check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def worker_thread(self, worker_id: int):
        """
        Worker thread for generating load.
        
        Args:
            worker_id: Unique worker identifier
        """
        session = requests.Session()
        session.timeout = 10.0
        obs_index = 0
        
        self.logger.debug(f"Worker {worker_id} started")
        
        while self.running:
            # Calculate timing for target RPS
            requests_per_worker = self.config.target_rps / self.config.workers
            interval = 1.0 / requests_per_worker if requests_per_worker > 0 else 0.1
            
            start_time = time.time()
            
            # Get observation (cycle through pre-generated ones)
            observation = self.observations[obs_index % len(self.observations)]
            obs_index += 1
            
            # Make request
            result = self._make_request(session, observation)
            
            # Record result
            with self.results_lock:
                self.results.append(result)
            
            # Maintain target rate
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.logger.debug(f"Worker {worker_id} stopped")
    
    def _make_request(self, session: requests.Session, observation: List[float]) -> RequestResult:
        """
        Make a single request and measure performance.
        
        Args:
            session: HTTP session
            observation: Observation data
            
        Returns:
            Request result
        """
        start_time = time.time()
        
        try:
            payload = {
                'observation': observation,
                'deterministic': True
            }
            
            response = session.post(
                f"{self.base_url}/act",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success', False):
                    return RequestResult(
                        timestamp=start_time,
                        success=True,
                        latency=latency
                    )
                else:
                    return RequestResult(
                        timestamp=start_time,
                        success=False,
                        latency=latency,
                        error_msg=data.get('error', 'Unknown error')
                    )
            else:
                return RequestResult(
                    timestamp=start_time,
                    success=False,
                    latency=latency,
                    error_msg=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            latency = time.time() - start_time
            return RequestResult(
                timestamp=start_time,
                success=False,
                latency=latency,
                error_msg=str(e)
            )
    
    def run_stress_test(self) -> Dict[str, Any]:
        """
        Run the complete stress test.
        
        Returns:
            Test results and SLA compliance status
        """
        self.logger.info("="*60)
        self.logger.info("BRIDGE SERVER STRESS TEST")
        self.logger.info("="*60)
        self.logger.info(f"Target: {self.config.target_rps} req/s for {self.config.duration_seconds}s")
        self.logger.info(f"SLA: ≤{self.config.max_error_rate*100}% errors, <{self.config.max_p50_latency*1000}ms p50")
        self.logger.info(f"Workers: {self.config.workers}")
        
        # Health check
        if not self.health_check():
            return {'success': False, 'error': 'Health check failed'}
        
        # Warmup phase
        if self.config.warmup_seconds > 0:
            self.logger.info(f"\n🔥 Warmup phase ({self.config.warmup_seconds}s)...")
            self._run_phase(self.config.warmup_seconds, warmup=True)
            time.sleep(1)
        
        # Main test phase
        self.logger.info(f"\n⚡ Main test phase ({self.config.duration_seconds}s)...")
        self.results.clear()  # Clear warmup results
        
        self.start_time = time.time()
        self.running = True
        
        # Start worker threads
        self.workers = []
        for i in range(self.config.workers):
            worker = threading.Thread(target=self.worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        # Monitor progress
        self._monitor_progress()
        
        # Stop workers
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        # Cooldown
        if self.config.cooldown_seconds > 0:
            self.logger.info(f"\n❄️  Cooldown phase ({self.config.cooldown_seconds}s)...")
            time.sleep(self.config.cooldown_seconds)
        
        # Analyze results
        return self._analyze_results()
    
    def _run_phase(self, duration: int, warmup: bool = False):
        """Run a test phase for specified duration."""
        self.running = True
        self.start_time = time.time()
        
        # Start single worker for warmup
        workers = 1 if warmup else self.config.workers
        temp_workers = []
        
        for i in range(workers):
            worker = threading.Thread(target=self.worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            temp_workers.append(worker)
        
        # Wait for duration
        time.sleep(duration)
        
        # Stop workers
        self.running = False
        for worker in temp_workers:
            worker.join(timeout=2.0)
    
    def _monitor_progress(self):
        """Monitor test progress and provide real-time feedback."""
        update_interval = 10  # Update every 10 seconds
        last_update = time.time()
        
        while self.running and (time.time() - self.start_time) < self.config.duration_seconds:
            time.sleep(1)
            
            if time.time() - last_update >= update_interval:
                elapsed = time.time() - self.start_time
                progress = (elapsed / self.config.duration_seconds) * 100
                
                with self.results_lock:
                    total_requests = len(self.results)
                    if total_requests > 0:
                        current_rps = total_requests / elapsed
                        errors = sum(1 for r in self.results if not r.success)
                        error_rate = errors / total_requests
                        
                        # Calculate recent latency
                        recent_results = [r for r in self.results if r.timestamp > time.time() - 30]
                        if recent_results:
                            recent_latencies = [r.latency for r in recent_results if r.success]
                            if recent_latencies:
                                recent_p50 = statistics.median(recent_latencies)
                                self.logger.info(
                                    f"Progress: {progress:.1f}% | "
                                    f"RPS: {current_rps:.1f} | "
                                    f"Errors: {error_rate*100:.2f}% | "
                                    f"P50: {recent_p50*1000:.1f}ms"
                                )
                
                last_update = time.time()
    
    def _analyze_results(self) -> Dict[str, Any]:
        """
        Analyze test results and check SLA compliance.
        
        Returns:
            Comprehensive results with SLA status
        """
        if not self.results:
            return {'success': False, 'error': 'No results collected'}
        
        # Basic statistics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        
        total_duration = max(r.timestamp for r in self.results) - min(r.timestamp for r in self.results)
        actual_rps = total_requests / total_duration if total_duration > 0 else 0
        
        error_rate = failed_requests / total_requests
        
        # Latency statistics (successful requests only)
        successful_latencies = [r.latency for r in self.results if r.success]
        
        latency_stats = {}
        if successful_latencies:
            latency_stats = {
                'p50': statistics.median(successful_latencies),
                'p95': np.percentile(successful_latencies, 95),
                'p99': np.percentile(successful_latencies, 99),
                'mean': statistics.mean(successful_latencies),
                'std': statistics.stdev(successful_latencies) if len(successful_latencies) > 1 else 0,
                'min': min(successful_latencies),
                'max': max(successful_latencies)
            }
        
        # SLA compliance check
        sla_compliance = {
            'throughput_met': actual_rps >= self.config.target_rps * 0.95,  # 5% tolerance
            'error_rate_met': error_rate <= self.config.max_error_rate,
            'latency_met': latency_stats.get('p50', float('inf')) <= self.config.max_p50_latency
        }
        
        all_sla_met = all(sla_compliance.values())
        
        # Error analysis
        error_breakdown = {}
        if failed_requests > 0:
            for result in self.results:
                if not result.success:
                    error_msg = result.error_msg or 'Unknown'
                    error_breakdown[error_msg] = error_breakdown.get(error_msg, 0) + 1
        
        results = {
            'success': all_sla_met,
            'test_config': {
                'duration': self.config.duration_seconds,
                'target_rps': self.config.target_rps,
                'max_error_rate': self.config.max_error_rate,
                'max_p50_latency': self.config.max_p50_latency,
                'workers': self.config.workers
            },
            'performance': {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'actual_rps': actual_rps,
                'error_rate': error_rate,
                'duration': total_duration
            },
            'latency': latency_stats,
            'sla_compliance': sla_compliance,
            'error_breakdown': error_breakdown
        }
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        self.logger.info("\n" + "="*60)
        self.logger.info("STRESS TEST RESULTS")
        self.logger.info("="*60)
        
        perf = results['performance']
        latency = results['latency']
        sla = results['sla_compliance']
        
        # Performance summary
        self.logger.info(f"\n📊 PERFORMANCE SUMMARY:")
        self.logger.info(f"  Total requests: {perf['total_requests']:,}")
        self.logger.info(f"  Successful: {perf['successful_requests']:,}")
        self.logger.info(f"  Failed: {perf['failed_requests']:,}")
        self.logger.info(f"  Duration: {perf['duration']:.1f}s")
        self.logger.info(f"  Throughput: {perf['actual_rps']:.2f} req/s")
        self.logger.info(f"  Error rate: {perf['error_rate']*100:.3f}%")
        
        # Latency summary
        if latency:
            self.logger.info(f"\n⏱️  LATENCY SUMMARY:")
            self.logger.info(f"  P50: {latency['p50']*1000:.1f}ms")
            self.logger.info(f"  P95: {latency['p95']*1000:.1f}ms")
            self.logger.info(f"  P99: {latency['p99']*1000:.1f}ms")
            self.logger.info(f"  Mean: {latency['mean']*1000:.1f}ms ± {latency['std']*1000:.1f}ms")
            self.logger.info(f"  Range: {latency['min']*1000:.1f}ms - {latency['max']*1000:.1f}ms")
        
        # SLA compliance
        self.logger.info(f"\n🎯 SLA COMPLIANCE:")
        self.logger.info(f"  Throughput (≥{self.config.target_rps} req/s): {'✅' if sla['throughput_met'] else '❌'}")
        self.logger.info(f"  Error rate (≤{self.config.max_error_rate*100}%): {'✅' if sla['error_rate_met'] else '❌'}")
        self.logger.info(f"  P50 latency (<{self.config.max_p50_latency*1000}ms): {'✅' if sla['latency_met'] else '❌'}")
        
        overall_status = "✅ PASSED" if results['success'] else "❌ FAILED"
        self.logger.info(f"\n🏆 OVERALL: {overall_status}")
        
        # Error breakdown
        if results['error_breakdown']:
            self.logger.info(f"\n🚨 ERROR BREAKDOWN:")
            for error, count in results['error_breakdown'].items():
                self.logger.info(f"  {error}: {count}")


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_path}")


def main():
    """Main entry point for stress test."""
    parser = argparse.ArgumentParser(description='Bridge Server Stress Test with SLA Monitoring')
    
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--duration', type=int, default=300, help='Test duration in seconds')
    parser.add_argument('--rps', type=float, default=25.0, help='Target requests per second')
    parser.add_argument('--max-error-rate', type=float, default=0.01, help='Maximum error rate (0.01 = 1%)')
    parser.add_argument('--max-p50-latency', type=float, default=0.06, help='Maximum P50 latency in seconds')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup duration in seconds')
    parser.add_argument('--cooldown', type=int, default=5, help='Cooldown duration in seconds')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--ci-mode', action='store_true', help='CI mode: 1-minute test with lower thresholds')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # CI mode adjustments
    if args.ci_mode:
        config = StressTestConfig(
            duration_seconds=60,      # 1 minute for CI
            target_rps=20.0,          # Slightly lower target for CI
            max_error_rate=0.02,      # 2% tolerance for CI
            max_p50_latency=0.08,     # 80ms tolerance for CI
            warmup_seconds=5,         # Shorter warmup
            cooldown_seconds=2,       # Shorter cooldown
            workers=2                 # Fewer workers for CI
        )
        print("🚀 Running in CI mode (1-minute abbreviated test)")
    else:
        config = StressTestConfig(
            duration_seconds=args.duration,
            target_rps=args.rps,
            max_error_rate=args.max_error_rate,
            max_p50_latency=args.max_p50_latency,
            warmup_seconds=args.warmup,
            cooldown_seconds=args.cooldown,
            workers=args.workers
        )
    
    # Run stress test
    runner = StressTestRunner(args.host, args.port, config)
    results = runner.run_stress_test()
    
    # Save results if requested
    if args.output:
        save_results(results, args.output)
    
    # Exit with appropriate code
    if results['success']:
        print("\n🎉 Stress test PASSED - all SLA requirements met!")
        sys.exit(0)
    else:
        print("\n💥 Stress test FAILED - SLA requirements not met!")
        sys.exit(1)


if __name__ == '__main__':
    main()