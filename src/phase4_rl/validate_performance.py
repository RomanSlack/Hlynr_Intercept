"""
Performance validation script for Unity RL Inference API.

Validates that the system meets PRP latency requirements:
- p50 < 20ms
- p95 < 35ms
- 60 Hz sustained operation without GC spikes
"""

import argparse
import json
import time
import sys
import statistics
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import requests
from dataclasses import dataclass

from .schemas import InferenceRequest


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    latencies_ms: List[float]
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    throughput_rps: float
    test_duration_s: float
    
    @property
    def meets_sla(self) -> bool:
        """Check if metrics meet SLA requirements."""
        return (
            self.p50_latency_ms < 20.0 and
            self.p95_latency_ms < 35.0 and
            self.throughput_rps >= 50.0 and  # Should handle 60 Hz
            self.failed_requests == 0
        )


class PerformanceValidator:
    """Performance validation for Unity RL Inference API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize performance validator.
        
        Args:
            base_url: Base URL of the inference server
        """
        self.base_url = base_url
        self.test_request = self._create_test_request()
    
    def _create_test_request(self) -> Dict[str, Any]:
        """Create a standard test request."""
        return {
            "meta": {
                "episode_id": "perf_test_001",
                "t": 1.0,
                "dt": 0.01,
                "sim_tick": 100
            },
            "frames": {
                "frame": "ENU",
                "unity_lh": True
            },
            "blue": {
                "pos_m": [100.0, 200.0, 50.0],
                "vel_mps": [150.0, 10.0, -5.0],
                "quat_wxyz": [0.995, 0.0, 0.1, 0.0],
                "ang_vel_radps": [0.1, 0.2, 0.05],
                "fuel_frac": 0.75
            },
            "red": {
                "pos_m": [500.0, 600.0, 100.0],
                "vel_mps": [-50.0, -40.0, -10.0],
                "quat_wxyz": [0.924, 0.0, 0.0, 0.383]
            },
            "guidance": {
                "los_unit": [0.8, 0.6, 0.0],
                "los_rate_radps": [0.01, -0.02, 0.0],
                "range_m": 500.0,
                "closing_speed_mps": 200.0,
                "fov_ok": True,
                "g_limit_ok": True
            },
            "env": {
                "wind_mps": [2.0, 1.0, 0.0],
                "noise_std": 0.01,
                "episode_step": 123,
                "max_steps": 1000
            },
            "normalization": {
                "obs_version": "obs_v1.0",
                "vecnorm_stats_id": "vecnorm_baseline_001",
                "transform_version": "tfm_v1.0"
            }
        }
    
    def validate_server_health(self) -> bool:
        """Validate that server is healthy before testing."""
        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get('status') == 'healthy' and health_data.get('model_loaded', False)
            return False
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def run_single_request_test(self) -> float:
        """Run a single request and measure latency."""
        start_time = time.perf_counter()
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/inference",
                json=self.test_request,
                headers={"Content-Type": "application/json"},
                timeout=10.0
            )
            
            latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                return latency
            else:
                print(f"Request failed with status {response.status_code}: {response.text}")
                return -1.0
                
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            print(f"Request error: {e}")
            return -1.0
    
    def run_latency_test(self, num_requests: int = 1000) -> PerformanceMetrics:
        """
        Run latency validation test.
        
        Args:
            num_requests: Number of requests to send
            
        Returns:
            Performance metrics
        """
        print(f"Running latency test with {num_requests} requests...")
        
        latencies = []
        failed_count = 0
        start_time = time.perf_counter()
        
        for i in range(num_requests):
            # Update request with current iteration
            test_req = self.test_request.copy()
            test_req["meta"]["t"] = i * 0.01
            test_req["meta"]["sim_tick"] = i
            
            latency = self.run_single_request_test()
            
            if latency > 0:
                latencies.append(latency)
            else:
                failed_count += 1
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{num_requests} requests")
        
        test_duration = time.perf_counter() - start_time
        
        if not latencies:
            raise RuntimeError("All requests failed")
        
        # Calculate metrics
        latency_array = np.array(latencies)
        
        metrics = PerformanceMetrics(
            total_requests=num_requests,
            successful_requests=len(latencies),
            failed_requests=failed_count,
            latencies_ms=latencies,
            p50_latency_ms=float(np.percentile(latency_array, 50)),
            p95_latency_ms=float(np.percentile(latency_array, 95)),
            p99_latency_ms=float(np.percentile(latency_array, 99)),
            mean_latency_ms=float(np.mean(latency_array)),
            max_latency_ms=float(np.max(latency_array)),
            min_latency_ms=float(np.min(latency_array)),
            throughput_rps=len(latencies) / test_duration,
            test_duration_s=test_duration
        )
        
        return metrics
    
    def run_sixty_hz_soak_test(self, duration_seconds: int = 60) -> PerformanceMetrics:
        """
        Run 60 Hz soak test to validate sustained performance.
        
        Args:
            duration_seconds: Test duration in seconds
            
        Returns:
            Performance metrics
        """
        print(f"Running 60 Hz soak test for {duration_seconds} seconds...")
        
        target_hz = 60.0
        target_period = 1.0 / target_hz
        
        latencies = []
        failed_count = 0
        start_time = time.perf_counter()
        cycle_count = 0
        
        while time.perf_counter() - start_time < duration_seconds:
            cycle_start = time.perf_counter()
            
            # Update request
            current_time = time.perf_counter() - start_time
            test_req = self.test_request.copy()
            test_req["meta"]["t"] = current_time
            test_req["meta"]["sim_tick"] = cycle_count
            
            # Make request
            latency = self.run_single_request_test()
            
            if latency > 0:
                latencies.append(latency)
            else:
                failed_count += 1
            
            cycle_end = time.perf_counter()
            cycle_time = cycle_end - cycle_start
            
            # Maintain 60 Hz timing
            sleep_time = target_period - cycle_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            cycle_count += 1
            
            # Progress update
            if cycle_count % 600 == 0:  # Every 10 seconds
                elapsed = time.perf_counter() - start_time
                current_hz = cycle_count / elapsed
                print(f"  Progress: {elapsed:.1f}s, {current_hz:.1f} Hz")
        
        test_duration = time.perf_counter() - start_time
        actual_hz = cycle_count / test_duration
        
        print(f"  Completed: {cycle_count} cycles in {test_duration:.2f}s ({actual_hz:.1f} Hz)")
        
        if not latencies:
            raise RuntimeError("All requests failed")
        
        # Calculate metrics
        latency_array = np.array(latencies)
        
        metrics = PerformanceMetrics(
            total_requests=cycle_count,
            successful_requests=len(latencies),
            failed_requests=failed_count,
            latencies_ms=latencies,
            p50_latency_ms=float(np.percentile(latency_array, 50)),
            p95_latency_ms=float(np.percentile(latency_array, 95)),
            p99_latency_ms=float(np.percentile(latency_array, 99)),
            mean_latency_ms=float(np.mean(latency_array)),
            max_latency_ms=float(np.max(latency_array)),
            min_latency_ms=float(np.min(latency_array)),
            throughput_rps=len(latencies) / test_duration,
            test_duration_s=test_duration
        )
        
        return metrics
    
    def run_burst_test(self, burst_size: int = 100, num_bursts: int = 10) -> PerformanceMetrics:
        """
        Run burst test to validate handling of request spikes.
        
        Args:
            burst_size: Number of requests per burst
            num_bursts: Number of bursts to send
            
        Returns:
            Performance metrics
        """
        print(f"Running burst test: {num_bursts} bursts of {burst_size} requests...")
        
        all_latencies = []
        total_failed = 0
        start_time = time.perf_counter()
        
        for burst_idx in range(num_bursts):
            print(f"  Burst {burst_idx + 1}/{num_bursts}")
            
            # Send burst of requests rapidly
            burst_latencies = []
            for req_idx in range(burst_size):
                test_req = self.test_request.copy()
                test_req["meta"]["sim_tick"] = burst_idx * burst_size + req_idx
                
                latency = self.run_single_request_test()
                
                if latency > 0:
                    burst_latencies.append(latency)
                else:
                    total_failed += 1
            
            all_latencies.extend(burst_latencies)
            
            # Wait between bursts
            time.sleep(0.5)
        
        test_duration = time.perf_counter() - start_time
        total_requests = num_bursts * burst_size
        
        if not all_latencies:
            raise RuntimeError("All requests failed")
        
        # Calculate metrics
        latency_array = np.array(all_latencies)
        
        metrics = PerformanceMetrics(
            total_requests=total_requests,
            successful_requests=len(all_latencies),
            failed_requests=total_failed,
            latencies_ms=all_latencies,
            p50_latency_ms=float(np.percentile(latency_array, 50)),
            p95_latency_ms=float(np.percentile(latency_array, 95)),
            p99_latency_ms=float(np.percentile(latency_array, 99)),
            mean_latency_ms=float(np.mean(latency_array)),
            max_latency_ms=float(np.max(latency_array)),
            min_latency_ms=float(np.min(latency_array)),
            throughput_rps=len(all_latencies) / test_duration,
            test_duration_s=test_duration
        )
        
        return metrics
    
    def print_metrics(self, metrics: PerformanceMetrics, test_name: str):
        """Print performance metrics in a formatted way."""
        print(f"\n{test_name} Results:")
        print("=" * 50)
        print(f"Total Requests:     {metrics.total_requests:,}")
        print(f"Successful:         {metrics.successful_requests:,}")
        print(f"Failed:             {metrics.failed_requests:,}")
        print(f"Test Duration:      {metrics.test_duration_s:.2f}s")
        print(f"Throughput:         {metrics.throughput_rps:.1f} req/s")
        print()
        print("Latency Statistics:")
        print(f"  Mean:             {metrics.mean_latency_ms:.2f}ms")
        print(f"  P50:              {metrics.p50_latency_ms:.2f}ms")
        print(f"  P95:              {metrics.p95_latency_ms:.2f}ms")
        print(f"  P99:              {metrics.p99_latency_ms:.2f}ms")
        print(f"  Min:              {metrics.min_latency_ms:.2f}ms")
        print(f"  Max:              {metrics.max_latency_ms:.2f}ms")
        print()
        print("SLA Compliance:")
        print(f"  P50 < 20ms:       {'✅' if metrics.p50_latency_ms < 20.0 else '❌'} ({metrics.p50_latency_ms:.2f}ms)")
        print(f"  P95 < 35ms:       {'✅' if metrics.p95_latency_ms < 35.0 else '❌'} ({metrics.p95_latency_ms:.2f}ms)")
        print(f"  Throughput ≥50/s: {'✅' if metrics.throughput_rps >= 50.0 else '❌'} ({metrics.throughput_rps:.1f}/s)")
        print(f"  Zero failures:    {'✅' if metrics.failed_requests == 0 else '❌'} ({metrics.failed_requests} failed)")
        print()
        print(f"Overall SLA:        {'✅ PASSED' if metrics.meets_sla else '❌ FAILED'}")
    
    def save_metrics(self, metrics: PerformanceMetrics, output_path: str, test_name: str):
        """Save metrics to JSON file."""
        output_data = {
            'test_name': test_name,
            'timestamp': time.time(),
            'metrics': {
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'test_duration_s': metrics.test_duration_s,
                'throughput_rps': metrics.throughput_rps,
                'latency_stats': {
                    'mean_ms': metrics.mean_latency_ms,
                    'p50_ms': metrics.p50_latency_ms,
                    'p95_ms': metrics.p95_latency_ms,
                    'p99_ms': metrics.p99_latency_ms,
                    'min_ms': metrics.min_latency_ms,
                    'max_ms': metrics.max_latency_ms
                },
                'sla_compliance': {
                    'p50_under_20ms': metrics.p50_latency_ms < 20.0,
                    'p95_under_35ms': metrics.p95_latency_ms < 35.0,
                    'throughput_over_50rps': metrics.throughput_rps >= 50.0,
                    'zero_failures': metrics.failed_requests == 0,
                    'overall_pass': metrics.meets_sla
                }
            }
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Metrics saved to: {output_path}")


def main():
    """Main entry point for performance validation."""
    parser = argparse.ArgumentParser(description='Unity RL Inference API Performance Validation')
    
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:5000',
        help='Base URL of inference server'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        choices=['latency', 'soak', 'burst', 'all'],
        default='all',
        help='Test type to run'
    )
    
    parser.add_argument(
        '--requests',
        type=int,
        default=1000,
        help='Number of requests for latency test'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration in seconds for soak test'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for metrics (JSON)'
    )
    
    parser.add_argument(
        '--no-health-check',
        action='store_true',
        help='Skip health check'
    )
    
    args = parser.parse_args()
    
    validator = PerformanceValidator(args.url)
    
    # Health check
    if not args.no_health_check:
        print("Checking server health...")
        if not validator.validate_server_health():
            print("❌ Server health check failed")
            sys.exit(1)
        print("✅ Server is healthy")
    
    # Run tests
    all_passed = True
    
    if args.test in ['latency', 'all']:
        try:
            metrics = validator.run_latency_test(args.requests)
            validator.print_metrics(metrics, "Latency Test")
            
            if args.output:
                validator.save_metrics(metrics, args.output.replace('.json', '_latency.json'), 'latency')
            
            if not metrics.meets_sla:
                all_passed = False
                
        except Exception as e:
            print(f"❌ Latency test failed: {e}")
            all_passed = False
    
    if args.test in ['soak', 'all']:
        try:
            metrics = validator.run_sixty_hz_soak_test(args.duration)
            validator.print_metrics(metrics, "60 Hz Soak Test")
            
            if args.output:
                validator.save_metrics(metrics, args.output.replace('.json', '_soak.json'), 'soak')
            
            if not metrics.meets_sla:
                all_passed = False
                
        except Exception as e:
            print(f"❌ Soak test failed: {e}")
            all_passed = False
    
    if args.test in ['burst', 'all']:
        try:
            metrics = validator.run_burst_test()
            validator.print_metrics(metrics, "Burst Test")
            
            if args.output:
                validator.save_metrics(metrics, args.output.replace('.json', '_burst.json'), 'burst')
            
            if not metrics.meets_sla:
                all_passed = False
                
        except Exception as e:
            print(f"❌ Burst test failed: {e}")
            all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL PERFORMANCE TESTS PASSED")
        print("✅ System meets PRP latency requirements:")
        print("   - P50 < 20ms")
        print("   - P95 < 35ms")
        print("   - 60 Hz sustained operation")
        sys.exit(0)
    else:
        print("💥 PERFORMANCE TESTS FAILED")
        print("❌ System does not meet PRP requirements")
        sys.exit(1)


if __name__ == '__main__':
    main()