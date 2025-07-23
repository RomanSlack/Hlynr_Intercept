#!/usr/bin/env python3
"""
Client stub for testing Unity bridge server.

This script provides a simple way to test the bridge server with dummy
observations and verify that the communication works correctly.
"""

import argparse
import json
import time
import sys
from typing import List, Dict, Any

import requests
import numpy as np


class BridgeClient:
    """Client for communicating with Unity bridge server."""
    
    def __init__(self, host: str = "localhost", port: int = 5000):
        """
        Initialize bridge client.
        
        Args:
            host: Server host address
            port: Server port
        """
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        self.session.timeout = 10.0  # 10 second timeout
        
    def health_check(self) -> Dict[str, Any]:
        """
        Check server health.
        
        Returns:
            Health status response
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': str(e), 'status': 'unreachable'}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.
        
        Returns:
            Server statistics response
        """
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': str(e)}
    
    def get_action(self, 
                   observation: List[float], 
                   deterministic: bool = True) -> Dict[str, Any]:
        """
        Get action for observation.
        
        Args:
            observation: Observation array
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action response
        """
        payload = {
            'observation': observation,
            'deterministic': deterministic
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/act",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': str(e), 'success': False}
    
    def reset_environment(self) -> Dict[str, Any]:
        """
        Reset server environment.
        
        Returns:
            Reset response
        """
        try:
            response = self.session.post(f"{self.base_url}/reset")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': str(e), 'success': False}


def generate_dummy_observation(obs_dim: int = 30) -> List[float]:
    """
    Generate a dummy observation for testing.
    
    Args:
        obs_dim: Observation dimension
        
    Returns:
        Dummy observation array
    """
    # Generate realistic-looking radar observation
    np.random.seed(42)  # For reproducible testing
    
    # Simulate radar observations (normalized between -1 and 1)
    observation = []
    
    # Missile positions and velocities (normalized)
    observation.extend([-0.1, -0.1, 0.2, 0.2, 0.9, 0.9])  # missile data
    
    # Interceptor positions and velocities
    observation.extend([0.4, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.3])  # interceptor data
    
    # Ground radar returns
    observation.extend([-0.1, -0.1, 0.8, 0.5])  # detected pos, confidence, range
    
    # Onboard radar returns  
    observation.extend([0.5, 0.2, 0.6])  # detections, bearing, range
    
    # Environmental data
    observation.extend([0.0, 0.0, 1.0, 0.5, 1.0, 0.1])  # wind, atmosphere, etc.
    
    # Relative positioning (interceptor-missile pairs)
    observation.extend([0.8, 0.3, 0.4])  # distance, bearing, rel_velocity
    
    # Pad or truncate to exact dimension
    if len(observation) < obs_dim:
        observation.extend([0.0] * (obs_dim - len(observation)))
    else:
        observation = observation[:obs_dim]
    
    return observation


def test_basic_functionality(client: BridgeClient):
    """Test basic server functionality."""
    print("=== Testing Basic Functionality ===")
    
    # Health check
    print("1. Health check...")
    health = client.health_check()
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Model loaded: {health.get('model_loaded', False)}")
    
    if not health.get('model_loaded', False):
        print("   ❌ Model not loaded - server may not be ready")
        return False
    
    # Get server stats
    print("2. Server statistics...")
    stats = client.get_stats()
    if 'error' not in stats:
        print(f"   Requests served: {stats.get('requests_served', 0)}")
        print(f"   Average inference time: {stats.get('average_inference_time', 0):.4f}s")
    else:
        print(f"   ❌ Error getting stats: {stats['error']}")
    
    return True


def test_inference(client: BridgeClient, num_tests: int = 5):
    """Test inference with dummy observations."""
    print(f"\n=== Testing Inference ({num_tests} requests) ===")
    
    inference_times = []
    
    for i in range(num_tests):
        print(f"{i+1}. Generating dummy observation...")
        
        # Generate dummy observation
        observation = generate_dummy_observation()
        print(f"   Observation shape: {len(observation)}")
        print(f"   Sample values: {observation[:5]}...")
        
        # Get action
        print("   Requesting action...")
        start_time = time.time()
        result = client.get_action(observation, deterministic=True)
        total_time = time.time() - start_time
        
        if result.get('success', False):
            action = result['action']
            inference_time = result.get('inference_time', 0)
            inference_times.append(inference_time)
            
            print(f"   ✅ Success!")
            print(f"   Action shape: {len(action)}")
            print(f"   Action values: {action}")
            print(f"   Inference time: {inference_time:.4f}s")
            print(f"   Total time: {total_time:.4f}s")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
            return False
        
        print()
    
    # Summary
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        max_inference_time = max(inference_times)
        min_inference_time = min(inference_times)
        
        print(f"Inference Summary:")
        print(f"  Average time: {avg_inference_time:.4f}s")
        print(f"  Min time: {min_inference_time:.4f}s")
        print(f"  Max time: {max_inference_time:.4f}s")
    
    return True


def test_error_handling(client: BridgeClient):
    """Test error handling with invalid requests."""
    print("\n=== Testing Error Handling ===")
    
    # Test invalid observation shape
    print("1. Testing invalid observation shape...")
    invalid_obs = [1.0, 2.0, 3.0]  # Too short
    result = client.get_action(invalid_obs)
    if not result.get('success', True):
        print(f"   ✅ Correctly rejected invalid shape: {result.get('error', 'No error message')}")
    else:
        print("   ❌ Should have rejected invalid observation shape")
    
    # Test invalid endpoint
    print("2. Testing invalid endpoint...")
    try:
        response = client.session.get(f"{client.base_url}/invalid_endpoint")
        if response.status_code == 404:
            print("   ✅ Correctly returned 404 for invalid endpoint")
        else:
            print(f"   ❌ Unexpected status code: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error testing invalid endpoint: {e}")
    
    return True


def benchmark_performance(client: BridgeClient, num_requests: int = 100):
    """Benchmark server performance."""
    print(f"\n=== Performance Benchmark ({num_requests} requests) ===")
    
    observation = generate_dummy_observation()
    inference_times = []
    errors = 0
    
    print("Running benchmark...")
    start_time = time.time()
    
    for i in range(num_requests):
        if i % 20 == 0:
            print(f"  Progress: {i}/{num_requests}")
        
        result = client.get_action(observation, deterministic=True)
        
        if result.get('success', False):
            inference_times.append(result.get('inference_time', 0))
        else:
            errors += 1
    
    total_time = time.time() - start_time
    
    # Results
    if inference_times:
        avg_inference = np.mean(inference_times)
        std_inference = np.std(inference_times)
        throughput = len(inference_times) / total_time
        
        print(f"\nBenchmark Results:")
        print(f"  Total requests: {num_requests}")
        print(f"  Successful: {len(inference_times)}")
        print(f"  Errors: {errors}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} requests/sec")
        print(f"  Average inference time: {avg_inference:.4f} ± {std_inference:.4f}s")
        print(f"  Min inference time: {min(inference_times):.4f}s")
        print(f"  Max inference time: {max(inference_times):.4f}s")
    else:
        print(f"❌ All requests failed!")
    
    return len(inference_times) > 0


def main():
    """Main entry point for client stub."""
    parser = argparse.ArgumentParser(description='Unity Bridge Server Client Stub')
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Server host address'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Server port'
    )
    
    parser.add_argument(
        '--test-type',
        choices=['basic', 'inference', 'errors', 'benchmark', 'all'],
        default='all',
        help='Type of test to run'
    )
    
    parser.add_argument(
        '--num-tests',
        type=int,
        default=5,
        help='Number of inference tests to run'
    )
    
    parser.add_argument(
        '--benchmark-size',
        type=int,
        default=100,
        help='Number of requests for benchmark test'
    )
    
    args = parser.parse_args()
    
    # Create client
    client = BridgeClient(host=args.host, port=args.port)
    
    print(f"Unity Bridge Server Client Stub")
    print(f"Connecting to: {client.base_url}")
    print()
    
    # Run tests based on type
    success = True
    
    if args.test_type in ['basic', 'all']:
        success &= test_basic_functionality(client)
    
    if args.test_type in ['inference', 'all'] and success:
        success &= test_inference(client, args.num_tests)
    
    if args.test_type in ['errors', 'all'] and success:
        success &= test_error_handling(client)
    
    if args.test_type in ['benchmark', 'all'] and success:
        success &= benchmark_performance(client, args.benchmark_size)
    
    # Final result
    print(f"\n{'='*50}")
    if success:
        print("✅ All tests passed!")
        print("Bridge server is working correctly.")
    else:
        print("❌ Some tests failed!")
        print("Check server logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()