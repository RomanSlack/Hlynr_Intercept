#!/usr/bin/env python3
"""
Test client for Hlynr Unity-RL Inference API
Tests all endpoints and visualizes the response data
"""

import json
import time
import asyncio
import aiohttp
import numpy as np
from typing import Dict, Any, List
import sys
from datetime import datetime
from pathlib import Path

# Server configuration
SERVER_URL = "http://localhost:5000"
INFERENCE_ENDPOINT = f"{SERVER_URL}/v1/inference"
HEALTH_ENDPOINT = f"{SERVER_URL}/healthz"
METRICS_ENDPOINT = f"{SERVER_URL}/metrics"

# Global response collection
collected_responses: List[Dict[str, Any]] = []

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text: str, color: str = Colors.ENDC):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.ENDC}")

def print_section(title: str):
    """Print a section header."""
    print_colored(f"\n{'='*60}", Colors.CYAN)
    print_colored(f"  {title}", Colors.BOLD + Colors.CYAN)
    print_colored(f"{'='*60}", Colors.CYAN)

def create_test_request(episode_step: int = 0, t: float = 0.0, vecnorm_stats_id: str = "vecnorm_checkpoints_obs_v1.0_43d32970") -> Dict[str, Any]:
    """Create a test inference request with realistic data."""
    # Simulate interceptor approaching threat
    intercept_distance = 5000.0 - (t * 100.0)  # Closing at 100 m/s
    
    return {
        "meta": {
            "episode_id": f"test_ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "t": t,
            "dt": 0.01,
            "sim_tick": episode_step
        },
        "frames": {
            "frame": "ENU",
            "unity_lh": True  # Unity uses left-handed coords
        },
        "blue": {  # Interceptor
            "pos_m": [0.0, 0.0, 1000.0],
            "vel_mps": [100.0, 0.0, 0.0],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "ang_vel_radps": [0.0, 0.0, 0.0],
            "fuel_frac": 0.9 - (t * 0.01)  # Fuel depleting
        },
        "red": {  # Threat
            "pos_m": [intercept_distance, 0.0, 1000.0],
            "vel_mps": [-50.0, 0.0, 0.0],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0]
        },
        "guidance": {
            "los_unit": [1.0, 0.0, 0.0],
            "los_rate_radps": [0.0, 0.01, 0.0],
            "range_m": intercept_distance,
            "closing_speed_mps": 150.0,
            "fov_ok": True,
            "g_limit_ok": True
        },
        "env": {
            "wind_mps": [2.0, 1.0, 0.0],
            "noise_std": 0.01,
            "episode_step": episode_step,
            "max_steps": 1000
        },
        "normalization": {
            "obs_version": "obs_v1.0",
            "vecnorm_stats_id": vecnorm_stats_id  # Use the provided ID
        }
    }

async def wait_for_server(session: aiohttp.ClientSession, max_retries: int = 30):
    """Wait for server to be ready."""
    print_colored("Waiting for server to start...", Colors.YELLOW)
    
    for i in range(max_retries):
        try:
            async with session.get(HEALTH_ENDPOINT, timeout=2) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('policy_loaded'):
                        print_colored("✓ Server is ready!", Colors.GREEN)
                        return True
                    else:
                        print(f"  Server running but model not loaded yet... ({i+1}/{max_retries})")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            print(f"  Waiting for server... ({i+1}/{max_retries})")
        
        await asyncio.sleep(2)
    
    return False

async def test_health_endpoint(session: aiohttp.ClientSession):
    """Test the health check endpoint."""
    print_section("Testing Health Endpoint")
    
    try:
        async with session.get(HEALTH_ENDPOINT) as response:
            data = await response.json()
            
            print_colored(f"Status Code: {response.status}", Colors.GREEN if response.status == 200 else Colors.RED)
            print("\nHealth Response:")
            
            # Key health indicators
            print(f"  • Server OK: {Colors.GREEN if data.get('ok') else Colors.RED}{data.get('ok')}{Colors.ENDC}")
            print(f"  • Policy Loaded: {Colors.GREEN if data.get('policy_loaded') else Colors.RED}{data.get('policy_loaded')}{Colors.ENDC}")
            print(f"  • Policy ID: {data.get('policy_id', 'N/A')}")
            print(f"  • VecNorm Stats ID: {data.get('vecnorm_stats_id', 'N/A')}")
            print(f"  • Obs Version: {data.get('obs_version', 'N/A')}")
            print(f"  • Transform Version: {data.get('transform_version', 'N/A')}")
            print(f"  • Seed: {data.get('seed', 'N/A')}")
            
            return data.get('ok', False)
            
    except Exception as e:
        print_colored(f"❌ Health check failed: {e}", Colors.RED)
        return False

async def get_server_config(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Get server configuration from health endpoint."""
    async with session.get(HEALTH_ENDPOINT) as response:
        return await response.json()

def save_response_to_collection(request_data: Dict[str, Any], response_data: Dict[str, Any], latency_ms: float, endpoint: str = "inference"):
    """Save a response to the global collection for later JSON export."""
    global collected_responses
    
    response_entry = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "latency_ms": latency_ms,
        "request": request_data,
        "response": response_data
    }
    collected_responses.append(response_entry)

def save_responses_to_json(filename: str = None):
    """Save all collected responses to a JSON file."""
    global collected_responses
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inference_test_responses_{timestamp}.json"
    
    # Create test results summary
    test_summary = {
        "test_metadata": {
            "timestamp": datetime.now().isoformat(),
            "server_url": SERVER_URL,
            "total_requests": len(collected_responses),
            "inference_requests": len([r for r in collected_responses if r["endpoint"] == "inference" or r["endpoint"].startswith("simulation_")]),
            "simulation_steps": len([r for r in collected_responses if r["endpoint"].startswith("simulation_")])
        },
        "responses": collected_responses
    }
    
    filepath = Path(filename)
    
    with open(filepath, 'w') as f:
        json.dump(test_summary, f, indent=2, default=str)
    
    print_colored(f"\n💾 Saved {len(collected_responses)} responses to: {filepath.absolute()}", Colors.GREEN)
    
    # Print summary statistics
    if collected_responses:
        latencies = [r["latency_ms"] for r in collected_responses if "latency_ms" in r]
        if latencies:
            print_colored(f"📊 Response Summary:", Colors.CYAN)
            print(f"  • Total Responses: {len(collected_responses)}")
            print(f"  • Avg Latency: {np.mean(latencies):.2f} ms")
            print(f"  • Max Latency: {max(latencies):.2f} ms")
            print(f"  • Min Latency: {min(latencies):.2f} ms")
            
            # Check if simulation states are present
            sim_states = [r for r in collected_responses if r.get("response", {}).get("simulation_state")]
            print(f"  • Responses with Simulation State: {len(sim_states)}/{len(collected_responses)}")
            
            if sim_states:
                print_colored("✅ Simulation state data is being returned correctly!", Colors.GREEN)
            else:
                print_colored("⚠️ No simulation state data found in responses", Colors.YELLOW)
    
    return filepath

async def test_inference_endpoint(session: aiohttp.ClientSession, request_data: Dict[str, Any]):
    """Test the inference endpoint."""
    print_section("Testing Inference Endpoint")
    
    try:
        start_time = time.perf_counter()
        
        async with session.post(
            INFERENCE_ENDPOINT,
            json=request_data,
            headers={'Content-Type': 'application/json'}
        ) as response:
            latency_ms = (time.perf_counter() - start_time) * 1000
            data = await response.json()
            
            # Save response to collection
            save_response_to_collection(request_data, data, latency_ms, "inference")
            
            print_colored(f"Status Code: {response.status}", Colors.GREEN if response.status == 200 else Colors.RED)
            print(f"Latency: {latency_ms:.2f} ms")
            
            if response.status == 200 and data.get('success', True):
                print_colored("\n✓ Inference Response Received!", Colors.GREEN)
                
                # Display action commands
                if 'action' in data:
                    action = data['action']
                    print("\n📍 Control Commands:")
                    print(f"  • Pitch Rate: {action['rate_cmd_radps']['pitch']:.3f} rad/s")
                    print(f"  • Yaw Rate: {action['rate_cmd_radps']['yaw']:.3f} rad/s")
                    print(f"  • Roll Rate: {action['rate_cmd_radps']['roll']:.3f} rad/s")
                    print(f"  • Thrust: {action['thrust_cmd']:.3f} [0-1]")
                
                # Display diagnostics
                if 'diagnostics' in data:
                    diag = data['diagnostics']
                    print("\n📊 Diagnostics:")
                    print(f"  • Policy Latency: {diag['policy_latency_ms']:.2f} ms")
                    print(f"  • Obs Clip Low: {diag['obs_clip_fractions']['low']:.3f}")
                    print(f"  • Obs Clip High: {diag['obs_clip_fractions']['high']:.3f}")
                
                # Display safety info
                if 'safety' in data:
                    safety = data['safety']
                    print("\n🛡️ Safety Status:")
                    if safety['clamped']:
                        print_colored(f"  • Commands Clamped: {safety['clamp_reason']}", Colors.YELLOW)
                    else:
                        print_colored("  • No Safety Clamps Applied", Colors.GREEN)
                
                # Display simulation state (NEW!)
                if 'simulation_state' in data and data['simulation_state']:
                    sim_state = data['simulation_state']
                    print_colored("\n🎮 Simulation State (Python → Unity):", Colors.CYAN)
                    
                    if 'blue' in sim_state:
                        blue = sim_state['blue']
                        print("\n  Interceptor (Blue):")
                        print(f"    • Position: [{blue['pos_m'][0]:.1f}, {blue['pos_m'][1]:.1f}, {blue['pos_m'][2]:.1f}] m")
                        print(f"    • Velocity: [{blue['vel_mps'][0]:.1f}, {blue['vel_mps'][1]:.1f}, {blue['vel_mps'][2]:.1f}] m/s")
                        print(f"    • Fuel: {blue['fuel_frac']:.2%}")
                    
                    if 'red' in sim_state:
                        red = sim_state['red']
                        print("\n  Threat (Red):")
                        print(f"    • Position: [{red['pos_m'][0]:.1f}, {red['pos_m'][1]:.1f}, {red['pos_m'][2]:.1f}] m")
                        print(f"    • Velocity: [{red['vel_mps'][0]:.1f}, {red['vel_mps'][1]:.1f}, {red['vel_mps'][2]:.1f}] m/s")
                    
                    # Calculate relative info
                    if 'blue' in sim_state and 'red' in sim_state:
                        blue_pos = np.array(sim_state['blue']['pos_m'])
                        red_pos = np.array(sim_state['red']['pos_m'])
                        distance = np.linalg.norm(red_pos - blue_pos)
                        print_colored(f"\n  📏 Distance: {distance:.1f} m", Colors.YELLOW)
                
                return True
            else:
                print_colored(f"\n❌ Inference failed: {data.get('error', 'Unknown error')}", Colors.RED)
                return False
                
    except Exception as e:
        print_colored(f"❌ Inference request failed: {e}", Colors.RED)
        return False

async def test_metrics_endpoint(session: aiohttp.ClientSession):
    """Test the metrics endpoint."""
    print_section("Testing Metrics Endpoint")
    
    try:
        async with session.get(METRICS_ENDPOINT) as response:
            # Metrics might be in Prometheus format or JSON
            content_type = response.headers.get('Content-Type', '')
            
            if 'text/plain' in content_type:
                # Prometheus format
                text = await response.text()
                print_colored("✓ Prometheus metrics received", Colors.GREEN)
                print("\nSample metrics (first 500 chars):")
                print(text[:500])
            else:
                # JSON format
                data = await response.json()
                print_colored("✓ JSON metrics received", Colors.GREEN)
                print("\nMetrics Summary:")
                print(f"  • Requests Served: {data.get('requests_served', 0)}")
                print(f"  • Requests Failed: {data.get('requests_failed', 0)}")
                print(f"  • Latency p50: {data.get('latency_p50_ms', 0):.2f} ms")
                print(f"  • Latency p95: {data.get('latency_p95_ms', 0):.2f} ms")
                print(f"  • Safety Clamps: {data.get('safety_clamps_total', 0)}")
            
            return True
            
    except Exception as e:
        print_colored(f"❌ Metrics request failed: {e}", Colors.RED)
        return False

async def run_simulation_loop(session: aiohttp.ClientSession, vecnorm_stats_id: str, num_steps: int = 10):
    """Run a simulated episode with multiple inference calls."""
    print_section("Running Simulated Episode")
    print(f"Simulating {num_steps} timesteps...\n")
    
    for step in range(num_steps):
        t = step * 0.01  # 100 Hz simulation
        request = create_test_request(episode_step=step, t=t, vecnorm_stats_id=vecnorm_stats_id)
        
        print_colored(f"\n⏱️ Step {step+1}/{num_steps} (t={t:.2f}s)", Colors.BLUE)
        
        try:
            start_time = time.perf_counter()
            
            async with session.post(
                INFERENCE_ENDPOINT,
                json=request,
                headers={'Content-Type': 'application/json'}
            ) as response:
                latency_ms = (time.perf_counter() - start_time) * 1000
                data = await response.json()
                
                # Save response to collection
                save_response_to_collection(request, data, latency_ms, f"simulation_step_{step+1}")
                
                if response.status == 200:
                    action = data.get('action', {})
                    thrust = action.get('thrust_cmd', 0)
                    pitch_rate = action.get('rate_cmd_radps', {}).get('pitch', 0)
                    
                    # Get simulation state
                    sim_state = data.get('simulation_state', {})
                    if sim_state and 'blue' in sim_state and 'red' in sim_state:
                        blue_pos = sim_state['blue']['pos_m']
                        red_pos = sim_state['red']['pos_m']
                        distance = np.linalg.norm(np.array(red_pos) - np.array(blue_pos))
                        
                        print(f"  Thrust: {thrust:.2f} | Pitch: {pitch_rate:+.3f} rad/s | Distance: {distance:.1f} m | Latency: {latency_ms:.1f} ms")
                    else:
                        print(f"  Thrust: {thrust:.2f} | Pitch: {pitch_rate:+.3f} rad/s | Latency: {latency_ms:.1f} ms")
                else:
                    print_colored(f"  Failed: {data.get('error', 'Unknown error')}", Colors.RED)
                    
        except Exception as e:
            print_colored(f"  Error: {e}", Colors.RED)
        
        # Small delay between requests
        await asyncio.sleep(0.05)
    
    print_colored("\n✓ Simulation loop completed!", Colors.GREEN)

async def main():
    """Main test sequence."""
    print_colored("""
╔═══════════════════════════════════════════════════════════╗
║     Hlynr Unity-RL Inference API Test Client             ║
║     Testing server at: http://localhost:5000             ║
╚═══════════════════════════════════════════════════════════╝
    """, Colors.CYAN + Colors.BOLD)
    
    async with aiohttp.ClientSession() as session:
        # Wait for server to be ready
        if not await wait_for_server(session):
            print_colored("\n❌ Server did not start within timeout period!", Colors.RED)
            print("Please ensure the server is running:")
            print("  uvicorn src.hlynr_bridge.server:app --host 0.0.0.0 --port 5000")
            return 1
        
        # Test health endpoint
        health_ok = await test_health_endpoint(session)
        if not health_ok:
            print_colored("\n⚠️ Server health check indicates issues", Colors.YELLOW)
        
        # Get server configuration
        server_config = await get_server_config(session)
        vecnorm_stats_id = server_config.get('vecnorm_stats_id', 'vecnorm_checkpoints_obs_v1.0_43d32970')
        
        # Test single inference with correct VecNormalize ID
        test_request = create_test_request(vecnorm_stats_id=vecnorm_stats_id)
        inference_ok = await test_inference_endpoint(session, test_request)
        if not inference_ok:
            print_colored("\n⚠️ Inference test failed", Colors.YELLOW)
        
        # Test metrics endpoint
        await test_metrics_endpoint(session)
        
        # Run simulation loop
        await run_simulation_loop(session, vecnorm_stats_id, num_steps=10)
        
        # Final metrics check
        print_section("Final Metrics")
        await test_metrics_endpoint(session)
        
        # Save all responses to JSON file
        json_filepath = save_responses_to_json()
        
        print_colored("""
╔═══════════════════════════════════════════════════════════╗
║                    Test Complete!                        ║
║     Check the output above for any issues                ║
║     Full response data saved to JSON file                ║
╚═══════════════════════════════════════════════════════════╝
        """, Colors.GREEN + Colors.BOLD)
        
        return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)