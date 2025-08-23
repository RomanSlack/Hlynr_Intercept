#!/usr/bin/env python3
"""
Test client for Hlynr Unity-RL Inference API
Tests all endpoints and visualizes the response data
Supports both standard testing and realistic episode simulation
"""

import json
import time
import asyncio
import aiohttp
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sys
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

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

# Episode simulation classes and configuration
class EpisodeOutcome(Enum):
    """Possible episode outcomes."""
    SUCCESS = "SUCCESS"
    MISS = "MISS"
    FUEL_OUT = "FUEL_OUT"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"

@dataclass
class CompletionConditions:
    """Episode completion condition thresholds."""
    intercept_distance_m: float = 200.0  # More generous for initial testing
    miss_distance_m: float = 12000.0     # Slightly more generous  
    fuel_threshold: float = 0.05
    max_duration_s: float = 120.0

@dataclass
class EpisodeConfig:
    """Configuration for episode simulation."""
    scenario_name: str = "default"
    update_rate_hz: float = 1.0
    max_duration_s: float = 120.0
    completion_conditions: CompletionConditions = None
    
    def __post_init__(self):
        if self.completion_conditions is None:
            self.completion_conditions = CompletionConditions()

@dataclass
class ScenarioConfig:
    """Initial conditions for a scenario."""
    name: str
    description: str
    blue_pos_m: List[float]
    blue_vel_mps: List[float]
    blue_fuel_frac: float
    red_pos_m: List[float]
    red_vel_mps: List[float]
    update_rate_hz: float = 1.0
    max_duration_s: float = 120.0

# Predefined scenarios
SCENARIOS = {
    "easy": ScenarioConfig(
        name="easy",
        description="Basic head-on intercept",
        blue_pos_m=[0.0, 0.0, 1000.0],
        blue_vel_mps=[200.0, 0.0, 0.0],
        blue_fuel_frac=1.0,
        red_pos_m=[8000.0, 0.0, 1000.0],
        red_vel_mps=[-100.0, 0.0, 0.0],
        update_rate_hz=1.0,
        max_duration_s=60.0
    ),
    "medium": ScenarioConfig(
        name="medium",
        description="Crossing trajectory intercept",
        blue_pos_m=[0.0, 0.0, 1000.0],
        blue_vel_mps=[150.0, 0.0, 0.0],
        blue_fuel_frac=0.8,
        red_pos_m=[6000.0, 2000.0, 1000.0],
        red_vel_mps=[-80.0, -40.0, 0.0],
        update_rate_hz=0.5,
        max_duration_s=90.0
    ),
    "hard": ScenarioConfig(
        name="hard",
        description="High-speed evasive target",
        blue_pos_m=[0.0, 0.0, 500.0],
        blue_vel_mps=[180.0, 0.0, 0.0],
        blue_fuel_frac=0.7,
        red_pos_m=[5000.0, 1000.0, 1500.0],
        red_vel_mps=[-120.0, -60.0, -20.0],
        update_rate_hz=0.5,
        max_duration_s=120.0
    )
}

@dataclass
class EpisodeMetrics:
    """Metrics collected during episode simulation."""
    outcome: EpisodeOutcome
    duration_s: float
    total_commands: int
    final_distance_m: float
    fuel_consumed: float
    average_latency_ms: float
    max_latency_ms: float
    intercept_time_s: Optional[float] = None
    trajectory_data: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.trajectory_data is None:
            self.trajectory_data = []

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

def create_episode_request(scenario: ScenarioConfig, episode_step: int, t: float, current_blue_state: Dict[str, Any], vecnorm_stats_id: str) -> Dict[str, Any]:
    """Create an inference request for episode simulation with dynamic state."""
    # Calculate relative guidance information
    blue_pos = np.array(current_blue_state["pos_m"])
    red_pos = np.array([
        scenario.red_pos_m[0] + scenario.red_vel_mps[0] * t,
        scenario.red_pos_m[1] + scenario.red_vel_mps[1] * t,
        scenario.red_pos_m[2] + scenario.red_vel_mps[2] * t
    ])
    
    # Line of sight calculations
    los_vector = red_pos - blue_pos
    range_m = np.linalg.norm(los_vector)
    los_unit = los_vector / range_m if range_m > 0.1 else np.array([1.0, 0.0, 0.0])
    
    # Closing velocity calculation
    blue_vel = np.array(current_blue_state["vel_mps"])
    red_vel = np.array(scenario.red_vel_mps)
    closing_speed_mps = -np.dot(los_unit, blue_vel - red_vel)
    
    return {
        "meta": {
            "episode_id": f"episode_{scenario.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "t": t,
            "dt": 1.0 / scenario.update_rate_hz,
            "sim_tick": episode_step
        },
        "frames": {
            "frame": "ENU",
            "unity_lh": True
        },
        "blue": current_blue_state,
        "red": {
            "pos_m": red_pos.tolist(),
            "vel_mps": scenario.red_vel_mps,
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0]
        },
        "guidance": {
            "los_unit": los_unit.tolist(),
            "los_rate_radps": [0.0, 0.01, 0.0],  # Simplified for now
            "range_m": float(range_m),
            "closing_speed_mps": float(closing_speed_mps),
            "fov_ok": bool(range_m < 15000.0),  # Within sensor range
            "g_limit_ok": True
        },
        "env": {
            "wind_mps": [2.0, 1.0, 0.0],
            "noise_std": 0.01,
            "episode_step": episode_step,
            "max_steps": int(scenario.max_duration_s * scenario.update_rate_hz)
        },
        "normalization": {
            "obs_version": "obs_v1.0",
            "vecnorm_stats_id": vecnorm_stats_id
        }
    }

def check_completion_conditions(simulation_state: Dict[str, Any], fuel_consumed: float, duration_s: float, conditions: CompletionConditions) -> Tuple[bool, EpisodeOutcome]:
    """Check if episode completion conditions are met."""
    if not simulation_state or 'blue' not in simulation_state or 'red' not in simulation_state:
        return False, EpisodeOutcome.ERROR
    
    blue_pos = np.array(simulation_state['blue']['pos_m'])
    red_pos = np.array(simulation_state['red']['pos_m'])
    distance = np.linalg.norm(red_pos - blue_pos)
    
    # Check intercept success
    if distance <= conditions.intercept_distance_m:
        return True, EpisodeOutcome.SUCCESS
    
    # Check if target is diverging (miss condition)
    if distance >= conditions.miss_distance_m:
        return True, EpisodeOutcome.MISS
    
    # Check fuel depletion
    fuel_remaining = simulation_state['blue'].get('fuel_frac', 1.0)
    if fuel_remaining <= conditions.fuel_threshold:
        return True, EpisodeOutcome.FUEL_OUT
    
    # Check timeout
    if duration_s >= conditions.max_duration_s:
        return True, EpisodeOutcome.TIMEOUT
    
    return False, EpisodeOutcome.ERROR

def update_blue_state_with_response(current_state: Dict[str, Any], response: Dict[str, Any], dt: float) -> Dict[str, Any]:
    """Update blue state based on simulation response and control commands."""
    # If simulation state is available, use it directly
    if 'simulation_state' in response and response['simulation_state'] and 'blue' in response['simulation_state']:
        return response['simulation_state']['blue'].copy()
    
    # Otherwise, integrate the control commands (simplified physics)
    new_state = current_state.copy()
    
    if 'action' in response:
        action = response['action']
        thrust_cmd = action.get('thrust_cmd', 0.0)
        rate_cmd = action.get('rate_cmd_radps', {})
        
        # Simple integration - in reality, this would involve full 6DOF dynamics
        current_vel = np.array(new_state['vel_mps'])
        
        # Apply thrust (simplified as acceleration along velocity vector)
        if np.linalg.norm(current_vel) > 0.1:
            vel_unit = current_vel / np.linalg.norm(current_vel)
            thrust_accel = thrust_cmd * 50.0  # Max 50 m/s² acceleration
            new_vel = current_vel + vel_unit * thrust_accel * dt
        else:
            new_vel = current_vel
        
        # Update position
        current_pos = np.array(new_state['pos_m'])
        new_pos = current_pos + current_vel * dt + 0.5 * (new_vel - current_vel) * dt
        
        # Update fuel (simplified consumption model)
        fuel_consumption_rate = 0.01 * thrust_cmd  # 1% per second at full thrust
        new_fuel = max(0.0, new_state['fuel_frac'] - fuel_consumption_rate * dt)
        
        new_state.update({
            'pos_m': new_pos.tolist(),
            'vel_mps': new_vel.tolist(),
            'fuel_frac': float(new_fuel)
        })
    
    return new_state

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

def save_responses_to_json(filename: str = None, episode_metrics: EpisodeMetrics = None):
    """Save all collected responses to a JSON file with optional episode metrics."""
    global collected_responses
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if episode_metrics:
            filename = f"episode_simulation_{episode_metrics.outcome.value.lower()}_{timestamp}.json"
        else:
            filename = f"inference_test_responses_{timestamp}.json"
    
    # Create test results summary
    test_summary = {
        "test_metadata": {
            "timestamp": datetime.now().isoformat(),
            "server_url": SERVER_URL,
            "test_type": "episode_simulation" if episode_metrics else "standard_testing",
            "total_requests": len(collected_responses),
            "inference_requests": len([r for r in collected_responses if r["endpoint"] == "inference" or r["endpoint"].startswith("simulation_") or r["endpoint"].startswith("episode_")]),
            "simulation_steps": len([r for r in collected_responses if r["endpoint"].startswith("simulation_") or r["endpoint"].startswith("episode_")])
        },
        "responses": collected_responses
    }
    
    # Add episode metrics if available
    if episode_metrics:
        test_summary["episode_metadata"] = {
            "scenario": episode_metrics.trajectory_data[0]["blue_state"] if episode_metrics.trajectory_data else "unknown",
            "outcome": episode_metrics.outcome.value,
            "duration": episode_metrics.duration_s,
            "total_commands": episode_metrics.total_commands,
            "final_distance": episode_metrics.final_distance_m,
            "fuel_consumed": episode_metrics.fuel_consumed,
            "intercept_time": episode_metrics.intercept_time_s,
            "performance_metrics": {
                "average_latency_ms": episode_metrics.average_latency_ms,
                "max_latency_ms": episode_metrics.max_latency_ms,
                "success_rate": 1.0 if episode_metrics.outcome == EpisodeOutcome.SUCCESS else 0.0
            }
        }
        
        # Add trajectory data for Unity replay
        if episode_metrics.trajectory_data:
            test_summary["trajectory_data"] = episode_metrics.trajectory_data
            test_summary["unity_replay_format"] = {
                "coordinate_system": "ENU_to_Unity",
                "update_rate_hz": episode_metrics.trajectory_data[1]["t"] - episode_metrics.trajectory_data[0]["t"] if len(episode_metrics.trajectory_data) > 1 else 1.0,
                "total_duration_s": episode_metrics.duration_s
            }
    
    filepath = Path(filename)
    
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        elif isinstance(obj, (np.str_, np.unicode_)):
            return str(obj)
        else:
            return str(obj)
    
    with open(filepath, 'w') as f:
        json.dump(test_summary, f, indent=2, default=convert_numpy_types)
    
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
            
            if episode_metrics:
                print_colored(f"📈 Episode Summary:", Colors.CYAN)
                print(f"  • Outcome: {episode_metrics.outcome.value}")
                print(f"  • Duration: {episode_metrics.duration_s:.1f} seconds")
                print(f"  • Commands Sent: {episode_metrics.total_commands}")
                print(f"  • Final Distance: {episode_metrics.final_distance_m:.1f} m")
                print(f"  • Fuel Consumed: {episode_metrics.fuel_consumed:.1%}")
                if episode_metrics.intercept_time_s:
                    print(f"  • Intercept Time: {episode_metrics.intercept_time_s:.1f} seconds")
            
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

async def run_realistic_episode_simulation(session: aiohttp.ClientSession, vecnorm_stats_id: str, scenario_name: str, update_rate_hz: float = 1.0, max_duration_s: Optional[float] = None) -> EpisodeMetrics:
    """Run a complete episode simulation with realistic timing and completion conditions."""
    scenario = SCENARIOS.get(scenario_name)
    if not scenario:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")
    
    # Override scenario parameters if provided
    if update_rate_hz != 1.0:
        scenario.update_rate_hz = update_rate_hz
    if max_duration_s:
        scenario.max_duration_s = max_duration_s
    
    print_section(f"Realistic Episode Simulation: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"Update Rate: {scenario.update_rate_hz:.1f} Hz ({1.0/scenario.update_rate_hz:.1f}s intervals)")
    print(f"Max Duration: {scenario.max_duration_s:.1f} seconds")
    print()
    
    # Initialize episode state
    episode_config = EpisodeConfig(
        scenario_name=scenario.name,
        update_rate_hz=scenario.update_rate_hz,
        max_duration_s=scenario.max_duration_s
    )
    
    # Initialize blue state from scenario
    current_blue_state = {
        "pos_m": scenario.blue_pos_m.copy(),
        "vel_mps": scenario.blue_vel_mps.copy(),
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "ang_vel_radps": [0.0, 0.0, 0.0],
        "fuel_frac": scenario.blue_fuel_frac
    }
    
    # Episode tracking
    start_time = time.perf_counter()
    episode_step = 0
    t = 0.0
    dt = 1.0 / scenario.update_rate_hz
    latencies = []
    trajectory_data = []
    initial_fuel = scenario.blue_fuel_frac
    
    print_colored("🚀 Episode Starting...", Colors.GREEN)
    print(f"Initial Distance: {np.linalg.norm(np.array(scenario.red_pos_m) - np.array(scenario.blue_pos_m)):.1f} m")
    
    while True:
        episode_start_step_time = time.perf_counter()
        
        try:
            # Create request for current state
            request = create_episode_request(scenario, episode_step, t, current_blue_state, vecnorm_stats_id)
            
            # Send inference request
            inference_start_time = time.perf_counter()
            async with session.post(
                INFERENCE_ENDPOINT,
                json=request,
                headers={'Content-Type': 'application/json'}
            ) as response:
                latency_ms = (time.perf_counter() - inference_start_time) * 1000
                data = await response.json()
                latencies.append(latency_ms)
                
                if response.status != 200:
                    print_colored(f"❌ Inference failed at step {episode_step}: {data.get('error', 'Unknown error')}", Colors.RED)
                    return EpisodeMetrics(
                        outcome=EpisodeOutcome.ERROR,
                        duration_s=float(t),
                        total_commands=int(episode_step),
                        final_distance_m=0.0,
                        fuel_consumed=0.0,
                        average_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
                        max_latency_ms=float(max(latencies)) if latencies else 0.0
                    )
                
                # Extract simulation state and control commands
                sim_state = data.get('simulation_state', {})
                action = data.get('action', {})
                
                # Save trajectory data point
                red_current_pos = [
                    float(scenario.red_pos_m[0] + scenario.red_vel_mps[0] * t),
                    float(scenario.red_pos_m[1] + scenario.red_vel_mps[1] * t),
                    float(scenario.red_pos_m[2] + scenario.red_vel_mps[2] * t)
                ]
                
                trajectory_point = {
                    "t": float(t),
                    "episode_step": int(episode_step),
                    "blue_state": current_blue_state.copy(),
                    "red_state": {
                        "pos_m": red_current_pos,
                        "vel_mps": list(scenario.red_vel_mps)
                    },
                    "command": action,
                    "latency_ms": float(latency_ms),
                    "diagnostics": data.get('diagnostics', {}),
                    "safety": data.get('safety', {})
                }
                trajectory_data.append(trajectory_point)
                
                # Save response to global collection
                save_response_to_collection(request, data, latency_ms, f"episode_{scenario.name}_step_{episode_step}")
                
                # Update blue state from server's simulation_state (if available)
                if sim_state and 'blue' in sim_state:
                    # Use server's physics simulation
                    current_blue_state = sim_state['blue'].copy()
                else:
                    # Fallback to client-side integration
                    current_blue_state = update_blue_state_with_response(current_blue_state, data, dt)
                
                # Calculate current distance
                blue_pos = np.array(current_blue_state['pos_m'])
                red_pos = np.array(trajectory_point["red_state"]["pos_m"])
                current_distance = float(np.linalg.norm(red_pos - blue_pos))
                
                # Display progress
                thrust = action.get('thrust_cmd', 0.0)
                rate_cmd = action.get('rate_cmd_radps', {})
                pitch_rate = rate_cmd.get('pitch', 0.0)
                yaw_rate = rate_cmd.get('yaw', 0.0)
                roll_rate = rate_cmd.get('roll', 0.0)
                fuel_remaining = current_blue_state.get('fuel_frac', 0.0)
                
                print(f"t={t:5.1f}s | Dist: {current_distance:6.1f}m | T: {thrust:.2f} | P: {pitch_rate:+.2f} Y: {yaw_rate:+.2f} R: {roll_rate:+.2f} | Fuel: {fuel_remaining:.1%} | {latency_ms:4.1f}ms")
                
                # Check completion conditions
                fuel_consumed = initial_fuel - fuel_remaining
                is_complete, outcome = check_completion_conditions(sim_state, fuel_consumed, t, episode_config.completion_conditions)
                
                # Debug completion conditions
                if episode_step % 10 == 0:  # Every 10 steps, show condition status
                    conditions = episode_config.completion_conditions
                    print(f"    Status: Distance {current_distance:.1f}m (need <{conditions.intercept_distance_m}m), Fuel {fuel_remaining:.1%} (need >{conditions.fuel_threshold:.1%}), Time {t:.1f}s (max {conditions.max_duration_s}s)")
                
                if is_complete:
                    total_duration = time.perf_counter() - start_time
                    
                    print_colored(f"\n🏁 Episode Complete: {outcome.value}", 
                                Colors.GREEN if outcome == EpisodeOutcome.SUCCESS else Colors.YELLOW)
                    print(f"Final Distance: {current_distance:.1f} m")
                    print(f"Duration: {t:.1f} seconds ({episode_step + 1} steps)")
                    print(f"Fuel Consumed: {fuel_consumed:.1%}")
                    print(f"Average Latency: {np.mean(latencies):.1f} ms")
                    print(f"Max Latency: {max(latencies):.1f} ms")
                    
                    return EpisodeMetrics(
                        outcome=outcome,
                        duration_s=float(t),
                        total_commands=int(episode_step + 1),
                        final_distance_m=float(current_distance),
                        fuel_consumed=float(fuel_consumed),
                        average_latency_ms=float(np.mean(latencies)),
                        max_latency_ms=float(max(latencies)),
                        intercept_time_s=float(t) if outcome == EpisodeOutcome.SUCCESS else None,
                        trajectory_data=trajectory_data
                    )
                
        except Exception as e:
            print_colored(f"❌ Error at step {episode_step}: {e}", Colors.RED)
            return EpisodeMetrics(
                outcome=EpisodeOutcome.ERROR,
                duration_s=float(t),
                total_commands=int(episode_step),
                final_distance_m=0.0,
                fuel_consumed=0.0,
                average_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
                max_latency_ms=float(max(latencies)) if latencies else 0.0
            )
        
        # Update timing
        episode_step += 1
        t += dt
        
        # Wait for next update (maintaining realistic timing)
        step_duration = time.perf_counter() - episode_start_step_time
        sleep_time = max(0, dt - step_duration)
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hlynr Unity-RL Inference API Test Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard testing (default)
  python test_inference_client.py
  
  # Episode simulation with default scenario (easy)
  python test_inference_client.py --episode
  
  # Episode simulation with specific scenario
  python test_inference_client.py --episode --scenario medium --update-rate 0.5
  
  # Episode simulation with custom duration
  python test_inference_client.py --episode --scenario hard --duration 90
        """
    )
    
    parser.add_argument(
        "--episode", 
        action="store_true",
        help="Run realistic episode simulation instead of standard testing"
    )
    
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="easy",
        help="Scenario to run in episode mode (default: easy)"
    )
    
    parser.add_argument(
        "--update-rate",
        type=float,
        default=1.0,
        help="Update rate in Hz for episode simulation (default: 1.0)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        help="Maximum episode duration in seconds (overrides scenario default)"
    )
    
    parser.add_argument(
        "--output",
        help="Output filename for JSON results (auto-generated if not specified)"
    )
    
    return parser.parse_args()

async def main():
    """Main test sequence with episode simulation support."""
    args = parse_arguments()
    
    # Display header based on mode
    if args.episode:
        print_colored(f"""
╔═══════════════════════════════════════════════════════════╗
║  Hlynr Unity-RL Inference API - Episode Simulation      ║
║  Server: http://localhost:5000                           ║
║  Scenario: {args.scenario:<15} Update Rate: {args.update_rate:.1f} Hz        ║
╚═══════════════════════════════════════════════════════════╝
        """, Colors.CYAN + Colors.BOLD)
    else:
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
            print("  ./start_inference_server.sh")
            print("  or manually: uvicorn src.hlynr_bridge.server:app --host 0.0.0.0 --port 5000")
            return 1
        
        # Test health endpoint
        health_ok = await test_health_endpoint(session)
        if not health_ok:
            print_colored("\n⚠️ Server health check indicates issues", Colors.YELLOW)
        
        # Get server configuration
        server_config = await get_server_config(session)
        vecnorm_stats_id = server_config.get('vecnorm_stats_id', 'vecnorm_checkpoints_obs_v1.0_43d32970')
        
        episode_metrics = None
        
        if args.episode:
            # Run realistic episode simulation
            try:
                episode_metrics = await run_realistic_episode_simulation(
                    session=session,
                    vecnorm_stats_id=vecnorm_stats_id,
                    scenario_name=args.scenario,
                    update_rate_hz=args.update_rate,
                    max_duration_s=args.duration
                )
            except Exception as e:
                print_colored(f"\n❌ Episode simulation failed: {e}", Colors.RED)
                return 1
        else:
            # Run standard testing sequence
            
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
        json_filepath = save_responses_to_json(filename=args.output, episode_metrics=episode_metrics)
        
        # Display completion message
        if args.episode:
            success_emoji = "🎯" if episode_metrics and episode_metrics.outcome == EpisodeOutcome.SUCCESS else "🏁"
            print_colored(f"""
╔═══════════════════════════════════════════════════════════╗
║             {success_emoji} Episode Simulation Complete! {success_emoji}             ║
║     Outcome: {episode_metrics.outcome.value if episode_metrics else 'Unknown':<20}                    ║
║     Unity-ready trajectory data saved to JSON file       ║
╚═══════════════════════════════════════════════════════════╝
            """, Colors.GREEN + Colors.BOLD if episode_metrics and episode_metrics.outcome == EpisodeOutcome.SUCCESS else Colors.YELLOW + Colors.BOLD)
        else:
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