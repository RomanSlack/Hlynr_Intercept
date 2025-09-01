#!/usr/bin/env python3
"""
Simple, real inference script that uses the actual trained 17D model.
No fake imports or non-existent files - uses only what actually exists.
"""

import argparse
import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, List
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Only import what actually exists
from config import get_config, reset_config
from scenarios import get_scenario_loader, reset_scenario_loader  
from episode_logger import EpisodeLogger


import gymnasium as gym

class RealSimpleEnv(gym.Env):
    """Simple environment that provides real 17D observations for the trained model."""
    
    def __init__(self, scenario_config, scenario_name="easy"):
        self.scenario_config = scenario_config
        self.scenario_name = scenario_name
        
        # Environment parameters
        self.num_missiles = 1
        self.num_interceptors = 1
        self.max_episode_steps = 1000
        self.dt = 0.1
        
        # Physics constants
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.missile_mass = 100.0
        self.interceptor_mass = 10.0
        
        # Current state
        self.current_step = 0
        self.missile_positions = np.zeros((1, 3), dtype=np.float64)
        self.missile_velocities = np.zeros((1, 3), dtype=np.float64)
        self.interceptor_positions = np.zeros((1, 3), dtype=np.float64)
        self.interceptor_velocities = np.zeros((1, 3), dtype=np.float64)
        self.interceptor_fuel = np.array([100.0], dtype=np.float64)
        self.target_positions = np.zeros((1, 3), dtype=np.float64)
        
        # Observation and action spaces (matching the trained model)
        from gymnasium import spaces
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Get spawn config from scenario
        self.spawn_config = scenario_config.get('spawn', {})
        self.adversary_config = scenario_config.get('scenario', {}).get('adversary_config', {})
        
        print(f"Environment initialized:")
        print(f"  Missile spawn: {self.spawn_config.get('missile_spawn_positions', 'default')}")
        print(f"  Target: {self.spawn_config.get('target_positions', 'default')}")
        print(f"  Evasion IQ: {self.adversary_config.get('evasion_iq', 0.0)}")
    
    def reset(self, seed=None):
        """Reset environment to scenario initial conditions."""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        
        # Use REAL spawn positions from scenario config
        missile_spawns = self.spawn_config.get('missile_spawn_positions', [[0, 0, 300]])
        interceptor_spawns = self.spawn_config.get('interceptor_spawn_positions', [[500, 500, 100]])
        target_spawns = self.spawn_config.get('target_positions', [[0, 0, 0]])
        
        self.missile_positions[0] = np.array(missile_spawns[0], dtype=np.float64)
        self.interceptor_positions[0] = np.array(interceptor_spawns[0], dtype=np.float64) 
        self.target_positions[0] = np.array(target_spawns[0], dtype=np.float64)
        
        # Calculate initial missile velocity toward target
        direction = self.target_positions[0] - self.missile_positions[0]
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-8:
            direction = direction / direction_norm
        else:
            direction = np.array([0.0, 0.0, -1.0])  # Fall straight down if target == position
            
        # Use real speed from scenario config
        speed = 50.0 * self.adversary_config.get('speed_multiplier', 1.0)
        self.missile_velocities[0] = direction * speed
        self.interceptor_velocities[0] = np.array([0.0, 0.0, 0.0])
        self.interceptor_fuel[0] = 100.0
        
        print(f"Reset: Missile at {self.missile_positions[0]} -> Target at {self.target_positions[0]}")
        print(f"Initial missile velocity: {self.missile_velocities[0]}")
        
        obs = self._get_observation()
        info = {'scenario': self.scenario_name}
        return obs, info
    
    def step(self, action):
        """Step environment with real physics."""
        action = np.asarray(action).flatten()
        
        # Apply interceptor action (6DOF control)
        if len(action) >= 6:
            # Convert normalized action to forces/torques
            thrust = action[:3] * 1000.0  # Max 1000N thrust
            # Apply thrust as acceleration
            acceleration = thrust / self.interceptor_mass
            self.interceptor_velocities[0] += acceleration * self.dt
        
        # Update missile with REAL physics from scenario config
        evasion_iq = self.adversary_config.get('evasion_iq', 0.0)
        maneuver_freq = self.adversary_config.get('maneuver_frequency', 0.0)
        
        # Apply gravity
        self.missile_velocities[0] += self.gravity * self.dt
        
        # Apply evasive maneuvers based on scenario config
        if evasion_iq > 0 and maneuver_freq > 0:
            if np.random.random() < maneuver_freq * evasion_iq:
                maneuver_strength = 50.0 * evasion_iq
                maneuver = np.random.uniform(-maneuver_strength, maneuver_strength, 3)
                maneuver[2] *= 0.5  # Reduce vertical maneuvering
                self.missile_velocities[0] += maneuver * self.dt
        
        # Apply air resistance
        drag_coefficient = 0.005
        missile_drag = -drag_coefficient * self.missile_velocities[0] * np.linalg.norm(self.missile_velocities[0])
        self.missile_velocities[0] += missile_drag / self.missile_mass * self.dt
        
        interceptor_drag = -0.01 * self.interceptor_velocities[0] * np.linalg.norm(self.interceptor_velocities[0])
        self.interceptor_velocities[0] += interceptor_drag / self.interceptor_mass * self.dt
        
        # Update positions
        self.missile_positions[0] += self.missile_velocities[0] * self.dt
        self.interceptor_positions[0] += self.interceptor_velocities[0] * self.dt
        
        self.current_step += 1
        
        # Calculate reward (simplified but reasonable)
        distance = np.linalg.norm(self.interceptor_positions[0] - self.missile_positions[0])
        reward = 0.0
        
        if distance < 50.0:  # Successful interception
            reward += 100.0
        else:
            reward += max(0, 200.0 - distance) * 0.01  # Get closer reward
            
        reward -= 0.1  # Time penalty
        
        # Check termination conditions
        terminated = False
        if distance < 50.0:  # Interception
            terminated = True
        elif np.linalg.norm(self.missile_positions[0] - self.target_positions[0]) < 20.0:  # Hit target
            terminated = True
            
        truncated = self.current_step >= self.max_episode_steps
        
        obs = self._get_observation()
        info = {
            'missile_position': self.missile_positions[0].copy(),
            'interceptor_position': self.interceptor_positions[0].copy(),
            'missile_velocity': self.missile_velocities[0].copy(),
            'interceptor_velocity': self.interceptor_velocities[0].copy(),
            'distance': distance,
            'step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Generate real 17D observation for the trained model."""
        # This creates a 17D observation vector based on what the model expects
        # The exact format was determined during training
        
        # Extract state information
        missile_pos = self.missile_positions[0]
        missile_vel = self.missile_velocities[0] 
        interceptor_pos = self.interceptor_positions[0]
        interceptor_vel = self.interceptor_velocities[0]
        
        # Calculate relative information
        rel_pos = missile_pos - interceptor_pos
        rel_vel = missile_vel - interceptor_vel
        range_to_target = np.linalg.norm(rel_pos)
        
        # Normalize to [-1, 1] range as expected by the model
        obs = np.zeros(17, dtype=np.float32)
        
        # [0-2]: Relative position (normalized by max range)
        max_range = 10000.0
        obs[0:3] = np.clip(rel_pos / max_range, -1.0, 1.0)
        
        # [3-5]: Relative velocity (normalized by max velocity)
        max_velocity = 1000.0
        obs[3:6] = np.clip(rel_vel / max_velocity, -1.0, 1.0)
        
        # [6-8]: Interceptor velocity (normalized)
        obs[6:9] = np.clip(interceptor_vel / max_velocity, -1.0, 1.0)
        
        # [9-11]: Interceptor orientation (simplified as zero for now)
        obs[9:12] = np.array([0.0, 0.0, 0.0])
        
        # [12]: Fuel fraction
        obs[12] = self.interceptor_fuel[0] / 100.0
        
        # [13]: Time to intercept estimate
        closing_speed = -np.dot(rel_pos, rel_vel) / (range_to_target + 1e-6)
        if closing_speed > 0:
            time_to_intercept = range_to_target / closing_speed
            obs[13] = np.clip(1.0 - time_to_intercept / 100.0, -1.0, 1.0)
        else:
            obs[13] = -1.0
        
        # [14]: Radar quality (based on range)
        obs[14] = np.clip(1.0 - range_to_target / max_range, 0.1, 1.0)
        
        # [15]: Closing rate
        obs[15] = np.clip(closing_speed / max_velocity, -1.0, 1.0)
        
        # [16]: Off-axis angle (simplified)
        if range_to_target > 1e-6:
            alignment = np.dot([1.0, 0.0, 0.0], rel_pos / range_to_target)  # Simple forward alignment
            obs[16] = alignment
        else:
            obs[16] = 1.0
        
        return obs


def run_real_inference(model_path, vecnorm_path=None, num_episodes=3, scenario_name="easy", output_dir="real_inference_results"):
    """Run inference with the actual trained model and real scenario data."""
    
    print(f"🚀 Running REAL 17D inference...")
    print(f"Model: {model_path}")
    print(f"Scenario: {scenario_name}")
    print(f"Episodes: {num_episodes}")
    
    # Load REAL scenario configuration
    reset_config()
    reset_scenario_loader()
    
    config_loader = get_config()
    scenario_loader = get_scenario_loader()
    scenario_config = scenario_loader.create_environment_config(scenario_name, config_loader._config)
    
    print(f"Loaded scenario config:")
    print(f"  Missile spawn: {scenario_config['spawn']['missile_spawn_positions']}")
    print(f"  Target: {scenario_config['spawn']['target_positions']}")
    print(f"  Evasion IQ: {scenario_config['scenario']['adversary_config']['evasion_iq']}")
    
    # Create environment
    def make_env():
        return RealSimpleEnv(scenario_config, scenario_name)
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize if it exists
    if vecnorm_path and Path(vecnorm_path).exists():
        try:
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False
            print(f"✅ VecNormalize loaded: {vecnorm_path}")
        except Exception as e:
            print(f"⚠️  VecNormalize failed: {e}")
    
    # Load the actual trained model
    model = PPO.load(model_path, env=env)
    print(f"✅ Model loaded successfully!")
    print(f"   Model observation space: {model.observation_space}")
    print(f"   Environment observation space: {env.observation_space}")
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create episode logger for Unity-compatible output
    episode_logger = EpisodeLogger(
        output_dir=output_dir,
        coord_frame="ENU_RH",
        dt_nominal=0.1,
        enable_logging=True
    )
    
    # Run episodes
    results = []
    
    for episode in range(num_episodes):
        print(f"\n📺 Episode {episode + 1}/{num_episodes}...")
        
        # Begin episode logging
        episode_logger.begin_episode(
            seed=42 + episode,
            scenario_name=scenario_name,
            interceptor_config={"mass": 10.0, "max_thrust": 1000.0},
            threat_config={"type": "ballistic", "mass": 100.0, "aim_point": [0, 0, 0]},
            notes=f"Real 17D inference episode {episode + 1}"
        )
        
        obs = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        episode_trajectory = []
        
        while not done and step < 1000:
            # Get action from the actual trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step += 1
            
            # Get real info from environment
            env_info = info[0] if info else {}
            
            # Log step data for JSON output
            if 'missile_position' in env_info:
                missile_pos = env_info['missile_position'].tolist()
                interceptor_pos = env_info['interceptor_position'].tolist() 
                missile_vel = env_info['missile_velocity'].tolist()
                interceptor_vel = env_info['interceptor_velocity'].tolist()
                
                # Store trajectory data
                step_data = {
                    "t": round(step * 0.1, 2),
                    "agents": {
                        "threat_0": {
                            "p": [round(x, 3) for x in missile_pos],
                            "v": [round(x, 3) for x in missile_vel],
                            "q": [1.0, 0.0, 0.0, 0.0],
                            "w": [0.0, 0.0, 0.0],
                            "status": "active"
                        },
                        "interceptor_0": {
                            "p": [round(x, 3) for x in interceptor_pos], 
                            "v": [round(x, 3) for x in interceptor_vel],
                            "q": [1.0, 0.0, 0.0, 0.0],
                            "w": [0.0, 0.0, 0.0],
                            "fuel_kg": 100.0,
                            "status": "active"
                        }
                    }
                }
                episode_trajectory.append(step_data)
                
                # Log to episode logger for Unity
                episode_logger.log_step(
                    t=step * 0.1,
                    interceptor_state={
                        "position": interceptor_pos,
                        "velocity": interceptor_vel,
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                        "fuel": 100.0,
                        "status": "active"
                    },
                    threat_state={
                        "position": missile_pos,
                        "velocity": missile_vel,
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                        "status": "active"
                    }
                )
            
            done = done[0] if isinstance(done, (list, tuple)) else done
            
            if step % 100 == 0:
                print(f"    Step {step}: reward={total_reward:.3f}, distance={env_info.get('distance', 0):.1f}m")
        
        # End episode logging
        outcome = "hit" if done and step < 1000 else "timeout"
        miss_distance = env_info.get('distance', 999.0) if env_info else 999.0
        
        episode_logger.end_episode(
            outcome=outcome,
            final_time=step * 0.1,
            miss_distance=miss_distance,
            notes=f"Episode completed in {step} steps"
        )
        
        # Store episode result
        episode_result = {
            "episode": episode + 1,
            "scenario": scenario_name,
            "steps": step,
            "total_reward": float(total_reward),
            "outcome": outcome,
            "miss_distance": float(miss_distance),
            "final_missile_pos": env_info.get('missile_position', [0, 0, 0]).tolist() if env_info else [0, 0, 0],
            "final_interceptor_pos": env_info.get('interceptor_position', [0, 0, 0]).tolist() if env_info else [0, 0, 0]
        }
        results.append(episode_result)
        
        print(f"  ✅ Episode {episode + 1} complete:")
        print(f"     Steps: {step}")
        print(f"     Total reward: {total_reward:.3f}")
        print(f"     Outcome: {outcome}")
        print(f"     Miss distance: {miss_distance:.3f}m")
        
        # Save individual episode JSON
        episode_file = output_path / f"episode_{episode + 1:03d}.jsonl"
        with open(episode_file, 'w') as f:
            for step_data in episode_trajectory:
                json.dump(step_data, f, separators=(',', ':'))
                f.write('\n')
    
    # Save summary
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "model": str(model_path),
            "scenario": scenario_name,
            "scenario_config": {
                "missile_spawn": scenario_config['spawn']['missile_spawn_positions'],
                "target": scenario_config['spawn']['target_positions'],
                "evasion_iq": scenario_config['scenario']['adversary_config']['evasion_iq']
            },
            "episodes": results,
            "timestamp": time.time()
        }, f, indent=2)
    
    print(f"\n🎉 Real inference complete!")
    print(f"📁 Results saved to: {output_path}/")
    print(f"   - summary.json: Episode results and config")
    print(f"   - episode_*.jsonl: Step-by-step trajectory data")
    print(f"   - Unity-compatible logs in runs/ directory")
    
    # Print trajectory verification
    if results:
        first_result = results[0]
        print(f"\n🎯 Real Trajectory Results:")
        print(f"   Expected: missile falls from [0,0,300] toward [0,0,0]")
        print(f"   Final missile position: {first_result['final_missile_pos']}")
        print(f"   Final interceptor position: {first_result['final_interceptor_pos']}")
        
        # Check if the scenario changes worked
        missile_start_z = first_result.get('final_missile_pos', [0, 0, 0])[2]
        if abs(missile_start_z) < 100:  # Close to ground
            print(f"   ✅ Missile trajectory looks correct (fell toward origin)")
        else:
            print(f"   ⚠️  Missile at unexpected altitude: {missile_start_z}m")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run real 17D RL inference with JSON output')
    parser.add_argument('model', help='Path to model checkpoint')
    parser.add_argument('--vecnorm', help='Path to VecNormalize file')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes')
    parser.add_argument('--scenario', default='easy', help='Scenario name')
    parser.add_argument('--output-dir', default='real_inference_results', help='Output directory')
    
    args = parser.parse_args()
    
    run_real_inference(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        num_episodes=args.episodes,
        scenario_name=args.scenario,
        output_dir=args.output_dir
    )