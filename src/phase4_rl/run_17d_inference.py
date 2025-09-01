#!/usr/bin/env python3
"""
Fixed 17D inference script that outputs proper JSON logs.
"""

import argparse
import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, List
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import modules
from config import get_config, reset_config
from scenarios import get_scenario_loader, reset_scenario_loader  
from radar_observations import Radar17DObservation
from episode_logger import EpisodeLogger


class Simple17DEnv:
    """Simple environment wrapper that forces 17D radar observations."""
    
    def __init__(self, scenario_config, scenario_name="easy"):
        self.scenario_config = scenario_config
        self.scenario_name = scenario_name
        
        # Environment parameters
        self.num_missiles = 1
        self.num_interceptors = 1
        self.max_episode_steps = 1000
        self.dt = 0.1
        
        # Physics
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.missile_mass = 100.0
        self.interceptor_mass = 10.0
        
        # Current state
        self.current_step = 0
        self.missile_positions = np.zeros((1, 3), dtype=np.float64)
        self.missile_velocities = np.zeros((1, 3), dtype=np.float64)
        self.interceptor_positions = np.zeros((1, 3), dtype=np.float64)
        self.interceptor_velocities = np.zeros((1, 3), dtype=np.float64)
        self.interceptor_orientations = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self.interceptor_fuel = np.array([100.0], dtype=np.float64)
        self.interceptor_active = np.array([True])
        self.missile_active = np.array([True])
        self.target_positions = np.zeros((1, 3), dtype=np.float64)
        
        # 17D radar observer
        self.radar_observer = Radar17DObservation(max_range=10000.0, max_velocity=1000.0)
        
        # Observation and action spaces
        from gymnasium import spaces
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Get spawn config from scenario
        self.spawn_config = scenario_config.get('spawn', {})
        self.adversary_config = scenario_config.get('scenario', {}).get('adversary_config', {})
    
    def reset(self, seed=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
            self.radar_observer.seed(seed)
            
        self.current_step = 0
        
        # Initialize positions from scenario config
        missile_spawns = self.spawn_config.get('missile_spawn_positions', [[0, 0, 300]])
        interceptor_spawns = self.spawn_config.get('interceptor_spawn_positions', [[500, 500, 100]])
        target_spawns = self.spawn_config.get('target_positions', [[0, 0, 0]])
        
        self.missile_positions[0] = np.array(missile_spawns[0], dtype=np.float64)
        self.interceptor_positions[0] = np.array(interceptor_spawns[0], dtype=np.float64) 
        self.target_positions[0] = np.array(target_spawns[0], dtype=np.float64)
        
        # Initialize velocities
        direction = self.target_positions[0] - self.missile_positions[0]
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-8:
            direction = direction / direction_norm
        else:
            direction = np.array([0.0, 0.0, -1.0])  # Fall straight down
            
        speed = 50.0 * self.adversary_config.get('speed_multiplier', 1.0)
        self.missile_velocities[0] = direction * speed
        self.interceptor_velocities[0] = np.array([0.0, 0.0, 0.0])
        
        # Reset other state
        self.interceptor_orientations[0] = np.array([1.0, 0.0, 0.0, 0.0])
        self.interceptor_fuel[0] = 100.0
        self.interceptor_active[0] = True
        self.missile_active[0] = True
        
        obs = self._get_observation()
        info = {'scenario': self.scenario_name}
        return obs, info
    
    def step(self, action):
        """Step environment."""
        action = np.asarray(action).flatten()
        
        # Apply interceptor action (simplified)
        if len(action) >= 6:
            thrust = action[:3] * 1000.0  # Scale thrust
            self.interceptor_velocities[0] += thrust / self.interceptor_mass * self.dt
        
        # Update missile (ballistic with some physics)
        if self.missile_active[0]:
            # Apply gravity  
            self.missile_velocities[0] += self.gravity * self.dt
            
            # Apply simple evasion if enabled
            evasion_iq = self.adversary_config.get('evasion_iq', 0.0)
            maneuver_freq = self.adversary_config.get('maneuver_frequency', 0.0)
            
            if np.random.random() < maneuver_freq * evasion_iq:
                maneuver = np.random.uniform(-50.0 * evasion_iq, 50.0 * evasion_iq, 3)
                maneuver[2] *= 0.5  # Reduce vertical maneuvering
                self.missile_velocities[0] += maneuver * self.dt
            
            # Apply air resistance
            drag = -0.005 * self.missile_velocities[0] * np.linalg.norm(self.missile_velocities[0])
            self.missile_velocities[0] += drag / self.missile_mass * self.dt
        
        # Update interceptor
        if self.interceptor_active[0]:
            # Simple air resistance
            drag = -0.01 * self.interceptor_velocities[0] * np.linalg.norm(self.interceptor_velocities[0])
            self.interceptor_velocities[0] += drag / self.interceptor_mass * self.dt
        
        # Update positions
        if self.missile_active[0]:
            self.missile_positions[0] += self.missile_velocities[0] * self.dt
        if self.interceptor_active[0]:
            self.interceptor_positions[0] += self.interceptor_velocities[0] * self.dt
        
        self.current_step += 1
        
        # Calculate reward
        distance = np.linalg.norm(self.interceptor_positions[0] - self.missile_positions[0])
        reward = 0.0
        
        if distance < 50.0:  # Successful interception
            reward += 100.0
        else:
            reward += max(0, 200.0 - distance) * 0.01
            
        reward -= 0.1  # Time penalty
        
        # Check termination
        terminated = False
        if distance < 50.0:  # Interception
            terminated = True
        elif np.linalg.norm(self.missile_positions[0] - self.target_positions[0]) < 20.0:  # Hit target
            terminated = True
        elif self.current_step >= self.max_episode_steps:
            terminated = True
            
        truncated = False
        
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
        """Generate 17D radar observation."""
        interceptor_state = {
            'position': self.interceptor_positions[0],
            'velocity': self.interceptor_velocities[0], 
            'orientation': self.interceptor_orientations[0],
            'fuel': self.interceptor_fuel[0]
        }
        
        missile_state = {
            'position': self.missile_positions[0],
            'velocity': self.missile_velocities[0]
        }
        
        # Calculate radar quality based on range
        rel_pos = missile_state['position'] - interceptor_state['position']
        range_to_target = np.linalg.norm(rel_pos)
        radar_quality = np.clip(1.0 - range_to_target / 10000.0, 0.1, 1.0)
        
        obs = self.radar_observer.compute_observation(
            interceptor_state=interceptor_state,
            missile_state=missile_state,
            radar_quality=radar_quality,
            radar_noise=0.02
        )
        
        return obs


def run_17d_inference(model_path, vecnorm_path=None, num_episodes=3, scenario_name="easy", output_dir="inference_17d_results"):
    """Run inference with 17D observations and generate JSON logs."""
    
    print(f"🚀 Running 17D inference...")
    print(f"Model: {model_path}")
    print(f"Scenario: {scenario_name}")
    print(f"Episodes: {num_episodes}")
    
    # Reset and load config
    reset_config()
    reset_scenario_loader()
    
    config_loader = get_config()
    scenario_loader = get_scenario_loader()
    scenario_config = scenario_loader.create_environment_config(scenario_name, config_loader._config)
    
    # Create environment
    def make_env():
        return Simple17DEnv(scenario_config, scenario_name)
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize if available
    if vecnorm_path and Path(vecnorm_path).exists():
        try:
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False
            print(f"✅ VecNormalize loaded: {vecnorm_path}")
        except Exception as e:
            print(f"⚠️  VecNormalize failed: {e}")
    
    # Load model
    model = PPO.load(model_path, env=env)
    print(f"✅ Model loaded successfully!")
    print(f"   Observation space: {model.observation_space}")
    print(f"   Action space: {model.action_space}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create episode logger
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
            notes=f"17D inference episode {episode + 1}"
        )
        
        obs = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        episode_data = []
        
        while not done and step < 1000:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step += 1
            
            # Get info from environment
            env_info = info[0] if info else {}
            
            # Log step data
            if 'missile_position' in env_info:
                missile_pos = env_info['missile_position'].tolist()
                interceptor_pos = env_info['interceptor_position'].tolist() 
                missile_vel = env_info['missile_velocity'].tolist()
                interceptor_vel = env_info['interceptor_velocity'].tolist()
                
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
                episode_data.append(step_data)
                
                # Log to episode logger
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
                print(f"    Step {step}: reward={total_reward:.3f}")
        
        # End episode logging
        outcome = "hit" if step < 1000 and done else "timeout"
        miss_distance = env_info.get('distance', 999.0) if env_info else 999.0
        
        episode_logger.end_episode(
            outcome=outcome,
            final_time=step * 0.1,
            miss_distance=miss_distance,
            notes=f"Episode completed in {step} steps"
        )
        
        # Save episode result
        episode_result = {
            "episode": episode + 1,
            "scenario": scenario_name,
            "steps": step,
            "total_reward": total_reward,
            "outcome": outcome,
            "miss_distance": miss_distance,
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
            for step_data in episode_data:
                json.dump(step_data, f, separators=(',', ':'))
                f.write('\n')
    
    # Save summary
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "model": str(model_path),
            "scenario": scenario_name,
            "episodes": results,
            "timestamp": time.time()
        }, f, indent=2)
    
    print(f"\n🎉 Inference complete!")
    print(f"📁 Results saved to: {output_path}/")
    print(f"   - summary.json: Episode results")
    print(f"   - episode_*.jsonl: Step-by-step trajectory data")
    print(f"   - Episode logs in runs/ directory")
    
    # Print trajectory verification
    if results:
        first_result = results[0]
        print(f"\n🎯 Trajectory Verification:")
        print(f"   Final missile position: {first_result['final_missile_pos']}")
        print(f"   Final interceptor position: {first_result['final_interceptor_pos']}")
        print(f"   Expected: missile falls from [0,0,300] toward [0,0,0]")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run 17D RL inference with JSON output')
    parser.add_argument('model', help='Path to model checkpoint')
    parser.add_argument('--vecnorm', help='Path to VecNormalize file')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes')
    parser.add_argument('--scenario', default='easy', help='Scenario name')
    parser.add_argument('--output-dir', default='inference_17d_results', help='Output directory')
    
    args = parser.parse_args()
    
    run_17d_inference(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        num_episodes=args.episodes,
        scenario_name=args.scenario,
        output_dir=args.output_dir
    )