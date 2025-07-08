#!/usr/bin/env python3
"""
Visualization script for trained AegisIntercept models.

This script loads a trained model and runs it with real-time 3D visualization.
"""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv


def dynamic_reset(env):
    """Handle both old and new Gym API reset returns."""
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
        return obs, info
    else:
        return reset_result, {}


def dynamic_step(env, action):
    """Handle both old (4-value) and new (5-value) Gym API step returns."""
    step_result = env.step(action)
    
    # Handle vectorized env info (list of dicts) vs single env info (dict)
    def process_info(info):
        if isinstance(info, list):
            # Vectorized env returns list of info dicts, take the first one
            return info[0] if len(info) > 0 else {}
        return info
    
    if len(step_result) == 4:
        # Old API: obs, reward, done, info
        obs, reward, done, info = step_result
        info = process_info(info)
        return obs, reward, done, done, info  # terminated = truncated = done
    elif len(step_result) == 5:
        # New API: obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = step_result
        info = process_info(info)
        return obs, reward, terminated, truncated, info
    else:
        raise ValueError(f"Unexpected step return format: {len(step_result)} values")


def visualize_episode(model_path: str, vec_normalize_path: str = None, num_episodes: int = 1):
    """Visualize episodes with trained model."""
    
    print(f"Loading model from: {model_path}")
    if num_episodes > 1:
        print(f"Will run {num_episodes} episodes with different random seeds")
    
    # Create environment
    env = Aegis6DOFEnv()
    
    # Load VecNormalize if available
    if vec_normalize_path and Path(vec_normalize_path).exists():
        print(f"Loading VecNormalize from: {vec_normalize_path}")
        # Wrap single environment in DummyVecEnv for VecNormalize compatibility
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Run multiple episodes
    episode_results = []
    
    for episode in range(num_episodes):
        print(f"\nRunning episode {episode + 1}/{num_episodes} with visualization...")
        
        # Set different random seed for each episode
        episode_seed = episode * 12345 + 67890  # Different seed each time
        np.random.seed(episode_seed)
        print(f"Using random seed: {episode_seed}")
        
        # Reset environment for this episode
        obs, info = dynamic_reset(env)
        
        # Debug: Print initial conditions
        try:
            if hasattr(env, 'envs') and len(env.envs) > 0:
                underlying_env = env.envs[0]
                interceptor_pos = underlying_env.interceptor.get_position()
                adversary_pos = underlying_env.adversary.get_position()
                target_pos = underlying_env.target_position
            elif hasattr(env, 'venv') and hasattr(env.venv, 'envs'):
                underlying_env = env.venv.envs[0]
                interceptor_pos = underlying_env.interceptor.get_position()
                adversary_pos = underlying_env.adversary.get_position()  
                target_pos = underlying_env.target_position
            else:
                interceptor_pos = env.interceptor.get_position()
                adversary_pos = env.adversary.get_position()
                target_pos = env.target_position
            
            print(f"Initial - Interceptor: {interceptor_pos}, Adversary: {adversary_pos}, Target: {target_pos}")
        except:
            print("Could not access initial positions for debug")
        
        # Setup visualization for this episode
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Storage for trajectory
        interceptor_trajectory = []
        adversary_trajectory = []
        
        step = 0
        done = False
        total_reward = 0
        
        while not done and step < 1000:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = dynamic_step(env, action)
            total_reward += reward
            done = terminated or truncated
        
            # Store positions - handle VecNormalize wrapped environments
            try:
                # Try direct access first (unwrapped environment)
                interceptor_pos = env.interceptor.get_position()
                adversary_pos = env.adversary.get_position()
            except AttributeError:
                # Environment is wrapped (VecNormalize + DummyVecEnv), access underlying env
                if hasattr(env, 'envs') and len(env.envs) > 0:
                    # Access through DummyVecEnv
                    underlying_env = env.envs[0]
                    interceptor_pos = underlying_env.interceptor.get_position()
                    adversary_pos = underlying_env.adversary.get_position()
                elif hasattr(env, 'venv') and hasattr(env.venv, 'envs'):
                    # Access through VecNormalize -> DummyVecEnv
                    underlying_env = env.venv.envs[0]
                    interceptor_pos = underlying_env.interceptor.get_position()
                    adversary_pos = underlying_env.adversary.get_position()
                else:
                    # Fallback: use info dict or default positions if available
                    interceptor_pos = info.get('interceptor_position', np.array([0., 0., 0.]))
                    adversary_pos = info.get('adversary_position', np.array([100., 100., 100.]))
                    print("Warning: Could not access interceptor/adversary positions directly")
            
            interceptor_trajectory.append(interceptor_pos.copy())
            adversary_trajectory.append(adversary_pos.copy())
            
            step += 1
        
        # Update visualization every 10 steps
        if step % 10 == 0:
            ax.clear()
            
            # Plot trajectories
            if len(interceptor_trajectory) > 1:
                traj = np.array(interceptor_trajectory)
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', label='Interceptor', linewidth=2)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='blue', s=100, marker='o')
            
            if len(adversary_trajectory) > 1:
                traj = np.array(adversary_trajectory)
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', label='Adversary', linewidth=2)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, marker='^')
            
            # Plot target - handle VecNormalize wrapped environments
            try:
                # Try direct access first (unwrapped environment)
                target_pos = env.target_position
            except AttributeError:
                # Environment is wrapped, access underlying env
                if hasattr(env, 'envs') and len(env.envs) > 0:
                    # Access through DummyVecEnv
                    target_pos = env.envs[0].target_position
                elif hasattr(env, 'venv') and hasattr(env.venv, 'envs'):
                    # Access through VecNormalize -> DummyVecEnv
                    target_pos = env.venv.envs[0].target_position
                else:
                    # Fallback: default target position
                    target_pos = info.get('target_position', np.array([1000., 1000., 100.]))
            
            ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='green', s=200, marker='*', label='Target')
            
            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            # Convert NumPy arrays/scalars to Python floats for formatting
            raw_dist = info.get("intercept_distance", 0)
            if hasattr(raw_dist, 'item'):
                dist = raw_dist.item()
            elif isinstance(raw_dist, np.ndarray):
                dist = float(raw_dist.flat[0])
            else:
                dist = float(raw_dist)
            
            raw_reward = total_reward
            if hasattr(raw_reward, 'item'):
                reward = raw_reward.item()
            elif isinstance(raw_reward, np.ndarray):
                reward = float(raw_reward.flat[0])
            else:
                reward = float(raw_reward)
            
            ax.set_title(f'Step {step}: Distance {dist:.1f}m, Reward {reward:.2f}')
            ax.legend()
            
            # Set equal aspect ratio
            max_range = 2000
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([0, 2000])
            
            plt.draw()
            plt.pause(0.05)
    
    # Final results
    success = info.get('intercept_distance', float('inf')) < 20.0
    status = "SUCCESS" if success else "FAILED"
    
    # Convert NumPy arrays/scalars to Python floats for formatting
    raw_reward = total_reward
    if hasattr(raw_reward, 'item'):
        reward = raw_reward.item()
    elif isinstance(raw_reward, np.ndarray):
        reward = float(raw_reward.flat[0])
    else:
        reward = float(raw_reward)
    
    raw_dist = info.get('intercept_distance', 0)
    if hasattr(raw_dist, 'item'):
        dist = raw_dist.item()
    elif isinstance(raw_dist, np.ndarray):
        dist = float(raw_dist.flat[0])
    else:
        dist = float(raw_dist)
    
    raw_fuel = info.get('fuel_remaining', 0)
    if hasattr(raw_fuel, 'item'):
        fuel = raw_fuel.item()
    elif isinstance(raw_fuel, np.ndarray):
        fuel = float(raw_fuel.flat[0])
    else:
        fuel = float(raw_fuel)
    
    print(f"\nEpisode complete!")
    print(f"Status: {status}")
    print(f"Total Reward: {reward:.2f}")
    print(f"Steps: {step}")
    print(f"Final Distance: {dist:.1f}m")
    print(f"Fuel Remaining: {fuel:.2f}")
    
    # Keep plot open
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize trained AegisIntercept model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.zip file)')
    parser.add_argument('--vec-normalize', type=str, help='Path to VecNormalize stats (.pkl file)')
    parser.add_argument('--checkpoint-dir', type=str, help='Load from checkpoint directory')
    
    args = parser.parse_args()
    
    if args.checkpoint_dir:
        # Auto-find files in checkpoint directory
        checkpoint_path = Path(args.checkpoint_dir)
        model_path = checkpoint_path / "model.zip"
        vec_normalize_path = checkpoint_path / "vec_normalize.pkl"
        
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            return 1
        
        visualize_episode(str(model_path), str(vec_normalize_path) if vec_normalize_path.exists() else None)
    else:
        visualize_episode(args.model, args.vec_normalize)
    
    return 0


if __name__ == "__main__":
    exit(main())