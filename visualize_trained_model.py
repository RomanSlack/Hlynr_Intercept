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
import gymnasium as gym


class BackwardCompatibilityWrapper(gym.ObservationWrapper):
    """Wrapper to make new environment compatible with old models."""
    
    def __init__(self, env, target_obs_size=32):
        super().__init__(env)
        self.target_obs_size = target_obs_size
        # Update observation space to match target size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(target_obs_size,), dtype=np.float32
        )
    
    def observation(self, obs):
        """Truncate observation to match old model expectations."""
        if len(obs) > self.target_obs_size:
            # Take the first target_obs_size elements (original format)
            return obs[:self.target_obs_size].astype(np.float32)
        return obs.astype(np.float32)


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


def get_positions(env, info):
    """Get interceptor, adversary, and target positions handling VecNormalize and wrappers."""
    try:
        # Try direct access first (unwrapped environment)
        interceptor_pos = env.interceptor.get_position()
        adversary_pos = env.adversary.get_position()
        target_pos = env.target_position
        return interceptor_pos, adversary_pos, target_pos
    except AttributeError:
        # Environment is wrapped, need to unwrap to get to the base environment
        current_env = env
        
        # Unwrap through multiple layers
        while hasattr(current_env, 'env') and not hasattr(current_env, 'interceptor'):
            current_env = current_env.env
        
        # Try the unwrapped environment
        if hasattr(current_env, 'interceptor'):
            interceptor_pos = current_env.interceptor.get_position()
            adversary_pos = current_env.adversary.get_position()
            target_pos = current_env.target_position
            return interceptor_pos, adversary_pos, target_pos
        
        # Environment is wrapped (VecNormalize + DummyVecEnv), access underlying env
        if hasattr(env, 'envs') and len(env.envs) > 0:
            # Access through DummyVecEnv
            underlying_env = env.envs[0]
            # Unwrap the underlying env too if needed
            while hasattr(underlying_env, 'env') and not hasattr(underlying_env, 'interceptor'):
                underlying_env = underlying_env.env
            interceptor_pos = underlying_env.interceptor.get_position()
            adversary_pos = underlying_env.adversary.get_position()
            target_pos = underlying_env.target_position
            return interceptor_pos, adversary_pos, target_pos
        elif hasattr(env, 'venv') and hasattr(env.venv, 'envs'):
            # Access through VecNormalize -> DummyVecEnv
            underlying_env = env.venv.envs[0]
            # Unwrap the underlying env too if needed
            while hasattr(underlying_env, 'env') and not hasattr(underlying_env, 'interceptor'):
                underlying_env = underlying_env.env
            interceptor_pos = underlying_env.interceptor.get_position()
            adversary_pos = underlying_env.adversary.get_position()
            target_pos = underlying_env.target_position
            return interceptor_pos, adversary_pos, target_pos
        else:
            # Fallback: use info dict or default positions if available
            interceptor_pos = info.get('interceptor_position', np.array([0., 0., 0.]))
            adversary_pos = info.get('adversary_position', np.array([100., 100., 100.]))
            target_pos = info.get('target_position', np.array([1000., 1000., 100.]))
            print("Warning: Could not access positions directly, using fallback")
            return interceptor_pos, adversary_pos, target_pos


def safe_float(value):
    """Convert NumPy arrays/scalars to Python floats safely."""
    if hasattr(value, 'item'):
        return value.item()
    elif isinstance(value, np.ndarray):
        return float(value.flat[0])
    else:
        return float(value)


def visualize_episodes(model_path: str, vec_normalize_path: str = None, num_episodes: int = 1):
    """Visualize episodes with trained model."""
    
    print(f"Loading model from: {model_path}")
    if num_episodes > 1:
        print(f"Will run {num_episodes} episodes with different random seeds")
    
    # Create environment
    env = Aegis6DOFEnv()
    
    # Check if this is an old checkpoint and apply compatibility wrapper if needed
    try:
        test_model = PPO.load(model_path)
        model_obs_space = test_model.observation_space.shape[0]
        env_obs_space = env.observation_space.shape[0]
        
        if model_obs_space != env_obs_space:
            print(f"⚠️  Model expects {model_obs_space}D observations, environment provides {env_obs_space}D")
            print(f"⚠️  Applying backward compatibility wrapper for old checkpoint")
            env = BackwardCompatibilityWrapper(env, target_obs_size=model_obs_space)
            print(f"✓ Environment observation space adjusted to {model_obs_space}D")
            
    except Exception as e:
        print(f"Warning: Could not check model compatibility: {e}")
        # Continue anyway
    
    # Load VecNormalize if available
    if vec_normalize_path and Path(vec_normalize_path).exists():
        print(f"Loading VecNormalize from: {vec_normalize_path}")
        # Wrap single environment in DummyVecEnv for VecNormalize compatibility
        env = DummyVecEnv([lambda: env])
        
        try:
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
            print("✓ VecNormalize loaded successfully")
        except AssertionError as e:
            if "spaces must have the same shape" in str(e):
                print(f"⚠️  VecNormalize shape mismatch: {e}")
                print("⚠️  Skipping VecNormalize (old checkpoint incompatible with updated environment)")
                print("⚠️  This is expected when testing updated code with old checkpoints")
                # Continue without VecNormalize - the model will still work
            else:
                raise
    
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
            interceptor_pos, adversary_pos, target_pos = get_positions(env, info)
            print(f"Initial - Interceptor: {interceptor_pos}")
            print(f"Initial - Adversary: {adversary_pos}")
            print(f"Initial - Target: {target_pos}")
        except Exception as e:
            print(f"Could not access initial positions for debug: {e}")
        
        # Setup visualization for this episode
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Storage for trajectory
        interceptor_trajectory = []
        adversary_trajectory = []
        
        step = 0
        done = False
        total_reward = 0
        
        # Store initial positions for visualization even if episode ends quickly
        try:
            interceptor_pos, adversary_pos, target_pos = get_positions(env, info)
            interceptor_trajectory.append(interceptor_pos.copy())
            adversary_trajectory.append(adversary_pos.copy())
        except Exception as e:
            print(f"Warning: Could not get initial positions: {e}")
        
        while not done and step < 1000:
            # Get action from trained model (use different determinism each episode)
            deterministic = episode % 2 == 0  # Alternate between deterministic and stochastic
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = dynamic_step(env, action)
            total_reward += reward
            done = terminated or truncated
            
            # Store positions
            try:
                interceptor_pos, adversary_pos, target_pos = get_positions(env, info)
                interceptor_trajectory.append(interceptor_pos.copy())
                adversary_trajectory.append(adversary_pos.copy())
            except Exception as e:
                print(f"Warning: Could not get positions at step {step}: {e}")
                # Use fallback positions
                interceptor_trajectory.append(np.array([0., 0., 0.]))
                adversary_trajectory.append(np.array([100., 100., 100.]))
            
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
                
                # Plot target
                try:
                    _, _, target_pos = get_positions(env, info)
                    ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='green', s=200, marker='*', label='Target')
                except:
                    # Fallback target position
                    ax.scatter(1000, 1000, 100, c='green', s=200, marker='*', label='Target')
                
                # Set labels and title
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                
                # Convert NumPy arrays/scalars to Python floats for formatting
                dist = safe_float(info.get("intercept_distance", 0))
                reward_val = safe_float(total_reward)
                
                ax.set_title(f'Episode {episode+1}, Step {step}: Distance {dist:.1f}m, Reward {reward_val:.2f}')
                ax.legend()
                
                # Set equal aspect ratio
                max_range = 2000
                ax.set_xlim([-max_range, max_range])
                ax.set_ylim([-max_range, max_range])
                ax.set_zlim([0, 2000])
                
                plt.draw()
                plt.pause(0.05)
        
        # Final plot for episodes that ended quickly
        if step <= 5:  # Episode ended very quickly, show final state
            ax.clear()
            
            # Plot trajectories (even if just initial positions)
            if len(interceptor_trajectory) >= 1:
                traj = np.array(interceptor_trajectory)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='blue', s=200, marker='o', label='Interceptor (Final)')
                if len(traj) > 1:
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=2)
            
            if len(adversary_trajectory) >= 1:
                traj = np.array(adversary_trajectory)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=200, marker='^', label='Adversary (Final)')
                if len(traj) > 1:
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', linewidth=2)
            
            # Plot target
            try:
                _, _, target_pos = get_positions(env, info)
                ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='green', s=300, marker='*', label='Target')
            except:
                ax.scatter(0, 0, 0, c='green', s=300, marker='*', label='Target')
            
            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            dist = safe_float(info.get("intercept_distance", 0))
            reward_val = safe_float(total_reward)
            termination_reason = info.get('termination_reason', 'unknown')
            
            ax.set_title(f'Episode {episode+1} (Quick End): {termination_reason}\nDistance: {dist:.1f}m, Reward: {reward_val:.2f}, Steps: {step}')
            ax.legend()
            
            # Set reasonable bounds for visualization
            max_range = 3000
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([0, 2000])
            
            plt.draw()
            plt.pause(1.0)  # Show for a moment
        
        # Final results for this episode
        success = info.get('intercept_distance', float('inf')) < 20.0
        status = "SUCCESS" if success else "FAILED"
        
        # Convert values safely for display
        reward_val = safe_float(total_reward)
        dist = safe_float(info.get('intercept_distance', 0))
        fuel = safe_float(info.get('fuel_remaining', 0))
        termination_reason = info.get('termination_reason', 'unknown')
        
        episode_result = {
            'episode': episode + 1,
            'status': status,
            'reward': reward_val,
            'steps': step,
            'distance': dist,
            'fuel': fuel,
            'seed': episode_seed,
            'termination_reason': termination_reason
        }
        episode_results.append(episode_result)
        
        print(f"\nEpisode {episode + 1} complete!")
        print(f"Status: {status}")
        print(f"Total Reward: {reward_val:.2f}")
        print(f"Steps: {step}")
        print(f"Final Distance: {dist:.1f}m")
        print(f"Fuel Remaining: {fuel:.2f}")
        print(f"Termination Reason: {termination_reason}")
        print(f"Random Seed: {episode_seed}")
        
        # Keep plot open briefly
        if episode < num_episodes - 1:
            plt.pause(2.0)  # Pause between episodes
            plt.close()
        else:
            plt.show()  # Keep final plot open
    
    # Summary of all episodes
    if num_episodes > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY OF {num_episodes} EPISODES")
        print(f"{'='*60}")
        for result in episode_results:
            print(f"Episode {result['episode']}: {result['status']} | "
                  f"Reward: {result['reward']:6.2f} | Steps: {result['steps']:3d} | "
                  f"Distance: {result['distance']:6.1f}m | Reason: {result['termination_reason']} | Seed: {result['seed']}")
        
        # Calculate statistics
        success_rate = sum(1 for r in episode_results if r['status'] == 'SUCCESS') / len(episode_results) * 100
        avg_reward = np.mean([r['reward'] for r in episode_results])
        avg_distance = np.mean([r['distance'] for r in episode_results])
        avg_steps = np.mean([r['steps'] for r in episode_results])
        
        # Termination reason analysis
        termination_counts = {}
        for result in episode_results:
            reason = result['termination_reason']
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
        
        print(f"\nStatistics:")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Final Distance: {avg_distance:.1f}m")
        
        print(f"\nTermination Reasons:")
        for reason, count in sorted(termination_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(episode_results) * 100
            print(f"  {reason}: {count}/{len(episode_results)} ({percentage:.1f}%)")
        
        print(f"{'='*60}")
    
    return episode_results


def main():
    parser = argparse.ArgumentParser(description='Visualize trained AegisIntercept model')
    parser.add_argument('--model', type=str, help='Path to trained model (.zip file)')
    parser.add_argument('--vec-normalize', type=str, help='Path to VecNormalize stats (.pkl file)')
    parser.add_argument('--checkpoint-dir', type=str, help='Load from checkpoint directory')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to visualize (default: 1)')
    
    args = parser.parse_args()
    
    if args.checkpoint_dir:
        # Auto-find files in checkpoint directory
        checkpoint_path = Path(args.checkpoint_dir)
        model_path = checkpoint_path / "model.zip"
        vec_normalize_path = checkpoint_path / "vec_normalize.pkl"
        
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            return 1
        
        visualize_episodes(str(model_path), str(vec_normalize_path) if vec_normalize_path.exists() else None, args.episodes)
    else:
        if not args.model:
            parser.error("Either --checkpoint-dir or --model must be provided")
        visualize_episodes(args.model, args.vec_normalize, args.episodes)
    
    return 0


if __name__ == "__main__":
    exit(main())