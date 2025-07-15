#!/usr/bin/env python3
"""
Checkpoint-based 3D Trajectory Visualization for AegisIntercept Phase 3.

This script loads trained models from checkpoints and creates 3D trajectory
visualizations showing the actual performance of the trained agent.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys
import warnings

# Add project root to path
sys.path.append('.')

from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv
from visualize_3d_trajectories import TrajectoryData, Trajectory3DVisualizer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CheckpointTrajectoryExtractor:
    """
    Extract trajectories from trained model checkpoints.
    
    This class handles loading models and running them to generate
    trajectory data for 3D visualization.
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint trajectory extractor.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_path = self.checkpoint_dir / "model.zip"
        self.vec_normalize_path = self.checkpoint_dir / "vec_normalize.pkl"
        
        # Load checkpoint metadata
        self.training_state = self._load_training_state()
        
        # Initialize environment
        self.env = None
        self.model = None
        
        print(f"Loaded checkpoint: {self.checkpoint_dir.name}")
        if self.training_state:
            print(f"  Timesteps: {self.training_state.get('num_timesteps', 0):,}")
            print(f"  Episodes: {self.training_state.get('callback_states', {}).get('curriculum', {}).get('episode_count', 0)}")
    
    def _load_training_state(self) -> Dict[str, Any]:
        """Load training state from checkpoint."""
        training_state_path = self.checkpoint_dir / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _setup_environment(self) -> bool:
        """Setup environment for trajectory extraction."""
        try:
            # Create environment
            self.env = Aegis6DOFEnv(curriculum_level="easy")
            
            # Try to load VecNormalize if available
            if self.vec_normalize_path.exists():
                try:
                    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
                    
                    # Wrap environment for VecNormalize
                    self.env = DummyVecEnv([lambda: self.env])
                    self.env = VecNormalize.load(self.vec_normalize_path, self.env)
                    self.env.training = False
                    self.env.norm_reward = False
                    
                    print("  ✓ VecNormalize loaded successfully")
                except Exception as e:
                    print(f"  ⚠ VecNormalize failed: {e}")
                    # Use unwrapped environment
                    self.env = Aegis6DOFEnv(curriculum_level="easy")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Environment setup failed: {e}")
            return False
    
    def _load_model(self) -> bool:
        """Load trained model."""
        try:
            from stable_baselines3 import PPO
            
            if not self.model_path.exists():
                print(f"  ❌ Model file not found: {self.model_path}")
                return False
            
            self.model = PPO.load(self.model_path)
            print("  ✓ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"  ❌ Model loading failed: {e}")
            return False
    
    def _get_positions(self, info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get positions from environment, handling wrapper complications."""
        try:
            # Try direct access first
            if hasattr(self.env, 'interceptor'):
                interceptor_pos = self.env.interceptor.get_position()
                adversary_pos = self.env.adversary.get_position()
                target_pos = self.env.target_position
                return interceptor_pos, adversary_pos, target_pos
            
            # Handle wrapped environments
            current_env = self.env
            while hasattr(current_env, 'env') and not hasattr(current_env, 'interceptor'):
                current_env = current_env.env
            
            if hasattr(current_env, 'interceptor'):
                interceptor_pos = current_env.interceptor.get_position()
                adversary_pos = current_env.adversary.get_position()
                target_pos = current_env.target_position
                return interceptor_pos, adversary_pos, target_pos
            
            # Handle VecEnv
            if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                underlying_env = self.env.envs[0]
                while hasattr(underlying_env, 'env') and not hasattr(underlying_env, 'interceptor'):
                    underlying_env = underlying_env.env
                interceptor_pos = underlying_env.interceptor.get_position()
                adversary_pos = underlying_env.adversary.get_position()
                target_pos = underlying_env.target_position
                return interceptor_pos, adversary_pos, target_pos
            
            # Handle VecNormalize
            if hasattr(self.env, 'venv') and hasattr(self.env.venv, 'envs'):
                underlying_env = self.env.venv.envs[0]
                while hasattr(underlying_env, 'env') and not hasattr(underlying_env, 'interceptor'):
                    underlying_env = underlying_env.env
                interceptor_pos = underlying_env.interceptor.get_position()
                adversary_pos = underlying_env.adversary.get_position()
                target_pos = underlying_env.target_position
                return interceptor_pos, adversary_pos, target_pos
            
            # Fallback to default positions
            return np.array([0., 0., 1000.]), np.array([1000., 1000., 1000.]), np.array([0., 0., 0.])
            
        except Exception as e:
            print(f"Warning: Could not get positions: {e}")
            return np.array([0., 0., 1000.]), np.array([1000., 1000., 1000.]), np.array([0., 0., 0.])
    
    def _dynamic_reset(self):
        """Handle different gym API versions for reset."""
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            return reset_result
        else:
            return reset_result, {}
    
    def _dynamic_step(self, action):
        """Handle different gym API versions for step."""
        step_result = self.env.step(action)
        
        def process_info(info):
            if isinstance(info, list):
                return info[0] if len(info) > 0 else {}
            return info
        
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            info = process_info(info)
            return obs, reward, done, done, info
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            info = process_info(info)
            return obs, reward, terminated, truncated, info
        else:
            raise ValueError(f"Unexpected step return format: {len(step_result)} values")
    
    def extract_trajectories(self, num_episodes: int = 5, 
                           max_steps: int = 500,
                           deterministic: bool = True) -> List[TrajectoryData]:
        """
        Extract trajectories from trained model.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            deterministic: Use deterministic policy
            
        Returns:
            List of trajectory data
        """
        if not self._setup_environment():
            return []
        
        if not self._load_model():
            return []
        
        print(f"\\nExtracting {num_episodes} trajectories...")
        
        trajectories = []
        
        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            obs, info = self._dynamic_reset()
            
            # Create trajectory container
            trajectory = TrajectoryData()
            trajectory.episode_id = episode + 1
            trajectory.seed = episode * 1000  # Different seed per episode
            
            # Get initial positions
            interceptor_pos, adversary_pos, target_pos = self._get_positions(info)
            trajectory.target_position = target_pos.copy()
            
            # Run episode
            step = 0
            done = False
            total_reward = 0
            
            while not done and step < max_steps:
                # Get action from trained model
                action, _ = self.model.predict(obs, deterministic=deterministic)
                
                # Step environment
                obs, reward, terminated, truncated, info = self._dynamic_step(action)
                total_reward += reward
                done = terminated or truncated
                
                # Get current positions
                interceptor_pos, adversary_pos, target_pos = self._get_positions(info)
                
                # Log trajectory data
                trajectory.times.append(step * 0.01)  # Assuming 0.01s timestep
                trajectory.interceptor_positions.append(interceptor_pos.copy())
                trajectory.adversary_positions.append(adversary_pos.copy())
                trajectory.rewards.append(reward)
                trajectory.distances.append(info.get('intercept_distance', 0))
                trajectory.fuel_levels.append(info.get('fuel_remaining', 0))
                
                step += 1
            
            # Finalize trajectory
            trajectory.success = info.get('intercept_distance', float('inf')) < 20.0
            trajectory.termination_reason = info.get('termination_reason', 'max_steps')
            
            trajectories.append(trajectory)
            
            status = "SUCCESS" if trajectory.success else "FAILED"
            final_distance = trajectory.distances[-1] if trajectory.distances else 0
            print(f"    Result: {status} | Steps: {step} | Distance: {final_distance:.1f}m | Reward: {total_reward:.2f}")
        
        print(f"\\nExtracted {len(trajectories)} trajectories")
        return trajectories


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Checkpoint-based 3D Trajectory Visualization')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--num-episodes', type=int, default=5,
                       help='Number of episodes to extract (default: 5)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy (default: True)')
    parser.add_argument('--output-dir', type=str, default='checkpoint_3d_output',
                       help='Output directory for visualizations')
    parser.add_argument('--world-scale', type=float, default=3000.0,
                       help='World scale in meters (default: 3000.0)')
    parser.add_argument('--show-individual', action='store_true',
                       help='Show individual trajectory plots')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    # Validate checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("AegisIntercept Phase 3 - Checkpoint 3D Trajectory Visualization")
    print("="*70)
    print(f"Checkpoint: {checkpoint_dir.name}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Deterministic: {args.deterministic}")
    print(f"World Scale: {args.world_scale}m")
    print(f"Output Directory: {output_dir}")
    print("="*70)
    
    # Extract trajectories
    extractor = CheckpointTrajectoryExtractor(args.checkpoint_dir)
    trajectories = extractor.extract_trajectories(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        deterministic=args.deterministic
    )
    
    if not trajectories:
        print("❌ No trajectories extracted")
        return 1
    
    # Create visualizer
    visualizer = Trajectory3DVisualizer(world_scale=args.world_scale)
    
    # Individual trajectory plots
    if args.show_individual:
        print("\\nCreating individual trajectory plots...")
        for i, trajectory in enumerate(trajectories):
            fig = visualizer.plot_single_trajectory(trajectory)
            
            if args.save_plots:
                filename = output_dir / f"checkpoint_trajectory_{trajectory.episode_id:03d}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filename}")
            
            plt.show()
    
    # Combined trajectory plot
    print("\\nCreating combined trajectory plot...")
    title = f"Trained Model Trajectories - {checkpoint_dir.name}"
    fig = visualizer.plot_multiple_trajectories(trajectories, title=title)
    
    if args.save_plots:
        filename = output_dir / f"checkpoint_trajectories_combined.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    plt.show()
    
    # Summary statistics
    success_count = sum(1 for t in trajectories if t.success)
    success_rate = (success_count / len(trajectories)) * 100
    avg_distance = np.mean([t.distances[-1] for t in trajectories if t.distances])
    avg_steps = np.mean([len(t.times) for t in trajectories])
    
    print("\\n" + "="*70)
    print("TRAINED MODEL TRAJECTORY ANALYSIS")
    print("="*70)
    print(f"Checkpoint: {checkpoint_dir.name}")
    print(f"Total Episodes: {len(trajectories)}")
    print(f"Successful Intercepts: {success_count}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Final Distance: {avg_distance:.1f}m")
    print(f"Average Episode Length: {avg_steps:.1f} steps")
    print("="*70)
    
    if args.save_plots:
        print(f"\\nAll plots saved to: {output_dir}/")
    
    return 0


if __name__ == "__main__":
    exit(main())