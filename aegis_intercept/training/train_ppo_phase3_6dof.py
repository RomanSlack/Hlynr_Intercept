#!/usr/bin/env python3
"""
PPO Training Pipeline for AegisIntercept Phase 3 - 6DOF System.

This script provides a complete training pipeline using Proximal Policy Optimization
with curriculum learning, advanced logging, and robust checkpoint management.

Features:
- Robust checkpointing with integrity validation
- Automatic resume from latest checkpoint
- Complete training state persistence (model, environment, callbacks, random states)
- Checkpoint cleanup and retention management
- Backward compatibility with legacy model loading

Usage:
    python train_ppo_phase3_6dof.py [options]

Examples:
    # New training
    python train_ppo_phase3_6dof.py --total-timesteps 1000000 --use-curriculum
    
    # Resume from latest checkpoint
    python train_ppo_phase3_6dof.py --resume-latest --total-timesteps 1000000
    
    # Resume from specific checkpoint
    python train_ppo_phase3_6dof.py --resume-from logs/checkpoints/robust_checkpoint_0050000
    
    # Validate checkpoints before training
    python train_ppo_phase3_6dof.py --validate-checkpoints --resume-latest
    
    # Force resume even with warnings
    python train_ppo_phase3_6dof.py --resume-latest --force-resume

Checkpoint Structure:
    checkpoints/robust_checkpoint_NNNNNNNNNN/
    ├── model.zip                    # PPO model
    ├── vec_normalize.pkl            # Environment normalization stats
    ├── training_state.pkl           # Complete training state (binary)
    ├── training_state.json          # Human-readable training state
    └── checksum.txt                 # Integrity checksum
"""

import argparse
import os
import sys
import time
import json
import pickle
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv
from aegis_intercept.curriculum import CurriculumManager
from aegis_intercept.logging import TrajectoryLogger, ExportManager

# Version for checkpoint compatibility
CHECKPOINT_VERSION = "1.0.0"


class RobustCheckpointCallback(BaseCallback):
    """
    Enhanced checkpoint callback with comprehensive training state management.
    
    This callback saves complete training state including model, environment
    normalization, callback states, and training progress to enable robust
    resuming from any checkpoint.
    """
    
    def __init__(self,
                 save_freq: int,
                 save_path: str,
                 name_prefix: str = 'robust_checkpoint',
                 max_checkpoints: int = 5,
                 verbose: int = 1):
        """
        Initialize robust checkpoint callback.
        
        Args:
            save_freq: Frequency for saving checkpoints (in timesteps)
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint filenames
            max_checkpoints: Maximum number of checkpoints to retain
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.max_checkpoints = max_checkpoints
        
        # Create checkpoint directory
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Track last save
        self.last_save_timestep = 0
        
        # References to other callbacks (set externally)
        self.curriculum_callback = None
        self.trajectory_callback = None
    
    def _on_step(self) -> bool:
        """Called at each environment step."""
        if self.num_timesteps - self.last_save_timestep >= self.save_freq:
            self._save_comprehensive_checkpoint()
            self.last_save_timestep = self.num_timesteps
        return True
    
    def _save_comprehensive_checkpoint(self):
        """Save complete training state."""
        try:
            checkpoint_name = f"{self.name_prefix}_{self.num_timesteps:010d}"
            checkpoint_dir = self.save_path / checkpoint_name
            temp_dir = self.save_path / f"{checkpoint_name}_temp"
            
            # Create temporary directory for atomic operations
            temp_dir.mkdir(exist_ok=True)
            
            # Save model
            model_path = temp_dir / "model.zip"
            self.model.save(str(model_path))
            
            # Save VecNormalize statistics
            if hasattr(self.training_env, 'save'):
                vecnorm_path = temp_dir / "vec_normalize.pkl"
                self.training_env.save(str(vecnorm_path))
            
            # Collect comprehensive training state
            training_state = self._collect_training_state()
            
            # Save training state
            state_path = temp_dir / "training_state.pkl"
            with open(state_path, 'wb') as f:
                pickle.dump(training_state, f)
            
            # Save training state as JSON (human readable)
            json_state = self._create_json_state(training_state)
            json_path = temp_dir / "training_state.json"
            with open(json_path, 'w') as f:
                json.dump(json_state, f, indent=2)
            
            # Create checksum for integrity verification
            checksum = self._calculate_checkpoint_checksum(temp_dir)
            checksum_path = temp_dir / "checksum.txt"
            with open(checksum_path, 'w') as f:
                f.write(checksum)
            
            # Atomic move: rename temp directory to final name
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            temp_dir.rename(checkpoint_dir)
            
            if self.verbose > 0:
                print(f"✓ Checkpoint saved: {checkpoint_name} (timestep {self.num_timesteps:,})")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            print(f"ERROR: Failed to save checkpoint: {e}")
            # Cleanup temp directory if it exists
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _collect_training_state(self) -> Dict[str, Any]:
        """Collect comprehensive training state."""
        state = {
            'checkpoint_version': CHECKPOINT_VERSION,
            'creation_timestamp': time.time(),
            'num_timesteps': self.num_timesteps,
            'training_start_time': getattr(self, 'training_start_time', time.time()),
            
            # Model and environment info
            'model_class': self.model.__class__.__name__,
            'env_info': {
                'n_envs': getattr(self.training_env, 'num_envs', 1),
                'observation_space': str(self.training_env.observation_space),
                'action_space': str(self.training_env.action_space)
            },
            
            # Random states for reproducibility
            'random_states': {
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'python': None  # Python random state if needed
            },
            
            # Training progress
            'training_progress': {
                'episodes_completed': getattr(self, 'episodes_completed', 0),
                'wall_time_elapsed': time.time() - getattr(self, 'training_start_time', time.time()),
                'timesteps_per_second': self.num_timesteps / max(1, time.time() - getattr(self, 'training_start_time', time.time()))
            }
        }
        
        # Collect callback states
        callback_states = {}
        
        if self.curriculum_callback:
            callback_states['curriculum'] = {
                'episode_count': self.curriculum_callback.episode_count,
                'episode_rewards': list(self.curriculum_callback.episode_rewards),
                'episode_successes': list(self.curriculum_callback.episode_successes),
                'curriculum_manager_state': self.curriculum_callback.curriculum_manager.get_statistics()
            }
        
        if self.trajectory_callback:
            callback_states['trajectory'] = {
                'episode_count': self.trajectory_callback.episode_count,
                'export_manager_stats': self.trajectory_callback.export_manager.get_export_statistics() if self.trajectory_callback.export_manager else None
            }
        
        state['callback_states'] = callback_states
        
        return state
    
    def _create_json_state(self, training_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create JSON-serializable version of training state."""
        json_state = {}
        
        for key, value in training_state.items():
            if key == 'random_states':
                # Skip non-serializable random states in JSON
                json_state[key] = {'note': 'Random states saved in pickle format only'}
            elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                json_state[key] = value
            else:
                json_state[key] = str(value)
        
        return json_state
    
    def _calculate_checkpoint_checksum(self, checkpoint_dir: Path) -> str:
        """Calculate MD5 checksum of checkpoint files."""
        hasher = hashlib.md5()
        
        # Get all files in sorted order for consistent checksum
        files = sorted(checkpoint_dir.glob('*'))
        
        for file_path in files:
            if file_path.is_file() and file_path.name != 'checksum.txt':
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        try:
            # Find all checkpoint directories
            checkpoint_dirs = [d for d in self.save_path.iterdir() 
                             if d.is_dir() and d.name.startswith(self.name_prefix)]
            
            # Sort by creation time (newest first)
            checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old checkpoints
            for old_checkpoint in checkpoint_dirs[self.max_checkpoints:]:
                if self.verbose > 1:
                    print(f"Removing old checkpoint: {old_checkpoint.name}")
                shutil.rmtree(old_checkpoint)
                
        except Exception as e:
            print(f"Warning: Failed to cleanup old checkpoints: {e}")
    
    def set_callback_references(self, curriculum_callback=None, trajectory_callback=None):
        """Set references to other callbacks for state collection."""
        self.curriculum_callback = curriculum_callback
        self.trajectory_callback = trajectory_callback
    
    def set_training_start_time(self, start_time: float):
        """Set training start time for progress calculations."""
        self.training_start_time = start_time


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning integration."""
    
    def __init__(self, 
                 curriculum_manager: CurriculumManager,
                 evaluation_interval: int = 1000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.evaluation_interval = evaluation_interval
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_successes = []
    
    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Check for episode completion
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                self.episode_count += 1
                
                # Get episode info
                info = self.locals.get('infos', [{}])[i]
                episode_reward = info.get('episode', {}).get('r', 0.0)
                episode_success = info.get('episode_metrics', {}).get('success', False)
                
                self.episode_rewards.append(episode_reward)
                self.episode_successes.append(episode_success)
                
                # Record in curriculum manager
                self.curriculum_manager.record_episode(
                    success=episode_success,
                    total_reward=episode_reward,
                    info=info
                )
                
                # Update environment difficulty if curriculum changed
                current_level = self.curriculum_manager.get_current_level()
                if hasattr(self.training_env, 'set_attr'):
                    self.training_env.set_attr('curriculum_level', current_level)
                
                # Print progress periodically
                if self.episode_count % 100 == 0:
                    recent_success_rate = np.mean(self.episode_successes[-100:])
                    recent_avg_reward = np.mean(self.episode_rewards[-100:])
                    
                    if self.verbose > 0:
                        print(f"Episode {self.episode_count}: "
                              f"Success Rate: {recent_success_rate:.2%}, "
                              f"Avg Reward: {recent_avg_reward:.2f}, "
                              f"Level: {current_level}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        self.curriculum_manager.print_progress_report()


class TrajectoryLoggingCallback(BaseCallback):
    """Callback for trajectory logging and export."""
    
    def __init__(self,
                 export_manager: ExportManager,
                 log_frequency: int = 100,
                 export_frequency: int = 1000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.export_manager = export_manager
        self.log_frequency = log_frequency
        self.export_frequency = export_frequency
        self.episode_count = 0
        self.trajectory_logger = TrajectoryLogger(
            enable_detailed_logging=False,  # Performance mode for training
            enable_performance_mode=True
        )
    
    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Log trajectory data periodically
        if self.num_timesteps % self.log_frequency == 0:
            # This would require environment integration for detailed logging
            pass
        
        # Export data periodically
        if self.num_timesteps % self.export_frequency == 0:
            self._export_training_data()
        
        return True
    
    def _export_training_data(self):
        """Export current training data."""
        try:
            timestamp = int(time.time())
            export_files = self.export_manager.export_all_episodes(
                formats=['json'],
                filename_prefix=f'training_checkpoint_{timestamp}'
            )
            
            if self.verbose > 0:
                print(f"Training data exported: {len(export_files)} files")
                
        except Exception as e:
            print(f"Warning: Failed to export training data: {e}")


# Checkpoint discovery and validation utilities

def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in the given directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        Path to latest checkpoint directory or None if not found
    """
    try:
        if not checkpoint_dir.exists():
            return None
        
        # Find all checkpoint directories
        checkpoint_dirs = [d for d in checkpoint_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('robust_checkpoint_')]
        
        if not checkpoint_dirs:
            return None
        
        # Sort by timestep (extracted from directory name)
        def extract_timestep(checkpoint_path):
            try:
                return int(checkpoint_path.name.split('_')[-1])
            except (ValueError, IndexError):
                return 0
        
        checkpoint_dirs.sort(key=extract_timestep, reverse=True)
        
        # Validate the latest checkpoint
        latest_checkpoint = checkpoint_dirs[0]
        if validate_checkpoint_integrity(latest_checkpoint):
            return latest_checkpoint
        
        # If latest is corrupt, try previous ones
        for checkpoint_dir in checkpoint_dirs[1:]:
            if validate_checkpoint_integrity(checkpoint_dir):
                print(f"Warning: Latest checkpoint corrupt, using: {checkpoint_dir.name}")
                return checkpoint_dir
        
        return None
        
    except Exception as e:
        print(f"Error finding latest checkpoint: {e}")
        return None


def validate_checkpoint_integrity(checkpoint_path: Path) -> bool:
    """
    Validate checkpoint integrity using checksums and required files.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        True if checkpoint is valid, False otherwise
    """
    try:
        if not checkpoint_path.is_dir():
            return False
        
        # Check required files
        required_files = ['model.zip', 'training_state.pkl', 'checksum.txt']
        for required_file in required_files:
            if not (checkpoint_path / required_file).exists():
                print(f"Missing required file: {required_file}")
                return False
        
        # Verify checksum
        checksum_file = checkpoint_path / 'checksum.txt'
        with open(checksum_file, 'r') as f:
            expected_checksum = f.read().strip()
        
        # Calculate current checksum
        actual_checksum = calculate_checkpoint_checksum(checkpoint_path)
        
        if actual_checksum != expected_checksum:
            print(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating checkpoint: {e}")
        return False


def calculate_checkpoint_checksum(checkpoint_dir: Path) -> str:
    """Calculate MD5 checksum of checkpoint files."""
    hasher = hashlib.md5()
    
    # Get all files in sorted order for consistent checksum
    files = sorted(checkpoint_dir.glob('*'))
    
    for file_path in files:
        if file_path.is_file() and file_path.name != 'checksum.txt':
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
    
    return hasher.hexdigest()


def load_training_state(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load comprehensive training state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Dictionary containing training state
    """
    try:
        state_file = checkpoint_path / 'training_state.pkl'
        with open(state_file, 'rb') as f:
            training_state = pickle.load(f)
        
        # Validate version compatibility
        checkpoint_version = training_state.get('checkpoint_version', '0.0.0')
        if checkpoint_version != CHECKPOINT_VERSION:
            print(f"Warning: Checkpoint version mismatch. "
                  f"Checkpoint: {checkpoint_version}, Current: {CHECKPOINT_VERSION}")
        
        return training_state
        
    except Exception as e:
        print(f"Error loading training state: {e}")
        raise


def validate_checkpoint_compatibility(training_state: Dict[str, Any], args: argparse.Namespace) -> bool:
    """
    Validate that checkpoint is compatible with current training configuration.
    
    Args:
        training_state: Loaded training state
        args: Current training arguments
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        # Check model class
        model_class = training_state.get('model_class', 'Unknown')
        if model_class != 'PPO':
            print(f"Model class mismatch: {model_class} != PPO")
            return False
        
        # Check environment compatibility
        env_info = training_state.get('env_info', {})
        expected_n_envs = getattr(args, 'n_envs', 1)
        saved_n_envs = env_info.get('n_envs', 1)
        
        if saved_n_envs != expected_n_envs:
            print(f"Warning: Environment count mismatch: {saved_n_envs} != {expected_n_envs}")
            # This is not a fatal error, but worth noting
        
        return True
        
    except Exception as e:
        print(f"Error validating compatibility: {e}")
        return False


def resume_training_from_checkpoint(checkpoint_path: Path, 
                                  env: VecNormalize, 
                                  args: argparse.Namespace) -> Tuple[PPO, VecNormalize, int, Dict[str, Any]]:
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        env: Current training environment
        args: Training arguments
        
    Returns:
        Tuple of (model, environment, completed_timesteps, training_state)
    """
    try:
        print(f"Resuming from checkpoint: {checkpoint_path.name}")
        
        # Load training state
        training_state = load_training_state(checkpoint_path)
        
        # Validate compatibility
        if not validate_checkpoint_compatibility(training_state, args):
            raise ValueError("Checkpoint incompatible with current configuration")
        
        # Load model
        model_path = checkpoint_path / 'model.zip'
        model = PPO.load(str(model_path), env=env)
        
        # Load VecNormalize statistics
        vecnorm_path = checkpoint_path / 'vec_normalize.pkl'
        if vecnorm_path.exists():
            env = VecNormalize.load(str(vecnorm_path), env)
            print("✓ Environment normalization statistics restored")
        
        # Restore random states
        random_states = training_state.get('random_states', {})
        if 'numpy' in random_states and random_states['numpy']:
            np.random.set_state(random_states['numpy'])
            print("✓ NumPy random state restored")
        
        if 'torch' in random_states and random_states['torch'] is not None:
            torch.set_rng_state(random_states['torch'])
            print("✓ PyTorch random state restored")
        
        # Get completed timesteps
        completed_timesteps = training_state.get('num_timesteps', 0)
        
        print(f"✓ Resumed from timestep {completed_timesteps:,}")
        print(f"✓ Wall time elapsed: {training_state.get('training_progress', {}).get('wall_time_elapsed', 0):.1f}s")
        
        return model, env, completed_timesteps, training_state
        
    except Exception as e:
        print(f"Failed to resume from checkpoint: {e}")
        raise


def create_training_environment(args: argparse.Namespace, rank: int = 0) -> gym.Env:
    """Create a single training environment."""
    def _init():
        # Create base environment
        env = Aegis6DOFEnv(
            render_mode=None,  # No rendering during training
            max_episode_steps=args.max_episode_steps,
            time_step=args.time_step,
            world_scale=args.world_scale,
            curriculum_level="easy"  # Will be updated by curriculum manager
        )
        
        # Wrap with Monitor for episode statistics
        log_dir = Path(args.log_dir) / "monitor"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(log_dir / f"monitor_{rank}.log"))
        
        return env
    
    return _init


def create_vectorized_env(args: argparse.Namespace) -> VecNormalize:
    """Create vectorized training environment."""
    # Create multiple environments for parallel training
    env_fns = [create_training_environment(args, i) for i in range(args.n_envs)]
    
    if args.n_envs == 1:
        # Use single environment
        vec_env = make_vec_env(env_fns[0], n_envs=1)
    else:
        # Use multiprocessing for parallel environments
        vec_env = SubprocVecEnv(env_fns)
    
    # Wrap with normalization (enhanced stability)
    vec_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=5.0,        # Reduced clipping for stability
        clip_reward=5.0,      # Reduced reward clipping
        gamma=args.gamma,
        epsilon=1e-8         # Stable epsilon for normalization
    )
    
    return vec_env


def create_ppo_model(env: VecNormalize, args: argparse.Namespace) -> PPO:
    """Create PPO model with specified hyperparameters."""
    
    # PPO hyperparameters
    model_params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_range': args.clip_range,
        'ent_coef': args.ent_coef,
        'vf_coef': args.vf_coef,
        'max_grad_norm': args.max_grad_norm,
        'verbose': 1,
        'device': 'auto',
        'tensorboard_log': str(Path(args.log_dir) / "tensorboard")
    }
    
    # Policy network architecture
    policy_kwargs = {
        'net_arch': [dict(pi=[256, 256], vf=[256, 256])],
        'activation_fn': torch.nn.ReLU
    }
    model_params['policy_kwargs'] = policy_kwargs
    
    return PPO(**model_params)


def setup_callbacks(args: argparse.Namespace, 
                   curriculum_manager: CurriculumManager,
                   export_manager: ExportManager,
                   resume_from_checkpoint: bool = False) -> Tuple[list, Optional[RobustCheckpointCallback]]:
    """Setup training callbacks."""
    callbacks = []
    
    # Robust checkpoint callback
    checkpoint_callback = RobustCheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(Path(args.log_dir) / "checkpoints"),
        name_prefix='robust_checkpoint',
        max_checkpoints=args.checkpoint_retention,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Curriculum callback
    curriculum_callback = None
    if args.use_curriculum:
        curriculum_callback = CurriculumCallback(
            curriculum_manager=curriculum_manager,
            evaluation_interval=1000,
            verbose=1
        )
        callbacks.append(curriculum_callback)
    
    # Trajectory logging callback
    trajectory_callback = None
    if args.enable_logging:
        trajectory_callback = TrajectoryLoggingCallback(
            export_manager=export_manager,
            log_frequency=args.log_frequency,
            export_frequency=args.export_frequency,
            verbose=1
        )
        callbacks.append(trajectory_callback)
    
    # Set callback references for state collection
    checkpoint_callback.set_callback_references(
        curriculum_callback=curriculum_callback,
        trajectory_callback=trajectory_callback
    )
    
    return callbacks, checkpoint_callback


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PPO on AegisIntercept Phase 3')
    
    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--max-episode-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    
    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Number of steps per environment per update')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for PPO updates')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Number of PPO epochs per update')
    parser.add_argument('--gamma', type=float, default=0.995,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda parameter')
    parser.add_argument('--clip-range', type=float, default=0.2,
                       help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                       help='Value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='Maximum gradient norm')
    
    # Environment parameters
    parser.add_argument('--time-step', type=float, default=0.01,
                       help='Physics simulation time step')
    parser.add_argument('--world-scale', type=float, default=5000.0,
                       help='World scale in meters')
    
    # Curriculum learning
    parser.add_argument('--use-curriculum', action='store_true',
                       help='Enable curriculum learning')
    parser.add_argument('--curriculum-config', type=str, default=None,
                       help='Path to curriculum configuration file')
    
    # Logging and checkpoints
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for logs and checkpoints')
    parser.add_argument('--checkpoint-freq', type=int, default=10000,
                       help='Checkpoint save frequency')
    parser.add_argument('--enable-logging', action='store_true',
                       help='Enable trajectory logging')
    parser.add_argument('--log-frequency', type=int, default=1000,
                       help='Trajectory logging frequency')
    parser.add_argument('--export-frequency', type=int, default=50000,
                       help='Data export frequency')
    
    # Model loading/saving
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to load existing model')
    parser.add_argument('--save-model', type=str, default='trained_model',
                       help='Path to save final model')
    
    # Robust checkpointing and resuming
    parser.add_argument('--resume-latest', action='store_true',
                       help='Resume training from the latest checkpoint')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume training from specific checkpoint directory')
    parser.add_argument('--checkpoint-retention', type=int, default=5,
                       help='Number of checkpoints to retain (default: 5)')
    parser.add_argument('--validate-checkpoints', action='store_true',
                       help='Validate checkpoint integrity on startup')
    parser.add_argument('--force-resume', action='store_true',
                       help='Force resume even with compatibility warnings')
    
    # Execution options
    parser.add_argument('--headless', action='store_true',
                       help='Run without visualization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("AegisIntercept Phase 3 - 6DOF PPO Training")
    print("="*60)
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Curriculum learning: {'Enabled' if args.use_curriculum else 'Disabled'}")
    print(f"Resume mode: {'Latest' if args.resume_latest else 'From checkpoint' if args.resume_from else 'New training'}")
    print(f"Log directory: {log_dir}")
    print("="*60)
    
    # Handle checkpoint validation if requested
    if args.validate_checkpoints:
        checkpoint_dir = log_dir / "checkpoints"
        if checkpoint_dir.exists():
            print("Validating existing checkpoints...")
            checkpoint_dirs = [d for d in checkpoint_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('robust_checkpoint_')]
            
            valid_count = 0
            for checkpoint_path in checkpoint_dirs:
                if validate_checkpoint_integrity(checkpoint_path):
                    valid_count += 1
                else:
                    print(f"❌ Invalid checkpoint: {checkpoint_path.name}")
            
            print(f"✓ {valid_count}/{len(checkpoint_dirs)} checkpoints are valid")
        else:
            print("No checkpoints directory found.")
    
    # Determine if resuming from checkpoint
    resume_checkpoint_path = None
    if args.resume_latest:
        checkpoint_dir = log_dir / "checkpoints"
        resume_checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if not resume_checkpoint_path:
            print("No valid checkpoint found for --resume-latest. Starting new training.")
        else:
            print(f"Found latest checkpoint: {resume_checkpoint_path.name}")
    
    elif args.resume_from:
        resume_checkpoint_path = Path(args.resume_from)
        if not resume_checkpoint_path.exists():
            print(f"Checkpoint path does not exist: {args.resume_from}")
            return 1
        
        if not validate_checkpoint_integrity(resume_checkpoint_path):
            if not args.force_resume:
                print("Checkpoint validation failed. Use --force-resume to override.")
                return 1
            else:
                print("Warning: Resuming from potentially corrupted checkpoint due to --force-resume")
    
    # Check for conflicting arguments
    if (args.resume_latest or args.resume_from) and args.load_model:
        print("Warning: --load-model ignored when resuming from checkpoint")
        args.load_model = None
    
    # Initialize curriculum manager
    curriculum_manager = None
    if args.use_curriculum:
        curriculum_manager = CurriculumManager(
            config_path=args.curriculum_config,
            promotion_threshold=0.85,
            evaluation_window=100,
            min_episodes=50
        )
        print(f"Curriculum initialized at level: {curriculum_manager.get_current_level()}")
    
    # Initialize export manager
    export_manager = None
    if args.enable_logging:
        export_manager = ExportManager(
            output_directory=str(log_dir / "exports"),
            auto_compress=True,
            max_episodes_per_file=100
        )
        print("Trajectory logging enabled")
    
    # Create training environment
    print("Creating training environment...")
    env = create_vectorized_env(args)
    print(f"Environment created with {args.n_envs} parallel instances")
    
    # Resume from checkpoint or create/load model
    completed_timesteps = 0
    training_state = None
    
    if resume_checkpoint_path:
        # Resume from checkpoint
        try:
            model, env, completed_timesteps, training_state = resume_training_from_checkpoint(
                resume_checkpoint_path, env, args
            )
        except Exception as e:
            print(f"Failed to resume from checkpoint: {e}")
            if not args.force_resume:
                return 1
            else:
                print("Continuing with new training due to resume failure...")
                model = create_ppo_model(env, args)
    
    elif args.load_model and Path(args.load_model).exists():
        # Load existing model (legacy support)
        print(f"Loading existing model from: {args.load_model}")
        model = PPO.load(args.load_model, env=env)
        
        # Load normalization stats if available
        norm_path = Path(args.load_model).parent / "vec_normalize.pkl"
        if norm_path.exists():
            env = VecNormalize.load(str(norm_path), env)
            print("Loaded normalization statistics")
    
    else:
        # Create new model
        print("Creating new PPO model...")
        model = create_ppo_model(env, args)
    
    # Setup TensorBoard logging
    tb_log_dir = log_dir / "tensorboard"
    tb_log_dir.mkdir(exist_ok=True)
    model.set_logger(configure(str(tb_log_dir), ["tensorboard"]))
    
    # Setup callbacks
    callbacks, checkpoint_callback = setup_callbacks(args, curriculum_manager, export_manager, 
                                                     resume_from_checkpoint=resume_checkpoint_path is not None)
    print(f"Training callbacks: {len(callbacks)} configured")
    
    # Restore callback states if resuming
    if training_state and 'callback_states' in training_state:
        print("Restoring callback states...")
        callback_states = training_state['callback_states']
        
        # Restore curriculum callback state
        if 'curriculum' in callback_states and curriculum_manager:
            curriculum_state = callback_states['curriculum']
            # Find curriculum callback in callbacks list
            for callback in callbacks:
                if isinstance(callback, CurriculumCallback):
                    callback.episode_count = curriculum_state.get('episode_count', 0)
                    callback.episode_rewards = list(curriculum_state.get('episode_rewards', []))
                    callback.episode_successes = list(curriculum_state.get('episode_successes', []))
                    
                    # Restore curriculum manager state
                    curriculum_manager_state = curriculum_state.get('curriculum_manager_state', {})
                    if 'current_level' in curriculum_manager_state:
                        curriculum_manager.set_level(curriculum_manager_state['current_level'])
                    
                    print(f"✓ Curriculum callback state restored (episodes: {callback.episode_count})")
                    break
        
        # Restore trajectory callback state  
        if 'trajectory' in callback_states and export_manager:
            trajectory_state = callback_states['trajectory']
            for callback in callbacks:
                if isinstance(callback, TrajectoryLoggingCallback):
                    callback.episode_count = trajectory_state.get('episode_count', 0)
                    print(f"✓ Trajectory callback state restored (episodes: {callback.episode_count})")
                    break
    
    # Set training start time for checkpoint callback
    training_start_time = time.time()
    if training_state:
        # Adjust for resumed training
        previous_start_time = training_state.get('training_start_time', training_start_time)
        elapsed_time = training_state.get('training_progress', {}).get('wall_time_elapsed', 0)
        training_start_time = training_start_time - elapsed_time
    
    checkpoint_callback.set_training_start_time(training_start_time)
    
    # Save training configuration
    config_data = {
        'args': vars(args),
        'model_params': {
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_range': args.clip_range
        },
        'environment_params': {
            'max_episode_steps': args.max_episode_steps,
            'time_step': args.time_step,
            'world_scale': args.world_scale
        },
        'training_start_time': time.time()
    }
    
    with open(log_dir / "training_config.json", 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Calculate remaining timesteps
    remaining_timesteps = args.total_timesteps - completed_timesteps
    
    if remaining_timesteps <= 0:
        print(f"\nTraining already completed! ({completed_timesteps:,}/{args.total_timesteps:,} timesteps)")
        return 0
    
    # Start training
    print(f"\nStarting training from timestep {completed_timesteps:,}")
    print(f"Remaining timesteps: {remaining_timesteps:,}")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            tb_log_name="ppo_6dof_training",
            reset_num_timesteps=False  # Don't reset timestep counter
        )
        
        training_time = time.time() - start_time
        total_timesteps_trained = completed_timesteps + remaining_timesteps
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Total timesteps trained: {total_timesteps_trained:,}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        return 1
    
    # Save final model
    save_path = Path(args.save_model)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add resume info to save path if we resumed
    if resume_checkpoint_path:
        save_path = save_path.parent / f"{save_path.stem}_resumed{save_path.suffix}"
    
    model.save(str(save_path))
    env.save(str(save_path.parent / "vec_normalize.pkl"))
    
    print(f"Model saved to: {save_path}")
    print(f"Normalization stats saved to: {save_path.parent / 'vec_normalize.pkl'}")
    
    # Save final training summary
    if resume_checkpoint_path or completed_timesteps > 0:
        final_summary = {
            'resumed_from_checkpoint': str(resume_checkpoint_path) if resume_checkpoint_path else None,
            'initial_timesteps': completed_timesteps,
            'total_timesteps_trained': args.total_timesteps,
            'final_training_time': time.time() - training_start_time,
            'resume_success': True
        }
        
        summary_path = log_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"Training summary saved to: {summary_path}")
    
    # Final statistics
    if curriculum_manager:
        curriculum_manager.print_progress_report()
        curriculum_manager.export_data(str(log_dir / "curriculum_final.json"))
    
    if export_manager:
        export_manager.create_analysis_report(str(log_dir / "training_analysis.json"))
        print(f"Training analysis saved")
    
    # Close environment
    env.close()
    
    print("\nTraining pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)