"""
Enhanced PPO training script for Phase 4 RL with multi-entity scenario support.

This script provides comprehensive training capabilities with scenario management,
checkpointing, and integration with the existing training infrastructure.
"""

import argparse
import os
import sys
from pathlib import Path
import time
import json
from typing import Dict, Any, Optional

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

try:
    from .config import get_config, reset_config
    from .scenarios import get_scenario_loader, reset_scenario_loader
    from .radar_env import RadarEnv
except ImportError:
    # Fallback for direct execution
    from config import get_config, reset_config
    from scenarios import get_scenario_loader, reset_scenario_loader
    from radar_env import RadarEnv


class Phase4Trainer:
    """Enhanced trainer for Phase 4 RL with scenario and configuration management."""
    
    def __init__(self, 
                 scenario_name: str = "easy",
                 config_path: Optional[str] = None,
                 checkpoint_dir: str = "checkpoints",
                 log_dir: str = "logs"):
        """
        Initialize trainer.
        
        Args:
            scenario_name: Name of scenario to train on
            config_path: Path to configuration file
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for logging
        """
        # Reset global instances to use new paths
        reset_config()
        reset_scenario_loader()
        
        # Load configuration
        self.config_loader = get_config(config_path)
        self.config = self.config_loader._config
        
        # Load scenario
        self.scenario_loader = get_scenario_loader()
        self.scenario_name = scenario_name
        self.scenario_config = self.scenario_loader.create_environment_config(
            scenario_name, self.config
        )
        
        # Training configuration
        self.training_config = self.config_loader.get_training_config()
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment and model
        self.env = None
        self.eval_env = None
        self.model = None
        self.vec_normalize = None
        
    def create_environment(self, n_envs: int = 1, eval_env: bool = False) -> gym.Env:
        """
        Create vectorized environment.
        
        Args:
            n_envs: Number of parallel environments
            eval_env: Whether this is for evaluation
            
        Returns:
            Vectorized environment
        """
        def make_env():
            def _init():
                env = RadarEnv(
                    config=self.scenario_config,
                    scenario_name=self.scenario_name,
                    render_mode=None
                )
                return env
            return _init
        
        if n_envs == 1:
            env = DummyVecEnv([make_env()])
        else:
            env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        
        # Apply normalization for training environments
        if not eval_env:
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=self.training_config.get('gamma', 0.99)
            )
            self.vec_normalize = env
        
        return env
    
    def setup_model(self, resume_checkpoint: Optional[str] = None) -> PPO:
        """
        Setup PPO model with configuration.
        
        Args:
            resume_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Configured PPO model
        """
        # Create training environment
        n_envs = self.training_config.get('n_envs', 8)
        self.env = self.create_environment(n_envs=n_envs)
        
        # Create evaluation environment
        self.eval_env = self.create_environment(n_envs=1, eval_env=True)
        
        # Configure model parameters
        model_params = {
            'learning_rate': self.training_config.get('learning_rate', 3e-4),
            'n_steps': 2048 // n_envs,  # Adjust for number of environments
            'batch_size': self.training_config.get('batch_size', 64),
            'n_epochs': self.training_config.get('n_epochs', 10),
            'gamma': self.training_config.get('gamma', 0.99),
            'gae_lambda': self.training_config.get('gae_lambda', 0.95),
            'clip_range': self.training_config.get('clip_range', 0.2),
            'ent_coef': self.training_config.get('ent_coef', 0.0),
            'vf_coef': self.training_config.get('vf_coef', 0.5),
            'max_grad_norm': self.training_config.get('max_grad_norm', 0.5),
            'tensorboard_log': str(self.log_dir / "tensorboard"),
            'verbose': 1
        }
        
        if resume_checkpoint:
            # Load existing model
            print(f"Resuming training from checkpoint: {resume_checkpoint}")
            self.model = PPO.load(
                resume_checkpoint,
                env=self.env,
                **model_params
            )
            
            # Load normalization parameters if they exist
            vec_normalize_path = Path(resume_checkpoint).parent / "vec_normalize.pkl"
            if vec_normalize_path.exists() and self.vec_normalize:
                self.vec_normalize = VecNormalize.load(vec_normalize_path, self.env)
                self.env = self.vec_normalize
        else:
            # Create new model
            self.model = PPO('MlpPolicy', self.env, **model_params)
        
        return self.model
    
    def setup_callbacks(self):
        """Setup training callbacks for checkpointing and evaluation."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_interval = self.training_config.get('checkpoint_interval', 10000)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_interval,
            save_path=str(self.checkpoint_dir),
            name_prefix=f"phase4_{self.scenario_name}_checkpoint"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.checkpoint_dir / "best_model"),
            log_path=str(self.log_dir),
            eval_freq=checkpoint_interval // 2,
            deterministic=True,
            render=False,
            n_eval_episodes=10
        )
        callbacks.append(eval_callback)
        
        return callbacks
    
    def train(self, total_timesteps: Optional[int] = None, resume_checkpoint: Optional[str] = None):
        """
        Train the model.
        
        Args:
            total_timesteps: Total training timesteps
            resume_checkpoint: Path to checkpoint to resume from
        """
        if total_timesteps is None:
            total_timesteps = self.training_config.get('total_timesteps', 1000000)
        
        print(f"Starting Phase 4 RL training:")
        print(f"  Scenario: {self.scenario_name}")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Checkpoint directory: {self.checkpoint_dir}")
        print(f"  Log directory: {self.log_dir}")
        print()
        
        # Setup model
        self.setup_model(resume_checkpoint)
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Configure logger
        new_logger = configure(str(self.log_dir), ["stdout", "csv", "tensorboard"])
        self.model.set_logger(new_logger)
        
        # Save training configuration
        self._save_training_config()
        
        # Start training
        start_time = time.time()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        training_time = time.time() - start_time
        
        # Save final model
        final_model_path = self.checkpoint_dir / f"phase4_{self.scenario_name}_final"
        self.model.save(final_model_path)
        
        # Save normalization parameters
        if self.vec_normalize:
            self.vec_normalize.save(self.checkpoint_dir / "vec_normalize_final.pkl")
        
        print(f"\nTraining completed!")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Final model saved: {final_model_path}")
        print(f"  Checkpoints saved: {self.checkpoint_dir}")
    
    def _save_training_config(self):
        """Save training configuration for reproducibility."""
        config_data = {
            'scenario_name': self.scenario_name,
            'scenario_config': self.scenario_config,
            'training_config': self.training_config,
            'timestamp': time.time(),
            'git_commit': self._get_git_commit(),
            'command_line': ' '.join(sys.argv)
        }
        
        config_path = self.log_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        print(f"Training configuration saved: {config_path}")
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description='Phase 4 RL Training with Scenario Support')
    
    parser.add_argument(
        '--scenario', 
        type=str, 
        default='easy',
        choices=['easy', 'medium', 'hard', 'impossible'],
        help='Scenario to train on'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Total training timesteps (overrides config)'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory for saving checkpoints'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str, 
        default='logs',
        help='Directory for logging'
    )
    
    parser.add_argument(
        '--list-scenarios',
        action='store_true',
        help='List available scenarios and exit'
    )
    
    args = parser.parse_args()
    
    # List scenarios if requested
    if args.list_scenarios:
        scenario_loader = get_scenario_loader()
        scenarios = scenario_loader.list_scenarios()
        print("Available scenarios:")
        for scenario in scenarios:
            try:
                config = scenario_loader.load_scenario(scenario)
                print(f"  {scenario}: {config.get('description', 'No description')}")
            except Exception as e:
                print(f"  {scenario}: Error loading scenario ({e})")
        return
    
    # Validate scenario
    scenario_loader = get_scenario_loader()
    available_scenarios = scenario_loader.list_scenarios()
    if args.scenario not in available_scenarios:
        print(f"Error: Scenario '{args.scenario}' not found.")
        print(f"Available scenarios: {', '.join(available_scenarios)}")
        sys.exit(1)
    
    # Create trainer and start training
    trainer = Phase4Trainer(
        scenario_name=args.scenario,
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    try:
        trainer.train(
            total_timesteps=args.timesteps,
            resume_checkpoint=args.resume
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        if trainer.model:
            interrupt_path = Path(args.checkpoint_dir) / f"phase4_{args.scenario}_interrupted"
            trainer.model.save(interrupt_path)
            print(f"Model saved: {interrupt_path}")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()