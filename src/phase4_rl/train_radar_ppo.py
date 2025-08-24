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
    from .fast_sim_env import FastSimEnv
    from .training_callbacks import (
        EntropyScheduleCallback,
        LearningRateSchedulerCallback,
        BestModelCallback,
        ClipRangeAdaptiveCallback
    )
except ImportError:
    # Fallback for direct execution
    from config import get_config, reset_config
    from scenarios import get_scenario_loader, reset_scenario_loader
    from radar_env import RadarEnv
    from fast_sim_env import FastSimEnv
    from training_callbacks import (
        EntropyScheduleCallback,
        LearningRateSchedulerCallback,
        BestModelCallback,
        ClipRangeAdaptiveCallback
    )


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
        
    def create_environment(self, n_envs: int = 1, training: bool = True) -> gym.Env:
        """
        Create vectorized environment.
        
        Args:
            n_envs: Number of parallel environments
            training: Whether this environment is for training (updates normalization stats)
            
        Returns:
            Vectorized environment
        """
        # Get episode logging configuration
        episode_logging_config = self.config.get('episode_logging', {})
        enable_logging = episode_logging_config.get('enabled', False)
        
        # Determine if we should log for this environment type
        if training:
            # Only log during training if explicitly enabled
            enable_logging = enable_logging and episode_logging_config.get('log_during_training', False)
        else:
            # Log during evaluation if enabled
            enable_logging = enable_logging and episode_logging_config.get('log_during_eval', True)
        
        def make_env():
            def _init():
                env = FastSimEnv(
                    config=self.scenario_config,
                    scenario_name=self.scenario_name,
                    enable_episode_logging=enable_logging,
                    episode_log_dir=episode_logging_config.get('output_dir', 'runs')
                )
                return env
            return _init
        
        if n_envs == 1:
            env = DummyVecEnv([make_env()])
        else:
            env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        
        # Apply normalization to all environments
        gamma = self.training_config.get('gamma', 0.99)
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
            training=training
        )
        
        # Store reference to training environment's VecNormalize wrapper
        if training:
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
        self.env = self.create_environment(n_envs=n_envs, training=True)
        
        # Log observation space dimensions
        print(f"📊 Observation space: {self.env.observation_space.shape[0]} dimensions")
        assert self.env.observation_space.shape[0] == 17, f"Expected 17D observations, got {self.env.observation_space.shape[0]}D"
        
        # Create evaluation environment with same normalization but training=False
        self.eval_env = self.create_environment(n_envs=1, training=False)
        
        # Configure model parameters
        model_params = {
            'learning_rate': self.training_config.get('learning_rate', 3e-4),
            'n_steps': 2048 // n_envs,  # Adjust for number of environments
            'batch_size': self.training_config.get('batch_size', 64),
            'n_epochs': self.training_config.get('n_epochs', 10),
            'gamma': self.training_config.get('gamma', 0.99),
            'gae_lambda': self.training_config.get('gae_lambda', 0.95),
            'clip_range': lambda _: self.training_config.get('clip_range', 0.2),  # Make it callable
            'ent_coef': self.training_config.get('ent_coef', 0.0),
            'vf_coef': self.training_config.get('vf_coef', 0.5),
            'max_grad_norm': self.training_config.get('max_grad_norm', 0.5),
            'tensorboard_log': None,  # Disable to avoid pickle issues
            'verbose': 0  # Set to 0 to avoid TTY pickle issues
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
        """Setup training callbacks for checkpointing, evaluation, and stability."""
        callbacks = []
        
        # Basic checkpoint callback
        checkpoint_interval = self.training_config.get('checkpoint_interval', 10000)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_interval,
            save_path=str(self.checkpoint_dir),
            name_prefix=f"phase4_{self.scenario_name}_checkpoint"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback (disable best model saving to avoid pickle issues)
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=None,  # Disable best model saving to avoid pickle issues
            log_path=str(self.log_dir),
            eval_freq=checkpoint_interval // 2,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=0  # Prevent TTY pickle issues
        )
        callbacks.append(eval_callback)
        
        # Training stability callbacks
        stability_config = self.training_config.get('stability', {})
        
        # Entropy scheduling callback
        if stability_config.get('entropy_schedule', {}).get('enabled', False):
            entropy_config = stability_config['entropy_schedule']
            entropy_callback = EntropyScheduleCallback(
                initial_entropy=entropy_config.get('initial_entropy', 0.01),
                final_entropy=entropy_config.get('final_entropy', 0.02),
                decay_steps=entropy_config.get('decay_steps', 500000),
                verbose=0
            )
            callbacks.append(entropy_callback)
            print(f"✅ Entropy scheduling enabled: {entropy_config['initial_entropy']} → {entropy_config['final_entropy']}")
        
        # Learning rate scheduling callback
        if stability_config.get('lr_schedule', {}).get('enabled', False):
            lr_config = stability_config['lr_schedule']
            lr_callback = LearningRateSchedulerCallback(
                monitor_key=lr_config.get('monitor_key', 'eval/mean_reward'),
                patience=lr_config.get('patience', 5),
                min_delta=lr_config.get('min_delta', 0.01),
                factor=lr_config.get('factor', 0.5),
                min_lr=lr_config.get('min_lr', 1e-6),
                early_stopping=lr_config.get('early_stopping', False),
                early_stopping_patience=lr_config.get('early_stopping_patience', 10),
                verbose=0
            )
            callbacks.append(lr_callback)
            print(f"✅ Learning rate scheduling enabled: patience={lr_config['patience']}, factor={lr_config['factor']}")
        
        # Best model callback (enhanced version)
        if stability_config.get('best_model', {}).get('enabled', False):
            best_config = stability_config['best_model']
            best_callback = BestModelCallback(
                monitor_key=best_config.get('monitor_key', 'eval/mean_reward'),
                save_path=str(self.checkpoint_dir / best_config.get('save_path', 'best_model')),
                name_prefix=best_config.get('name_prefix', 'best'),
                save_freq=checkpoint_interval // 2,
                save_vecnormalize=best_config.get('save_vecnormalize', True),
                verbose=0
            )
            callbacks.append(best_callback)
            print(f"✅ Enhanced best model checkpointing enabled: monitor={best_config.get('monitor_key', 'eval/mean_reward')}")
        
        # Clip range adaptive callback
        if stability_config.get('clip_range_adaptive', {}).get('enabled', False):
            clip_config = stability_config['clip_range_adaptive']
            clip_callback = ClipRangeAdaptiveCallback(
                clip_fraction_threshold=clip_config.get('clip_fraction_threshold', 0.2),
                consecutive_threshold=clip_config.get('consecutive_threshold', 3),
                reduction_factor=clip_config.get('reduction_factor', 0.75),
                min_clip_range=clip_config.get('min_clip_range', 0.05),
                check_freq=checkpoint_interval // 2,
                verbose=0
            )
            callbacks.append(clip_callback)
            print(f"✅ Adaptive clip range enabled: threshold={clip_config['clip_fraction_threshold']}")
        
        print(f"📋 Total callbacks configured: {len(callbacks)}")
        
        return callbacks
    
    def train(self, total_timesteps: Optional[int] = None, resume_checkpoint: Optional[str] = None, seed: Optional[int] = None):
        """
        Train the model.
        
        Args:
            total_timesteps: Total training timesteps
            resume_checkpoint: Path to checkpoint to resume from
            seed: Random seed for reproducible training
        """
        if total_timesteps is None:
            total_timesteps = self.training_config.get('total_timesteps', 1000000)
        
        # Set random seed if provided
        if seed is not None:
            import random
            import torch
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            print(f"🎲 Random seed set to: {seed}")
        
        print(f"Starting Phase 4 RL training:")
        print(f"  Scenario: {self.scenario_name}")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Checkpoint directory: {self.checkpoint_dir}")
        print(f"  Log directory: {self.log_dir}")
        if seed is not None:
            print(f"  Random seed: {seed}")
        print()
        
        # Setup model
        self.setup_model(resume_checkpoint)
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Configure logger (exclude stdout to avoid TTY pickle issues, include tensorboard here)
        tensorboard_dir = str(self.log_dir / "tensorboard")
        new_logger = configure(tensorboard_dir, ["csv", "tensorboard"])
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
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible training'
    )
    
    parser.add_argument(
        '--enable-episode-logging',
        action='store_true',
        help='Enable episode logging for Unity replay visualization'
    )
    
    parser.add_argument(
        '--episode-log-dir',
        type=str,
        default='runs',
        help='Directory for episode logs (if logging enabled)'
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
    
    # Override config if episode logging is requested via CLI
    config_path = args.config
    if args.enable_episode_logging:
        # Need to modify config before creating trainer
        from config import get_config
        config_loader = get_config(config_path)
        if 'episode_logging' not in config_loader._config:
            config_loader._config['episode_logging'] = {}
        config_loader._config['episode_logging']['enabled'] = True
        config_loader._config['episode_logging']['output_dir'] = args.episode_log_dir
        print(f"Episode logging enabled. Logs will be saved to: {args.episode_log_dir}")
        
        # Save modified config to temp file
        import tempfile
        import yaml
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_loader._config, temp_config, default_flow_style=False)
        temp_config.close()
        config_path = temp_config.name
    
    # Create trainer and start training
    trainer = Phase4Trainer(
        scenario_name=args.scenario,
        config_path=config_path,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    try:
        trainer.train(
            total_timesteps=args.timesteps,
            resume_checkpoint=args.resume,
            seed=args.seed
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