"""
PPO training script with comprehensive logging and stability features.
"""

import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from environment import InterceptEnvironment
from logger import UnifiedLogger, EpisodeEvent


class CustomTrainingCallback(BaseCallback):
    """Custom callback for enhanced logging and adaptive training."""
    
    def __init__(self, logger: UnifiedLogger, config: Dict[str, Any]):
        super().__init__()
        self.logger = logger
        self.config = config
        self.tb_writer = logger.get_tensorboard_writer()
        
        # Entropy scheduling
        self.entropy_schedule = config.get('entropy_schedule', {})
        self.initial_entropy = self.entropy_schedule.get('initial', 0.02)
        self.final_entropy = self.entropy_schedule.get('final', 0.01)
        self.entropy_decay_steps = self.entropy_schedule.get('decay_steps', 500000)
        
        # Learning rate scheduling
        self.lr_schedule = config.get('lr_schedule', {})
        self.lr_patience = self.lr_schedule.get('patience', 5)
        self.lr_factor = self.lr_schedule.get('factor', 0.5)
        self.min_lr = self.lr_schedule.get('min_lr', 1e-6)
        self.best_reward = -float('inf')
        self.patience_counter = 0
        
        # Clip range adaptation
        self.clip_adapt = config.get('clip_adapt', {})
        self.clip_threshold = self.clip_adapt.get('threshold', 0.2)
        self.clip_reduction = self.clip_adapt.get('reduction', 0.75)
        self.min_clip = self.clip_adapt.get('min_clip', 0.05)
        self.clip_violations = 0
        
    def _on_training_start(self):
        """Initialize training."""
        self.logger.logger.info("Training started")
        
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Update entropy coefficient
        if self.entropy_schedule.get('enabled', True):
            progress = min(self.num_timesteps / self.entropy_decay_steps, 1.0)
            entropy_coef = self.initial_entropy * (1 - progress) + self.final_entropy * progress
            self.model.ent_coef = entropy_coef
        
        return True
    
    def _on_rollout_end(self):
        """Called at the end of a rollout."""
        # Log training metrics
        if self.tb_writer:
            # Get rollout info from the model's logger
            if hasattr(self.model, '_logger') and self.model._logger:
                rollout_info = self.model._logger.name_to_value
                
                # Log key metrics
                for key in ['policy_gradient_loss', 'value_loss', 'entropy_loss', 'approx_kl', 'clip_fraction']:
                    if key in rollout_info:
                        self.tb_writer.add_scalar(f'train/{key}', rollout_info[key], self.num_timesteps)
                
                # Adaptive clip range based on KL divergence
                if 'clip_fraction' in rollout_info:
                    clip_fraction = rollout_info['clip_fraction']
                    
                    if self.clip_adapt.get('enabled', True) and clip_fraction > self.clip_threshold:
                        self.clip_violations += 1
                        if self.clip_violations >= 3:
                            current_clip = self.model.clip_range
                            if callable(current_clip):
                                current_clip = current_clip(1.0)
                            new_clip = max(current_clip * self.clip_reduction, self.min_clip)
                            self.model.clip_range = new_clip
                            self.logger.logger.info(f"Reduced clip range to {new_clip:.3f}")
                            self.clip_violations = 0
                    else:
                        self.clip_violations = 0
            
            # Log current learning rate and entropy
            self.tb_writer.add_scalar('train/learning_rate', self.model.learning_rate, self.num_timesteps)
            self.tb_writer.add_scalar('train/entropy_coef', self.model.ent_coef, self.num_timesteps)
        
        # Log to unified logger
        metrics = {
            'timesteps': self.num_timesteps,
            'entropy_coef': self.model.ent_coef,
            'learning_rate': self.model.learning_rate
        }
        
        if hasattr(self.model, 'logger') and self.model.logger:
            if hasattr(self.model.logger, 'name_to_value'):
                metrics.update(self.model.logger.name_to_value)
        
        self.logger.log_training_step(self.num_timesteps, metrics)


class EpisodeLoggingCallback(BaseCallback):
    """Callback for detailed episode logging."""
    
    def __init__(self, logger: UnifiedLogger, log_interval: int = 100):
        super().__init__()
        self.logger = logger
        self.log_interval = log_interval
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """Log episode data."""
        # Check for episode end
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            # Get episode info
            info = self.locals.get('infos', [{}])[0]
            episode_reward = info.get('episode', {}).get('r', 0)
            episode_length = info.get('episode', {}).get('l', 0)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Log every N episodes
            if self.episode_count % self.log_interval == 0:
                self.logger.log_metrics({
                    'episode': self.episode_count,
                    'mean_reward': np.mean(self.episode_rewards[-100:]),
                    'mean_length': np.mean(self.episode_lengths[-100:]),
                    'total_timesteps': self.num_timesteps
                })
        
        return True


def create_env(config: Dict[str, Any], rank: int = 0) -> InterceptEnvironment:
    """Create and wrap environment."""
    def _init():
        env = InterceptEnvironment(config)
        env = Monitor(env)
        return env
    return _init


def train(config_path: str):
    """Main training function."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create logger
    logger = UnifiedLogger(
        log_dir=config.get('logging', {}).get('log_dir', 'logs'),
        run_name="training"
    )
    
    logger.logger.info(f"Loaded configuration from {config_path}")
    
    # Create environments
    n_envs = config['training'].get('n_envs', 8)
    env_config = config.get('environment', {})
    
    envs = DummyVecEnv([create_env(env_config, i) for i in range(n_envs)])
    
    # Add VecNormalize wrapper
    envs = VecNormalize(
        envs,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create model
    policy_kwargs = dict(
        net_arch=config['training'].get('net_arch', [256, 256]),
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=config['training'].get('learning_rate', 3e-4),
        n_steps=config['training'].get('n_steps', 2048),
        batch_size=config['training'].get('batch_size', 64),
        n_epochs=config['training'].get('n_epochs', 10),
        gamma=config['training'].get('gamma', 0.99),
        gae_lambda=config['training'].get('gae_lambda', 0.95),
        clip_range=config['training'].get('clip_range', 0.2),
        ent_coef=config['training'].get('ent_coef', 0.01),
        vf_coef=config['training'].get('vf_coef', 0.5),
        max_grad_norm=config['training'].get('max_grad_norm', 0.5),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(logger.log_dir / "tensorboard"),
        verbose=1
    )
    
    logger.logger.info(f"Created PPO model with {n_envs} parallel environments")
    
    # Create callbacks
    callbacks = []
    
    # Custom training callback
    callbacks.append(CustomTrainingCallback(logger, config['training']))
    
    # Episode logging callback
    callbacks.append(EpisodeLoggingCallback(logger))
    
    # Checkpoint callback
    checkpoint_dir = Path(config.get('checkpointing', {}).get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)
    
    callbacks.append(CheckpointCallback(
        save_freq=config['training'].get('checkpoint_freq', 10000),
        save_path=str(checkpoint_dir),
        name_prefix="model",
        save_vecnormalize=True
    ))
    
    # Evaluation callback
    eval_env = DummyVecEnv([create_env(env_config)])
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
    
    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(logger.log_dir / "eval"),
        eval_freq=config['training'].get('eval_freq', 10000),
        n_eval_episodes=config['training'].get('n_eval_episodes', 10),
        deterministic=True
    ))
    
    callback_list = CallbackList(callbacks)
    
    # Train model
    logger.logger.info(f"Starting training for {config['training']['total_timesteps']} timesteps")
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callback_list,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.logger.info("Training interrupted by user")
    except Exception as e:
        logger.logger.error(f"Training failed: {e}")
        raise
    
    # Save final model
    final_path = checkpoint_dir / "final"
    final_path.mkdir(exist_ok=True)
    model.save(str(final_path / "model"))
    envs.save(str(final_path / "vec_normalize.pkl"))
    
    logger.logger.info(f"Training completed. Final model saved to {final_path}")
    
    # Create manifest
    logger.create_manifest()
    
    return model, envs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for missile interception")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    train(args.config)