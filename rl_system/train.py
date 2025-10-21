"""
PPO training script with comprehensive logging and stability features.
"""

import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Fix CUDA initialization issues by ensuring clean environment
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# Uncomment to force specific GPU: os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from environment import InterceptEnvironment
from logger import UnifiedLogger, EpisodeEvent


def get_device() -> str:
    """
    Intelligently detect and select compute device with graceful fallback.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        try:
            # Clear any stale CUDA state
            torch.cuda.empty_cache()

            # Test CUDA availability with a small operation
            test_tensor = torch.zeros(1, device='cuda')
            _ = test_tensor + 1  # Simple operation to verify GPU works
            del test_tensor
            torch.cuda.empty_cache()

            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ Using CUDA device: {gpu_name} ({gpu_memory:.1f}GB)")
            return 'cuda'
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                print(f"⚠️  GPU out of memory, falling back to CPU")
            elif "busy" in error_msg.lower():
                print(f"⚠️  GPU busy with another process, falling back to CPU")
            else:
                print(f"⚠️  CUDA error: {error_msg}")
                print(f"⚠️  Falling back to CPU (training will be slower but functional)")
            return 'cpu'
    elif torch.backends.mps.is_available():
        print(f"✓ Using MPS (Apple Silicon GPU) acceleration")
        return 'mps'
    else:
        print(f"ℹ️  No GPU available, using CPU")
        return 'cpu'


class CustomTrainingCallback(BaseCallback):
    """Custom callback for enhanced logging and adaptive training."""
    
    def __init__(self, logger: UnifiedLogger, config: Dict[str, Any]):
        super().__init__()
        self.unified_logger = logger
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
        self.unified_logger.logger.info("Training started")
        
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Update entropy coefficient
        if self.entropy_schedule.get('enabled', True):
            progress = min(self.num_timesteps / self.entropy_decay_steps, 1.0)
            entropy_coef = self.initial_entropy * (1 - progress) + self.final_entropy * progress
            self.model.ent_coef = entropy_coef

        # Update curriculum learning progress in all environments
        # This allows intercept radius to gradually decrease during training
        if hasattr(self.training_env, 'env_method'):
            try:
                self.training_env.env_method('set_training_step_count', self.num_timesteps)
            except AttributeError:
                pass  # Environment doesn't support curriculum learning

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
                            self.unified_logger.logger.info(f"Reduced clip range to {new_clip:.3f}")
                            self.clip_violations = 0
                    else:
                        self.clip_violations = 0

                # Log interception success rate from episode buffer if available
                if 'ep_success_buffer' in rollout_info:
                    success_rate = rollout_info['ep_success_buffer'] * 100
                    self.tb_writer.add_scalar('rollout/success_rate_pct', success_rate, self.num_timesteps)

            # Log current learning rate and entropy
            self.tb_writer.add_scalar('train/learning_rate', self.model.learning_rate, self.num_timesteps)
            self.tb_writer.add_scalar('train/entropy_coef', self.model.ent_coef, self.num_timesteps)

            # Log curriculum learning progress (intercept radius + radar parameters)
            if hasattr(self.training_env, 'get_attr'):
                try:
                    # Get current intercept radius from first environment
                    radius = self.training_env.get_attr('get_current_intercept_radius')[0]()
                    self.tb_writer.add_scalar('curriculum/intercept_radius_m', radius, self.num_timesteps)

                    # Get radar curriculum parameters from first environment
                    obs_gen = self.training_env.get_attr('observation_generator')[0]
                    self.tb_writer.add_scalar('curriculum/radar_beam_width_deg', obs_gen.radar_beam_width, self.num_timesteps)
                    self.tb_writer.add_scalar('curriculum/onboard_detection_reliability', obs_gen.onboard_detection_reliability, self.num_timesteps)
                    self.tb_writer.add_scalar('curriculum/ground_detection_reliability', obs_gen.ground_detection_reliability, self.num_timesteps)
                    self.tb_writer.add_scalar('curriculum/measurement_noise_level', obs_gen.measurement_noise_level, self.num_timesteps)
                except (AttributeError, IndexError):
                    pass  # Environment doesn't support curriculum or method not available

        # Log to unified logger
        metrics = {
            'timesteps': self.num_timesteps,
            'entropy_coef': self.model.ent_coef,
            'learning_rate': self.model.learning_rate
        }

        if hasattr(self.model, 'logger') and self.model.logger:
            if hasattr(self.model.logger, 'name_to_value'):
                metrics.update(self.model.logger.name_to_value)

        self.unified_logger.log_training_step(self.num_timesteps, metrics)


class EpisodeLoggingCallback(BaseCallback):
    """Callback for detailed episode logging with interception tracking."""

    def __init__(self, logger: UnifiedLogger, log_interval: int = 100):
        super().__init__()
        self.unified_logger = logger
        self.tb_writer = None  # Will be set in _on_training_start from model's logger
        self.log_interval = log_interval
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.interceptions = []  # Track successful interceptions

    def _on_training_start(self) -> None:
        """Initialize TensorBoard writer from model's logger."""
        # Use the model's TensorBoard writer (created by PPO)
        if self.model.logger and hasattr(self.model.logger, 'output_formats'):
            for output_format in self.model.logger.output_formats:
                if hasattr(output_format, 'writer'):  # TensorBoardOutputFormat
                    self.tb_writer = output_format.writer
                    self.unified_logger.logger.info("EpisodeCallback connected to model's TensorBoard writer")
                    break

        if self.tb_writer is None:
            self.unified_logger.logger.warning("Could not connect to TensorBoard writer!")

    def _on_step(self) -> bool:
        """Log episode data including interception success."""
        # VecEnv returns lists - check ALL environments for episode ends
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])

        # DEBUG: Log first time we see dones
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            self.unified_logger.logger.debug(f"EpisodeCallback first step - dones type: {type(dones)}, len: {len(dones) if hasattr(dones, '__len__') else 'N/A'}")

        # Iterate through all environments
        for i, (done, info) in enumerate(zip(dones, infos)):
            # Only process when episode actually ends
            if done:
                # Monitor wrapper adds 'episode' dict with cumulative stats on episode end
                if 'episode' in info:
                    self.episode_count += 1

                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']

                    # Track interception success (our custom info from environment)
                    intercepted = info.get('intercepted', False)

                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.interceptions.append(1.0 if intercepted else 0.0)

                    # DEBUG: Log first few episodes
                    if self.episode_count <= 3:
                        self.unified_logger.logger.info(f"Episode {self.episode_count}: intercepted={intercepted}, reward={episode_reward:.0f}, steps={episode_length}")

                    # Log to TensorBoard immediately for each episode
                    if self.tb_writer:
                        self.tb_writer.add_scalar(
                            'episode/intercepted',
                            1.0 if intercepted else 0.0,
                            self.num_timesteps
                        )
                        self.tb_writer.add_scalar(
                            'episode/reward',
                            episode_reward,
                            self.num_timesteps
                        )

                    # Log aggregated metrics every N episodes
                    if self.episode_count % self.log_interval == 0:
                        success_rate = np.mean(self.interceptions[-100:]) * 100
                        self.unified_logger.log_metrics({
                            'episode': self.episode_count,
                            'mean_reward': np.mean(self.episode_rewards[-100:]),
                            'mean_length': np.mean(self.episode_lengths[-100:]),
                            'success_rate_pct': success_rate,
                            'total_timesteps': self.num_timesteps
                        })

                        # Log rolling success rate to TensorBoard
                        if self.tb_writer:
                            self.tb_writer.add_scalar(
                                'episode/success_rate_pct',
                                success_rate,
                                self.num_timesteps
                            )

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

    # OPTIMIZATION: VecNormalize removed to prevent double normalization
    # Observations are already manually normalized in core.py observation generator
    # Double normalization creates non-stationary input distributions during curriculum
    #
    # If normalization is needed in future, use ONLY VecNormalize OR manual normalization, not both
    # envs = VecNormalize(
    #     envs,
    #     norm_obs=True,
    #     norm_reward=True,
    #     clip_obs=10.0,
    #     clip_reward=10.0
    # )
    
    # Detect available device with graceful fallback
    device = get_device()
    logger.logger.info(f"Using device: {device}")

    # Create model with LSTM support for handling partial observability
    # LSTM policy maintains hidden state across timesteps to integrate radar observations
    use_lstm = config['training'].get('use_lstm', True)

    if use_lstm:
        # RecurrentPPO with LSTM for temporal integration of radar detections
        policy_kwargs = dict(
            net_arch=config['training'].get('net_arch', [256, 256]),
            activation_fn=torch.nn.ReLU,
            lstm_hidden_size=config['training'].get('lstm_hidden_size', 256),
            n_lstm_layers=config['training'].get('n_lstm_layers', 1),
            enable_critic_lstm=config['training'].get('enable_critic_lstm', True),
            lstm_kwargs=dict(dropout=0.0)  # No dropout for stable training
        )

        # Use RecurrentPPO for LSTM support
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO(
            "MlpLstmPolicy",
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
            device=device,
            verbose=1
        )
        logger.logger.info("Using RecurrentPPO with LSTM policy for temporal integration")
    else:
        # Standard MLP policy (memoryless, for comparison)
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
            device=device,
            verbose=1
        )
        logger.logger.info("Using standard MLP policy (memoryless)")
    
    logger.logger.info(f"Created PPO model with {n_envs} parallel environments")
    
    # Create callbacks
    callbacks = []
    
    # Custom training callback
    callbacks.append(CustomTrainingCallback(logger, config['training']))
    
    # Episode logging callback
    callbacks.append(EpisodeLoggingCallback(logger))
    
    # Checkpoint callback - create timestamped directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_steps = config['training']['total_timesteps']
    checkpoint_base = Path(config.get('checkpointing', {}).get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir = checkpoint_base / f"training_{timestamp}_{total_steps}steps"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    callbacks.append(CheckpointCallback(
        save_freq=config['training'].get('checkpoint_freq', 10000),
        save_path=str(checkpoint_dir / "checkpoints"),
        name_prefix="model",
        save_vecnormalize=True
    ))

    # Evaluation callback
    eval_env = DummyVecEnv([create_env(env_config)])
    # VecNormalize removed from training, keep it commented for eval too
    # eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(checkpoint_dir / "eval_logs"),
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

    # VecNormalize removed - skip saving it
    # Only save if VecNormalize wrapper is actually used
    if hasattr(envs, 'save'):
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