#!/usr/bin/env python3
"""
HRL Specialist Pre-Training Script - Full Implementation

Trains individual specialist policies (SEARCH, TRACK, TERMINAL) using PPO with LSTM.
Each specialist is trained separately to learn option-specific behaviors before
being integrated into the hierarchical system.

Usage:
    # Train search specialist
    python scripts/train_hrl_pretrain.py --agent search --config configs/hrl/search_specialist.yaml

    # Train track specialist
    python scripts/train_hrl_pretrain.py --agent track --config configs/hrl/track_specialist.yaml

    # Train terminal specialist
    python scripts/train_hrl_pretrain.py --agent terminal --config configs/hrl/terminal_specialist.yaml

    # Train all specialists sequentially
    python scripts/train_hrl_pretrain.py --agent all --config configs/hrl/hrl_curriculum.yaml
"""

import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Ensure clean CUDA environment
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import RecurrentPPO for LSTM support
try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    RecurrentPPO = None
    HAS_RECURRENT_PPO = False
    print("Warning: sb3_contrib not installed, LSTM policies unavailable")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import InterceptEnvironment
from logger import UnifiedLogger
from hrl.wrappers import SpecialistRewardWrapper


class CustomMLP(BaseFeaturesExtractor):
    """
    Custom MLP features extractor with orthogonal initialization.
    Same architecture as flat PPO for consistency.
    """

    def __init__(self, observation_space, features_dim: int = 256,
                 net_arch: list = [512, 512, 256],
                 use_orthogonal_init: bool = True,
                 use_layer_norm: bool = True):
        super().__init__(observation_space, features_dim)

        self.net_arch = net_arch
        self.use_layer_norm = use_layer_norm

        # Build network layers
        layers = []
        input_dim = observation_space.shape[0]

        for i, hidden_dim in enumerate(net_arch):
            linear = nn.Linear(input_dim, hidden_dim)

            if use_orthogonal_init:
                nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
                nn.init.constant_(linear.bias, 0.0)

            layers.append(linear)

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self._features_dim = net_arch[-1]

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


def get_device() -> str:
    """Detect and select compute device."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            test_tensor = torch.zeros(1, device='cuda')
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.empty_cache()

            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ Using CUDA device: {gpu_name} ({gpu_memory:.1f}GB)")
            return 'cuda'
        except RuntimeError as e:
            print(f"⚠️  CUDA error: {e}, falling back to CPU")
            return 'cpu'
    elif torch.backends.mps.is_available():
        print(f"✓ Using MPS (Apple Silicon GPU) acceleration")
        return 'mps'
    else:
        print(f"ℹ️  No GPU available, using CPU")
        return 'cpu'


class SpecialistTrainingCallback(BaseCallback):
    """Callback for specialist-specific logging and metrics."""

    def __init__(self, logger: UnifiedLogger, specialist_name: str):
        super().__init__()
        self.unified_logger = logger
        self.specialist_name = specialist_name
        self.tb_writer = logger.get_tensorboard_writer()

        # Track validation metrics to catch measurement bugs early
        self.episode_min_distances = []
        self.episode_successes = []

    def _on_training_start(self):
        self.unified_logger.logger.info(f"Training {self.specialist_name} specialist started")

    def _on_step(self) -> bool:
        # Extract info from environments to track REAL distances
        # This catches the normalized observation bug early
        for info in self.locals.get('infos', []):
            if 'interceptor_pos' in info and 'missile_pos' in info:
                int_pos = np.array(info['interceptor_pos'])
                mis_pos = np.array(info['missile_pos'])
                real_distance = np.linalg.norm(int_pos - mis_pos)

                # Log to tensorboard periodically
                if self.num_timesteps % 1000 == 0:
                    if self.tb_writer:
                        self.tb_writer.add_scalar(
                            f'specialists/{self.specialist_name}/real_distance',
                            real_distance,
                            self.num_timesteps
                        )

            # Track episode outcomes
            if info.get('terminal', False) or info.get('TimeLimit.truncated', False):
                self.episode_successes.append(info.get('intercepted', False))
                if 'distance' in info:
                    self.episode_min_distances.append(info['distance'])

        return True

    def _on_rollout_end(self):
        """Log specialist-specific metrics."""
        if self.tb_writer and hasattr(self.model, '_logger'):
            rollout_info = self.model._logger.name_to_value

            # Log under specialist namespace
            for key in ['policy_gradient_loss', 'value_loss', 'entropy_loss', 'approx_kl']:
                if key in rollout_info:
                    self.tb_writer.add_scalar(
                        f'specialists/{self.specialist_name}/{key}',
                        rollout_info[key],
                        self.num_timesteps
                    )

        # Log validation metrics every rollout
        if self.episode_min_distances:
            mean_dist = np.mean(self.episode_min_distances[-100:])  # Last 100 episodes
            if self.tb_writer:
                self.tb_writer.add_scalar(
                    f'specialists/{self.specialist_name}/mean_min_distance_m',
                    mean_dist,
                    self.num_timesteps
                )

            # SANITY CHECK: Warn if distances are implausibly small
            if mean_dist < 1.0:  # Less than 1 meter
                self.unified_logger.logger.warning(
                    f"⚠️  SANITY CHECK: Mean distance {mean_dist:.2f}m is suspiciously small!\n"
                    f"    This may indicate a measurement bug (using normalized obs?)."
                )

        if self.episode_successes:
            success_rate = np.mean(self.episode_successes[-100:])
            if self.tb_writer:
                self.tb_writer.add_scalar(
                    f'specialists/{self.specialist_name}/success_rate',
                    success_rate,
                    self.num_timesteps
                )


def create_env(config: Dict[str, Any], rank: int = 0, specialist_type: str = None) -> InterceptEnvironment:
    """
    Create and wrap environment with specialist-specific rewards.

    Args:
        config: Environment configuration
        rank: Environment rank for parallel training
        specialist_type: 'search', 'track', or 'terminal' for tactical rewards
    """
    def _init():
        env = InterceptEnvironment(config)

        # CRITICAL: Apply specialist reward wrapper for tactical rewards
        # Without this, specialists train on base reward which doesn't
        # differentiate between search/track/terminal objectives
        if specialist_type is not None:
            env = SpecialistRewardWrapper(
                env,
                specialist_type=specialist_type,
                blend_with_base=0.3,  # 70% tactical + 30% base reward
            )

        env = Monitor(env)
        return env
    return _init


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load parent config if specified
    if 'parent_config' in config:
        parent_path = config_path.parent / config['parent_config']
        if parent_path.exists():
            with open(parent_path, 'r') as f:
                parent_config = yaml.safe_load(f)

            # Merge configs (specialist config overrides parent)
            merged = parent_config.copy()
            for key in config:
                if key != 'parent_config':
                    if isinstance(config[key], dict) and key in merged:
                        merged[key].update(config[key])
                    else:
                        merged[key] = config[key]
            config = merged

    return config


def train_specialist(
    agent: str,
    config: Dict[str, Any],
    checkpoint_dir: Optional[Path] = None,
    steps_override: Optional[int] = None,
) -> PPO:
    """
    Train a single specialist policy.

    Args:
        agent: Specialist type ('search', 'track', 'terminal')
        config: Configuration dictionary
        checkpoint_dir: Directory to save checkpoints
        steps_override: Override total training steps

    Returns:
        Trained PPO model
    """
    # Create logger
    logger = UnifiedLogger(
        log_dir=config.get('logging', {}).get('log_dir', 'logs'),
        run_name=f"specialist_{agent}"
    )

    logger.logger.info(f"\n{'='*60}")
    logger.logger.info(f"Training {agent.upper()} Specialist")
    logger.logger.info(f"{'='*60}\n")

    # Create environments with specialist-specific reward wrapper
    n_envs = config['training'].get('n_envs', 4)
    env_config = config.get('environment', {})

    logger.logger.info(f"Creating {n_envs} environments with {agent.upper()} tactical rewards")
    envs = DummyVecEnv([create_env(env_config, i, specialist_type=agent) for i in range(n_envs)])

    # Add frame-stacking
    frame_stack = config['training'].get('frame_stack', 4)
    if frame_stack > 1:
        logger.logger.info(f"Adding frame-stacking: {frame_stack} frames")
        envs = VecFrameStack(envs, n_stack=frame_stack)

    # Add observation normalization
    logger.logger.info("Adding VecNormalize for observation normalization")
    envs = VecNormalize(
        envs,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=config['training'].get('gamma', 0.99)
    )

    # Detect device
    device = get_device()

    # Get network architecture
    net_arch = config['training'].get('net_arch', [512, 512, 256])
    use_orthogonal_init = config['training'].get('use_orthogonal_init', True)
    use_layer_norm = config['training'].get('use_layer_norm', True)
    use_lstm = config['training'].get('use_lstm', False)

    logger.logger.info(f"Network architecture: {net_arch}")
    logger.logger.info(f"  - LSTM: {use_lstm}")
    logger.logger.info(f"  - Orthogonal init: {use_orthogonal_init}")
    logger.logger.info(f"  - Layer normalization: {use_layer_norm}")

    # Configure policy and create model
    if use_lstm:
        if not HAS_RECURRENT_PPO:
            raise ImportError(
                "LSTM enabled but sb3_contrib not installed. "
                "Install with: pip install sb3-contrib"
            )

        # RecurrentPPO uses different policy kwargs structure
        policy_kwargs = dict(
            features_extractor_class=CustomMLP,
            features_extractor_kwargs=dict(
                features_dim=net_arch[-1],
                net_arch=net_arch,
                use_orthogonal_init=use_orthogonal_init,
                use_layer_norm=use_layer_norm
            ),
            net_arch=[],  # Network handled by features extractor
            lstm_hidden_size=config['training'].get('lstm_hidden_dim', 256),
            n_lstm_layers=1,
            shared_lstm=False,  # Separate LSTM for actor and critic
            enable_critic_lstm=True,
        )

        logger.logger.info(f"Creating RecurrentPPO model with LSTM (hidden_dim={config['training'].get('lstm_hidden_dim', 256)})")
        model = RecurrentPPO(
            "MlpLstmPolicy",
            envs,
            learning_rate=config['training'].get('learning_rate', 3e-4),
            n_steps=config['training'].get('n_steps', 1024),
            batch_size=config['training'].get('batch_size', 256),
            n_epochs=config['training'].get('n_epochs', 10),
            gamma=config['training'].get('gamma', 0.99),
            gae_lambda=config['training'].get('gae_lambda', 0.95),
            clip_range=config['training'].get('clip_range', 0.2),
            ent_coef=config['training'].get('ent_coef', 0.02),
            vf_coef=config['training'].get('vf_coef', 0.5),
            max_grad_norm=config['training'].get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(logger.log_dir / "tensorboard"),
            device=device,
            verbose=1
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=CustomMLP,
            features_extractor_kwargs=dict(
                features_dim=net_arch[-1],
                net_arch=net_arch,
                use_orthogonal_init=use_orthogonal_init,
                use_layer_norm=use_layer_norm
            ),
            net_arch=[],
            activation_fn=torch.nn.ReLU
        )

        logger.logger.info(f"Creating PPO model (MLP)")
        model = PPO(
            "MlpPolicy",
            envs,
            learning_rate=config['training'].get('learning_rate', 3e-4),
            n_steps=config['training'].get('n_steps', 1024),
            batch_size=config['training'].get('batch_size', 256),
            n_epochs=config['training'].get('n_epochs', 10),
            gamma=config['training'].get('gamma', 0.99),
            gae_lambda=config['training'].get('gae_lambda', 0.95),
            clip_range=config['training'].get('clip_range', 0.2),
            ent_coef=config['training'].get('ent_coef', 0.02),
            vf_coef=config['training'].get('vf_coef', 0.5),
            max_grad_norm=config['training'].get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(logger.log_dir / "tensorboard"),
            device=device,
            verbose=1
        )

    logger.logger.info(f"Created PPO model with {n_envs} parallel environments")

    # Setup checkpoint directory
    if checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_steps = steps_override or config['training'].get('total_timesteps', 100000)
        checkpoint_base = Path(config.get('checkpointing', {}).get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir = checkpoint_base / f"hrl/specialists/{agent}/{timestamp}_{total_steps}steps"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Create callbacks
    callbacks = []

    # Specialist training callback
    callbacks.append(SpecialistTrainingCallback(logger, agent))

    # Checkpoint callback
    callbacks.append(CheckpointCallback(
        save_freq=config['training'].get('checkpoint_freq', 10000),
        save_path=str(checkpoint_dir / "checkpoints"),
        name_prefix=f"{agent}_model",
        save_vecnormalize=True
    ))

    # Evaluation callback (also use specialist rewards for consistent evaluation)
    eval_env = DummyVecEnv([create_env(env_config, specialist_type=agent)])
    if frame_stack > 1:
        eval_env = VecFrameStack(eval_env, n_stack=frame_stack)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
        gamma=config['training'].get('gamma', 0.99)
    )

    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(checkpoint_dir / "eval_logs"),
        eval_freq=config['training'].get('eval_freq', 10000),
        n_eval_episodes=config['training'].get('n_eval_episodes', 10),
        deterministic=True
    ))

    callback_list = CallbackList(callbacks)

    # Train
    total_timesteps = steps_override or config['training'].get('total_timesteps', 100000)
    logger.logger.info(f"Starting training for {total_timesteps} timesteps")

    try:
        model.learn(
            total_timesteps=total_timesteps,
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

    # Save VecNormalize statistics
    if isinstance(envs, VecNormalize):
        envs.save(str(final_path / "vec_normalize.pkl"))
        logger.logger.info("Saved VecNormalize statistics")

    logger.logger.info(f"Training completed. Final model saved to {final_path}")

    # Create manifest
    logger.create_manifest()

    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-train HRL specialist policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-train search specialist
  python scripts/train_hrl_pretrain.py --agent search --config configs/hrl/search_specialist.yaml

  # Pre-train track specialist
  python scripts/train_hrl_pretrain.py --agent track --config configs/hrl/track_specialist.yaml

  # Pre-train terminal specialist
  python scripts/train_hrl_pretrain.py --agent terminal --config configs/hrl/terminal_specialist.yaml

  # Pre-train all specialists
  python scripts/train_hrl_pretrain.py --agent all --config configs/hrl/hrl_curriculum.yaml
        """
    )

    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["search", "track", "terminal", "all"],
        help="Which specialist to pre-train"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to specialist config YAML"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total training steps"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("\n🚀 HRL Specialist Pre-Training")
    print("=" * 60)

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"\n❌ Error loading config: {e}")
        return 1

    # Prepare checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    # Train specialists
    if args.agent == "all":
        print("\nTraining all specialists sequentially...")
        for agent in ["search", "track", "terminal"]:
            print(f"\n{'='*60}")
            print(f"Starting {agent} specialist training")
            print(f"{'='*60}\n")

            # Load agent-specific config
            agent_config_path = Path(args.config).parent / f"{agent}_specialist.yaml"
            if agent_config_path.exists():
                agent_config = load_config(str(agent_config_path))
            else:
                agent_config = config

            train_specialist(
                agent=agent,
                config=agent_config,
                checkpoint_dir=checkpoint_dir / agent if checkpoint_dir else None,
                steps_override=args.steps
            )
            print(f"\n✅ {agent} specialist training completed\n")

        print("\n✅ All specialists trained successfully")
    else:
        # Train single specialist
        train_specialist(
            agent=args.agent,
            config=config,
            checkpoint_dir=checkpoint_dir,
            steps_override=args.steps
        )
        print(f"\n✅ {args.agent} specialist training completed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
