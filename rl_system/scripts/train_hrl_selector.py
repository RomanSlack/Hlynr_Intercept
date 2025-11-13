#!/usr/bin/env python3
"""
HRL Selector Training Script

Trains the high-level selector policy to choose between specialist options
(SEARCH, TRACK, TERMINAL) based on abstract 7D observations. Specialists
are frozen during selector training.

Usage:
    # Train selector with pre-trained specialists
    python scripts/train_hrl_selector.py \
        --config configs/hrl/selector_config.yaml \
        --specialist-dir checkpoints/hrl/specialists/

    # Resume from checkpoint
    python scripts/train_hrl_selector.py \
        --config configs/hrl/selector_config.yaml \
        --specialist-dir checkpoints/hrl/specialists/ \
        --resume checkpoints/hrl/selector/model.zip
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import InterceptEnvironment
from logger import UnifiedLogger
from hrl import (
    HierarchicalManager,
    SelectorPolicy,
    SearchSpecialist,
    TrackSpecialist,
    TerminalSpecialist,
    Option,
    make_hrl_env,
)
from hrl.wrappers import AbstractObservationWrapper


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


class SelectorTrainingCallback(BaseCallback):
    """Callback for selector-specific logging and metrics."""

    def __init__(self, logger: UnifiedLogger):
        super().__init__()
        self.unified_logger = logger
        self.tb_writer = logger.get_tensorboard_writer()
        self.option_counts = {opt: 0 for opt in Option}
        self.episode_count = 0

    def _on_training_start(self):
        self.unified_logger.logger.info("Selector training started")

    def _on_step(self) -> bool:
        # Track option usage from info dict
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'option' in info:
                option_name = info['option']
                # Convert name to Option enum
                for opt in Option:
                    if opt.name == option_name:
                        self.option_counts[opt] += 1
                        break

        return True

    def _on_rollout_end(self):
        """Log selector-specific metrics."""
        if self.tb_writer and hasattr(self.model, '_logger'):
            rollout_info = self.model._logger.name_to_value

            # Log training metrics under selector namespace
            for key in ['policy_gradient_loss', 'value_loss', 'entropy_loss', 'approx_kl']:
                if key in rollout_info:
                    self.tb_writer.add_scalar(
                        f'selector/{key}',
                        rollout_info[key],
                        self.num_timesteps
                    )

            # Log option usage statistics
            total_decisions = sum(self.option_counts.values())
            if total_decisions > 0:
                for opt, count in self.option_counts.items():
                    usage_pct = (count / total_decisions) * 100
                    self.tb_writer.add_scalar(
                        f'selector/option_usage/{opt.name}_pct',
                        usage_pct,
                        self.num_timesteps
                    )


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

            # Merge configs
            merged = parent_config.copy()
            for key in config:
                if key != 'parent_config':
                    if isinstance(config[key], dict) and key in merged:
                        merged[key].update(config[key])
                    else:
                        merged[key] = config[key]
            config = merged

    return config


def load_specialists(specialist_dir: Path, device: str = 'auto') -> Dict[Option, Any]:
    """
    Load pre-trained specialist checkpoints.

    Args:
        specialist_dir: Directory containing specialist subdirectories
        device: Device to load models on

    Returns:
        Dictionary mapping Option to loaded specialist policies
    """
    specialists = {}

    for option, specialist_class in [
        (Option.SEARCH, SearchSpecialist),
        (Option.TRACK, TrackSpecialist),
        (Option.TERMINAL, TerminalSpecialist),
    ]:
        specialist_name = option.name.lower()
        specialist_base_dir = specialist_dir / specialist_name

        # Try different checkpoint paths
        possible_paths = [
            specialist_base_dir / "final" / "model.zip",
            specialist_base_dir / "best" / "best_model.zip",
            specialist_base_dir / "model.zip",
        ]

        # Also check for timestamped training directories (e.g., 20251109_183052_100000steps/final/)
        if specialist_base_dir.exists():
            for subdir in sorted(specialist_base_dir.iterdir(), reverse=True):
                if subdir.is_dir():
                    possible_paths.extend([
                        subdir / "final" / "model.zip",
                        subdir / "best" / "best_model.zip",
                        subdir / "checkpoints" / "model.zip",
                    ])

        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(
                f"Could not find {specialist_name} specialist checkpoint in {specialist_base_dir}\n"
                f"Tried paths: {[str(p) for p in possible_paths[:6]]}"
            )

        print(f"Loading {specialist_name} specialist from: {model_path}")
        specialist = specialist_class(model_path=str(model_path))
        specialists[option] = specialist

    print(f"✓ Loaded all 3 specialists")
    return specialists


def create_hrl_env_for_selector(config: Dict[str, Any], specialists: Dict[Option, Any], rank: int = 0):
    """
    Create HRL environment for selector training.

    CRITICAL: Frame-stacking is applied at single-env level BEFORE HRL wrappers
    so specialists receive 104D observations (26D × 4 frames) as they were trained.

    Wrapper hierarchy:
        Monitor(AbstractObservationWrapper(HRLActionWrapper(FrameStack(InterceptEnv))))
                                                             ↑ 26D → 104D here

    Returns a function that creates the wrapped environment.
    """
    def _init():
        from gymnasium.wrappers import FrameStackObservation

        # Create base environment (26D observations)
        base_env = InterceptEnvironment(config.get('environment', {}))

        # Apply frame-stacking at single-env level (26D → 104D)
        # CRITICAL: This must match the frame_stack used during specialist training
        frame_stack = 4  # Must match specialist training configuration
        stacked_env = FrameStackObservation(base_env, stack_size=frame_stack)

        # Create HRL manager with frozen specialists
        decision_interval = config.get('hrl', {}).get('decision_interval_steps', 100)

        manager = HierarchicalManager(
            selector=None,  # Will be trained
            specialists=specialists,
            decision_interval=decision_interval,
            enable_forced_transitions=True,
        )

        # Wrap with HRL environment (receives 104D frame-stacked observations)
        # CRITICAL: training_selector=True enables external selector training mode
        from hrl.wrappers import HRLActionWrapper
        hrl_env = HRLActionWrapper(
            stacked_env,
            manager,
            return_hrl_info=True,
            training_selector=True  # Accept discrete actions from PPO selector
        )

        # Wrap with abstract observation for selector
        abstract_env = AbstractObservationWrapper(hrl_env)

        # Monitor wrapper for episode tracking
        monitored_env = Monitor(abstract_env)

        return monitored_env

    return _init


def train_selector(
    config: Dict[str, Any],
    specialists: Dict[Option, Any],
    checkpoint_dir: Optional[Path] = None,
    resume_path: Optional[str] = None,
    steps_override: Optional[int] = None,
) -> PPO:
    """
    Train selector policy with frozen specialists.

    Args:
        config: Configuration dictionary
        specialists: Pre-loaded specialist policies (frozen)
        checkpoint_dir: Directory to save checkpoints
        resume_path: Path to resume training from
        steps_override: Override total training steps

    Returns:
        Trained PPO model
    """
    # Create logger
    logger = UnifiedLogger(
        log_dir=config.get('logging', {}).get('log_dir', 'logs'),
        run_name="selector_training"
    )

    logger.logger.info("\n" + "="*60)
    logger.logger.info("Training HRL Selector Policy")
    logger.logger.info("="*60 + "\n")

    # Create environments with HRL wrapper and abstract observations
    n_envs = config['training'].get('n_envs', 4)

    envs = DummyVecEnv([
        create_hrl_env_for_selector(config, specialists, i)
        for i in range(n_envs)
    ])

    # NOTE: Frame-stacking is already applied at single-env level (see create_hrl_env_for_selector)
    # - Specialists receive 104D frame-stacked observations (26D × 4 frames)
    # - Selector receives 7D abstract observations (converted by AbstractObservationWrapper)
    # DO NOT apply VecFrameStack here as it would stack the 7D abstract observations

    # Add observation normalization for abstract state (7D)
    logger.logger.info("Adding VecNormalize for abstract observation normalization (7D)")
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

    # Get network architecture (smaller for discrete action space)
    net_arch = config['training'].get('net_arch', [256, 256])

    logger.logger.info(f"Selector network architecture: {net_arch}")
    logger.logger.info(f"  - Input: 7D abstract observation")
    logger.logger.info(f"  - Output: Discrete(3) - {{SEARCH, TRACK, TERMINAL}}")

    # Configure policy for discrete action space
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn=torch.nn.ReLU
    )

    # Create or load PPO model
    if resume_path:
        logger.logger.info(f"Resuming from checkpoint: {resume_path}")
        model = PPO.load(
            resume_path,
            env=envs,
            device=device,
        )
    else:
        model = PPO(
            "MlpPolicy",
            envs,
            learning_rate=config['training'].get('learning_rate', 3e-4),
            n_steps=config['training'].get('n_steps', 512),
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

    logger.logger.info(f"Created selector PPO model with {n_envs} parallel environments")

    # Setup checkpoint directory
    if checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_steps = steps_override or config['training'].get('total_timesteps', 50000)
        checkpoint_base = Path(config.get('checkpointing', {}).get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir = checkpoint_base / f"hrl/selector/{timestamp}_{total_steps}steps"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Create callbacks
    callbacks = []

    # Selector training callback
    callbacks.append(SelectorTrainingCallback(logger))

    # Checkpoint callback
    # CRITICAL: Convert checkpoint_freq from timesteps to callback invocations
    # With n_envs parallel environments, callback is called once per n_envs timesteps
    checkpoint_freq_timesteps = config['training'].get('checkpoint_freq', 10000)
    checkpoint_freq_callbacks = max(1, checkpoint_freq_timesteps // n_envs)
    callbacks.append(CheckpointCallback(
        save_freq=checkpoint_freq_callbacks,
        save_path=str(checkpoint_dir / "checkpoints"),
        name_prefix="selector_model",
        save_vecnormalize=True
    ))
    if checkpoint_freq_callbacks != checkpoint_freq_timesteps:
        logger.logger.info(f"Checkpoint frequency: {checkpoint_freq_timesteps} timesteps -> {checkpoint_freq_callbacks} callback invocations (n_envs={n_envs})")

    # Evaluation callback
    eval_env = DummyVecEnv([create_hrl_env_for_selector(config, specialists)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
        gamma=config['training'].get('gamma', 0.99)
    )

    # CRITICAL: Convert eval_freq from timesteps to callback invocations
    # With n_envs parallel environments, callback is called once per n_envs timesteps
    eval_freq_timesteps = config['training'].get('eval_freq', 10000)
    eval_freq_callbacks = max(1, eval_freq_timesteps // n_envs)
    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(checkpoint_dir / "eval_logs"),
        eval_freq=eval_freq_callbacks,
        n_eval_episodes=config['training'].get('n_eval_episodes', 10),
        deterministic=True
    ))
    if eval_freq_callbacks != eval_freq_timesteps:
        logger.logger.info(f"Evaluation frequency: {eval_freq_timesteps} timesteps -> {eval_freq_callbacks} callback invocations (n_envs={n_envs})")

    callback_list = CallbackList(callbacks)

    # Train
    total_timesteps = steps_override or config['training'].get('total_timesteps', 50000)
    logger.logger.info(f"Starting selector training for {total_timesteps} timesteps")

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

    logger.logger.info(f"Selector training completed. Final model saved to {final_path}")

    # Create manifest
    logger.create_manifest()

    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train HRL selector policy with frozen specialists",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train selector with pre-trained specialists
  python scripts/train_hrl_selector.py \\
      --config configs/hrl/selector_config.yaml \\
      --specialist-dir checkpoints/hrl/specialists/

  # Resume from checkpoint
  python scripts/train_hrl_selector.py \\
      --config configs/hrl/selector_config.yaml \\
      --specialist-dir checkpoints/hrl/specialists/ \\
      --resume checkpoints/hrl/selector/model.zip
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to selector config YAML"
    )

    parser.add_argument(
        "--specialist-dir",
        type=str,
        required=True,
        help="Directory containing specialist checkpoints (search/, track/, terminal/)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to selector checkpoint to resume from"
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

    print("\n🚀 HRL Selector Training")
    print("=" * 60)

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"\n❌ Error loading config: {e}")
        return 1

    # Load specialists
    try:
        specialist_dir = Path(args.specialist_dir)
        if not specialist_dir.exists():
            raise FileNotFoundError(f"Specialist directory not found: {specialist_dir}")

        device = get_device()
        specialists = load_specialists(specialist_dir, device=device)
    except Exception as e:
        print(f"\n❌ Error loading specialists: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Prepare checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    # Train selector
    try:
        train_selector(
            config=config,
            specialists=specialists,
            checkpoint_dir=checkpoint_dir,
            resume_path=args.resume,
            steps_override=args.steps
        )
        print("\n✅ Selector training completed successfully")
        return 0
    except Exception as e:
        print(f"\n❌ Selector training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
