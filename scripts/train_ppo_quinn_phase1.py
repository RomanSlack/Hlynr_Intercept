#!/usr/bin/env python3
"""
Phase-1 PPO Training Script for AegisIntercept - Quinn's Version

Original implementation using Stable-Baselines3 PPO with vectorized environments,
TensorBoard logging, and Apple Silicon optimization.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

from aegis_intercept.envs import Aegis2DInterceptEnvQuinn


class EarlyStoppingCallback(BaseCallback):
    """
    Custom callback for early stopping when mean reward threshold is reached.
    """
    
    def __init__(self, reward_threshold: float = 0.5, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Collect episode rewards from all environments
        for env_idx in range(self.training_env.num_envs):
            if len(self.locals.get("infos", [])) > env_idx:
                info = self.locals["infos"][env_idx]
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_count += 1
        
        # Check stopping condition every check_freq steps
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) >= 50:
            mean_reward = np.mean(self.episode_rewards[-50:])
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Mean reward over last 50 episodes: {mean_reward:.3f}")
            
            if mean_reward >= self.reward_threshold:
                if self.verbose > 0:
                    print(f"Early stopping: Mean reward {mean_reward:.3f} >= {self.reward_threshold}")
                return False
        
        return True


class TensorBoardCallback(BaseCallback):
    """
    Custom callback for enhanced TensorBoard logging.
    """
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_training_start(self) -> None:
        self.writer = SummaryWriter(self.log_dir)
    
    def _on_step(self) -> bool:
        # Log episode statistics
        for env_idx in range(self.training_env.num_envs):
            if len(self.locals.get("infos", [])) > env_idx:
                info = self.locals["infos"][env_idx]
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    self.writer.add_scalar("episode/reward", ep_reward, self.n_calls)
                    self.writer.add_scalar("episode/length", ep_length, self.n_calls)
        
        # Log training metrics every 1000 steps
        if self.n_calls % 1000 == 0:
            if len(self.episode_rewards) > 0:
                self.writer.add_scalar("train/mean_reward", np.mean(self.episode_rewards[-10:]), self.n_calls)
                self.writer.add_scalar("train/mean_episode_length", np.mean(self.episode_lengths[-10:]), self.n_calls)
        
        return True
    
    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()


def detect_device() -> str:
    """
    Auto-detect best available device for Apple Silicon Macs.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def make_env(rank: int = 0, seed: int = 0) -> gym.Env:
    """
    Create a single environment instance.
    """
    def _init() -> gym.Env:
        env = Aegis2DInterceptEnvQuinn()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_vectorized_env(n_envs: int = 4, seed: int = 42) -> SubprocVecEnv:
    """
    Create vectorized environment using SubprocVecEnv.
    """
    env_fns = [make_env(rank=i, seed=seed) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    return vec_env


def evaluate_policy(model: PPO, env: gym.Env, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
    """
    Evaluate trained policy over multiple episodes.
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        terminated = False
        
        while not terminated:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards)
    }


def train_ppo_model(
    total_timesteps: int = 100000,
    save_path: str = "models/ppo_quinn.zip",
    render_every: int = 0,
    n_envs: int = 4,
    device: str = "auto",
    early_stopping: bool = True,
    verbose: int = 1
) -> PPO:
    """
    Train PPO model on AegisIntercept environment.
    """
    # Auto-detect device if not specified
    if device == "auto":
        device = detect_device()
    
    print(f"Training on device: {device}")
    print(f"Using {n_envs} parallel environments")
    
    # Create directories
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = "./runs/ppo_quinn_phase1"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create vectorized environment
    vec_env = create_vectorized_env(n_envs=n_envs)
    
    # Wrap with VecNormalize for observation normalization
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Configure PPO with Apple Silicon optimizations
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=verbose,
        tensorboard_log=log_dir
    )
    
    # Setup callbacks
    callbacks = []
    
    # TensorBoard logging callback
    tb_callback = TensorBoardCallback(log_dir=log_dir, verbose=verbose)
    callbacks.append(tb_callback)
    
    # Early stopping callback
    if early_stopping:
        early_stop_callback = EarlyStoppingCallback(
            reward_threshold=0.5,
            check_freq=1000,
            verbose=verbose
        )
        callbacks.append(early_stop_callback)
    
    # Rendering setup (for first environment only)
    render_env = None
    if render_every > 0:
        render_env = Aegis2DInterceptEnvQuinn(render_mode="human")
        print(f"Rendering first environment every {render_every} seconds")
    
    # Train the model
    print(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the trained model
    model.save(save_path)
    print(f"Model saved to: {save_path}")
    
    # Save the VecNormalize statistics
    vec_normalize_path = save_path.replace(".zip", "_vecnormalize.pkl")
    vec_env.save(vec_normalize_path)
    print(f"VecNormalize stats saved to: {vec_normalize_path}")
    
    # Clean up
    vec_env.close()
    if render_env:
        render_env.close()
    
    return model


def main():
    """
    Main training function with command-line interface.
    """
    parser = argparse.ArgumentParser(description="Train PPO agent on AegisIntercept environment")
    
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (default: 100000)"
    )
    
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/ppo_quinn.zip",
        help="Path to save trained model (default: models/ppo_quinn.zip)"
    )
    
    parser.add_argument(
        "--render-every",
        type=int,
        default=0,
        help="Render first environment every N seconds (0 to disable, default: 0)"
    )
    
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training (default: auto)"
    )
    
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping callback"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate trained model after training"
    )
    
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Train the model
    model = train_ppo_model(
        total_timesteps=args.total_timesteps,
        save_path=args.save_path,
        render_every=args.render_every,
        n_envs=args.n_envs,
        device=args.device,
        early_stopping=not args.no_early_stopping,
        verbose=args.verbose
    )
    
    # Optional evaluation
    if args.evaluate:
        print("\nEvaluating trained model...")
        eval_env = Aegis2DInterceptEnvQuinn()
        eval_env = Monitor(eval_env)
        
        # Load VecNormalize stats for consistent evaluation
        vec_normalize_path = args.save_path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vec_normalize_path):
            vec_norm = VecNormalize.load(vec_normalize_path, eval_env)
            vec_norm.training = False
            vec_norm.norm_reward = False
            eval_env = vec_norm
        
        eval_stats = evaluate_policy(model, eval_env, n_episodes=10, deterministic=True)
        
        print("Evaluation Results (10 episodes):")
        print(f"  Mean Reward: {eval_stats['mean_reward']:.3f} +/- {eval_stats['std_reward']:.3f}")
        print(f"  Mean Length: {eval_stats['mean_length']:.1f} +/- {eval_stats['std_length']:.1f}")
        print(f"  Reward Range: [{eval_stats['min_reward']:.3f}, {eval_stats['max_reward']:.3f}]")
        
        eval_env.close()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()