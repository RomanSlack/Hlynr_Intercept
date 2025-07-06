#!/usr/bin/env python3
"""
Phase-2 PPO Training Script for AegisIntercept 3D Environment

This script trains a PPO agent on the 3D missile intercept environment with:
- Full 3D physics including gravity and thrust vectoring
- Smart adversary with evasion behavior
- Comprehensive checkpointing system for training resume
- Enhanced reward shaping for 3D space
- Flexible rendering modes (headless/visual)

Usage:
    python -m scripts.train_ppo_phase2 --headless --checkpoint-interval 10000 --device cpu
    python train_ppo_phase2.py --total-timesteps 500000 --n-envs 8
    python -m scripts.train_ppo_phase2.py --auto-resume --total-timesteps 500000 --device cpu

To visualize a trained model:
    python -m scripts.train_ppo_phase2 --load-model models/ppo_phase2.zip --render-eval --device cpu
"""

import argparse
import os
import time
import glob
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

from aegis_intercept.envs import Aegis3DEnv


class EarlyStoppingCallback(BaseCallback):
    """
    Enhanced early stopping callback for 3D intercept missions.
    Stops training when agent consistently achieves successful intercepts.
    """
    
    def __init__(self, reward_threshold: float = 5.0, check_freq: int = 2000, 
                 min_episodes: int = 100, success_rate: float = 0.8, verbose: int = 0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq
        self.min_episodes = min_episodes
        self.success_rate = success_rate
        self.episode_rewards = []
        self.episode_count = 0
        self.success_count = 0
    
    def _on_step(self) -> bool:
        # Collect episode rewards from all environments
        for env_idx in range(self.training_env.num_envs):
            if len(self.locals.get("infos", [])) > env_idx:
                info = self.locals["infos"][env_idx]
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    self.episode_rewards.append(ep_reward)
                    self.episode_count += 1
                    
                    # Count successful intercepts
                    if ep_reward >= self.reward_threshold:
                        self.success_count += 1
        
        # Check stopping condition
        if (self.n_calls % self.check_freq == 0 and 
            len(self.episode_rewards) >= self.min_episodes):
            
            recent_rewards = self.episode_rewards[-self.min_episodes:]
            recent_successes = sum(1 for r in recent_rewards if r >= self.reward_threshold)
            current_success_rate = recent_successes / len(recent_rewards)
            mean_reward = np.mean(recent_rewards)
            
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Success rate: {current_success_rate:.2%}, "
                      f"Mean reward: {mean_reward:.3f}")
            
            if current_success_rate >= self.success_rate:
                if self.verbose > 0:
                    print(f"Early stopping: Success rate {current_success_rate:.2%} >= "
                          f"{self.success_rate:.2%}")
                return False
        
        return True


class Enhanced3DTensorBoardCallback(BaseCallback):
    """
    Enhanced TensorBoard callback for 3D environment metrics.
    """
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.intercept_distances = []
    
    def _on_training_start(self) -> None:
        self.writer = SummaryWriter(self.log_dir)
    
    def _on_step(self) -> bool:
        # Log episode statistics and 3D-specific metrics
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
                    
                    # Log success/failure
                    success = 1 if ep_reward >= 5.0 else 0
                    self.writer.add_scalar("episode/success", success, self.n_calls)
                
                # Log 3D-specific metrics if available
                if hasattr(info, 'distance_to_adversary'):
                    self.writer.add_scalar("metrics/distance_to_adversary", 
                                         info['distance_to_adversary'], self.n_calls)
        
        # Log training metrics every 1000 steps
        if self.n_calls % 1000 == 0 and len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-20:]
            recent_lengths = self.episode_lengths[-20:]
            
            self.writer.add_scalar("train/mean_reward", np.mean(recent_rewards), self.n_calls)
            self.writer.add_scalar("train/mean_episode_length", np.mean(recent_lengths), self.n_calls)
            
            # Success rate calculation
            recent_successes = sum(1 for r in recent_rewards if r >= 5.0)
            success_rate = recent_successes / len(recent_rewards) if recent_rewards else 0
            self.writer.add_scalar("train/success_rate", success_rate, self.n_calls)
        
        return True
    
    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()


def detect_device() -> str:
    """
    Auto-detect best available device for training.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def make_env(rank: int = 0, seed: int = 0, render_mode: Optional[str] = None) -> gym.Env:
    """
    Create a single 3D environment instance.
    """
    def _init() -> gym.Env:
        env = Aegis3DEnv(render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_vectorized_env(n_envs: int = 4, seed: int = 42, 
                         render_mode: Optional[str] = None) -> SubprocVecEnv:
    """
    Create vectorized 3D environment using SubprocVecEnv.
    """
    env_fns = [make_env(rank=i, seed=seed, render_mode=render_mode) 
               for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    return vec_env


def evaluate_policy(model: PPO, env: gym.Env, n_episodes: int = 10, 
                   deterministic: bool = True, render: bool = False) -> Dict[str, float]:
    """
    Evaluate trained policy over multiple episodes with 3D metrics.
    """
    episode_rewards = []
    episode_lengths = []
    intercept_distances = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        terminated = False
        min_distance = float('inf')
        
        while not terminated:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Track minimum intercept distance
            if 'distance_to_adversary' in info:
                min_distance = min(min_distance, info['distance_to_adversary'])
            
            if render:
                env.render()
            
            if truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        intercept_distances.append(min_distance)
        
        if episode_reward >= 5.0:  # Success threshold
            success_count += 1
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "success_rate": success_count / n_episodes,
        "mean_min_distance": np.mean(intercept_distances)
    }


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """Find the latest checkpoint file."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "rl_model_*_steps.zip"))
    if not checkpoint_files:
        return None
    
    # Sort by step number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]))
    return checkpoint_files[-1]


def train_ppo_model(
    total_timesteps: int = 200000,
    save_path: str = "models/ppo_phase2.zip",
    checkpoint_interval: int = 10000,
    resume_checkpoint: Optional[str] = None,
    n_envs: int = 4,
    device: str = "auto",
    early_stopping: bool = True,
    headless: bool = False,
    verbose: int = 1
) -> PPO:
    """
    Train PPO model on Aegis3D environment with enhanced features.
    """
    # Auto-detect device if not specified
    if device == "auto":
        device = detect_device()
    
    print(f"Training PPO on Aegis3D Environment")
    print(f"Device: {device}")
    print(f"Parallel environments: {n_envs}")
    print(f"Headless mode: {headless}")
    
    # Set environment variable for headless rendering
    if headless:
        os.environ["AEGIS_HEADLESS"] = "true"
        render_mode = None
    else:
        render_mode = "human"
    
    # Create directories
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = "./runs/ppo_phase2"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = "./checkpoints"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Create vectorized environment
    vec_env = create_vectorized_env(n_envs=n_envs, render_mode=render_mode)
    
    # Wrap with VecNormalize for observation normalization
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Configure PPO for 3D environment
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),  # Larger network for 3D
        activation_fn=torch.nn.ReLU
    )
    
    # Initialize or load model
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        model = PPO.load(resume_checkpoint, env=vec_env, device=device)
        
        # Load VecNormalize stats if available
        vecnorm_path = resume_checkpoint.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            print(f"Loaded VecNormalize stats from {vecnorm_path}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=2e-4,  # Slightly lower for 3D complexity
            n_steps=2048,
            batch_size=128,      # Larger batch for stability
            n_epochs=10,
            gamma=0.995,         # Higher gamma for longer episodes
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.005,      # Lower entropy for more focused policy
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose,
            tensorboard_log=log_dir
        )
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=checkpoint_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # Enhanced TensorBoard logging
    tb_callback = Enhanced3DTensorBoardCallback(log_dir=log_dir, verbose=verbose)
    callbacks.append(tb_callback)
    
    # Early stopping callback
    if early_stopping:
        early_stop_callback = EarlyStoppingCallback(
            reward_threshold=5.0,
            check_freq=2000,
            min_episodes=100,
            success_rate=0.8,
            verbose=verbose
        )
        callbacks.append(early_stop_callback)
    
    # Train the model
    print(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the final trained model
    model.save(save_path)
    print(f"Final model saved to: {save_path}")
    
    # Save the VecNormalize statistics
    vec_normalize_path = save_path.replace(".zip", "_vecnormalize.pkl")
    vec_env.save(vec_normalize_path)
    print(f"VecNormalize stats saved to: {vec_normalize_path}")
    
    # Clean up
    vec_env.close()
    
    return model


def main():
    """
    Main training function with comprehensive command-line interface.
    """
    parser = argparse.ArgumentParser(description="Train PPO agent on Aegis3D environment")
    
    # Training parameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200000,
        help="Total training timesteps (default: 200000)"
    )
    
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/ppo_phase2.zip",
        help="Path to save trained model (default: models/ppo_phase2.zip)"
    )
    
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Save checkpoint every N steps (default: 10000)"
    )
    
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from latest checkpoint"
    )
    
    # Environment parameters
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no rendering)"
    )
    
    # Training configuration
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
    
    # Evaluation
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate trained model after training"
    )
    
    parser.add_argument(
        "--load-model",
        type=str,
        help="Load and evaluate a specific model file"
    )
    
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation (default: 10)"
    )
    
    parser.add_argument(
        "--render-eval",
        action="store_true",
        help="Render during evaluation"
    )
    
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Handle auto-resume
    if args.auto_resume and not args.resume_checkpoint:
        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint:
            args.resume_checkpoint = latest_checkpoint
            print(f"Auto-resuming from: {latest_checkpoint}")
    
    # Evaluation mode
    if args.load_model:
        print(f"Loading model for evaluation: {args.load_model}")
        
        # Create single environment for evaluation
        eval_env = Aegis3DEnv(render_mode="human" if args.render_eval else None)
        eval_env = Monitor(eval_env)
        
        # Load model
        model = PPO.load(args.load_model, device=args.device)
        
        # Load VecNormalize stats if available
        vec_normalize_path = args.load_model.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vec_normalize_path):
            dummy_vec_env = create_vectorized_env(n_envs=1)
            vec_norm = VecNormalize.load(vec_normalize_path, dummy_vec_env)
            vec_norm.training = False
            vec_norm.norm_reward = False
            # Note: For evaluation, we use unnormalized environment
            dummy_vec_env.close()
        
        eval_stats = evaluate_policy(
            model, eval_env, 
            n_episodes=args.eval_episodes, 
            deterministic=True,
            render=args.render_eval
        )
        
        print(f"\nEvaluation Results ({args.eval_episodes} episodes):")
        print(f"  Mean Reward: {eval_stats['mean_reward']:.3f} +/- {eval_stats['std_reward']:.3f}")
        print(f"  Success Rate: {eval_stats['success_rate']:.2%}")
        print(f"  Mean Length: {eval_stats['mean_length']:.1f} +/- {eval_stats['std_length']:.1f}")
        print(f"  Reward Range: [{eval_stats['min_reward']:.3f}, {eval_stats['max_reward']:.3f}]")
        print(f"  Mean Min Distance: {eval_stats['mean_min_distance']:.3f}")
        
        eval_env.close()
        return
    
    # Train the model
    model = train_ppo_model(
        total_timesteps=args.total_timesteps,
        save_path=args.save_path,
        checkpoint_interval=args.checkpoint_interval,
        resume_checkpoint=args.resume_checkpoint,
        n_envs=args.n_envs,
        device=args.device,
        early_stopping=not args.no_early_stopping,
        headless=args.headless,
        verbose=args.verbose
    )
    
    # Optional evaluation
    if args.evaluate:
        print("\nEvaluating trained model...")
        eval_env = Aegis3DEnv(render_mode="human" if args.render_eval else None)
        eval_env = Monitor(eval_env)
        
        eval_stats = evaluate_policy(
            model, eval_env, 
            n_episodes=args.eval_episodes, 
            deterministic=True,
            render=args.render_eval
        )
        
        print("Final Evaluation Results:")
        print(f"  Mean Reward: {eval_stats['mean_reward']:.3f} +/- {eval_stats['std_reward']:.3f}")
        print(f"  Success Rate: {eval_stats['success_rate']:.2%}")
        print(f"  Mean Length: {eval_stats['mean_length']:.1f} +/- {eval_stats['std_length']:.1f}")
        print(f"  Reward Range: [{eval_stats['min_reward']:.3f}, {eval_stats['max_reward']:.3f}]")
        print(f"  Mean Min Distance: {eval_stats['mean_min_distance']:.3f}")
        
        eval_env.close()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()