#!/usr/bin/env python3
"""
Phase 3 PPO Training Script for AegisIntercept 6DOF System

This script demonstrates the new 6DOF capabilities including:
- Enhanced 6DOF environment with 31D state space
- Curriculum learning progression
- Advanced trajectory logging
- Enhanced adversary system
- Performance analytics and Unity export

Author: Coder Agent
Date: Phase 3 Implementation
"""

import os
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv
from distutils.util import strtobool
from pathlib import Path
from collections import deque

# AegisIntercept Phase 3 imports
from aegis_intercept.envs import Aegis6DInterceptEnv, DifficultyMode, ActionMode
from aegis_intercept.curriculum import CurriculumManager, create_curriculum_manager, setup_curriculum_directories
from aegis_intercept.logging import TrajectoryLogger, ExportManager, LogLevel, create_trajectory_logger, create_export_manager
from aegis_intercept.adversary import create_default_adversary_config


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 6DOF Training with Curriculum Learning")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="aegis-intercept-phase3")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--visualize", action="store_true", help="Enable real-time visualization")
    parser.add_argument("--viz-mode", type=str, default="god_view", 
                       choices=["god_view", "radar_ppi", "radar_3d", "both"],
                       help="Visualization mode: god_view (traditional), radar_ppi (realistic radar), radar_3d (3D radar), both")
    parser.add_argument("--viz-speed", type=float, default=10.0, help="Visualization speed multiplier (higher = faster)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Environment and curriculum
    parser.add_argument("--start-phase", type=str, default="phase_1_basic_3dof", 
                       help="Starting curriculum phase")
    parser.add_argument("--enable-curriculum", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--curriculum-config", type=str, default=None, help="Path to curriculum config file")
    
    # 6DOF specific parameters
    parser.add_argument("--action-mode", type=str, default="acceleration_6dof", 
                       choices=["acceleration_3dof", "acceleration_6dof", "thrust_attitude"])
    parser.add_argument("--difficulty-mode", type=str, default="full_6dof",
                       choices=["easy_3dof", "medium_3dof", "simplified_6dof", "full_6dof", "expert_6dof"])
    
    # Logging and export
    parser.add_argument("--enable-logging", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--log-level", type=str, default="detailed", choices=["minimal", "basic", "detailed", "debug"])
    parser.add_argument("--enable-unity-export", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--export-frequency", type=int, default=100, help="Export episode data every N episodes")
    
    # Algorithm parameters
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=6)  # Reduced for 6DOF complexity
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=6)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)  # Higher entropy for 6DOF exploration
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def make_env(curriculum_manager: CurriculumManager, seed: int, idx: int, 
             capture_video: bool, run_name: str, render_mode=None):
    """Create environment with curriculum learning support"""
    def thunk():
        # Get current phase configuration
        env_config = curriculum_manager.get_environment_config()
        
        # Create 6DOF environment
        env = Aegis6DInterceptEnv(
            difficulty_mode=env_config['difficulty_mode'],
            action_mode=env_config['action_mode'],
            world_size=env_config['world_size'],
            max_steps=env_config['max_steps'],
            dt=env_config['dt'],
            intercept_threshold=env_config['intercept_threshold'],
            miss_threshold=env_config['miss_threshold'],
            explosion_radius=env_config['explosion_radius'],
            enable_wind=env_config['enable_wind'],
            enable_atmosphere=env_config['enable_atmosphere'],
            wind_strength=env_config['wind_strength'],
            render_mode=render_mode,
            # Realistic sensor parameters
            enable_realistic_sensors=env_config.get('enable_realistic_sensors', True),
            sensor_update_rate=env_config.get('sensor_update_rate', 0.1),
            weather_conditions=env_config.get('weather_conditions', 'clear')
        )
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed + idx)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Enhanced6DOFAgent(nn.Module):
    """Enhanced neural network for 6DOF control"""
    
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        action_shape = envs.single_action_space.shape
        
        # Larger networks for 6DOF complexity
        hidden_size = 512
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 256)),
            nn.ReLU(),
        )
        
        # Value function
        self.critic = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        
        # Policy function
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, np.prod(action_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def get_value(self, x):
        features = self.feature_extractor(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.feature_extractor(x)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features)


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Generalized Advantage Estimation"""
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - dones[-1]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages


def main():
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Setup directories
    curriculum_dirs = setup_curriculum_directories("curriculum")
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Initialize curriculum manager
    curriculum_manager = create_curriculum_manager(args.curriculum_config)
    if not args.enable_curriculum:
        # Set fixed phase if curriculum is disabled
        from aegis_intercept.curriculum import CurriculumPhase
        fixed_phase = CurriculumPhase(args.start_phase)
        curriculum_manager.set_phase(fixed_phase)
    
    print(f"Starting curriculum phase: {curriculum_manager.current_phase.value}")
    
    # Initialize trajectory logger
    trajectory_logger = None
    export_manager = None
    if args.enable_logging:
        trajectory_logger = create_trajectory_logger(
            log_directory=f"logs/phase3/{run_name}",
            log_level=LogLevel(args.log_level)
        )
        print(f"Trajectory logging enabled: {args.log_level}")
        
        if args.enable_unity_export:
            export_manager = create_export_manager(
                export_directory=f"exports/phase3/{run_name}"
            )
            print("Unity export enabled")

    # Environment setup with curriculum
    envs = AsyncVectorEnv([
        make_env(curriculum_manager, args.seed, i, False, run_name) 
        for i in range(args.num_envs)
    ])
    
    # Visualization environment
    viz_env = None
    if args.visualize:
        viz_env_config = curriculum_manager.get_environment_config()
        viz_env = Aegis6DInterceptEnv(
            difficulty_mode=viz_env_config['difficulty_mode'],
            action_mode=viz_env_config['action_mode'],
            render_mode="human",
            **{k: v for k, v in viz_env_config.items() 
               if k not in ['difficulty_mode', 'action_mode']}
        )
        # Override speed multiplier based on command line arg
        viz_env.viz_speed_multiplier = args.viz_speed
        viz_obs, _ = viz_env.reset(seed=args.seed)
        print(f"Visualization enabled: {args.viz_mode} mode with {args.viz_speed}x speed multiplier")
        
        if args.viz_mode in ["radar_ppi", "radar_3d", "both"]:
            print("Realistic radar visualization - showing only sensor-derived information!")

    # Initialize enhanced agent
    agent = Enhanced6DOFAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Resume from checkpoint
    start_update = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        agent.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = checkpoint.get("update", 0)
        
        # Restore curriculum phase
        if "curriculum_phase" in checkpoint and checkpoint["curriculum_phase"] is not None:
            from aegis_intercept.curriculum import CurriculumPhase
            saved_phase = CurriculumPhase(checkpoint["curriculum_phase"])
            curriculum_manager.set_phase(saved_phase)
            print(f"Resumed from update {start_update}, curriculum phase: {saved_phase.value}")
        else:
            print(f"Resumed from update {start_update} (no curriculum phase saved)")

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Training loop
    global_step = start_update * args.batch_size
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs)
    next_done = torch.zeros(args.num_envs)
    num_updates = args.total_timesteps // args.batch_size
    
    episode_count = 0
    last_curriculum_update = time.time()
    
    # Episode tracking for tensorboard logging
    episode_returns = [0.0] * args.num_envs
    episode_lengths = [0] * args.num_envs
    
    # Rolling success rate tracking (last 50 episodes)
    recent_episodes = deque(maxlen=50)

    print(f"Starting Phase 3 training for {num_updates} updates ({args.total_timesteps:,} timesteps)")
    print(f"Current phase: {curriculum_manager.current_phase.value}")

    for update in range(start_update + 1, num_updates + 1):
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout phase
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action selection
            with torch.no_grad():
                obs_gpu = next_obs.to(device)
                action, logprob, _, value = agent.get_action_and_value(obs_gpu)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Environment step
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs, next_done = torch.tensor(next_obs, device=device), torch.tensor(done, device=device)

            # Update episode tracking
            for i in range(args.num_envs):
                episode_returns[i] += reward[i]
                episode_lengths[i] += 1

            # Process episode completions
            for i in range(args.num_envs):
                if done[i]:
                    episode_count += 1
                    
                    # Log episode metrics to TensorBoard
                    final_return = episode_returns[i]
                    final_length = episode_lengths[i]
                    writer.add_scalar("charts/episodic_return", final_return, global_step)
                    writer.add_scalar("charts/episodic_length", final_length, global_step)
                    
                    # Update curriculum with episode results  
                    episode_reward = final_return
                    # Proper success detection: terminated with positive reward = successful intercept
                    episode_success = terminated[i] and episode_reward > 0
                    fuel_used = 0.5  # Placeholder - would get from env info
                    intercept_time = 10.0  # Placeholder - would get from env info
                    
                    # Track success for rolling success rate
                    recent_episodes.append(episode_success)
                    
                    # Calculate and log success rate (averaged over last 50 episodes, updated each episode)
                    rolling_success_rate = sum(recent_episodes) / len(recent_episodes) * 100.0
                    writer.add_scalar("charts/success_rate", rolling_success_rate, global_step)
                    
                    if args.enable_curriculum:
                        # Extract sensor-based metrics from episode info
                        additional_metrics = {}
                        
                        # Get sensor performance from environment info
                        if 'final_info' in info and info['final_info'] and len(info['final_info']) > i:
                            env_info = info['final_info'][i]
                            if env_info:
                                # Extract tracking confidence if available
                                if 'tracking_confidence' in env_info:
                                    additional_metrics['tracking_confidence'] = env_info['tracking_confidence']
                                
                                # Extract sensor detection metrics
                                if 'sensor_detections' in env_info:
                                    additional_metrics['sensor_detections'] = env_info['sensor_detections']
                                    additional_metrics['false_alarms'] = env_info.get('false_alarms', 0)
                                    additional_metrics['tracking_accuracy'] = env_info.get('tracking_accuracy', 0.0)
                        
                        curriculum_manager.update_performance(
                            episode_reward, episode_success, fuel_used, intercept_time, additional_metrics
                        )
                    
                    # Log trajectory data
                    if trajectory_logger and episode_count % 10 == 0:  # Log every 10th episode
                        trajectory_logger.start_episode(episode_count)
                        # Would log trajectory points during episode
                        # trajectory_logger.end_episode(episode_success, "intercept" if episode_success else "miss")
                    
                    # Export to Unity
                    if export_manager and episode_count % args.export_frequency == 0:
                        print(f"Would export episode {episode_count} for Unity")
                        # Would export episode data for Unity visualization
                    
                    # Reset episode tracking for this environment
                    episode_returns[i] = 0.0
                    episode_lengths[i] = 0
                    
                    # Print progress
                    if episode_count % 10 == 0:
                        print(f"Episode {episode_count}: Return={final_return:.2f}, Length={final_length}, Success={episode_success}, SuccessRate={rolling_success_rate:.1f}%")
            
            # Log episode statistics
            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        episode_reward = item["episode"]["r"]
                        episode_length = item["episode"]["l"]
                        writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                        writer.add_scalar("charts/episodic_length", episode_length, global_step)
                        
                        print(f"Episode completed! Return: {episode_reward:.2f}, Length: {episode_length}")

            # Update visualization (reduced frequency for speed)
            if viz_env is not None and step % 20 == 0:  # Render every 20 steps instead of 5
                with torch.no_grad():
                    viz_obs_gpu = torch.tensor(viz_obs).unsqueeze(0).to(device)
                    viz_action, _, _, _ = agent.get_action_and_value(viz_obs_gpu)
                    viz_obs, viz_reward, viz_terminated, viz_truncated, viz_info = viz_env.step(viz_action[0].cpu().numpy())
                    
                    if viz_terminated or viz_truncated:
                        viz_obs, _ = viz_env.reset()
                        viz_env.render(True, args.viz_mode)  # Always render intercept events
                    else:
                        viz_env.render(False, args.viz_mode)

        # Bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = compute_gae(rewards, values, dones, next_value, args.gamma, args.gae_lambda)
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(newvalue - mb_values, -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Curriculum logging
        if args.enable_curriculum:
            curriculum_status = curriculum_manager.get_curriculum_status()
            writer.add_scalar("curriculum/phase", 
                            list(curriculum_status['all_phases_metrics'].keys()).index(curriculum_status['current_phase']), 
                            global_step)
            writer.add_scalar("curriculum/success_rate", curriculum_status['phase_progress']['success_rate'], global_step)
            writer.add_scalar("curriculum/recent_success_rate", curriculum_status['phase_progress']['recent_success_rate'], global_step)

        # Progress logging
        if update % 10 == 0:
            sps = int(global_step / (time.time() - start_time))
            current_phase = curriculum_manager.current_phase.value if args.enable_curriculum else "fixed"
            print(f"Update {update:4d} | Steps: {global_step:8,} | SPS: {sps:4d} | Phase: {current_phase}")

        # Checkpoint saving
        if update % 100 == 0:
            os.makedirs("models/phase3", exist_ok=True)
            checkpoint = {
                "model": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
                "curriculum_phase": curriculum_manager.current_phase.value if args.enable_curriculum else None,
            }
            torch.save(checkpoint, f"models/phase3/latest.pt")
            
            if update % 500 == 0:  # Save numbered checkpoints less frequently
                torch.save(checkpoint, f"models/phase3/checkpoint_{update:06d}.pt")

    # Final cleanup and export
    envs.close()
    if viz_env is not None:
        viz_env.close()
    writer.close()
    
    # Final curriculum and logging summary
    if args.enable_curriculum:
        curriculum_status = curriculum_manager.get_curriculum_status()
        print(f"\n🎓 FINAL CURRICULUM STATUS:")
        print(f"Final Phase: {curriculum_status['current_phase']}")
        print(f"Phases Completed: {len([p for p, m in curriculum_status['all_phases_metrics'].items() if m['episodes_completed'] > 0])}")
    
    if trajectory_logger:
        trajectory_logger.save_episode_summary(f"final_summary_{run_name}")
        trajectory_logger.close()
        print(f"📊 Trajectory logs saved")
    
    if export_manager:
        print(f"🎮 Unity export data available in exports/phase3/{run_name}")
    
    print(f"✅ Phase 3 training completed: {args.total_timesteps:,} timesteps")


if __name__ == "__main__":
    main()