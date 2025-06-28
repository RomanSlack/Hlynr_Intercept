#!/usr/bin/env python3
"""
Phase-2 PPO Training Script for AegisIntercept (3D)
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv as SubprocVecEnv
from distutils.util import strtobool
import matplotlib.pyplot as plt
from collections import deque

from aegis_intercept.envs.aegis_3d_env import Aegis3DInterceptEnv

def make_env(env_id, seed, idx, capture_video, run_name, render_mode=None):
    def thunk():
        env = Aegis3DInterceptEnv(render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed + idx)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=True`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanrl", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--visualize", action="store_true", help="Enable real-time visualization")
    parser.add_argument("--resume", type=str, default=None, help="the path to the model checkpoint to resume from")
    parser.add_argument("--reset", action="store_true", help="Delete all existing checkpoints and restart training from scratch")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Aegis3DIntercept-v0", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    # Handle reset flag - delete existing checkpoints
    if args.reset:
        import shutil
        if os.path.exists("models/phase2"):
            shutil.rmtree("models/phase2")
            print("Deleted existing checkpoints. Starting fresh training.")
        if os.path.exists("runs"):
            for run_dir in os.listdir("runs"):
                if args.env_id in run_dir:
                    shutil.rmtree(os.path.join("runs", run_dir))
                    print(f"Deleted tensorboard logs: {run_dir}")
        args.resume = None  # Ensure we don't try to resume

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        "|param|value|\n|-|-|\n%s" % ("|\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: Using CPU instead of GPU - training will be slower")

    # env setup
    envs = SubprocVecEnv([
        make_env(args.env_id, args.seed, i, args.capture_video, run_name, render_mode=None) for i in range(args.num_envs)
    ])
    
    # Setup visualization environment if requested
    viz_env = None
    viz_obs = None
    viz_step_counter = 0
    viz_render_every = 5  # Render every 5 steps to maintain performance
    if args.visualize:
        viz_env = Aegis3DInterceptEnv(render_mode="human")
        viz_obs, _ = viz_env.reset(seed=args.seed)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.resume:
        agent.load_state_dict(torch.load(args.resume))

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    
    # Episode tracking for progress monitoring
    episode_returns = []
    episode_lengths = []
    last_progress_update = 0
    
    # Real-time score graphing setup
    score_graph = None
    score_history = deque(maxlen=100)  # Keep last 100 episodes
    # Always show score graph unless explicitly visualizing 3D
    if not args.visualize:
        plt.ion()
        plt.rcParams['figure.raise_window'] = True  # Try to bring window to front
        score_graph = plt.figure(figsize=(10, 6))
        score_graph.suptitle("Training Progress - Waiting for Episodes...")
        
        # Create initial empty plot
        plt.subplot(111)
        plt.xlabel("Episode")
        plt.ylabel("Score (Return)")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Neutral')
        plt.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Perfect (+1.0)')
        plt.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Worst (-1.0)')
        plt.legend()
        plt.ylim(-1.5, 1.5)
        plt.xlim(0, 10)
        
        # Force window to show and come to front
        plt.show(block=False)
        plt.pause(0.1)
        try:
            # Try to bring window to front (platform dependent)
            score_graph.canvas.manager.window.wm_attributes('-topmost', 1)
            score_graph.canvas.manager.window.wm_attributes('-topmost', 0)
        except:
            try:
                # Alternative for different window managers
                score_graph.canvas.manager.window.raise_()
            except:
                pass  # If all fails, just continue
        plt.draw()
        score_graph.canvas.flush_events()
        print("=== SCORE GRAPH WINDOW OPENED ===")
        print("Look for a matplotlib window titled 'Training Progress'")
        print("If you don't see it, check your taskbar or run: 'python -c \"import matplotlib; print(matplotlib.get_backend())\"'")
        print("=======================================")
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            # Debug: Check if ANY episodes are completing
            if global_step % 1000 == 0:  # Every 1k steps instead of 10k
                print(f"[DEBUG] Step {global_step}: Terminated={terminated.sum()}, Truncated={truncated.sum()}, Done={done.sum()}")
                print(f"[DEBUG] Rewards range: {reward.min():.3f} to {reward.max():.3f}")
                if 'final_info' in info:
                    print(f"[DEBUG] Final info available: {len([i for i in info['final_info'] if i is not None])}")
            
            # Manual episode completion handling since RecordEpisodeStatistics isn't working properly
            if done.sum() > 0:
                print(f"[TRAINING] Episode completions detected! Done={done.sum()}, Step={global_step}")
                
                # Check if we have episode stats in the regular info dict
                if 'episode' in info:
                    # Episodes completed this step - extract their statistics
                    completed_count = 0
                    for i in range(len(done)):
                        if done[i]:  # This environment completed an episode
                            # Create fake final_info structure for our processing
                            if not hasattr(info, 'get'):
                                episode_return = -0.3  # Default timeout reward
                                episode_length = 50     # Default episode length
                            else:
                                episode_return = reward[i]  # Use the reward from this step
                                episode_length = 50         # We know episodes are 50 steps
                            
                            completed_count += 1
                            
                            # Manually add to our episode tracking
                            episode_returns.append(float(episode_return))
                            episode_lengths.append(int(episode_length))
                            score_history.append(float(episode_return))
                            
                            print(f"[TRAINING] Manual episode {completed_count}: return={episode_return:.3f}, length={episode_length}")
                            
                            # Update graph immediately
                            if score_graph is not None:
                                plt.figure(score_graph.number)
                                plt.clf()
                                
                                if len(score_history) >= 1:
                                    episodes = list(range(1, len(score_history) + 1))
                                    scores = list(score_history)
                                    
                                    # Plot individual episode scores as light dots
                                    plt.scatter(episodes, scores, alpha=0.4, s=10, color='lightblue', label='Individual Episodes')
                                    
                                    # Plot moving average as main line
                                    if len(scores) >= 3:
                                        window_size = min(10, max(3, len(scores) // 3))
                                        moving_avg = []
                                        for j in range(len(scores)):
                                            start_idx = max(0, j - window_size + 1)
                                            avg = np.mean(scores[start_idx:j+1])
                                            moving_avg.append(avg)
                                        plt.plot(episodes, moving_avg, 'r-', linewidth=3, label=f'Moving Average ({window_size})')
                                        current_avg = moving_avg[-1]
                                    else:
                                        current_avg = np.mean(scores)
                                    
                                    # Reference lines
                                    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral (0)')
                                    plt.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Perfect (+1.0)')
                                    plt.axhline(y=-1, color='red', linestyle='--', alpha=0.7, label='Worst (-1.0)')
                                    
                                    # Title with key info
                                    plt.title(f'Training Progress | Episodes: {len(scores)} | Avg Score: {current_avg:.3f} | Steps: {global_step:,}', fontsize=12)
                                    
                                    # Smart axis limits
                                    min_score = min(scores)
                                    max_score = max(scores)
                                    margin = max(0.1, abs(max_score - min_score) * 0.15)
                                    plt.ylim(min(min_score - margin, -1.3), max(max_score + margin, 1.3))
                                    plt.xlim(0.5, max(10, len(scores) + 0.5))
                                
                                plt.xlabel('Episode Number')
                                plt.ylabel('Episode Score (Return)')
                                plt.legend(loc='best')
                                plt.grid(True, alpha=0.3)
                                plt.tight_layout()
                                plt.draw()
                                score_graph.canvas.flush_events()
                                plt.pause(0.01)
                            
                            # Add to tensorboard
                            writer.add_scalar("charts/episodic_return", episode_return, global_step)
                            writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    
                    print(f"[TRAINING] Manually processed {completed_count} episodes")
            
            # Visualization: step the viz environment with the agent's action
            if args.visualize and viz_env is not None:
                # Get action for visualization environment using its current observation
                with torch.no_grad():
                    viz_obs_tensor = torch.Tensor(viz_obs).unsqueeze(0).to(device)
                    viz_action, _, _, _ = agent.get_action_and_value(viz_obs_tensor)
                    viz_action = viz_action[0].cpu().numpy()
                
                # Step the visualization environment
                viz_obs, viz_reward, viz_terminated, viz_truncated, viz_info = viz_env.step(viz_action)
                
                # Render at reduced frequency to maintain performance
                viz_step_counter += 1
                if viz_step_counter % viz_render_every == 0:
                    viz_env.render(viz_terminated and viz_reward > 0.5)  # Pass intercept success
                
                # Reset viz environment if episode ends
                if viz_terminated or viz_truncated:
                    viz_obs, _ = viz_env.reset(seed=args.seed)

            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        episode_return = item['episode']['r']
                        episode_length = item['episode']['l']
                        
                        # Track episodes for progress monitoring
                        episode_returns.append(episode_return)
                        episode_lengths.append(episode_length)
                        score_history.append(episode_return)
                        
                        # Update real-time score graph  
                        if score_graph is not None:
                            plt.figure(score_graph.number)
                            plt.clf()
                            
                            if len(score_history) >= 1:
                                episodes = list(range(1, len(score_history) + 1))
                                scores = list(score_history)
                                
                                # Plot individual episode scores as light dots
                                plt.scatter(episodes, scores, alpha=0.4, s=10, color='lightblue', label='Individual Episodes')
                                
                                # Plot moving average as main line
                                if len(scores) >= 3:
                                    window_size = min(10, max(3, len(scores) // 3))
                                    moving_avg = []
                                    for i in range(len(scores)):
                                        start_idx = max(0, i - window_size + 1)
                                        avg = np.mean(scores[start_idx:i+1])
                                        moving_avg.append(avg)
                                    plt.plot(episodes, moving_avg, 'r-', linewidth=3, label=f'Moving Average ({window_size})')
                                    current_avg = moving_avg[-1]
                                else:
                                    current_avg = np.mean(scores)
                                
                                # Reference lines
                                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral (0)')
                                plt.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Perfect (+1.0)')
                                plt.axhline(y=-1, color='red', linestyle='--', alpha=0.7, label='Worst (-1.0)')
                                
                                # Title with key info
                                plt.title(f'Training Progress | Episodes: {len(scores)} | Avg Score: {current_avg:.3f} | Training Steps: {global_step:,}', fontsize=12)
                                
                                # Smart axis limits
                                min_score = min(scores)
                                max_score = max(scores)
                                margin = max(0.1, abs(max_score - min_score) * 0.15)
                                plt.ylim(min(min_score - margin, -1.3), max(max_score + margin, 1.3))
                                plt.xlim(0.5, max(10, len(scores) + 0.5))
                                
                            else:
                                plt.title(f'Waiting for First Episode | Training Steps: {global_step:,}', fontsize=12)
                                plt.ylim(-1.3, 1.3)
                                plt.xlim(0, 10)
                                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                                plt.axhline(y=1, color='green', linestyle='--', alpha=0.7)
                                plt.axhline(y=-1, color='red', linestyle='--', alpha=0.7)
                            
                            plt.xlabel('Episode Number')
                            plt.ylabel('Episode Score (Return)')
                            plt.legend(loc='best')
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.draw()
                            score_graph.canvas.flush_events()
                            plt.pause(0.01)
                        
                        # Enhanced logging for training progress
                        if not args.visualize:  # More detailed output when not visualizing
                            print(f"Episode Complete | Step: {global_step:,} | Return: {episode_return:.2f} | Length: {episode_length} | SPS: {int(global_step / (time.time() - start_time))}")
                        else:
                            print(f"global_step={global_step}, episodic_return={episode_return}")
                            
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", episode_length, global_step)
                        break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/explained_variance", explained_var, global_step)
        
        # Enhanced progress reporting
        current_sps = int(global_step / (time.time() - start_time))
        if not args.visualize:
            # Show progress summary every 10 updates
            if update % 10 == 0 and len(episode_returns) > 0:
                recent_returns = episode_returns[-50:] if len(episode_returns) >= 50 else episode_returns
                avg_return = np.mean(recent_returns)
                avg_length = np.mean(episode_lengths[-50:] if len(episode_lengths) >= 50 else episode_lengths)
                print(f"Update {update:4d} | Steps: {global_step:7,} | Episodes: {len(episode_returns):4d} | Avg Return: {avg_return:+6.2f} | Avg Length: {avg_length:6.1f} | SPS: {current_sps:4d}")
            elif update % 5 == 0:  # Show basic progress every 5 updates
                if len(episode_returns) == 0:
                    print(f"Update {update:4d} | Steps: {global_step:7,} | Episodes: {len(episode_returns):4d} | SPS: {current_sps:4d} | [Waiting for first episode to complete...]")
                else:
                    print(f"Update {update:4d} | Steps: {global_step:7,} | Episodes: {len(episode_returns):4d} | SPS: {current_sps:4d}")
        else:
            print("SPS:", current_sps)
            
        writer.add_scalar("charts/SPS", current_sps, global_step)

        if update % 5 == 0:
            os.makedirs("models/phase2", exist_ok=True)
            torch.save(agent.state_dict(), f"models/phase2/ppo_{global_step}.pt")

    envs.close()
    if viz_env is not None:
        viz_env.close()
    writer.close()