#!/usr/bin/env python3
"""
Phase-1 PPO Training Script for AegisIntercept
Simple training script using CleanRL PPO implementation for the 2D missile intercept environment.
"""

import os
import time
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from aegis_intercept.envs import Aegis2DInterceptEnv


def make_env(env_id: str, idx: int, enable_render: bool) -> Callable:
    """Create environment factory function."""
    def thunk():
        if env_id == "Aegis2D":
            render_mode = "human" if enable_render and idx == 0 else None
            env = Aegis2DInterceptEnv(render_mode=render_mode)
        else:
            env = gym.make(env_id)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize neural network layer."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Agent with continuous action space."""
    
    def __init__(self, envs):
        super().__init__()
        
        # Shared feature extractor
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Actor network (mean)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        
        # Actor network (log std)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def train_ppo(enable_visualization=False, save_interval=50, resume_from_checkpoint=True):
    """Main training function."""
    
    # Hyperparameters
    exp_name = "aegis_ppo_phase1"
    seed = 1
    torch_deterministic = True
    cuda = torch.cuda.is_available()
    
    # Environment parameters
    env_id = "Aegis2D"
    num_envs = 8
    num_steps = 256  # Increased rollout length for better training
    total_timesteps = 250000  # Increased for better convergence
    
    # PPO parameters
    learning_rate = 3e-4
    anneal_lr = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 8  # Increased for better gradient estimates
    update_epochs = 4
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.01  # Maintain exploration
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    # Derived parameters
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size
    
    # Setup complete
    
    # Seeding
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    
    # Environment setup
    envs = gym.vector.SyncVectorEnv([
        make_env(env_id, i, enable_visualization) for i in range(num_envs)
    ])
    
    # Agent setup
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    # Load checkpoint if resuming
    start_iteration = 1
    if resume_from_checkpoint:
        model_path = "models/phase1_ppo.pt"
        if os.path.exists(model_path):
            print(f"Loading checkpoint from {model_path}")
            agent.load_state_dict(torch.load(model_path, map_location=device))
            # Find the latest checkpoint to determine iteration
            checkpoints = [f for f in os.listdir("models") if f.startswith("checkpoint_iter_") and f.endswith(".pt")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))
                start_iteration = int(latest_checkpoint.split("_")[2].split(".")[0]) + 1
                print(f"Resuming from iteration {start_iteration}")
        else:
            print("No checkpoint found, starting fresh training")
    
    # Storage setup
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    
    # Training loop
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    
    print(f"Starting training for {num_iterations} iterations ({total_timesteps} total timesteps)")
    
    for iteration in range(start_iteration, num_iterations + 1):
        # Annealing the rate if instructed to do so
        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Rollout phase
        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            # Get action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            # Execute action
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            # Render if visualization enabled
            if enable_visualization:
                envs.render()
            
            # Logging
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        episode_reward = info['episode']['r']
                        episode_length = info['episode']['l']
                        print(f"Episode: reward={episode_reward:.2f}, length={episode_length}, global_step={global_step}")
        
        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        
        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -clip_coef, clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
        
        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Get the final values from the last minibatch
        loss_val = loss.item()
        pg_loss_val = pg_loss.item()
        v_loss_val = v_loss.item()
        entropy_val = entropy_loss.item()
        
        print(f"Iteration {iteration}/{num_iterations}")
        print(f"  SPS: {int(global_step / (time.time() - start_time))}")
        print(f"  Loss: {loss_val:.4f} | Policy: {pg_loss_val:.4f} | Value: {v_loss_val:.4f}")
        print(f"  Entropy: {entropy_val:.4f} | Explained Var: {explained_var:.4f}")
        print(f"  Avg Reward: {rewards.mean().item():.3f} | Avg Value: {values.mean().item():.3f}")
        print("")
        
        # Save checkpoint at specified intervals
        if iteration % save_interval == 0:
            checkpoint_path = f"models/checkpoint_iter_{iteration}.pt"
            os.makedirs("models", exist_ok=True)
            torch.save({
                'iteration': iteration,
                'global_step': global_step,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), "models/phase1_ppo.pt")
    print("Model saved to models/phase1_ppo.pt")
    
    envs.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO agent for AegisIntercept Phase-1")
    parser.add_argument("--visualize", action="store_true", 
                       help="Enable real-time visualization (slows down training)")
    parser.add_argument("--save-interval", type=int, default=50,
                       help="Save checkpoint every N iterations (default: 50)")
    
    args = parser.parse_args()
    
    print(f"Starting training with visualization={'ON (slower)' if args.visualize else 'OFF (headless)'}")
    print(f"Checkpoints will be saved every {args.save_interval} iterations")
    
    train_ppo(enable_visualization=args.visualize, save_interval=args.save_interval, resume_from_checkpoint=True)