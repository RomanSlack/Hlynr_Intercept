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


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str) -> Callable:
    """Create environment factory function."""
    def thunk():
        if env_id == "Aegis2D":
            env = Aegis2DInterceptEnv(render_mode="human" if capture_video and idx == 0 else None)
        else:
            env = gym.make(env_id)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
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


def train_ppo():
    """Main training function."""
    
    # Hyperparameters
    exp_name = "aegis_ppo_phase1"
    seed = 1
    torch_deterministic = True
    cuda = torch.cuda.is_available()
    
    # Environment parameters
    env_id = "Aegis2D"
    num_envs = 8
    num_steps = 128
    total_timesteps = 100000
    
    # PPO parameters
    learning_rate = 3e-4
    anneal_lr = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 4
    update_epochs = 4
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    # Derived parameters
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size
    
    # Setup
    run_name = f"{exp_name}__{seed}__{int(time.time())}"
    
    # Seeding
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    
    # Environment setup
    envs = gym.vector.SyncVectorEnv([
        make_env(env_id, i, False, run_name) for i in range(num_envs)
    ])
    
    # Agent setup
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
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
    
    for iteration in range(1, num_iterations + 1):
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
            
            # Logging
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        
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
        
        print(f"Iteration {iteration}")
        print(f"  SPS: {int(global_step / (time.time() - start_time))}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Policy Loss: {pg_loss.item():.4f}")
        print(f"  Value Loss: {v_loss.item():.4f}")
        print(f"  Entropy: {entropy_loss.item():.4f}")
        print(f"  Explained Variance: {explained_var:.4f}")
        print("")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), "models/phase1_ppo.pt")
    print("Model saved to models/phase1_ppo.pt")
    
    envs.close()


if __name__ == "__main__":
    train_ppo()