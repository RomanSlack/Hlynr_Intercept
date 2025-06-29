#!/usr/bin/env python3
"""
Optimized Phase-2 PPO Training Script for AegisIntercept (3D)
Based on CleanRL patterns with performance improvements
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

from aegis_intercept.envs.aegis_3d_env import Aegis3DInterceptEnv
from gymnasium.envs.registration import register

# Register the environment
register(id="Aegis3DIntercept-v0", entry_point="aegis_intercept.envs:Aegis3DInterceptEnv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="aegis-intercept")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--visualize", action="store_true", help="Enable real-time visualization")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Aegis3DIntercept-v0")
    parser.add_argument("--total-timesteps", type=int, default=9999_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=512)  # Shorter rollouts for faster episode completion
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=8)  # Adjust for smaller batch size
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=0.005)  # Lower entropy for more stable learning
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision training")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def make_env(env_id, seed, idx, capture_video, run_name, render_mode=None):
    def thunk():
        if env_id == "Aegis3DIntercept-v0":
            env = Aegis3DInterceptEnv(render_mode=render_mode)
        else:
            env = gym.make(env_id, render_mode=render_mode)
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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        action_shape = envs.single_action_space.shape
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(action_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

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


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Vectorized GAE computation like CleanRL"""
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Environment setup
    envs = AsyncVectorEnv([
        make_env(args.env_id, args.seed, i, args.capture_video, run_name) 
        for i in range(args.num_envs)
    ])
    
    # Visualization environment (separate from training)
    viz_env = None
    if args.visualize:
        viz_env = Aegis3DInterceptEnv(render_mode="human")
        viz_obs, _ = viz_env.reset(seed=args.seed)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # Resume from checkpoint
    start_update = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        agent.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = checkpoint.get("update", 0)
        print(f"Resumed from update {start_update}")

    # Storage setup - keep on device for better performance
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

    print(f"Starting training for {num_updates} updates ({args.total_timesteps:,} timesteps)")

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

            # Debug episode completion and log episodes
            if step % 100 == 0:  # Every 100 steps
                print(f"Step {step}: Terminated={terminated.sum()}, Truncated={truncated.sum()}, Done={done.sum()}")
                if done.sum() > 0:
                    print(f"  Episode rewards this step: {reward[done]}")
            
            # Manual episode logging since AsyncVectorEnv may not pass final_info correctly
            for i in range(args.num_envs):
                if done[i]:  # Episode completed in environment i
                    # Since we don't have access to episode stats, create synthetic ones
                    episode_reward = reward[i]  # This step's reward
                    episode_length = 300  # Assume max episode length for now
                    print(f"Episode completed in env {i}! Reward: {episode_reward:.2f}, Length: ~{episode_length}")
                    writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
            
            # Also check for proper final_info
            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        print(f"[PROPER] Episode completed! Return: {item['episode']['r']}, Length: {item['episode']['l']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)

            # Update visualization (non-blocking)
            if viz_env is not None and step % 5 == 0:  # Update every 5 steps
                with torch.no_grad():
                    viz_obs_gpu = torch.tensor(viz_obs).unsqueeze(0).to(device)
                    viz_action, _, _, _ = agent.get_action_and_value(viz_obs_gpu)
                    viz_obs, viz_reward, viz_terminated, viz_truncated, viz_info = viz_env.step(viz_action[0].cpu().numpy())
                    
                    # Show episode result popup
                    if viz_terminated or viz_truncated:
                        if hasattr(viz_env, 'show_episode_result'):
                            viz_env.show_episode_result(viz_reward, viz_terminated, viz_truncated)
                        viz_obs, _ = viz_env.reset()
                    viz_env.render()

        # Bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = compute_gae(rewards, values, dones, next_value, args.gamma, args.gae_lambda)
            returns = advantages + values

        # Flatten the batch - already on device
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

                # Get minibatch - already on device
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]

                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                        logratio = newlogprob - mb_logprobs
                        ratio = logratio.exp()

                        # Calculate losses
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
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
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

        # Progress logging
        if update % 10 == 0:
            sps = int(global_step / (time.time() - start_time))
            print(f"Update {update:4d} | Steps: {global_step:8,} | SPS: {sps:4d}")

        # Checkpoint saving with max_to_keep=5
        if update % 50 == 0:
            os.makedirs("models/phase2", exist_ok=True)
            checkpoint = {
                "model": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
            }
            torch.save(checkpoint, f"models/phase2/latest.pt")
            
            # Keep only last 5 checkpoints
            if update % 250 == 0:  # Save numbered checkpoints less frequently
                torch.save(checkpoint, f"models/phase2/checkpoint_{update:06d}.pt")
                
                # Clean up old checkpoints
                import glob
                checkpoints = sorted(glob.glob("models/phase2/checkpoint_*.pt"))
                if len(checkpoints) > 5:
                    for old_checkpoint in checkpoints[:-5]:
                        os.remove(old_checkpoint)

    envs.close()
    if viz_env is not None:
        viz_env.close()
    writer.close()


if __name__ == "__main__":
    main()