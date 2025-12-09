#!/usr/bin/env python3
"""
Evaluate Terminal Specialist with 360° LOS Frame Config

Uses the proper eval_360_los.yaml config that matches training.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from stable_baselines3 import PPO
from environment import InterceptEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model.zip')
    parser.add_argument('--config', type=str, default='configs/eval_360_los.yaml', help='Config file')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print(f"Model: {args.model}")
    print(f"Observation mode: {config['environment'].get('observation_mode', 'world')}")
    print(f"Episodes: {args.episodes}, Seed: {args.seed}")
    print()

    # Load model
    model = PPO.load(args.model)

    # Create environment with proper config
    env = InterceptEnvironment(config['environment'])

    # Frame stacking
    n_frames = 4

    min_distances = []
    successes = 0
    rewards_list = []

    # Track by quadrant
    quadrant_results = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}

    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        obs, info = env.reset()

        # Get azimuth for quadrant tracking
        mis_pos = env.missile_state['position']
        int_pos = env.interceptor_state['position']
        rel_pos = np.array(mis_pos) - np.array(int_pos)
        azimuth = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))

        if 0 <= azimuth < 90:
            quadrant = 'Q1'
        elif 90 <= azimuth <= 180:
            quadrant = 'Q2'
        elif -180 <= azimuth < -90:
            quadrant = 'Q3'
        else:
            quadrant = 'Q4'

        # Initialize frame stack
        frame_stack = deque([obs.copy() for _ in range(n_frames)], maxlen=n_frames)

        min_dist = float('inf')
        total_reward = 0

        for step in range(2000):
            stacked_obs = np.concatenate(list(frame_stack))
            action, _ = model.predict(stacked_obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            frame_stack.append(obs.copy())
            total_reward += reward

            # Track min distance
            dist = np.linalg.norm(
                np.array(env.missile_state['position']) -
                np.array(env.interceptor_state['position'])
            )
            min_dist = min(min_dist, dist)

            # Debug: print distance every 100 steps for first episode
            if ep == 0 and step % 100 == 0:
                print(f"  step {step}: dist={dist:.1f}m")

            if done or truncated:
                if ep < 3:  # Debug first 3 episodes
                    mis_z = env.missile_state['position'][2]
                    int_z = env.interceptor_state['position'][2]
                    print(f"  -> Ended step {step}: done={done} trunc={truncated} dist={dist:.1f}m min={min_dist:.1f}m | mis_z={mis_z:.1f} int_z={int_z:.1f}")
                break

        min_distances.append(min_dist)
        rewards_list.append(total_reward)
        quadrant_results[quadrant].append(min_dist)

        success = min_dist < 50.0
        if success:
            successes += 1

        status = "✓" if success else "✗"
        print(f"Ep {ep+1:2d}: {quadrant} az={azimuth:+6.1f}° -> {min_dist:6.1f}m {status}")

    env.close()

    # Summary
    print()
    print("=" * 60)
    print("TERMINAL STANDALONE (360° LOS FRAME) RESULTS")
    print("=" * 60)
    print(f"Success Rate: {successes}/{args.episodes} = {successes/args.episodes*100:.1f}%")
    print(f"Mean Min Distance: {np.mean(min_distances):.1f}m ± {np.std(min_distances):.1f}m")
    print(f"Median: {np.median(min_distances):.1f}m")
    print(f"Best: {np.min(min_distances):.1f}m, Worst: {np.max(min_distances):.1f}m")
    print(f"Sub-50m: {sum(1 for d in min_distances if d < 50)}/{args.episodes}")
    print(f"Sub-100m: {sum(1 for d in min_distances if d < 100)}/{args.episodes}")
    print()
    print("BY QUADRANT:")
    for q, dists in quadrant_results.items():
        if len(dists) > 0:
            sr = sum(1 for d in dists if d < 50) / len(dists) * 100
            print(f"  {q}: n={len(dists):2d}, success={sr:5.1f}%, median={np.median(dists):6.1f}m")
    print()
    print(f"Mean Reward: {np.mean(rewards_list):.1f}")


if __name__ == "__main__":
    main()