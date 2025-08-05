#!/usr/bin/env python3
"""Test script for episode logging integration."""

import numpy as np
from pathlib import Path
import json

from fast_sim_env import make_fast_sim_env


def test_episode_logging():
    """Test the episode logging functionality."""
    print("Testing episode logging integration...")
    
    # Create environment with logging enabled and explicit 3D config
    config = {
        'environment': {
            'num_missiles': 1,
            'num_interceptors': 1,
            'max_episode_steps': 100
        },
        'spawn': {
            'missile_spawn_area': [[-100, -100, 200], [100, 100, 500]],
            'interceptor_spawn_area': [[400, 400, 50], [600, 600, 200]],
            'target_area': [[800, 800, 0], [1000, 1000, 10]]
        }
    }
    
    from fast_sim_env import FastSimEnv
    env = FastSimEnv(
        config=config,
        scenario_name="easy",
        enable_episode_logging=True,
        episode_log_dir="test_runs"
    )
    
    # Run a few episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        obs, info = env.reset(seed=42 + episode)
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            # Random action for testing
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
        
        print(f"  Steps: {step_count}")
        print(f"  Final reward: {reward:.3f}")
        if "episode" in info:
            print(f"  Episode info: {info['episode']}")
    
    # Check that logs were created
    log_dir = Path("test_runs")
    if log_dir.exists():
        # Find the most recent run directory
        run_dirs = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if run_dirs:
            latest_run = run_dirs[-1]
            print(f"\nLogs created in: {latest_run}")
            
            # Check manifest
            manifest_path = latest_run / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                print(f"Manifest contains {len(manifest['episodes'])} episodes")
                print(f"Coordinate frame: {manifest['coord_frame']}")
                print(f"Timestep: {manifest['dt_nominal']}s")
            
            # Check episode files
            episode_files = list(latest_run.glob("ep_*.jsonl"))
            print(f"Found {len(episode_files)} episode files")
            
            if episode_files:
                # Sample first episode
                with open(episode_files[0]) as f:
                    lines = f.readlines()
                    print(f"First episode has {len(lines)} timesteps")
                    
                    # Check header
                    header = json.loads(lines[0])
                    if "meta" in header:
                        print(f"Episode metadata: {header['meta']}")
                    
                    # Check a sample timestep
                    if len(lines) > 10:
                        sample = json.loads(lines[10])
                        print(f"Sample timestep at t={sample['t']}s:")
                        if "agents" in sample:
                            for agent_id, agent_data in sample["agents"].items():
                                print(f"  {agent_id}: pos={agent_data['p']}, status={agent_data['status']}")
    
    print("\nEpisode logging test completed!")


if __name__ == "__main__":
    test_episode_logging()