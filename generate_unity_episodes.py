#!/usr/bin/env python3
"""
Simple script to generate JSONL episodes for Unity visualization.
Runs the RL policy on scenarios and saves the trajectories.
"""

import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "phase4_rl"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fast_sim_env import FastSimEnv
from episode_logger import EpisodeLogger
from scenarios import get_scenario_loader
from config import get_config


def generate_episodes(
    checkpoint_path="checkpoints/best_model/best_model.zip",
    vecnorm_path="checkpoints/vec_normalize.pkl", 
    scenario="easy",
    num_episodes=5,
    output_dir="unity_episodes"
):
    """Generate episodes with RL policy and save to JSONL."""
    
    print(f"Generating {num_episodes} episodes for scenario: {scenario}")
    print(f"Output directory: {output_dir}")
    
    # Setup paths
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint}")
        return
    
    # Load config and scenario
    config = get_config()
    scenario_loader = get_scenario_loader()
    scenario_config = scenario_loader.create_environment_config(scenario, config)
    
    # Create environment
    def make_env():
        return FastSimEnv(
            config=scenario_config,
            scenario_name=scenario,
            enable_episode_logging=True,  # Enable logging
            episode_log_dir=output_dir
        )
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize if available
    vecnorm = Path(vecnorm_path)
    if vecnorm.exists():
        print(f"Loading VecNormalize from {vecnorm}")
        env = VecNormalize.load(str(vecnorm), env)
        env.training = False  # Set to inference mode
        env.norm_reward = False
    
    # Load trained model
    print(f"Loading model from {checkpoint}")
    model = PPO.load(str(checkpoint), env=env)
    
    # Run episodes
    for episode_num in range(num_episodes):
        print(f"\nRunning episode {episode_num + 1}/{num_episodes}...")
        
        obs = env.reset()
        done = False
        step_count = 0
        
        while not done:
            # Get action from policy (deterministic for consistent replay)
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"  Step {step_count}...")
        
        # Get episode outcome from info
        outcome = "completed"
        if info and len(info) > 0:
            outcome = info[0].get('outcome', 'completed')
            miss_distance = info[0].get('miss_distance_m', 0.0)
            print(f"  Episode {episode_num + 1} finished: {outcome}, miss distance: {miss_distance:.2f}m")
        else:
            print(f"  Episode {episode_num + 1} finished: {outcome}")
    
    print(f"\n✅ Generated {num_episodes} episodes in {output_dir}/")
    print(f"Check the manifest.json and ep_XXXXXX.jsonl files")


def examine_episode(episode_path):
    """Examine a generated episode file."""
    import json
    
    print(f"\nExamining episode: {episode_path}")
    print("-" * 60)
    
    with open(episode_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    
    # Parse first line (header)
    header = json.loads(lines[0])
    print(f"Episode ID: {header['meta']['ep_id']}")
    print(f"Scenario: {header['meta']['scenario']}")
    print(f"Coordinate frame: {header['meta']['coord_frame']}")
    print(f"Timestep: {header['meta']['dt_nominal']}s ({1/header['meta']['dt_nominal']:.0f} Hz)")
    
    # Check a sample timestep
    if len(lines) > 10:
        sample = json.loads(lines[10])
        print(f"\nSample timestep at t={sample['t']:.2f}s:")
        
        if 'interceptor_0' in sample['agents']:
            interceptor = sample['agents']['interceptor_0']
            print(f"  Interceptor position: {interceptor['p']}")
            print(f"  Interceptor velocity: {interceptor['v']}")
            print(f"  Interceptor fuel: {interceptor.get('fuel_kg', 'N/A')} kg")
            print(f"  Interceptor action: {interceptor.get('u', 'N/A')}")
        
        if 'threat_0' in sample['agents']:
            threat = sample['agents']['threat_0']
            print(f"  Threat position: {threat['p']}")
            print(f"  Threat velocity: {threat['v']}")
    
    # Check last line (summary)
    if lines[-1].strip():
        summary_line = json.loads(lines[-1])
        if 'summary' in summary_line:
            summary = summary_line['summary']
            print(f"\nEpisode summary:")
            print(f"  Outcome: {summary['outcome']}")
            print(f"  Duration: {summary['episode_duration']:.2f}s")
            print(f"  Miss distance: {summary.get('miss_distance_m', 'N/A')}m")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate JSONL episodes for Unity')
    parser.add_argument('--checkpoint', default='checkpoints/best_model/best_model.zip',
                        help='Path to model checkpoint')
    parser.add_argument('--vecnorm', default='checkpoints/vec_normalize.pkl',
                        help='Path to VecNormalize stats')
    parser.add_argument('--scenario', default='easy',
                        choices=['easy', 'medium', 'hard'],
                        help='Scenario difficulty')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to generate')
    parser.add_argument('--output-dir', default='unity_episodes',
                        help='Output directory for episodes')
    parser.add_argument('--examine', type=str,
                        help='Path to episode file to examine')
    
    args = parser.parse_args()
    
    if args.examine:
        examine_episode(args.examine)
    else:
        generate_episodes(
            checkpoint_path=args.checkpoint,
            vecnorm_path=args.vecnorm,
            scenario=args.scenario,
            num_episodes=args.episodes,
            output_dir=args.output_dir
        )