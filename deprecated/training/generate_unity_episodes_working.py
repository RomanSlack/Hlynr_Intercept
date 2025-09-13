#!/usr/bin/env python3
"""
Working script to generate JSONL episodes for Unity visualization.
Uses the actual working pattern from test_episode_logging.py
"""

import numpy as np
from pathlib import Path
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fast_sim_env import FastSimEnv
from config import get_config
from scenarios import get_scenario_loader


def generate_unity_episodes(
    checkpoint_path="checkpoints/phase4_easy_final.zip",
    vecnorm_path="checkpoints/vec_normalize_final.pkl", 
    scenario="easy",
    num_episodes=5,
    output_dir="unity_episodes"
):
    """Generate episodes with trained RL policy for Unity visualization."""
    
    print(f"🎯 Generating {num_episodes} episodes for Unity...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🏆 Model: {checkpoint_path}")
    print(f"📊 Scenario: {scenario}")
    
    # Check if model exists
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        print(f"❌ ERROR: Model checkpoint not found at {checkpoint}")
        return False
    
    # Load config and scenario
    try:
        config = get_config()
        scenario_loader = get_scenario_loader()
        scenario_config = scenario_loader.create_environment_config(scenario, config._config)
    except:
        # Use simple config if the full system doesn't work
        scenario_config = {
            'environment': {
                'num_missiles': 1,
                'num_interceptors': 1,
                'max_episode_steps': 1000
            },
            'spawn': {
                'missile_spawn_area': [[-100, -100, 200], [100, 100, 500]],
                'interceptor_spawn_area': [[400, 400, 50], [600, 600, 200]], 
                'target_area': [[800, 800, 0], [1000, 1000, 10]]
            }
        }
    
    # Create environment with episode logging enabled
    env = FastSimEnv(
        config=scenario_config,
        scenario_name=scenario,
        enable_episode_logging=True,  # This is the key!
        episode_log_dir=output_dir
    )
    
    # Wrap in DummyVecEnv for model compatibility
    vec_env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize if available
    vecnorm_file = Path(vecnorm_path)
    if vecnorm_file.exists():
        print(f"📊 Loading VecNormalize from {vecnorm_file}")
        try:
            vec_env = VecNormalize.load(str(vecnorm_file), vec_env)
            vec_env.training = False  # Set to inference mode
            vec_env.norm_reward = False
        except Exception as e:
            print(f"⚠️  VecNormalize loading failed: {e}")
    
    # Load trained model
    print(f"🧠 Loading trained RL policy...")
    try:
        model = PPO.load(str(checkpoint), env=vec_env)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Generate episodes
    print(f"\n🚀 Starting episode generation...")
    successful_episodes = 0
    
    for episode_num in range(num_episodes):
        print(f"\n📺 Episode {episode_num + 1}/{num_episodes}...")
        
        try:
            obs = vec_env.reset()
            done = False
            step_count = 0
            total_reward = 0
            
            while not done and step_count < 2000:  # Safety limit
                # Get action from trained RL policy
                action, _states = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, info = vec_env.step(action)
                total_reward += reward[0]
                step_count += 1
                
                # Check if done
                done = done[0]
                
                # Progress indicator
                if step_count % 200 == 0:
                    print(f"    Step {step_count}... (reward: {total_reward:.2f})")
            
            # Episode completed
            outcome = "completed"
            miss_distance = "unknown"
            if info and len(info) > 0 and isinstance(info[0], dict):
                outcome = info[0].get('outcome', 'completed')
                miss_distance = info[0].get('miss_distance_m', 'unknown')
            
            print(f"  ✅ Episode {episode_num + 1} finished:")
            print(f"     Steps: {step_count}")
            print(f"     Total reward: {total_reward:.3f}")
            print(f"     Outcome: {outcome}")
            print(f"     Miss distance: {miss_distance}")
            
            successful_episodes += 1
            
        except Exception as e:
            print(f"  ❌ Episode {episode_num + 1} failed: {e}")
    
    # Check results
    output_path = Path(output_dir)
    if output_path.exists():
        # Find the most recent run directory
        run_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if run_dirs:
            latest_run = run_dirs[-1]
            print(f"\n🎉 SUCCESS! Episodes generated in: {latest_run}")
            
            # Check what was created
            manifest_path = latest_run / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                print(f"📋 Manifest: {len(manifest['episodes'])} episodes")
                print(f"🌍 Coordinate frame: {manifest['coord_frame']}")
                print(f"⏱️  Sampling rate: {1/manifest['dt_nominal']:.0f} Hz")
            
            # List episode files
            episode_files = list(latest_run.glob("ep_*.jsonl"))
            print(f"📁 Episode files: {len(episode_files)} JSONL files")
            
            if episode_files:
                # Show sample from first episode
                with open(episode_files[0]) as f:
                    lines = f.readlines()
                    print(f"📊 Sample episode: {len(lines)} timesteps")
                    
                    if len(lines) > 10:
                        sample = json.loads(lines[10])
                        print(f"🎯 Sample data at t={sample['t']:.2f}s:")
                        if "agents" in sample:
                            for agent_id, agent_data in sample["agents"].items():
                                pos = agent_data['p']
                                print(f"   {agent_id}: pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
            
            print(f"\n🎮 Ready for Unity! Use files in: {latest_run}")
            print(f"🔥 Generated {successful_episodes}/{num_episodes} episodes successfully")
            return True
    else:
        print(f"❌ No output directory created")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Unity episodes with trained RL policy')
    parser.add_argument('--checkpoint', default='checkpoints/phase4_easy_final.zip',
                        help='Path to model checkpoint')
    parser.add_argument('--vecnorm', default='checkpoints/vec_normalize_final.pkl',
                        help='Path to VecNormalize stats')  
    parser.add_argument('--scenario', default='easy',
                        choices=['easy', 'medium', 'hard'],
                        help='Scenario difficulty')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to generate')
    parser.add_argument('--output-dir', default='unity_episodes',
                        help='Output directory for episodes')
    
    args = parser.parse_args()
    
    success = generate_unity_episodes(
        checkpoint_path=args.checkpoint,
        vecnorm_path=args.vecnorm,
        scenario=args.scenario,
        num_episodes=args.episodes,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n🎉 All done! Episodes ready for Unity visualization!")
    else:
        print("\n❌ Episode generation failed. Check errors above.")