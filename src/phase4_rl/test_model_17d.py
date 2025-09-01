#!/usr/bin/env python3
"""
Test the 17D radar model with proper observation space matching.
"""

import numpy as np
import json
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from .fast_sim_env import FastSimEnv
    from .config import get_config
    from .scenarios import get_scenario_loader
    from .radar_observations import Radar17DObservation
except ImportError:
    from fast_sim_env import FastSimEnv
    from config import get_config
    from scenarios import get_scenario_loader
    from radar_observations import Radar17DObservation


def test_17d_model(model_path="checkpoints_new/phase4_easy_final.zip",
                   vecnorm_path="checkpoints_new/vec_normalize_final.pkl",
                   num_episodes=3):
    """Test the 17D radar model with proper setup."""
    
    print(f"🧠 Loading 17D radar model: {model_path}")
    
    # Load configuration
    config = get_config()
    scenario_loader = get_scenario_loader()
    scenario_config = scenario_loader.create_environment_config("easy", config._config)
    
    # Create environment that forces 17D observations
    def make_env():
        # Create a radar environment that uses 17D observations
        from radar_env import RadarEnv
        env = RadarEnv(
            config=scenario_config,
            scenario_name="easy",
            num_missiles=1,
            num_interceptors=1,
            render_mode=None
        )
        # Force the environment to use 17D radar observations
        env.use_radar_17d = True
        env.radar_observer = Radar17DObservation(max_range=10000.0, max_velocity=1000.0)
        
        # Update observation space to exactly 17 dimensions
        from gymnasium import spaces
        env.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(17,), dtype=np.float32
        )
        return env
    
    vec_env = DummyVecEnv([make_env])
    
    # Try to load VecNormalize if it exists and works
    try:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"✅ VecNormalize loaded from: {vecnorm_path}")
    except Exception as e:
        print(f"⚠️  VecNormalize loading failed: {e}")
        print("   Proceeding without normalization...")
    
    # Load model
    model = PPO.load(model_path, env=vec_env)
    print(f"✅ Model loaded successfully!")
    print(f"   Model observation space: {model.observation_space}")
    print(f"   Environment observation space: {vec_env.observation_space}")
    
    # Run episodes
    results = []
    episode_data = []
    
    print(f"\n🚀 Running {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        print(f"\n📺 Episode {episode + 1}/{num_episodes}")
        
        obs = vec_env.reset()
        done = False
        step = 0
        total_reward = 0.0
        episode_steps = []
        
        # Get initial state for logging
        env_instance = vec_env.envs[0]
        
        while not done and step < 1000:
            # Get action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment  
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            
            # Log state for JSON output
            if hasattr(env_instance, 'missile_positions'):
                missile_pos = env_instance.missile_positions[0].tolist()
                interceptor_pos = env_instance.interceptor_positions[0].tolist()
                missile_vel = env_instance.missile_velocities[0].tolist()
                interceptor_vel = env_instance.interceptor_velocities[0].tolist()
                
                step_data = {
                    "t": round(step * 0.1, 2),
                    "agents": {
                        "threat_0": {
                            "p": [round(x, 3) for x in missile_pos],
                            "v": [round(x, 3) for x in missile_vel], 
                            "status": "active"
                        },
                        "interceptor_0": {
                            "p": [round(x, 3) for x in interceptor_pos],
                            "v": [round(x, 3) for x in interceptor_vel],
                            "status": "active"
                        }
                    }
                }
                episode_steps.append(step_data)
            
            step += 1
            done = done[0] if isinstance(done, (list, tuple)) else done
            
            if step % 100 == 0:
                print(f"  Step {step}: reward={total_reward:.3f}")
        
        # Episode complete
        episode_result = {
            "episode": episode + 1,
            "steps": step,
            "total_reward": total_reward,
            "outcome": "completed"
        }
        
        results.append(episode_result)
        episode_data.append(episode_steps)
        
        print(f"  ✅ Episode {episode + 1} complete:")
        print(f"     Steps: {step}")
        print(f"     Total reward: {total_reward:.3f}")
    
    # Save episode data as JSON
    output_dir = Path("test_17d_episodes")
    output_dir.mkdir(exist_ok=True)
    
    # Save summary
    summary_file = output_dir / "summary.json" 
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save individual episode data
    for i, episode_steps in enumerate(episode_data):
        episode_file = output_dir / f"episode_{i+1:03d}.jsonl"
        with open(episode_file, 'w') as f:
            for step_data in episode_steps:
                json.dump(step_data, f, separators=(',', ':'))
                f.write('\n')
    
    print(f"\n🎉 Testing complete!")
    print(f"📁 Episode data saved to: {output_dir}/")
    print(f"   - summary.json: Episode results")
    print(f"   - episode_*.jsonl: Step-by-step trajectory data")
    
    # Print final positions to verify scenario
    if episode_data and episode_data[0]:
        first_episode = episode_data[0]
        initial_threat_pos = first_episode[0]["agents"]["threat_0"]["p"]
        final_threat_pos = first_episode[-1]["agents"]["threat_0"]["p"]
        
        print(f"\n🎯 Trajectory Verification:")
        print(f"   Threat initial position: {initial_threat_pos}")
        print(f"   Threat final position: {final_threat_pos}")
        print(f"   Expected: missile falls from [0,0,300] toward [0,0,0]")
        
        if abs(initial_threat_pos[2] - 300) < 50:  # Started near 300m altitude
            print(f"   ✅ Correct starting position!")
        else:
            print(f"   ⚠️  Unexpected starting position")

if __name__ == "__main__":
    test_17d_model()