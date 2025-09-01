#!/usr/bin/env python3
"""
Test the fixed reward function and action interpretation.
"""

import numpy as np
from config import get_config, reset_config
from scenarios import get_scenario_loader, reset_scenario_loader
from radar_env import RadarEnv

def test_fixes():
    """Test if our fixes to the reward function and actions work correctly."""
    
    print("🔧 Testing Fixed Training Environment")
    print("=" * 50)
    
    # Load real scenario config
    reset_config()
    reset_scenario_loader()
    
    config_loader = get_config()
    scenario_loader = get_scenario_loader()
    scenario_config = scenario_loader.create_environment_config("easy", config_loader._config)
    
    # Create environment with our fixes
    env = RadarEnv(
        config=scenario_config,
        scenario_name="easy",
        num_missiles=1,
        num_interceptors=1,
        render_mode=None
    )
    
    print(f"✅ Environment created with observation space: {env.observation_space.shape}")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"✅ Environment reset successfully")
    
    # Check initial positions
    missile_pos = env.missile_positions[0]
    interceptor_pos = env.interceptor_positions[0]
    target_pos = env.target_positions[0]
    
    print(f"Initial positions:")
    print(f"  Missile: {missile_pos}")
    print(f"  Interceptor: {interceptor_pos}")  
    print(f"  Target: {target_pos}")
    
    # Test reward function
    initial_distance = np.linalg.norm(interceptor_pos - missile_pos)
    initial_reward = env._calculate_reward()
    print(f"\nInitial distance: {initial_distance:.1f}m")
    print(f"Initial reward: {initial_reward:.3f}")
    
    # Test action that should move interceptor TOWARD missile
    # Missile at [0,0,300], Interceptor at [500,500,100]
    # To intercept: need negative X, negative Y, positive Z thrust
    good_action = np.array([-0.5, -0.5, 0.3, 0.0, 0.0, 0.0])  # Move toward missile
    bad_action = np.array([0.5, 0.5, -0.3, 0.0, 0.0, 0.0])    # Move away from missile
    
    print(f"\n🧪 Testing GOOD action (should move toward missile):")
    print(f"Action: {good_action}")
    
    # Step with good action
    obs, reward_good, done, truncated, info = env.step(good_action)
    
    new_interceptor_pos = env.interceptor_positions[0]
    new_distance = np.linalg.norm(new_interceptor_pos - env.missile_positions[0])
    
    print(f"New interceptor position: {new_interceptor_pos}")
    print(f"New distance: {new_distance:.1f}m")
    print(f"Distance change: {new_distance - initial_distance:.3f}m")
    print(f"Reward: {reward_good:.3f}")
    
    # Check if interceptor moved in right direction
    pos_change = new_interceptor_pos - interceptor_pos
    print(f"Position change: {pos_change}")
    
    if pos_change[0] < 0 and pos_change[1] < 0:
        print("✅ CORRECT: Interceptor moved toward missile (negative X,Y)")
    else:
        print("❌ WRONG: Interceptor moved away from missile")
        
    if new_distance < initial_distance:
        print("✅ CORRECT: Distance decreased")
    else:
        print("❌ WRONG: Distance increased or stayed same")
        
    if reward_good > -1.0:  # Should be much better than old -0.1 time penalty
        print("✅ CORRECT: Reward is reasonable (not dominated by time penalty)")
    else:
        print("❌ WRONG: Reward still too negative")
    
    # Reset for bad action test
    env.reset(seed=42)
    
    print(f"\n🧪 Testing BAD action (should move away from missile):")
    print(f"Action: {bad_action}")
    
    obs, reward_bad, done, truncated, info = env.step(bad_action)
    
    new_interceptor_pos_bad = env.interceptor_positions[0]
    new_distance_bad = np.linalg.norm(new_interceptor_pos_bad - env.missile_positions[0])
    
    print(f"New distance: {new_distance_bad:.1f}m")
    print(f"Distance change: {new_distance_bad - initial_distance:.3f}m")
    print(f"Reward: {reward_bad:.3f}")
    
    if new_distance_bad > initial_distance:
        print("✅ EXPECTED: Bad action increased distance")
    else:
        print("❌ UNEXPECTED: Bad action decreased distance")
        
    if reward_good > reward_bad:
        print("✅ CORRECT: Good action gets better reward than bad action")
    else:
        print("❌ WRONG: Bad action gets better reward - reward function still broken")
    
    print(f"\n🏆 SUMMARY:")
    print(f"Good action reward: {reward_good:.3f}")
    print(f"Bad action reward: {reward_bad:.3f}")
    print(f"Reward difference: {reward_good - reward_bad:.3f}")
    
    if reward_good > reward_bad and reward_good > -0.5:
        print("✅ SUCCESS: Fixes appear to be working!")
        print("✅ Ready to retrain the model")
    else:
        print("❌ FAILURE: Fixes not working correctly")
        print("❌ Need more debugging before retraining")

if __name__ == "__main__":
    test_fixes()