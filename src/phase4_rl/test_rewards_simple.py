#!/usr/bin/env python3
"""
Simple test of just the reward function logic.
"""

import numpy as np

def test_reward_function():
    """Test the new reward function logic."""
    
    print("🔧 Testing Fixed Reward Function")
    print("=" * 50)
    
    # Simulate the new reward calculation
    def calculate_reward(interceptor_pos, missile_pos, prev_distance=None):
        """Simulate the fixed reward function."""
        distance = np.linalg.norm(interceptor_pos - missile_pos)
        reward = 0.0
        
        # Large reward for successful interception
        if distance < 50.0:
            reward += 1000.0
        else:
            # Distance reward: closer = better
            distance_reward = max(0, (1000.0 - distance)) / 100.0
            reward += distance_reward
            
            # Progress reward
            if prev_distance is not None:
                progress = prev_distance - distance  # Positive when getting closer
                reward += progress * 10.0
        
        # Small time penalty
        reward -= 0.01
        
        return reward, distance
    
    # Test scenario: missile at [0,0,300], interceptor starts at [500,500,100]
    missile_pos = np.array([0.0, 0.0, 300.0])
    
    # Initial position
    interceptor_pos_initial = np.array([500.0, 500.0, 100.0])
    initial_reward, initial_distance = calculate_reward(interceptor_pos_initial, missile_pos)
    
    print(f"Initial setup:")
    print(f"  Missile: {missile_pos}")
    print(f"  Interceptor: {interceptor_pos_initial}")
    print(f"  Distance: {initial_distance:.1f}m")
    print(f"  Reward: {initial_reward:.3f}")
    
    # Test moving TOWARD missile (good action)
    interceptor_pos_good = interceptor_pos_initial + np.array([-10.0, -10.0, 5.0])  # Move toward missile
    good_reward, good_distance = calculate_reward(interceptor_pos_good, missile_pos, initial_distance)
    
    print(f"\nAfter GOOD action (move toward missile):")
    print(f"  New interceptor pos: {interceptor_pos_good}")
    print(f"  New distance: {good_distance:.1f}m")
    print(f"  Distance change: {good_distance - initial_distance:.1f}m")
    print(f"  Reward: {good_reward:.3f}")
    
    # Test moving AWAY from missile (bad action)
    interceptor_pos_bad = interceptor_pos_initial + np.array([10.0, 10.0, -5.0])  # Move away from missile
    bad_reward, bad_distance = calculate_reward(interceptor_pos_bad, missile_pos, initial_distance)
    
    print(f"\nAfter BAD action (move away from missile):")
    print(f"  New interceptor pos: {interceptor_pos_bad}")
    print(f"  New distance: {bad_distance:.1f}m")
    print(f"  Distance change: {bad_distance - initial_distance:.1f}m")
    print(f"  Reward: {bad_reward:.3f}")
    
    # Analysis
    print(f"\n🏆 ANALYSIS:")
    print(f"Good action reward: {good_reward:.3f}")
    print(f"Bad action reward: {bad_reward:.3f}")
    print(f"Reward difference: {good_reward - bad_reward:.3f}")
    
    if good_reward > bad_reward:
        print("✅ CORRECT: Good action gets better reward")
    else:
        print("❌ WRONG: Bad action gets better reward")
    
    if good_reward > 0:
        print("✅ CORRECT: Good actions get positive reward")
    else:
        print("⚠️  WARNING: Good actions still get negative reward")
        
    if bad_reward < initial_reward:
        print("✅ CORRECT: Bad actions get worse reward than doing nothing")
    else:
        print("❌ WRONG: Bad actions get better reward than doing nothing")
    
    # Test interception reward
    interceptor_pos_intercept = missile_pos + np.array([5.0, 5.0, -5.0])  # Very close to missile
    intercept_reward, intercept_distance = calculate_reward(interceptor_pos_intercept, missile_pos, initial_distance)
    
    print(f"\nInterception test (distance={intercept_distance:.1f}m):")
    print(f"  Reward: {intercept_reward:.3f}")
    
    if intercept_reward > 500:
        print("✅ CORRECT: Interception gets huge reward")
    else:
        print("❌ WRONG: Interception reward too small")
    
    print(f"\n{'='*50}")
    if good_reward > bad_reward and intercept_reward > 500:
        print("🎉 SUCCESS: Reward function is now working correctly!")
        print("🚀 Ready to retrain the model with fixed rewards")
    else:
        print("❌ FAILURE: Reward function still has issues")

if __name__ == "__main__":
    test_reward_function()