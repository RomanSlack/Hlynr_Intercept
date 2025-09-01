#!/usr/bin/env python3
"""
Quick test to see if inverting model actions fixes the interceptor behavior.
"""

import numpy as np
from stable_baselines3 import PPO

# Load the model
model = PPO.load('checkpoints_new/phase4_easy_final.zip', env=None)

# Test different observation scenarios
print("🔍 Testing Model Action Behavior")
print("=" * 50)

# Scenario: Missile at [0,0,300], Interceptor at [500,500,100]
# To intercept, interceptor should move toward [0,0,300]
# That means positive X and Y thrust is WRONG (moves away)
# Negative X and Y thrust would be CORRECT (moves toward origin)

print("Test Case: Missile falling from [0,0,300], Interceptor at [500,500,100]")
print("Expected: Interceptor should thrust toward [0,0,300] (negative X,Y)")

# Create realistic observation
rel_pos = np.array([0, 0, 300]) - np.array([500, 500, 100])  # [-500, -500, 200]
rel_pos_norm = rel_pos / 10000.0  # Normalize

rel_vel = np.array([0, 0, -50]) - np.array([0, 0, 0])  # [0, 0, -50]  
rel_vel_norm = rel_vel / 1000.0

obs = np.array([
    rel_pos_norm[0], rel_pos_norm[1], rel_pos_norm[2],  # [-0.05, -0.05, 0.02]
    rel_vel_norm[0], rel_vel_norm[1], rel_vel_norm[2],  # [0, 0, -0.05]
    0.0, 0.0, 0.0,  # Interceptor velocity
    0.0, 0.0, 0.0,  # Orientation  
    1.0,            # Fuel
    0.5,            # Time to intercept
    0.9,            # Radar quality
    -0.05,          # Closing rate
    0.0             # Alignment
]).reshape(1, 17).astype(np.float32)

print(f"Observation: rel_pos={rel_pos}, rel_vel={rel_vel}")
print(f"Normalized obs: {obs[0][:6]}")

action, _ = model.predict(obs, deterministic=True)
print(f"Model action: {action[0]}")

thrust = action[0][:3] * 1000.0
print(f"Thrust forces: {thrust}")

print("\nAnalysis:")
if thrust[0] > 0:
    print("❌ WRONG: Positive X thrust pushes interceptor AWAY from missile (500→600)")
else:
    print("✅ CORRECT: Negative X thrust pushes interceptor TOWARD missile (500→400)")
    
if thrust[1] > 0:
    print("❌ WRONG: Positive Y thrust pushes interceptor AWAY from missile (500→600)")  
else:
    print("✅ CORRECT: Negative Y thrust pushes interceptor TOWARD missile (500→400)")

if thrust[2] > 0:
    print("✅ MAYBE: Positive Z thrust pushes interceptor UP toward missile (100→400)")
else:
    print("❌ MAYBE: Negative Z thrust pushes interceptor DOWN away from missile (100→0)")

print(f"\n🏆 Verdict: Model is {'CORRECT' if thrust[0] < 0 and thrust[1] < 0 else 'WRONG'}")

# Test the opposite observation (what if we flip the relative position?)
print("\n" + "="*50)
print("🔄 Testing with FLIPPED relative position signs:")

obs_flipped = obs.copy()
obs_flipped[0][0] = -obs_flipped[0][0]  # Flip X
obs_flipped[0][1] = -obs_flipped[0][1]  # Flip Y
obs_flipped[0][2] = -obs_flipped[0][2]  # Flip Z

action_flipped, _ = model.predict(obs_flipped, deterministic=True)
thrust_flipped = action_flipped[0][:3] * 1000.0

print(f"Flipped observation: {obs_flipped[0][:6]}")
print(f"Flipped action: {action_flipped[0]}")
print(f"Flipped thrust: {thrust_flipped}")

print(f"\n🏆 Flipped Verdict: Model is {'CORRECT' if thrust_flipped[0] < 0 and thrust_flipped[1] < 0 else 'WRONG'}")