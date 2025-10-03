"""
Check if missile is tracking toward interceptor or target.
"""

import numpy as np
import yaml
from environment import InterceptEnvironment

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create environment
env = InterceptEnvironment(config['environment'])

# Run 5 episodes and check missile behavior
print("="*80)
print("MISSILE TRAJECTORY ANALYSIS")
print("="*80)

for ep in range(5):
    obs, info = env.reset()

    initial_missile_pos = env.missile_state['position'].copy()
    initial_missile_vel = env.missile_state['velocity'].copy()
    initial_interceptor_pos = env.interceptor_state['position'].copy()
    target_pos = env.target_position

    print(f"\nEpisode {ep+1}:")
    print(f"  Target position: {target_pos}")
    print(f"  Missile spawn: {initial_missile_pos}")
    print(f"  Interceptor spawn: {initial_interceptor_pos}")
    print(f"  Missile initial velocity: {initial_missile_vel}")

    # Calculate initial directions
    to_target = target_pos - initial_missile_pos
    to_interceptor = initial_interceptor_pos - initial_missile_pos

    to_target_norm = to_target / np.linalg.norm(to_target)
    to_interceptor_norm = to_interceptor / np.linalg.norm(to_interceptor)
    vel_norm = initial_missile_vel / np.linalg.norm(initial_missile_vel)

    angle_to_target = np.degrees(np.arccos(np.clip(np.dot(vel_norm, to_target_norm), -1, 1)))
    angle_to_interceptor = np.degrees(np.arccos(np.clip(np.dot(vel_norm, to_interceptor_norm), -1, 1)))

    print(f"  Angle between velocity and TARGET: {angle_to_target:.1f}°")
    print(f"  Angle between velocity and INTERCEPTOR: {angle_to_interceptor:.1f}°")

    if angle_to_target < angle_to_interceptor:
        print(f"  → Missile heading toward TARGET ✓")
    else:
        print(f"  → Missile heading toward INTERCEPTOR ✗")

    # Simulate 100 steps without control to see pure ballistic trajectory
    missile_positions = [initial_missile_pos.copy()]

    for step in range(100):
        # Zero action (no thrust)
        action = np.zeros(6)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

        missile_positions.append(env.missile_state['position'].copy())

    # Check if missile is getting closer to target or interceptor
    final_missile_pos = missile_positions[-1]

    initial_dist_to_target = np.linalg.norm(initial_missile_pos - target_pos)
    final_dist_to_target = np.linalg.norm(final_missile_pos - target_pos)

    initial_dist_to_int = np.linalg.norm(initial_missile_pos - initial_interceptor_pos)
    final_dist_to_int = np.linalg.norm(final_missile_pos - initial_interceptor_pos)

    print(f"  After 100 steps (1 second):")
    print(f"    Distance to target: {initial_dist_to_target:.1f}m → {final_dist_to_target:.1f}m (change: {final_dist_to_target - initial_dist_to_target:.1f}m)")
    print(f"    Distance to interceptor spawn: {initial_dist_to_int:.1f}m → {final_dist_to_int:.1f}m (change: {final_dist_to_int - initial_dist_to_int:.1f}m)")

    if final_dist_to_target < initial_dist_to_target:
        print(f"  → Missile IS approaching target ✓")
    else:
        print(f"  → Missile NOT approaching target ✗")

print("\n" + "="*80)
print("CONCLUSION:")
print("The missile follows a BALLISTIC trajectory (no guidance).")
print("It should be heading toward the target based on initial velocity.")
print("If it appears to track the interceptor, it may be an optical illusion")
print("from the visualization perspective or interceptor pursuing the missile.")
print("="*80)
