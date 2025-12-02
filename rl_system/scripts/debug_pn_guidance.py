#!/usr/bin/env python3
"""
Debug script for Proportional Navigation (PN) guidance testing.

Tests whether the LOS observation and action frames are correctly aligned
for PN guidance to work.

PN Law: a_lateral = N * V_c * LOS_rate
- N is navigation ratio (typically 3-5)
- V_c is closing velocity
- LOS_rate is angular rate of LOS rotation

In our action space:
- action[0]: thrust along LOS (toward target, for closing)
- action[1]: thrust along los_horizontal (lateral correction)
- action[2]: thrust along los_vertical (vertical correction)

In our observation space:
- obs[2]: LOS azimuth rate (decomposed onto los_horizontal)
- obs[3]: LOS elevation rate (decomposed onto los_vertical)

For PN to work correctly:
- If obs[2] > 0 (LOS rotating in +los_horizontal direction)
- We need action[1] > 0 (thrust in +los_horizontal) to cancel it
- This is because we want to accelerate TOWARD where the target is going

Let's test this with a concrete example.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from environment import InterceptEnvironment

def test_pn_sign_convention():
    """Test that PN guidance signs are correct."""
    print("=" * 60)
    print("PROPORTIONAL NAVIGATION SIGN CONVENTION TEST")
    print("=" * 60)

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'configs/eval_terminal_standalone.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create env
    env = InterceptEnvironment(config['environment'])

    # Set up controlled scenario
    np.random.seed(42)

    # Run multiple tests with different geometries
    n_tests = 10
    results = []

    for test_idx in range(n_tests):
        obs, info = env.reset()

        # Wait for radar lock
        for _ in range(50):
            obs, _, done, _, _ = env.step(np.zeros(6, dtype=np.float32))
            if done:
                break
            # Check if we have radar lock (obs[0] != -2.0)
            if obs[0] != -2.0:
                break

        if obs[0] == -2.0:
            print(f"Test {test_idx}: No radar lock after 50 steps, skipping")
            continue

        # Get initial state
        int_pos = np.array(env.interceptor_state['position'])
        mis_pos = np.array(env.missile_state['position'])
        int_vel = np.array(env.interceptor_state['velocity'])
        mis_vel = np.array(env.missile_state['velocity'])

        # Ground truth LOS rate calculation
        rel_pos = mis_pos - int_pos
        rel_vel = mis_vel - int_vel
        range_to_target = np.linalg.norm(rel_pos)
        los_unit = rel_pos / range_to_target

        # Ground truth LOS rate vector
        closing = np.dot(rel_vel, los_unit)
        tangent_vel = rel_vel - closing * los_unit
        los_rate_vec = tangent_vel / range_to_target

        # Decompose into our frame
        world_up = np.array([0.0, 0.0, 1.0])
        los_right = np.cross(los_unit, world_up)
        los_right_norm = np.linalg.norm(los_right)
        if los_right_norm > 1e-6:
            los_horizontal = los_right / los_right_norm
        else:
            los_horizontal = np.array([1.0, 0.0, 0.0])
        los_vertical = np.cross(los_unit, los_horizontal)

        gt_azimuth_rate = np.dot(los_rate_vec, los_horizontal)
        gt_elevation_rate = np.dot(los_rate_vec, los_vertical)

        # Compare with observation
        obs_azimuth_rate = obs[2] * 0.5  # Denormalize (max_los_rate = 0.5)
        obs_elevation_rate = obs[3] * 0.5

        print(f"\nTest {test_idx}:")
        print(f"  Range: {range_to_target:.1f}m")
        print(f"  Closing rate: {-closing:.1f} m/s")  # Positive = closing
        print(f"  Ground truth azimuth rate:   {gt_azimuth_rate:+.4f} rad/s")
        print(f"  Observation azimuth rate:    {obs_azimuth_rate:+.4f} rad/s")
        print(f"  Ground truth elevation rate: {gt_elevation_rate:+.4f} rad/s")
        print(f"  Observation elevation rate:  {obs_elevation_rate:+.4f} rad/s")

        # Check sign match
        az_sign_match = (gt_azimuth_rate * obs_azimuth_rate >= 0) or abs(gt_azimuth_rate) < 0.01
        el_sign_match = (gt_elevation_rate * obs_elevation_rate >= 0) or abs(gt_elevation_rate) < 0.01
        print(f"  Azimuth sign match: {az_sign_match}")
        print(f"  Elevation sign match: {el_sign_match}")

        results.append({
            'az_sign_match': az_sign_match,
            'el_sign_match': el_sign_match,
            'gt_az': gt_azimuth_rate,
            'obs_az': obs_azimuth_rate,
            'gt_el': gt_elevation_rate,
            'obs_el': obs_elevation_rate
        })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    az_matches = sum(1 for r in results if r['az_sign_match'])
    el_matches = sum(1 for r in results if r['el_sign_match'])
    print(f"Azimuth sign matches: {az_matches}/{len(results)}")
    print(f"Elevation sign matches: {el_matches}/{len(results)}")

    env.close()
    return results


def test_action_effect():
    """Test that actions have the expected effect on LOS rate."""
    print("\n" + "=" * 60)
    print("ACTION EFFECT TEST")
    print("=" * 60)

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'configs/eval_terminal_standalone.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env = InterceptEnvironment(config['environment'])
    np.random.seed(123)

    # Test: Apply positive horizontal thrust and see if LOS azimuth rate changes as expected
    obs, info = env.reset()

    # Wait for lock
    for _ in range(50):
        obs, _, done, _, _ = env.step(np.zeros(6, dtype=np.float32))
        if done or obs[0] != -2.0:
            break

    if obs[0] == -2.0:
        print("No radar lock, cannot test")
        env.close()
        return

    initial_az_rate = obs[2]
    initial_el_rate = obs[3]
    print(f"Initial LOS azimuth rate: {initial_az_rate:+.4f}")
    print(f"Initial LOS elevation rate: {initial_el_rate:+.4f}")

    # Apply pure horizontal thrust (action[1] = 1.0)
    print("\nApplying action[1] = +1.0 (positive horizontal thrust)")
    action = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    for step in range(10):
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break

    final_az_rate = obs[2]
    final_el_rate = obs[3]
    print(f"After 10 steps:")
    print(f"  LOS azimuth rate: {final_az_rate:+.4f} (delta: {final_az_rate - initial_az_rate:+.4f})")
    print(f"  LOS elevation rate: {final_el_rate:+.4f} (delta: {final_el_rate - initial_el_rate:+.4f})")

    # For PN to work: if initial_az_rate > 0, we want positive horizontal thrust
    # to REDUCE the azimuth rate (bring it toward zero)
    print("\n--- Interpretation ---")
    print("For PN: positive horizontal thrust should reduce positive azimuth rate")
    print("(We want to accelerate toward where target is going)")
    if initial_az_rate > 0.05:
        if final_az_rate < initial_az_rate:
            print("✓ Correct: positive thrust reduced positive azimuth rate")
        else:
            print("✗ WRONG: positive thrust increased positive azimuth rate (sign flip needed)")
    elif initial_az_rate < -0.05:
        print("Initial azimuth rate is negative - need to test with positive rate")
    else:
        print("Initial azimuth rate near zero - cannot determine sign relationship")

    env.close()


def test_pn_policy_simulation():
    """Simulate a pure PN policy and check if it converges."""
    print("\n" + "=" * 60)
    print("PN POLICY SIMULATION TEST")
    print("=" * 60)

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'configs/eval_terminal_standalone.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env = InterceptEnvironment(config['environment'])

    # Test both positive and negative PN gain
    for N in [3.0, -3.0]:
        np.random.seed(456)
        obs, info = env.reset()

        print(f"\n--- PN with N = {N} ---")

        min_distance = float('inf')

        for step in range(1000):
            if obs[0] == -2.0:  # No radar lock
                action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                # PN policy
                az_rate = obs[2]  # Already normalized by 0.5 rad/s
                el_rate = obs[3]

                # action[0]: thrust toward target
                # action[1]: lateral acceleration = N * LOS_az_rate
                # action[2]: vertical acceleration = N * LOS_el_rate
                action = np.array([
                    1.0,          # Full thrust toward target
                    N * az_rate,  # PN lateral correction
                    N * el_rate,  # PN vertical correction
                    0.0, 0.0, 0.0
                ], dtype=np.float32)
                action = np.clip(action, -1.0, 1.0)

            obs, reward, done, truncated, info = env.step(action)

            # Track minimum distance
            int_pos = env.interceptor_state['position']
            mis_pos = env.missile_state['position']
            dist = np.linalg.norm(np.array(mis_pos) - np.array(int_pos))
            min_distance = min(min_distance, dist)

            if done or truncated:
                break

        print(f"  Minimum distance achieved: {min_distance:.1f}m")
        print(f"  Success (< 50m): {'Yes' if min_distance < 50 else 'No'}")

    env.close()


if __name__ == "__main__":
    test_pn_sign_convention()
    test_action_effect()
    test_pn_policy_simulation()
