"""
Test for verifying that seeding fix works correctly.
"""

import numpy as np
import pytest
from radar_env import RadarEnv


def test_seeding_fix():
    """Test that seeding produces reproducible results after the fix."""
    env = RadarEnv()
    
    # Reset with same seed should produce same initial observations
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    
    # Observations should be identical
    np.testing.assert_array_equal(obs1, obs2)


def test_step_reproducibility_fixed():
    """Test that seeded environments behave reproducibly after the fix."""
    env1 = RadarEnv()
    env2 = RadarEnv()
    
    # Reset both with same seed
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    # Take same action in both - use deterministic action instead of random
    action = np.zeros(env1.action_space.shape, dtype=env1.action_space.dtype)
    
    result1 = env1.step(action)
    result2 = env2.step(action)
    
    # Results should be identical (or very close due to floating point)
    np.testing.assert_array_almost_equal(result1[0], result2[0], decimal=5)
    assert abs(result1[1] - result2[1]) < 1e-6  # Rewards should be very close


if __name__ == '__main__':
    pytest.main([__file__])