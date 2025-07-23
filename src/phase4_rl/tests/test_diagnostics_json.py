"""
Test for verifying diagnostics JSON record/reload cycle.
"""

import json
import tempfile
import os
from pathlib import Path
import pytest
import numpy as np

try:
    from ..diagnostics import Logger
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('..')
    from diagnostics import Logger


def test_json_record_reload_cycle():
    """Test that diagnostics can record episode data and reload it successfully."""
    # Create logger
    logger = Logger()
    logger.reset_episode()
    
    # Log some sample step data
    sample_steps = [
        {
            'step': 0,
            'observation': np.array([1.0, 2.0, 3.0], dtype=np.float32),
            'action': np.array([0.5, -0.5], dtype=np.float32),
            'reward': 0.1,
            'done': False,
            'info': {
                'missile_positions': [[100.0, 200.0]],
                'interceptor_positions': [[400.0, 500.0]],
                'missile_velocities': [[10.0, 15.0]],
                'interceptor_velocities': [[5.0, -8.0]],
                'min_interception_distances': [300.0]
            }
        },
        {
            'step': 1,
            'observation': np.array([1.1, 2.1, 3.1], dtype=np.float32),
            'action': np.array([0.6, -0.4], dtype=np.float32),
            'reward': 0.2,
            'done': False,
            'info': {
                'missile_positions': [[110.0, 215.0]],
                'interceptor_positions': [[405.0, 492.0]],
                'missile_velocities': [[10.0, 15.0]],
                'interceptor_velocities': [[5.0, -8.0]],
                'min_interception_distances': [285.0]
            }
        },
        {
            'step': 2,
            'observation': np.array([1.2, 2.2, 3.2], dtype=np.float32),
            'action': np.array([0.7, -0.3], dtype=np.float32),
            'reward': 0.3,
            'done': True,
            'info': {
                'missile_positions': [[120.0, 230.0]],
                'interceptor_positions': [[410.0, 484.0]],
                'missile_velocities': [[10.0, 15.0]],
                'interceptor_velocities': [[5.0, -8.0]],
                'min_interception_distances': [270.0]
            }
        }
    ]
    
    # Log all steps
    for step_data in sample_steps:
        logger.log_step(step_data)
    
    # Get episode metrics
    original_metrics = logger.get_episode_metrics()
    
    # Save to temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        logger.save_episode(temp_path)
    
    try:
        # Verify file was created and has content
        assert os.path.exists(temp_path), "JSON file was not created"
        assert os.path.getsize(temp_path) > 0, "JSON file is empty"
        
        # Load the JSON file
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        # Verify required keys exist
        required_keys = ['metrics', 'step_data', 'timestamp']
        for key in required_keys:
            assert key in loaded_data, f"Missing required key: {key}"
        
        # Verify metrics match
        loaded_metrics = loaded_data['metrics']
        
        # Check basic metrics
        assert loaded_metrics['total_steps'] == original_metrics['total_steps']
        assert abs(loaded_metrics['total_reward'] - original_metrics['total_reward']) < 1e-6
        assert loaded_metrics['closest_approach'] == original_metrics['closest_approach']
        
        # Verify step data
        loaded_step_data = loaded_data['step_data']
        assert len(loaded_step_data) == len(sample_steps)
        
        # Check specific step data
        for i, (original_step, loaded_step) in enumerate(zip(sample_steps, loaded_step_data)):
            assert loaded_step['step'] == original_step['step']
            assert abs(loaded_step['reward'] - original_step['reward']) < 1e-6
            assert loaded_step['done'] == original_step['done']
            
            # Verify trajectory data was extracted correctly
            if 'missile_positions' in loaded_step:
                assert loaded_step['missile_positions'] == original_step['info']['missile_positions']
            if 'interceptor_positions' in loaded_step:
                assert loaded_step['interceptor_positions'] == original_step['info']['interceptor_positions']
        
        # Verify trajectories were computed correctly
        if 'missile_trajectories' in loaded_metrics:
            missile_trajectories = loaded_metrics['missile_trajectories']
            assert len(missile_trajectories) == 1  # One missile
            assert len(missile_trajectories[0]) == 3  # Three steps
            
            # Check first missile trajectory point
            assert missile_trajectories[0][0] == [100.0, 200.0]
            assert missile_trajectories[0][1] == [110.0, 215.0] 
            assert missile_trajectories[0][2] == [120.0, 230.0]
        
        print("JSON record/reload cycle test passed successfully!")
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_empty_logger_handling():
    """Test that logger handles empty episode gracefully."""
    logger = Logger()
    logger.reset_episode()
    
    # Get metrics without logging any steps
    metrics = logger.get_episode_metrics()
    
    # Should return empty but valid metrics
    assert isinstance(metrics, dict)
    assert metrics.get('total_steps', 0) == 0
    assert metrics.get('total_reward', 0) == 0


def test_metrics_calculation():
    """Test that episode metrics are calculated correctly."""
    logger = Logger()
    logger.reset_episode()
    
    # Log steps with known values
    rewards = [0.1, 0.2, -0.1, 0.3]
    for i, reward in enumerate(rewards):
        step_data = {
            'step': i,
            'observation': np.array([1.0] * 5),
            'action': np.array([0.0] * 2),
            'reward': reward,
            'done': i == len(rewards) - 1,
            'info': {}
        }
        logger.log_step(step_data)
    
    metrics = logger.get_episode_metrics()
    
    # Verify calculated metrics
    expected_total_reward = sum(rewards)
    assert abs(metrics['total_reward'] - expected_total_reward) < 1e-6
    assert metrics['total_steps'] == len(rewards)
    assert abs(metrics['average_reward'] - np.mean(rewards)) < 1e-6
    assert abs(metrics['min_reward'] - min(rewards)) < 1e-6
    assert abs(metrics['max_reward'] - max(rewards)) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__])