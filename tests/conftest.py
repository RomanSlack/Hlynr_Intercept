"""
Pytest Configuration and Shared Fixtures

This module provides pytest configuration and shared fixtures for the
6DOF test suite including:
- Test markers for different test categories
- Shared test fixtures and utilities
- Test configuration and setup
- Performance monitoring fixtures
- Environment setup helpers

Author: Tester Agent
Date: Phase 3 Testing Framework
"""

import pytest
import numpy as np
import tempfile
import os
import psutil
import time
from typing import Dict, Any, Generator, Optional
import logging

from aegis_intercept.envs.aegis_6dof_env import Aegis6DInterceptEnv, DifficultyMode, ActionMode
from aegis_intercept.curriculum.curriculum_manager import CurriculumManager
from aegis_intercept.logging.trajectory_logger import TrajectoryLogger
from aegis_intercept.utils.physics6dof import RigidBody6DOF, VehicleType


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take more than 30 seconds)"
    )
    config.addinivalue_line(
        "markers", "physics: marks tests that validate physics accuracy"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests that benchmark performance"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests that test component integration"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests that prevent regression"
    )
    config.addinivalue_line(
        "markers", "realworld: marks tests that validate real-world applicability"
    )


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def basic_6dof_env():
    """Provide a basic 6DOF environment for testing"""
    env = Aegis6DInterceptEnv(
        difficulty_mode=DifficultyMode.FULL_6DOF,
        max_steps=100,
        world_size=300.0
    )
    yield env
    if hasattr(env, 'close'):
        env.close()


@pytest.fixture
def legacy_3dof_env():
    """Provide a legacy 3DOF environment for compatibility testing"""
    env = Aegis6DInterceptEnv(
        legacy_3dof_mode=True,
        difficulty_mode=DifficultyMode.EASY_3DOF,
        action_mode=ActionMode.ACCELERATION_3DOF,
        max_steps=100
    )
    yield env
    if hasattr(env, 'close'):
        env.close()


@pytest.fixture
def curriculum_manager():
    """Provide a curriculum manager for testing"""
    manager = CurriculumManager()
    yield manager


@pytest.fixture
def trajectory_logger(temp_dir):
    """Provide a trajectory logger for testing"""
    logger = TrajectoryLogger(log_dir=temp_dir)
    yield logger


@pytest.fixture
def interceptor_6dof():
    """Provide a 6DOF interceptor rigid body for physics testing"""
    rigid_body = RigidBody6DOF(
        VehicleType.INTERCEPTOR,
        initial_position=np.array([0, 0, 1000]),
        initial_velocity=np.array([100, 0, 0]),
        initial_orientation=np.array([1, 0, 0, 0]),
        initial_angular_velocity=np.array([0, 0, 0])
    )
    yield rigid_body


@pytest.fixture
def missile_6dof():
    """Provide a 6DOF missile rigid body for physics testing"""
    rigid_body = RigidBody6DOF(
        VehicleType.MISSILE,
        initial_position=np.array([1000, 0, 1000]),
        initial_velocity=np.array([-50, 0, 0]),
        initial_orientation=np.array([1, 0, 0, 0]),
        initial_angular_velocity=np.array([0, 0, 0])
    )
    yield rigid_body


@pytest.fixture
def performance_monitor():
    """Provide a performance monitor for benchmarking tests"""
    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_memory = None
            self.start_time = None
        
        def start(self):
            self.start_memory = self.process.memory_info().rss
            self.start_time = time.time()
        
        def stop(self):
            end_memory = self.process.memory_info().rss
            end_time = time.time()
            
            return {
                'duration': end_time - self.start_time,
                'memory_growth': end_memory - self.start_memory,
                'memory_start': self.start_memory,
                'memory_end': end_memory
            }
    
    monitor = PerformanceMonitor()
    yield monitor


@pytest.fixture
def deterministic_seed():
    """Provide a deterministic seed for reproducible tests"""
    return 42


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress specific loggers that might be noisy during tests
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)


@pytest.fixture
def test_scenario_config():
    """Provide a test scenario configuration"""
    return {
        'name': 'test_scenario',
        'description': 'Standard test scenario',
        'interceptor_position_range': {'x': (50, 150), 'y': (50, 150), 'z': (0, 50)},
        'interceptor_velocity_range': {'x': (-10, 10), 'y': (-10, 10), 'z': (0, 20)},
        'missile_position_range': {'x': (200, 400), 'y': (200, 400), 'z': (100, 300)},
        'missile_velocity_range': {'x': (-50, 50), 'y': (-50, 50), 'z': (-20, 0)},
        'target_position_range': {'x': (90, 110), 'y': (90, 110), 'z': (0, 10)},
        'wind_conditions': {'enabled': True, 'strength': 1.0},
        'atmospheric_conditions': {'standard': True},
        'evasion_aggressiveness': 1.0
    }


@pytest.fixture
def multiple_env_configs():
    """Provide multiple environment configurations for testing"""
    return [
        {
            'name': '3dof_legacy',
            'config': {
                'legacy_3dof_mode': True,
                'difficulty_mode': DifficultyMode.EASY_3DOF,
                'action_mode': ActionMode.ACCELERATION_3DOF
            }
        },
        {
            'name': '6dof_simplified',
            'config': {
                'difficulty_mode': DifficultyMode.SIMPLIFIED_6DOF,
                'action_mode': ActionMode.THRUST_ATTITUDE
            }
        },
        {
            'name': '6dof_full',
            'config': {
                'difficulty_mode': DifficultyMode.FULL_6DOF,
                'action_mode': ActionMode.ACCELERATION_6DOF
            }
        },
        {
            'name': '6dof_expert',
            'config': {
                'difficulty_mode': DifficultyMode.EXPERT_6DOF,
                'action_mode': ActionMode.ACCELERATION_6DOF
            }
        }
    ]


# Helper functions for tests
def create_test_environment(config_name: str = 'default', **kwargs) -> Aegis6DInterceptEnv:
    """Create a test environment with specified configuration"""
    default_configs = {
        'default': {
            'difficulty_mode': DifficultyMode.FULL_6DOF,
            'max_steps': 100,
            'world_size': 300.0
        },
        'fast': {
            'difficulty_mode': DifficultyMode.SIMPLIFIED_6DOF,
            'max_steps': 50,
            'world_size': 200.0
        },
        'legacy': {
            'legacy_3dof_mode': True,
            'difficulty_mode': DifficultyMode.EASY_3DOF,
            'action_mode': ActionMode.ACCELERATION_3DOF,
            'max_steps': 50
        }
    }
    
    config = default_configs.get(config_name, default_configs['default'])
    config.update(kwargs)
    
    return Aegis6DInterceptEnv(**config)


def run_basic_episode(env: Aegis6DInterceptEnv, seed: int = 42, 
                     policy: str = 'random') -> Dict[str, Any]:
    """Run a basic episode and return results"""
    obs, info = env.reset(seed=seed)
    
    episode_data = {
        'rewards': [],
        'actions': [],
        'observations': [obs.copy()],
        'infos': [info.copy()],
        'terminated': False,
        'truncated': False,
        'total_reward': 0.0,
        'episode_length': 0
    }
    
    for step in range(env.max_steps):
        if policy == 'random':
            action = env.action_space.sample()
        elif policy == 'pursuit':
            # Simple pursuit policy
            if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                int_pos = env.interceptor_6dof.position
                mis_pos = env.missile_6dof.position
                direction = mis_pos - int_pos
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                action = np.concatenate([direction * 0.8, direction * 0.2, [0.0]])
            else:
                action = env.action_space.sample()
        else:
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_data['rewards'].append(reward)
        episode_data['actions'].append(action.copy())
        episode_data['observations'].append(obs.copy())
        episode_data['infos'].append(info.copy())
        episode_data['total_reward'] += reward
        episode_data['episode_length'] += 1
        
        if terminated or truncated:
            episode_data['terminated'] = terminated
            episode_data['truncated'] = truncated
            break
    
    return episode_data


def assert_environment_valid(env: Aegis6DInterceptEnv):
    """Assert that an environment is in a valid state"""
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.max_steps > 0
    assert env.world_size > 0
    assert env.dt > 0


def assert_observation_valid(obs: np.ndarray, env: Aegis6DInterceptEnv):
    """Assert that an observation is valid"""
    assert env.observation_space.contains(obs), f"Observation out of bounds: {obs}"
    assert np.all(np.isfinite(obs)), f"Observation contains non-finite values: {obs}"
    assert obs.dtype == np.float32, f"Observation wrong dtype: {obs.dtype}"


def assert_action_valid(action: np.ndarray, env: Aegis6DInterceptEnv):
    """Assert that an action is valid"""
    assert env.action_space.contains(action), f"Action out of bounds: {action}"
    assert np.all(np.isfinite(action)), f"Action contains non-finite values: {action}"
    assert action.dtype == np.float32, f"Action wrong dtype: {action.dtype}"


def assert_reward_valid(reward: float):
    """Assert that a reward is valid"""
    assert np.isfinite(reward), f"Reward is not finite: {reward}"
    assert isinstance(reward, (int, float)), f"Reward wrong type: {type(reward)}"


def measure_performance(func, *args, **kwargs) -> Dict[str, Any]:
    """Measure performance of a function"""
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    end_memory = process.memory_info().rss
    
    return {
        'result': result,
        'duration': end_time - start_time,
        'memory_growth': end_memory - start_memory,
        'start_memory': start_memory,
        'end_memory': end_memory
    }


# Test data generators
def generate_test_trajectories(num_trajectories: int = 10, 
                              trajectory_length: int = 50) -> List[Dict[str, Any]]:
    """Generate test trajectory data"""
    trajectories = []
    
    for i in range(num_trajectories):
        trajectory = {
            'trajectory_id': f'test_traj_{i}',
            'positions': np.random.uniform(-100, 100, (trajectory_length, 3)),
            'velocities': np.random.uniform(-50, 50, (trajectory_length, 3)),
            'orientations': np.random.uniform(-1, 1, (trajectory_length, 4)),  # Will need normalization
            'timestamps': np.linspace(0, trajectory_length * 0.05, trajectory_length)
        }
        
        # Normalize quaternions
        for j in range(trajectory_length):
            quat = trajectory['orientations'][j]
            trajectory['orientations'][j] = quat / np.linalg.norm(quat)
        
        trajectories.append(trajectory)
    
    return trajectories


def validate_test_environment():
    """Validate that the test environment is properly set up"""
    try:
        # Test basic environment creation
        env = Aegis6DInterceptEnv()
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        if hasattr(env, 'close'):
            env.close()
        return True
    except Exception as e:
        print(f"Test environment validation failed: {e}")
        return False


# Pytest hooks for custom behavior
def pytest_runtest_setup(item):
    """Setup for each test"""
    # Validate test environment before running tests
    if not validate_test_environment():
        pytest.skip("Test environment validation failed")


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers based on test names/paths
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid.lower() or any(mark.name == "slow" for mark in item.iter_markers()):
            item.add_marker(pytest.mark.slow)
        
        # Mark performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)
        
        # Mark physics tests
        if "physics" in item.nodeid.lower():
            item.add_marker(pytest.mark.physics)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)


def pytest_report_header(config):
    """Add custom header to test report"""
    return [
        "AegisIntercept 6DOF Test Suite",
        "================================",
        "Testing Phase 3 6DOF implementation with comprehensive validation",
        ""
    ]