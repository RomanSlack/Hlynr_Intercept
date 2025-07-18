#!/usr/bin/env python3
"""
Quick integration test script for Phase 4 RL components.

This script tests the basic functionality of all major components to ensure
they work together correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from config import get_config, reset_config
        from scenarios import get_scenario_loader, reset_scenario_loader
        from radar_env import RadarEnv
        from diagnostics import Logger, export_to_csv, export_to_json
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration management."""
    print("\nTesting configuration...")
    
    try:
        from config import get_config, reset_config
        
        # Reset to ensure clean state
        reset_config()
        
        # Test config loading
        config = get_config()
        assert 'environment' in config._config
        assert 'training' in config._config
        
        # Test config access
        num_missiles = config.get('environment.num_missiles')
        assert num_missiles == 1
        
        # Test config modification
        config.set('environment.num_missiles', 3)
        assert config.get('environment.num_missiles') == 3
        
        print("✓ Configuration tests passed")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_scenarios():
    """Test scenario management."""
    print("\nTesting scenarios...")
    
    try:
        from scenarios import get_scenario_loader, reset_scenario_loader
        
        # Reset to ensure clean state
        reset_scenario_loader()
        
        # Test scenario loading
        loader = get_scenario_loader()
        scenarios = loader.list_scenarios()
        
        if scenarios:
            print(f"  Found scenarios: {scenarios}")
            
            # Test loading first scenario
            first_scenario = scenarios[0]
            scenario_config = loader.load_scenario(first_scenario)
            
            assert 'name' in scenario_config
            assert 'difficulty_level' in scenario_config
            assert 'num_missiles' in scenario_config
            assert 'num_interceptors' in scenario_config
            
            print(f"✓ Successfully loaded scenario: {first_scenario}")
        else:
            print("⚠ No scenarios found, but loader works")
        
        return True
    except Exception as e:
        print(f"✗ Scenario test failed: {e}")
        return False

def test_environment():
    """Test environment functionality."""
    print("\nTesting environment...")
    
    try:
        from radar_env import RadarEnv
        from config import get_config
        
        # Test basic environment creation
        env = RadarEnv()
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
        
        # Test step
        action = env.action_space.sample()
        step_result = env.step(action)
        assert len(step_result) == 5
        
        obs, reward, terminated, truncated, info = step_result
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()
        print("✓ Environment tests passed")
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_diagnostics():
    """Test diagnostics functionality."""
    print("\nTesting diagnostics...")
    
    try:
        from diagnostics import Logger
        
        # Test logger
        logger = Logger()
        logger.reset_episode()
        
        # Log some dummy steps
        for i in range(3):
            step_data = {
                'step': i,
                'observation': [1.0, 2.0, 3.0],
                'action': [0.5, -0.5],
                'reward': i * 0.1,
                'done': False,
                'info': {}
            }
            logger.log_step(step_data)
        
        # Get metrics
        metrics = logger.get_episode_metrics()
        assert 'total_reward' in metrics
        assert 'total_steps' in metrics
        assert metrics['total_steps'] == 3
        
        print("✓ Diagnostics tests passed")
        return True
    except Exception as e:
        print(f"✗ Diagnostics test failed: {e}")
        return False

def test_integration():
    """Test full integration with scenarios."""
    print("\nTesting full integration...")
    
    try:
        from config import get_config, reset_config
        from scenarios import get_scenario_loader, reset_scenario_loader
        from radar_env import RadarEnv
        
        # Reset to clean state
        reset_config()
        reset_scenario_loader()
        
        # Get configuration and scenarios
        config_loader = get_config()
        scenario_loader = get_scenario_loader()
        
        scenarios = scenario_loader.list_scenarios()
        if scenarios:
            # Test with first available scenario
            scenario_name = scenarios[0]
            env_config = scenario_loader.create_environment_config(
                scenario_name, config_loader._config
            )
            
            # Create environment with scenario
            env = RadarEnv(config=env_config, scenario_name=scenario_name)
            
            # Test basic functionality
            obs, info = env.reset()
            # Note: scenario info may not be in reset info, check environment instead
            assert hasattr(env, 'scenario_name') or 'scenario' in info
            
            action = env.action_space.sample()
            step_result = env.step(action)
            assert len(step_result) == 5
            
            env.close()
            print(f"✓ Full integration test passed with scenario: {scenario_name}")
        else:
            print("⚠ No scenarios available for integration test")
        
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("Phase 4 RL Integration Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_configuration,
        test_scenarios,
        test_environment,
        test_diagnostics,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'=' * 40}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("⚠ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())