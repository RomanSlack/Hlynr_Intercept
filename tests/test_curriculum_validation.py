"""
Curriculum Learning Validation Tests

This module contains comprehensive tests to validate the curriculum learning
system including:
- Progressive difficulty verification
- Phase transition logic validation
- JSON configuration handling
- Performance tracking accuracy
- Scenario generation consistency
- Advancement criteria validation

Author: Tester Agent
Date: Phase 3 Testing Framework
"""

import pytest
import numpy as np
import json
import tempfile
import os
import time
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from aegis_intercept.curriculum.curriculum_manager import (
    CurriculumManager, CurriculumPhase, AdvancementCriteria, 
    PhaseConfig, ScenarioConfig, PerformanceMetrics,
    create_curriculum_manager, setup_curriculum_directories
)
from aegis_intercept.envs.aegis_6dof_env import DifficultyMode, ActionMode


class TestCurriculumPhaseProgression:
    """Test curriculum phase progression and difficulty scaling"""
    
    def test_phase_sequence_ordering(self):
        """Test that curriculum phases follow logical progression"""
        manager = CurriculumManager()
        
        phases = list(CurriculumPhase)
        expected_order = [
            CurriculumPhase.PHASE_1_BASIC_3DOF,
            CurriculumPhase.PHASE_2_ADVANCED_3DOF,
            CurriculumPhase.PHASE_3_SIMPLIFIED_6DOF,
            CurriculumPhase.PHASE_4_FULL_6DOF,
            CurriculumPhase.PHASE_5_EXPERT_6DOF
        ]
        
        assert phases == expected_order, "Phase order should follow logical progression"
        
        # Test difficulty progression
        prev_complexity = 0
        for phase in phases:
            config = manager.phase_configs[phase]
            
            # Calculate complexity score
            complexity = 0
            if config.difficulty_mode in [DifficultyMode.SIMPLIFIED_6DOF, DifficultyMode.FULL_6DOF, DifficultyMode.EXPERT_6DOF]:
                complexity += 3
            if config.enable_wind:
                complexity += 1
            if config.action_mode == ActionMode.ACCELERATION_6DOF:
                complexity += 2
            
            complexity += config.wind_strength
            complexity += (500 - config.world_size) / 100  # Larger world = more complex
            
            # Generally expect increasing complexity (allow some flexibility)
            if phase != CurriculumPhase.PHASE_1_BASIC_3DOF:
                assert complexity >= prev_complexity - 1, f"Phase {phase} complexity regression too large"
            
            prev_complexity = complexity
    
    def test_phase_advancement_criteria(self):
        """Test phase advancement criteria logic"""
        manager = CurriculumManager()
        
        # Test advancement from Phase 1
        manager.current_phase = CurriculumPhase.PHASE_1_BASIC_3DOF
        metrics = manager.phase_metrics[manager.current_phase]
        
        # Should not advance without meeting requirements
        assert not manager._should_advance_phase(), "Should not advance without meeting criteria"
        
        # Meet episode requirement but not performance
        metrics.episodes_completed = 1000
        metrics.recent_success_rate = 0.5  # Below threshold
        assert not manager._should_advance_phase(), "Should not advance with low success rate"
        
        # Meet all requirements
        metrics.recent_success_rate = 0.8  # Above threshold
        metrics.recent_average_reward = 10.0  # Above threshold
        assert manager._should_advance_phase(), "Should advance when all criteria met"
    
    def test_automatic_phase_advancement(self):
        """Test automatic phase advancement functionality"""
        manager = CurriculumManager()
        initial_phase = manager.current_phase
        
        # Simulate meeting advancement criteria
        metrics = manager.phase_metrics[manager.current_phase]
        config = manager.phase_configs[manager.current_phase]
        
        # Force advancement criteria to be met
        metrics.episodes_completed = config.episodes_required
        metrics.recent_success_rate = config.success_rate_threshold + 0.1
        metrics.recent_average_reward = config.average_reward_threshold + 1.0
        
        # Update performance to trigger advancement check
        manager.update_performance(15.0, True, 50.0, 8.0)
        
        # Should have advanced to next phase
        assert manager.current_phase != initial_phase, "Should have advanced to next phase"
        
        # Should have reset scenario counter
        assert manager.current_scenario == 0, "Scenario counter should reset on phase advancement"
    
    def test_manual_phase_setting(self):
        """Test manual phase setting functionality"""
        manager = CurriculumManager()
        
        # Test valid phase setting
        target_phase = CurriculumPhase.PHASE_4_FULL_6DOF
        manager.set_phase(target_phase)
        
        assert manager.current_phase == target_phase, "Phase should be set correctly"
        assert manager.current_scenario == 0, "Scenario should reset"
        
        # Test invalid phase setting
        with pytest.raises(ValueError):
            manager.set_phase("invalid_phase")
    
    def test_consecutive_success_criteria(self):
        """Test consecutive success advancement criteria"""
        manager = CurriculumManager()
        
        # Modify criteria to use consecutive successes
        config = manager.phase_configs[manager.current_phase]
        config.advancement_criteria = [AdvancementCriteria.CONSECUTIVE_SUCCESSES]
        config.episodes_required = 50  # Lower for testing
        
        metrics = manager.phase_metrics[manager.current_phase]
        
        # Add mixed results
        for i in range(60):
            success = i >= 50  # Last 10 should be successful
            manager.update_performance(10.0 if success else 5.0, success, 50.0, 8.0)
        
        # Should advance with 10 consecutive successes
        assert manager._should_advance_phase(), "Should advance with consecutive successes"


class TestPerformanceMetricsTracking:
    """Test performance metrics tracking accuracy"""
    
    def test_performance_metrics_calculation(self):
        """Test accuracy of performance metrics calculations"""
        metrics = PerformanceMetrics()
        
        # Test series of episodes
        test_data = [
            (10.0, True, 30.0, 5.0),   # reward, success, fuel, time
            (5.0, False, 45.0, 8.0),
            (15.0, True, 25.0, 4.0),
            (8.0, True, 35.0, 6.0),
            (-2.0, False, 50.0, 10.0),
        ]
        
        for reward, success, fuel, time_to_intercept in test_data:
            metrics.update(reward, success, fuel, time_to_intercept)
        
        # Verify calculations
        expected_episodes = 5
        expected_successes = 3
        expected_success_rate = 3/5
        expected_avg_reward = (10 + 5 + 15 + 8 - 2) / 5
        expected_fuel_efficiency = (30 + 25 + 35) / 3  # Only successful episodes
        expected_avg_time = (5 + 4 + 6) / 3  # Only successful episodes
        
        assert metrics.episodes_completed == expected_episodes
        assert metrics.episodes_successful == expected_successes
        assert abs(metrics.success_rate - expected_success_rate) < 1e-10
        assert abs(metrics.average_reward - expected_avg_reward) < 1e-10
        assert abs(metrics.average_fuel_efficiency - expected_fuel_efficiency) < 1e-10
        assert abs(metrics.average_intercept_time - expected_avg_time) < 1e-10
    
    def test_recent_performance_tracking(self):
        """Test recent performance tracking for stability"""
        metrics = PerformanceMetrics(recent_episodes=5)
        
        # Add more episodes than recent window
        for i in range(10):
            success = i >= 7  # Last 3 successful
            reward = 10.0 if success else 2.0
            metrics.update(reward, success, 30.0, 5.0)
        
        # Recent metrics should only consider last 5 episodes
        expected_recent_successes = 3  # Episodes 7, 8, 9 were successful
        expected_recent_success_rate = 3/5
        
        assert abs(metrics.recent_success_rate - expected_recent_success_rate) < 1e-10
        
        # Check that tracking arrays are limited
        assert len(metrics.episode_rewards) == 5
        assert len(metrics.episode_success) == 5
    
    def test_edge_case_metrics(self):
        """Test metrics calculation edge cases"""
        metrics = PerformanceMetrics()
        
        # Test with no episodes
        assert metrics.success_rate == 0.0
        assert metrics.average_reward == 0.0
        assert metrics.average_fuel_efficiency == 0.0
        assert metrics.average_intercept_time == 0.0
        
        # Test with no successes
        metrics.update(-5.0, False, 50.0, 0.0)
        metrics.update(-3.0, False, 45.0, 0.0)
        
        assert metrics.success_rate == 0.0
        assert metrics.average_fuel_efficiency == 0.0
        assert metrics.average_intercept_time == 0.0
        assert metrics.average_reward < 0


class TestScenarioConfiguration:
    """Test scenario configuration and management"""
    
    def test_default_scenario_loading(self):
        """Test loading of default scenarios for each phase"""
        manager = CurriculumManager()
        
        for phase in CurriculumPhase:
            if phase in manager.scenarios:
                scenarios = manager.scenarios[phase]
                assert len(scenarios) > 0, f"Phase {phase} should have scenarios"
                
                for scenario in scenarios:
                    # Validate scenario structure
                    assert hasattr(scenario, 'name'), "Scenario should have name"
                    assert hasattr(scenario, 'description'), "Scenario should have description"
                    assert hasattr(scenario, 'interceptor_position_range'), "Should have interceptor position range"
                    assert hasattr(scenario, 'missile_position_range'), "Should have missile position range"
                    
                    # Validate range structure
                    pos_range = scenario.interceptor_position_range
                    assert 'x' in pos_range and 'y' in pos_range and 'z' in pos_range
                    
                    for coord in ['x', 'y', 'z']:
                        range_tuple = pos_range[coord]
                        assert len(range_tuple) == 2, f"Range for {coord} should be tuple of 2"
                        assert range_tuple[0] <= range_tuple[1], f"Range min should be <= max for {coord}"
    
    def test_scenario_cycling(self):
        """Test scenario cycling within phases"""
        manager = CurriculumManager()
        
        # Get scenarios for current phase
        current_scenarios = manager.scenarios.get(manager.current_phase, [])
        if not current_scenarios:
            pytest.skip("No scenarios defined for current phase")
        
        # Test cycling through scenarios
        initial_scenario = manager.current_scenario
        scenario_configs = []
        
        for i in range(len(current_scenarios) + 2):  # Go past end to test cycling
            config = manager.get_current_scenario_config()
            scenario_configs.append(config)
            manager.current_scenario = (manager.current_scenario + 1) % len(current_scenarios)
        
        # Should cycle back to beginning
        assert scenario_configs[0].name == scenario_configs[len(current_scenarios)].name
    
    def test_scenario_parameter_validation(self):
        """Test validation of scenario parameters"""
        manager = CurriculumManager()
        
        # Test valid scenario
        valid_scenario = ScenarioConfig(
            name="test_scenario",
            description="Test scenario",
            interceptor_position_range={"x": (0, 100), "y": (0, 100), "z": (0, 50)},
            interceptor_velocity_range={"x": (-10, 10), "y": (-10, 10), "z": (0, 20)},
            missile_position_range={"x": (200, 300), "y": (200, 300), "z": (100, 200)},
            missile_velocity_range={"x": (-50, 50), "y": (-50, 50), "z": (-20, 0)},
            target_position_range={"x": (45, 55), "y": (45, 55), "z": (0, 10)},
            wind_conditions={"enabled": True},
            atmospheric_conditions={"standard": True}
        )
        
        # Should create without errors
        assert valid_scenario.name == "test_scenario"
        assert valid_scenario.evasion_aggressiveness == 1.0  # Default value


class TestJSONConfigurationHandling:
    """Test JSON configuration loading and validation"""
    
    def test_curriculum_config_save_load(self):
        """Test saving and loading curriculum configuration"""
        manager = CurriculumManager()
        
        # Modify some configurations
        config = manager.phase_configs[CurriculumPhase.PHASE_1_BASIC_3DOF]
        config.success_rate_threshold = 0.85
        config.episodes_required = 750
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            manager.save_curriculum_config(config_path)
            
            # Create new manager and load config
            new_manager = CurriculumManager()
            original_threshold = new_manager.phase_configs[CurriculumPhase.PHASE_1_BASIC_3DOF].success_rate_threshold
            
            new_manager.load_curriculum_config(config_path)
            
            # Should have loaded modified values
            loaded_config = new_manager.phase_configs[CurriculumPhase.PHASE_1_BASIC_3DOF]
            assert loaded_config.success_rate_threshold == 0.85
            assert loaded_config.episodes_required == 750
            
        finally:
            os.unlink(config_path)
    
    def test_invalid_json_config_handling(self):
        """Test handling of invalid JSON configurations"""
        manager = CurriculumManager()
        
        # Test with invalid JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            invalid_path = f.name
        
        try:
            # Should handle gracefully without crashing
            manager.load_curriculum_config(invalid_path)
            # Should still be functional
            assert manager.current_phase == CurriculumPhase.PHASE_1_BASIC_3DOF
            
        finally:
            os.unlink(invalid_path)
    
    def test_partial_config_loading(self):
        """Test loading configuration with missing fields"""
        manager = CurriculumManager()
        
        # Create partial configuration
        partial_config = {
            "phases": {
                "phase_1_basic_3dof": {
                    "success_rate_threshold": 0.9,
                    "episodes_required": 999
                    # Missing other fields
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(partial_config, f)
            config_path = f.name
        
        try:
            manager.load_curriculum_config(config_path)
            
            # Should have updated specified fields
            config = manager.phase_configs[CurriculumPhase.PHASE_1_BASIC_3DOF]
            assert config.success_rate_threshold == 0.9
            assert config.episodes_required == 999
            
            # Should have kept default values for other fields
            assert config.world_size == 300.0  # Default value
            
        finally:
            os.unlink(config_path)
    
    def test_scenario_config_json_loading(self):
        """Test loading scenario configurations from JSON"""
        manager = CurriculumManager()
        
        # Create scenario configuration
        scenario_config = {
            "scenarios": {
                "phase_1_basic_3dof": [
                    {
                        "name": "custom_scenario",
                        "description": "Custom test scenario",
                        "interceptor_position_range": {"x": [10, 20], "y": [10, 20], "z": [0, 5]},
                        "interceptor_velocity_range": {"x": [-5, 5], "y": [-5, 5], "z": [0, 10]},
                        "missile_position_range": {"x": [100, 200], "y": [100, 200], "z": [50, 100]},
                        "missile_velocity_range": {"x": [-30, 30], "y": [-30, 30], "z": [-10, 0]},
                        "target_position_range": {"x": [15, 25], "y": [15, 25], "z": [0, 5]},
                        "wind_conditions": {"enabled": False},
                        "atmospheric_conditions": {"standard": True},
                        "evasion_aggressiveness": 0.5
                    }
                ]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(scenario_config, f)
            config_path = f.name
        
        try:
            manager.load_curriculum_config(config_path)
            
            # Should have loaded custom scenario
            scenarios = manager.scenarios[CurriculumPhase.PHASE_1_BASIC_3DOF]
            custom_scenario = scenarios[0]  # Should replace default scenarios
            
            assert custom_scenario.name == "custom_scenario"
            assert custom_scenario.evasion_aggressiveness == 0.5
            
        finally:
            os.unlink(config_path)


class TestDynamicDifficultyAdjustment:
    """Test dynamic difficulty adjustment system"""
    
    def test_difficulty_increase_conditions(self):
        """Test conditions that trigger difficulty increase"""
        manager = CurriculumManager(enable_dynamic_difficulty=True)
        
        # Simulate high success rate
        metrics = manager.phase_metrics[manager.current_phase]
        metrics.episodes_completed = 200  # Trigger adjustment check
        metrics.recent_success_rate = 0.9  # High success rate
        
        original_config = manager.phase_configs[manager.current_phase]
        original_threshold = original_config.intercept_threshold
        original_wind = original_config.wind_strength
        
        # Force adjustment timing
        manager.last_adjustment_time = time.time() - 400  # 6+ minutes ago
        
        # Trigger adjustment
        manager._adjust_difficulty()
        
        # Should have increased difficulty
        assert original_config.intercept_threshold <= original_threshold, "Should reduce intercept threshold"
        if original_config.enable_wind:
            assert original_config.wind_strength >= original_wind, "Should increase wind strength"
    
    def test_difficulty_decrease_conditions(self):
        """Test conditions that trigger difficulty decrease"""
        manager = CurriculumManager(enable_dynamic_difficulty=True)
        
        # Simulate low success rate
        metrics = manager.phase_metrics[manager.current_phase]
        metrics.episodes_completed = 200  # Trigger adjustment check
        metrics.recent_success_rate = 0.3  # Low success rate
        
        original_config = manager.phase_configs[manager.current_phase]
        original_threshold = original_config.intercept_threshold
        original_wind = original_config.wind_strength
        
        # Force adjustment timing
        manager.last_adjustment_time = time.time() - 400
        
        # Trigger adjustment
        manager._adjust_difficulty()
        
        # Should have decreased difficulty
        assert original_config.intercept_threshold >= original_threshold, "Should increase intercept threshold"
        if original_config.enable_wind:
            assert original_config.wind_strength <= original_wind, "Should decrease wind strength"
    
    def test_adjustment_frequency_limiting(self):
        """Test that difficulty adjustments are rate-limited"""
        manager = CurriculumManager(enable_dynamic_difficulty=True)
        
        metrics = manager.phase_metrics[manager.current_phase]
        metrics.episodes_completed = 200
        metrics.recent_success_rate = 0.9
        
        # Recent adjustment
        manager.last_adjustment_time = time.time() - 100  # 1.6 minutes ago
        
        original_threshold = manager.phase_configs[manager.current_phase].intercept_threshold
        
        # Should not adjust due to timing
        manager._adjust_difficulty()
        
        # Threshold should not have changed
        assert manager.phase_configs[manager.current_phase].intercept_threshold == original_threshold
    
    def test_adjustment_history_tracking(self):
        """Test that difficulty adjustments are properly logged"""
        manager = CurriculumManager(enable_dynamic_difficulty=True)
        
        metrics = manager.phase_metrics[manager.current_phase]
        metrics.episodes_completed = 200
        metrics.recent_success_rate = 0.9
        
        manager.last_adjustment_time = time.time() - 400
        
        initial_history_length = len(manager.difficulty_adjustment_history)
        
        # Trigger adjustment
        manager._adjust_difficulty()
        
        # Should have added history entry
        assert len(manager.difficulty_adjustment_history) == initial_history_length + 1
        
        # Check history entry structure
        if manager.difficulty_adjustment_history:
            entry = manager.difficulty_adjustment_history[-1]
            assert 'time' in entry
            assert 'phase' in entry
            assert 'adjustment' in entry
            assert entry['adjustment'] in ['increase', 'decrease']


class TestCurriculumIntegration:
    """Test curriculum integration with environment"""
    
    def test_environment_config_generation(self):
        """Test environment configuration generation"""
        manager = CurriculumManager()
        
        for phase in CurriculumPhase:
            manager.set_phase(phase)
            env_config = manager.get_environment_config()
            
            # Should contain required fields
            required_fields = [
                'difficulty_mode', 'action_mode', 'world_size', 'max_steps',
                'intercept_threshold', 'enable_wind', 'enable_atmosphere'
            ]
            
            for field in required_fields:
                assert field in env_config, f"Missing field {field} in config"
            
            # Values should be appropriate for phase
            assert isinstance(env_config['difficulty_mode'], DifficultyMode)
            assert isinstance(env_config['action_mode'], ActionMode)
            assert env_config['world_size'] > 0
            assert env_config['max_steps'] > 0
    
    def test_curriculum_status_reporting(self):
        """Test curriculum status reporting"""
        manager = CurriculumManager()
        
        # Add some performance data
        manager.update_performance(10.0, True, 30.0, 5.0)
        manager.update_performance(8.0, False, 45.0, 8.0)
        
        status = manager.get_curriculum_status()
        
        # Should contain expected fields
        assert 'current_phase' in status
        assert 'phase_progress' in status
        assert 'advancement_criteria' in status
        assert 'all_phases_metrics' in status
        
        # Phase progress should have performance data
        progress = status['phase_progress']
        assert progress['episodes_completed'] == 2
        assert progress['success_rate'] == 0.5
        
        # All phases should be represented
        assert len(status['all_phases_metrics']) == len(CurriculumPhase)
    
    def test_phase_reset_functionality(self):
        """Test phase reset functionality"""
        manager = CurriculumManager()
        
        # Add performance data
        manager.update_performance(10.0, True, 30.0, 5.0)
        manager.update_performance(8.0, False, 45.0, 8.0)
        
        phase = manager.current_phase
        assert manager.phase_metrics[phase].episodes_completed == 2
        
        # Reset phase
        manager.reset_phase(phase)
        
        # Should be reset to zero
        assert manager.phase_metrics[phase].episodes_completed == 0
        assert manager.phase_metrics[phase].episodes_successful == 0
        assert len(manager.phase_metrics[phase].episode_rewards) == 0


class TestCurriculumUtilities:
    """Test curriculum utility functions"""
    
    def test_curriculum_manager_creation(self):
        """Test curriculum manager creation utility"""
        manager = create_curriculum_manager()
        
        assert isinstance(manager, CurriculumManager)
        assert manager.current_phase == CurriculumPhase.PHASE_1_BASIC_3DOF
    
    def test_directory_setup(self):
        """Test curriculum directory setup utility"""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.join(temp_dir, "test_curriculum")
            
            dirs = setup_curriculum_directories(base_path)
            
            # Check that all directories were created
            expected_dirs = ['base', 'configs', 'logs', 'checkpoints', 'scenarios']
            for dir_key in expected_dirs:
                assert dir_key in dirs
                assert os.path.exists(dirs[dir_key])
                assert os.path.isdir(dirs[dir_key])
    
    def test_config_path_handling(self):
        """Test configuration path handling"""
        # Test with non-existent config path
        manager = CurriculumManager(config_path="/non/existent/path.json")
        
        # Should still initialize with defaults
        assert manager.current_phase == CurriculumPhase.PHASE_1_BASIC_3DOF
        assert len(manager.phase_configs) == len(CurriculumPhase)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])