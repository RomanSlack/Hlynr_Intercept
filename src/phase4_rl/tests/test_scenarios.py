"""
Tests for scenario loading and management system.
"""

import pytest
import tempfile
import json
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios import ScenarioLoader, get_scenario_loader, reset_scenario_loader


class TestScenarioLoader:
    """Test cases for ScenarioLoader class."""
    
    def setup_method(self):
        """Setup for each test method."""
        reset_scenario_loader()
        
        # Create temporary directory with test scenarios
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir_path = Path(self.temp_dir)
        
        # Create test scenario files
        self._create_test_scenarios()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_scenarios(self):
        """Create test scenario files."""
        # Valid test scenario
        valid_scenario = {
            "name": "test_easy",
            "description": "Test easy scenario",
            "difficulty_level": 1,
            "num_missiles": 1,
            "num_interceptors": 1,
            "spawn_positions": {
                "missiles": [[0, 0]],
                "interceptors": [[100, 100]],
                "targets": [[200, 200]]
            },
            "wind_settings": {
                "speed": 5.0,
                "direction": 0.0,
                "variability": 0.1
            },
            "adversary_config": {
                "speed_multiplier": 1.0,
                "evasion_iq": 0.2
            },
            "radar_config": {
                "range": 1000.0,
                "noise_level": 0.05
            }
        }
        
        with open(self.temp_dir_path / "test_easy.json", 'w') as f:
            json.dump(valid_scenario, f)
        
        # Invalid scenario (missing required fields)
        invalid_scenario = {
            "name": "test_invalid",
            "description": "Invalid test scenario"
            # Missing required fields
        }
        
        with open(self.temp_dir_path / "test_invalid.json", 'w') as f:
            json.dump(invalid_scenario, f)
        
        # Scenario with mismatched entity counts
        mismatch_scenario = {
            "name": "test_mismatch",
            "description": "Test scenario with mismatched counts",
            "difficulty_level": 2,
            "num_missiles": 2,  # Says 2 missiles
            "num_interceptors": 1,
            "spawn_positions": {
                "missiles": [[0, 0]],  # But only 1 spawn position
                "interceptors": [[100, 100]],
                "targets": [[200, 200]]
            },
            "wind_settings": {"speed": 5.0, "direction": 0.0, "variability": 0.1},
            "adversary_config": {"speed_multiplier": 1.0, "evasion_iq": 0.2},
            "radar_config": {"range": 1000.0, "noise_level": 0.05}
        }
        
        with open(self.temp_dir_path / "test_mismatch.json", 'w') as f:
            json.dump(mismatch_scenario, f)
    
    def test_scenario_discovery(self):
        """Test automatic scenario discovery."""
        loader = ScenarioLoader(self.temp_dir)
        scenarios = loader.list_scenarios()
        
        assert "test_easy" in scenarios
        assert "test_invalid" in scenarios
        assert "test_mismatch" in scenarios
        assert len(scenarios) == 3
    
    def test_valid_scenario_loading(self):
        """Test loading of valid scenario."""
        loader = ScenarioLoader(self.temp_dir)
        scenario = loader.load_scenario("test_easy")
        
        assert scenario["name"] == "test_easy"
        assert scenario["difficulty_level"] == 1
        assert scenario["num_missiles"] == 1
        assert scenario["num_interceptors"] == 1
        assert len(scenario["spawn_positions"]["missiles"]) == 1
    
    def test_invalid_scenario_loading(self):
        """Test loading of invalid scenario raises appropriate error."""
        loader = ScenarioLoader(self.temp_dir)
        
        with pytest.raises(ValueError, match="missing required fields"):
            loader.load_scenario("test_invalid")
    
    def test_mismatched_entity_counts(self):
        """Test scenario with mismatched entity counts raises error."""
        loader = ScenarioLoader(self.temp_dir)
        
        with pytest.raises(ValueError, match="doesn't match spawn positions"):
            loader.load_scenario("test_mismatch")
    
    def test_nonexistent_scenario(self):
        """Test loading nonexistent scenario raises appropriate error."""
        loader = ScenarioLoader(self.temp_dir)
        
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_scenario("nonexistent_scenario")
    
    def test_scenario_caching(self):
        """Test that scenarios are cached after first load."""
        loader = ScenarioLoader(self.temp_dir)
        
        # First load
        scenario1 = loader.load_scenario("test_easy")
        
        # Second load should return cached version
        scenario2 = loader.load_scenario("test_easy")
        
        # Should be equal but different objects (copy)
        assert scenario1 == scenario2
        assert scenario1 is not scenario2
    
    def test_difficulty_level_mapping(self):
        """Test difficulty level mapping functionality."""
        loader = ScenarioLoader(self.temp_dir)
        
        scenario_name = loader.get_scenario_by_difficulty(1)
        assert scenario_name == "test_easy"
        
        nonexistent = loader.get_scenario_by_difficulty(99)
        assert nonexistent is None
        
        difficulty_map = loader.get_scenarios_by_difficulty()
        assert 1 in difficulty_map
        assert difficulty_map[1] == "test_easy"
    
    def test_config_merging(self):
        """Test merging scenario config with base config."""
        loader = ScenarioLoader(self.temp_dir)
        
        base_config = {
            'environment': {'max_episode_steps': 1000},
            'radar': {'update_rate': 10},
            'spawn': {},
            'environment_conditions': {}
        }
        
        scenario_config = loader.load_scenario("test_easy")
        merged = loader.merge_with_base_config(scenario_config, base_config)
        
        # Check that scenario values override base config
        assert merged['environment']['num_missiles'] == 1
        assert merged['environment']['num_interceptors'] == 1
        
        # Check that base config values are preserved
        assert merged['environment']['max_episode_steps'] == 1000
        
        # Check radar config merging
        assert merged['radar']['range'] == 1000.0
        assert merged['radar']['update_rate'] == 10  # From base config
        
        # Check scenario metadata is added
        assert 'scenario' in merged
        assert merged['scenario']['name'] == "test_easy"
    
    def test_environment_config_creation(self):
        """Test creation of complete environment configuration."""
        loader = ScenarioLoader(self.temp_dir)
        
        # Mock base configuration
        from config import ConfigLoader
        config_loader = ConfigLoader()
        base_config = config_loader._config
        
        env_config = loader.create_environment_config("test_easy", base_config)
        
        # Should have scenario-specific settings
        assert env_config['environment']['num_missiles'] == 1
        assert env_config['environment']['num_interceptors'] == 1
        
        # Should have scenario metadata
        assert 'scenario' in env_config
        assert env_config['scenario']['name'] == "test_easy"
    
    def test_scenario_validation(self):
        """Test validation of all scenarios."""
        loader = ScenarioLoader(self.temp_dir)
        
        validation_results = loader.validate_all_scenarios()
        
        assert validation_results["test_easy"] is True
        assert validation_results["test_invalid"] is False
        assert validation_results["test_mismatch"] is False
    
    def test_global_scenario_loader(self):
        """Test global scenario loader instance management."""
        # First call should create instance
        loader1 = get_scenario_loader(self.temp_dir)
        
        # Second call should return same instance
        loader2 = get_scenario_loader()
        assert loader1 is loader2
        
        # Reset should clear instance
        reset_scenario_loader()
        loader3 = get_scenario_loader(self.temp_dir)
        assert loader3 is not loader1


class TestScenarioValidation:
    """Test cases for scenario validation logic."""
    
    def test_missing_required_fields(self):
        """Test validation of scenarios missing required fields."""
        loader = ScenarioLoader()
        
        incomplete_scenario = {
            "name": "incomplete",
            "description": "Missing fields"
            # Missing many required fields
        }
        
        with pytest.raises(ValueError, match="missing required fields"):
            loader._validate_scenario(incomplete_scenario, "incomplete")
    
    def test_invalid_difficulty_level(self):
        """Test validation of invalid difficulty levels."""
        loader = ScenarioLoader()
        
        # Create minimal valid scenario
        scenario = {
            "name": "test",
            "description": "Test",
            "difficulty_level": 5,  # Invalid (should be 1-4)
            "num_missiles": 1,
            "num_interceptors": 1,
            "spawn_positions": {
                "missiles": [[0, 0]],
                "interceptors": [[100, 100]],
                "targets": [[200, 200]]
            },
            "wind_settings": {},
            "adversary_config": {},
            "radar_config": {}
        }
        
        with pytest.raises(ValueError, match="difficulty_level must be integer 1-4"):
            loader._validate_scenario(scenario, "test")
    
    def test_spawn_position_validation(self):
        """Test validation of spawn positions."""
        loader = ScenarioLoader()
        
        # Missing spawn position fields
        scenario_missing_spawn = {
            "name": "test",
            "description": "Test", 
            "difficulty_level": 1,
            "num_missiles": 1,
            "num_interceptors": 1,
            "spawn_positions": {
                "missiles": [[0, 0]]
                # Missing interceptors and targets
            },
            "wind_settings": {},
            "adversary_config": {},
            "radar_config": {}
        }
        
        with pytest.raises(ValueError, match="spawn_positions missing fields"):
            loader._validate_scenario(scenario_missing_spawn, "test")


if __name__ == '__main__':
    pytest.main([__file__])