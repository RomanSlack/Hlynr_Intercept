"""
Tests for configuration management system.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigLoader, get_config, reset_config


class TestConfigLoader:
    """Test cases for ConfigLoader class."""
    
    def setup_method(self):
        """Setup for each test method."""
        reset_config()
    
    def test_default_config_loading(self):
        """Test loading default configuration."""
        config = ConfigLoader()
        
        # Test that basic configuration sections exist
        assert 'environment' in config._config
        assert 'training' in config._config
        assert 'radar' in config._config
        
        # Test default values
        assert config.get('environment.num_missiles') == 1
        assert config.get('environment.num_interceptors') == 1
        assert config.get('training.algorithm') == 'PPO'
    
    def test_custom_config_file(self):
        """Test loading from custom configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            custom_config = {
                'environment': {
                    'num_missiles': 3,
                    'num_interceptors': 2
                },
                'training': {
                    'learning_rate': 0.001
                }
            }
            yaml.dump(custom_config, f)
            temp_path = f.name
        
        try:
            config = ConfigLoader(temp_path)
            assert config.get('environment.num_missiles') == 3
            assert config.get('environment.num_interceptors') == 2
            assert config.get('training.learning_rate') == 0.001
        finally:
            os.unlink(temp_path)
    
    def test_nonexistent_config_file(self):
        """Test handling of nonexistent configuration file."""
        config = ConfigLoader('/nonexistent/path/config.yaml')
        
        # Should fall back to default configuration
        assert config.get('environment.num_missiles') == 1
        assert config.get('training.algorithm') == 'PPO'
    
    def test_dot_notation_access(self):
        """Test dot notation for configuration access."""
        config = ConfigLoader()
        
        # Test getting values
        assert config.get('environment.num_missiles') == 1
        assert config.get('nonexistent.key', 'default') == 'default'
        
        # Test setting values
        config.set('environment.num_missiles', 5)
        assert config.get('environment.num_missiles') == 5
        
        # Test setting nested values
        config.set('new_section.new_key', 'new_value')
        assert config.get('new_section.new_key') == 'new_value'
    
    def test_config_update(self):
        """Test configuration updating."""
        config = ConfigLoader()
        
        update_data = {
            'environment': {
                'num_missiles': 10
            },
            'new_section': {
                'new_key': 'new_value'
            }
        }
        
        config.update(update_data)
        
        assert config.get('environment.num_missiles') == 10
        assert config.get('environment.num_interceptors') == 1  # Should preserve existing
        assert config.get('new_section.new_key') == 'new_value'
    
    def test_config_save(self):
        """Test configuration saving."""
        config = ConfigLoader()
        config.set('test.value', 42)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            
            # Load saved configuration
            with open(temp_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            assert saved_config['test']['value'] == 42
        finally:
            os.unlink(temp_path)
    
    def test_environment_config(self):
        """Test environment-specific configuration extraction."""
        config = ConfigLoader()
        env_config = config.get_environment_config()
        
        assert 'num_missiles' in env_config
        assert 'num_interceptors' in env_config
        assert 'radar_config' in env_config
        assert 'spawn_config' in env_config
    
    def test_training_config(self):
        """Test training-specific configuration extraction."""
        config = ConfigLoader()
        training_config = config.get_training_config()
        
        assert 'algorithm' in training_config
        assert 'learning_rate' in training_config
        assert 'total_timesteps' in training_config
    
    def test_global_config_instance(self):
        """Test global configuration instance management."""
        # First call should create instance
        config1 = get_config()
        
        # Second call should return same instance
        config2 = get_config()
        assert config1 is config2
        
        # Reset should clear instance
        reset_config()
        config3 = get_config()
        assert config3 is not config1


if __name__ == '__main__':
    pytest.main([__file__])