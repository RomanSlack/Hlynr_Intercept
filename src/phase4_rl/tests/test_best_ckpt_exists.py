#!/usr/bin/env python3
"""
Tests for best model checkpointing functionality.

Verifies that the BestModelCallback correctly saves the best model
when evaluation performance improves, and that the files are created
with the correct naming convention.
"""

import pytest
import numpy as np
import tempfile
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Import the callback to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_callbacks import BestModelCallback


class TestBestModelCallback:
    """Test suite for BestModelCallback."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = Path(self.temp_dir) / "best_model"
        
        # Create callback instance
        self.callback = BestModelCallback(
            monitor_key="eval/mean_reward",
            save_path=str(self.save_path),
            name_prefix="best",
            save_freq=1000,
            verbose=0
        )
        
        # Mock model and environment
        self.mock_model = Mock()
        self.mock_model.save = Mock()
        self.mock_logger = Mock()
        self.mock_logger.name_to_value = {}
        self.mock_training_env = Mock()
        self.mock_training_env.save = Mock()
        
        # Setup callback with mocks
        self.callback.model = self.mock_model
        self.callback.logger = self.mock_logger  # This is the BaseCallback logger property
        self.callback.training_env = self.mock_training_env
        self.callback.locals = {}
        self.callback.globals = {}
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test callback initialization."""
        callback = BestModelCallback(
            monitor_key="test/reward",
            save_path="test_path",
            name_prefix="test",
            save_freq=5000,
            verbose=1
        )
        
        assert callback.monitor_key == "test/reward"
        assert callback.save_path == Path("test_path")
        assert callback.name_prefix == "test"
        assert callback.save_freq == 5000
        assert callback.best_value == -np.inf
        assert callback.n_saves == 0
    
    def test_training_start_creates_directory(self):
        """Test that training start creates the save directory."""
        self.callback._on_training_start()
        
        # Directory should be created
        assert self.save_path.exists()
        assert self.save_path.is_dir()
    
    def test_save_on_improvement(self):
        """Test that model is saved when performance improves."""
        self.callback._on_training_start()
        
        # Setup mock logger with improving rewards
        rewards = [10.0, 15.0, 12.0, 20.0]  # 15.0 and 20.0 should trigger saves
        
        for i, reward in enumerate(rewards):
            self.callback.num_timesteps = (i + 1) * 1000  # Align with save_freq
            self.mock_logger.name_to_value = {"eval/mean_reward": reward}
            
            self.callback._on_step()
        
        # Should have saved 2 times (for 15.0 and 20.0)
        assert self.callback.n_saves == 2
        assert self.callback.best_value == 20.0
        assert self.callback.best_timestep == 4000
        
        # Model.save should have been called
        assert self.mock_model.save.call_count == 2
    
    def test_no_save_without_improvement(self):
        """Test that model is not saved when performance doesn't improve."""
        self.callback._on_training_start()
        
        # Setup mock logger with non-improving rewards
        rewards = [10.0, 8.0, 9.0, 7.0]  # Only first should trigger save
        
        for i, reward in enumerate(rewards):
            self.callback.num_timesteps = (i + 1) * 1000
            self.mock_logger.name_to_value = {"eval/mean_reward": reward}
            
            self.callback._on_step()
        
        # Should have saved only once (for the first 10.0)
        assert self.callback.n_saves == 1
        assert self.callback.best_value == 10.0
        assert self.mock_model.save.call_count == 1
    
    def test_save_frequency_respected(self):
        """Test that saves only happen at specified frequency."""
        self.callback._on_training_start()
        
        # Setup improving reward but wrong timesteps
        self.mock_logger.name_to_value = {"eval/mean_reward": 15.0}
        
        # These timesteps shouldn't trigger save (not multiples of save_freq)
        for timestep in [500, 1500, 2300]:
            self.callback.num_timesteps = timestep
            self.callback._on_step()
        
        # No saves should have occurred
        assert self.callback.n_saves == 0
        assert self.mock_model.save.call_count == 0
        
        # This should trigger save
        self.callback.num_timesteps = 2000  # Multiple of save_freq
        self.callback._on_step()
        
        assert self.callback.n_saves == 1
        assert self.mock_model.save.call_count == 1
    
    def test_save_file_naming(self):
        """Test that saved files use correct naming convention."""
        self.callback._on_training_start()
        
        # Trigger a save
        self.callback.num_timesteps = 1000
        self.mock_logger.name_to_value = {"eval/mean_reward": 15.0}
        self.callback._on_step()
        
        # Check that model.save was called with correct path
        expected_path = self.save_path / "best_model"
        actual_call = self.mock_model.save.call_args[0][0]
        assert actual_call == expected_path
    
    def test_metadata_saved(self):
        """Test that metadata is saved alongside the model."""
        self.callback._on_training_start()
        
        # Trigger a save
        self.callback.num_timesteps = 2000
        self.mock_logger.name_to_value = {"eval/mean_reward": 25.0}
        
        with patch('time.time', return_value=1234567890.0):
            self.callback._on_step()
        
        # Check that metadata file was created
        metadata_path = self.save_path / "best_metadata.json"
        assert metadata_path.exists()
        
        # Check metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['timestep'] == 2000
        assert metadata['best_value'] == 25.0
        assert metadata['metric'] == "eval/mean_reward"
        assert metadata['timestamp'] == 1234567890.0
        assert metadata['save_count'] == 1
        assert metadata['model_path'].endswith('best_model.zip')
    
    def test_vecnormalize_saving(self):
        """Test that VecNormalize parameters are saved when enabled."""
        # Create callback with VecNormalize saving enabled
        callback = BestModelCallback(
            save_path=str(self.save_path),
            save_vecnormalize=True,
            verbose=0
        )
        
        callback.model = self.mock_model
        callback.logger = self.mock_logger
        callback.training_env = self.mock_training_env
        callback._on_training_start()
        
        # Trigger a save
        callback.num_timesteps = 1000
        self.mock_logger.name_to_value = {"eval/mean_reward": 15.0}
        callback._on_step()
        
        # VecNormalize save should have been called
        expected_path = self.save_path / "best_vecnormalize.pkl"
        self.mock_training_env.save.assert_called_once_with(expected_path)
    
    def test_vecnormalize_saving_disabled(self):
        """Test that VecNormalize parameters are not saved when disabled."""
        # Create callback with VecNormalize saving disabled
        callback = BestModelCallback(
            save_path=str(self.save_path),
            save_vecnormalize=False,
            verbose=0
        )
        
        callback.model = self.mock_model
        callback.logger = self.mock_logger
        callback.training_env = self.mock_training_env
        callback._on_training_start()
        
        # Trigger a save
        callback.num_timesteps = 1000
        self.mock_logger.name_to_value = {"eval/mean_reward": 15.0}
        callback._on_step()
        
        # VecNormalize save should NOT have been called
        self.mock_training_env.save.assert_not_called()
    
    def test_missing_metric_handling(self):
        """Test callback behavior when monitored metric is missing."""
        self.callback._on_training_start()
        
        # No metric in logger
        self.mock_logger.name_to_value = {}
        
        # Should not crash or save
        self.callback.num_timesteps = 1000
        self.callback._on_step()
        
        assert self.callback.n_saves == 0
        assert self.mock_model.save.call_count == 0
    
    def test_custom_monitor_key(self):
        """Test callback with custom monitor key."""
        callback = BestModelCallback(
            monitor_key="custom/metric",
            save_path=str(self.save_path),
            verbose=0
        )
        
        callback.model = self.mock_model
        callback.logger = self.mock_logger
        callback._on_training_start()
        
        # Use custom metric
        callback.num_timesteps = 1000
        self.mock_logger.name_to_value = {"custom/metric": 42.0}
        callback._on_step()
        
        assert callback.n_saves == 1
        assert callback.best_value == 42.0
    
    def test_save_history_tracking(self):
        """Test that save history is properly tracked."""
        self.callback._on_training_start()
        
        # Multiple saves
        saves_data = [(1000, 10.0), (2000, 15.0), (3000, 20.0)]
        
        for timestep, reward in saves_data:
            self.callback.num_timesteps = timestep
            self.mock_logger.name_to_value = {"eval/mean_reward": reward}
            self.callback._on_step()
        
        # Check save history
        assert len(self.callback.save_history) == 3
        
        for i, (timestep, reward) in enumerate(saves_data):
            history_entry = self.callback.save_history[i]
            assert history_entry['timestep'] == timestep
            assert history_entry['best_value'] == reward
            assert history_entry['save_count'] == i + 1
    
    def test_training_end_summary(self):
        """Test training end summary logging."""
        self.callback._on_training_start()
        
        # Trigger some saves
        self.callback.num_timesteps = 1000
        self.mock_logger.name_to_value = {"eval/mean_reward": 15.0}
        self.callback._on_step()
        
        # Should not crash
        self.callback._on_training_end()
        
        assert self.callback.n_saves == 1
        assert self.callback.best_value == 15.0
    
    def test_training_end_no_saves(self):
        """Test training end when no saves occurred."""
        self.callback._on_training_start()
        
        # No saves triggered
        self.callback._on_training_end()
        
        # Should handle gracefully
        assert self.callback.n_saves == 0


class TestBestModelIntegration:
    """Integration tests for best model checkpointing."""
    
    def test_short_training_run_creates_best_model(self):
        """Test that a short training run creates a best model checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "best_model"
            
            callback = BestModelCallback(
                save_path=str(save_path),
                save_freq=100,  # Check every 100 steps
                verbose=1
            )
            
            # Mock model that actually creates files
            mock_model = Mock()
            def mock_save(path):
                # Create a dummy file to simulate model saving
                Path(str(path) + '.zip').touch()
            mock_model.save.side_effect = mock_save
            
            mock_logger = Mock()
            callback.model = mock_model
            callback.logger = mock_logger  # BaseCallback logger property
            
            # Simulate short training run with improving performance
            callback._on_training_start()
            
            rewards = [5.0, 8.0, 12.0, 10.0, 15.0]  # Improvements at 8.0, 12.0, 15.0
            
            for i, reward in enumerate(rewards):
                callback.num_timesteps = (i + 1) * 100
                mock_logger.name_to_value = {"eval/mean_reward": reward}
                callback._on_step()
            
            callback._on_training_end()
            
            # Check that best model files exist
            assert (save_path / "best_model.zip").exists()
            assert (save_path / "best_metadata.json").exists()
            
            # Check metadata
            with open(save_path / "best_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            assert metadata['best_value'] == 15.0
            assert metadata['save_count'] == 3  # Three improvements
            assert metadata['timestep'] == 500
    
    def test_no_improvement_no_best_model(self):
        """Test that no best model is saved if performance never improves."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "best_model"
            
            callback = BestModelCallback(
                save_path=str(save_path),
                save_freq=100,
                verbose=0
            )
            
            mock_model = Mock()
            mock_logger = Mock()
            callback.model = mock_model
            callback.logger = mock_logger  # BaseCallback logger property
            
            callback._on_training_start()
            
            # No improving rewards (all worse than -inf initial)
            # Wait, actually first reward will always be better than -inf
            # Let's test with decreasing rewards after first
            rewards = [10.0, 8.0, 6.0, 4.0]  # Only first should save
            
            for i, reward in enumerate(rewards):
                callback.num_timesteps = (i + 1) * 100
                mock_logger.name_to_value = {"eval/mean_reward": reward}
                callback._on_step()
            
            # Only one save should have occurred (for the first reward)
            assert callback.n_saves == 1
            assert callback.best_value == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])