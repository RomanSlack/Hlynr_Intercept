#!/usr/bin/env python3
"""
Tests for entropy scheduling callback.

Verifies that the EntropyScheduleCallback correctly implements linear decay
with a floor value that is never violated.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Import the callback to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_callbacks import EntropyScheduleCallback


class TestEntropyScheduleCallback:
    """Test suite for EntropyScheduleCallback."""
    
    def setup_method(self):
        """Setup test environment."""
        self.initial_entropy = 0.02
        self.final_entropy = 0.01
        self.decay_steps = 1000
        
        # Create callback instance
        self.callback = EntropyScheduleCallback(
            initial_entropy=self.initial_entropy,
            final_entropy=self.final_entropy,
            decay_steps=self.decay_steps,
            verbose=0
        )
        
        # Mock model and logger
        self.mock_model = Mock()
        self.mock_model.ent_coef = self.initial_entropy
        self.mock_logger = Mock()
        
        # Setup callback with mocks
        self.callback.model = self.mock_model
        self.callback.logger = self.mock_logger  # This is the BaseCallback logger property
        self.callback.training_env = Mock()
        self.callback.locals = {}
        self.callback.globals = {}
    
    def test_initialization_valid_parameters(self):
        """Test callback initialization with valid parameters."""
        callback = EntropyScheduleCallback(
            initial_entropy=0.01,
            final_entropy=0.005,
            decay_steps=10000
        )
        
        assert callback.initial_entropy == 0.01
        assert callback.final_entropy == 0.005
        assert callback.decay_steps == 10000
        assert callback.current_entropy == 0.01
    
    def test_initialization_invalid_parameters(self):
        """Test callback initialization with invalid parameters."""
        # final_entropy >= initial_entropy
        with pytest.raises(ValueError, match="final_entropy.*must be less than initial_entropy"):
            EntropyScheduleCallback(
                initial_entropy=0.01,
                final_entropy=0.02,  # Greater than initial
                decay_steps=1000
            )
        
        # Negative final_entropy
        with pytest.raises(ValueError, match="final_entropy.*must be non-negative"):
            EntropyScheduleCallback(
                initial_entropy=0.01,
                final_entropy=-0.01,
                decay_steps=1000
            )
        
        # Non-positive decay_steps
        with pytest.raises(ValueError, match="decay_steps.*must be positive"):
            EntropyScheduleCallback(
                initial_entropy=0.01,
                final_entropy=0.005,
                decay_steps=0
            )
    
    def test_training_start(self):
        """Test callback behavior at training start."""
        self.callback._on_training_start()
        
        # Should set model's entropy coefficient to initial value
        assert self.mock_model.ent_coef == self.initial_entropy
        assert self.callback.current_entropy == self.initial_entropy
    
    def test_linear_decay_progression(self):
        """Test that entropy decays linearly over specified steps."""
        self.callback._on_training_start()
        
        # Test at various points in training
        test_points = [0, 250, 500, 750, 1000, 1500]  # Beyond decay_steps too
        expected_entropies = []
        
        for timestep in test_points:
            self.callback.num_timesteps = timestep
            self.callback._on_step()
            
            # Calculate expected entropy
            progress = min(timestep / self.decay_steps, 1.0)
            expected = self.initial_entropy + progress * (self.final_entropy - self.initial_entropy)
            expected = max(expected, self.final_entropy)  # Ensure floor is respected
            
            expected_entropies.append(expected)
            
            # Check that current entropy matches expected
            assert abs(self.callback.current_entropy - expected) < 1e-6, \
                f"At step {timestep}: expected {expected}, got {self.callback.current_entropy}"
            
            # Check that model's entropy coefficient is updated
            assert abs(self.mock_model.ent_coef - expected) < 1e-6
        
        # Verify monotonic increase (since final > initial in this test)
        for i in range(1, len(expected_entropies)):
            assert expected_entropies[i] >= expected_entropies[i-1], \
                "Entropy should increase monotonically when final > initial"
    
    def test_entropy_floor_never_violated(self):
        """Test that entropy never goes below the specified floor."""
        # Test with decreasing entropy (more common case)
        callback = EntropyScheduleCallback(
            initial_entropy=0.1,
            final_entropy=0.02,  # Floor value
            decay_steps=1000,
            verbose=0
        )
        
        callback.model = self.mock_model
        callback.logger = self.mock_logger
        callback._on_training_start()
        
        # Test over extended training period (beyond decay_steps)
        for timestep in range(0, 2000, 50):
            callback.num_timesteps = timestep
            callback._on_step()
            
            # Entropy should never go below floor
            assert callback.current_entropy >= callback.final_entropy, \
                f"At step {timestep}: entropy {callback.current_entropy} below floor {callback.final_entropy}"
            
            # Model should reflect this
            assert self.mock_model.ent_coef >= callback.final_entropy
    
    def test_entropy_reaches_floor_exactly(self):
        """Test that entropy reaches exactly the floor value after decay_steps."""
        callback = EntropyScheduleCallback(
            initial_entropy=0.1,
            final_entropy=0.02,
            decay_steps=1000,
            verbose=0
        )
        
        callback.model = self.mock_model
        callback.logger = self.mock_logger
        callback._on_training_start()
        
        # Test at and beyond decay_steps
        for timestep in [1000, 1500, 2000]:
            callback.num_timesteps = timestep
            callback._on_step()
            
            # Should be at floor value
            assert abs(callback.current_entropy - callback.final_entropy) < 1e-6, \
                f"At step {timestep}: expected floor {callback.final_entropy}, got {callback.current_entropy}"
    
    def test_logging_calls(self):
        """Test that callback logs entropy coefficient values."""
        self.callback._on_training_start()
        
        # Simulate a few steps
        for timestep in [0, 500, 1000]:
            self.callback.num_timesteps = timestep
            self.callback._on_step()
        
        # Should have recorded entropy values
        expected_calls = len([0, 500, 1000])
        assert self.mock_logger.record.call_count == expected_calls
        
        # Check that it's logging the right key
        call_args_list = self.mock_logger.record.call_args_list
        for call_args in call_args_list:
            args, kwargs = call_args
            assert args[0] == "train/entropy_coef"
            assert isinstance(args[1], (int, float))
    
    def test_zero_decay_steps(self):
        """Test edge case with very small decay_steps."""
        with pytest.raises(ValueError):
            EntropyScheduleCallback(
                initial_entropy=0.01,
                final_entropy=0.005,
                decay_steps=0
            )
    
    def test_single_step_decay(self):
        """Test decay over a single step."""
        callback = EntropyScheduleCallback(
            initial_entropy=0.1,
            final_entropy=0.02,
            decay_steps=1,
            verbose=0
        )
        
        callback.model = self.mock_model
        callback.logger = self.mock_logger
        callback._on_training_start()
        
        # At step 0
        callback.num_timesteps = 0
        callback._on_step()
        assert abs(callback.current_entropy - 0.1) < 1e-6
        
        # At step 1 (should reach floor)
        callback.num_timesteps = 1
        callback._on_step()
        assert abs(callback.current_entropy - 0.02) < 1e-6
    
    def test_model_without_ent_coef_attribute(self):
        """Test callback behavior when model doesn't have ent_coef attribute."""
        # Create mock model without ent_coef
        mock_model_no_attr = Mock(spec=[])  # Empty spec means no attributes
        
        callback = EntropyScheduleCallback(
            initial_entropy=0.01,
            final_entropy=0.005,
            decay_steps=1000,
            verbose=0
        )
        
        callback.model = mock_model_no_attr
        callback.logger = self.mock_logger
        
        # Should not raise error even if model doesn't have ent_coef
        callback._on_training_start()
        callback.num_timesteps = 500
        callback._on_step()
        
        # Callback should still track entropy internally
        assert callback.current_entropy > 0
    
    def test_verbose_output(self):
        """Test that verbose mode produces appropriate output."""
        # This is harder to test directly, but we can at least verify
        # that verbose initialization doesn't crash
        callback = EntropyScheduleCallback(
            initial_entropy=0.01,
            final_entropy=0.005,
            decay_steps=1000,
            verbose=2  # High verbosity
        )
        
        callback.model = self.mock_model
        callback.logger = self.mock_logger
        
        # Should not raise any exceptions
        callback._on_training_start()
        callback.num_timesteps = 500
        callback._on_step()
        callback._on_training_end()


class TestEntropyScheduleIntegration:
    """Integration tests for entropy scheduling."""
    
    def test_realistic_training_scenario(self):
        """Test entropy scheduling in a realistic training scenario."""
        # Typical PPO entropy scheduling
        callback = EntropyScheduleCallback(
            initial_entropy=0.01,
            final_entropy=0.001,
            decay_steps=500000,
            verbose=1
        )
        
        # Mock model
        mock_model = Mock()
        mock_model.ent_coef = 0.01
        mock_logger = Mock()
        
        callback.model = mock_model
        callback.logger = mock_logger  # BaseCallback logger property
        
        # Start training
        callback._on_training_start()
        
        # Simulate training checkpoints
        checkpoints = [0, 50000, 100000, 250000, 500000, 750000, 1000000]
        entropies = []
        
        for step in checkpoints:
            callback.num_timesteps = step
            callback._on_step()
            entropies.append(callback.current_entropy)
        
        # Verify entropy behavior
        assert entropies[0] == 0.01  # Initial
        assert entropies[-1] == 0.001  # Floor reached
        assert entropies[-2] == 0.001  # Still at floor
        
        # Verify monotonic decrease
        for i in range(1, len(entropies)):
            assert entropies[i] <= entropies[i-1], \
                f"Entropy should decrease: {entropies[i-1]} -> {entropies[i]}"
        
        # Verify floor is never violated
        for entropy in entropies:
            assert entropy >= 0.001, f"Entropy {entropy} below floor 0.001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])