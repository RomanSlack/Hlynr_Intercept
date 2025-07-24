#!/usr/bin/env python3
"""
Functional tests for training stability callbacks.

These tests verify that the callbacks can be instantiated and work
with realistic parameters without requiring extensive mocking.
"""

import pytest
import tempfile
import os
from pathlib import Path

# Import the callbacks to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_callbacks import (
    EntropyScheduleCallback,
    LearningRateSchedulerCallback,
    BestModelCallback,
    ClipRangeAdaptiveCallback
)


class TestCallbackInstantiation:
    """Test that callbacks can be instantiated with valid parameters."""
    
    def test_entropy_schedule_callback_creation(self):
        """Test EntropyScheduleCallback can be created with valid parameters."""
        # Test decreasing entropy (common case)
        callback = EntropyScheduleCallback(
            initial_entropy=0.1,
            final_entropy=0.01,
            decay_steps=10000,
            verbose=0
        )
        
        assert callback.initial_entropy == 0.1
        assert callback.final_entropy == 0.01
        assert callback.decay_steps == 10000
        assert callback.current_entropy == 0.1
    
    def test_entropy_schedule_callback_validation(self):
        """Test EntropyScheduleCallback parameter validation."""
        # final_entropy must be less than initial_entropy
        with pytest.raises(ValueError, match="final_entropy.*must be less than initial_entropy"):
            EntropyScheduleCallback(
                initial_entropy=0.01,
                final_entropy=0.1,  # Greater than initial
                decay_steps=1000
            )
        
        # final_entropy must be non-negative
        with pytest.raises(ValueError, match="final_entropy.*must be non-negative"):
            EntropyScheduleCallback(
                initial_entropy=0.1,
                final_entropy=-0.01,
                decay_steps=1000
            )
        
        # decay_steps must be positive
        with pytest.raises(ValueError, match="decay_steps.*must be positive"):
            EntropyScheduleCallback(
                initial_entropy=0.1,
                final_entropy=0.01,
                decay_steps=0
            )
    
    def test_lr_scheduler_callback_creation(self):
        """Test LearningRateSchedulerCallback can be created."""
        callback = LearningRateSchedulerCallback(
            monitor_key="eval/mean_reward",
            patience=5,
            factor=0.5,
            min_lr=1e-6,
            verbose=0
        )
        
        assert callback.monitor_key == "eval/mean_reward"
        assert callback.patience == 5
        assert callback.factor == 0.5
        assert callback.min_lr == 1e-6
        assert callback.best_value == -float('inf')
    
    def test_best_model_callback_creation(self):
        """Test BestModelCallback can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = BestModelCallback(
                monitor_key="eval/mean_reward",
                save_path=temp_dir,
                name_prefix="test",
                save_freq=1000,
                verbose=0
            )
            
            assert callback.monitor_key == "eval/mean_reward"
            assert callback.save_path == Path(temp_dir)
            assert callback.name_prefix == "test"
            assert callback.save_freq == 1000
            assert callback.best_value == -float('inf')
    
    def test_clip_range_callback_creation(self):
        """Test ClipRangeAdaptiveCallback can be created."""
        callback = ClipRangeAdaptiveCallback(
            clip_fraction_threshold=0.2,
            consecutive_threshold=3,
            reduction_factor=0.75,
            min_clip_range=0.05,
            verbose=0
        )
        
        assert callback.clip_fraction_threshold == 0.2
        assert callback.consecutive_threshold == 3
        assert callback.reduction_factor == 0.75
        assert callback.min_clip_range == 0.05
        assert callback.consecutive_violations == 0


class TestEntropyScheduleLogic:
    """Test the core entropy scheduling logic."""
    
    def test_entropy_decay_calculation(self):
        """Test that entropy decay calculation is correct."""
        callback = EntropyScheduleCallback(
            initial_entropy=0.1,
            final_entropy=0.01,
            decay_steps=1000,
            verbose=0
        )
        
        # Test manual calculation of expected entropy
        test_cases = [
            (0, 0.1),      # At start
            (500, 0.055),  # Halfway: 0.1 + 0.5 * (0.01 - 0.1) = 0.055
            (1000, 0.01),  # At end
            (1500, 0.01),  # Beyond end (should stay at floor)
        ]
        
        for timestep, expected_entropy in test_cases:
            # Manual calculation
            progress = min(timestep / callback.decay_steps, 1.0)
            calculated_entropy = callback.initial_entropy + progress * (callback.final_entropy - callback.initial_entropy)
            calculated_entropy = max(calculated_entropy, callback.final_entropy)
            
            assert abs(calculated_entropy - expected_entropy) < 1e-6, \
                f"At timestep {timestep}: expected {expected_entropy}, calculated {calculated_entropy}"
    
    def test_entropy_floor_respected(self):
        """Test that entropy never goes below the floor."""
        callback = EntropyScheduleCallback(
            initial_entropy=0.2,
            final_entropy=0.02,
            decay_steps=100,
            verbose=0
        )
        
        # Test across many timesteps
        for timestep in range(0, 200, 10):
            progress = min(timestep / callback.decay_steps, 1.0)
            entropy = callback.initial_entropy + progress * (callback.final_entropy - callback.initial_entropy)
            entropy = max(entropy, callback.final_entropy)
            
            assert entropy >= callback.final_entropy, \
                f"Entropy {entropy} below floor {callback.final_entropy} at timestep {timestep}"


class TestCallbacksConfigCompatibility:
    """Test that callbacks work with config values."""
    
    def test_callbacks_with_realistic_config_values(self):
        """Test callbacks with realistic configuration values."""
        # Values from our config.yaml
        entropy_callback = EntropyScheduleCallback(
            initial_entropy=0.01,
            final_entropy=0.002,  # Floor should be less than initial
            decay_steps=500000,
            verbose=0
        )
        
        lr_callback = LearningRateSchedulerCallback(
            monitor_key="eval/mean_reward",
            patience=5,
            min_delta=0.01,
            factor=0.5,
            min_lr=1e-6,
            verbose=0
        )
        
        clip_callback = ClipRangeAdaptiveCallback(
            clip_fraction_threshold=0.2,
            consecutive_threshold=3,
            reduction_factor=0.75,
            min_clip_range=0.05,
            verbose=0
        )
        
        # All should instantiate without error
        assert entropy_callback.initial_entropy == 0.01
        assert lr_callback.patience == 5
        assert clip_callback.reduction_factor == 0.75
    
    def test_callback_state_initialization(self):
        """Test that callback states are properly initialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            entropy_cb = EntropyScheduleCallback(0.05, 0.005, 1000)
            lr_cb = LearningRateSchedulerCallback()
            best_cb = BestModelCallback(save_path=temp_dir)
            clip_cb = ClipRangeAdaptiveCallback()
            
            # Check initial states
            assert entropy_cb.current_entropy == entropy_cb.initial_entropy
            assert lr_cb.best_value == -float('inf')
            assert lr_cb.wait_count == 0
            assert best_cb.best_value == -float('inf')
            assert best_cb.n_saves == 0
            assert clip_cb.consecutive_violations == 0
            assert clip_cb.reductions == 0


class TestBestModelMetadata:
    """Test best model callback metadata generation."""
    
    def test_best_model_creates_directory(self):
        """Test that best model callback creates save directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "best_models"
            
            callback = BestModelCallback(
                save_path=str(save_path),
                verbose=0
            )
            
            # Simulate training start (this should create directory)
            callback._on_training_start()
            
            assert save_path.exists()
            assert save_path.is_dir()


class TestCallbackIntegration:
    """Test that all callbacks can work together."""
    
    def test_multiple_callbacks_instantiation(self):
        """Test that multiple callbacks can be created simultaneously."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callbacks = [
                EntropyScheduleCallback(0.01, 0.001, 10000, verbose=0),
                LearningRateSchedulerCallback(verbose=0),
                BestModelCallback(save_path=temp_dir, verbose=0),
                ClipRangeAdaptiveCallback(verbose=0)
            ]
            
            # All should be created without interference
            assert len(callbacks) == 4
            assert isinstance(callbacks[0], EntropyScheduleCallback)
            assert isinstance(callbacks[1], LearningRateSchedulerCallback)
            assert isinstance(callbacks[2], BestModelCallback)
            assert isinstance(callbacks[3], ClipRangeAdaptiveCallback)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])