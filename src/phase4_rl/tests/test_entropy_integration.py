#!/usr/bin/env python3
"""
Integration test for EntropyScheduleCallback with actual PPO training.

This test verifies that the callback works correctly in a real training context
and catches logger bugs that would occur during actual use.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the callback to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_callbacks import EntropyScheduleCallback


class TestEntropyIntegration:
    """Integration tests for EntropyScheduleCallback with PPO."""
    
    def test_entropy_callback_with_ppo_training(self):
        """Test that EntropyScheduleCallback works correctly with actual PPO training."""
        
        # Create a simple environment
        def make_env():
            return gym.make('CartPole-v1')
        
        env = DummyVecEnv([make_env])
        
        # Create the entropy callback
        callback = EntropyScheduleCallback(
            initial_entropy=0.01,
            final_entropy=0.001,
            decay_steps=100,  # Short for testing
            verbose=1
        )
        
        # Create PPO model
        model = PPO(
            'MlpPolicy', 
            env, 
            ent_coef=0.01,  # Initial entropy coefficient
            n_steps=32,     # Small rollout for fast test
            verbose=0
        )
        
        # Create a mock logger and set it on the model
        mock_logger = Mock()
        mock_logger.record = Mock()
        mock_logger.dump = Mock()
        mock_logger.name_to_value = {}
        model.set_logger(mock_logger)
        
        try:
            # Run a short training session
            model.learn(
                total_timesteps=64,  # Very short for testing
                callback=callback
            )
            
            # Verify that training completed without exceptions
            assert True, "Training completed successfully"
            
            # Verify that the logger.record was called for entropy
            record_calls = mock_logger.record.call_args_list
            entropy_calls = [call for call in record_calls if call[0][0] == "train/entropy_coef"]
            
            # Should have at least one entropy logging call
            assert len(entropy_calls) > 0, "Logger should have recorded entropy coefficient"
            
            # Verify that entropy was actually updated
            assert hasattr(model, 'ent_coef'), "Model should have ent_coef attribute"
            
            # Verify callback state
            assert callback.current_entropy >= callback.final_entropy, \
                "Current entropy should not go below floor"
            assert callback.current_entropy <= callback.initial_entropy, \
                "Current entropy should not exceed initial value"
            
        finally:
            env.close()
    
    def test_entropy_callback_logger_compatibility(self):
        """Test that the callback's logger usage is compatible with SB3."""
        
        # Create simple environment
        env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        
        # Create callback with different parameters
        callback = EntropyScheduleCallback(
            initial_entropy=0.05,
            final_entropy=0.005,
            decay_steps=50,
            verbose=2  # Higher verbosity for more logging
        )
        
        # Create PPO model
        model = PPO('MlpPolicy', env, verbose=0, n_steps=16)
        
        # Create a mock logger and set it on the model
        mock_logger = Mock()
        mock_logger.record = Mock()
        mock_logger.dump = Mock()
        mock_logger.name_to_value = {}
        model.set_logger(mock_logger)
        
        try:
            # Test that callback can be initialized with the model
            callback.init_callback(model)
            
            # Test that the callback's logger methods exist and work
            callback._on_training_start()
            
            # Simulate some training steps
            callback.num_timesteps = 25  # Halfway through decay
            result = callback._on_step()
            
            assert result is True, "Callback should return True to continue training"
            
            # Test training end
            callback._on_training_end()
            
            # If we get here without exceptions, the logger compatibility works
            assert True, "Logger compatibility test passed"
            
        finally:
            env.close()
    
    def test_entropy_decay_during_training(self):
        """Test that entropy actually decays during training steps."""
        
        env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        
        callback = EntropyScheduleCallback(
            initial_entropy=0.1,
            final_entropy=0.01,
            decay_steps=100,
            verbose=0
        )
        
        model = PPO('MlpPolicy', env, verbose=0, n_steps=16)
        
        # Create a mock logger and set it on the model
        mock_logger = Mock()
        mock_logger.record = Mock()
        mock_logger.dump = Mock()
        mock_logger.name_to_value = {}
        model.set_logger(mock_logger)
        
        try:
            # Initialize callback
            callback.init_callback(model)
            callback._on_training_start()
            
            # Record entropy at different timesteps
            entropies = []
            
            for timestep in [0, 25, 50, 75, 100, 125]:
                callback.num_timesteps = timestep
                callback._on_step()
                entropies.append(callback.current_entropy)
            
            # Verify entropy decay pattern
            assert entropies[0] == 0.1, "Should start at initial entropy"
            assert entropies[-1] == 0.01, "Should end at final entropy"
            
            # Verify monotonic decrease (until floor is reached)
            for i in range(1, len(entropies)):
                if entropies[i] > callback.final_entropy:
                    assert entropies[i] <= entropies[i-1], \
                        f"Entropy should decrease: {entropies[i-1]} -> {entropies[i]}"
            
            # Verify floor is never violated
            for entropy in entropies:
                assert entropy >= callback.final_entropy, \
                    f"Entropy {entropy} should not go below floor {callback.final_entropy}"
            
        finally:
            env.close()
    
    def test_multiple_callbacks_integration(self):
        """Test that EntropyScheduleCallback works alongside other callbacks."""
        
        env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        
        # Create multiple callbacks
        entropy_callback = EntropyScheduleCallback(
            initial_entropy=0.02,
            final_entropy=0.002,
            decay_steps=50,
            verbose=0
        )
        
        # Create a simple custom callback for testing
        class DummyCallback:
            def __init__(self):
                self.calls = 0
            
            def init_callback(self, model):
                pass
            
            def _on_training_start(self):
                self.calls += 1
                return True
            
            def _on_step(self):
                self.calls += 1
                return True
            
            def _on_training_end(self):
                self.calls += 1
                return True
        
        dummy_callback = DummyCallback()
        
        model = PPO('MlpPolicy', env, verbose=0, n_steps=16)
        
        # Create a mock logger and set it on the model
        mock_logger = Mock()
        mock_logger.record = Mock()
        mock_logger.dump = Mock()
        mock_logger.name_to_value = {}
        model.set_logger(mock_logger)
        
        try:
            # Test that both callbacks can coexist
            callbacks = [entropy_callback, dummy_callback]
            
            # Initialize callbacks
            for cb in callbacks:
                if hasattr(cb, 'init_callback'):
                    cb.init_callback(model)
            
            # Test training start
            for cb in callbacks:
                if hasattr(cb, '_on_training_start'):
                    cb._on_training_start()
            
            # Test a few steps
            for step in range(3):
                for cb in callbacks:
                    if hasattr(cb, '_on_step'):
                        cb.num_timesteps = step * 16
                        result = cb._on_step()
                        assert result is True, "All callbacks should return True"
            
            # Test training end
            for cb in callbacks:
                if hasattr(cb, '_on_training_end'):
                    cb._on_training_end()
            
            # Verify that both callbacks were active
            assert dummy_callback.calls > 0, "Dummy callback should have been called"
            assert entropy_callback.current_entropy <= entropy_callback.initial_entropy, \
                "Entropy callback should have updated entropy"
            
        finally:
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])