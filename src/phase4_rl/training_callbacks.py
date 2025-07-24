#!/usr/bin/env python3
"""
Advanced training stability callbacks for Phase 4 RL.

This module provides custom callbacks for PPO training to improve stability
and performance through adaptive hyperparameter scheduling and monitoring.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger


class EntropyScheduleCallback(BaseCallback):
    """
    Callback for scheduling entropy coefficient with linear decay to a floor.
    
    Implements linear decay from initial entropy coefficient to a minimum floor
    (typically 0.01-0.02) to maintain exploration while preventing premature
    convergence.
    """
    
    def __init__(self, 
                 initial_entropy: float = 0.01,
                 final_entropy: float = 0.02,
                 decay_steps: int = 500000,
                 verbose: int = 0):
        """
        Initialize entropy schedule callback.
        
        Args:
            initial_entropy: Starting entropy coefficient
            final_entropy: Minimum entropy coefficient (floor)
            decay_steps: Number of timesteps over which to decay
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        
        self.initial_entropy = initial_entropy
        self.final_entropy = final_entropy
        self.decay_steps = decay_steps
        self.current_entropy = initial_entropy
        
        # Validation
        if final_entropy >= initial_entropy:
            raise ValueError(f"final_entropy ({final_entropy}) must be less than initial_entropy ({initial_entropy})")
        if final_entropy < 0:
            raise ValueError(f"final_entropy ({final_entropy}) must be non-negative")
        if decay_steps <= 0:
            raise ValueError(f"decay_steps ({decay_steps}) must be positive")
        
        self._callback_logger = logging.getLogger(self.__class__.__name__)
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.current_entropy = self.initial_entropy
        
        # Set initial entropy coefficient
        if hasattr(self.model, 'ent_coef'):
            self.model.ent_coef = self.current_entropy
            
        self._callback_logger.info(f"Entropy schedule initialized:")
        self._callback_logger.info(f"  Initial entropy: {self.initial_entropy}")
        self._callback_logger.info(f"  Final entropy floor: {self.final_entropy}")
        self._callback_logger.info(f"  Decay steps: {self.decay_steps}")
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Calculate current entropy based on linear decay
        progress = min(self.num_timesteps / self.decay_steps, 1.0)
        
        # Linear interpolation from initial to final
        self.current_entropy = self.initial_entropy + progress * (self.final_entropy - self.initial_entropy)
        
        # Ensure we never go below the floor
        self.current_entropy = max(self.current_entropy, self.final_entropy)
        
        # Update model's entropy coefficient
        if hasattr(self.model, 'ent_coef'):
            self.model.ent_coef = self.current_entropy
        
        # Log entropy coefficient periodically
        if self.num_timesteps % 10000 == 0 or self.verbose >= 2:
            self._callback_logger.debug(f"Step {self.num_timesteps}: entropy_coef = {self.current_entropy:.6f}")
            
        # Log to tensorboard/CSV
        self._callback_logger.record("train/entropy_coef", self.current_entropy)
        
        return True
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        self._callback_logger.info(f"Entropy schedule completed. Final entropy: {self.current_entropy:.6f}")


class LearningRateSchedulerCallback(BaseCallback):
    """
    Callback for adaptive learning rate scheduling based on evaluation performance.
    
    Monitors evaluation reward and reduces learning rate when performance plateaus,
    with optional early stopping based on patience.
    """
    
    def __init__(self,
                 monitor_key: str = "eval/mean_reward",
                 patience: int = 5,
                 min_delta: float = 0.01,
                 factor: float = 0.5,
                 min_lr: float = 1e-6,
                 early_stopping: bool = False,
                 early_stopping_patience: int = 10,
                 verbose: int = 0):
        """
        Initialize learning rate scheduler callback.
        
        Args:
            monitor_key: Metric to monitor for plateau detection
            patience: Number of evaluations without improvement before LR reduction
            min_delta: Minimum change to qualify as improvement
            factor: Factor by which to reduce LR (new_lr = old_lr * factor)
            min_lr: Minimum learning rate threshold
            early_stopping: Whether to enable early stopping
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.monitor_key = monitor_key
        self.patience = patience
        self.min_delta = min_delta
        self.factor = factor
        self.min_lr = min_lr
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        
        # State tracking
        self.best_value = -np.inf
        self.wait_count = 0
        self.early_stop_wait = 0
        self.lr_reductions = 0
        self.initial_lr = None
        
        # History
        self.lr_history = []
        self.performance_history = []
        
        self._callback_logger = logging.getLogger(self.__class__.__name__)
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        # Get initial learning rate
        if hasattr(self.model.policy.optimizer, 'param_groups'):
            self.initial_lr = self.model.policy.optimizer.param_groups[0]['lr']
        else:
            self.initial_lr = self.model.learning_rate
            
        self._callback_logger.info(f"Learning rate scheduler initialized:")
        self._callback_logger.info(f"  Monitor: {self.monitor_key}")
        self._callback_logger.info(f"  Initial LR: {self.initial_lr}")
        self._callback_logger.info(f"  Patience: {self.patience}")
        self._callback_logger.info(f"  Early stopping: {self.early_stopping}")
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Check if we have the monitored metric
        if self.monitor_key in self.model.logger.name_to_value:
            current_value = self.model.logger.name_to_value[self.monitor_key]
            self.performance_history.append(current_value)
            
            # Check for improvement
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.wait_count = 0
                self.early_stop_wait = 0
                
                if self.verbose >= 1:
                    self._callback_logger.info(f"New best {self.monitor_key}: {current_value:.4f}")
            else:
                self.wait_count += 1
                self.early_stop_wait += 1
                
                # Check if we should reduce learning rate
                if self.wait_count >= self.patience:
                    self._reduce_learning_rate()
                    self.wait_count = 0
                
                # Check for early stopping
                if self.early_stopping and self.early_stop_wait >= self.early_stopping_patience:
                    self._callback_logger.info(f"Early stopping triggered after {self.early_stop_wait} evaluations without improvement")
                    return False
        
        # Log current learning rate
        current_lr = self._get_current_lr()
        self._callback_logger.record("train/learning_rate", current_lr)
        
        return True
    
    def _reduce_learning_rate(self):
        """Reduce learning rate by the specified factor."""
        current_lr = self._get_current_lr()
        new_lr = max(current_lr * self.factor, self.min_lr)
        
        if new_lr < current_lr:
            self._set_learning_rate(new_lr)
            self.lr_reductions += 1
            self.lr_history.append((self.num_timesteps, new_lr))
            
            self._callback_logger.info(f"Reducing learning rate from {current_lr:.2e} to {new_lr:.2e} "
                           f"(reduction #{self.lr_reductions})")
        else:
            self._callback_logger.info(f"Learning rate already at minimum ({self.min_lr:.2e})")
    
    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        if hasattr(self.model.policy.optimizer, 'param_groups'):
            return self.model.policy.optimizer.param_groups[0]['lr']
        else:
            return self.model.learning_rate
    
    def _set_learning_rate(self, new_lr: float):
        """Set new learning rate."""
        if hasattr(self.model.policy.optimizer, 'param_groups'):
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            self.model.learning_rate = new_lr
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        final_lr = self._get_current_lr()
        self._callback_logger.info(f"Learning rate scheduling completed:")
        self._callback_logger.info(f"  Initial LR: {self.initial_lr:.2e}")
        self._callback_logger.info(f"  Final LR: {final_lr:.2e}")
        self._callback_logger.info(f"  Total reductions: {self.lr_reductions}")
        self._callback_logger.info(f"  Best {self.monitor_key}: {self.best_value:.4f}")


class BestModelCallback(BaseCallback):
    """
    Enhanced callback for saving the best model based on evaluation performance.
    
    Extends the functionality of EvalCallback by providing more sophisticated
    best model tracking and metadata saving.
    """
    
    def __init__(self,
                 monitor_key: str = "eval/mean_reward",
                 save_path: str = "best_model",
                 name_prefix: str = "best",
                 save_freq: int = 10000,
                 save_replay_buffer: bool = False,
                 save_vecnormalize: bool = True,
                 verbose: int = 0):
        """
        Initialize best model callback.
        
        Args:
            monitor_key: Metric to monitor for best model selection
            save_path: Directory to save best model
            name_prefix: Prefix for saved files
            save_freq: Frequency of best model checks (in timesteps)
            save_replay_buffer: Whether to save replay buffer
            save_vecnormalize: Whether to save VecNormalize parameters
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.monitor_key = monitor_key
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_freq = save_freq
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        
        # State tracking
        self.best_value = -np.inf
        self.best_timestep = 0
        self.n_saves = 0
        self.last_check = 0
        
        # Metadata
        self.save_history = []
        
        self._callback_logger = logging.getLogger(self.__class__.__name__)
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self._callback_logger.info(f"Best model callback initialized:")
        self._callback_logger.info(f"  Monitor: {self.monitor_key}")
        self._callback_logger.info(f"  Save path: {self.save_path}")
        self._callback_logger.info(f"  Save frequency: {self.save_freq}")
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Check if it's time to evaluate
        if self.num_timesteps - self.last_check >= self.save_freq:
            self.last_check = self.num_timesteps
            
            # Check if we have the monitored metric
            if self.monitor_key in self.model.logger.name_to_value:
                current_value = self.model.logger.name_to_value[self.monitor_key]
                
                # Check if this is a new best
                if current_value > self.best_value:
                    self.best_value = current_value
                    self.best_timestep = self.num_timesteps
                    self._save_best_model()
        
        return True
    
    def _save_best_model(self):
        """Save the current model as the best model."""
        self.n_saves += 1
        timestamp = time.time()
        
        # Create timestamped filename
        model_path = self.save_path / f"{self.name_prefix}_model"
        
        # Save the model
        self.model.save(model_path)
        
        # Save VecNormalize if requested and available
        if self.save_vecnormalize and hasattr(self.training_env, 'save'):
            normalize_path = self.save_path / f"{self.name_prefix}_vecnormalize.pkl"
            self.training_env.save(normalize_path)
        
        # Save metadata
        metadata = {
            'timestep': self.best_timestep,
            'best_value': float(self.best_value),
            'metric': self.monitor_key,
            'timestamp': timestamp,
            'save_count': self.n_saves,
            'model_path': str(model_path) + '.zip'
        }
        
        metadata_path = self.save_path / f"{self.name_prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update save history
        self.save_history.append(metadata)
        
        # Log the save
        self._callback_logger.info(f"Saved new best model (save #{self.n_saves}):")
        self._callback_logger.info(f"  {self.monitor_key}: {self.best_value:.4f}")
        self._callback_logger.info(f"  Timestep: {self.best_timestep}")
        self._callback_logger.info(f"  Path: {model_path}.zip")
        
        # Log to tensorboard
        self._callback_logger.record("train/best_model_saves", self.n_saves)
        self._callback_logger.record(f"train/best_{self.monitor_key.replace('/', '_')}", self.best_value)
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.n_saves > 0:
            self._callback_logger.info(f"Best model callback completed:")
            self._callback_logger.info(f"  Total saves: {self.n_saves}")
            self._callback_logger.info(f"  Best {self.monitor_key}: {self.best_value:.4f}")
            self._callback_logger.info(f"  Best timestep: {self.best_timestep}")
        else:
            self._callback_logger.warning("No best models were saved during training")


class ClipRangeAdaptiveCallback(BaseCallback):
    """
    Callback for adaptive clip range adjustment based on policy update statistics.
    
    Monitors clip_fraction and automatically reduces clip_range when the fraction
    consistently exceeds a threshold, indicating that the policy updates may be
    too large and potentially destabilizing.
    """
    
    def __init__(self,
                 clip_fraction_threshold: float = 0.2,
                 consecutive_threshold: int = 3,
                 reduction_factor: float = 0.75,
                 min_clip_range: float = 0.05,
                 check_freq: int = 10000,
                 verbose: int = 0):
        """
        Initialize clip range adaptive callback.
        
        Args:
            clip_fraction_threshold: Clip fraction threshold for triggering reduction
            consecutive_threshold: Number of consecutive violations before reduction
            reduction_factor: Factor by which to reduce clip range
            min_clip_range: Minimum allowed clip range
            check_freq: Frequency of clip fraction checks (in timesteps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.clip_fraction_threshold = clip_fraction_threshold
        self.consecutive_threshold = consecutive_threshold
        self.reduction_factor = reduction_factor
        self.min_clip_range = min_clip_range
        self.check_freq = check_freq
        
        # State tracking
        self.consecutive_violations = 0
        self.last_check = 0
        self.reductions = 0
        self.initial_clip_range = None
        
        # History
        self.clip_fraction_history = []
        self.clip_range_history = []
        
        self._callback_logger = logging.getLogger(self.__class__.__name__)
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        # Get initial clip range
        if hasattr(self.model, 'clip_range'):
            if callable(self.model.clip_range):
                # If clip_range is a function, get its initial value
                self.initial_clip_range = self.model.clip_range(1.0)
            else:
                self.initial_clip_range = self.model.clip_range
        else:
            self.initial_clip_range = 0.2  # Default PPO clip range
            
        self._callback_logger.info(f"Clip range adaptive callback initialized:")
        self._callback_logger.info(f"  Initial clip range: {self.initial_clip_range}")
        self._callback_logger.info(f"  Threshold: {self.clip_fraction_threshold}")
        self._callback_logger.info(f"  Consecutive threshold: {self.consecutive_threshold}")
        self._callback_logger.info(f"  Reduction factor: {self.reduction_factor}")
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Check if it's time to evaluate
        if self.num_timesteps - self.last_check >= self.check_freq:
            self.last_check = self.num_timesteps
            
            # Check if we have clip fraction data
            if "train/clip_fraction" in self.model.logger.name_to_value:
                clip_fraction = self.model.logger.name_to_value["train/clip_fraction"]
                self.clip_fraction_history.append(clip_fraction)
                
                # Check if clip fraction exceeds threshold
                if clip_fraction > self.clip_fraction_threshold:
                    self.consecutive_violations += 1
                    
                    if self.verbose >= 2:
                        self._callback_logger.debug(f"Clip fraction violation #{self.consecutive_violations}: {clip_fraction:.3f}")
                    
                    # Check if we should reduce clip range
                    if self.consecutive_violations >= self.consecutive_threshold:
                        self._reduce_clip_range()
                        self.consecutive_violations = 0
                else:
                    self.consecutive_violations = 0
        
        # Log current clip range
        current_clip_range = self._get_current_clip_range()
        self._callback_logger.record("train/clip_range", current_clip_range)
        
        return True
    
    def _reduce_clip_range(self):
        """Reduce clip range by the specified factor."""
        current_clip_range = self._get_current_clip_range()
        new_clip_range = max(current_clip_range * self.reduction_factor, self.min_clip_range)
        
        if new_clip_range < current_clip_range:
            self._set_clip_range(new_clip_range)
            self.reductions += 1
            self.clip_range_history.append((self.num_timesteps, new_clip_range))
            
            self._callback_logger.info(f"Reducing clip range from {current_clip_range:.3f} to {new_clip_range:.3f} "
                           f"(reduction #{self.reductions})")
        else:
            self._callback_logger.info(f"Clip range already at minimum ({self.min_clip_range:.3f})")
    
    def _get_current_clip_range(self) -> float:
        """Get current clip range."""
        if hasattr(self.model, 'clip_range'):
            if callable(self.model.clip_range):
                # If clip_range is a function, evaluate it at progress=1.0
                return self.model.clip_range(1.0)
            else:
                return self.model.clip_range
        else:
            return 0.2  # Default
    
    def _set_clip_range(self, new_clip_range: float):
        """Set new clip range."""
        if hasattr(self.model, 'clip_range'):
            # Replace the clip_range with a constant value
            self.model.clip_range = new_clip_range
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        final_clip_range = self._get_current_clip_range()
        self._callback_logger.info(f"Clip range adaptation completed:")
        self._callback_logger.info(f"  Initial clip range: {self.initial_clip_range:.3f}")
        self._callback_logger.info(f"  Final clip range: {final_clip_range:.3f}")
        self._callback_logger.info(f"  Total reductions: {self.reductions}")
        
        if self.clip_fraction_history:
            avg_clip_fraction = np.mean(self.clip_fraction_history)
            self._callback_logger.info(f"  Average clip fraction: {avg_clip_fraction:.3f}")