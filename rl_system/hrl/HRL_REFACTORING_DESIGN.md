# Hierarchical RL Integration: Refactoring Design

**Version:** 1.0
**Date:** 2025-11-05
**Status:** Design Review

## Executive Summary

This document outlines a comprehensive refactoring plan to integrate Hierarchical Reinforcement Learning (HRL) into the existing `rl_system/` codebase while maintaining full backward compatibility with the current flat PPO implementation. The design prioritizes modularity, testability, and minimal disruption to existing workflows.

**Core Principle:** Composition over modification - HRL functionality will be added through new modules and wrappers, not by modifying existing core files.

---

## 1. Proposed Directory Structure

```
rl_system/
├── # ============================================
├── # EXISTING FILES (UNCHANGED)
├── # ============================================
├── core.py                      # 26D radar obs, Kalman, safety (NO CHANGES)
├── environment.py               # InterceptEnvironment Gymnasium env (NO CHANGES)
├── train.py                     # PPO training loop (MINIMAL CHANGES - see migration)
├── inference.py                 # FastAPI server + offline eval (MINIMAL CHANGES)
├── physics_models.py            # 6DOF dynamics (NO CHANGES)
├── physics_randomizer.py        # Domain randomization (NO CHANGES)
├── logger.py                    # Unified logging (NO CHANGES)
├── monitor_training.py          # Training monitor (NO CHANGES)
├── diagnose_radar.py            # Radar diagnostics (NO CHANGES)
├── visualize.py                 # Episode visualization (NO CHANGES)
├── validate_setup.py            # Setup validation (NO CHANGES)
├── config.yaml                  # Main config (KEPT for backward compat)
├── config_*.yaml                # Alternative configs (KEPT)
├── requirements.txt             # Dependencies (UPDATED with HRL libs)
├── README.md                    # Documentation (UPDATED)
│
├── # ============================================
├── # EXISTING DIRECTORIES (MINIMAL CHANGES)
├── # ============================================
├── scenarios/                   # RENAMED → configs/scenarios/ (Phase 1)
│   ├── easy.yaml
│   ├── medium.yaml
│   └── hard.yaml
│
├── helpers/                     # Utility scripts (NO CHANGES)
│   ├── analyze_metrics.py
│   ├── check_missile_trajectory.py
│   ├── visualize_episode.py
│   └── visualize_physics.py
│
├── checkpoints/                 # REORGANIZED (see structure below)
│   ├── flat_ppo/               # Move existing checkpoints here
│   │   ├── model_*.zip
│   │   ├── best/
│   │   └── vec_normalize.pkl
│   └── hrl/                    # New HRL checkpoints
│       ├── selector/
│       │   ├── model.zip
│       │   └── vec_normalize.pkl
│       └── specialists/
│           ├── search/
│           ├── track/
│           └── terminal/
│
├── logs/                       # Logs (NO CHANGES, auto-managed)
│   └── run_YYYYMMDD_HHMMSS/
│
├── # ============================================
├── # NEW DIRECTORIES (HRL IMPLEMENTATION)
├── # ============================================
├── hrl/                        # All HRL-specific code
│   ├── __init__.py            # Public API exports
│   │
│   ├── # Core HRL Components
│   ├── manager.py             # HierarchicalManager: coordinates policies
│   ├── selector_policy.py     # High-level discrete option selection
│   ├── specialist_policies.py # Low-level continuous specialists
│   ├── option_manager.py      # Option switching logic, forced transitions
│   │
│   ├── # State Abstraction & Rewards
│   ├── observation_abstraction.py  # 26D → abstract state mapping
│   ├── reward_decomposition.py     # Strategic + tactical rewards
│   │
│   ├── # Gymnasium Integration
│   ├── wrappers.py            # HRL-compatible env wrappers
│   ├── hierarchical_env.py    # Main HRL environment wrapper
│   │
│   ├── # Training Infrastructure
│   ├── callbacks.py           # HRL-specific training callbacks
│   ├── metrics.py             # Option transition tracking, usage stats
│   ├── curriculum.py          # Curriculum learning for pre-training
│   │
│   └── # Utilities
│       ├── option_definitions.py   # Enum for options, shared constants
│       └── visualization.py        # HRL-specific visualizations
│
├── configs/                   # Configuration management
│   ├── # Backward compatibility
│   ├── config.yaml           # Symlink to ../config.yaml
│   │
│   ├── # Scenario presets (moved from scenarios/)
│   ├── scenarios/
│   │   ├── easy.yaml
│   │   ├── medium.yaml
│   │   └── hard.yaml
│   │
│   ├── # HRL-specific configs
│   ├── hrl/
│   │   ├── hrl_base.yaml              # Base HRL configuration
│   │   ├── selector_config.yaml       # Selector policy settings
│   │   ├── search_specialist.yaml     # Search specialist config
│   │   ├── track_specialist.yaml      # Track specialist config
│   │   ├── terminal_specialist.yaml   # Terminal specialist config
│   │   ├── hrl_curriculum.yaml        # Pre-training curriculum
│   │   └── hrl_full_training.yaml     # End-to-end HRL training
│   │
│   └── # Experimental configs
│       └── experimental/
│           ├── goal_conditioned.yaml
│           └── multi_agent.yaml
│
├── scripts/                  # Training & deployment scripts
│   ├── # Backward compatibility wrapper
│   ├── train_flat_ppo.py     # Refactored flat PPO (extracted from train.py)
│   │
│   ├── # HRL training pipeline
│   ├── train_hrl_pretrain.py      # Pre-train specialists separately
│   ├── train_hrl_selector.py      # Train selector with frozen specialists
│   ├── train_hrl_joint.py         # Joint fine-tuning (optional)
│   ├── train_hrl_full.py          # End-to-end training orchestrator
│   │
│   ├── # Evaluation & deployment
│   ├── evaluate_hrl.py            # Offline HRL evaluation
│   ├── compare_policies.py        # Flat vs HRL comparison
│   │
│   └── # Utilities
│       ├── migrate_checkpoints.py # Migrate old checkpoints
│       └── visualize_options.py   # Visualize option usage
│
├── tests/                    # Unit & integration tests
│   ├── __init__.py
│   │
│   ├── # Core HRL tests
│   ├── test_hrl_manager.py
│   ├── test_selector_policy.py
│   ├── test_specialist_policies.py
│   ├── test_option_manager.py
│   │
│   ├── # Abstraction & Reward tests
│   ├── test_observation_abstraction.py
│   ├── test_reward_decomposition.py
│   │
│   ├── # Environment tests
│   ├── test_hierarchical_env.py
│   ├── test_wrappers.py
│   │
│   ├── # Integration tests
│   ├── test_end_to_end.py
│   ├── test_backward_compatibility.py
│   │
│   └── # Fixtures & utilities
│       ├── conftest.py        # Pytest fixtures
│       └── test_utils.py      # Test helper functions
│
└── docs/                     # Documentation (NEW)
    ├── hrl/
    │   ├── architecture.md
    │   ├── training_guide.md
    │   ├── api_reference.md
    │   └── migration_guide.md
    └── diagrams/
        ├── hrl_data_flow.png
        └── option_transitions.png
```

---

## 2. Module Interfaces and Responsibilities

### 2.1 Core HRL Components

#### `hrl/option_definitions.py`
```python
"""
Shared constants and enumerations for HRL options.
"""
from enum import IntEnum
from typing import Dict, Any


class Option(IntEnum):
    """High-level options for missile defense."""
    SEARCH = 0      # Wide-area scanning for target acquisition
    TRACK = 1       # Maintain lock and close distance
    TERMINAL = 2    # Final intercept guidance


# Option metadata
OPTION_METADATA = {
    Option.SEARCH: {
        'name': 'Search',
        'description': 'Wide-area scanning, large angular changes',
        'expected_duration': 200,  # steps
        'forced_exit_conditions': ['radar_lock_acquired'],
        'color': '#FF6B6B',  # Red for visualization
    },
    Option.TRACK: {
        'name': 'Track',
        'description': 'Maintain radar lock, approach target',
        'expected_duration': 500,
        'forced_exit_conditions': ['radar_lock_lost', 'close_range_threshold'],
        'color': '#4ECDC4',  # Teal
    },
    Option.TERMINAL: {
        'name': 'Terminal',
        'description': 'Final guidance, high precision',
        'expected_duration': 100,
        'forced_exit_conditions': ['intercept_complete', 'miss_imminent'],
        'color': '#95E1D3',  # Light green
    },
}


# Physical thresholds for forced transitions
FORCED_TRANSITION_THRESHOLDS = {
    'radar_lock_quality_min': 0.3,      # Below this, lose track
    'radar_lock_quality_search': 0.7,   # Above this, exit search
    'close_range_threshold': 100.0,     # meters - trigger terminal guidance
    'terminal_fuel_min': 0.1,           # Min fuel to enter terminal
}
```

---

#### `hrl/manager.py`
```python
"""
Hierarchical Manager coordinates selector and specialists during training/inference.
"""
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .option_definitions import Option, FORCED_TRANSITION_THRESHOLDS
from .selector_policy import SelectorPolicy
from .specialist_policies import SpecialistPolicy
from .option_manager import OptionManager
from .observation_abstraction import abstract_observation


@dataclass
class HRLState:
    """Current state of the hierarchical controller."""
    current_option: Option
    steps_in_option: int
    lstm_states: Dict[Option, Any]  # Per-specialist LSTM hidden states
    last_decision_step: int
    total_steps: int


class HierarchicalManager:
    """
    Coordinates selector and specialists during inference/training.

    Design Pattern: Mediator - coordinates between selector and specialists.
    Thread Safety: NOT thread-safe, use one instance per environment.
    """

    def __init__(
        self,
        selector_policy: SelectorPolicy,
        specialists: Dict[Option, SpecialistPolicy],
        decision_interval: int = 100,  # High-level decides every N steps (1Hz @ 100Hz sim)
        enable_forced_transitions: bool = True,
        default_option: Option = Option.SEARCH,
    ):
        """
        Args:
            selector_policy: High-level policy for option selection
            specialists: Dict mapping Option -> SpecialistPolicy
            decision_interval: Steps between high-level decisions
            enable_forced_transitions: Allow environment state to force option changes
            default_option: Initial option on reset
        """
        self.selector = selector_policy
        self.specialists = specialists
        self.decision_interval = decision_interval
        self.enable_forced_transitions = enable_forced_transitions
        self.default_option = default_option

        # Option switching logic
        self.option_manager = OptionManager(
            thresholds=FORCED_TRANSITION_THRESHOLDS,
            enable_forced=enable_forced_transitions,
        )

        # State tracking
        self.state = None
        self.reset()

    def reset(self):
        """Reset to initial state (called at episode start)."""
        self.state = HRLState(
            current_option=self.default_option,
            steps_in_option=0,
            lstm_states={opt: None for opt in Option},
            last_decision_step=0,
            total_steps=0,
        )

        # Reset all specialist LSTM states
        for specialist in self.specialists.values():
            if hasattr(specialist, 'reset_lstm'):
                specialist.reset_lstm()

    def select_action(
        self,
        full_obs: np.ndarray,  # 26D or 104D frame-stacked observation
        env_state: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action using hierarchical policy.

        Args:
            full_obs: Full low-level observation (26D base or 104D stacked)
            env_state: Environment state for forced transition checks
            deterministic: Use deterministic action selection

        Returns:
            action: 6D continuous action from active specialist
            info: Dictionary with debugging information
        """
        # 1. Check for forced option transitions (e.g., radar lock lost)
        forced_option = None
        if self.enable_forced_transitions and env_state is not None:
            forced_option = self.option_manager.get_forced_transition(
                current_option=self.state.current_option,
                env_state=env_state,
            )

        # 2. Check if it's time for high-level decision
        option_switched = False
        if forced_option is not None:
            new_option = forced_option
            option_switched = True
            switch_reason = "forced"
        elif self.should_switch_option():
            # High-level decision: select new option
            abstract_state = abstract_observation(full_obs)
            new_option_idx = self.selector.predict(
                abstract_state,
                deterministic=deterministic
            )
            new_option = Option(new_option_idx)
            option_switched = (new_option != self.state.current_option)
            switch_reason = "selector"
        else:
            new_option = self.state.current_option
            switch_reason = "continue"

        # 3. Handle option switching
        if option_switched:
            self._switch_option(new_option)

        # 4. Execute low-level action from active specialist
        active_specialist = self.specialists[self.state.current_option]

        if hasattr(active_specialist, 'predict_with_lstm'):
            # Specialist uses LSTM
            action, new_lstm_state = active_specialist.predict_with_lstm(
                full_obs,
                self.state.lstm_states[self.state.current_option],
                deterministic=deterministic,
            )
            self.state.lstm_states[self.state.current_option] = new_lstm_state
        else:
            # Specialist without LSTM
            action = active_specialist.predict(full_obs, deterministic=deterministic)

        # 5. Update state
        self.state.steps_in_option += 1
        self.state.total_steps += 1

        # 6. Build info dict for logging/debugging
        info = {
            'option': self.state.current_option.name,
            'option_index': int(self.state.current_option),
            'steps_in_option': self.state.steps_in_option,
            'option_switched': option_switched,
            'switch_reason': switch_reason,
            'forced_transition': forced_option is not None,
        }

        return action, info

    def should_switch_option(self) -> bool:
        """Check if it's time for high-level to make new decision."""
        return (self.state.total_steps - self.state.last_decision_step) >= self.decision_interval

    def _switch_option(self, new_option: Option):
        """Internal: Handle option switching."""
        old_option = self.state.current_option

        # Reset specialist LSTM state on switch
        if hasattr(self.specialists[old_option], 'reset_lstm'):
            self.specialists[old_option].reset_lstm()

        # Update state
        self.state.current_option = new_option
        self.state.steps_in_option = 0
        self.state.last_decision_step = self.state.total_steps

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for analysis."""
        return {
            'current_option': self.state.current_option.name,
            'total_steps': self.state.total_steps,
            'steps_in_current_option': self.state.steps_in_option,
        }
```

---

#### `hrl/selector_policy.py`
```python
"""
High-level selector policy that chooses options.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from .option_definitions import Option


class SelectorPolicy:
    """
    High-level policy that selects options (search/track/terminal).

    Design: Wraps a Stable-Baselines3 PPO model with discrete action space.
    Action Space: Discrete(3) - {SEARCH, TRACK, TERMINAL}
    Observation: Abstract state (5-7D)
    """

    def __init__(
        self,
        obs_dim: int,
        n_options: int = 3,
        model_path: Optional[str] = None,
        device: str = 'auto',
    ):
        """
        Args:
            obs_dim: Dimension of abstract observation space
            n_options: Number of discrete options (default 3)
            model_path: Path to pre-trained model (optional)
            device: 'cuda', 'cpu', or 'auto'
        """
        self.obs_dim = obs_dim
        self.n_options = n_options
        self.device = device

        # Create dummy environment for SB3 (we'll use predict directly)
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_options)

        # Load or initialize model
        if model_path:
            self.model = PPO.load(model_path, device=device)
        else:
            # Will be initialized during training
            self.model = None

    def predict(
        self,
        abstract_obs: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """
        Select option based on abstract observation.

        Args:
            abstract_obs: Abstract state vector (5-7D)
            deterministic: Use deterministic policy (default True for inference)

        Returns:
            option_index: {0, 1, 2} corresponding to {SEARCH, TRACK, TERMINAL}
        """
        if self.model is None:
            # Fallback: random option (used during initialization)
            return np.random.randint(0, self.n_options)

        # Ensure observation is properly shaped
        obs = np.array(abstract_obs, dtype=np.float32).reshape(1, -1)

        # Get action from policy
        action, _states = self.model.predict(obs, deterministic=deterministic)

        return int(action)

    def save(self, path: str):
        """Save model to disk."""
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str):
        """Load model from disk."""
        self.model = PPO.load(path, device=self.device)
```

---

#### `hrl/specialist_policies.py`
```python
"""
Low-level specialist policies for each option.
"""
import numpy as np
from typing import Optional, Tuple, Any
from stable_baselines3 import PPO

from .option_definitions import Option


class SpecialistPolicy:
    """
    Base class for low-level specialists.

    Each specialist is a continuous action PPO policy specialized for one option.
    Uses LSTM for temporal reasoning and frame stacking for rich state representation.
    """

    def __init__(
        self,
        option_type: Option,
        obs_dim: int = 104,  # 26D base * 4 frame stack
        action_dim: int = 6,  # 3D thrust + 3D angular
        use_lstm: bool = True,
        lstm_hidden_dim: int = 256,
        model_path: Optional[str] = None,
        device: str = 'auto',
    ):
        """
        Args:
            option_type: Which option this specialist handles
            obs_dim: Observation dimension (104 with frame stacking)
            action_dim: Action dimension (6D continuous)
            use_lstm: Enable LSTM for temporal reasoning
            lstm_hidden_dim: LSTM hidden state dimension
            model_path: Path to pre-trained model
            device: Compute device
        """
        self.option_type = option_type
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_lstm = use_lstm
        self.lstm_hidden_dim = lstm_hidden_dim
        self.device = device

        # LSTM state tracking
        self.lstm_state = None

        # Load or initialize model
        if model_path:
            self.model = PPO.load(model_path, device=device)
        else:
            self.model = None

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict action (without LSTM state management).

        Args:
            obs: Full observation (104D stacked)
            deterministic: Use deterministic policy

        Returns:
            action: 6D continuous action
        """
        if self.model is None:
            # Fallback: zero action
            return np.zeros(self.action_dim, dtype=np.float32)

        obs = np.array(obs, dtype=np.float32).reshape(1, -1)
        action, _states = self.model.predict(obs, deterministic=deterministic)

        return action

    def predict_with_lstm(
        self,
        obs: np.ndarray,
        lstm_state: Optional[Tuple],
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Tuple]:
        """
        Predict action with explicit LSTM state management.

        Args:
            obs: Full observation (104D stacked)
            lstm_state: Previous LSTM hidden state (or None)
            deterministic: Use deterministic policy

        Returns:
            action: 6D continuous action
            new_lstm_state: Updated LSTM hidden state
        """
        if self.model is None:
            return np.zeros(self.action_dim, dtype=np.float32), None

        obs = np.array(obs, dtype=np.float32).reshape(1, -1)

        # SB3 PPO with LSTM: pass state as episode_start flag
        action, new_state = self.model.predict(
            obs,
            state=lstm_state,
            episode_start=np.array([lstm_state is None]),
            deterministic=deterministic,
        )

        return action, new_state

    def reset_lstm(self):
        """Reset LSTM hidden state (called on option switch)."""
        self.lstm_state = None

    def save(self, path: str):
        """Save model to disk."""
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str):
        """Load model from disk."""
        self.model = PPO.load(path, device=self.device)


class SearchSpecialist(SpecialistPolicy):
    """
    Specialist for wide-area scanning.

    Objective: Maximize scan coverage, acquire radar lock quickly.
    Action Space: Large angular changes, moderate thrust.
    Reward Focus: Scan coverage, lock acquisition bonus.
    """

    def __init__(self, **kwargs):
        super().__init__(option_type=Option.SEARCH, **kwargs)


class TrackSpecialist(SpecialistPolicy):
    """
    Specialist for maintaining lock and approaching target.

    Objective: Maintain radar lock, reduce distance, optimize intercept geometry.
    Action Space: Smooth tracking, balanced thrust.
    Reward Focus: Lock maintenance, approach progress, closing rate.
    """

    def __init__(self, **kwargs):
        super().__init__(option_type=Option.TRACK, **kwargs)


class TerminalSpecialist(SpecialistPolicy):
    """
    Specialist for final intercept guidance.

    Objective: Minimize miss distance with high precision.
    Action Space: High-precision corrections, maximum thrust.
    Reward Focus: Miss distance shaping, closure rate optimization.
    """

    def __init__(self, **kwargs):
        super().__init__(option_type=Option.TERMINAL, **kwargs)
```

---

### 2.2 State Abstraction & Rewards

#### `hrl/observation_abstraction.py`
```python
"""
Convert 26D radar observations to abstract state for high-level policy.
"""
import numpy as np
from typing import Tuple


def abstract_observation(full_obs: np.ndarray) -> np.ndarray:
    """
    Convert 26D/104D observation to abstract state for selector policy.

    Args:
        full_obs: Either 26D base observation or 104D frame-stacked (26*4)

    Returns:
        abstract_state: 7D vector:
            [0] distance_to_target (normalized)
            [1] closing_rate (normalized)
            [2] radar_lock_quality (0-1)
            [3] fuel_fraction (0-1)
            [4] off_axis_angle (normalized)
            [5] time_to_intercept_estimate (normalized)
            [6] relative_altitude (normalized)

    Design Notes:
    - Selector needs strategic information, not low-level sensor data
    - Normalized to [-1, 1] or [0, 1] for stability
    - Independent of frame stacking dimension
    """
    # Handle frame-stacked observations (extract latest frame)
    if len(full_obs.shape) > 1 or full_obs.shape[0] > 26:
        # Assuming frame stacking: last 26 elements are most recent
        obs = full_obs[-26:] if full_obs.ndim == 1 else full_obs.flatten()[-26:]
    else:
        obs = full_obs

    # Extract features from 26D observation
    # Observation layout (from core.py Radar26DObservation):
    # [0-2]:   relative_position (target - interceptor)
    # [3-5]:   relative_velocity
    # [6-8]:   interceptor_velocity
    # [9-11]:  interceptor_orientation (euler)
    # [12]:    fuel_fraction
    # [13]:    time_to_intercept
    # [14]:    radar_lock_quality
    # [15]:    closing_rate
    # [16]:    off_axis_angle
    # [17-25]: (reserved/padding)

    # 1. Distance to target (normalized by max detection range)
    relative_position = obs[0:3]
    distance = np.linalg.norm(relative_position)
    distance_norm = np.clip(distance / 5000.0, 0, 1)  # 5km max radar range

    # 2. Closing rate (normalized, centered at 0)
    closing_rate = obs[15]
    closing_rate_norm = np.clip(closing_rate / 500.0, -1, 1)  # +/-500 m/s

    # 3. Radar lock quality (already 0-1)
    lock_quality = obs[14]

    # 4. Fuel fraction (already 0-1)
    fuel = obs[12]

    # 5. Off-axis angle (normalized)
    off_axis = obs[16]
    off_axis_norm = np.clip(off_axis / np.pi, -1, 1)  # +/- pi radians

    # 6. Time to intercept estimate (normalized)
    tti = obs[13]
    tti_norm = np.clip(tti / 10.0, 0, 1)  # 0-10 seconds

    # 7. Relative altitude (derived from position)
    rel_altitude = relative_position[2]  # Z component
    altitude_norm = np.clip(rel_altitude / 1000.0, -1, 1)  # +/-1km

    abstract_state = np.array([
        distance_norm,
        closing_rate_norm,
        lock_quality,
        fuel,
        off_axis_norm,
        tti_norm,
        altitude_norm,
    ], dtype=np.float32)

    return abstract_state


def extract_env_state_for_transitions(full_obs: np.ndarray) -> dict:
    """
    Extract environment state features for forced transition checks.

    Returns:
        Dict with keys: 'lock_quality', 'distance', 'fuel', 'closing_rate'
    """
    if len(full_obs.shape) > 1 or full_obs.shape[0] > 26:
        obs = full_obs[-26:] if full_obs.ndim == 1 else full_obs.flatten()[-26:]
    else:
        obs = full_obs

    distance = np.linalg.norm(obs[0:3])

    return {
        'lock_quality': float(obs[14]),
        'distance': float(distance),
        'fuel': float(obs[12]),
        'closing_rate': float(obs[15]),
    }
```

---

#### `hrl/reward_decomposition.py`
```python
"""
Hierarchical reward decomposition: strategic (selector) + tactical (specialists).
"""
import numpy as np
from typing import Dict, Any
from .option_definitions import Option


def compute_strategic_reward(
    env_state: Dict[str, Any],
    option: Option,
    next_env_state: Dict[str, Any],
    episode_done: bool,
    episode_info: Dict[str, Any],
) -> float:
    """
    High-level reward for option selection (selector policy).

    Focus: Long-term strategic outcomes (intercept success, efficiency).

    Args:
        env_state: Current state features
        option: Selected option
        next_env_state: Next state features
        episode_done: Whether episode terminated
        episode_info: Episode termination info

    Returns:
        reward: Strategic reward signal
    """
    reward = 0.0

    # Terminal rewards (only at episode end)
    if episode_done:
        if episode_info.get('intercept_success', False):
            # Major bonus for successful intercept
            reward += 1000.0

            # Efficiency bonus (fuel remaining)
            fuel_remaining = next_env_state.get('fuel', 0.0)
            reward += fuel_remaining * 100.0

        elif episode_info.get('reason') == 'out_of_fuel':
            # Penalty for fuel depletion
            reward -= 200.0

        elif episode_info.get('reason') == 'timeout':
            # Penalty for timeout
            reward -= 100.0

    # Shaping rewards (every step)
    # Penalize distance to encourage progress
    distance = next_env_state.get('distance', float('inf'))
    if distance < float('inf'):
        reward -= distance * 0.01  # Small penalty proportional to distance

    # Reward closing rate (approach progress)
    closing_rate = next_env_state.get('closing_rate', 0.0)
    if closing_rate > 0:
        reward += closing_rate * 0.1

    return reward


def compute_tactical_reward(
    env_state: Dict[str, Any],
    action: np.ndarray,
    option: Option,
    next_env_state: Dict[str, Any],
    episode_done: bool,
) -> float:
    """
    Low-level reward for specialist execution (specialist policies).

    Focus: Option-specific tactical objectives.

    Args:
        env_state: Current state features
        action: 6D continuous action taken
        option: Active option
        next_env_state: Next state features
        episode_done: Whether episode terminated

    Returns:
        reward: Tactical reward signal
    """
    if option == Option.SEARCH:
        return compute_search_reward(env_state, action, next_env_state, episode_done)
    elif option == Option.TRACK:
        return compute_track_reward(env_state, action, next_env_state, episode_done)
    elif option == Option.TERMINAL:
        return compute_terminal_reward(env_state, action, next_env_state, episode_done)
    else:
        return 0.0


def compute_search_reward(
    env_state: Dict[str, Any],
    action: np.ndarray,
    next_env_state: Dict[str, Any],
    episode_done: bool,
) -> float:
    """Reward for search specialist: maximize scan coverage, acquire lock."""
    reward = 0.0

    # Reward for acquiring radar lock
    lock_quality_before = env_state.get('lock_quality', 0.0)
    lock_quality_after = next_env_state.get('lock_quality', 0.0)

    if lock_quality_before < 0.3 and lock_quality_after >= 0.3:
        # Lock acquired!
        reward += 50.0

    # Reward for improving lock quality
    lock_improvement = lock_quality_after - lock_quality_before
    reward += lock_improvement * 10.0

    # Encourage angular diversity (scan coverage)
    angular_action = action[3:6]  # Angular components
    angular_magnitude = np.linalg.norm(angular_action)
    reward += angular_magnitude * 0.5  # Small bonus for exploration

    # Penalize fuel waste during search
    fuel_usage = env_state.get('fuel', 1.0) - next_env_state.get('fuel', 1.0)
    reward -= fuel_usage * 5.0

    return reward


def compute_track_reward(
    env_state: Dict[str, Any],
    action: np.ndarray,
    next_env_state: Dict[str, Any],
    episode_done: bool,
) -> float:
    """Reward for track specialist: maintain lock, close distance."""
    reward = 0.0

    # Critical: Maintain radar lock
    lock_quality = next_env_state.get('lock_quality', 0.0)
    if lock_quality >= 0.7:
        reward += 2.0  # Steady bonus for good lock
    elif lock_quality < 0.3:
        reward -= 10.0  # Penalty for losing lock

    # Reward approach progress
    distance_before = env_state.get('distance', float('inf'))
    distance_after = next_env_state.get('distance', float('inf'))
    distance_reduction = distance_before - distance_after
    reward += distance_reduction * 1.0

    # Reward positive closing rate
    closing_rate = next_env_state.get('closing_rate', 0.0)
    if closing_rate > 0:
        reward += closing_rate * 0.5

    # Penalize excessive angular action (should be tracking smoothly)
    angular_action = action[3:6]
    angular_magnitude = np.linalg.norm(angular_action)
    reward -= angular_magnitude * 0.2  # Penalize jerky movements

    return reward


def compute_terminal_reward(
    env_state: Dict[str, Any],
    action: np.ndarray,
    next_env_state: Dict[str, Any],
    episode_done: bool,
) -> float:
    """Reward for terminal specialist: minimize miss distance."""
    reward = 0.0

    # Primary objective: minimize miss distance
    distance = next_env_state.get('distance', float('inf'))
    if distance < float('inf'):
        # Exponential reward for getting very close
        reward += np.exp(-distance / 10.0) * 10.0

        # Strong penalty for increasing distance
        distance_before = env_state.get('distance', float('inf'))
        if distance > distance_before:
            reward -= (distance - distance_before) * 5.0

    # Reward high closing rate (impact speed)
    closing_rate = next_env_state.get('closing_rate', 0.0)
    reward += closing_rate * 1.0

    # Encourage maximum thrust in terminal phase
    thrust_action = action[0:3]
    thrust_magnitude = np.linalg.norm(thrust_action)
    reward += thrust_magnitude * 0.5

    return reward
```

---

### 2.3 Gymnasium Integration

#### `hrl/wrappers.py`
```python
"""
Gymnasium wrappers for hierarchical RL.
"""
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional

from .manager import HierarchicalManager
from .observation_abstraction import abstract_observation, extract_env_state_for_transitions


class HierarchicalActionWrapper(gym.Wrapper):
    """
    Wrapper to use HierarchicalManager for action selection.

    This allows existing InterceptEnvironment to work with HRL without modification.
    """

    def __init__(
        self,
        env: gym.Env,
        manager: HierarchicalManager,
        return_hrl_info: bool = True,
    ):
        """
        Args:
            env: Base InterceptEnvironment
            manager: HierarchicalManager instance
            return_hrl_info: Include HRL info in step() info dict
        """
        super().__init__(env)
        self.manager = manager
        self.return_hrl_info = return_hrl_info

        # Action space remains unchanged (6D continuous from specialist)
        # Observation space remains unchanged (26D or 104D with frame stacking)

    def reset(self, **kwargs):
        """Reset environment and HRL manager."""
        obs, info = self.env.reset(**kwargs)
        self.manager.reset()
        return obs, info

    def step(self, action=None):
        """
        Step with hierarchical action selection.

        Args:
            action: Ignored - actions come from HRL manager

        Returns:
            Standard Gymnasium step output with HRL info
        """
        # Get current observation (needed for manager)
        # Note: This assumes we store last observation, or we get it from env state
        # For simplicity, we'll extract env state from observation

        # Get action from manager
        obs = self._get_current_obs()
        env_state = extract_env_state_for_transitions(obs)
        action, hrl_info = self.manager.select_action(obs, env_state)

        # Execute in base environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Add HRL info
        if self.return_hrl_info:
            info.update(hrl_info)

        return next_obs, reward, terminated, truncated, info

    def _get_current_obs(self) -> np.ndarray:
        """
        Get current observation from environment.

        Workaround: Gymnasium doesn't provide current obs in step().
        We store it from last reset/step call.
        """
        # This would be set in a more complete implementation
        # For now, return dummy observation
        return np.zeros(self.observation_space.shape, dtype=np.float32)


class AbstractObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert 26D → abstract state for selector training.

    Use this when training the selector policy.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Override observation space to abstract state dimension
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),  # Abstract state dimension
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert full obs to abstract state."""
        return abstract_observation(obs)
```

---

## 3. Migration Plan

### Phase 1: Setup (No Disruption) - Day 1

**Objective:** Create directory structure and update imports without breaking existing code.

**Steps:**

1. **Create new directories:**
   ```bash
   cd /Users/quinnhasse/Hlynr_Intercept/rl_system
   mkdir -p hrl tests scripts docs/hrl
   mkdir -p configs/scenarios configs/hrl configs/experimental
   ```

2. **Move scenario files:**
   ```bash
   # Backup existing scenarios
   cp -r scenarios scenarios_backup

   # Move to new location
   mv scenarios/* configs/scenarios/

   # Create symlink for backward compatibility
   ln -s configs/scenarios scenarios
   ```

3. **Update requirements.txt:**
   ```txt
   # Existing dependencies (unchanged)
   gymnasium>=0.29.0
   stable-baselines3>=2.0.0
   torch>=2.0.0
   numpy>=1.24.0
   pyyaml>=6.0
   fastapi>=0.104.0
   uvicorn>=0.24.0
   matplotlib>=3.7.0
   tensorboard>=2.14.0

   # New dependencies for HRL
   pytest>=7.4.0
   pytest-cov>=4.1.0
   ```

4. **Create placeholder __init__.py files:**
   ```bash
   touch hrl/__init__.py
   touch tests/__init__.py
   touch scripts/__init__.py
   ```

**Validation:**
```bash
# Existing training should still work
python train.py --config config.yaml
```

---

### Phase 2: Core Modules (Parallel Development) - Days 2-4

**Objective:** Implement HRL core modules without touching existing code.

**Day 2: Option Definitions & Observation Abstraction**

1. Implement `hrl/option_definitions.py` (see interface above)
2. Implement `hrl/observation_abstraction.py` (see interface above)
3. Write tests: `tests/test_observation_abstraction.py`

**Day 3: Specialist Policies**

1. Implement `hrl/specialist_policies.py` (see interface above)
2. Write tests: `tests/test_specialist_policies.py`
3. Test loading existing PPO checkpoints

**Day 4: Selector Policy & Manager**

1. Implement `hrl/selector_policy.py` (see interface above)
2. Implement `hrl/option_manager.py` (forced transitions)
3. Implement `hrl/manager.py` (see interface above)
4. Write tests: `tests/test_hrl_manager.py`

**Validation:**
```bash
# Run unit tests
cd /Users/quinnhasse/Hlynr_Intercept/rl_system
pytest tests/ -v
```

---

### Phase 3: Reward Decomposition & Wrappers - Days 5-6

**Day 5: Reward System**

1. Implement `hrl/reward_decomposition.py` (see interface above)
2. Write tests: `tests/test_reward_decomposition.py`

**Day 6: Gymnasium Integration**

1. Implement `hrl/wrappers.py` (see interface above)
2. Implement `hrl/hierarchical_env.py` (complete wrapper)
3. Write tests: `tests/test_wrappers.py`, `tests/test_hierarchical_env.py`

**Validation:**
```bash
# Test HRL environment wrapper
python -c "from hrl.hierarchical_env import create_hrl_env; env = create_hrl_env(); print('OK')"
```

---

### Phase 4: Training Scripts - Days 7-9

**Day 7: Refactor Flat PPO**

1. Extract training logic from `train.py` → `scripts/train_flat_ppo.py`
2. Make `train.py` a thin wrapper:
   ```python
   # train.py (modified)
   if __name__ == "__main__":
       from scripts.train_flat_ppo import main
       main()
   ```
3. Verify backward compatibility:
   ```bash
   python train.py --config config.yaml  # Should work identically
   ```

**Day 8: HRL Pre-training**

1. Implement `scripts/train_hrl_pretrain.py`
   - Train search specialist on search-only episodes
   - Train track specialist on tracking scenarios
   - Train terminal specialist on close-range scenarios
2. Create configs: `configs/hrl/search_specialist.yaml`, etc.

**Day 9: HRL Selector Training**

1. Implement `scripts/train_hrl_selector.py`
   - Load frozen specialist checkpoints
   - Train selector on abstract observations
2. Implement `scripts/train_hrl_full.py` (end-to-end orchestrator)

---

### Phase 5: Testing & Documentation - Days 10-12

**Day 10: Integration Testing**

1. Write `tests/test_end_to_end.py`
2. Write `tests/test_backward_compatibility.py`
3. Run full test suite

**Day 11: Checkpoint Migration**

1. Implement `scripts/migrate_checkpoints.py`
2. Move existing checkpoints to `checkpoints/flat_ppo/`

**Day 12: Documentation**

1. Write `docs/hrl/architecture.md`
2. Write `docs/hrl/training_guide.md`
3. Update main `README.md`

---

## 4. Backward Compatibility Strategy

### Principle: Zero Breaking Changes

**Existing workflows must work identically:**

```bash
# OLD (still works)
cd rl_system/
python train.py --config config.yaml
python inference.py --model checkpoints/best

# NEW (parallel)
python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml
python scripts/evaluate_hrl.py --model checkpoints/hrl/selector/
```

### Implementation Details

**1. Symlink Strategy for Scenarios:**
```bash
# scenarios/ → configs/scenarios/ (symlink)
# Code using 'scenarios/easy.yaml' will still work
```

**2. train.py Wrapper Pattern:**
```python
# train.py (backward compatible wrapper)
import sys
import warnings

def main():
    """Backward compatible entry point."""
    # Detect HRL mode
    if "--hrl" in sys.argv:
        warnings.warn(
            "Using --hrl flag is deprecated. "
            "Use scripts/train_hrl_full.py instead.",
            DeprecationWarning
        )
        from scripts.train_hrl_full import main as hrl_main
        return hrl_main()
    else:
        # Existing flat PPO training
        from scripts.train_flat_ppo import main as flat_main
        return flat_main()

if __name__ == "__main__":
    main()
```

**3. Config Path Resolution:**
```python
# Automatic path resolution for backward compat
def resolve_config_path(path: str) -> str:
    """Resolve config path with backward compatibility."""
    if os.path.exists(path):
        return path

    # Try old location (scenarios/)
    old_path = path.replace("configs/scenarios/", "scenarios/")
    if os.path.exists(old_path):
        warnings.warn(f"Using deprecated path {old_path}, use {path} instead")
        return old_path

    # Try new location
    new_path = path.replace("scenarios/", "configs/scenarios/")
    if os.path.exists(new_path):
        return new_path

    raise FileNotFoundError(f"Config not found: {path}")
```

---

## 5. Configuration File Design

### Base HRL Configuration

**`configs/hrl/hrl_base.yaml`**

```yaml
# HRL Base Configuration
# Inherits from: ../config.yaml

# Hierarchy Settings
hrl:
  enabled: true

  hierarchy:
    levels: 2  # Selector (high) + Specialists (low)
    decision_interval: 100  # High-level decides every 100 steps (1Hz @ 100Hz sim)
    forced_transitions: true  # Allow environment to force option changes
    default_option: 0  # Start with SEARCH

  # Selector Policy (High-Level)
  selector:
    algorithm: PPO
    obs_dim: 7  # Abstract state dimension
    n_options: 3  # {SEARCH, TRACK, TERMINAL}

    # Training hyperparameters
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01

    # Network architecture
    net_arch:
      pi: [256, 256]
      vf: [256, 256]

    # Checkpoints
    checkpoint_dir: checkpoints/hrl/selector/
    save_freq: 10000

  # Specialist Policies (Low-Level)
  specialists:
    # Search Specialist
    - name: search
      option_id: 0
      checkpoint_dir: checkpoints/hrl/specialists/search/

      # Pre-training configuration
      pretrain:
        enabled: true
        episodes: 200
        curriculum:
          - scenario: easy
            episodes: 100
          - scenario: medium
            episodes: 100

      # Training hyperparameters (same as flat PPO)
      learning_rate: 0.0003
      n_steps: 2048
      batch_size: 64
      n_epochs: 10
      gamma: 0.99
      gae_lambda: 0.95
      clip_range: 0.2
      ent_coef: 0.02  # Higher entropy for exploration

      # Network architecture
      use_lstm: true
      lstm_hidden_dim: 256
      net_arch: [512, 512, 256]

    # Track Specialist
    - name: track
      option_id: 1
      checkpoint_dir: checkpoints/hrl/specialists/track/

      pretrain:
        enabled: true
        episodes: 200
        # Start from scenarios with existing lock
        initial_lock_quality: 0.7

      learning_rate: 0.0003
      use_lstm: true
      lstm_hidden_dim: 256
      net_arch: [512, 512, 256]

    # Terminal Specialist
    - name: terminal
      option_id: 2
      checkpoint_dir: checkpoints/hrl/specialists/terminal/

      pretrain:
        enabled: true
        episodes: 200
        # Start from close-range scenarios
        initial_distance_max: 200.0

      learning_rate: 0.0003
      use_lstm: true
      lstm_hidden_dim: 256
      net_arch: [512, 512, 256]

  # Reward Decomposition
  rewards:
    # Strategic rewards (for selector)
    strategic:
      intercept_success: 1000.0
      fuel_efficiency_bonus: 100.0  # Multiplied by fuel_remaining
      timeout_penalty: -100.0
      out_of_fuel_penalty: -200.0
      distance_shaping: -0.01  # Per meter
      closing_rate_bonus: 0.1  # Per m/s

    # Tactical rewards (for specialists)
    tactical:
      search:
        lock_acquisition_bonus: 50.0
        lock_improvement: 10.0  # Per 0.1 increase
        angular_diversity_bonus: 0.5
        fuel_waste_penalty: -5.0

      track:
        lock_maintenance_bonus: 2.0  # Per step with good lock
        lock_loss_penalty: -10.0
        distance_reduction: 1.0  # Per meter closed
        closing_rate_bonus: 0.5
        jerky_movement_penalty: -0.2

      terminal:
        proximity_bonus_scale: 10.0  # exp(-dist/10) * scale
        distance_increase_penalty: -5.0
        closing_rate_bonus: 1.0
        max_thrust_bonus: 0.5

  # Forced Transition Thresholds
  transitions:
    radar_lock_quality_min: 0.3  # Below → lose track
    radar_lock_quality_search: 0.7  # Above → exit search
    close_range_threshold: 100.0  # meters → trigger terminal
    terminal_fuel_min: 0.1  # Min fuel for terminal entry

  # Training Curriculum
  curriculum:
    enabled: true

    stages:
      # Stage 1: Pre-train specialists
      - name: pretrain_specialists
        duration_episodes: 600  # 200 per specialist
        train_selector: false
        train_specialists: true
        freeze_specialists: false

      # Stage 2: Train selector with frozen specialists
      - name: train_selector
        duration_episodes: 1000
        train_selector: true
        train_specialists: false
        freeze_specialists: true

      # Stage 3: Joint fine-tuning (optional)
      - name: joint_finetuning
        duration_episodes: 500
        train_selector: true
        train_specialists: true
        freeze_specialists: false
        learning_rate_multiplier: 0.1  # Lower LR for stability

# Inherit all physics settings from base config
physics_enhancements:
  enabled: true
  # ... (same as config.yaml)

# Environment settings (same as base)
environment:
  dt: 0.01
  max_steps: 1000
  # ... (same as config.yaml)
```

---

## 6. Testing Strategy

### Unit Tests (80%+ Coverage Required)

**`tests/test_observation_abstraction.py`**

```python
import numpy as np
import pytest
from hrl.observation_abstraction import abstract_observation, extract_env_state_for_transitions


def test_abstract_observation_shape():
    """Abstract state should be 7D."""
    full_obs = np.random.randn(26)
    abstract = abstract_observation(full_obs)
    assert abstract.shape == (7,)


def test_abstract_observation_normalized():
    """All features should be normalized."""
    full_obs = np.random.randn(26)
    abstract = abstract_observation(full_obs)
    assert np.all(np.abs(abstract) <= 1.0)


def test_abstract_observation_frame_stacked():
    """Should handle frame-stacked observations."""
    stacked_obs = np.random.randn(104)  # 26 * 4
    abstract = abstract_observation(stacked_obs)
    assert abstract.shape == (7,)


def test_extract_env_state():
    """Extract env state should return dict with required keys."""
    full_obs = np.zeros(26)
    full_obs[14] = 0.8  # lock quality
    full_obs[12] = 0.5  # fuel

    env_state = extract_env_state_for_transitions(full_obs)

    assert 'lock_quality' in env_state
    assert 'distance' in env_state
    assert 'fuel' in env_state
    assert 'closing_rate' in env_state
    assert env_state['lock_quality'] == 0.8
```

**`tests/test_hrl_manager.py`**

```python
import numpy as np
import pytest
from hrl.manager import HierarchicalManager
from hrl.selector_policy import SelectorPolicy
from hrl.specialist_policies import SearchSpecialist, TrackSpecialist, TerminalSpecialist
from hrl.option_definitions import Option


@pytest.fixture
def dummy_policies():
    """Create dummy policies for testing."""
    selector = SelectorPolicy(obs_dim=7, n_options=3)
    specialists = {
        Option.SEARCH: SearchSpecialist(),
        Option.TRACK: TrackSpecialist(),
        Option.TERMINAL: TerminalSpecialist(),
    }
    return selector, specialists


def test_manager_initialization(dummy_policies):
    """Manager should initialize correctly."""
    selector, specialists = dummy_policies
    manager = HierarchicalManager(selector, specialists)

    assert manager.state.current_option == Option.SEARCH
    assert manager.state.steps_in_option == 0


def test_manager_action_selection(dummy_policies):
    """Manager should return valid actions."""
    selector, specialists = dummy_policies
    manager = HierarchicalManager(selector, specialists)

    obs = np.random.randn(26)
    env_state = {'lock_quality': 0.5, 'distance': 1000.0, 'fuel': 0.8}

    action, info = manager.select_action(obs, env_state)

    assert action.shape == (6,)  # 6D continuous action
    assert 'option' in info
    assert 'option_switched' in info


def test_manager_forced_transition(dummy_policies):
    """Manager should handle forced transitions."""
    selector, specialists = dummy_policies
    manager = HierarchicalManager(selector, specialists, enable_forced_transitions=True)

    # Force transition by losing radar lock
    manager.state.current_option = Option.TRACK
    env_state = {
        'lock_quality': 0.2,  # Below minimum
        'distance': 1000.0,
        'fuel': 0.8,
    }

    obs = np.random.randn(26)
    action, info = manager.select_action(obs, env_state)

    assert info['forced_transition'] == True
```

### Integration Tests

**`tests/test_end_to_end.py`**

```python
def test_hrl_full_episode():
    """Run full episode with HRL."""
    from hrl.hierarchical_env import create_hrl_env

    env = create_hrl_env()
    obs, info = env.reset()

    total_reward = 0
    done = False
    step_count = 0

    while not done and step_count < 1000:
        action = env.action_space.sample()  # Random actions for test
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_count += 1

    assert step_count > 0
    assert 'option' in info  # HRL info should be present
```

**`tests/test_backward_compatibility.py`**

```python
def test_flat_ppo_unchanged():
    """Verify flat PPO training still works."""
    from environment import InterceptEnvironment
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env = InterceptEnvironment(config)
    obs, info = env.reset()

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
```

---

## 7. Common Pitfalls to Avoid

### Don't Do This

**1. Modifying Core Files Extensively**
```python
# BAD: Editing environment.py to add HRL logic
class InterceptEnvironment(gym.Env):
    def __init__(self, config, use_hrl=False):  # ❌ Breaks backward compat
        if use_hrl:
            # HRL-specific setup
            ...
```

**2. Hard-Coding Option Indices**
```python
# BAD: Magic numbers everywhere
if option == 0:  # ❌ What is 0?
    ...
elif option == 1:
    ...
```

**3. Circular Dependencies**
```python
# BAD: manager.py imports hierarchical_env.py which imports manager.py
# ❌ Creates import loops
```

**4. Mixing Abstractions**
```python
# BAD: Selector policy accessing low-level sensor data
def select_option(self, full_26d_obs):  # ❌ Should use abstract state
    radar_noise = full_26d_obs[17]  # Breaks abstraction
```

### Do This Instead

**1. Use Composition Over Modification**
```python
# GOOD: Wrapper pattern
class HierarchicalEnvWrapper(gym.Wrapper):
    def __init__(self, env, manager):
        super().__init__(env)
        self.manager = manager
```

**2. Use Enums for Options**
```python
# GOOD: Type-safe, self-documenting
from hrl.option_definitions import Option

if option == Option.SEARCH:
    ...
```

**3. Clear Module Boundaries**
```python
# GOOD: One-way dependencies
# manager.py imports selector_policy.py ✓
# selector_policy.py does NOT import manager.py ✓
```

**4. Respect Abstraction Layers**
```python
# GOOD: Selector uses abstract state
def select_option(self, abstract_state: np.ndarray):
    distance, closing_rate, lock_quality, fuel, ... = abstract_state
```

---

## 8. Implementation Checklist

### Pre-Implementation

- [ ] Create feature branch `feature/hrl-integration`
- [ ] Run existing tests: `pytest tests/` (establish baseline)
- [ ] Document current flat PPO metrics (intercept rate, training time)
- [ ] Backup checkpoints: `cp -r checkpoints checkpoints_backup`

### Phase 1: Setup (Day 1)

- [ ] Create directory structure (`hrl/`, `configs/`, `scripts/`, `tests/`)
- [ ] Move `scenarios/` → `configs/scenarios/`
- [ ] Create symlink `scenarios` → `configs/scenarios`
- [ ] Update `requirements.txt` with pytest, pytest-cov
- [ ] Create `__init__.py` files
- [ ] Verify: `python train.py --config config.yaml` still works

### Phase 2: Core Modules (Days 2-4)

- [ ] Implement `hrl/option_definitions.py`
- [ ] Implement `hrl/observation_abstraction.py`
- [ ] Write `tests/test_observation_abstraction.py`
- [ ] Implement `hrl/specialist_policies.py`
- [ ] Write `tests/test_specialist_policies.py`
- [ ] Implement `hrl/selector_policy.py`
- [ ] Implement `hrl/option_manager.py`
- [ ] Implement `hrl/manager.py`
- [ ] Write `tests/test_hrl_manager.py`
- [ ] Run: `pytest tests/ -v` (all pass)

### Phase 3: Rewards & Wrappers (Days 5-6)

- [ ] Implement `hrl/reward_decomposition.py`
- [ ] Write `tests/test_reward_decomposition.py`
- [ ] Implement `hrl/wrappers.py`
- [ ] Implement `hrl/hierarchical_env.py`
- [ ] Write `tests/test_hierarchical_env.py`
- [ ] Run: `pytest tests/ -v --cov=hrl` (>80% coverage)

### Phase 4: Training Scripts (Days 7-9)

- [ ] Extract `train.py` logic → `scripts/train_flat_ppo.py`
- [ ] Make `train.py` a thin wrapper
- [ ] Verify: Flat PPO training unchanged
- [ ] Implement `scripts/train_hrl_pretrain.py`
- [ ] Create `configs/hrl/*.yaml` files
- [ ] Implement `scripts/train_hrl_selector.py`
- [ ] Implement `scripts/train_hrl_full.py`
- [ ] Test: Run 1000-step training for each script

### Phase 5: Testing & Docs (Days 10-12)

- [ ] Write `tests/test_end_to_end.py`
- [ ] Write `tests/test_backward_compatibility.py`
- [ ] Run full test suite: `pytest tests/ -v --cov=hrl`
- [ ] Implement `scripts/migrate_checkpoints.py`
- [ ] Migrate existing checkpoints to `checkpoints/flat_ppo/`
- [ ] Write `docs/hrl/architecture.md`
- [ ] Write `docs/hrl/training_guide.md`
- [ ] Update `README.md` with HRL section

### Validation

- [ ] Flat PPO training: 5M steps, verify 75%+ intercept rate
- [ ] HRL pre-training: Train all specialists, check convergence
- [ ] HRL selector training: Train selector, verify option switching
- [ ] End-to-end HRL: Full training, compare vs flat PPO
- [ ] Checkpoint loading: Verify all models load correctly
- [ ] TensorBoard logs: Verify metrics are interpretable
- [ ] Code coverage: >80% for all `hrl/` modules

---

## 9. Success Criteria

### Functional Requirements

1. **Backward Compatibility (Critical)**
   - Existing `train.py` workflow works identically
   - All existing checkpoints load successfully
   - No breaking changes to `core.py`, `environment.py`, `physics_*.py`

2. **HRL Pipeline (Core)**
   - Specialists pre-train successfully (convergence in 200 episodes)
   - Selector trains with frozen specialists
   - Full HRL inference works end-to-end
   - Option switching occurs based on state/time

3. **Code Quality (Essential)**
   - All modules have >80% test coverage
   - No circular dependencies
   - Clear separation of concerns
   - Type hints where appropriate

### Performance Targets

1. **Training Efficiency**
   - Specialist pre-training: <30 minutes per specialist (5M steps)
   - Selector training: <20 minutes (3M steps)
   - End-to-end HRL: <2 hours total

2. **Inference Performance**
   - HRL decision overhead: <5ms per step
   - Option switching: <1ms
   - No memory leaks over 10,000 steps

3. **Policy Performance (Baseline)**
   - HRL intercept rate: ≥70% (within 10% of flat PPO)
   - Fuel efficiency: Comparable or better than flat PPO
   - Option diversity: Use all 3 options in >50% of episodes

---

## 10. Future Extensions (Out of Scope for Initial Implementation)

### Post-V1 Enhancements

1. **Goal-Conditioned Specialists**
   - Pass abstract goals to specialists
   - Enable multi-target scenarios

2. **Meta-Learning**
   - Rapid adaptation to new missile types
   - Few-shot learning for new scenarios

3. **Multi-Agent Coordination**
   - Multiple interceptors with shared selector
   - Coordinated volley defense

4. **Skill Discovery**
   - Automatic option discovery (not hand-designed)
   - Hierarchical Actor-Critic (HAC)

---

## Appendix A: File Dependency Graph

```
Core Dependencies (No HRL)
===========================
environment.py
├── core.py
├── physics_models.py
└── physics_randomizer.py

train.py
├── environment.py
└── logger.py

inference.py
├── environment.py
└── logger.py


HRL Dependencies (New)
======================
hrl/option_definitions.py
└── (no dependencies)

hrl/observation_abstraction.py
└── numpy

hrl/specialist_policies.py
├── hrl/option_definitions.py
└── stable_baselines3

hrl/selector_policy.py
├── hrl/option_definitions.py
└── stable_baselines3

hrl/option_manager.py
└── hrl/option_definitions.py

hrl/manager.py
├── hrl/option_definitions.py
├── hrl/selector_policy.py
├── hrl/specialist_policies.py
├── hrl/option_manager.py
└── hrl/observation_abstraction.py

hrl/reward_decomposition.py
└── hrl/option_definitions.py

hrl/wrappers.py
├── hrl/manager.py
└── hrl/observation_abstraction.py

hrl/hierarchical_env.py
├── environment.py
├── hrl/manager.py
└── hrl/wrappers.py

scripts/train_hrl_*.py
├── environment.py
├── hrl/hierarchical_env.py
└── logger.py
```

---

## Appendix B: Quick Start Commands

### Backward Compatible (Existing Workflow)

```bash
# Activate environment
cd /Users/quinnhasse/Hlynr_Intercept/rl_system

# Train flat PPO (unchanged)
python train.py --config config.yaml

# Inference (unchanged)
python inference.py --model checkpoints/best --mode offline --episodes 100
```

### New HRL Workflow

```bash
# Pre-train specialists
python scripts/train_hrl_pretrain.py --config configs/hrl/hrl_curriculum.yaml

# Train selector
python scripts/train_hrl_selector.py --config configs/hrl/hrl_curriculum.yaml

# Full HRL training (end-to-end)
python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml

# Evaluate HRL
python scripts/evaluate_hrl.py --model checkpoints/hrl/selector/ --episodes 100

# Compare policies
python scripts/compare_policies.py \
    --flat checkpoints/flat_ppo/best/ \
    --hrl checkpoints/hrl/selector/ \
    --episodes 100
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=hrl --cov-report=html

# Specific test
pytest tests/test_hrl_manager.py -v
```

---

## Summary

This refactoring design achieves the following objectives:

1. **Zero Disruption:** Existing flat PPO workflow remains unchanged
2. **Clean Separation:** All HRL code isolated in `hrl/` module
3. **Testable:** Comprehensive unit and integration tests
4. **Extensible:** Clear interfaces for future enhancements
5. **Production-Ready:** Type hints, error handling, logging

The design prioritizes **composition over modification**, ensuring the existing codebase remains stable while adding powerful hierarchical RL capabilities in a modular, maintainable way.

**Next Steps:**
1. Review this design document with team
2. Create feature branch
3. Begin Phase 1 implementation (setup)
4. Iterate with frequent testing

**Estimated Timeline:** 12 days for full implementation + testing + documentation.
