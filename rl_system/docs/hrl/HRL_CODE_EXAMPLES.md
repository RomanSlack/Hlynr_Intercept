# HRL Implementation: Code Examples

This document provides concrete code examples for implementing the HRL system.

## Table of Contents

1. [Option Definitions](#option-definitions)
2. [Observation Abstraction](#observation-abstraction)
3. [Specialist Policies](#specialist-policies)
4. [Selector Policy](#selector-policy)
5. [Option Manager](#option-manager)
6. [Hierarchical Manager](#hierarchical-manager)
7. [Reward Decomposition](#reward-decomposition)
8. [Environment Wrappers](#environment-wrappers)
9. [Training Scripts](#training-scripts)
10. [Testing Examples](#testing-examples)

---

## Option Definitions

**File:** `hrl/option_definitions.py`

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
OPTION_METADATA: Dict[Option, Dict[str, Any]] = {
    Option.SEARCH: {
        'name': 'Search',
        'description': 'Wide-area scanning, large angular changes',
        'expected_duration': 200,  # steps
        'forced_exit_conditions': ['radar_lock_acquired'],
        'color': '#FF6B6B',  # Red for visualization
        'default_reward_weight': 1.0,
    },
    Option.TRACK: {
        'name': 'Track',
        'description': 'Maintain radar lock, approach target',
        'expected_duration': 500,
        'forced_exit_conditions': ['radar_lock_lost', 'close_range_threshold'],
        'color': '#4ECDC4',  # Teal
        'default_reward_weight': 1.0,
    },
    Option.TERMINAL: {
        'name': 'Terminal',
        'description': 'Final guidance, high precision',
        'expected_duration': 100,
        'forced_exit_conditions': ['intercept_complete', 'miss_imminent'],
        'color': '#95E1D3',  # Light green
        'default_reward_weight': 2.0,  # Higher weight for critical phase
    },
}


# Physical thresholds for forced transitions
FORCED_TRANSITION_THRESHOLDS = {
    'radar_lock_quality_min': 0.3,      # Below this, lose track → SEARCH
    'radar_lock_quality_search': 0.7,   # Above this, exit search → TRACK
    'close_range_threshold': 100.0,     # meters - trigger TERMINAL
    'terminal_fuel_min': 0.1,           # Min fuel to enter terminal
    'miss_imminent_distance': 200.0,    # If distance increasing beyond this, abort
}


def get_option_name(option: Option) -> str:
    """Get human-readable name for option."""
    return OPTION_METADATA[option]['name']


def get_option_color(option: Option) -> str:
    """Get color for visualization."""
    return OPTION_METADATA[option]['color']


def get_expected_duration(option: Option) -> int:
    """Get expected duration in steps."""
    return OPTION_METADATA[option]['expected_duration']
```

---

## Observation Abstraction

**File:** `hrl/observation_abstraction.py`

```python
"""
Convert 26D radar observations to abstract state for high-level policy.
"""
import numpy as np
from typing import Dict, Any


def abstract_observation(full_obs: np.ndarray) -> np.ndarray:
    """
    Convert 26D/104D observation to abstract state for selector policy.

    Args:
        full_obs: Either 26D base observation or 104D frame-stacked (26*4)

    Returns:
        abstract_state: 7D vector:
            [0] distance_to_target (normalized to [0, 1])
            [1] closing_rate (normalized to [-1, 1])
            [2] radar_lock_quality (0-1)
            [3] fuel_fraction (0-1)
            [4] off_axis_angle (normalized to [-1, 1])
            [5] time_to_intercept_estimate (normalized to [0, 1])
            [6] relative_altitude (normalized to [-1, 1])
    """
    # Handle frame-stacked observations (extract latest frame)
    if len(full_obs.shape) > 1 or full_obs.shape[0] > 26:
        # Assuming frame stacking: last 26 elements are most recent
        obs = full_obs[-26:] if full_obs.ndim == 1 else full_obs.flatten()[-26:]
    else:
        obs = full_obs

    # Observation layout (from core.py):
    # [0-2]:   relative_position (target - interceptor)
    # [3-5]:   relative_velocity
    # [6-8]:   interceptor_velocity
    # [9-11]:  interceptor_orientation (euler)
    # [12]:    fuel_fraction
    # [13]:    time_to_intercept
    # [14]:    radar_lock_quality
    # [15]:    closing_rate
    # [16]:    off_axis_angle

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


def extract_env_state_for_transitions(full_obs: np.ndarray) -> Dict[str, float]:
    """
    Extract environment state features for forced transition checks.

    Args:
        full_obs: 26D or 104D observation

    Returns:
        Dict with keys: 'lock_quality', 'distance', 'fuel', 'closing_rate'
    """
    # Handle frame stacking
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


def is_valid_abstract_state(abstract_state: np.ndarray) -> bool:
    """
    Validate abstract state.

    Args:
        abstract_state: 7D abstract state

    Returns:
        True if valid, False otherwise
    """
    if abstract_state.shape != (7,):
        return False

    # Check normalization bounds
    if not np.all(np.isfinite(abstract_state)):
        return False

    # Distance, lock, fuel, TTI should be [0, 1]
    if not (0 <= abstract_state[0] <= 1):
        return False
    if not (0 <= abstract_state[2] <= 1):
        return False
    if not (0 <= abstract_state[3] <= 1):
        return False
    if not (0 <= abstract_state[5] <= 1):
        return False

    # Closing rate, off-axis, altitude should be [-1, 1]
    if not (-1 <= abstract_state[1] <= 1):
        return False
    if not (-1 <= abstract_state[4] <= 1):
        return False
    if not (-1 <= abstract_state[6] <= 1):
        return False

    return True
```

---

## Option Manager

**File:** `hrl/option_manager.py`

```python
"""
Option switching logic and forced transitions.
"""
import numpy as np
from typing import Optional, Dict, Any

from .option_definitions import Option, FORCED_TRANSITION_THRESHOLDS


class OptionManager:
    """
    Manages option switching logic, including forced transitions.

    Forced transitions occur when environment state requires option change
    (e.g., radar lock lost, close range reached).
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        enable_forced: bool = True,
    ):
        """
        Args:
            thresholds: Override default forced transition thresholds
            enable_forced: Enable forced transitions (disable for testing)
        """
        self.thresholds = thresholds or FORCED_TRANSITION_THRESHOLDS
        self.enable_forced = enable_forced

        # Statistics tracking
        self.transition_counts = {
            'forced': 0,
            'selector': 0,
            'timeout': 0,
        }

    def get_forced_transition(
        self,
        current_option: Option,
        env_state: Dict[str, Any],
    ) -> Optional[Option]:
        """
        Check if environment state forces an option change.

        Args:
            current_option: Current active option
            env_state: Environment state dict with keys:
                - lock_quality: float [0, 1]
                - distance: float (meters)
                - fuel: float [0, 1]
                - closing_rate: float (m/s)

        Returns:
            Required option, or None if no forced transition
        """
        if not self.enable_forced:
            return None

        lock_quality = env_state.get('lock_quality', 0.0)
        distance = env_state.get('distance', float('inf'))
        fuel = env_state.get('fuel', 1.0)

        # Rule 1: Lost radar lock → SEARCH
        if current_option in [Option.TRACK, Option.TERMINAL]:
            if lock_quality < self.thresholds['radar_lock_quality_min']:
                self.transition_counts['forced'] += 1
                return Option.SEARCH

        # Rule 2: Acquired lock during search → TRACK
        if current_option == Option.SEARCH:
            if lock_quality >= self.thresholds['radar_lock_quality_search']:
                self.transition_counts['forced'] += 1
                return Option.TRACK

        # Rule 3: Close range → TERMINAL
        if current_option == Option.TRACK:
            if distance < self.thresholds['close_range_threshold']:
                if fuel >= self.thresholds['terminal_fuel_min']:
                    self.transition_counts['forced'] += 1
                    return Option.TERMINAL

        # Rule 4: Out of terminal range → TRACK
        if current_option == Option.TERMINAL:
            if distance > self.thresholds['miss_imminent_distance']:
                # Miss imminent, return to tracking
                self.transition_counts['forced'] += 1
                return Option.TRACK

        return None

    def should_timeout_option(
        self,
        current_option: Option,
        steps_in_option: int,
    ) -> bool:
        """
        Check if option has been active too long (timeout).

        Args:
            current_option: Current option
            steps_in_option: Steps since option started

        Returns:
            True if option should timeout
        """
        # Get expected duration
        from .option_definitions import get_expected_duration
        expected_duration = get_expected_duration(current_option)

        # Allow 2x expected duration before forcing timeout
        max_duration = expected_duration * 2

        if steps_in_option > max_duration:
            self.transition_counts['timeout'] += 1
            return True

        return False

    def get_statistics(self) -> Dict[str, int]:
        """Get transition statistics."""
        return self.transition_counts.copy()

    def reset_statistics(self):
        """Reset statistics counters."""
        self.transition_counts = {
            'forced': 0,
            'selector': 0,
            'timeout': 0,
        }
```

---

## Hierarchical Manager (Complete Implementation)

**File:** `hrl/manager.py`

```python
"""
Hierarchical Manager coordinates selector and specialists.
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
    lstm_states: Dict[Option, Any]  # Per-specialist LSTM states
    last_decision_step: int
    total_steps: int


class HierarchicalManager:
    """
    Coordinates selector and specialists during inference/training.
    """

    def __init__(
        self,
        selector_policy: SelectorPolicy,
        specialists: Dict[Option, SpecialistPolicy],
        decision_interval: int = 100,
        enable_forced_transitions: bool = True,
        default_option: Option = Option.SEARCH,
    ):
        """
        Args:
            selector_policy: High-level policy
            specialists: Dict mapping Option -> SpecialistPolicy
            decision_interval: Steps between high-level decisions
            enable_forced_transitions: Allow forced option changes
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
        """Reset to initial state."""
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
        full_obs: np.ndarray,
        env_state: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action using hierarchical policy.

        Args:
            full_obs: 26D or 104D observation
            env_state: Environment state for forced transitions
            deterministic: Use deterministic policies

        Returns:
            action: 6D continuous action
            info: Debug information
        """
        # 1. Check for forced transitions
        forced_option = None
        if self.enable_forced_transitions and env_state is not None:
            forced_option = self.option_manager.get_forced_transition(
                current_option=self.state.current_option,
                env_state=env_state,
            )

        # 2. Check if selector should decide
        option_switched = False
        if forced_option is not None:
            new_option = forced_option
            option_switched = True
            switch_reason = "forced"
        elif self.should_switch_option():
            # Selector decision
            abstract_state = abstract_observation(full_obs)
            new_option_idx = self.selector.predict(abstract_state, deterministic)
            new_option = Option(new_option_idx)
            option_switched = (new_option != self.state.current_option)
            switch_reason = "selector"
        else:
            new_option = self.state.current_option
            switch_reason = "continue"

        # 3. Handle option switching
        if option_switched:
            self._switch_option(new_option)

        # 4. Execute specialist
        active_specialist = self.specialists[self.state.current_option]

        if hasattr(active_specialist, 'predict_with_lstm'):
            action, new_lstm_state = active_specialist.predict_with_lstm(
                full_obs,
                self.state.lstm_states[self.state.current_option],
                deterministic,
            )
            self.state.lstm_states[self.state.current_option] = new_lstm_state
        else:
            action = active_specialist.predict(full_obs, deterministic)

        # 5. Update state
        self.state.steps_in_option += 1
        self.state.total_steps += 1

        # 6. Build info
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
        """Check if it's time for selector decision."""
        return (self.state.total_steps - self.state.last_decision_step) >= self.decision_interval

    def _switch_option(self, new_option: Option):
        """Handle option switching."""
        old_option = self.state.current_option

        # Reset old specialist LSTM
        if hasattr(self.specialists[old_option], 'reset_lstm'):
            self.specialists[old_option].reset_lstm()

        # Update state
        self.state.current_option = new_option
        self.state.steps_in_option = 0
        self.state.last_decision_step = self.state.total_steps

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'current_option': self.state.current_option.name,
            'total_steps': self.state.total_steps,
            'steps_in_current_option': self.state.steps_in_option,
            'transition_stats': self.option_manager.get_statistics(),
        }
```

---

## Training Script Example

**File:** `scripts/train_hrl_pretrain.py`

```python
"""
Pre-train HRL specialists.
"""
import os
import yaml
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

import sys
sys.path.append(str(Path(__file__).parent.parent))

from environment import InterceptEnvironment
from hrl.option_definitions import Option
from logger import UnifiedLogger


def create_search_scenario_env(base_config):
    """Create environment for search specialist pre-training."""
    config = base_config.copy()

    # Modify scenario for search training
    # Start without lock, wide search area
    config['interceptor_spawn'] = {
        'position': [[400, 400, 50], [600, 600, 200]],
        'velocity': [[0, 0, 0], [50, 50, 20]],
    }
    config['missile_spawn'] = {
        'position': [[-500, -500, 200], [500, 500, 500]],
        'velocity': [[50, 50, -20], [150, 150, -50]],
    }

    # Wide radar beam for easier acquisition
    config['radar'] = config.get('radar', {})
    config['radar']['beam_width'] = 90.0  # degrees

    return InterceptEnvironment(config)


def create_track_scenario_env(base_config):
    """Create environment for track specialist pre-training."""
    config = base_config.copy()

    # Start with radar lock, medium distance
    # This would require environment modifications to force initial lock
    # For now, use standard environment with modified spawn

    return InterceptEnvironment(config)


def create_terminal_scenario_env(base_config):
    """Create environment for terminal specialist pre-training."""
    config = base_config.copy()

    # Start close to target
    config['interceptor_spawn'] = {
        'position': [[800, 800, 100], [900, 900, 150]],
        'velocity': [[100, 100, 50], [150, 150, 80]],
    }
    config['missile_spawn'] = {
        'position': [[850, 850, 120], [950, 950, 180]],
        'velocity': [[50, 50, -20], [100, 100, -40]],
    }

    return InterceptEnvironment(config)


def train_specialist(
    specialist_name: str,
    config: dict,
    output_dir: str,
    total_timesteps: int = 1000000,
):
    """
    Train a single specialist.

    Args:
        specialist_name: 'search', 'track', or 'terminal'
        config: Training configuration
        output_dir: Directory to save checkpoints
        total_timesteps: Training duration
    """
    print(f"\n{'='*60}")
    print(f"Training {specialist_name.upper()} Specialist")
    print(f"{'='*60}\n")

    # Create specialized environment
    if specialist_name == 'search':
        env_fn = lambda: create_search_scenario_env(config)
    elif specialist_name == 'track':
        env_fn = lambda: create_track_scenario_env(config)
    elif specialist_name == 'terminal':
        env_fn = lambda: create_terminal_scenario_env(config)
    else:
        raise ValueError(f"Unknown specialist: {specialist_name}")

    # Create vectorized environment
    env = DummyVecEnv([env_fn])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env = VecFrameStack(env, n_stack=4)

    # Create logger
    logger = UnifiedLogger(
        run_name=f"{specialist_name}_specialist",
        config=config,
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        verbose=1,
        tensorboard_log=logger.log_dir,
        policy_kwargs={
            'net_arch': [512, 512, 256],
            'lstm_hidden_size': 256,
        },
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "model"))
    env.save(os.path.join(output_dir, "vec_normalize.pkl"))

    print(f"\n✓ {specialist_name} specialist training complete!")
    print(f"  Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/hrl/hrl_base.yaml')
    parser.add_argument('--specialist', type=str, choices=['search', 'track', 'terminal', 'all'])
    parser.add_argument('--steps', type=int, default=1000000)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Train specialists
    specialists = [args.specialist] if args.specialist != 'all' else ['search', 'track', 'terminal']

    for specialist in specialists:
        output_dir = f"checkpoints/hrl/specialists/{specialist}"
        train_specialist(specialist, config, output_dir, args.steps)


if __name__ == "__main__":
    main()
```

---

## Testing Example

**File:** `tests/test_hrl_manager.py`

```python
"""
Unit tests for HierarchicalManager.
"""
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from hrl.manager import HierarchicalManager, HRLState
from hrl.selector_policy import SelectorPolicy
from hrl.specialist_policies import SearchSpecialist, TrackSpecialist, TerminalSpecialist
from hrl.option_definitions import Option


@pytest.fixture
def mock_policies():
    """Create mock policies for testing."""
    # Mock selector
    selector = Mock(spec=SelectorPolicy)
    selector.predict = Mock(return_value=0)  # Always return SEARCH

    # Mock specialists
    search_specialist = Mock(spec=SearchSpecialist)
    search_specialist.predict = Mock(return_value=np.zeros(6, dtype=np.float32))

    track_specialist = Mock(spec=TrackSpecialist)
    track_specialist.predict = Mock(return_value=np.zeros(6, dtype=np.float32))

    terminal_specialist = Mock(spec=TerminalSpecialist)
    terminal_specialist.predict = Mock(return_value=np.zeros(6, dtype=np.float32))

    specialists = {
        Option.SEARCH: search_specialist,
        Option.TRACK: track_specialist,
        Option.TERMINAL: terminal_specialist,
    }

    return selector, specialists


def test_manager_initialization(mock_policies):
    """Test manager initializes correctly."""
    selector, specialists = mock_policies
    manager = HierarchicalManager(selector, specialists)

    assert manager.state.current_option == Option.SEARCH
    assert manager.state.steps_in_option == 0
    assert manager.state.total_steps == 0


def test_manager_reset(mock_policies):
    """Test manager reset."""
    selector, specialists = mock_policies
    manager = HierarchicalManager(selector, specialists)

    # Advance state
    obs = np.zeros(26)
    manager.select_action(obs)
    assert manager.state.total_steps == 1

    # Reset
    manager.reset()
    assert manager.state.total_steps == 0
    assert manager.state.current_option == Option.SEARCH


def test_manager_action_selection(mock_policies):
    """Test action selection returns valid output."""
    selector, specialists = mock_policies
    manager = HierarchicalManager(selector, specialists)

    obs = np.zeros(26)
    env_state = {'lock_quality': 0.5, 'distance': 1000.0, 'fuel': 0.8, 'closing_rate': 50.0}

    action, info = manager.select_action(obs, env_state)

    # Check action shape
    assert action.shape == (6,)
    assert action.dtype == np.float32

    # Check info dict
    assert 'option' in info
    assert 'option_index' in info
    assert 'steps_in_option' in info
    assert 'option_switched' in info


def test_manager_forced_transition_lock_acquired(mock_policies):
    """Test forced transition when lock acquired."""
    selector, specialists = mock_policies
    manager = HierarchicalManager(selector, specialists, enable_forced_transitions=True)

    # Start in SEARCH
    assert manager.state.current_option == Option.SEARCH

    # Acquire lock
    obs = np.zeros(26)
    env_state = {
        'lock_quality': 0.8,  # Above threshold
        'distance': 1000.0,
        'fuel': 0.8,
        'closing_rate': 50.0,
    }

    action, info = manager.select_action(obs, env_state)

    # Should force transition to TRACK
    assert manager.state.current_option == Option.TRACK
    assert info['forced_transition'] == True
    assert info['switch_reason'] == 'forced'


def test_manager_forced_transition_close_range(mock_policies):
    """Test forced transition when close range reached."""
    selector, specialists = mock_policies
    manager = HierarchicalManager(selector, specialists, enable_forced_transitions=True)

    # Start in TRACK
    manager.state.current_option = Option.TRACK

    # Get close
    obs = np.zeros(26)
    env_state = {
        'lock_quality': 0.9,
        'distance': 80.0,  # Below 100m threshold
        'fuel': 0.5,
        'closing_rate': 100.0,
    }

    action, info = manager.select_action(obs, env_state)

    # Should force transition to TERMINAL
    assert manager.state.current_option == Option.TERMINAL
    assert info['forced_transition'] == True


def test_manager_selector_decision_interval(mock_policies):
    """Test selector makes decision at correct interval."""
    selector, specialists = mock_policies
    manager = HierarchicalManager(
        selector,
        specialists,
        decision_interval=100,
        enable_forced_transitions=False,
    )

    obs = np.zeros(26)

    # First 99 steps: no selector decision
    for i in range(99):
        action, info = manager.select_action(obs)
        assert info['switch_reason'] == 'continue'

    # Step 100: selector decision
    action, info = manager.select_action(obs)
    # Selector should have been called
    selector.predict.assert_called()


def test_manager_statistics(mock_policies):
    """Test statistics tracking."""
    selector, specialists = mock_policies
    manager = HierarchicalManager(selector, specialists)

    obs = np.zeros(26)

    # Run for 10 steps
    for _ in range(10):
        manager.select_action(obs)

    stats = manager.get_statistics()

    assert stats['total_steps'] == 10
    assert stats['current_option'] == 'Search'
    assert 'transition_stats' in stats


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
```

---

## Usage Examples

### Example 1: Training Specialists

```bash
# Pre-train search specialist
python scripts/train_hrl_pretrain.py \
    --config configs/hrl/hrl_base.yaml \
    --specialist search \
    --steps 1000000

# Pre-train all specialists in sequence
python scripts/train_hrl_pretrain.py \
    --config configs/hrl/hrl_base.yaml \
    --specialist all \
    --steps 1000000
```

### Example 2: Using HierarchicalManager in Inference

```python
from hrl.manager import HierarchicalManager
from hrl.selector_policy import SelectorPolicy
from hrl.specialist_policies import SearchSpecialist, TrackSpecialist, TerminalSpecialist
from hrl.option_definitions import Option
from environment import InterceptEnvironment
import yaml

# Load config
with open('configs/hrl/hrl_base.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create environment
env = InterceptEnvironment(config)

# Load policies
selector = SelectorPolicy(obs_dim=7, n_options=3)
selector.load('checkpoints/hrl/selector/model')

specialists = {
    Option.SEARCH: SearchSpecialist(model_path='checkpoints/hrl/specialists/search/model'),
    Option.TRACK: TrackSpecialist(model_path='checkpoints/hrl/specialists/track/model'),
    Option.TERMINAL: TerminalSpecialist(model_path='checkpoints/hrl/specialists/terminal/model'),
}

# Create manager
manager = HierarchicalManager(
    selector_policy=selector,
    specialists=specialists,
    decision_interval=100,
)

# Run episode
obs, info = env.reset()
manager.reset()

done = False
total_reward = 0

while not done:
    # Extract env state for forced transitions
    from hrl.observation_abstraction import extract_env_state_for_transitions
    env_state = extract_env_state_for_transitions(obs)

    # Get action from HRL manager
    action, hrl_info = manager.select_action(obs, env_state, deterministic=True)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

    # Log option transitions
    if hrl_info['option_switched']:
        print(f"Step {manager.state.total_steps}: Switched to {hrl_info['option']} ({hrl_info['switch_reason']})")

print(f"Episode complete! Total reward: {total_reward:.2f}")
print(f"Final statistics: {manager.get_statistics()}")
```

---

## Summary

These code examples provide:

1. **Complete implementations** of key modules
2. **Testing patterns** for validation
3. **Usage examples** for training and inference
4. **Clear documentation** with type hints and docstrings

All code follows the design principles:
- Modularity: Each module has clear responsibilities
- Testability: Mock-friendly interfaces
- Backward compatibility: No changes to existing code
- Production-ready: Error handling, logging, type safety

**Next Steps:**
1. Review these examples
2. Start implementing Phase 1 (directory setup)
3. Implement modules incrementally with tests
4. Validate backward compatibility at each phase
