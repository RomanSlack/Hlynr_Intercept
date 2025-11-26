"""
Gymnasium wrappers for hierarchical RL.

CRITICAL: This wrapper manages observation persistence for HRL manager.
The manager needs the current observation to make decisions, but Gymnasium
doesn't provide it during step(). We store _last_obs to solve this.
"""
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional

from .manager import HierarchicalManager
from .observation_abstraction import extract_env_state_for_transitions, abstract_observation
from .selector_policy import Option
from .reward_decomposition import compute_tactical_reward


class HRLActionWrapper(gym.Wrapper):
    """
    Wrapper to use HierarchicalManager for action selection.

    Supports two modes:
    1. Autonomous mode (selector=internal): Manager uses its own selector policy
    2. Training mode (selector=None): Accepts discrete option indices from external selector
    """

    def __init__(
        self,
        env: gym.Env,
        manager: HierarchicalManager,
        return_hrl_info: bool = True,
        training_selector: bool = False,
    ):
        """
        Args:
            env: Base InterceptEnvironment
            manager: HierarchicalManager instance
            return_hrl_info: Include HRL info in step() info dict
            training_selector: If True, wrapper expects discrete option indices as actions.
                             If False, manager uses its own selector (autonomous mode).
        """
        super().__init__(env)
        self.manager = manager
        self.return_hrl_info = return_hrl_info
        self.training_selector = training_selector
        self._last_obs = None

        # Track transition statistics for selector training mode
        if training_selector:
            self._num_switches = 0
            self._forced_transitions = 0
            self._selector_decisions = 0

        # Override action space for selector training mode
        if training_selector:
            self.action_space = gym.spaces.Discrete(3)  # SEARCH, TRACK, TERMINAL

    def reset(self, **kwargs):
        """Reset environment and HRL manager."""
        obs, info = self.env.reset(**kwargs)
        self.manager.reset()
        self._last_obs = obs

        # Reset training statistics
        if self.training_selector:
            self._num_switches = 0
            self._forced_transitions = 0
            self._selector_decisions = 0

        return obs, info

    def step(self, action=None):
        """
        Step with hierarchical action selection.

        Args:
            action: If training_selector=True, expects discrete option index (0, 1, or 2).
                   If training_selector=False, action is ignored (manager selects internally).

        Returns:
            Standard Gymnasium step output with HRL info

        Note:
            Uses stored _last_obs for manager decision. After env step,
            updates _last_obs with next_obs for the next iteration.
        """
        # CRITICAL: Use stored observation from previous step
        # This is the current observation needed for manager decision
        if self._last_obs is None:
            raise RuntimeError(
                "HRLActionWrapper.step() called before reset(). "
                "Must call reset() first to initialize observation."
            )

        obs = self._last_obs

        # Handle frame-stacked observations: flatten (4, 26) → (104,)
        # FrameStackObservation returns shape (stack_size, obs_dim) which needs flattening
        # for specialists that expect (104,) flattened observations
        if obs.ndim > 1:
            obs = obs.flatten()

        # Get environment state for forced transitions
        env_state = extract_env_state_for_transitions(obs, env_info=None)

        if self.training_selector:
            # TRAINING MODE: Use external selector's discrete action
            if action is None:
                raise ValueError(
                    "training_selector=True but no action provided. "
                    "Expected discrete option index (0, 1, or 2)."
                )

            # Convert discrete index to Option enum
            option_index = int(action)
            if option_index not in [0, 1, 2]:
                raise ValueError(f"Invalid option index: {option_index}. Expected 0, 1, or 2.")

            selected_option = Option(option_index)

            # Check for forced transitions (override selector choice if necessary)
            forced_option = None
            if self.manager.enable_forced_transitions:
                forced_option = self.manager.option_manager.get_forced_transition(
                    current_option=self.manager.state.current_option,
                    env_state=env_state,
                )

            if forced_option is not None:
                selected_option = forced_option
                forced_transition = True
            else:
                forced_transition = False

            # Update manager state and track statistics
            if self.manager.state.current_option != selected_option:
                self._num_switches += 1
                if forced_transition:
                    self._forced_transitions += 1
                else:
                    self._selector_decisions += 1

            self.manager.state.current_option = selected_option
            self.manager.state.steps_in_option = 0

            # Get action from the selected specialist
            specialist = self.manager.specialists[selected_option]
            specialist_action = specialist.predict(obs, deterministic=True)

            # Create HRL info
            hrl_info = {
                'hrl/option': selected_option.name,
                'hrl/option_index': selected_option.value,
                'hrl/forced_transition': forced_transition,
                'hrl/num_switches': self._num_switches,
                'hrl/forced_transitions': self._forced_transitions,
                'hrl/selector_decisions': self._selector_decisions,
            }

            final_action = specialist_action
        else:
            # AUTONOMOUS MODE: Manager uses its own selector
            final_action, hrl_info = self.manager.select_action(obs, env_state)

        # Execute in base environment
        next_obs, reward, terminated, truncated, info = self.env.step(final_action)

        # CRITICAL: Update stored observation for next step
        self._last_obs = next_obs

        # Add HRL info
        if self.return_hrl_info:
            info.update(hrl_info)

        return next_obs, reward, terminated, truncated, info


class AbstractObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert 26D/104D observations to abstract 7D state for selector training.

    This wrapper is used when training the selector policy, which operates on
    abstract strategic information rather than raw sensor data.
    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env: Base environment (should output 26D base or 104D frame-stacked obs)
        """
        super().__init__(env)

        # Override observation space to abstract state dimension
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),  # Abstract state dimension
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert full observation to abstract state."""
        return abstract_observation(obs)


class SpecialistRewardWrapper(gym.Wrapper):
    """
    Wrapper that applies specialist-specific tactical rewards during pre-training.

    This is CRITICAL for specialist learning - each specialist needs rewards
    tailored to their specific objective:
    - SEARCH: Lock acquisition, scan coverage, fuel efficiency
    - TRACK: Lock maintenance, distance reduction, closing rate
    - TERMINAL: Proximity (exponential), closing rate, thrust usage

    Without this wrapper, specialists train on the base environment reward,
    which doesn't differentiate between search/track/terminal objectives.
    """

    def __init__(
        self,
        env: gym.Env,
        specialist_type: str,
        blend_with_base: float = 0.3,
    ):
        """
        Args:
            env: Base InterceptEnvironment
            specialist_type: 'search', 'track', or 'terminal'
            blend_with_base: How much of base reward to include (0.0 = only tactical,
                           1.0 = only base). Default 0.3 means 70% tactical + 30% base.
        """
        super().__init__(env)

        # Map string to Option enum
        type_map = {
            'search': Option.SEARCH,
            'track': Option.TRACK,
            'terminal': Option.TERMINAL,
        }
        if specialist_type.lower() not in type_map:
            raise ValueError(f"Unknown specialist type: {specialist_type}. "
                           f"Expected one of: {list(type_map.keys())}")

        self.specialist_option = type_map[specialist_type.lower()]
        self.specialist_type = specialist_type.lower()
        self.blend_with_base = blend_with_base

        # Track previous state for reward computation
        self._prev_env_state = None
        self._last_action = None

    def reset(self, **kwargs):
        """Reset environment and state tracking."""
        obs, info = self.env.reset(**kwargs)

        # Initialize previous state from first observation
        self._prev_env_state = self._extract_env_state(obs, info)
        self._last_action = None

        return obs, info

    def step(self, action):
        """Step with specialist-specific reward."""
        # Store action for reward computation
        self._last_action = action

        # Execute in base environment
        next_obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Extract current state for reward computation
        curr_env_state = self._extract_env_state(next_obs, info)

        # Compute tactical reward for this specialist
        tactical_reward = compute_tactical_reward(
            env_state=self._prev_env_state,
            action=action,
            option=self.specialist_option,
            next_env_state=curr_env_state,
            episode_done=(terminated or truncated),
        )

        # Blend tactical and base rewards
        # Higher tactical weight helps specialists learn their specific objectives
        blended_reward = (
            (1.0 - self.blend_with_base) * tactical_reward +
            self.blend_with_base * base_reward
        )

        # Update previous state for next step
        self._prev_env_state = curr_env_state

        # Add reward breakdown to info for debugging
        info['reward/tactical'] = tactical_reward
        info['reward/base'] = base_reward
        info['reward/blended'] = blended_reward

        return next_obs, blended_reward, terminated, truncated, info

    def _extract_env_state(self, obs: np.ndarray, info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract environment state features needed for reward computation.

        Uses info dict when available (ground truth), falls back to observation
        if info not provided.
        """
        # Handle frame-stacked observations: use latest frame
        if obs.ndim > 1:
            # Shape is (stack_size, obs_dim), use last frame
            latest_obs = obs[-1]
        else:
            # Already flattened or single frame
            if len(obs) > 26:
                # Flattened frame stack, extract last 26 elements
                latest_obs = obs[-26:]
            else:
                latest_obs = obs

        # Extract from info dict if available (ground truth)
        if info is not None:
            distance = info.get('distance', None)
            if distance is None and 'interceptor_pos' in info and 'missile_pos' in info:
                int_pos = np.array(info['interceptor_pos'])
                mis_pos = np.array(info['missile_pos'])
                distance = float(np.linalg.norm(int_pos - mis_pos))

            return {
                'lock_quality': info.get('radar_quality', latest_obs[14] if len(latest_obs) > 14 else 0.0),
                'distance': distance if distance is not None else float(np.linalg.norm(latest_obs[0:3]) * 10000.0),
                'fuel': info.get('fuel_remaining', latest_obs[12] if len(latest_obs) > 12 else 1.0),
                'closing_rate': latest_obs[15] * 1000.0 if len(latest_obs) > 15 else 0.0,  # Unnormalize
            }

        # Fallback: extract from observation (normalized values)
        return {
            'lock_quality': float(latest_obs[14]) if len(latest_obs) > 14 else 0.0,
            'distance': float(np.linalg.norm(latest_obs[0:3]) * 10000.0),  # Unnormalize position
            'fuel': float(latest_obs[12]) if len(latest_obs) > 12 else 1.0,
            'closing_rate': float(latest_obs[15] * 1000.0) if len(latest_obs) > 15 else 0.0,  # Unnormalize
        }
