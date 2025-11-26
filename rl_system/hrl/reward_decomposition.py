"""
Hierarchical reward decomposition: strategic (selector) + tactical (specialists).

This module implements the two-level reward structure:
- Strategic rewards: Long-term mission success (intercept, fuel efficiency)
- Tactical rewards: Option-specific immediate objectives (lock acquisition, tracking, terminal guidance)

Design principle: Dense tactical rewards accelerate specialist learning while sparse strategic
rewards guide high-level option selection.
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
        env_state: Current state features {'lock_quality', 'distance', 'fuel', 'closing_rate'}
        option: Selected option
        next_env_state: Next state features
        episode_done: Whether episode terminated
        episode_info: Episode termination info {'intercept_success', 'reason', ...}

    Returns:
        Strategic reward signal

    Reward Structure:
        Terminal:
            +1000: Intercept success
            +100 * fuel_remaining: Efficiency bonus
            -200: Out of fuel
            -100: Timeout
        Shaping (every step):
            -0.01 * distance: Encourage progress toward target
            +0.1 * closing_rate: Reward approach velocity
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
        action: 6D continuous action taken [thrust_x, thrust_y, thrust_z, gimbal_pitch, gimbal_yaw, gimbal_roll]
        option: Active option
        next_env_state: Next state features
        episode_done: Whether episode terminated

    Returns:
        Tactical reward signal
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
    """
    Reward for search specialist: maximize scan coverage, acquire lock.

    Objectives:
        - Acquire radar lock quickly
        - Cover large search areas efficiently
        - Conserve fuel during search phase

    Args:
        env_state: Current state features
        action: 6D continuous action
        next_env_state: Next state features
        episode_done: Whether episode terminated

    Returns:
        Search-specific reward
    """
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
    angular_action = action[3:6]  # Angular components [gimbal_pitch, gimbal_yaw, gimbal_roll]
    angular_magnitude = np.linalg.norm(angular_action)
    reward += angular_magnitude * 0.5  # Small bonus for exploration

    # Penalize fuel waste during search
    fuel_before = env_state.get('fuel', 1.0)
    fuel_after = next_env_state.get('fuel', 1.0)
    fuel_usage = fuel_before - fuel_after
    reward -= fuel_usage * 5.0

    return reward


def compute_track_reward(
    env_state: Dict[str, Any],
    action: np.ndarray,
    next_env_state: Dict[str, Any],
    episode_done: bool,
) -> float:
    """
    Reward for track specialist: maintain lock, close distance.

    Objectives:
        - Maintain radar lock quality above threshold
        - Reduce distance to target
        - Maximize closing rate (approach speed)
        - Smooth tracking (minimize jerky movements)

    Args:
        env_state: Current state features
        action: 6D continuous action
        next_env_state: Next state features
        episode_done: Whether episode terminated

    Returns:
        Track-specific reward
    """
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
    if distance_before < float('inf') and distance_after < float('inf'):
        distance_reduction = distance_before - distance_after
        reward += distance_reduction * 1.0

    # Reward positive closing rate
    closing_rate = next_env_state.get('closing_rate', 0.0)
    if closing_rate > 0:
        reward += closing_rate * 0.5

    # Penalize excessive angular action (should be tracking smoothly)
    angular_action = action[3:6]  # Angular components
    angular_magnitude = np.linalg.norm(angular_action)
    reward -= angular_magnitude * 0.2  # Penalize jerky movements

    return reward


def compute_terminal_reward(
    env_state: Dict[str, Any],
    action: np.ndarray,
    next_env_state: Dict[str, Any],
    episode_done: bool,
) -> float:
    """
    Reward for terminal specialist: minimize miss distance.

    Objectives:
        - Get extremely close to target (exponential proximity reward)
        - Maximize closing rate for impact
        - Use maximum available thrust

    Args:
        env_state: Current state features
        action: 6D continuous action
        next_env_state: Next state features
        episode_done: Whether episode terminated

    Returns:
        Terminal-specific reward
    """
    reward = 0.0

    # Primary objective: minimize miss distance
    distance = next_env_state.get('distance', float('inf'))
    distance_before = env_state.get('distance', float('inf'))

    if distance < float('inf'):
        # Multi-scale exponential reward for getting very close
        # Scale 1: Coarse guidance (effective 0-100m)
        reward += np.exp(-distance / 50.0) * 5.0

        # Scale 2: Fine guidance (effective 0-20m)
        reward += np.exp(-distance / 10.0) * 10.0

        # Scale 3: Precision guidance (effective 0-5m) - big bonus for sub-5m
        reward += np.exp(-distance / 2.0) * 20.0

        # Milestone bonuses for crossing distance thresholds
        if distance_before < float('inf'):
            if distance_before >= 50.0 and distance < 50.0:
                reward += 10.0  # Crossed 50m threshold
            if distance_before >= 20.0 and distance < 20.0:
                reward += 25.0  # Crossed 20m threshold
            if distance_before >= 10.0 and distance < 10.0:
                reward += 50.0  # Crossed 10m threshold
            if distance_before >= 5.0 and distance < 5.0:
                reward += 100.0  # Crossed 5m threshold - realistic intercept!

        # Strong penalty for increasing distance in terminal phase
        if distance_before < float('inf') and distance > distance_before:
            reward -= (distance - distance_before) * 10.0  # Increased penalty

    # Reward high closing rate (impact speed)
    closing_rate = next_env_state.get('closing_rate', 0.0)
    reward += closing_rate * 1.5  # Increased weight

    # Encourage maximum thrust in terminal phase
    thrust_action = action[0:3]  # Thrust components [thrust_x, thrust_y, thrust_z]
    thrust_magnitude = np.linalg.norm(thrust_action)
    reward += thrust_magnitude * 0.5

    return reward


def compute_combined_reward(
    env_state: Dict[str, Any],
    action: np.ndarray,
    option: Option,
    next_env_state: Dict[str, Any],
    episode_done: bool,
    episode_info: Dict[str, Any],
    strategic_weight: float = 0.5,
    tactical_weight: float = 0.5,
) -> float:
    """
    Combine strategic and tactical rewards with configurable weights.

    This is useful during joint training phases where both selector and specialists
    are being optimized simultaneously.

    Args:
        env_state: Current state features
        action: 6D continuous action
        option: Active option
        next_env_state: Next state features
        episode_done: Whether episode terminated
        episode_info: Episode termination info
        strategic_weight: Weight for strategic reward (default 0.5)
        tactical_weight: Weight for tactical reward (default 0.5)

    Returns:
        Weighted combination of strategic and tactical rewards
    """
    strategic = compute_strategic_reward(env_state, option, next_env_state, episode_done, episode_info)
    tactical = compute_tactical_reward(env_state, action, option, next_env_state, episode_done)

    return strategic_weight * strategic + tactical_weight * tactical
