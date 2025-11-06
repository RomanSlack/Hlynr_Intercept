"""
Low-level specialist policies for each option.

Phase 2: Loads RecurrentPPO models with LSTM support.
"""
import numpy as np
from typing import Optional, Tuple, Any
from pathlib import Path

try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    RecurrentPPO = None
    HAS_RECURRENT_PPO = False

from .option_definitions import Option


class SpecialistPolicy:
    """
    Base class for low-level specialists using RecurrentPPO with LSTM.

    Phase 2: Loads trained RecurrentPPO models, manages LSTM state per option.
    Falls back to zero actions if no model provided (backward compatibility).
    """

    def __init__(
        self,
        option_type: Option,
        obs_dim: int = 104,  # 26D base * 4 frame stack
        action_dim: int = 6,  # 3D thrust + 3D angular
        model_path: Optional[str] = None,
    ):
        """
        Args:
            option_type: Which option this specialist handles
            obs_dim: Observation dimension (104 with frame stacking)
            action_dim: Action dimension (6D continuous)
            model_path: Path to pre-trained RecurrentPPO model (optional)
        """
        self.option_type = option_type
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model_path = model_path

        # LSTM state tracking (tuple of hidden and cell states)
        # Shape: (num_layers, batch_size, hidden_size)
        # For RecurrentPPO default: (1, 1, 256)
        self.lstm_state = None

        # Load model if path provided
        self.model = None
        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """
        Load RecurrentPPO model from disk.

        Args:
            model_path: Path to saved RecurrentPPO model (.zip)

        Raises:
            FileNotFoundError: If model file doesn't exist
            ImportError: If sb3_contrib not installed
            RuntimeError: If model loading fails
        """
        if not HAS_RECURRENT_PPO:
            raise ImportError(
                "sb3_contrib is required for RecurrentPPO. "
                "Install with: pip install sb3-contrib>=2.0.0"
            )

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Expected a trained RecurrentPPO model for {self.option_type.name}"
            )

        try:
            self.model = RecurrentPPO.load(model_path)
            print(f"Loaded RecurrentPPO model for {self.option_type.name} from {model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load RecurrentPPO model from {model_path}: {e}"
            )

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict action without explicit LSTM state (uses internal state).

        Args:
            obs: Full observation (104D stacked)
            deterministic: Use deterministic policy

        Returns:
            action: 6D continuous action

        Note:
            This method uses the internal self.lstm_state. For explicit state
            management, use predict_with_lstm() instead.
        """
        if self.model is None:
            # Stub behavior: return zeros (backward compatibility)
            return np.zeros(self.action_dim, dtype=np.float32)

        # RecurrentPPO.predict() interface:
        # action, lstm_state = model.predict(obs, state=lstm_state,
        #                                     episode_start=episode_start,
        #                                     deterministic=deterministic)

        # Reshape obs to (1, obs_dim) for batch processing
        obs_batch = obs.reshape(1, -1)

        # Episode start flag: True if lstm_state is None (triggers LSTM reset)
        episode_start = np.array([self.lstm_state is None])

        action, self.lstm_state = self.model.predict(
            obs_batch,
            state=self.lstm_state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

        # RecurrentPPO returns action as (1, action_dim), flatten to (action_dim,)
        return action.flatten()

    def predict_with_lstm(
        self,
        obs: np.ndarray,
        lstm_state: Optional[Tuple],
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        """
        Predict action with explicit LSTM state management.

        Args:
            obs: Full observation (104D stacked)
            lstm_state: Previous LSTM hidden state (or None to reset)
            deterministic: Use deterministic policy

        Returns:
            action: 6D continuous action
            new_lstm_state: Updated LSTM hidden state

        Note:
            This method does NOT modify self.lstm_state. It's stateless from
            the caller's perspective, useful for testing and explicit control.
        """
        if self.model is None:
            # Stub behavior: return zeros with None state
            return np.zeros(self.action_dim, dtype=np.float32), None

        # Reshape obs to (1, obs_dim) for batch processing
        obs_batch = obs.reshape(1, -1)

        # Episode start flag: True if lstm_state is None
        episode_start = np.array([lstm_state is None])

        action, new_lstm_state = self.model.predict(
            obs_batch,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

        # Flatten action from (1, action_dim) to (action_dim,)
        return action.flatten(), new_lstm_state

    def reset_lstm(self):
        """
        Reset LSTM hidden state (called on option switch or episode reset).

        Setting lstm_state to None causes the next predict() call to trigger
        an internal LSTM reset via episode_start=True.
        """
        self.lstm_state = None


class SearchSpecialist(SpecialistPolicy):
    """
    Specialist for wide-area scanning.

    Objective: Maximize scan coverage, acquire radar lock quickly.
    """

    def __init__(self, **kwargs):
        super().__init__(option_type=Option.SEARCH, **kwargs)


class TrackSpecialist(SpecialistPolicy):
    """
    Specialist for maintaining lock and approaching target.

    Objective: Maintain radar lock, reduce distance, optimize intercept geometry.
    """

    def __init__(self, **kwargs):
        super().__init__(option_type=Option.TRACK, **kwargs)


class TerminalSpecialist(SpecialistPolicy):
    """
    Specialist for final intercept guidance.

    Objective: Minimize miss distance with high precision.
    """

    def __init__(self, **kwargs):
        super().__init__(option_type=Option.TERMINAL, **kwargs)
