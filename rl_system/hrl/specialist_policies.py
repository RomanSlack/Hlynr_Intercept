"""
Low-level specialist policies for each option.

Phase 1 stub: Returns zero actions (no training logic yet).
"""
import numpy as np
from typing import Optional, Tuple, Any

from .option_definitions import Option


class SpecialistPolicy:
    """
    Base class for low-level specialists (stub implementation).

    Phase 1: Returns zero actions. Will be replaced with PPO policies in Phase 2.
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
            model_path: Path to pre-trained model (unused in stub)
        """
        self.option_type = option_type
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model_path = model_path

        # LSTM state tracking (stub)
        self.lstm_state = None

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict action (stub: returns zeros).

        Args:
            obs: Full observation (104D stacked)
            deterministic: Use deterministic policy

        Returns:
            action: 6D continuous action (zeros for now)
        """
        return np.zeros(self.action_dim, dtype=np.float32)

    def predict_with_lstm(
        self,
        obs: np.ndarray,
        lstm_state: Optional[Tuple],
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        """
        Predict action with explicit LSTM state management (stub).

        Args:
            obs: Full observation (104D stacked)
            lstm_state: Previous LSTM hidden state (or None)
            deterministic: Use deterministic policy

        Returns:
            action: 6D continuous action (zeros for now)
            new_lstm_state: Updated LSTM hidden state (None for stub)
        """
        return np.zeros(self.action_dim, dtype=np.float32), None

    def reset_lstm(self):
        """Reset LSTM hidden state (called on option switch)."""
        self.lstm_state = None


class SearchSpecialist(SpecialistPolicy):
    """
    Specialist for wide-area scanning (stub).

    Objective: Maximize scan coverage, acquire radar lock quickly.
    """

    def __init__(self, **kwargs):
        super().__init__(option_type=Option.SEARCH, **kwargs)


class TrackSpecialist(SpecialistPolicy):
    """
    Specialist for maintaining lock and approaching target (stub).

    Objective: Maintain radar lock, reduce distance, optimize intercept geometry.
    """

    def __init__(self, **kwargs):
        super().__init__(option_type=Option.TRACK, **kwargs)


class TerminalSpecialist(SpecialistPolicy):
    """
    Specialist for final intercept guidance (stub).

    Objective: Minimize miss distance with high precision.
    """

    def __init__(self, **kwargs):
        super().__init__(option_type=Option.TERMINAL, **kwargs)
