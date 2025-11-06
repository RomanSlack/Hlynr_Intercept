"""
High-level selector policy that chooses options.

Phase 1 stub: Returns random or fixed options (no training logic yet).
"""
import numpy as np
from typing import Optional

from .option_definitions import Option


class SelectorPolicy:
    """
    High-level policy that selects options (search/track/terminal).

    Phase 1 stub: Returns fixed SEARCH option or random selection.
    """

    def __init__(
        self,
        obs_dim: int = 7,  # Abstract state dimension
        n_options: int = 3,
        model_path: Optional[str] = None,
        mode: str = "fixed",  # "fixed" or "random"
    ):
        """
        Args:
            obs_dim: Dimension of abstract observation space
            n_options: Number of discrete options (default 3)
            model_path: Path to pre-trained model (unused in stub)
            mode: "fixed" returns SEARCH always, "random" returns random option
        """
        self.obs_dim = obs_dim
        self.n_options = n_options
        self.model_path = model_path
        self.mode = mode

    def predict(
        self,
        abstract_obs: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """
        Select option based on abstract observation (stub).

        Args:
            abstract_obs: Abstract state vector (7D)
            deterministic: Use deterministic policy

        Returns:
            option_index: {0, 1, 2} corresponding to {SEARCH, TRACK, TERMINAL}
        """
        if self.mode == "fixed":
            # Always return SEARCH for safety
            return int(Option.SEARCH)
        elif self.mode == "random":
            # Random option selection
            return np.random.randint(0, self.n_options)
        else:
            return int(Option.SEARCH)
