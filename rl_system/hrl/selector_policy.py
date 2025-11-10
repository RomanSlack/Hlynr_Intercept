"""
High-level selector policy that chooses options.

Phase 2: Supports model-based selection, rule-based fallback, and stub modes.
"""
import os
import numpy as np
from typing import Optional
from pathlib import Path

try:
    from stable_baselines3 import PPO  # Discrete action space
    HAS_PPO = True
except ImportError:
    PPO = None
    HAS_PPO = False

from .option_definitions import Option


class SelectorPolicy:
    """
    High-level policy that selects options (search/track/terminal).

    Modes:
        - "fixed": Always returns SEARCH (safe default)
        - "random": Uniform random selection (debugging)
        - "rules": Physically-motivated decision rules (intelligent fallback)
        - "model": Trained PPO model (requires model_path)

    The "rules" mode is useful for:
        - Initial testing before selector training
        - Fallback when model fails to load
        - Debugging specialist performance in isolation
    """

    def __init__(self, obs_dim: int = 7, n_options: int = 3,
                 model_path: str | None = None, mode: str = "rules"):
        self.obs_dim = obs_dim
        self.n_options = n_options
        self.model_path = model_path
        self.mode = mode
        self.model = None

        if self.mode == "model":
            if self.model_path is None:
                # REQUIRED by test_model_mode_requires_model_path
                raise ValueError("model_path must be provided when mode='model'.")
            # Note: Path validation moved to _load_model() for smart resolution
            # (handles directories, .zip files, and paths without extension)
            self._load_model(self.model_path)

    def _load_model(self, model_path: str):
        """
        Load PPO model for discrete action space.
        Handles both directory and file paths (auto-appends model.zip if needed).

        Args:
            model_path: Path to selector model (directory or .zip file)

        Raises:
            FileNotFoundError: If model not found
            ImportError: If stable_baselines3 not installed
            RuntimeError: If model loading fails
        """
        if not HAS_PPO:
            raise ImportError(
                "stable_baselines3 is required for selector policy. "
                "Install with: pip install stable-baselines3>=2.0.0"
            )

        model_path_obj = Path(model_path)

        # DEBUG: Print for troubleshooting
        print(f"[DEBUG] _load_model called with: {repr(model_path)}")
        print(f"[DEBUG] Is directory: {model_path_obj.is_dir()}")
        print(f"[DEBUG] Exists: {model_path_obj.exists()}")

        # Smart path resolution (matches specialist training script pattern)
        if model_path_obj.is_dir():
            # Directory provided - look for model.zip inside
            if (model_path_obj / "model.zip").exists():
                model_file = model_path_obj / "model.zip"
            else:
                raise FileNotFoundError(
                    f"Directory provided but model.zip not found: {model_path}\n"
                    f"Expected {model_path}/model.zip"
                )
        elif model_path_obj.suffix == ".zip":
            # Explicit .zip file provided
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model_file = model_path_obj
        else:
            # Neither directory nor .zip - try appending .zip
            if Path(f"{model_path}.zip").exists():
                model_file = Path(f"{model_path}.zip")
            else:
                raise FileNotFoundError(
                    f"Model not found at: {model_path}\n"
                    f"Tried: {model_path}, {model_path}.zip"
                )

        try:
            self.model = PPO.load(str(model_file))
            print(f"Loaded selector PPO model from {model_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to load selector model from {model_file}: {e}")

    def predict(
        self,
        abstract_obs: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """
        Select option based on abstract observation.

        Args:
            abstract_obs: Abstract state vector (7D):
                [0] distance_norm [0-1]
                [1] closing_rate_norm [-1, 1]
                [2] lock_quality [0-1]
                [3] fuel [0-1]
                [4] off_axis_norm [-1, 1]
                [5] tti_norm [0-1]
                [6] altitude_norm [-1, 1]
            deterministic: Use deterministic policy

        Returns:
            option_index: {0, 1, 2} corresponding to {SEARCH, TRACK, TERMINAL}
        """
        if self.mode == "fixed":
            # Always return SEARCH for safety
            return int(Option.SEARCH)

        elif self.mode == "random":
            # Uniform random selection (for testing)
            return int(np.random.randint(0, self.n_options))

        elif self.mode == "rules":
            # Physically-motivated decision rules
            return self._rule_based_selection(abstract_obs)

        elif self.mode == "model":
            if self.model is None:
                # Fallback if someone toggled mode to 'model' but didn't set a model
                return self._rule_based_selection(abstract_obs)

            # Batch to (1, 7) for SB3-style predict
            obs = np.asarray(abstract_obs, dtype=np.float32).reshape(1, -1)

            action, _ = self.model.predict(obs, deterministic=deterministic)

            # Robust scalar extraction: handles 1, array([1]), [[1]], etc.
            action_idx = int(np.asarray(action).reshape(-1)[0])
            return action_idx

        else:
            # Unknown mode, fall back to SEARCH
            print(f"Warning: Unknown selector mode '{self.mode}', defaulting to SEARCH")
            return int(Option.SEARCH)

    def _rule_based_selection(self, abstract_obs: np.ndarray) -> int:
        """
        Physically-motivated rule-based option selection.

        Rules (in priority order):
            1. No radar lock (lock_quality < 0.3) → SEARCH
            2. Close range (distance < 100m) + good lock → TERMINAL
            3. Good lock (lock_quality ≥ 0.3) → TRACK
            4. Default → SEARCH

        Args:
            abstract_obs: 7D abstract state

        Returns:
            option_index: {0, 1, 2}
        """
        # Extract key features from abstract state
        distance_norm = abstract_obs[0]  # [0-1], normalized by 5000m
        lock_quality = abstract_obs[2]   # [0-1]
        fuel = abstract_obs[3]           # [0-1]

        # Convert normalized distance back to meters for threshold checks
        distance_meters = distance_norm * 5000.0

        # Rule 1: No lock → SEARCH
        if lock_quality < 0.3:
            return int(Option.SEARCH)

        # Rule 2: Close range + good lock + sufficient fuel → TERMINAL
        if distance_meters < 100.0 and lock_quality > 0.7 and fuel > 0.1:
            return int(Option.TERMINAL)

        # Rule 3: Good lock but not close enough → TRACK
        if lock_quality >= 0.3:
            return int(Option.TRACK)

        # Rule 4: Default fallback → SEARCH
        return int(Option.SEARCH)
