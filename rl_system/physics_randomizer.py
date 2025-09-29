"""
Domain randomization framework for robust physics-based training.

This module provides systematic randomization of physics parameters
to improve sim-to-real transfer and training robustness.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class RandomizationParameters:
    """Configuration for physics parameter randomization."""
    # Atmospheric parameters
    air_density_variation: float = 0.1  # ±10% variation
    temperature_variation: float = 0.05  # ±5% variation

    # Drag coefficient randomization
    drag_coefficient_variation: float = 0.2  # ±20% variation
    mach_curve_variation: float = 0.15  # ±15% variation

    # Sensor delays and noise
    sensor_delay_variation: float = 0.5  # ±50% variation
    radar_noise_variation: float = 0.3  # ±30% variation
    radar_quality_variation: float = 0.1  # ±10% variation

    # Thrust and propulsion
    thrust_response_variation: float = 0.3  # ±30% variation
    fuel_consumption_variation: float = 0.2  # ±20% variation

    # Wind and environmental
    wind_velocity_variation: float = 0.3  # ±30% variation
    turbulence_variation: float = 0.4  # ±40% variation

    # Vehicle mass and properties
    mass_variation: float = 0.1  # ±10% variation
    moment_of_inertia_variation: float = 0.15  # ±15% variation


@dataclass
class RandomizedValues:
    """Container for randomized physics values for one episode."""
    # Atmospheric
    air_density_multiplier: float = 1.0
    temperature_offset: float = 0.0

    # Aerodynamics
    drag_coefficient_multiplier: float = 1.0
    mach_effects_multiplier: float = 1.0

    # Sensors
    sensor_delay_multiplier: float = 1.0
    radar_noise_multiplier: float = 1.0
    radar_quality_multiplier: float = 1.0

    # Propulsion
    thrust_response_multiplier: float = 1.0
    fuel_consumption_multiplier: float = 1.0

    # Environment
    wind_velocity_multiplier: float = 1.0
    turbulence_multiplier: float = 1.0

    # Vehicle properties
    interceptor_mass_multiplier: float = 1.0
    missile_mass_multiplier: float = 1.0

    # Episode metadata
    episode_seed: Optional[int] = None
    episode_id: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class PhysicsRandomizer:
    """
    Domain randomization framework for physics parameters.

    Provides systematic randomization of physics parameters to improve
    training robustness and sim-to-real transfer.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 randomization_params: Optional[RandomizationParameters] = None):
        """
        Initialize physics randomizer.

        Args:
            config: Domain randomization configuration
            randomization_params: Explicit randomization parameters
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)

        # Randomization parameters
        if randomization_params:
            self.params = randomization_params
        else:
            # Load from config
            self.params = RandomizationParameters()
            if 'drag_coefficient_variation' in self.config:
                self.params.drag_coefficient_variation = self.config['drag_coefficient_variation']
            if 'air_density_variation' in self.config:
                self.params.air_density_variation = self.config['air_density_variation']
            if 'sensor_delay_variation' in self.config:
                self.params.sensor_delay_variation = self.config['sensor_delay_variation']
            if 'thrust_response_variation' in self.config:
                self.params.thrust_response_variation = self.config['thrust_response_variation']
            if 'wind_variation' in self.config:
                self.params.wind_velocity_variation = self.config['wind_variation']

        # Randomization settings
        self.randomize_per_episode = self.config.get('randomize_per_episode', True)
        self.randomize_per_reset = self.config.get('randomize_per_reset', True)

        # Logging
        self.log_randomization = self.config.get('log_randomization', True)
        self.randomization_history: List[RandomizedValues] = []

        # Random number generator
        self.rng = np.random.default_rng()

        # Current randomized values
        self.current_values = RandomizedValues()

    def seed(self, seed: int):
        """Set random seed for reproducible randomization."""
        self.rng = np.random.default_rng(seed)

    def randomize_for_episode(self, episode_id: Optional[str] = None,
                             episode_seed: Optional[int] = None) -> RandomizedValues:
        """
        Generate new randomized values for an episode.

        Args:
            episode_id: Unique identifier for the episode
            episode_seed: Seed for this episode's randomization

        Returns:
            RandomizedValues object with new parameters
        """
        if not self.enabled:
            return RandomizedValues()

        # Set episode seed if provided
        if episode_seed is not None:
            episode_rng = np.random.default_rng(episode_seed)
        else:
            episode_rng = self.rng

        # Generate randomized values
        values = RandomizedValues(
            episode_seed=episode_seed,
            episode_id=episode_id,
            timestamp=np.random.default_rng().random()  # Use system time
        )

        # Atmospheric randomization
        values.air_density_multiplier = self._randomize_multiplier(
            episode_rng, self.params.air_density_variation
        )
        values.temperature_offset = episode_rng.normal(
            0, self.params.temperature_variation * 20.0  # ±20K typical variation
        )

        # Aerodynamic randomization
        values.drag_coefficient_multiplier = self._randomize_multiplier(
            episode_rng, self.params.drag_coefficient_variation
        )
        values.mach_effects_multiplier = self._randomize_multiplier(
            episode_rng, self.params.mach_curve_variation
        )

        # Sensor randomization
        values.sensor_delay_multiplier = self._randomize_multiplier(
            episode_rng, self.params.sensor_delay_variation
        )
        values.radar_noise_multiplier = self._randomize_multiplier(
            episode_rng, self.params.radar_noise_variation
        )
        values.radar_quality_multiplier = self._randomize_multiplier(
            episode_rng, self.params.radar_quality_variation, min_val=0.5, max_val=1.0
        )

        # Propulsion randomization
        values.thrust_response_multiplier = self._randomize_multiplier(
            episode_rng, self.params.thrust_response_variation, min_val=0.5, max_val=2.0
        )
        values.fuel_consumption_multiplier = self._randomize_multiplier(
            episode_rng, self.params.fuel_consumption_variation
        )

        # Environmental randomization
        values.wind_velocity_multiplier = self._randomize_multiplier(
            episode_rng, self.params.wind_velocity_variation
        )
        values.turbulence_multiplier = self._randomize_multiplier(
            episode_rng, self.params.turbulence_variation
        )

        # Vehicle mass randomization
        values.interceptor_mass_multiplier = self._randomize_multiplier(
            episode_rng, self.params.mass_variation
        )
        values.missile_mass_multiplier = self._randomize_multiplier(
            episode_rng, self.params.mass_variation
        )

        # Store and return
        self.current_values = values
        if self.log_randomization:
            self.randomization_history.append(values)

        return values

    def _randomize_multiplier(self, rng: np.random.Generator, variation: float,
                             min_val: float = 0.1, max_val: float = 3.0) -> float:
        """
        Generate a randomized multiplier within specified bounds.

        Args:
            rng: Random number generator
            variation: Variation fraction (e.g., 0.2 for ±20%)
            min_val: Minimum allowed multiplier
            max_val: Maximum allowed multiplier

        Returns:
            Random multiplier centered around 1.0
        """
        # Generate normal distribution around 1.0
        multiplier = rng.normal(1.0, variation)

        # Clamp to bounds
        return np.clip(multiplier, min_val, max_val)

    def apply_to_atmospheric_model(self, atmospheric_model, values: Optional[RandomizedValues] = None):
        """Apply randomization to atmospheric model."""
        if not self.enabled or atmospheric_model is None:
            return

        values = values or self.current_values

        # Modify atmospheric constants
        if hasattr(atmospheric_model, 'constants'):
            original_density = atmospheric_model.constants.SEA_LEVEL_DENSITY
            atmospheric_model.constants.SEA_LEVEL_DENSITY = (
                original_density * values.air_density_multiplier
            )

            # Temperature offset
            original_temp = atmospheric_model.constants.SEA_LEVEL_TEMPERATURE
            atmospheric_model.constants.SEA_LEVEL_TEMPERATURE = (
                original_temp + values.temperature_offset
            )

    def apply_to_drag_model(self, mach_drag_model, values: Optional[RandomizedValues] = None):
        """Apply randomization to Mach drag model."""
        if not self.enabled or mach_drag_model is None:
            return

        values = values or self.current_values

        # Modify base drag coefficient
        if hasattr(mach_drag_model, 'base_cd'):
            original_cd = 0.3  # Store original value
            mach_drag_model.base_cd = original_cd * values.drag_coefficient_multiplier

        # Modify Mach effects
        if hasattr(mach_drag_model, 'transonic_peak_multiplier'):
            original_peak = 3.0
            mach_drag_model.transonic_peak_multiplier = (
                original_peak * values.mach_effects_multiplier
            )

    def apply_to_sensor_delays(self, observation_generator, values: Optional[RandomizedValues] = None):
        """Apply randomization to sensor delays."""
        if not self.enabled or observation_generator is None:
            return

        values = values or self.current_values

        # Modify sensor delay buffer if present
        if hasattr(observation_generator, 'sensor_delay_buffer') and observation_generator.sensor_delay_buffer:
            original_delay = 3  # 30ms at 100Hz
            new_delay = int(original_delay * values.sensor_delay_multiplier)
            new_delay = max(1, min(new_delay, 10))  # Clamp to reasonable range

            # Recreate buffer with new delay
            from core import SensorDelayBuffer
            observation_generator.sensor_delay_buffer = SensorDelayBuffer(new_delay)

    def get_current_values(self) -> RandomizedValues:
        """Get currently active randomized values."""
        return self.current_values

    def save_randomization_log(self, filepath: Path):
        """Save randomization history to file."""
        if not self.log_randomization:
            return

        log_data = {
            'randomization_params': asdict(self.params),
            'config': self.config,
            'history': [values.to_dict() for values in self.randomization_history]
        }

        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

    def load_randomization_log(self, filepath: Path) -> List[RandomizedValues]:
        """Load randomization history from file."""
        with open(filepath, 'r') as f:
            log_data = json.load(f)

        history = []
        for entry in log_data.get('history', []):
            values = RandomizedValues(**entry)
            history.append(values)

        return history

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of randomization history."""
        if not self.randomization_history:
            return {}

        stats = {}

        # Get all numeric fields from RandomizedValues
        sample_values = self.randomization_history[0]
        numeric_fields = []
        for field, value in asdict(sample_values).items():
            if isinstance(value, (int, float)) and field not in ['timestamp', 'episode_seed']:
                numeric_fields.append(field)

        # Calculate statistics for each field
        for field in numeric_fields:
            values = [getattr(rv, field) for rv in self.randomization_history]
            stats[field] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }

        return stats