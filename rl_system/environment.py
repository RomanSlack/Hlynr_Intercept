"""
Gymnasium environment for missile interception with 6DOF physics.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
from core import Radar26DObservation, SafetyClamp, SafetyLimits, euler_to_quaternion, SensorDelayBuffer
from physics_models import AtmosphericModel, MachDragModel, EnhancedWindModel
from physics_randomizer import PhysicsRandomizer
import time


class InterceptEnvironment(gym.Env):
    """Clean missile interception environment with realistic physics."""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        # Default configuration
        self.config = config or {}
        self.dt = self.config.get('dt', 0.01)  # 100Hz simulation
        self.max_steps = self.config.get('max_steps', 1000)
        self.max_range = self.config.get('max_range', 10000.0)
        self.max_velocity = self.config.get('max_velocity', 1000.0)

        # Spawn configuration
        self.missile_spawn_range = self.config.get('missile_spawn', {
            'position': [[-500, -500, 200], [500, 500, 500]],
            'velocity': [[50, 50, -20], [150, 150, -50]]
        })
        self.interceptor_spawn_range = self.config.get('interceptor_spawn', {
            'position': [[400, 400, 50], [600, 600, 200]],
            'velocity': [[0, 0, 0], [50, 50, 20]]
        })
        self.target_position = np.array(self.config.get('target_position', [900, 900, 5]), dtype=np.float32)

        # Volley mode configuration (can be set via config or reset options)
        self.volley_mode = self.config.get('volley_mode', False)
        self.volley_size = self.config.get('volley_size', 1)
        self.missile_states = []  # List of missile states for volley mode
        
        # Basic physics parameters
        self.gravity = np.array([0, 0, -9.81], dtype=np.float32)
        self.drag_coefficient = 0.3
        self.air_density = 1.225  # kg/m^3 at sea level (fallback value)

        # Advanced Physics v2.0 - Initialize enhanced physics models
        self.physics_config = self.config.get('physics_enhancements', {})
        self.physics_enabled = self.physics_config.get('enabled', True)

        # Atmospheric model
        self.atmospheric_model = None
        if self.physics_enabled and self.physics_config.get('atmospheric_model', {}).get('enabled', True):
            self.atmospheric_model = AtmosphericModel()

        # Mach-dependent drag model
        self.mach_drag_model = None
        if self.physics_enabled and self.physics_config.get('mach_effects', {}).get('enabled', True):
            mach_config = self.physics_config.get('mach_effects', {})
            self.mach_drag_model = MachDragModel(base_drag_coefficient=self.drag_coefficient)
            self.mach_drag_model.subsonic_mach = mach_config.get('subsonic_mach', 0.8)
            self.mach_drag_model.supersonic_mach = mach_config.get('supersonic_mach', 1.2)
            self.mach_drag_model.transonic_peak_multiplier = mach_config.get('transonic_peak_multiplier', 3.0)
            self.mach_drag_model.supersonic_multiplier = mach_config.get('supersonic_multiplier', 2.5)

        # Wind configuration (must be defined before enhanced wind model)
        wind_config = self.config.get('wind', {})
        self.base_wind = np.array(wind_config.get('velocity', [5.0, 0.0, 0.0]), dtype=np.float32)
        self.wind_variability = wind_config.get('variability', 0.1)

        # Enhanced wind model
        self.enhanced_wind_model = None
        if self.physics_enabled and self.physics_config.get('enhanced_wind', {}).get('enabled', True):
            enhanced_wind_config = self.physics_config.get('enhanced_wind', {})
            self.enhanced_wind_model = EnhancedWindModel(
                base_wind=self.base_wind,
                boundary_layer_height=enhanced_wind_config.get('boundary_layer_height', 1000.0),
                surface_roughness=enhanced_wind_config.get('surface_roughness', 0.1)
            )
            self.enhanced_wind_model.turbulence_intensity = enhanced_wind_config.get('turbulence_intensity', 0.1)
            self.enhanced_wind_model.gust_scale = enhanced_wind_config.get('max_gust_speed', 5.0)

        # Thrust dynamics
        self.thrust_dynamics_enabled = (
            self.physics_enabled and
            self.physics_config.get('thrust_dynamics', {}).get('enabled', True)
        )
        if self.thrust_dynamics_enabled:
            thrust_config = self.physics_config.get('thrust_dynamics', {})
            self.thrust_time_constant = thrust_config.get('response_time_constant', 0.1)
            # Initialize actual thrust state
            self.interceptor_thrust_actual = np.zeros(3, dtype=np.float32)

        # Domain randomization
        self.physics_randomizer = None
        if self.physics_enabled and self.physics_config.get('domain_randomization', {}).get('enabled', False):
            randomization_config = self.physics_config.get('domain_randomization', {})
            self.physics_randomizer = PhysicsRandomizer(config=randomization_config)

        # Performance monitoring
        self.physics_timing_enabled = self.physics_config.get('performance', {}).get('log_physics_timing', False)
        self.physics_validation_enabled = self.physics_config.get('performance', {}).get('enable_physics_validation', True)
        
        # Curriculum learning configuration - setup FIRST before using it
        # Progressive intercept radius: Start with larger radius (easier), gradually decrease to realistic 20m
        # This allows agent to first learn "get close" before learning "precise intercept"
        self.curriculum_config = self.config.get('curriculum', {})
        self.use_curriculum = self.curriculum_config.get('enabled', True)
        self.initial_intercept_radius = self.curriculum_config.get('initial_radius', 200.0)  # Start easy
        self.final_intercept_radius = self.curriculum_config.get('final_radius', 20.0)  # End realistic
        self.curriculum_steps = self.curriculum_config.get('curriculum_steps', 5000000)  # Over 5M steps
        self.training_step_count = 0  # Track global training steps for curriculum

        # Precision mode: When enabled, episode continues after crossing intercept threshold
        # This allows the model to learn sub-threshold precision by receiving rewards
        # based on minimum distance achieved, not just threshold crossing
        self.precision_mode = self.curriculum_config.get('precision_mode', False)
        self._episode_min_distance = float('inf')  # Track minimum distance achieved in episode
        self._crossed_threshold = False  # Track if we've crossed the intercept threshold

        # Proximity fuze termination: Terminate episode when min distance drops below kill radius
        # This matches real missile defense behavior where proximity fuze detonates at closest approach
        # Success is achieving min_distance < proximity_kill_radius, not final_distance
        self.proximity_fuze_enabled = self.config.get('proximity_fuze_enabled', False)
        self.proximity_kill_radius = self.config.get('proximity_kill_radius', 20.0)  # meters

        # Radar curriculum learning: Progressive radar difficulty
        self.radar_curriculum_config = self.curriculum_config.get('radar_curriculum', {})
        self.use_radar_curriculum = self.radar_curriculum_config.get('enabled', True)

        # Radar configuration
        radar_config = self.config.get('radar', {})
        self.radar_noise = radar_config.get('radar_noise', 0.05)
        self.radar_quality = radar_config.get('radar_quality', 1.0)

        # Sensor delay configuration
        sensor_delay_ms = 0.0  # Default no delay
        if self.physics_enabled and self.physics_config.get('sensor_delays', {}).get('enabled', True):
            sensor_delay_ms = self.physics_config.get('sensor_delays', {}).get('radar_delay_ms', 30.0)

        # Ground radar configuration
        ground_radar_config = self.config.get('ground_radar', {})

        # Weather factor for ground radar (can vary per episode)
        self.weather_factor = 1.0

        # Components - now using 26D observation space with ground radar
        # Initial beam width will be updated by curriculum if enabled
        initial_beam_width = radar_config.get('radar_beam_width', 60.0)
        if self.use_radar_curriculum and self.radar_curriculum_config:
            initial_beam_width = self.radar_curriculum_config.get('initial_beam_width', 120.0)

        # Observation mode configuration:
        # - "world_frame": Original XYZ in world coordinates (direction-dependent, original behavior)
        # - "body_frame": XYZ in interceptor body frame (partially direction-invariant)
        # - "los_frame": Line-of-sight based (fully direction-invariant, best for 360° coverage)
        self.observation_mode = self.config.get('observation_mode', 'world_frame')

        # Backward compatibility: rotation_invariant=True maps to body_frame mode
        self.rotation_invariant = self.config.get('rotation_invariant', False)
        if self.rotation_invariant and self.observation_mode == 'world_frame':
            self.observation_mode = 'body_frame'

        self.observation_generator = Radar26DObservation(
            max_range=self.max_range,
            max_velocity=self.max_velocity,
            radar_range=radar_config.get('radar_range', 5000.0),
            min_detection_range=radar_config.get('min_detection_range', 50.0),
            sensor_delay_ms=sensor_delay_ms,
            simulation_dt=self.dt,
            ground_radar_config=ground_radar_config,
            radar_beam_width=initial_beam_width,
            rotation_invariant=self.rotation_invariant,
            observation_mode=self.observation_mode
        )

        # Set initial curriculum parameters for radar (perfect detection at start)
        if self.use_radar_curriculum and self.radar_curriculum_config:
            self.observation_generator.onboard_detection_reliability = self.radar_curriculum_config.get('initial_detection_reliability', 1.0)
            self.observation_generator.ground_detection_reliability = self.radar_curriculum_config.get('initial_ground_reliability', 1.0)
            self.observation_generator.measurement_noise_level = self.radar_curriculum_config.get('initial_noise_level', 0.0)
        self.safety_clamp = SafetyClamp(SafetyLimits())

        # Spaces - updated to 26D for ground radar support
        # NOTE: Observation space bounds extended to [-2.0, 1.0] to accommodate sentinel value
        # -2.0 = NO_DETECTION_SENTINEL (used when radar loses lock)
        # [-1.0, 1.0] = Normal observation range (radar measurements, internal state)
        self.observation_space = spaces.Box(
            low=-2.0, high=1.0, shape=(26,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # State variables
        self.missile_state = None
        self.interceptor_state = None
        self.current_wind = None
        self.steps = 0
        self.total_fuel_used = 0

        # LOS frame vectors for action transformation (updated each step)
        # These define the coordinate frame for LOS-relative actions
        self._los_unit = None          # Unit vector pointing from interceptor to missile
        self._los_horizontal = None    # Unit vector perpendicular to LOS in horizontal plane
        self._los_vertical = None      # Unit vector perpendicular to LOS in vertical plane

        # Track last detection info for reward calculation
        self.last_detection_info = None

        # Smart early termination tracking
        self._distance_worsening_count = 0
        self._last_distance = None

        # Volley mode tracking
        self.intercepted_missile_indices = []  # Track which missiles were intercepted
        self.missile_min_distances = []  # Track minimum distance achieved to each missile

    def get_current_intercept_radius(self) -> float:
        """
        Calculate current intercept radius based on curriculum progress.
        Linearly interpolates from initial (easy) to final (realistic) radius.
        """
        if not self.use_curriculum:
            return self.final_intercept_radius

        # Linear interpolation based on training progress
        progress = min(1.0, self.training_step_count / self.curriculum_steps)
        radius = self.initial_intercept_radius * (1.0 - progress) + self.final_intercept_radius * progress
        return radius

    def _select_priority_missile(self) -> Optional[Dict[str, Any]]:
        """
        Select the highest priority missile for observation and tracking.

        In volley mode, prioritizes:
        1. Closest active missile (most immediate threat)
        2. Falls back to any active missile if none are particularly close

        Returns:
            Priority missile state dict, or None if no active missiles
        """
        if not self.volley_mode or not self.missile_states:
            return self.missile_state

        # Filter active missiles
        active_missiles = [ms for ms in self.missile_states if ms.get('active', True)]

        if not active_missiles:
            return None

        # Find closest active missile to interceptor
        int_pos = self.interceptor_state['position']
        closest_missile = active_missiles[0]
        closest_dist = np.linalg.norm(closest_missile['position'] - int_pos)

        for ms in active_missiles[1:]:
            dist = np.linalg.norm(ms['position'] - int_pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_missile = ms

        return closest_missile

    def set_training_step_count(self, step_count: int):
        """Update global training step count for curriculum learning."""
        self.training_step_count = step_count
        self._update_radar_curriculum()

    def _update_radar_curriculum(self):
        """
        Update radar parameters based on STAGGERED curriculum progress.

        OPTIMIZATION: Each difficulty factor transitions independently to prevent cliff.
        - Beam width: Transitions 3M-5M steps (Phase 2-3)
        - Detection reliability: Transitions 4.5M-6M steps (Phase 3)
        - Measurement noise: Transitions 6M-7M steps (Phase 4)

        This allows policy to learn each skill incrementally without catastrophic forgetting.
        """
        if not self.use_radar_curriculum or not self.radar_curriculum_config:
            return

        current_step = self.training_step_count

        # === BEAM WIDTH CURRICULUM (Phase 2-3) ===
        initial_beam = self.radar_curriculum_config.get('initial_beam_width', 120.0)
        final_beam = self.radar_curriculum_config.get('final_beam_width', 60.0)
        beam_start = self.radar_curriculum_config.get('beam_width_transition_start', 3000000)
        beam_end = self.radar_curriculum_config.get('beam_width_transition_end', 5000000)

        if current_step < beam_start:
            current_beam_width = initial_beam
        elif current_step >= beam_end:
            current_beam_width = final_beam
        else:
            # Linear interpolation during transition
            beam_progress = (current_step - beam_start) / (beam_end - beam_start)
            current_beam_width = initial_beam * (1.0 - beam_progress) + final_beam * beam_progress

        # === ONBOARD DETECTION RELIABILITY CURRICULUM (Phase 3) ===
        initial_reliability = self.radar_curriculum_config.get('initial_detection_reliability', 1.0)
        final_reliability = self.radar_curriculum_config.get('final_detection_reliability', 0.75)
        reliability_start = self.radar_curriculum_config.get('reliability_transition_start', 4500000)
        reliability_end = self.radar_curriculum_config.get('reliability_transition_end', 6000000)

        if current_step < reliability_start:
            current_reliability = initial_reliability
        elif current_step >= reliability_end:
            current_reliability = final_reliability
        else:
            reliability_progress = (current_step - reliability_start) / (reliability_end - reliability_start)
            current_reliability = initial_reliability * (1.0 - reliability_progress) + final_reliability * reliability_progress

        # === GROUND RADAR RELIABILITY CURRICULUM (Phase 3) ===
        initial_ground = self.radar_curriculum_config.get('initial_ground_reliability', 1.0)
        final_ground = self.radar_curriculum_config.get('final_ground_reliability', 0.85)
        ground_start = self.radar_curriculum_config.get('ground_reliability_transition_start', 4500000)
        ground_end = self.radar_curriculum_config.get('ground_reliability_transition_end', 6000000)

        if current_step < ground_start:
            current_ground_reliability = initial_ground
        elif current_step >= ground_end:
            current_ground_reliability = final_ground
        else:
            ground_progress = (current_step - ground_start) / (ground_end - ground_start)
            current_ground_reliability = initial_ground * (1.0 - ground_progress) + final_ground * ground_progress

        # === MEASUREMENT NOISE CURRICULUM (Phase 4) ===
        initial_noise = self.radar_curriculum_config.get('initial_noise_level', 0.0)
        final_noise = self.radar_curriculum_config.get('final_noise_level', 0.05)
        noise_start = self.radar_curriculum_config.get('noise_transition_start', 6000000)
        noise_end = self.radar_curriculum_config.get('noise_transition_end', 7000000)

        if current_step < noise_start:
            current_noise = initial_noise
        elif current_step >= noise_end:
            current_noise = final_noise
        else:
            noise_progress = (current_step - noise_start) / (noise_end - noise_start)
            current_noise = initial_noise * (1.0 - noise_progress) + final_noise * noise_progress

        # Update observation generator with staggered curriculum parameters
        self.observation_generator.radar_beam_width = current_beam_width
        self.observation_generator.onboard_detection_reliability = current_reliability
        self.observation_generator.ground_detection_reliability = current_ground_reliability
        self.observation_generator.measurement_noise_level = current_noise

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            self.observation_generator.seed(seed)

        # Reset sensor delays for new episode
        self.observation_generator.reset_sensor_delays()

        # Check for volley mode in options (overrides config if provided)
        if options is not None:
            self.volley_mode = options.get('volley_mode', self.config.get('volley_mode', False))
            self.volley_size = options.get('volley_size', self.config.get('volley_size', 1))
        # Otherwise use config values (already set in __init__)

        # Reset volley tracking
        self.intercepted_missile_indices = []
        self.missile_states = []
        self.missile_min_distances = []

        # Initialize missile state(s)
        mis_pos_min, mis_pos_max = self.missile_spawn_range['position']
        mis_vel_min, mis_vel_max = self.missile_spawn_range['velocity']

        # Calculate velocity magnitude range from config
        vel_mag_min = np.linalg.norm(mis_vel_min)
        vel_mag_max = np.linalg.norm(mis_vel_max)

        # Check for spherical spawn mode (full 360° coverage)
        spawn_mode = self.missile_spawn_range.get('position_mode', 'box')

        # Spawn missiles (single or volley)
        num_missiles = self.volley_size if self.volley_mode else 1

        for i in range(num_missiles):
            if spawn_mode == 'spherical':
                # Spherical spawn: full 360° azimuth, configurable elevation
                radius_min = self.missile_spawn_range.get('radius_min', 800.0)
                radius_max = self.missile_spawn_range.get('radius_max', 1500.0)
                azimuth_range = self.missile_spawn_range.get('azimuth_range', [0, 360])
                elevation_range = self.missile_spawn_range.get('elevation_range', [10, 60])

                # Random radius, azimuth, elevation
                radius = np.random.uniform(radius_min, radius_max)
                azimuth = np.random.uniform(azimuth_range[0], azimuth_range[1]) * np.pi / 180.0
                elevation = np.random.uniform(elevation_range[0], elevation_range[1]) * np.pi / 180.0

                # Convert spherical to Cartesian (relative to target)
                x = radius * np.cos(elevation) * np.cos(azimuth)
                y = radius * np.cos(elevation) * np.sin(azimuth)
                z = radius * np.sin(elevation)
                missile_pos = (self.target_position + np.array([x, y, z])).astype(np.float32)
            else:
                # Box spawn (original behavior)
                missile_pos = np.random.uniform(mis_pos_min, mis_pos_max).astype(np.float32)

            # Get velocity settings
            velocity_mode = self.missile_spawn_range.get('velocity_mode', 'toward_target')
            speed_min = self.missile_spawn_range.get('speed_min', vel_mag_min)
            speed_max = self.missile_spawn_range.get('speed_max', vel_mag_max)
            desired_speed = np.random.uniform(speed_min, speed_max)

            # Calculate direction from missile spawn to target
            to_target = self.target_position - missile_pos
            to_target_dist = np.linalg.norm(to_target)

            if to_target_dist > 1e-6:
                # Point velocity toward target with desired speed
                missile_vel = (to_target / to_target_dist) * desired_speed
            else:
                # Fallback if somehow spawned on target
                missile_vel = np.random.uniform(mis_vel_min, mis_vel_max).astype(np.float32)

            missile_state = {
                'position': missile_pos,
                'velocity': missile_vel.astype(np.float32),
                'orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'angular_velocity': np.zeros(3, dtype=np.float32),
                'active': True
            }
            self.missile_states.append(missile_state)

        # Set primary missile state (for backward compatibility and observation generation)
        # In volley mode, this will be dynamically updated to the closest/most threatening missile
        self.missile_state = self.missile_states[0] if self.missile_states else None

        # Initialize interceptor state
        int_pos_min, int_pos_max = self.interceptor_spawn_range['position']
        int_vel_min, int_vel_max = self.interceptor_spawn_range['velocity']

        interceptor_pos = np.random.uniform(int_pos_min, int_pos_max).astype(np.float32)
        interceptor_vel = np.random.uniform(int_vel_min, int_vel_max).astype(np.float32)

        # Initialize minimum distance tracking for all missiles (now that interceptor position is known)
        if self.volley_mode:
            for missile_state in self.missile_states:
                initial_dist = np.linalg.norm(missile_state['position'] - interceptor_pos)
                self.missile_min_distances.append(initial_dist)

        # Calculate initial orientation: point interceptor toward primary missile for radar acquisition
        # In volley mode, point at closest missile
        # This uses only the positions (which radar would know), no omniscient data
        if self.volley_mode and len(self.missile_states) > 0:
            # Point at closest missile
            closest_dist = float('inf')
            closest_missile = self.missile_states[0]
            for ms in self.missile_states:
                dist = np.linalg.norm(ms['position'] - interceptor_pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_missile = ms
            rel_vector = closest_missile['position'] - interceptor_pos
        else:
            rel_vector = self.missile_state['position'] - interceptor_pos
        rel_distance = np.linalg.norm(rel_vector)

        if rel_distance > 1e-6:
            # Create quaternion that points forward vector toward missile
            forward_direction = rel_vector / rel_distance

            # Use simple approach: create quaternion from direction vector
            # Default forward is +Z axis [0, 0, 1], rotate to point at missile
            default_forward = np.array([0.0, 0.0, 1.0])

            # Calculate rotation axis (cross product)
            rotation_axis = np.cross(default_forward, forward_direction)
            rotation_axis_length = np.linalg.norm(rotation_axis)

            if rotation_axis_length > 1e-6:
                # Normalize rotation axis
                rotation_axis = rotation_axis / rotation_axis_length

                # Calculate rotation angle
                cos_angle = np.dot(default_forward, forward_direction)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                # Create quaternion from axis-angle
                half_angle = angle / 2.0
                sin_half = np.sin(half_angle)
                initial_orientation = np.array([
                    np.cos(half_angle),  # w
                    rotation_axis[0] * sin_half,  # x
                    rotation_axis[1] * sin_half,  # y
                    rotation_axis[2] * sin_half   # z
                ], dtype=np.float32)
            else:
                # Already aligned or opposite direction
                if cos_angle > 0:
                    initial_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    # 180 degree rotation around X axis
                    initial_orientation = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        else:
            # Fallback if somehow at same position
            initial_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.interceptor_state = {
            'position': interceptor_pos,
            'velocity': interceptor_vel,
            'orientation': initial_orientation,
            'angular_velocity': np.zeros(3, dtype=np.float32),
            'fuel': 100.0,
            'active': True
        }
        
        # Initialize wind
        self.current_wind = self.base_wind.copy()

        # Initialize enhanced physics models
        if self.enhanced_wind_model:
            self.enhanced_wind_model.seed(seed if seed else 0)

        if self.thrust_dynamics_enabled:
            self.interceptor_thrust_actual = np.zeros(3, dtype=np.float32)

        # Apply domain randomization for new episode
        if self.physics_randomizer:
            episode_id = f"episode_{self.steps if hasattr(self, 'steps') else 0}"
            randomized_values = self.physics_randomizer.randomize_for_episode(
                episode_id=episode_id,
                episode_seed=seed
            )

            # Apply randomizations to physics models
            self.physics_randomizer.apply_to_atmospheric_model(self.atmospheric_model, randomized_values)
            self.physics_randomizer.apply_to_drag_model(self.mach_drag_model, randomized_values)
            self.physics_randomizer.apply_to_sensor_delays(self.observation_generator, randomized_values)

        # Reset counters
        self.steps = 0
        self.total_fuel_used = 0
        
        # Compute initial observation using radar detection (with ground radar)
        obs = self.observation_generator.compute_radar_detection(
            self.interceptor_state, self.missile_state,
            self.radar_quality, self.radar_noise, self.weather_factor
        )

        # Store detection info (but don't pass omniscient data to policy)
        self.last_detection_info = self.observation_generator.get_last_detection_info()

        # Calculate initial distance for reward shaping
        self._prev_distance = np.linalg.norm(
            self.missile_state['position'] - self.interceptor_state['position']
        )

        # Reset smart early termination tracking
        self._distance_worsening_count = 0
        self._last_distance = self._prev_distance

        # Reset precision mode tracking
        self._episode_min_distance = self._prev_distance
        self._crossed_threshold = False

        # Initialize LOS frame for LOS-relative action transformation
        if self.observation_mode == "los_frame":
            self._update_los_frame()

        info = {
            'missile_pos': self.missile_state['position'].copy(),
            'interceptor_pos': self.interceptor_state['position'].copy(),
            'distance': self._prev_distance,
            'radar_detected': self.last_detection_info.get('detected', False) if self.last_detection_info else False,
            'radar_quality': self.last_detection_info.get('radar_quality', 0.0) if self.last_detection_info else 0.0
        }

        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.steps += 1

        # LOS-relative action transformation
        # When observation_mode is "los_frame", actions are interpreted as:
        # - action[0]: thrust along LOS (toward target)
        # - action[1]: thrust perpendicular to LOS (horizontal)
        # - action[2]: thrust perpendicular to LOS (vertical)
        # These are transformed to world-frame thrust before physics
        if self.observation_mode == "los_frame":
            self._update_los_frame()
            action = self._transform_los_action_to_world(action)

        # Apply safety constraints
        clamped_action, clamp_info = self.safety_clamp.apply(
            action, self.interceptor_state['fuel']
        )

        # Update interceptor physics
        if self.interceptor_state['active']:
            self._update_interceptor(clamped_action)

        # Update missile physics (all missiles in volley mode)
        if self.volley_mode:
            for i, missile_state in enumerate(self.missile_states):
                if missile_state['active']:
                    self._update_missile_state(missile_state)
        else:
            if self.missile_state['active']:
                self._update_missile()

        # Update wind
        self._update_wind()

        # Select priority missile for observation and distance calculation
        priority_missile = self._select_priority_missile()
        if priority_missile is None:
            # No active missiles (all intercepted or hit ground)
            priority_missile = self.missile_states[0] if self.volley_mode else self.missile_state

        # Update primary missile state for observation generation
        self.missile_state = priority_missile

        # Check interception for all missiles in volley mode
        # NOTE: This uses true positions for ground truth termination/reward calculation.
        # The POLICY only receives radar observations through the 26D observation vector.
        # Using privileged info for reward shaping is standard RL practice and doesn't
        # violate the radar-only constraint on the policy's observations.
        current_radius = self.get_current_intercept_radius()
        intercepted = False
        any_missile_hit_target = False

        if self.volley_mode:
            # Check all missiles for interception and update minimum distances
            for i, missile_state in enumerate(self.missile_states):
                if not missile_state['active']:
                    continue

                dist = np.linalg.norm(
                    missile_state['position'] - self.interceptor_state['position']
                )

                # Update minimum distance for this missile
                if dist < self.missile_min_distances[i]:
                    self.missile_min_distances[i] = dist

                # Check if this missile was intercepted
                # When proximity fuze is enabled, use proximity_kill_radius instead of curriculum radius
                intercept_threshold = self.proximity_kill_radius if self.proximity_fuze_enabled else current_radius
                if dist < intercept_threshold and i not in self.intercepted_missile_indices:
                    intercepted = True
                    self.intercepted_missile_indices.append(i)
                    missile_state['active'] = False  # Mark as neutralized

            # Calculate distance to closest active missile for reward
            active_distances = []
            for missile_state in self.missile_states:
                if missile_state['active']:
                    dist = np.linalg.norm(
                        missile_state['position'] - self.interceptor_state['position']
                    )
                    active_distances.append(dist)

            distance = min(active_distances) if active_distances else 0.0
        else:
            # Single missile mode (original behavior)
            distance = np.linalg.norm(
                self.missile_state['position'] - self.interceptor_state['position']
            )
            # When proximity fuze is enabled, use proximity_kill_radius instead of curriculum radius
            # This allows the model to fly past the curriculum threshold and trigger proximity fuze
            if self.proximity_fuze_enabled:
                intercepted = distance < self.proximity_kill_radius
            else:
                intercepted = distance < current_radius

        # Track minimum distance achieved in episode (for precision_mode rewards)
        self._episode_min_distance = min(self._episode_min_distance, distance)

        # Track threshold crossing (for precision_mode)
        if intercepted and not self._crossed_threshold:
            self._crossed_threshold = True

        # PROXIMITY FUZE TERMINATION: If enabled, terminate when min distance < kill radius
        # This matches real missile defense behavior where proximity fuze detonates at closest approach
        proximity_fuze_triggered = False
        if self.proximity_fuze_enabled and self._episode_min_distance < self.proximity_kill_radius:
            proximity_fuze_triggered = True
            intercepted = True  # Count as successful intercept

        # Check termination conditions
        terminated = False
        truncated = False
        missile_hit_target = False

        if self.volley_mode:
            # Volley mode termination logic
            # Check if ANY missile hit the target
            all_missiles_inactive = True
            for missile_state in self.missile_states:
                if missile_state['position'][2] <= 0:  # Missile hit ground
                    missile_state['active'] = False
                    # Check if missile hit near the target (mission failure)
                    missile_ground_pos = missile_state['position'][:2]  # X, Y only
                    target_ground_pos = self.target_position[:2]
                    ground_distance = np.linalg.norm(missile_ground_pos - target_ground_pos)

                    if ground_distance < 500.0:  # Within 500m of target = mission failure
                        any_missile_hit_target = True
                        missile_hit_target = True

                if missile_state['active']:
                    all_missiles_inactive = False

            # Episode ends when all missiles are neutralized (intercepted or hit ground)
            if all_missiles_inactive:
                terminated = True
            # Proximity fuze termination in volley mode
            elif proximity_fuze_triggered:
                terminated = True
        else:
            # Single missile mode termination logic
            # PRECISION MODE: Don't terminate on threshold crossing, continue to learn sub-threshold precision
            if self.precision_mode:
                # In precision mode, we continue the episode after crossing threshold
                # Termination only on: missile hits ground, interceptor crash, fuel out, timeout
                if self.missile_state['position'][2] <= 0:  # Missile hit ground
                    # Check if missile hit near the target (mission failure)
                    missile_ground_pos = self.missile_state['position'][:2]  # X, Y only
                    target_ground_pos = self.target_position[:2]
                    ground_distance = np.linalg.norm(missile_ground_pos - target_ground_pos)

                    if ground_distance < 500.0:  # Within 500m of target = mission failure
                        missile_hit_target = True
                        terminated = True
                    else:
                        # Missile hit ground far from target (harmless)
                        terminated = True
                # Note: intercepted threshold crossing does NOT terminate in precision_mode
            else:
                # Standard mode (original behavior): terminate on intercept
                if intercepted:
                    terminated = True
                # Proximity fuze termination (terminates when min_distance < kill_radius)
                elif proximity_fuze_triggered:
                    terminated = True
                elif self.missile_state['position'][2] <= 0:  # Missile hit ground
                    # Check if missile hit near the target (mission failure)
                    missile_ground_pos = self.missile_state['position'][:2]  # X, Y only
                    target_ground_pos = self.target_position[:2]
                    ground_distance = np.linalg.norm(missile_ground_pos - target_ground_pos)

                    if ground_distance < 500.0:  # Within 500m of target = mission failure
                        missile_hit_target = True
                        terminated = True
                    else:
                        # Missile hit ground far from target (harmless)
                        terminated = True

        # Common termination conditions (both modes)
        if self.interceptor_state['position'][2] < 0:  # Interceptor crashed
            terminated = True
        elif self.interceptor_state['fuel'] <= 0:  # Out of fuel
            terminated = True
        # SMART EARLY TERMINATION: Only terminate if consistently getting WORSE
        # Track if distance has been increasing (not improving) for sustained period
        elif self.steps > 1000:
            # Update worsening counter based on distance trend
            if self._last_distance is not None:
                if distance > self._last_distance:
                    # Distance increasing (getting worse) - increment counter
                    self._distance_worsening_count += 1
                else:
                    # Distance decreasing (improving) - decay counter quickly
                    self._distance_worsening_count = max(0, self._distance_worsening_count - 5)

            # Update last distance for next comparison
            self._last_distance = distance

            # Only terminate if distance has been worsening for 500 consecutive steps (5 seconds)
            # AND we're far away (>2500m), indicating policy has completely failed
            if self._distance_worsening_count > 500 and distance > 2500.0:
                terminated = True

        if self.steps >= self.max_steps:
            truncated = True
        
        # Generate observation using radar detection (with ground radar)
        obs = self.observation_generator.compute_radar_detection(
            self.interceptor_state, self.missile_state,
            self.radar_quality, self.radar_noise, self.weather_factor
        )

        # Store detection info for reward calculation (internal only, not passed to policy)
        self.last_detection_info = self.observation_generator.get_last_detection_info()

        # Calculate reward with detection info
        reward = self._calculate_reward(distance, intercepted, terminated, missile_hit_target)

        # Info dict (detection info for logging only, not passed to policy observations)
        info = {
            'distance': distance,
            'intercepted': intercepted,
            'missile_hit_target': missile_hit_target,
            'fuel_remaining': self.interceptor_state['fuel'],
            'fuel_used': self.total_fuel_used,
            'clamped': clamp_info['clamped'],
            'missile_pos': self.missile_state['position'].copy(),
            'interceptor_pos': self.interceptor_state['position'].copy(),
            'steps': self.steps,
            'radar_detected': self.last_detection_info.get('detected', False) if self.last_detection_info else False,
            'radar_quality': self.last_detection_info.get('radar_quality', 0.0) if self.last_detection_info else 0.0,
            # Comprehensive radar debug info for logging
            'radar_debug': self.observation_generator.get_last_radar_debug_info(),
            # Volley mode specific info
            'volley_mode': self.volley_mode,
            'volley_size': self.volley_size if self.volley_mode else 1,
            'missiles_intercepted': len(self.intercepted_missile_indices) if self.volley_mode else (1 if intercepted else 0),
            'missiles_remaining': sum(1 for ms in self.missile_states if ms['active']) if self.volley_mode else (0 if intercepted else 1),
            'missile_min_distances': self.missile_min_distances.copy() if self.volley_mode else [distance],
            # Precision mode tracking
            'min_distance': self._episode_min_distance,
            'crossed_threshold': self._crossed_threshold,
            'precision_mode': self.precision_mode,
            # Proximity fuze tracking
            'proximity_fuze_enabled': self.proximity_fuze_enabled,
            'proximity_fuze_triggered': proximity_fuze_triggered,
            'proximity_kill_radius': self.proximity_kill_radius,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _update_interceptor(self, action: np.ndarray):
        """Update interceptor state with enhanced 6DOF physics."""
        physics_start = time.time() if self.physics_timing_enabled else None

        # Extract thrust and angular commands from action
        # PAC-3 interceptor has extremely high thrust-to-weight ratio (>20)
        # This gives ~40 m/s^2 acceleration capability for aggressive pursuit
        # ACTION SCALING OPTIMIZATION: Balanced scaling for stable gradient flow
        # Both thrust and angular scaled proportionally to prevent gradient imbalance
        thrust_cmd_desired = action[0:3] * 10000.0  # Scale to N (realistic interceptor thrust)
        angular_cmd = action[3:6] * 20.0   # Scale to rad/s (10x increase for balanced gradients)

        # Apply thrust dynamics (first-order lag response)
        if self.thrust_dynamics_enabled:
            # First-order response: actual_thrust += (desired - actual) * dt/tau
            thrust_error = thrust_cmd_desired - self.interceptor_thrust_actual
            self.interceptor_thrust_actual += thrust_error * self.dt / self.thrust_time_constant
            thrust_cmd = self.interceptor_thrust_actual
        else:
            thrust_cmd = thrust_cmd_desired

        # Calculate fuel consumption based on actual thrust
        thrust_mag = np.linalg.norm(thrust_cmd)
        fuel_consumed = (thrust_mag / 500.0) * self.safety_clamp.limits.fuel_depletion_rate * self.dt
        self.interceptor_state['fuel'] -= fuel_consumed
        self.total_fuel_used += fuel_consumed

        if self.interceptor_state['fuel'] <= 0:
            self.interceptor_state['fuel'] = 0
            thrust_cmd *= 0
            if self.thrust_dynamics_enabled:
                self.interceptor_thrust_actual *= 0

        # Vehicle properties
        mass = 500.0  # kg
        thrust_accel = thrust_cmd / mass

        # Get atmospheric properties at current altitude
        altitude = max(0.0, self.interceptor_state['position'][2])
        if self.atmospheric_model:
            atm_props = self.atmospheric_model.get_atmospheric_properties(altitude)
            air_density = atm_props['density']
            speed_of_sound = atm_props['speed_of_sound']
        else:
            air_density = self.air_density  # Fallback to constant density
            speed_of_sound = 343.0  # Standard speed of sound at sea level

        # Calculate drag with enhanced physics
        vel = self.interceptor_state['velocity']
        vel_air = vel - self.current_wind

        if self.mach_drag_model and np.linalg.norm(vel_air) > 1e-6:
            # Use Mach-dependent drag model
            drag_force = self.mach_drag_model.get_drag_force(
                vel_air, air_density, speed_of_sound
            )
            drag_accel = drag_force / mass
        else:
            # Fallback to simple drag model
            drag_accel = -0.5 * self.drag_coefficient * air_density * \
                         np.linalg.norm(vel_air) * vel_air / mass

        # Total acceleration
        total_accel = thrust_accel + drag_accel + self.gravity

        # Physics validation
        if self.physics_validation_enabled:
            if not np.isfinite(total_accel).all():
                print(f"Warning: Non-finite acceleration detected: {total_accel}")
                total_accel = np.nan_to_num(total_accel, nan=0.0, posinf=50.0, neginf=-50.0)
        
        # Update velocity and position
        self.interceptor_state['velocity'] += total_accel * self.dt
        self.interceptor_state['position'] += self.interceptor_state['velocity'] * self.dt

        # Update angular velocity and orientation
        self.interceptor_state['angular_velocity'] = angular_cmd

        # Simple quaternion integration (small angle approximation)
        w = self.interceptor_state['angular_velocity']
        angle = np.linalg.norm(w) * self.dt
        if angle > 1e-6:
            axis = w / np.linalg.norm(w)
            dq = np.array([
                np.cos(angle/2),
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2)
            ])
            self.interceptor_state['orientation'] = self._quaternion_multiply(
                dq, self.interceptor_state['orientation']
            )
            # Normalize quaternion
            self.interceptor_state['orientation'] /= np.linalg.norm(
                self.interceptor_state['orientation']
            )

        # Log physics timing if enabled
        if self.physics_timing_enabled:
            physics_time = (time.time() - physics_start) * 1000.0  # Convert to ms
            max_time = self.physics_config.get('performance', {}).get('max_physics_time_ms', 10.0)
            if physics_time > max_time:
                print(f"Warning: Physics computation took {physics_time:.2f}ms (exceeds {max_time}ms threshold)")
    
    def _update_los_frame(self):
        """
        Update the LOS (Line-of-Sight) coordinate frame vectors.

        Called at the start of each step to establish the frame for action transformation.
        The LOS frame is defined as:
        - los_unit: Points from interceptor toward missile (along LOS)
        - los_horizontal: Perpendicular to LOS in the horizontal plane (for lateral maneuvers)
        - los_vertical: Perpendicular to LOS in the vertical plane (for altitude maneuvers)

        This enables direction-invariant control: "thrust along LOS" means the same thing
        regardless of where in the world the engagement is happening.
        """
        int_pos = np.array(self.interceptor_state['position'], dtype=np.float32)
        mis_pos = np.array(self.missile_state['position'], dtype=np.float32)

        # Relative position from interceptor to missile
        rel_pos = mis_pos - int_pos
        range_to_target = np.linalg.norm(rel_pos)

        if range_to_target > 1e-6:
            # LOS unit vector (points toward target)
            self._los_unit = rel_pos / range_to_target
        else:
            # Fallback if on top of target
            self._los_unit = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # World up vector for reference
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Horizontal component of LOS (project onto XY plane)
        los_horizontal_proj = self._los_unit - np.dot(self._los_unit, world_up) * world_up
        los_horizontal_norm = np.linalg.norm(los_horizontal_proj)

        if los_horizontal_norm > 1e-6:
            los_horizontal_unit = los_horizontal_proj / los_horizontal_norm
            # Horizontal perpendicular: cross world_up with horizontal LOS
            # This gives a vector perpendicular to LOS in the horizontal plane
            self._los_horizontal = np.cross(world_up, los_horizontal_unit)
            horiz_norm = np.linalg.norm(self._los_horizontal)
            if horiz_norm > 1e-6:
                self._los_horizontal = self._los_horizontal / horiz_norm
            else:
                self._los_horizontal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            # LOS is nearly vertical - use arbitrary horizontal reference
            self._los_horizontal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Vertical perpendicular: cross LOS with horizontal perpendicular
        # This gives a vector perpendicular to LOS in the vertical plane
        self._los_vertical = np.cross(self._los_unit, self._los_horizontal)
        vert_norm = np.linalg.norm(self._los_vertical)
        if vert_norm > 1e-6:
            self._los_vertical = self._los_vertical / vert_norm
        else:
            self._los_vertical = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _transform_los_action_to_world(self, action: np.ndarray) -> np.ndarray:
        """
        Transform LOS-relative action to world-frame action.

        LOS-relative action space:
        - action[0]: Thrust along LOS (positive = toward target)
        - action[1]: Thrust perpendicular to LOS in horizontal plane (lateral correction)
        - action[2]: Thrust perpendicular to LOS in vertical plane (vertical correction)
        - action[3:6]: Angular rates (unchanged, body-relative)

        This transformation enables direction-invariant learning:
        The same action "thrust toward target" works from any approach direction.

        For proportional navigation guidance, the policy can learn:
        - action[0] > 0: Close the distance
        - action[1], action[2]: Correct based on LOS rates to achieve collision course
        """
        if self._los_unit is None:
            # LOS frame not yet initialized - return action unchanged
            return action

        world_action = action.copy()

        # Transform thrust components from LOS frame to world frame
        # thrust_world = thrust_los * los_unit + thrust_horiz * los_horizontal + thrust_vert * los_vertical
        thrust_los = action[0]      # Along LOS (toward target)
        thrust_horiz = action[1]    # Perpendicular horizontal
        thrust_vert = action[2]     # Perpendicular vertical

        world_thrust = (
            thrust_los * self._los_unit +
            thrust_horiz * self._los_horizontal +
            thrust_vert * self._los_vertical
        )

        world_action[0:3] = world_thrust

        # Angular commands remain unchanged (body-relative)
        # action[3:6] are angular rates in body frame

        return world_action

    def _update_missile(self):
        """Update missile state with enhanced ballistic trajectory."""
        self._update_missile_state(self.missile_state)

    def _update_missile_state(self, missile_state: Dict[str, Any]):
        """Update a specific missile state with enhanced ballistic trajectory."""
        mass = 1000.0  # kg

        # Get atmospheric properties at missile altitude
        altitude = max(0.0, missile_state['position'][2])
        if self.atmospheric_model:
            atm_props = self.atmospheric_model.get_atmospheric_properties(altitude)
            air_density = atm_props['density']
            speed_of_sound = atm_props['speed_of_sound']
        else:
            air_density = self.air_density
            speed_of_sound = 343.0

        # Calculate drag with enhanced physics
        vel = missile_state['velocity']
        vel_air = vel - self.current_wind

        if self.mach_drag_model and np.linalg.norm(vel_air) > 1e-6:
            # Use Mach-dependent drag for missile too (typically larger, less agile)
            # Create a separate drag model for missile with higher base drag
            missile_base_cd = self.drag_coefficient * 1.5  # Missiles typically have higher drag
            drag_force = self.mach_drag_model.get_drag_force(
                vel_air, air_density, speed_of_sound, reference_area=2.0  # Larger cross-section
            )
            # Scale by missile drag coefficient ratio
            drag_force *= (missile_base_cd / self.drag_coefficient)
            drag_accel = drag_force / mass
        else:
            # Fallback to simple drag
            drag_accel = -0.5 * self.drag_coefficient * air_density * \
                         np.linalg.norm(vel_air) * vel_air / mass

        # Add slight evasion (if configured)
        evasion = np.zeros(3)
        if self.config.get('missile_evasion', False):
            evasion = np.random.randn(3) * 2.0  # Small random acceleration

        # Total acceleration
        total_accel = drag_accel + self.gravity + evasion

        # Physics validation for missile too
        if self.physics_validation_enabled:
            if not np.isfinite(total_accel).all():
                total_accel = np.nan_to_num(total_accel, nan=0.0, posinf=20.0, neginf=-20.0)

        # Update velocity and position
        missile_state['velocity'] += total_accel * self.dt
        missile_state['position'] += missile_state['velocity'] * self.dt
    
    def _update_wind(self):
        """Update wind with enhanced modeling."""
        if self.enhanced_wind_model:
            # Use enhanced wind model - get wind at interceptor position
            interceptor_pos = self.interceptor_state['position']
            self.current_wind = self.enhanced_wind_model.get_wind_vector(interceptor_pos, self.dt)
        else:
            # Fallback to simple wind variability
            if self.wind_variability > 0:
                wind_change = np.random.randn(3) * self.wind_variability
                self.current_wind = 0.95 * self.current_wind + 0.05 * (self.base_wind + wind_change)
    
    def _calculate_reward(self, distance: float, intercepted: bool, terminated: bool,
                         missile_hit_target: bool = False) -> float:
        """
        Reward function with precision_mode support.

        Standard mode:
        - Terminal reward on intercept (+5000)
        - Distance-based penalty on failure
        - Small per-step gradient rewards

        Precision mode:
        - Episode continues after threshold crossing
        - Rewards based on minimum distance achieved (exponential scaling)
        - Strong incentive to get closer than threshold

        This ensures: intercept_reward >> any_per_step_accumulation * max_episode_length
        """
        reward = 0.0

        # === PRECISION MODE REWARDS ===
        if self.precision_mode:
            # In precision mode, we reward based on minimum distance achieved
            # rather than just threshold crossing

            if terminated:
                min_dist = self._episode_min_distance

                if self._crossed_threshold:
                    # Successfully crossed threshold - now reward based on HOW close we got
                    # Base reward for crossing threshold
                    reward = 3000.0

                    # Exponential bonus for sub-threshold precision
                    # This creates strong gradient to get closer than the threshold
                    current_radius = self.get_current_intercept_radius()

                    # Multi-scale exponential rewards for precision
                    # Scale 1: Reward for beating threshold significantly
                    if min_dist < current_radius:
                        improvement_ratio = (current_radius - min_dist) / current_radius
                        reward += improvement_ratio * 1000.0  # Up to +1000 for much better than threshold

                    # Scale 2: Exponential bonus for close range (effective 0-50m)
                    reward += np.exp(-min_dist / 25.0) * 500.0

                    # Scale 3: Strong exponential bonus for very close (effective 0-20m)
                    reward += np.exp(-min_dist / 10.0) * 1000.0

                    # Scale 4: Extreme precision bonus (effective 0-5m)
                    reward += np.exp(-min_dist / 3.0) * 500.0

                    # Time bonus for quick intercept
                    time_bonus = (self.max_steps - self.steps) * 0.3
                    reward += time_bonus

                else:
                    # Never crossed threshold - failure
                    # Penalty proportional to how far we were from threshold
                    current_radius = self.get_current_intercept_radius()
                    miss_amount = min_dist - current_radius
                    reward = -min_dist * 0.5  # Penalty for distance
                    reward = max(reward, -2000.0)  # Cap penalty

                    if missile_hit_target:
                        reward -= 1000.0
                    elif self.interceptor_state['position'][2] < 0:
                        reward -= 500.0
                    elif self.interceptor_state['fuel'] <= 0:
                        reward -= 300.0

                return reward

            # Per-step rewards in precision mode
            # Strong gradient toward target, especially at close range
            prev_distance = getattr(self, '_prev_distance', distance)
            distance_delta = prev_distance - distance

            # Graduated per-step rewards based on distance
            if distance < 50.0:
                # Very close: strong gradient for final approach
                reward += distance_delta * 3.0
                # Bonus for staying very close
                reward += np.exp(-distance / 10.0) * 0.5
            elif distance < 200.0:
                # Close range: moderate gradient
                reward += distance_delta * 1.5
            else:
                # Far range: weak gradient
                reward += distance_delta * 0.5

            # Small time penalty to encourage efficiency
            reward -= 0.3

            # Store distance for next step
            self._prev_distance = distance
            return reward

        # === STANDARD MODE REWARDS (original behavior) ===
        if intercepted:
            # MASSIVE success reward to make interception the dominant objective
            reward = 5000.0

            # Large time bonus for quick intercept
            time_bonus = (self.max_steps - self.steps) * 0.5
            reward += time_bonus

            return reward

        if terminated:
            # Failure penalty proportional to how close we got
            distance_penalty = -distance * 0.5
            reward = max(distance_penalty, -2000.0)

            if missile_hit_target:
                reward -= 1000.0
            elif self.interceptor_state['position'][2] < 0:
                reward -= 500.0
            elif self.interceptor_state['fuel'] <= 0:
                reward -= 300.0

            return reward

        # Per-step rewards (standard mode)
        prev_distance = getattr(self, '_prev_distance', distance)
        distance_delta = prev_distance - distance

        if distance < 500.0:
            reward += distance_delta * 1.0
        else:
            reward += distance_delta * 0.5

        reward -= 0.5

        # Store distance for next step
        self._prev_distance = distance

        return reward
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float32)
    
    def render(self):
        """Render environment (placeholder for future implementation)."""
        pass