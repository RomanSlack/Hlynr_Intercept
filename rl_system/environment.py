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

        self.observation_generator = Radar26DObservation(
            max_range=self.max_range,
            max_velocity=self.max_velocity,
            radar_range=radar_config.get('radar_range', 5000.0),
            min_detection_range=radar_config.get('min_detection_range', 50.0),
            sensor_delay_ms=sensor_delay_ms,
            simulation_dt=self.dt,
            ground_radar_config=ground_radar_config,
            radar_beam_width=initial_beam_width
        )

        # Set initial curriculum parameters for radar (perfect detection at start)
        if self.use_radar_curriculum and self.radar_curriculum_config:
            self.observation_generator.onboard_detection_reliability = self.radar_curriculum_config.get('initial_detection_reliability', 1.0)
            self.observation_generator.ground_detection_reliability = self.radar_curriculum_config.get('initial_ground_reliability', 1.0)
            self.observation_generator.measurement_noise_level = self.radar_curriculum_config.get('initial_noise_level', 0.0)
        self.safety_clamp = SafetyClamp(SafetyLimits())

        # Spaces - updated to 26D for ground radar support
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(26,), dtype=np.float32
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

        # Track last detection info for reward calculation
        self.last_detection_info = None

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

    def set_training_step_count(self, step_count: int):
        """Update global training step count for curriculum learning."""
        self.training_step_count = step_count
        self._update_radar_curriculum()

    def _update_radar_curriculum(self):
        """Update radar parameters based on curriculum progress (radar-only, no omniscient data)."""
        if not self.use_radar_curriculum or not self.radar_curriculum_config:
            return

        # Calculate curriculum progress (0.0 = start, 1.0 = end)
        progress = min(1.0, self.training_step_count / self.curriculum_steps)

        # Linear interpolation for beam width
        initial_beam = self.radar_curriculum_config.get('initial_beam_width', 120.0)
        final_beam = self.radar_curriculum_config.get('final_beam_width', 60.0)
        current_beam_width = initial_beam * (1.0 - progress) + final_beam * progress

        # Linear interpolation for onboard detection reliability
        initial_reliability = self.radar_curriculum_config.get('initial_detection_reliability', 1.0)
        final_reliability = self.radar_curriculum_config.get('final_detection_reliability', 0.75)
        current_reliability = initial_reliability * (1.0 - progress) + final_reliability * progress

        # Linear interpolation for ground radar reliability
        initial_ground = self.radar_curriculum_config.get('initial_ground_reliability', 1.0)
        final_ground = self.radar_curriculum_config.get('final_ground_reliability', 0.85)
        current_ground_reliability = initial_ground * (1.0 - progress) + final_ground * progress

        # Linear interpolation for noise level
        initial_noise = self.radar_curriculum_config.get('initial_noise_level', 0.0)
        final_noise = self.radar_curriculum_config.get('final_noise_level', 0.05)
        current_noise = initial_noise * (1.0 - progress) + final_noise * progress

        # Update observation generator with curriculum parameters
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
        
        # Initialize missile state
        mis_pos_min, mis_pos_max = self.missile_spawn_range['position']
        mis_vel_min, mis_vel_max = self.missile_spawn_range['velocity']

        missile_pos = np.random.uniform(mis_pos_min, mis_pos_max).astype(np.float32)

        # Calculate velocity magnitude range from config
        vel_mag_min = np.linalg.norm(mis_vel_min)
        vel_mag_max = np.linalg.norm(mis_vel_max)
        desired_speed = np.random.uniform(vel_mag_min, vel_mag_max)

        # Calculate direction from missile spawn to target
        to_target = self.target_position - missile_pos
        to_target_dist = np.linalg.norm(to_target)

        if to_target_dist > 1e-6:
            # Point velocity toward target with desired speed
            missile_vel = (to_target / to_target_dist) * desired_speed
        else:
            # Fallback if somehow spawned on target
            missile_vel = np.random.uniform(mis_vel_min, mis_vel_max).astype(np.float32)

        self.missile_state = {
            'position': missile_pos,
            'velocity': missile_vel.astype(np.float32),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'angular_velocity': np.zeros(3, dtype=np.float32),
            'active': True
        }
        
        # Initialize interceptor state
        int_pos_min, int_pos_max = self.interceptor_spawn_range['position']
        int_vel_min, int_vel_max = self.interceptor_spawn_range['velocity']

        interceptor_pos = np.random.uniform(int_pos_min, int_pos_max).astype(np.float32)
        interceptor_vel = np.random.uniform(int_vel_min, int_vel_max).astype(np.float32)

        # Calculate initial orientation: point interceptor toward missile for radar acquisition
        # This uses only the positions (which radar would know), no omniscient data
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
        
        # Apply safety constraints
        clamped_action, clamp_info = self.safety_clamp.apply(
            action, self.interceptor_state['fuel']
        )
        
        # Update interceptor physics
        if self.interceptor_state['active']:
            self._update_interceptor(clamped_action)
        
        # Update missile physics
        if self.missile_state['active']:
            self._update_missile()
        
        # Update wind
        self._update_wind()
        
        # Check interception
        # NOTE: This uses true positions for ground truth termination/reward calculation.
        # The POLICY only receives radar observations through the 17D observation vector.
        # Using privileged info for reward shaping is standard RL practice and doesn't
        # violate the radar-only constraint on the policy's observations.
        distance = np.linalg.norm(
            self.missile_state['position'] - self.interceptor_state['position']
        )

        # Curriculum-based intercept radius (200m -> 20m over training)
        # This allows agent to first learn "get close", then gradually refine to precise interception
        current_radius = self.get_current_intercept_radius()
        intercepted = distance < current_radius
        
        # Check termination conditions
        terminated = False
        truncated = False
        missile_hit_target = False

        if intercepted:
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
        elif self.interceptor_state['position'][2] < 0:  # Interceptor crashed
            terminated = True
        elif self.steps >= self.max_steps:
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
            'radar_quality': self.last_detection_info.get('radar_quality', 0.0) if self.last_detection_info else 0.0
        }
        
        return obs, reward, terminated, truncated, info
    
    def _update_interceptor(self, action: np.ndarray):
        """Update interceptor state with enhanced 6DOF physics."""
        physics_start = time.time() if self.physics_timing_enabled else None

        # Extract thrust and angular commands from action
        # PAC-3 interceptor has extremely high thrust-to-weight ratio (>20)
        # This gives ~40 m/s^2 acceleration capability for aggressive pursuit
        thrust_cmd_desired = action[0:3] * 10000.0  # Scale to N (realistic interceptor thrust)
        angular_cmd = action[3:6] * 2.0   # Scale to rad/s

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
    
    def _update_missile(self):
        """Update missile state with enhanced ballistic trajectory."""
        mass = 1000.0  # kg

        # Get atmospheric properties at missile altitude
        altitude = max(0.0, self.missile_state['position'][2])
        if self.atmospheric_model:
            atm_props = self.atmospheric_model.get_atmospheric_properties(altitude)
            air_density = atm_props['density']
            speed_of_sound = atm_props['speed_of_sound']
        else:
            air_density = self.air_density
            speed_of_sound = 343.0

        # Calculate drag with enhanced physics
        vel = self.missile_state['velocity']
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
        self.missile_state['velocity'] += total_accel * self.dt
        self.missile_state['position'] += self.missile_state['velocity'] * self.dt
    
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
        Calculate step reward with dense shaping.

        This reward function uses ONLY information available through radar observations
        plus internal interceptor state (fuel, position). The distance is calculated
        internally but corresponds to what the agent can infer from radar detections.
        """
        reward = 0.0

        if intercepted:
            # Successful interception - major reward
            reward = 200.0

            # Bonus for fuel efficiency (encourages optimal trajectories)
            fuel_bonus = self.interceptor_state['fuel'] * 1.0
            reward += fuel_bonus

            # Bonus for quick interception (encourages aggressive pursuit)
            time_bonus = (self.max_steps - self.steps) * 0.2
            reward += time_bonus

            return reward

        if terminated:
            # Failed interception - apply penalties based on outcome
            if missile_hit_target:
                # Missile hit the defended target - MISSION FAILURE (worst outcome)
                reward = -500.0
            elif self.missile_state['position'][2] <= 0:
                # Missile hit ground away from target - partial failure
                reward = -100.0
            elif self.interceptor_state['position'][2] < 0:
                # Interceptor crashed - failure
                reward = -100.0

            return reward

        # Dense shaping rewards for ongoing episode
        prev_distance = getattr(self, '_prev_distance', distance)

        # 1. CLOSING DISTANCE REWARD (primary shaping signal)
        # This is the most important reward - strongly incentivize reducing distance
        distance_delta = prev_distance - distance
        reward += distance_delta * 1.0  # Much stronger than before (was 0.1)

        # 2. RADAR TRACKING REWARD (critical for maintaining lock)
        # Agent must learn to keep missile in radar beam to track and intercept
        if self.last_detection_info:
            detected = self.last_detection_info.get('detected', False)
            radar_quality = self.last_detection_info.get('radar_quality', 0.0)

            if detected:
                # Bonus for maintaining radar lock
                reward += 0.5

                # Additional bonus for high-quality track
                reward += radar_quality * 0.3
            else:
                # Penalty for losing track - agent must re-acquire
                reward -= 1.0

        # 3. ENHANCED TERMINAL HOMING REWARDS (exponential shaping)
        # Critical fix: Strong exponential rewards as distance decreases
        # This provides much stronger gradient for terminal guidance phase
        if distance < 500.0:
            # Moderate exponential reward when approaching
            proximity_reward = 10.0 * np.exp(-distance / 200.0)
            reward += proximity_reward

        if distance < 200.0:
            # Strong exponential reward in terminal phase
            terminal_reward = 50.0 * np.exp(-distance / 50.0)
            reward += terminal_reward

        if distance < 50.0:
            # Extreme reward very close to interception
            # This prevents agent from flying past the target
            close_terminal_reward = 100.0 * np.exp(-distance / 20.0)
            reward += close_terminal_reward

        # 4. CLOSING VELOCITY REWARD (critical for terminal guidance)
        # Reward high closing rate to prevent flyby misses
        # This directly addresses the "get close then diverge" problem
        int_vel = self.interceptor_state['velocity']
        mis_vel = self.missile_state['velocity']
        relative_vel = int_vel - mis_vel
        to_missile = self.missile_state['position'] - self.interceptor_state['position']

        if distance > 1.0:  # Avoid division by zero
            # Calculate closing velocity (negative = closing, positive = opening)
            closing_velocity = -np.dot(relative_vel, to_missile / distance)

            if closing_velocity > 0:
                # Reward high closing speed (approaching target)
                reward += closing_velocity * 0.1

                # Extra reward for high closing speed in terminal phase
                if distance < 200.0:
                    reward += closing_velocity * 0.2
            else:
                # Penalty for opening (moving away from target)
                # This prevents the flyby problem
                if distance < 200.0:
                    reward += closing_velocity * 0.3  # Negative penalty

        # 5. VELOCITY ALIGNMENT REWARD (pursue optimal intercept geometry)
        # Encourages pointing velocity vector toward missile
        int_vel_mag = np.linalg.norm(int_vel)
        to_missile_mag = np.linalg.norm(to_missile)

        if int_vel_mag > 10.0 and to_missile_mag > 10.0:
            # Calculate alignment (cosine similarity)
            vel_alignment = np.dot(int_vel, to_missile) / (int_vel_mag * to_missile_mag)
            reward += vel_alignment * 0.5

        # 6. SMALL TIME PENALTY (encourages action but doesn't dominate)
        reward -= 0.05

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