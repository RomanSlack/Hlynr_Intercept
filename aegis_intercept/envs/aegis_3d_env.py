"""3D Missile Intercept Environment for AegisIntercept"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from ..utils.physics3d import distance, linear_drag
from ..rendering.viewer3d import Viewer3D

class Aegis3DInterceptEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        world_size: float = 300.0,  # New coordinate system 0-600
        max_steps: int = 250,  # Longer episodes since missiles start farther away
        dt: float = 0.05,
        intercept_threshold: float = 30.0,  # Slightly easier intercepts
        miss_threshold: float = 10.0,  # Reasonable target protection  
        max_velocity: float = 50.0,  # Very fast interceptor
        max_accel: float = 25.0,  # High acceleration
        drag_coefficient: float = 0.03,  # Low drag for missile-like behavior
        missile_speed: float = 25.0,  # Faster incoming missile
        evasion_freq: int = 15,  # More frequent evasion
        evasion_magnitude: float = 4.0,  # Larger evasion maneuvers
        max_fuel: float = 100.0,  # Limited fuel for interceptor
        fuel_burn_rate: float = 0.5,  # Fuel consumed per acceleration
        explosion_radius: float = 40.0,  # Explosion effective radius
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.world_size = world_size
        self.max_steps = max_steps
        self.dt = dt
        self.intercept_threshold = intercept_threshold
        self.miss_threshold = miss_threshold
        self.max_velocity = max_velocity
        self.max_accel = max_accel
        self.drag_coefficient = drag_coefficient
        self.missile_speed = missile_speed
        self.evasion_freq = evasion_freq
        self.evasion_magnitude = evasion_magnitude
        self.max_fuel = max_fuel
        self.fuel_burn_rate = fuel_burn_rate
        self.explosion_radius = explosion_radius
        self.render_mode = render_mode

        # State: [interceptor_pos(3), interceptor_vel(3), missile_pos(3), missile_vel(3), time_remaining, fuel_remaining]
        # Fix observation bounds: positions [0, world*2], velocities [-max_vel, +max_vel], time [0, 1], fuel [0, 1]
        low = np.array(
            [0, 0, 0] +  # interceptor_pos
            [-max_velocity, -max_velocity, -max_velocity] +  # interceptor_vel
            [0, 0, 0] +  # missile_pos
            [-missile_speed, -missile_speed, -missile_speed] +  # missile_vel
            [0] +  # time_remaining
            [0],  # fuel_remaining
            dtype=np.float32
        )
        high = np.array(
            [world_size * 2, world_size * 2, world_size * 2] +  # interceptor_pos
            [max_velocity, max_velocity, max_velocity] +  # interceptor_vel
            [world_size * 2, world_size * 2, world_size * 2] +  # missile_pos
            [missile_speed, missile_speed, missile_speed] +  # missile_vel
            [1] +  # time_remaining
            [1],  # fuel_remaining
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action: [accel_x, accel_y, accel_z, explode]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        self.interceptor_pos = np.zeros(3, dtype=np.float32)
        self.interceptor_vel = np.zeros(3, dtype=np.float32)
        self.missile_pos = np.zeros(3, dtype=np.float32)
        self.missile_vel = np.zeros(3, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)

        self.step_count = 0
        self.viewer = None
        self.popup_message = None
        self.popup_timer = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.step_count = 0

        # Target position (what we're defending) - at ground level in center
        self.target_pos = np.array([
            self.world_size + self.np_random.uniform(-20, 20),  # Center around 300
            self.world_size + self.np_random.uniform(-20, 20),  # Center around 300
            0.0  # Ground level (z=0)
        ], dtype=np.float32)
        
        # Interceptor starts at ground level near target (launch site)
        interceptor_offset = self.np_random.uniform(10, 30)
        interceptor_angle = self.np_random.uniform(0, 2 * np.pi)
        self.interceptor_pos = self.target_pos + np.array([
            interceptor_offset * np.cos(interceptor_angle),
            interceptor_offset * np.sin(interceptor_angle),
            0.0  # Start at ground level
        ], dtype=np.float32)
        # Start with strong upward launch velocity and prevent ground collision
        self.interceptor_vel = np.array([0.0, 0.0, 8.0], dtype=np.float32)
        
        # Initialize fuel
        self.fuel_remaining = self.max_fuel
        
        # Track interceptor state for constraints
        self.max_altitude_reached = self.interceptor_pos[2]
        self.initial_direction = None  # Will be set after first movement

        # Missile starts from random direction at longer range for realism
        missile_distance_from_target = self.np_random.uniform(200, 350)  # Further back
        
        # Random approach direction (full 360 degrees around target)
        missile_angle = self.np_random.uniform(0, 2 * np.pi)
        
        # Calculate initial position
        missile_x = self.target_pos[0] + missile_distance_from_target * np.cos(missile_angle)
        missile_y = self.target_pos[1] + missile_distance_from_target * np.sin(missile_angle)
        
        # Clamp to world bounds with some margin
        missile_x = np.clip(missile_x, 30, self.world_size * 2 - 30)
        missile_y = np.clip(missile_y, 30, self.world_size * 2 - 30)
        
        # Random height - can come from high, medium, or low altitude
        missile_height = self.np_random.uniform(100, 400)  # Higher and more varied altitude
        
        self.missile_pos = np.array([
            missile_x,
            missile_y,
            missile_height
        ], dtype=np.float32)
        
        # Initialize missile trajectory toward target
        direction = (self.target_pos - self.missile_pos) / np.linalg.norm(self.target_pos - self.missile_pos)
        self.missile_vel = direction * self.missile_speed
        
        # Initialize evasion state for more realistic flight patterns
        self.missile_evasion_timer = 0
        self.current_evasion_direction = np.zeros(3, dtype=np.float32)
        self.evasion_duration = 0
        self.base_direction = direction.copy()  # Store original direction

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Debug: Print step count occasionally to see if environment is running
        if self.step_count == 0:
            print(f"[ENV] Starting new episode")
        elif self.step_count == 125:
            print(f"[ENV] Halfway through episode (step {self.step_count})")
            
        action = np.clip(action, -1.0, 1.0)
        
        # Extract explosion command (threshold of 0.5 to trigger explosion)
        explosion_command = action[3] > 0.5
        thrust_action = action[:3]  # Only use first 3 components for thrust
        
        # Calculate fuel consumption based on thrust magnitude
        thrust_magnitude = np.linalg.norm(thrust_action)
        fuel_consumed = thrust_magnitude * self.fuel_burn_rate * self.dt
        self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_consumed)
        
        # Apply thrust only if fuel is available
        if self.fuel_remaining > 0:
            thrust = thrust_action * self.max_accel
        else:
            thrust = np.zeros(3, dtype=np.float32)  # No thrust without fuel

        # Apply physics - drag and thrust
        drag = linear_drag(self.interceptor_vel, self.drag_coefficient)
        self.interceptor_vel += (thrust + drag) * self.dt
        
        # Velocity limits
        vel_mag = np.linalg.norm(self.interceptor_vel)
        if vel_mag > self.max_velocity:
            self.interceptor_vel = self.interceptor_vel / vel_mag * self.max_velocity
        
        # Update position - allow going underground (realistic for intercept)
        old_pos = self.interceptor_pos.copy()
        self.interceptor_pos = self.interceptor_pos + self.interceptor_vel * self.dt
        
        # Track altitude and backward movement
        self.max_altitude_reached = max(self.max_altitude_reached, self.interceptor_pos[2])
        
        # Set initial direction after first movement
        if self.initial_direction is None and self.step_count > 0:
            movement = self.interceptor_pos - old_pos
            if np.linalg.norm(movement) > 0.1:  # Only set if meaningful movement
                self.initial_direction = movement / np.linalg.norm(movement)
        
        # Check for backward movement (dot product with initial direction)
        backward_movement = False
        if self.initial_direction is not None and self.step_count > 10:  # Allow some initial adjustment
            movement = self.interceptor_pos - old_pos
            if np.linalg.norm(movement) > 0.1:
                movement_direction = movement / np.linalg.norm(movement)
                # If moving opposite to initial direction (dot product < -0.5)
                backward_movement = np.dot(movement_direction, self.initial_direction) < -0.5
        
        # Check for altitude loss (more than 10m drop from max)
        altitude_loss = (self.max_altitude_reached - self.interceptor_pos[2]) > 10.0

        # Advanced missile evasion behavior
        self.missile_evasion_timer += 1
        
        # Update base direction toward target
        self.base_direction = (self.target_pos - self.missile_pos) / np.linalg.norm(self.target_pos - self.missile_pos)
        
        # Start new evasive maneuver
        if self.missile_evasion_timer % self.evasion_freq == 0:
            # Random evasive maneuver duration
            self.evasion_duration = self.np_random.integers(5, 20)  # 5-20 steps
            
            # Generate random evasion direction (can include vertical now)
            evasion_direction = self.np_random.normal(0, 1, 3)
            # Bias toward horizontal evasion but allow some vertical
            evasion_direction[2] *= 0.3  # Reduce vertical component
            
            if np.linalg.norm(evasion_direction) > 0:
                self.current_evasion_direction = evasion_direction / np.linalg.norm(evasion_direction)
            else:
                self.current_evasion_direction = np.zeros(3, dtype=np.float32)
        
        # Apply evasive maneuver if active
        if self.evasion_duration > 0:
            self.evasion_duration -= 1
            # Blend evasion with base direction (70% target, 30% evasion)
            evasion_strength = 0.3 * (self.evasion_duration / 20.0)  # Fade out over time
            final_direction = (0.7 * self.base_direction + 
                             evasion_strength * self.current_evasion_direction)
            final_direction = final_direction / np.linalg.norm(final_direction)
        else:
            # No evasion, go straight to target with small random variance
            noise = self.np_random.normal(0, 0.05, 3)  # Small random flight variance
            final_direction = self.base_direction + noise
            final_direction = final_direction / np.linalg.norm(final_direction)
        
        # Update missile velocity and position
        self.missile_vel = final_direction * self.missile_speed
        self.missile_pos += self.missile_vel * self.dt

        intercept_dist = distance(self.interceptor_pos, self.missile_pos)
        miss_dist = distance(self.missile_pos, self.target_pos)

        # Check for explosion-based interception
        exploded = explosion_command and intercept_dist < self.explosion_radius
        # Keep original close-proximity interception as backup
        intercepted = exploded or intercept_dist < self.intercept_threshold
        missile_hit = miss_dist < self.miss_threshold
        
        # Check if interceptor went above the missile (punishment condition)
        interceptor_above_missile = self.interceptor_pos[2] > self.missile_pos[2] + 15.0  # 15m buffer
        
        # Check if interceptor ran out of fuel
        out_of_fuel = self.fuel_remaining <= 0
        
        # Check bounds for 0-600 coordinate system
        above_ceiling = self.interceptor_pos[2] > self.world_size * 2
        outside_horizontal = (self.interceptor_pos[0] < 0) or (self.interceptor_pos[0] > self.world_size * 2) or (self.interceptor_pos[1] < 0) or (self.interceptor_pos[1] > self.world_size * 2)
        # Allow going underground now (realistic for intercepts)
        out_of_bounds = above_ceiling or outside_horizontal
        max_steps_reached = self.step_count >= self.max_steps

        # Separate terminated (episode done due to success/failure) from truncated (timeout)
        terminated = intercepted or missile_hit or out_of_bounds or interceptor_above_missile or out_of_fuel or backward_movement or altitude_loss
        truncated = max_steps_reached and not terminated  # Only truncate if not already terminated
        
        # Improved reward function for realistic intercept behavior
        reward = 0.0
        
        if intercepted:
            # Bonus for intercepting early (more fuel remaining = better)
            fuel_bonus = (self.fuel_remaining / self.max_fuel) * 5.0
            time_bonus = (self.max_steps - self.step_count) / self.max_steps * 3.0
            # Extra bonus for explosion-based interception (realistic behavior)
            explosion_bonus = 2.0 if exploded else 0.0
            reward = 15.0 + fuel_bonus + time_bonus + explosion_bonus  # 15-25 based on efficiency
        elif missile_hit:
            reward = -8.0  # Big penalty for mission failure
        elif backward_movement:
            reward = -5.0  # Penalty for going backward
        elif altitude_loss:
            reward = -4.5  # Penalty for losing altitude
        elif interceptor_above_missile:
            reward = -4.0  # Punishment for going above the missile
        elif out_of_fuel:
            reward = -3.0  # Penalty for running out of fuel
        elif out_of_bounds:
            reward = -2.0  # Penalty for going out of bounds
        elif truncated:
            # Final reward based on how close we got
            final_distance_reward = max(0, 5.0 - intercept_dist / 100.0)  # 0-5 based on final distance
            reward = final_distance_reward - 1.0  # -1 to +4 based on performance
        else:
            # Time pressure - small penalty each step to encourage quick action
            time_penalty = -0.02  # Small time pressure
            
            # Fuel efficiency reward/penalty
            fuel_efficiency = -0.01 if self.fuel_remaining < self.max_fuel * 0.9 else 0.0
            
            # Distance-based reward (smaller than before)
            max_dist = 400.0
            distance_reward = max(0, 0.5 * (1.0 - intercept_dist / max_dist))  # 0-0.5 based on distance
            
            # Approach velocity reward - reward for moving toward missile at high speed
            if np.linalg.norm(self.interceptor_vel) > 5.0:  # Only if moving fast
                to_missile = self.missile_pos - self.interceptor_pos
                if np.linalg.norm(to_missile) > 0:
                    to_missile_normalized = to_missile / np.linalg.norm(to_missile)
                    vel_normalized = self.interceptor_vel / np.linalg.norm(self.interceptor_vel)
                    
                    # Reward for high speed approach
                    alignment = np.dot(vel_normalized, to_missile_normalized)
                    speed_factor = min(1.0, np.linalg.norm(self.interceptor_vel) / 30.0)
                    approach_reward = max(0, alignment) * speed_factor * 0.4
                    distance_reward += approach_reward
            
            # Penalty for being too close without intercepting (discourages following)
            if intercept_dist < 40 and self.step_count > 50:
                distance_reward -= 0.2  # Penalty for hanging around
                
            # Progress bonus (smaller than before)
            if hasattr(self, 'prev_distance'):
                if intercept_dist < self.prev_distance:
                    progress_reward = min(0.1, (self.prev_distance - intercept_dist) / 10.0)
                    distance_reward += progress_reward
                else:
                    distance_reward -= 0.05  # Penalty for moving away
            
            self.prev_distance = intercept_dist
            reward = time_penalty + fuel_efficiency + distance_reward

        self.step_count += 1
        
        # Debug: Print important terminations and progress info
        if terminated or truncated:
            if intercepted:
                intercept_type = "EXPLODED" if exploded else "CLOSE-CONTACT"
                print(f"[ENV] ✅ INTERCEPTED ({intercept_type})! Distance: {intercept_dist:.1f}, Reward: {reward:.1f}, Steps: {self.step_count}")
            elif missile_hit:
                print(f"[ENV] ❌ MISSILE HIT TARGET! Distance: {miss_dist:.1f}, Reward: {reward:.1f}, Steps: {self.step_count}")  
            elif backward_movement:
                print(f"[ENV] ⬅️ BACKWARD MOVEMENT! Reward: {reward:.1f}, Steps: {self.step_count}")
            elif altitude_loss:
                print(f"[ENV] ⬇️ ALTITUDE LOSS! Max: {self.max_altitude_reached:.1f}, Current: {self.interceptor_pos[2]:.1f}, Reward: {reward:.1f}, Steps: {self.step_count}")
            elif interceptor_above_missile:
                print(f"[ENV] ⬆️ INTERCEPTOR WENT ABOVE MISSILE! Interceptor Z: {self.interceptor_pos[2]:.1f}, Missile Z: {self.missile_pos[2]:.1f}, Reward: {reward:.1f}, Steps: {self.step_count}")
            elif out_of_fuel:
                print(f"[ENV] ⛽ OUT OF FUEL! Fuel: {self.fuel_remaining:.1f}, Reward: {reward:.1f}, Steps: {self.step_count}")
            elif out_of_bounds:
                print(f"[ENV] 🚫 OUT OF BOUNDS, Steps: {self.step_count}, Position: {self.interceptor_pos}")
            elif truncated:
                final_distance = distance(self.interceptor_pos, self.missile_pos)
                print(f"[ENV] ⏱️ TIMEOUT - Final distance: {final_distance:.1f}, Reward: {reward:.1f}, Steps: {self.step_count}")
        
        # Occasional progress updates during episodes
        elif self.step_count % 100 == 0:
            current_distance = distance(self.interceptor_pos, self.missile_pos)
            missile_to_target = distance(self.missile_pos, self.target_pos)
            fuel_pct = (self.fuel_remaining / self.max_fuel) * 100
            print(f"[ENV] Step {self.step_count}: Distance to missile: {current_distance:.1f}, Missile to target: {missile_to_target:.1f}, Fuel: {fuel_pct:.0f}%, Reward: {reward:.2f}")
                
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def show_episode_result(self, reward: float, terminated: bool, truncated: bool):
        """Show a popup message for episode results"""
        if terminated:
            if reward > 10:  # Successful intercept
                self.popup_message = "🎯 INTERCEPT SUCCESS!"
                self.popup_color = (0, 255, 0)  # Green
            elif reward < -3:  # Serious failure
                if reward < -7:  # Missile hit target
                    self.popup_message = "💥 MISSILE HIT TARGET!"
                    self.popup_color = (255, 0, 0)  # Red
                elif reward > -5:  # Went above missile or out of fuel
                    self.popup_message = "⚠️ INTERCEPTOR FAILED!"
                    self.popup_color = (255, 165, 0)  # Orange
                else:
                    self.popup_message = "❌ MISSION FAILED!"
                    self.popup_color = (255, 0, 0)  # Red
            else:
                self.popup_message = "⛔ OUT OF BOUNDS!"
                self.popup_color = (128, 128, 128)  # Gray
        elif truncated:
            self.popup_message = "⏰ TIMEOUT!"
            self.popup_color = (255, 255, 0)  # Yellow
        
        self.popup_timer = 120  # Show for 2 seconds at 60fps

    def _get_observation(self) -> np.ndarray:
        time_remaining = (self.max_steps - self.step_count) / self.max_steps
        fuel_remaining = self.fuel_remaining / self.max_fuel  # Normalize fuel to 0-1
        return np.concatenate([
            self.interceptor_pos,
            self.interceptor_vel,
            self.missile_pos,
            self.missile_vel,
            [time_remaining],
            [fuel_remaining]
        ]).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "interceptor_pos": self.interceptor_pos.copy(),
            "missile_pos": self.missile_pos.copy(),
            "distance_to_missile": distance(self.interceptor_pos, self.missile_pos),
        }

    def render(self, intercepted: bool = False):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = Viewer3D(self.world_size)
            
            # Update popup timer
            if self.popup_timer > 0:
                self.popup_timer -= 1
                if self.popup_timer == 0:
                    self.popup_message = None
            
            # Pass popup info to viewer
            popup_info = None
            if self.popup_message and self.popup_timer > 0:
                popup_info = {
                    'message': self.popup_message,
                    'color': self.popup_color,
                    'timer': self.popup_timer
                }
            
            self.viewer.render(self.interceptor_pos, self.missile_pos, self.target_pos, intercepted, popup_info)

    def close(self):
        if self.viewer:
            self.viewer.close()
