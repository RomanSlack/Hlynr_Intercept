import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time

from ..core.world import World
from ..core.missiles import Missile
from ..sensors.radar import GroundRadar, InterceptorRadar
from ..sensors.scope import RadarScope
from ..render.engine import RenderEngine


class MissileInterceptSim:
    """Clean RL-ready API for 3D missile intercept simulation"""
    
    def __init__(self, 
                 render_enabled: bool = True,
                 render_width: int = 1200,
                 render_height: int = 800,
                 dt: float = 1/60):
        
        self.dt = dt
        self.render_enabled = render_enabled
        
        # Core simulation components
        self.world = World(dt)
        self.ground_radars: List[GroundRadar] = []
        self.interceptor_radars: List[InterceptorRadar] = []
        
        # Rendering
        self.render_engine = None
        self.radar_scope = None
        if render_enabled:
            self.render_engine = RenderEngine(render_width, render_height)
            # Create radar scope after OpenGL context is established
            self.radar_scope = None  # Will be created on first render call
            
        # State tracking
        self.episode_length = 0
        self.max_episode_length = 7200  # 2 minutes at 60fps
        self.done = False
        
        # Action/observation spaces (for RL compatibility)
        self.action_space_size = 4  # [thrust, pitch, yaw, roll] for interceptor
        self.observation_space_size = 64  # Flattened radar + state observations
        
    def reset(self, scenario: str = "standard") -> Dict[str, Any]:
        """Reset simulation to initial state"""
        # Reset world
        self.world.reset(scenario)
        
        # Clear radar systems
        self.ground_radars.clear()
        self.interceptor_radars.clear()
        
        # Create ground radars from world ground sites
        for site in self.world.ground_sites:
            if site['type'] == 'radar':
                radar = GroundRadar(
                    position=site['position'],
                    max_range=80000,  # 80km range
                    scan_rate=6.0     # 6 RPM
                )
                self.ground_radars.append(radar)
                
        # Create interceptor radars
        interceptors = self.world.get_missiles_by_type("interceptor")
        for interceptor in interceptors:
            radar = InterceptorRadar(interceptor)
            self.interceptor_radars.append(radar)
            
        # Reset state
        self.episode_length = 0
        self.done = False
        
        # Clear radar scope
        if self.radar_scope:
            self.radar_scope.clear_contacts()
            
        return self.get_observation()
        
    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Step simulation forward by one time step
        
        Args:
            action: Control action [thrust_fraction, pitch_control, yaw_control, roll_control]
                   Each element in range [-1, 1]
                   
        Returns:
            observation: Current state observation
            reward: Reward signal (for RL)
            done: Episode termination flag
            info: Additional information
        """
        if self.done:
            return self.get_observation(), 0.0, True, {"error": "Episode already done"}
            
        # Apply action to interceptor (if provided)
        if action is not None:
            self._apply_action(action)
            
        # Update world simulation
        self.world.step()
        
        # Update radar systems
        current_time = self.world.time
        missiles = self.world.get_active_missiles()
        
        # Update ground radars
        for radar in self.ground_radars:
            radar.update(missiles, current_time, self.dt)
            
        # Update interceptor radars
        for radar in self.interceptor_radars:
            if radar.missile.active:
                radar.update(missiles, current_time, self.dt)
                
        # Update radar scope
        if self.radar_scope:
            self.radar_scope.update_ground_radar(self.ground_radars, current_time)
            self.radar_scope.update_interceptor_radar(self.interceptor_radars, current_time)
            
        # Check termination conditions
        self.episode_length += 1
        self.done = self._check_done()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get observation
        observation = self.get_observation()
        
        # Info dict with more debug info
        active_missiles = len(missiles)
        total_missiles = len(self.world.missiles)
        info = {
            "time": self.world.time,
            "active_missiles": active_missiles,
            "total_missiles": total_missiles,
            "episode_length": self.episode_length,
            "scenario_complete": self.world.is_scenario_complete()
        }
        
        # Debug output only at start
        if self.episode_length == 1:
            print(f"DEBUG: Episode start - Active: {active_missiles}/{total_missiles}")
            for i, missile in enumerate(self.world.missiles):
                print(f"  Missile {i}: {missile.type}, active={missile.active}, pos={missile.position}, altitude={missile.get_altitude():.1f}m")
        
        return observation, reward, self.done, info
        
    def _apply_action(self, action: np.ndarray):
        """Apply control action to interceptor missile"""
        interceptors = self.world.get_missiles_by_type("interceptor")
        if not interceptors or not interceptors[0].active:
            return
            
        interceptor = interceptors[0]
        
        # Parse action
        thrust_fraction = np.clip(action[0], 0.0, 1.0)  # Thrust must be positive
        pitch_control = np.clip(action[1], -1.0, 1.0)
        yaw_control = np.clip(action[2], -1.0, 1.0) 
        roll_control = np.clip(action[3], -1.0, 1.0)
        
        # Apply to missile
        interceptor.set_thrust_fraction(thrust_fraction)
        interceptor.set_control_inputs(pitch_control, yaw_control, roll_control)
        
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation state"""
        observation = {
            # World state
            "time": self.world.time,
            "dt": self.dt,
            
            # Missile states
            "missiles": [],
            
            # Radar observations  
            "ground_radar_contacts": [],
            "interceptor_radar_contacts": [],
            
            # Derived features
            "threat_assessment": self._assess_threats(),
            "intercept_geometry": self._calculate_intercept_geometry()
        }
        
        # Add missile states
        for missile in self.world.missiles:
            missile_obs = {
                "type": missile.type,
                "active": missile.active,
                "position": missile.position.copy(),
                "velocity": missile.velocity.copy(),
                "orientation": missile.orientation.copy(),
                "angular_velocity": missile.angular_velocity.copy(),
                "mass": missile.get_mass(),
                "fuel_remaining": missile.fuel_remaining,
                "thrust_fraction": missile.thrust_fraction,
                "speed": missile.get_speed(),
                "altitude": missile.get_altitude(),
                "range": missile.get_range()
            }
            observation["missiles"].append(missile_obs)
            
        # Add radar contacts (only observable information)
        for radar in self.ground_radars:
            for contact in radar.contacts:
                contact_obs = {
                    "bearing": contact.bearing,
                    "elevation": contact.elevation,
                    "range": contact.range,
                    "doppler_velocity": contact.doppler_velocity,
                    "snr": contact.snr,
                    "timestamp": contact.timestamp,
                    "radar_type": "ground"
                }
                observation["ground_radar_contacts"].append(contact_obs)
                
        for radar in self.interceptor_radars:
            if radar.missile.active:
                for contact in radar.contacts:
                    contact_obs = {
                        "bearing": contact.bearing,
                        "range": contact.range,
                        "doppler_velocity": contact.doppler_velocity,
                        "snr": contact.snr,
                        "timestamp": contact.timestamp,
                        "radar_type": "interceptor"
                    }
                    observation["interceptor_radar_contacts"].append(contact_obs)
                    
        return observation
        
    def get_flattened_observation(self) -> np.ndarray:
        """Get flattened observation vector for RL algorithms"""
        obs = self.get_observation()
        
        # Create fixed-size observation vector
        obs_vector = np.zeros(self.observation_space_size)
        
        idx = 0
        
        # Time and basic state (4 elements)
        obs_vector[idx:idx+2] = [obs["time"], obs["dt"]]
        idx += 2
        
        # Interceptor state (16 elements)
        interceptors = [m for m in obs["missiles"] if m["type"] == "interceptor"]
        if interceptors:
            interceptor = interceptors[0]
            obs_vector[idx:idx+3] = interceptor["position"]
            obs_vector[idx+3:idx+6] = interceptor["velocity"] 
            obs_vector[idx+6:idx+9] = interceptor["orientation"]
            obs_vector[idx+9:idx+12] = interceptor["angular_velocity"]
            obs_vector[idx+12] = interceptor["mass"] / 200.0  # Normalize
            obs_vector[idx+13] = interceptor["fuel_remaining"] / 50.0  # Normalize
            obs_vector[idx+14] = interceptor["thrust_fraction"]
            obs_vector[idx+15] = int(interceptor["active"])
        idx += 16
        
        # Closest attacker state (16 elements)
        attackers = [m for m in obs["missiles"] if m["type"] == "attacker" and m["active"]]
        if attackers:
            # Find closest attacker
            closest = min(attackers, key=lambda m: np.linalg.norm(m["position"]))
            obs_vector[idx:idx+3] = closest["position"]
            obs_vector[idx+3:idx+6] = closest["velocity"]
            obs_vector[idx+6:idx+9] = closest["orientation"] 
            obs_vector[idx+9:idx+12] = closest["angular_velocity"]
            obs_vector[idx+12] = closest["mass"] / 200.0  # Normalize
            obs_vector[idx+13] = closest["fuel_remaining"] / 50.0  # Normalize
            obs_vector[idx+14] = closest["speed"] / 2000.0  # Normalize to km/s
            obs_vector[idx+15] = int(closest["active"])
        idx += 16
        
        # Radar contacts (remaining space)
        radar_contacts = obs["ground_radar_contacts"] + obs["interceptor_radar_contacts"]
        max_contacts = (self.observation_space_size - idx) // 6
        
        for i, contact in enumerate(radar_contacts[:max_contacts]):
            base_idx = idx + i * 6
            obs_vector[base_idx:base_idx+6] = [
                contact["bearing"] / (2 * np.pi),  # Normalize bearing
                contact["elevation"] / (np.pi/2),  # Normalize elevation
                contact["range"] / 100000.0,       # Normalize range to 100km
                contact["doppler_velocity"] / 2000.0,  # Normalize velocity
                contact["snr"] / 30.0,             # Normalize SNR
                1.0 if contact["radar_type"] == "ground" else 0.0
            ]
            
        return obs_vector
        
    def _assess_threats(self) -> Dict[str, Any]:
        """Assess current threat situation"""
        attackers = [m for m in self.world.missiles if m.type == "attacker" and m.active]
        interceptors = [m for m in self.world.missiles if m.type == "interceptor" and m.active]
        
        assessment = {
            "num_active_attackers": len(attackers),
            "num_active_interceptors": len(interceptors),
            "closest_attacker_range": float('inf'),
            "time_to_impact": float('inf'),
            "intercept_possible": False
        }
        
        if attackers:
            # Find closest attacker to target
            target = self.world.target_position
            closest_attacker = min(attackers, key=lambda m: np.linalg.norm(m.position - target))
            assessment["closest_attacker_range"] = np.linalg.norm(closest_attacker.position - target)
            
            # Estimate time to impact (simplified)
            velocity_to_target = np.dot(closest_attacker.velocity, target - closest_attacker.position)
            if velocity_to_target > 0:
                assessment["time_to_impact"] = assessment["closest_attacker_range"] / velocity_to_target
                
        return assessment
        
    def _calculate_intercept_geometry(self) -> Dict[str, Any]:
        """Calculate intercept geometry parameters"""
        geometry = {
            "intercept_angle": 0.0,
            "closure_rate": 0.0,
            "lateral_separation": 0.0,
            "time_to_closest_approach": float('inf')
        }
        
        attackers = [m for m in self.world.missiles if m.type == "attacker" and m.active]
        interceptors = [m for m in self.world.missiles if m.type == "interceptor" and m.active]
        
        if attackers and interceptors:
            attacker = attackers[0]
            interceptor = interceptors[0]
            
            # Relative position and velocity
            rel_pos = attacker.position - interceptor.position
            rel_vel = attacker.velocity - interceptor.velocity
            
            # Closure rate (negative = closing)
            range_to_target = np.linalg.norm(rel_pos)
            if range_to_target > 0:
                geometry["closure_rate"] = np.dot(rel_vel, rel_pos) / range_to_target
                
                # Time to closest approach
                if geometry["closure_rate"] < 0:
                    geometry["time_to_closest_approach"] = -range_to_target / geometry["closure_rate"]
                    
        return geometry
        
    def _calculate_reward(self) -> float:
        """Calculate reward signal for RL training"""
        reward = 0.0
        
        # Basic survival reward
        reward += 0.1
        
        # Penalty for time (encourage faster intercepts)
        reward -= 0.01
        
        # Check for intercept success
        attackers = [m for m in self.world.missiles if m.type == "attacker" and m.active]
        interceptors = [m for m in self.world.missiles if m.type == "interceptor" and m.active]
        
        # Large positive reward for successful intercept
        if len(attackers) < len(self.world.get_missiles_by_type("attacker")):
            reward += 1000.0
            
        # Large negative reward if interceptor is destroyed
        if len(interceptors) == 0:
            reward -= 500.0
            
        # Reward for closing distance to threats
        if attackers and interceptors:
            attacker = attackers[0]
            interceptor = interceptors[0]
            
            distance = np.linalg.norm(attacker.position - interceptor.position)
            reward += max(0, 10000 - distance) / 10000  # Closer is better
            
        return reward
        
    def _check_done(self) -> bool:
        """Check if episode should terminate"""
        # Time limit reached
        if self.episode_length >= self.max_episode_length:
            return True
            
        # For manual control mode, don't auto-end the episode
        # Only end if interceptor is destroyed or user requests it
        interceptors = [m for m in self.world.missiles if m.type == "interceptor" and m.active]
        if len(interceptors) == 0:
            print("Interceptor destroyed!")
            return True
            
        return False
        
    def render(self):
        """Render current simulation state"""
        if not self.render_enabled or not self.render_engine:
            return
            
        # Render 3D scene
        self.render_engine.render_frame(self.world)
        
        # Create radar scope on first render (when OpenGL context is available)
        if self.radar_scope is None:
            self.radar_scope = RadarScope(
                self.render_engine.width - 320, 10, 300, 300
            )
        
        # Render radar scope overlay (simplified text-only version)
        if self.radar_scope:
            self.radar_scope.render(self.world.time)
            
    def set_manual_control(self, enabled: bool):
        """Enable/disable manual control mode"""
        # This would integrate with the ManualControls class
        pass
        
    def get_debug_info(self) -> Dict[str, Any]:
        """Get detailed debug information"""
        return {
            "world_state": self.world.get_world_state(),
            "ground_radars": [
                {
                    "position": radar.position.tolist(),
                    "current_azimuth": radar.current_azimuth,
                    "contacts": len(radar.contacts)
                }
                for radar in self.ground_radars
            ],
            "interceptor_radars": [
                {
                    "missile_active": radar.missile.active,
                    "gimbal_pointing": radar.gimbal_pointing.tolist(),
                    "contacts": len(radar.contacts)
                }
                for radar in self.interceptor_radars
            ],
            "episode_info": {
                "length": self.episode_length,
                "done": self.done,
                "max_length": self.max_episode_length
            }
        }
        
    def close(self):
        """Cleanup resources"""
        if self.render_engine:
            self.render_engine.cleanup()