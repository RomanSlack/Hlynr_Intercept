import numpy as np
from typing import List, Dict, Any, Optional
from .physics import Physics6DOF
from .missiles import Missile


class World:
    """World state manager for 3D missile simulation"""
    
    def __init__(self, dt: float = 1/60):
        self.dt = dt
        self.physics = Physics6DOF(dt)
        
        # Simulation state
        self.time = 0.0
        self.missiles: List[Missile] = []
        self.ground_sites: List[Dict] = []
        
        # Environment parameters
        self.terrain_size = 50000  # 50km x 50km terrain
        self.max_altitude = 20000  # 20km ceiling
        
        # Weather
        self.wind_velocity = np.array([0.0, 0.0, 0.0])  # m/s
        self.physics.set_wind(self.wind_velocity)
        
        # Scenario parameters  
        self.attacker_launch_site = np.array([50000, 0, 15000]) # 50km away, 15km altitude
        self.defender_launch_site = np.array([0, 0, 0])         # Origin
        self.target_position = np.array([-2000, 0, 0])          # 2km behind defender
        
    def reset(self, scenario: str = "standard"):
        """Reset world to initial scenario state"""
        self.time = 0.0
        self.missiles.clear()
        self.ground_sites.clear()
        
        if scenario == "standard":
            self._setup_standard_scenario()
        elif scenario == "multiple_attackers":
            self._setup_multiple_attackers_scenario()
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
            
    def _setup_standard_scenario(self):
        """Setup standard scenario - start with interceptor only"""
        # Only spawn interceptor initially - attacker will be spawned manually
        interceptor = self.add_missile(
            missile_type="interceptor", 
            position=self.defender_launch_site.copy() + np.array([0, 0, 100]),  # 100m above ground
            velocity=np.array([0, 0, 50]),  # Start with upward velocity
            orientation=np.array([0, np.radians(60), 0])  # Pointing steeply up
        )
        
        # Give it some initial thrust to stay airborne
        interceptor.set_thrust_fraction(0.3)  # 30% thrust to maintain altitude
        
        # Ground radar sites
        self.add_ground_site("radar", np.array([0, 0, 10]))      # Main site
        self.add_ground_site("radar", np.array([5000, 0, 10]))   # Auxiliary site
        self.add_ground_site("radar", np.array([-5000, 0, 10]))  # Auxiliary site
        
    def _setup_multiple_attackers_scenario(self):
        """Setup scenario with multiple incoming missiles"""
        for i in range(3):
            offset = np.array([0, (i-1) * 5000, 0])  # Spread attackers
            launch_site = self.attacker_launch_site + offset
            
            velocity = self._calculate_ballistic_velocity(
                launch_site, self.target_position, flight_time=50.0 + i*5
            )
            
            self.add_missile(
                missile_type="attacker",
                position=launch_site,
                velocity=velocity,
                orientation=np.array([0, np.radians(-35), np.radians(180)])
            )
            
        # Single interceptor
        self.add_missile(
            missile_type="interceptor",
            position=self.defender_launch_site.copy() + np.array([0, 0, 2]),
            velocity=np.array([0, 0, 0]),
            orientation=np.array([0, np.radians(60), 0])
        )
        
        # Enhanced radar network
        radar_positions = [
            np.array([0, 0, 15]),
            np.array([3000, 3000, 12]),
            np.array([-3000, 3000, 12]),
            np.array([3000, -3000, 12]),
            np.array([-3000, -3000, 12])
        ]
        
        for pos in radar_positions:
            self.add_ground_site("radar", pos)
            
    def _calculate_ballistic_velocity(self, start: np.ndarray, target: np.ndarray, 
                                    flight_time: float) -> np.ndarray:
        """Calculate initial velocity for ballistic trajectory"""
        delta_pos = target - start
        horizontal_distance = np.linalg.norm(delta_pos[:2])
        height_diff = delta_pos[2]
        
        # Simple ballistic calculation (assumes no air resistance for initial velocity)
        v_horizontal = horizontal_distance / flight_time
        v_vertical = height_diff / flight_time + 0.5 * self.physics.g * flight_time
        
        # Direction vector
        horizontal_dir = delta_pos[:2] / horizontal_distance if horizontal_distance > 0 else np.array([1, 0])
        
        velocity = np.array([
            horizontal_dir[0] * v_horizontal,
            horizontal_dir[1] * v_horizontal, 
            v_vertical
        ])
        
        return velocity
        
    def spawn_attacker_missile(self):
        """Spawn attacker missile with realistic ballistic trajectory"""
        # Calculate ballistic trajectory for 90 second flight time
        attacker_velocity = self._calculate_ballistic_velocity(
            self.attacker_launch_site, self.target_position, flight_time=90.0
        )
        
        # Calculate pitch angle for initial trajectory
        speed = np.linalg.norm(attacker_velocity)
        pitch_angle = np.arcsin(attacker_velocity[2] / speed) if speed > 0 else 0
        yaw_angle = np.arctan2(attacker_velocity[1], attacker_velocity[0])
        
        attacker = self.add_missile(
            missile_type="attacker",
            position=self.attacker_launch_site.copy(),
            velocity=attacker_velocity,
            orientation=np.array([0, pitch_angle, yaw_angle])
        )
        
        print(f"Attacker spawned at {self.attacker_launch_site} with velocity {attacker_velocity}")
        print(f"Flight time to target: ~90 seconds")
        return attacker
        
    def add_missile(self, missile_type: str, position: np.ndarray, 
                   velocity: np.ndarray, orientation: np.ndarray) -> Missile:
        """Add new missile to world"""
        missile = Missile(missile_type, position, velocity, orientation, self.physics)
        self.missiles.append(missile)
        return missile
        
    def add_ground_site(self, site_type: str, position: np.ndarray):
        """Add ground installation (radar, launcher, etc.)"""
        site = {
            "type": site_type,
            "position": position.copy(),
            "active": True
        }
        self.ground_sites.append(site)
        
    def step(self):
        """Advance simulation by one time step"""
        # Update all missiles
        for missile in self.missiles:
            if missile.active:
                missile.update(self.dt)
                
        # Update simulation time
        self.time += self.dt
        
        # Check for intercepts
        self._check_intercepts()
        
    def _check_intercepts(self):
        """Check for missile-to-missile intercepts"""
        attackers = [m for m in self.missiles if m.type == "attacker" and m.active]
        interceptors = [m for m in self.missiles if m.type == "interceptor" and m.active]
        
        for attacker in attackers:
            for interceptor in interceptors:
                distance = np.linalg.norm(attacker.position - interceptor.position)
                
                # Intercept threshold (combined missile lengths plus margin)
                intercept_distance = (attacker.length + interceptor.length) / 2 + 5.0  # 5m margin
                
                if distance < intercept_distance:
                    attacker.active = False
                    interceptor.active = False
                    # Could add explosion effects here
                    
    def get_active_missiles(self) -> List[Missile]:
        """Get list of active missiles"""
        return [m for m in self.missiles if m.active]
        
    def get_missiles_by_type(self, missile_type: str) -> List[Missile]:
        """Get missiles of specific type"""
        return [m for m in self.missiles if m.type == missile_type]
        
    def set_wind(self, wind_velocity: np.ndarray):
        """Set wind velocity for all physics calculations"""
        self.wind_velocity = wind_velocity.copy()
        self.physics.set_wind(wind_velocity)
        
    def get_world_state(self) -> Dict[str, Any]:
        """Get complete world state"""
        return {
            "time": self.time,
            "missiles": [m.get_state_dict() for m in self.missiles],
            "ground_sites": self.ground_sites.copy(),
            "wind_velocity": self.wind_velocity.copy(),
            "active_missiles": len(self.get_active_missiles())
        }
        
    def is_scenario_complete(self) -> bool:
        """Check if scenario has ended (all missiles inactive or time limit)"""
        active_count = len(self.get_active_missiles())
        time_limit_reached = self.time > 120.0  # 2 minute limit
        
        return active_count == 0 or time_limit_reached
        
    def get_bounds(self) -> Dict[str, float]:
        """Get world coordinate bounds"""
        return {
            "x_min": -self.terrain_size // 2,
            "x_max": self.terrain_size // 2,
            "y_min": -self.terrain_size // 2, 
            "y_max": self.terrain_size // 2,
            "z_min": 0,
            "z_max": self.max_altitude
        }