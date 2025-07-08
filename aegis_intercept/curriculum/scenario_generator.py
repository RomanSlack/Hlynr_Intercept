"""
Scenario Generator for AegisIntercept Phase 3.

This module generates random scenarios within curriculum tier bounds,
providing varied training experiences while maintaining difficulty consistency.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class ScenarioGenerator:
    """
    Generates random scenarios for training within curriculum constraints.
    
    This class creates varied initial conditions, environmental parameters,
    and mission configurations while ensuring they remain within the bounds
    of the current curriculum tier.
    """
    
    def __init__(self, tier_config: Dict[str, Any], seed: Optional[int] = None):
        """
        Initialize scenario generator.
        
        Args:
            tier_config: Configuration for current curriculum tier
            seed: Random seed for reproducibility
        """
        self.tier_config = tier_config
        self.rng = np.random.RandomState(seed)
        
        # Scenario variation parameters
        self.variation_params = {
            'position_noise': 0.1,      # Position variation as fraction of spawn separation
            'velocity_noise': 0.2,      # Velocity variation as fraction of base speed
            'altitude_range': [500, 2000],  # Altitude variation in meters
            'wind_variation': 0.3,      # Wind parameter variation
            'fuel_variation': 0.1,      # Fuel capacity variation
            'mass_variation': 0.1       # Mass variation
        }
        
        # Predefined scenario templates
        self.scenario_templates = self._create_scenario_templates()
    
    def _create_scenario_templates(self) -> List[Dict[str, Any]]:
        """Create predefined scenario templates for variety."""
        templates = [
            {
                'name': 'head_on',
                'description': 'Head-on engagement',
                'interceptor_angle_bias': 0.0,
                'adversary_approach_angle': 0.0,
                'relative_altitude': 0.0,
                'wind_bias': [0.0, 0.0, 0.0]
            },
            {
                'name': 'side_attack',
                'description': 'Side attack geometry',
                'interceptor_angle_bias': np.pi/2,
                'adversary_approach_angle': 0.0,
                'relative_altitude': 200.0,
                'wind_bias': [0.0, 0.0, 0.0]
            },
            {
                'name': 'rear_chase',
                'description': 'Rear chase engagement',
                'interceptor_angle_bias': np.pi,
                'adversary_approach_angle': 0.0,
                'relative_altitude': -100.0,
                'wind_bias': [0.0, 0.0, 0.0]
            },
            {
                'name': 'high_altitude',
                'description': 'High altitude engagement',
                'interceptor_angle_bias': None,  # Random
                'adversary_approach_angle': None,  # Random
                'relative_altitude': 1000.0,
                'wind_bias': [0.0, 0.0, 0.0]
            },
            {
                'name': 'crosswind',
                'description': 'Strong crosswind conditions',
                'interceptor_angle_bias': None,
                'adversary_approach_angle': None,
                'relative_altitude': 0.0,
                'wind_bias': [0.0, 15.0, 0.0]
            },
            {
                'name': 'vertical_intercept',
                'description': 'Vertical interception geometry',
                'interceptor_angle_bias': None,
                'adversary_approach_angle': -np.pi/4,
                'relative_altitude': 500.0,
                'wind_bias': [0.0, 0.0, 0.0]
            }
        ]
        
        return templates
    
    def generate_scenario(self, template_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a random scenario within tier constraints.
        
        Args:
            template_name: Optional specific template to use
            
        Returns:
            Dictionary containing scenario parameters
        """
        # Select template
        if template_name:
            template = next((t for t in self.scenario_templates if t['name'] == template_name), None)
            if not template:
                template = self.rng.choice(self.scenario_templates)
        else:
            template = self.rng.choice(self.scenario_templates)
        
        # Generate scenario parameters
        scenario = {
            'template': template,
            'initial_conditions': self._generate_initial_conditions(template),
            'environment': self._generate_environment_config(template),
            'adversary_config': self._generate_adversary_config(),
            'mission_config': self._generate_mission_config()
        }
        
        return scenario
    
    def _generate_initial_conditions(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial positions and velocities."""
        # Base spawn separation from tier config
        base_separation = self.tier_config['spawn_separation']
        
        # Add variation
        separation_noise = self.variation_params['position_noise']
        actual_separation = base_separation * (1.0 + self.rng.uniform(-separation_noise, separation_noise))
        
        # Target position (always at origin for now)
        target_pos = np.array([0.0, 0.0, 0.0])
        
        # Adversary position
        if template['adversary_approach_angle'] is not None:
            adversary_angle = template['adversary_approach_angle']
        else:
            adversary_angle = self.rng.uniform(0, 2 * np.pi)
        
        adversary_elevation = self.rng.uniform(-np.pi/6, np.pi/6)
        adversary_distance = actual_separation * self.rng.uniform(0.8, 1.2)
        
        adversary_pos = np.array([
            adversary_distance * np.cos(adversary_angle) * np.cos(adversary_elevation),
            adversary_distance * np.sin(adversary_angle) * np.cos(adversary_elevation),
            adversary_distance * np.sin(adversary_elevation) + 
            self.rng.uniform(*self.variation_params['altitude_range'])
        ])
        
        # Adversary velocity
        speed_range = self.tier_config['adversary_speed_range']
        adversary_speed = self.rng.uniform(speed_range[0], speed_range[1])
        
        target_direction = target_pos - adversary_pos
        target_direction = target_direction / np.linalg.norm(target_direction)
        
        # Add some randomness to adversary velocity direction
        noise_magnitude = 0.1
        velocity_noise = self.rng.normal(0, noise_magnitude, 3)
        adversary_vel_direction = target_direction + velocity_noise
        adversary_vel_direction = adversary_vel_direction / np.linalg.norm(adversary_vel_direction)
        
        adversary_vel = adversary_speed * adversary_vel_direction
        
        # Interceptor position
        if template['interceptor_angle_bias'] is not None:
            interceptor_angle = template['interceptor_angle_bias']
        else:
            interceptor_angle = self.rng.uniform(0, 2 * np.pi)
        
        interceptor_distance = actual_separation * self.rng.uniform(0.6, 0.9)
        interceptor_altitude = (adversary_pos[2] + 
                              template['relative_altitude'] + 
                              self.rng.uniform(-100, 100))
        
        interceptor_pos = np.array([
            interceptor_distance * np.cos(interceptor_angle),
            interceptor_distance * np.sin(interceptor_angle),
            max(100, interceptor_altitude)  # Ensure minimum altitude
        ])
        
        # Interceptor velocity
        intercept_direction = adversary_pos - interceptor_pos
        intercept_direction = intercept_direction / np.linalg.norm(intercept_direction)
        
        interceptor_speed = self.rng.uniform(100, 200)
        interceptor_vel = interceptor_speed * intercept_direction
        
        # Add velocity noise
        vel_noise = self.variation_params['velocity_noise']
        interceptor_vel += self.rng.normal(0, interceptor_speed * vel_noise, 3)
        adversary_vel += self.rng.normal(0, adversary_speed * vel_noise, 3)
        
        return {
            'target_position': target_pos,
            'interceptor_position': interceptor_pos,
            'interceptor_velocity': interceptor_vel,
            'adversary_position': adversary_pos,
            'adversary_velocity': adversary_vel,
            'separation_distance': actual_separation
        }
    
    def _generate_environment_config(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate environment configuration."""
        # Base wind from tier config
        base_wind_severity = self.tier_config['wind_severity']
        
        # Apply template wind bias
        wind_bias = np.array(template['wind_bias'])
        wind_variation = self.variation_params['wind_variation']
        
        # Generate wind configuration
        wind_config = {
            'steady_wind': [
                wind_bias[0] + self.rng.uniform(-10, 10) * base_wind_severity,
                wind_bias[1] + self.rng.uniform(-10, 10) * base_wind_severity,
                wind_bias[2] + self.rng.uniform(-5, 5) * base_wind_severity
            ],
            'turbulence_intensity': base_wind_severity * self.rng.uniform(0.5, 1.5),
            'gust_amplitude': 10.0 * base_wind_severity * self.rng.uniform(0.5, 2.0),
            'gust_frequency': 0.1 * self.rng.uniform(0.5, 2.0),
            'wind_profile': self.rng.choice(['uniform', 'logarithmic', 'power_law']),
            'wind_shear_enabled': self.rng.random() < base_wind_severity
        }
        
        # Atmospheric conditions
        atmospheric_config = {
            'density_variation': self.rng.uniform(-0.1, 0.1),
            'temperature_offset': self.rng.uniform(-10, 10),  # Celsius
            'pressure_variation': self.rng.uniform(-0.05, 0.05)
        }
        
        return {
            'wind_config': wind_config,
            'atmospheric_config': atmospheric_config,
            'visibility': self.rng.uniform(5000, 50000),  # meters
            'cloud_cover': self.rng.uniform(0, 1)
        }
    
    def _generate_adversary_config(self) -> Dict[str, Any]:
        """Generate adversary configuration."""
        base_aggressiveness = self.tier_config['adversary_evasion_aggressiveness']
        
        # Add variation to evasion parameters
        evasion_config = {
            'aggressiveness': base_aggressiveness * self.rng.uniform(0.7, 1.3),
            'evasion_threshold': 300.0 * self.rng.uniform(0.8, 1.2),
            'jink_frequency': 0.5 * self.rng.uniform(0.5, 2.0),
            'spiral_rate': 0.2 * self.rng.uniform(0.5, 1.5),
            'barrel_roll_rate': 0.1 * self.rng.uniform(0.3, 1.0)
        }
        
        # Physical parameters
        mass_variation = self.variation_params['mass_variation']
        base_mass = 200.0
        
        physical_config = {
            'mass': base_mass * (1.0 + self.rng.uniform(-mass_variation, mass_variation)),
            'thrust_capacity': self.rng.uniform(2000, 5000),
            'drag_coefficient': self.rng.uniform(0.4, 0.8),
            'reference_area': self.rng.uniform(0.02, 0.05)
        }
        
        return {
            'evasion_config': evasion_config,
            'physical_config': physical_config,
            'guidance_noise': self.rng.uniform(0.0, 0.1)
        }
    
    def _generate_mission_config(self) -> Dict[str, Any]:
        """Generate mission-specific configuration."""
        # Fuel parameters
        base_fuel = self.tier_config.get('fuel_capacity', 1.0)
        fuel_variation = self.variation_params['fuel_variation']
        
        fuel_config = {
            'initial_fuel': base_fuel * (1.0 + self.rng.uniform(-fuel_variation, fuel_variation)),
            'consumption_rate': 0.001 * self.rng.uniform(0.8, 1.2),
            'thrust_efficiency': self.rng.uniform(0.8, 1.2)
        }
        
        # Mission constraints
        mission_constraints = {
            'max_episode_steps': self.tier_config['max_episode_steps'],
            'kill_distance': self.tier_config['kill_distance'],
            'max_range': self.tier_config['spawn_separation'] * 2.0,
            'min_altitude': 50.0,
            'max_altitude': 5000.0
        }
        
        # Success criteria
        success_criteria = {
            'primary_objective': 'intercept_adversary',
            'secondary_objectives': [
                'minimize_fuel_consumption',
                'minimize_time_to_intercept',
                'maintain_flight_envelope'
            ],
            'bonus_conditions': [
                'perfect_intercept',  # Within 1m
                'efficient_fuel_use',  # <50% fuel used
                'quick_intercept'     # <50% max time
            ]
        }
        
        return {
            'fuel_config': fuel_config,
            'mission_constraints': mission_constraints,
            'success_criteria': success_criteria,
            'scenario_id': self.rng.randint(0, 999999)
        }
    
    def generate_scenario_batch(self, batch_size: int, 
                              template_distribution: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Generate a batch of scenarios.
        
        Args:
            batch_size: Number of scenarios to generate
            template_distribution: Optional distribution of templates to use
            
        Returns:
            List of scenario dictionaries
        """
        scenarios = []
        
        for _ in range(batch_size):
            # Select template based on distribution
            if template_distribution:
                template_names = list(template_distribution.keys())
                template_probs = list(template_distribution.values())
                template_name = self.rng.choice(template_names, p=template_probs)
            else:
                template_name = None
            
            scenario = self.generate_scenario(template_name)
            scenarios.append(scenario)
        
        return scenarios
    
    def save_scenario(self, scenario: Dict[str, Any], filename: str):
        """Save scenario to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            else:
                return obj
        
        scenario_json = convert_arrays(scenario)
        
        try:
            with open(filename, 'w') as f:
                json.dump(scenario_json, f, indent=2)
        except Exception as e:
            print(f"Error saving scenario: {e}")
    
    def load_scenario(self, filename: str) -> Dict[str, Any]:
        """Load scenario from JSON file."""
        try:
            with open(filename, 'r') as f:
                scenario_json = json.load(f)
            
            # Convert lists back to numpy arrays
            def convert_lists(obj):
                if isinstance(obj, dict):
                    return {k: convert_lists(v) for k, v in obj.items()}
                elif isinstance(obj, list) and len(obj) == 3 and all(isinstance(x, (int, float)) for x in obj):
                    return np.array(obj)
                elif isinstance(obj, list):
                    return [convert_lists(item) for item in obj]
                else:
                    return obj
            
            return convert_lists(scenario_json)
            
        except Exception as e:
            print(f"Error loading scenario: {e}")
            return {}
    
    def get_scenario_statistics(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics for a batch of scenarios."""
        if not scenarios:
            return {}
        
        # Extract statistics
        template_counts = {}
        separations = []
        adversary_speeds = []
        wind_severities = []
        
        for scenario in scenarios:
            template_name = scenario['template']['name']
            template_counts[template_name] = template_counts.get(template_name, 0) + 1
            
            separations.append(scenario['initial_conditions']['separation_distance'])
            
            adv_vel = scenario['initial_conditions']['adversary_velocity']
            adversary_speeds.append(np.linalg.norm(adv_vel))
            
            wind_config = scenario['environment']['wind_config']
            wind_severities.append(wind_config['turbulence_intensity'])
        
        return {
            'template_distribution': template_counts,
            'separation_stats': {
                'mean': np.mean(separations),
                'std': np.std(separations),
                'min': np.min(separations),
                'max': np.max(separations)
            },
            'adversary_speed_stats': {
                'mean': np.mean(adversary_speeds),
                'std': np.std(adversary_speeds),
                'min': np.min(adversary_speeds),
                'max': np.max(adversary_speeds)
            },
            'wind_severity_stats': {
                'mean': np.mean(wind_severities),
                'std': np.std(wind_severities),
                'min': np.min(wind_severities),
                'max': np.max(wind_severities)
            }
        }