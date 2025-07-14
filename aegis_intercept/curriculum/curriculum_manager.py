"""
Curriculum Learning System for AegisIntercept Phase 3

This module implements a comprehensive curriculum learning framework that
progressively trains the agent from simple 3DOF scenarios to complex 6DOF
intercept missions. It supports:

- JSON-based scenario configuration
- Dynamic difficulty adjustment based on performance
- Phase progression logic (3DOF → simplified 6DOF → full 6DOF)
- Teacher-student curriculum learning
- Performance-based advancement criteria
- Automatic scenario generation

Author: Coder Agent
Date: Phase 3 Implementation
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import logging

from ..envs.aegis_6dof_env import DifficultyMode, ActionMode


class CurriculumPhase(Enum):
    """Curriculum learning phases"""
    PHASE_1_BASIC_3DOF = "phase_1_basic_3dof"
    PHASE_2_ADVANCED_3DOF = "phase_2_advanced_3dof"
    PHASE_3_SIMPLIFIED_6DOF = "phase_3_simplified_6dof"
    PHASE_4_FULL_6DOF = "phase_4_full_6dof"
    PHASE_5_EXPERT_6DOF = "phase_5_expert_6dof"


class AdvancementCriteria(Enum):
    """Criteria for advancing to next phase"""
    EPISODE_SUCCESS_RATE = "episode_success_rate"
    AVERAGE_REWARD = "average_reward"
    TIME_TO_INTERCEPT = "time_to_intercept"
    FUEL_EFFICIENCY = "fuel_efficiency"
    CONSECUTIVE_SUCCESSES = "consecutive_successes"
    COMBINED_METRICS = "combined_metrics"


@dataclass
class PhaseConfig:
    """Configuration for a curriculum phase"""
    phase: CurriculumPhase
    difficulty_mode: DifficultyMode
    action_mode: ActionMode
    
    # Environment parameters
    world_size: float = 300.0
    max_steps: int = 300
    dt: float = 0.05
    
    # Mission parameters
    intercept_threshold: float = 30.0
    miss_threshold: float = 10.0
    explosion_radius: float = 50.0
    
    # Physics parameters
    enable_wind: bool = True
    enable_atmosphere: bool = True
    wind_strength: float = 1.0
    
    # Training parameters
    episodes_required: int = 1000
    success_rate_threshold: float = 0.75
    average_reward_threshold: float = 10.0
    
    # Advancement criteria
    advancement_criteria: List[AdvancementCriteria] = None
    criteria_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.advancement_criteria is None:
            self.advancement_criteria = [AdvancementCriteria.EPISODE_SUCCESS_RATE]
        if self.criteria_weights is None:
            self.criteria_weights = {"episode_success_rate": 1.0}


@dataclass
class ScenarioConfig:
    """Configuration for a specific training scenario"""
    name: str
    description: str
    
    # Initial conditions
    interceptor_position_range: Dict[str, Tuple[float, float]]
    interceptor_velocity_range: Dict[str, Tuple[float, float]]
    missile_position_range: Dict[str, Tuple[float, float]]
    missile_velocity_range: Dict[str, Tuple[float, float]]
    target_position_range: Dict[str, Tuple[float, float]]
    
    # Environmental conditions
    wind_conditions: Dict[str, Any]
    atmospheric_conditions: Dict[str, Any]
    
    # Mission parameters
    time_of_day: str = "day"  # "day", "night", "dawn", "dusk"
    weather: str = "clear"    # "clear", "cloudy", "stormy"
    
    # Difficulty modifiers
    evasion_aggressiveness: float = 1.0
    sensor_noise: float = 0.0
    communication_delay: float = 0.0
    
    # Success criteria
    mission_success_reward: float = 20.0
    mission_failure_penalty: float = -10.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for curriculum advancement"""
    episodes_completed: int = 0
    episodes_successful: int = 0
    total_reward: float = 0.0
    total_fuel_used: float = 0.0
    total_time_to_intercept: float = 0.0
    
    # Recent performance (for stability)
    recent_success_rate: float = 0.0
    recent_average_reward: float = 0.0
    recent_episodes: int = 100
    
    # Tracking arrays
    episode_rewards: List[float] = None
    episode_success: List[bool] = None
    episode_times: List[float] = None
    
    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.episode_success is None:
            self.episode_success = []
        if self.episode_times is None:
            self.episode_times = []
    
    @property
    def success_rate(self) -> float:
        """Overall success rate"""
        if self.episodes_completed == 0:
            return 0.0
        return self.episodes_successful / self.episodes_completed
    
    @property
    def average_reward(self) -> float:
        """Average reward per episode"""
        if self.episodes_completed == 0:
            return 0.0
        return self.total_reward / self.episodes_completed
    
    @property
    def average_fuel_efficiency(self) -> float:
        """Average fuel efficiency (fuel used per successful intercept)"""
        if self.episodes_successful == 0:
            return 0.0
        return self.total_fuel_used / self.episodes_successful
    
    @property
    def average_intercept_time(self) -> float:
        """Average time to intercept for successful episodes"""
        if self.episodes_successful == 0:
            return 0.0
        return self.total_time_to_intercept / self.episodes_successful
    
    def update(self, reward: float, success: bool, fuel_used: float, intercept_time: float):
        """Update metrics with new episode data"""
        self.episodes_completed += 1
        self.total_reward += reward
        self.total_fuel_used += fuel_used
        
        if success:
            self.episodes_successful += 1
            self.total_time_to_intercept += intercept_time
        
        # Update tracking arrays
        self.episode_rewards.append(reward)
        self.episode_success.append(success)
        self.episode_times.append(intercept_time)
        
        # Keep only recent episodes for stability calculation
        if len(self.episode_rewards) > self.recent_episodes:
            self.episode_rewards = self.episode_rewards[-self.recent_episodes:]
            self.episode_success = self.episode_success[-self.recent_episodes:]
            self.episode_times = self.episode_times[-self.recent_episodes:]
        
        # Update recent performance metrics
        self.recent_success_rate = sum(self.episode_success) / len(self.episode_success)
        self.recent_average_reward = sum(self.episode_rewards) / len(self.episode_rewards)


class CurriculumManager:
    """Main curriculum learning manager"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 enable_dynamic_difficulty: bool = True,
                 enable_scenario_generation: bool = True,
                 log_file: Optional[str] = None):
        """
        Initialize curriculum manager
        
        Args:
            config_path: Path to curriculum configuration file
            enable_dynamic_difficulty: Whether to dynamically adjust difficulty
            enable_scenario_generation: Whether to generate new scenarios
            log_file: Path to log file for curriculum progress
        """
        self.enable_dynamic_difficulty = enable_dynamic_difficulty
        self.enable_scenario_generation = enable_scenario_generation
        
        # Setup logging
        self.logger = logging.getLogger("CurriculumManager")
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize phase configurations
        self.phase_configs = self._initialize_default_phases()
        
        # Load custom configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_curriculum_config(config_path)
        
        # Current state
        self.current_phase = CurriculumPhase.PHASE_1_BASIC_3DOF
        self.current_scenario = 0
        self.phase_metrics = {phase: PerformanceMetrics() for phase in CurriculumPhase}
        
        # Scenario management
        self.scenarios = self._initialize_default_scenarios()
        
        # Dynamic difficulty adjustment
        self.difficulty_adjustment_history = []
        self.last_adjustment_time = time.time()
        
        self.logger.info("Curriculum Manager initialized")
    
    def _initialize_default_phases(self) -> Dict[CurriculumPhase, PhaseConfig]:
        """Initialize default phase configurations"""
        return {
            CurriculumPhase.PHASE_1_BASIC_3DOF: PhaseConfig(
                phase=CurriculumPhase.PHASE_1_BASIC_3DOF,
                difficulty_mode=DifficultyMode.EASY_3DOF,
                action_mode=ActionMode.ACCELERATION_3DOF,
                world_size=300.0,
                max_steps=250,
                intercept_threshold=35.0,
                enable_wind=False,
                episodes_required=500,
                success_rate_threshold=0.70,
                average_reward_threshold=8.0
            ),
            
            CurriculumPhase.PHASE_2_ADVANCED_3DOF: PhaseConfig(
                phase=CurriculumPhase.PHASE_2_ADVANCED_3DOF,
                difficulty_mode=DifficultyMode.MEDIUM_3DOF,
                action_mode=ActionMode.ACCELERATION_3DOF,
                world_size=400.0,
                max_steps=300,
                intercept_threshold=30.0,
                enable_wind=True,
                wind_strength=0.5,
                episodes_required=750,
                success_rate_threshold=0.75,
                average_reward_threshold=10.0
            ),
            
            CurriculumPhase.PHASE_3_SIMPLIFIED_6DOF: PhaseConfig(
                phase=CurriculumPhase.PHASE_3_SIMPLIFIED_6DOF,
                difficulty_mode=DifficultyMode.SIMPLIFIED_6DOF,
                action_mode=ActionMode.THRUST_ATTITUDE,
                world_size=500.0,
                max_steps=350,
                intercept_threshold=25.0,
                enable_wind=True,
                wind_strength=0.8,
                episodes_required=1000,
                success_rate_threshold=0.65,
                average_reward_threshold=12.0
            ),
            
            CurriculumPhase.PHASE_4_FULL_6DOF: PhaseConfig(
                phase=CurriculumPhase.PHASE_4_FULL_6DOF,
                difficulty_mode=DifficultyMode.FULL_6DOF,
                action_mode=ActionMode.ACCELERATION_6DOF,
                world_size=600.0,
                max_steps=400,
                intercept_threshold=20.0,
                enable_wind=True,
                wind_strength=1.0,
                episodes_required=1500,
                success_rate_threshold=0.70,
                average_reward_threshold=15.0
            ),
            
            CurriculumPhase.PHASE_5_EXPERT_6DOF: PhaseConfig(
                phase=CurriculumPhase.PHASE_5_EXPERT_6DOF,
                difficulty_mode=DifficultyMode.EXPERT_6DOF,
                action_mode=ActionMode.ACCELERATION_6DOF,
                world_size=800.0,
                max_steps=500,
                intercept_threshold=15.0,
                enable_wind=True,
                wind_strength=1.2,
                episodes_required=2000,
                success_rate_threshold=0.80,
                average_reward_threshold=18.0
            )
        }
    
    def _initialize_default_scenarios(self) -> Dict[CurriculumPhase, List[ScenarioConfig]]:
        """Initialize default scenario configurations for each phase"""
        scenarios = {}
        
        # Phase 1: Basic 3DOF scenarios
        scenarios[CurriculumPhase.PHASE_1_BASIC_3DOF] = [
            ScenarioConfig(
                name="basic_intercept",
                description="Simple head-on intercept with slow missile",
                interceptor_position_range={"x": (250, 350), "y": (250, 350), "z": (0, 10)},
                interceptor_velocity_range={"x": (-5, 5), "y": (-5, 5), "z": (5, 15)},
                missile_position_range={"x": (100, 500), "y": (100, 500), "z": (100, 300)},
                missile_velocity_range={"x": (-30, 30), "y": (-30, 30), "z": (-10, 0)},
                target_position_range={"x": (290, 310), "y": (290, 310), "z": (0, 5)},
                wind_conditions={"enabled": False},
                atmospheric_conditions={"standard": True},
                evasion_aggressiveness=0.3
            ),
            ScenarioConfig(
                name="angled_approach",
                description="Missile approaching from various angles",
                interceptor_position_range={"x": (280, 320), "y": (280, 320), "z": (0, 20)},
                interceptor_velocity_range={"x": (-10, 10), "y": (-10, 10), "z": (8, 20)},
                missile_position_range={"x": (50, 550), "y": (50, 550), "z": (150, 400)},
                missile_velocity_range={"x": (-35, 35), "y": (-35, 35), "z": (-15, 5)},
                target_position_range={"x": (290, 310), "y": (290, 310), "z": (0, 5)},
                wind_conditions={"enabled": False},
                atmospheric_conditions={"standard": True},
                evasion_aggressiveness=0.5
            )
        ]
        
        # Phase 2: Advanced 3DOF with evasion
        scenarios[CurriculumPhase.PHASE_2_ADVANCED_3DOF] = [
            ScenarioConfig(
                name="evasive_missile",
                description="Missile with moderate evasive maneuvers",
                interceptor_position_range={"x": (300, 400), "y": (300, 400), "z": (0, 30)},
                interceptor_velocity_range={"x": (-15, 15), "y": (-15, 15), "z": (10, 25)},
                missile_position_range={"x": (0, 600), "y": (0, 600), "z": (200, 500)},
                missile_velocity_range={"x": (-40, 40), "y": (-40, 40), "z": (-20, 0)},
                target_position_range={"x": (340, 360), "y": (340, 360), "z": (0, 10)},
                wind_conditions={"enabled": True, "strength": 0.5},
                atmospheric_conditions={"standard": True},
                evasion_aggressiveness=0.8
            )
        ]
        
        # Phase 3: Simplified 6DOF
        scenarios[CurriculumPhase.PHASE_3_SIMPLIFIED_6DOF] = [
            ScenarioConfig(
                name="attitude_control",
                description="6DOF intercept with attitude control focus",
                interceptor_position_range={"x": (400, 600), "y": (400, 600), "z": (0, 50)},
                interceptor_velocity_range={"x": (-20, 20), "y": (-20, 20), "z": (15, 35)},
                missile_position_range={"x": (0, 800), "y": (0, 800), "z": (300, 700)},
                missile_velocity_range={"x": (-50, 50), "y": (-50, 50), "z": (-30, 0)},
                target_position_range={"x": (490, 510), "y": (490, 510), "z": (0, 20)},
                wind_conditions={"enabled": True, "strength": 0.8},
                atmospheric_conditions={"variable": True},
                evasion_aggressiveness=1.0
            )
        ]
        
        # Phase 4: Full 6DOF
        scenarios[CurriculumPhase.PHASE_4_FULL_6DOF] = [
            ScenarioConfig(
                name="full_6dof_intercept",
                description="Complete 6DOF intercept with all physics",
                interceptor_position_range={"x": (500, 700), "y": (500, 700), "z": (0, 100)},
                interceptor_velocity_range={"x": (-30, 30), "y": (-30, 30), "z": (20, 50)},
                missile_position_range={"x": (0, 1000), "y": (0, 1000), "z": (400, 900)},
                missile_velocity_range={"x": (-60, 60), "y": (-60, 60), "z": (-40, 0)},
                target_position_range={"x": (590, 610), "y": (590, 610), "z": (0, 30)},
                wind_conditions={"enabled": True, "strength": 1.0, "turbulence": True},
                atmospheric_conditions={"variable": True, "altitude_effects": True},
                evasion_aggressiveness=1.2
            )
        ]
        
        # Phase 5: Expert 6DOF
        scenarios[CurriculumPhase.PHASE_5_EXPERT_6DOF] = [
            ScenarioConfig(
                name="expert_challenge",
                description="Highly evasive missile with complex environment",
                interceptor_position_range={"x": (600, 1000), "y": (600, 1000), "z": (0, 200)},
                interceptor_velocity_range={"x": (-50, 50), "y": (-50, 50), "z": (30, 80)},
                missile_position_range={"x": (0, 1200), "y": (0, 1200), "z": (500, 1200)},
                missile_velocity_range={"x": (-80, 80), "y": (-80, 80), "z": (-60, 0)},
                target_position_range={"x": (790, 810), "y": (790, 810), "z": (0, 50)},
                wind_conditions={"enabled": True, "strength": 1.2, "turbulence": True, "shear": True},
                atmospheric_conditions={"variable": True, "altitude_effects": True, "weather": "adverse"},
                evasion_aggressiveness=1.5
            )
        ]
        
        return scenarios
    
    def get_current_phase_config(self) -> PhaseConfig:
        """Get configuration for current phase"""
        return self.phase_configs[self.current_phase]
    
    def get_current_scenario_config(self) -> ScenarioConfig:
        """Get configuration for current scenario"""
        phase_scenarios = self.scenarios.get(self.current_phase, [])
        if not phase_scenarios:
            return None
        return phase_scenarios[self.current_scenario % len(phase_scenarios)]
    
    def update_performance(self, reward: float, success: bool, 
                          fuel_used: float, intercept_time: float, 
                          additional_metrics: Optional[Dict[str, float]] = None):
        """Update performance metrics for current phase"""
        metrics = self.phase_metrics[self.current_phase]
        metrics.update(reward, success, fuel_used, intercept_time)
        
        self.logger.info(f"Phase {self.current_phase.value}: Episode {metrics.episodes_completed}, "
                        f"Success: {success}, Reward: {reward:.2f}, "
                        f"Recent Success Rate: {metrics.recent_success_rate:.3f}")
        
        # Check for phase advancement
        if self._should_advance_phase():
            self._advance_to_next_phase()
        
        # Dynamic difficulty adjustment
        if self.enable_dynamic_difficulty:
            self._adjust_difficulty()
    
    def _should_advance_phase(self) -> bool:
        """Check if current phase advancement criteria are met"""
        config = self.phase_configs[self.current_phase]
        metrics = self.phase_metrics[self.current_phase]
        
        # Minimum episodes requirement
        if metrics.episodes_completed < config.episodes_required:
            return False
        
        # Check advancement criteria
        criteria_met = True
        
        for criterion in config.advancement_criteria:
            if criterion == AdvancementCriteria.EPISODE_SUCCESS_RATE:
                if metrics.recent_success_rate < config.success_rate_threshold:
                    criteria_met = False
                    break
            elif criterion == AdvancementCriteria.AVERAGE_REWARD:
                if metrics.recent_average_reward < config.average_reward_threshold:
                    criteria_met = False
                    break
            elif criterion == AdvancementCriteria.CONSECUTIVE_SUCCESSES:
                # Check last 10 episodes for consecutive successes
                if len(metrics.episode_success) >= 10:
                    recent_successes = sum(metrics.episode_success[-10:])
                    if recent_successes < 8:  # At least 8/10 success
                        criteria_met = False
                        break
                        
        # Additional sensor-based advancement criteria
        # Only advance if the agent can consistently track targets
        if hasattr(self, '_track_quality_history'):
            recent_track_quality = self._track_quality_history[-20:] if len(self._track_quality_history) >= 20 else self._track_quality_history
            avg_track_quality = sum(recent_track_quality) / len(recent_track_quality) if recent_track_quality else 0.0
            
            # Require minimum tracking performance for advancement
            min_track_quality = 0.4 if self.current_phase.value.endswith('3dof') else 0.6
            if avg_track_quality < min_track_quality:
                criteria_met = False
        
        return criteria_met
    
    def _advance_to_next_phase(self):
        """Advance to the next curriculum phase"""
        phases = list(CurriculumPhase)
        current_index = phases.index(self.current_phase)
        
        if current_index < len(phases) - 1:
            old_phase = self.current_phase
            self.current_phase = phases[current_index + 1]
            self.current_scenario = 0  # Reset scenario counter
            
            self.logger.info(f"Advanced from {old_phase.value} to {self.current_phase.value}")
            print(f"🎓 CURRICULUM ADVANCEMENT: {old_phase.value} → {self.current_phase.value}")
        else:
            self.logger.info("Curriculum completed - reached expert level")
            print("🏆 CURRICULUM COMPLETED - Expert level achieved!")
    
    def _adjust_difficulty(self):
        """Dynamically adjust difficulty based on recent performance"""
        current_time = time.time()
        
        # Only adjust every 100 episodes
        metrics = self.phase_metrics[self.current_phase]
        if metrics.episodes_completed % 100 != 0:
            return
        
        # Don't adjust too frequently
        if current_time - self.last_adjustment_time < 300:  # 5 minutes
            return
        
        config = self.phase_configs[self.current_phase]
        
        # Adjust based on recent performance
        if metrics.recent_success_rate > 0.85:
            # Too easy - increase difficulty
            self._increase_difficulty(config)
            self.logger.info(f"Increased difficulty due to high success rate: {metrics.recent_success_rate:.3f}")
        elif metrics.recent_success_rate < 0.40:
            # Too hard - decrease difficulty
            self._decrease_difficulty(config)
            self.logger.info(f"Decreased difficulty due to low success rate: {metrics.recent_success_rate:.3f}")
        
        self.last_adjustment_time = current_time
    
    def _increase_difficulty(self, config: PhaseConfig):
        """Increase difficulty of current phase"""
        # Reduce intercept threshold
        config.intercept_threshold = max(10.0, config.intercept_threshold * 0.95)
        
        # Increase wind strength
        if config.enable_wind:
            config.wind_strength = min(2.0, config.wind_strength * 1.1)
        
        # Reduce maximum steps slightly
        config.max_steps = max(200, int(config.max_steps * 0.98))
        
        self.difficulty_adjustment_history.append({
            'time': time.time(),
            'phase': config.phase.value,
            'adjustment': 'increase',
            'intercept_threshold': config.intercept_threshold,
            'wind_strength': config.wind_strength,
            'max_steps': config.max_steps
        })
    
    def _decrease_difficulty(self, config: PhaseConfig):
        """Decrease difficulty of current phase"""
        # Increase intercept threshold
        config.intercept_threshold = min(50.0, config.intercept_threshold * 1.05)
        
        # Decrease wind strength
        if config.enable_wind:
            config.wind_strength = max(0.1, config.wind_strength * 0.9)
        
        # Increase maximum steps slightly
        config.max_steps = min(600, int(config.max_steps * 1.02))
        
        self.difficulty_adjustment_history.append({
            'time': time.time(),
            'phase': config.phase.value,
            'adjustment': 'decrease',
            'intercept_threshold': config.intercept_threshold,
            'wind_strength': config.wind_strength,
            'max_steps': config.max_steps
        })
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get current environment configuration for the current phase"""
        phase_config = self.get_current_phase_config()
        scenario_config = self.get_current_scenario_config()
        
        env_config = {
            'difficulty_mode': phase_config.difficulty_mode,
            'action_mode': phase_config.action_mode,
            'world_size': phase_config.world_size,
            'max_steps': phase_config.max_steps,
            'dt': phase_config.dt,
            'intercept_threshold': phase_config.intercept_threshold,
            'miss_threshold': phase_config.miss_threshold,
            'explosion_radius': phase_config.explosion_radius,
            'enable_wind': phase_config.enable_wind,
            'enable_atmosphere': phase_config.enable_atmosphere,
            'wind_strength': phase_config.wind_strength,
            
            # Realistic sensor system parameters
            'enable_realistic_sensors': True,
            'sensor_update_rate': 0.1,
            'weather_conditions': 'clear',
        }
        
        if scenario_config:
            env_config['scenario_config'] = asdict(scenario_config)
        
        return env_config
    
    def load_curriculum_config(self, config_path: str):
        """Load curriculum configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update phase configurations
            if 'phases' in config_data:
                for phase_name, phase_data in config_data['phases'].items():
                    try:
                        phase_enum = CurriculumPhase(phase_name)
                        if phase_enum in self.phase_configs:
                            # Update existing configuration
                            config = self.phase_configs[phase_enum]
                            for key, value in phase_data.items():
                                if hasattr(config, key):
                                    setattr(config, key, value)
                    except ValueError:
                        self.logger.warning(f"Unknown phase in config: {phase_name}")
            
            # Load custom scenarios
            if 'scenarios' in config_data:
                for phase_name, scenarios_data in config_data['scenarios'].items():
                    try:
                        phase_enum = CurriculumPhase(phase_name)
                        scenarios = []
                        for scenario_data in scenarios_data:
                            scenario = ScenarioConfig(**scenario_data)
                            scenarios.append(scenario)
                        self.scenarios[phase_enum] = scenarios
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error loading scenarios for {phase_name}: {e}")
            
            self.logger.info(f"Loaded curriculum configuration from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load curriculum configuration: {e}")
    
    def save_curriculum_config(self, config_path: str):
        """Save current curriculum configuration to JSON file"""
        try:
            config_data = {
                'phases': {},
                'scenarios': {}
            }
            
            # Save phase configurations
            for phase, config in self.phase_configs.items():
                config_data['phases'][phase.value] = asdict(config)
            
            # Save scenarios
            for phase, scenarios in self.scenarios.items():
                config_data['scenarios'][phase.value] = [asdict(scenario) for scenario in scenarios]
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved curriculum configuration to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save curriculum configuration: {e}")
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get comprehensive curriculum status"""
        current_metrics = self.phase_metrics[self.current_phase]
        
        status = {
            'current_phase': self.current_phase.value,
            'current_scenario': self.current_scenario,
            'phase_progress': {
                'episodes_completed': current_metrics.episodes_completed,
                'episodes_required': self.phase_configs[self.current_phase].episodes_required,
                'success_rate': current_metrics.success_rate,
                'recent_success_rate': current_metrics.recent_success_rate,
                'average_reward': current_metrics.average_reward,
                'recent_average_reward': current_metrics.recent_average_reward,
            },
            'advancement_criteria': {
                'success_rate_threshold': self.phase_configs[self.current_phase].success_rate_threshold,
                'average_reward_threshold': self.phase_configs[self.current_phase].average_reward_threshold,
                'criteria_met': self._should_advance_phase()
            },
            'all_phases_metrics': {}
        }
        
        # Add metrics for all phases
        for phase, metrics in self.phase_metrics.items():
            status['all_phases_metrics'][phase.value] = {
                'episodes_completed': metrics.episodes_completed,
                'success_rate': metrics.success_rate,
                'average_reward': metrics.average_reward,
                'recent_success_rate': metrics.recent_success_rate
            }
        
        return status
    
    def reset_phase(self, phase: Optional[CurriculumPhase] = None):
        """Reset a specific phase or current phase"""
        if phase is None:
            phase = self.current_phase
        
        self.phase_metrics[phase] = PerformanceMetrics()
        self.logger.info(f"Reset phase {phase.value}")
    
    def set_phase(self, phase: CurriculumPhase):
        """Manually set the current phase"""
        if phase in self.phase_configs:
            old_phase = self.current_phase
            self.current_phase = phase
            self.current_scenario = 0
            self.logger.info(f"Manually set phase from {old_phase.value} to {phase.value}")
        else:
            raise ValueError(f"Invalid phase: {phase}")


# Utility functions for curriculum integration
def create_curriculum_manager(config_path: Optional[str] = None) -> CurriculumManager:
    """Create and initialize a curriculum manager"""
    return CurriculumManager(config_path=config_path)


def setup_curriculum_directories(base_path: str = "curriculum"):
    """Setup directory structure for curriculum learning"""
    Path(base_path).mkdir(exist_ok=True)
    Path(f"{base_path}/configs").mkdir(exist_ok=True)
    Path(f"{base_path}/logs").mkdir(exist_ok=True)
    Path(f"{base_path}/checkpoints").mkdir(exist_ok=True)
    Path(f"{base_path}/scenarios").mkdir(exist_ok=True)
    
    return {
        'base': base_path,
        'configs': f"{base_path}/configs",
        'logs': f"{base_path}/logs", 
        'checkpoints': f"{base_path}/checkpoints",
        'scenarios': f"{base_path}/scenarios"
    }