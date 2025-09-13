"""
Phase 4 RL: Enhanced Multi-Entity Radar Environment for Reinforcement Learning

This package provides a comprehensive framework for training and evaluating RL agents
in multi-missile/interceptor scenarios with radar-based observations.

Key Components:
- RadarEnv: Multi-entity gymnasium environment with radar processing
- Configuration: Centralized configuration management 
- Scenarios: Template-driven scenario management system
- Diagnostics: Multi-entity logging and analysis tools
- Training: Enhanced PPO training with scenario support
- Inference: Comprehensive evaluation and testing framework

Example Usage:

    # Basic environment usage
    from phase4_rl import RadarEnv
    env = RadarEnv()
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Configuration and scenario usage
    from phase4_rl.config import get_config
    from phase4_rl.scenarios import get_scenario_loader
    
    config = get_config()
    scenario_loader = get_scenario_loader()
    env_config = scenario_loader.create_environment_config("easy", config._config)
    env = RadarEnv(config=env_config, scenario_name="easy")

    # Training
    from phase4_rl.train_radar_ppo import Phase4Trainer
    trainer = Phase4Trainer(scenario_name="easy")
    trainer.train()

    # Inference
    from phase4_rl.run_inference import Phase4InferenceRunner
    runner = Phase4InferenceRunner("model.zip", scenario_name="easy")
    results = runner.run_episodes(num_episodes=10)
"""

__version__ = "0.1.0"
__author__ = "Quinn Hasse"

# Core imports
from .radar_env import RadarEnv
from .config import ConfigLoader, get_config
from .scenarios import ScenarioLoader, get_scenario_loader
from .diagnostics import Logger, export_to_csv, export_to_json, plot_metrics

# Training and inference
from .train_radar_ppo import Phase4Trainer
from .run_inference import Phase4InferenceRunner

__all__ = [
    # Core classes
    'RadarEnv',
    'ConfigLoader', 
    'get_config',
    'ScenarioLoader',
    'get_scenario_loader',
    'Logger',
    
    # Training and inference
    'Phase4Trainer',
    'Phase4InferenceRunner',
    
    # Utility functions
    'export_to_csv',
    'export_to_json', 
    'plot_metrics',
]