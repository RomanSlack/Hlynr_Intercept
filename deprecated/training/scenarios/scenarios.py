"""
Scenario loading and management for Phase 4 RL training.

This module provides the ScenarioLoader class for loading and validating
scenario templates from JSON files.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class ScenarioLoader:
    """Loads and manages scenario templates for different difficulty levels."""
    
    def __init__(self, scenarios_dir: Optional[str] = None):
        """
        Initialize scenario loader.
        
        Args:
            scenarios_dir: Directory containing scenario JSON files.
                          If None, uses directory containing this file.
        """
        if scenarios_dir is None:
            scenarios_dir = Path(__file__).parent
        
        self.scenarios_dir = Path(scenarios_dir)
        self._scenarios_cache = {}
        self._available_scenarios = self._discover_scenarios()
    
    def _discover_scenarios(self) -> List[str]:
        """Discover available scenario files in the scenarios directory."""
        scenarios = []
        if self.scenarios_dir.exists():
            for file_path in self.scenarios_dir.glob("*.json"):
                scenario_name = file_path.stem
                scenarios.append(scenario_name)
        return sorted(scenarios)
    
    def list_scenarios(self) -> List[str]:
        """
        Get list of available scenario names.
        
        Returns:
            List of scenario names (without .json extension)
        """
        return self._available_scenarios.copy()
    
    def load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Load scenario configuration from JSON file.
        
        Args:
            scenario_name: Name of scenario (without .json extension)
            
        Returns:
            Dictionary containing scenario configuration
            
        Raises:
            FileNotFoundError: If scenario file doesn't exist
            ValueError: If scenario JSON is invalid
        """
        # Check cache first
        if scenario_name in self._scenarios_cache:
            return self._scenarios_cache[scenario_name].copy()
        
        scenario_path = self.scenarios_dir / f"{scenario_name}.json"
        
        if not scenario_path.exists():
            available = ", ".join(self._available_scenarios)
            raise FileNotFoundError(
                f"Scenario '{scenario_name}' not found. "
                f"Available scenarios: {available}"
            )
        
        try:
            with open(scenario_path, 'r') as f:
                scenario_config = json.load(f)
            
            # Validate scenario structure
            self._validate_scenario(scenario_config, scenario_name)
            
            # Cache the scenario
            self._scenarios_cache[scenario_name] = scenario_config
            
            return scenario_config.copy()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in scenario '{scenario_name}': {e}")
        except Exception as e:
            raise ValueError(f"Error loading scenario '{scenario_name}': {e}")
    
    def _validate_scenario(self, scenario: Dict[str, Any], name: str) -> None:
        """
        Validate scenario configuration structure.
        
        Args:
            scenario: Scenario configuration dictionary
            name: Scenario name for error messages
            
        Raises:
            ValueError: If scenario structure is invalid
        """
        required_fields = [
            'name', 'description', 'difficulty_level',
            'num_missiles', 'num_interceptors', 'spawn_positions',
            'wind_settings', 'adversary_config', 'radar_config'
        ]
        
        missing_fields = [field for field in required_fields if field not in scenario]
        if missing_fields:
            raise ValueError(
                f"Scenario '{name}' missing required fields: {missing_fields}"
            )
        
        # Validate spawn positions structure
        spawn_pos = scenario['spawn_positions']
        required_spawn_fields = ['missiles', 'interceptors', 'targets']
        missing_spawn = [field for field in required_spawn_fields if field not in spawn_pos]
        if missing_spawn:
            raise ValueError(
                f"Scenario '{name}' spawn_positions missing fields: {missing_spawn}"
            )
        
        # Validate entity counts match spawn positions
        num_missiles = scenario['num_missiles']
        num_interceptors = scenario['num_interceptors']
        
        if len(spawn_pos['missiles']) != num_missiles:
            raise ValueError(
                f"Scenario '{name}': num_missiles ({num_missiles}) doesn't match "
                f"spawn positions ({len(spawn_pos['missiles'])})"
            )
        
        if len(spawn_pos['interceptors']) != num_interceptors:
            raise ValueError(
                f"Scenario '{name}': num_interceptors ({num_interceptors}) doesn't match "
                f"spawn positions ({len(spawn_pos['interceptors'])})"
            )
        
        # Validate difficulty level
        difficulty = scenario['difficulty_level']
        if not isinstance(difficulty, int) or difficulty < 1 or difficulty > 4:
            raise ValueError(
                f"Scenario '{name}': difficulty_level must be integer 1-4, got {difficulty}"
            )
    
    def get_scenario_by_difficulty(self, difficulty: int) -> Optional[str]:
        """
        Get scenario name for a given difficulty level.
        
        Args:
            difficulty: Difficulty level (1-4)
            
        Returns:
            Scenario name or None if not found
        """
        for scenario_name in self._available_scenarios:
            try:
                scenario = self.load_scenario(scenario_name)
                if scenario['difficulty_level'] == difficulty:
                    return scenario_name
            except (FileNotFoundError, ValueError):
                continue
        return None
    
    def get_scenarios_by_difficulty(self) -> Dict[int, str]:
        """
        Get mapping of difficulty levels to scenario names.
        
        Returns:
            Dictionary mapping difficulty level to scenario name
        """
        difficulty_map = {}
        for scenario_name in self._available_scenarios:
            try:
                scenario = self.load_scenario(scenario_name)
                difficulty = scenario['difficulty_level']
                difficulty_map[difficulty] = scenario_name
            except (FileNotFoundError, ValueError):
                continue
        return difficulty_map
    
    def merge_with_base_config(self, scenario_config: Dict[str, Any], 
                              base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge scenario configuration with base configuration.
        
        Args:
            scenario_config: Scenario-specific configuration
            base_config: Base configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        # Update environment settings
        if 'environment' in merged:
            merged['environment']['num_missiles'] = scenario_config['num_missiles']
            merged['environment']['num_interceptors'] = scenario_config['num_interceptors']
        
        # Update radar configuration
        if 'radar' in merged and 'radar_config' in scenario_config:
            merged['radar'].update(scenario_config['radar_config'])
        
        # Update spawn configuration  
        if 'spawn' in merged and 'spawn_positions' in scenario_config:
            merged['spawn']['missile_spawn_positions'] = scenario_config['spawn_positions']['missiles']
            merged['spawn']['interceptor_spawn_positions'] = scenario_config['spawn_positions']['interceptors']
            merged['spawn']['target_positions'] = scenario_config['spawn_positions']['targets']
        
        # Update environmental conditions
        if 'environment_conditions' in merged and 'wind_settings' in scenario_config:
            wind = scenario_config['wind_settings']
            merged['environment_conditions']['wind_speed'] = wind['speed']
            merged['environment_conditions']['wind_direction'] = wind['direction']
            merged['environment_conditions']['wind_variability'] = wind['variability']
        
        # Add scenario-specific metadata
        merged['scenario'] = {
            'name': scenario_config['name'],
            'description': scenario_config['description'],
            'difficulty_level': scenario_config['difficulty_level'],
            'adversary_config': scenario_config['adversary_config'],
            'success_criteria': scenario_config.get('success_criteria', {})
        }
        
        return merged
    
    def create_environment_config(self, scenario_name: str, 
                                 base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create environment configuration for a specific scenario.
        
        Args:
            scenario_name: Name of scenario to load
            base_config: Base configuration to merge with. If None, uses default.
            
        Returns:
            Complete environment configuration dictionary
        """
        scenario = self.load_scenario(scenario_name)
        
        if base_config is None:
            # Import here to avoid circular imports
            try:
                from ..config import get_config
            except ImportError:
                from config import get_config
            config_loader = get_config()
            base_config = config_loader._config
        
        return self.merge_with_base_config(scenario, base_config)
    
    def validate_all_scenarios(self) -> Dict[str, bool]:
        """
        Validate all available scenarios.
        
        Returns:
            Dictionary mapping scenario names to validation status (True/False)
        """
        results = {}
        for scenario_name in self._available_scenarios:
            try:
                self.load_scenario(scenario_name)
                results[scenario_name] = True
            except (FileNotFoundError, ValueError):
                results[scenario_name] = False
        return results
    
    def __repr__(self) -> str:
        return f"ScenarioLoader(scenarios_dir={self.scenarios_dir}, available={len(self._available_scenarios)})"


# Global scenario loader instance
_global_scenario_loader = None

def get_scenario_loader(scenarios_dir: Optional[str] = None) -> ScenarioLoader:
    """
    Get global scenario loader instance.
    
    Args:
        scenarios_dir: Directory containing scenarios. Only used for first call.
        
    Returns:
        Global ScenarioLoader instance
    """
    global _global_scenario_loader
    if _global_scenario_loader is None:
        _global_scenario_loader = ScenarioLoader(scenarios_dir)
    return _global_scenario_loader

def reset_scenario_loader() -> None:
    """Reset global scenario loader instance (mainly for testing)."""
    global _global_scenario_loader
    _global_scenario_loader = None