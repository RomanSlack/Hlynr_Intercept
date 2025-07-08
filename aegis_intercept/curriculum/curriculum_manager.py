"""
Curriculum Manager for AegisIntercept Phase 3.

This module manages automatic curriculum progression based on training performance,
promoting to higher difficulty levels when success criteria are met.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os
from collections import deque


class CurriculumManager:
    """
    Manages curriculum progression for training.
    
    This class tracks training performance and automatically promotes to
    higher difficulty levels when the agent achieves sufficient success
    rates. It supports configurable promotion criteria and difficulty tiers.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 promotion_threshold: float = 0.85,
                 evaluation_window: int = 100,
                 min_episodes: int = 50):
        """
        Initialize curriculum manager.
        
        Args:
            config_path: Path to curriculum configuration file
            promotion_threshold: Success rate threshold for promotion (0-1)
            evaluation_window: Number of episodes to evaluate for promotion
            min_episodes: Minimum episodes before considering promotion
        """
        self.promotion_threshold = promotion_threshold
        self.evaluation_window = evaluation_window
        self.min_episodes = min_episodes
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config = self._load_config(config_path)
        else:
            self.config = self._get_default_config()
        
        # Current curriculum state
        self.current_level = "easy"
        self.current_tier_index = 0
        self.available_tiers = list(self.config["tiers"].keys())
        
        # Performance tracking
        self.episode_history = deque(maxlen=self.evaluation_window)
        self.level_statistics = {level: {"episodes": 0, "successes": 0, "total_reward": 0.0} 
                                for level in self.available_tiers}
        
        # Promotion history
        self.promotion_history = []
        self.episodes_in_current_level = 0
        
        print(f"Curriculum Manager initialized with {len(self.available_tiers)} tiers")
        print(f"Starting level: {self.current_level}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default curriculum configuration."""
        return {
            "tiers": {
                "easy": {
                    "spawn_separation": 500.0,
                    "adversary_speed_range": [150, 200],
                    "adversary_evasion_aggressiveness": 0.2,
                    "wind_severity": 0.1,
                    "max_episode_steps": 800,
                    "fuel_capacity": 1.0,
                    "kill_distance": 5.0,
                    "description": "Basic interception with minimal evasion"
                },
                "medium": {
                    "spawn_separation": 800.0,
                    "adversary_speed_range": [180, 250],
                    "adversary_evasion_aggressiveness": 0.5,
                    "wind_severity": 0.2,
                    "max_episode_steps": 1000,
                    "fuel_capacity": 0.8,
                    "kill_distance": 3.0,
                    "description": "Moderate evasion with environmental challenges"
                },
                "hard": {
                    "spawn_separation": 1200.0,
                    "adversary_speed_range": [200, 300],
                    "adversary_evasion_aggressiveness": 0.8,
                    "wind_severity": 0.3,
                    "max_episode_steps": 1200,
                    "fuel_capacity": 0.6,
                    "kill_distance": 2.0,
                    "description": "Aggressive evasion with limited fuel"
                },
                "impossible": {
                    "spawn_separation": 1500.0,
                    "adversary_speed_range": [250, 350],
                    "adversary_evasion_aggressiveness": 1.0,
                    "wind_severity": 0.4,
                    "max_episode_steps": 1500,
                    "fuel_capacity": 0.4,
                    "kill_distance": 1.5,
                    "description": "Maximum difficulty with all challenges"
                }
            },
            "promotion_criteria": {
                "success_rate_threshold": 0.85,
                "evaluation_window": 100,
                "min_episodes": 50,
                "stability_requirement": 10  # Consecutive episodes above threshold
            }
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load curriculum configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading curriculum config: {e}")
            return self._get_default_config()
    
    def save_config(self, config_path: str):
        """Save current curriculum configuration to JSON file."""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving curriculum config: {e}")
    
    def record_episode(self, success: bool, total_reward: float, info: Dict[str, Any]):
        """
        Record episode results for curriculum evaluation.
        
        Args:
            success: Whether the episode was successful
            total_reward: Total reward achieved
            info: Additional episode information
        """
        # Record episode data
        episode_data = {
            'success': success,
            'reward': total_reward,
            'level': self.current_level,
            'episode_num': self.episodes_in_current_level,
            'info': info
        }
        
        self.episode_history.append(episode_data)
        
        # Update level statistics
        self.level_statistics[self.current_level]["episodes"] += 1
        self.level_statistics[self.current_level]["total_reward"] += total_reward
        if success:
            self.level_statistics[self.current_level]["successes"] += 1
        
        self.episodes_in_current_level += 1
        
        # Check for promotion
        if self._should_promote():
            self._promote_to_next_level()
    
    def _should_promote(self) -> bool:
        """Check if agent should be promoted to next difficulty level."""
        # Must have minimum episodes in current level
        if self.episodes_in_current_level < self.min_episodes:
            return False
        
        # Must not be at highest level
        if self.current_tier_index >= len(self.available_tiers) - 1:
            return False
        
        # Must have sufficient recent episodes
        if len(self.episode_history) < self.evaluation_window:
            return False
        
        # Calculate success rate over evaluation window
        recent_episodes = list(self.episode_history)[-self.evaluation_window:]
        current_level_episodes = [ep for ep in recent_episodes if ep['level'] == self.current_level]
        
        if len(current_level_episodes) < self.min_episodes:
            return False
        
        success_rate = sum(ep['success'] for ep in current_level_episodes) / len(current_level_episodes)
        
        # Check if success rate meets threshold
        return success_rate >= self.promotion_threshold
    
    def _promote_to_next_level(self):
        """Promote agent to next difficulty level."""
        if self.current_tier_index < len(self.available_tiers) - 1:
            old_level = self.current_level
            self.current_tier_index += 1
            self.current_level = self.available_tiers[self.current_tier_index]
            
            # Record promotion
            promotion_data = {
                'from_level': old_level,
                'to_level': self.current_level,
                'total_episodes': sum(stats['episodes'] for stats in self.level_statistics.values()),
                'episodes_in_level': self.episodes_in_current_level,
                'success_rate': self.get_current_success_rate()
            }
            
            self.promotion_history.append(promotion_data)
            self.episodes_in_current_level = 0
            
            print(f"🎯 PROMOTED: {old_level} → {self.current_level}")
            print(f"   Success rate: {promotion_data['success_rate']:.2%}")
            print(f"   Episodes in level: {promotion_data['episodes_in_level']}")
    
    def get_current_level(self) -> str:
        """Get current curriculum level."""
        return self.current_level
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get configuration for current difficulty level."""
        return self.config["tiers"][self.current_level].copy()
    
    def get_current_success_rate(self) -> float:
        """Get success rate for current level."""
        stats = self.level_statistics[self.current_level]
        if stats["episodes"] == 0:
            return 0.0
        return stats["successes"] / stats["episodes"]
    
    def get_overall_success_rate(self) -> float:
        """Get overall success rate across all levels."""
        if not self.episode_history:
            return 0.0
        
        total_successes = sum(ep['success'] for ep in self.episode_history)
        return total_successes / len(self.episode_history)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive curriculum statistics."""
        return {
            'current_level': self.current_level,
            'current_tier_index': self.current_tier_index,
            'episodes_in_current_level': self.episodes_in_current_level,
            'total_episodes': sum(stats['episodes'] for stats in self.level_statistics.values()),
            'level_statistics': self.level_statistics.copy(),
            'promotion_history': self.promotion_history.copy(),
            'current_success_rate': self.get_current_success_rate(),
            'overall_success_rate': self.get_overall_success_rate(),
            'evaluation_window_size': len(self.episode_history)
        }
    
    def set_level(self, level: str):
        """Manually set curriculum level."""
        if level in self.available_tiers:
            self.current_level = level
            self.current_tier_index = self.available_tiers.index(level)
            self.episodes_in_current_level = 0
            print(f"Curriculum level manually set to: {level}")
        else:
            print(f"Invalid level: {level}. Available levels: {self.available_tiers}")
    
    def reset_statistics(self):
        """Reset all curriculum statistics."""
        self.episode_history.clear()
        self.level_statistics = {level: {"episodes": 0, "successes": 0, "total_reward": 0.0} 
                                for level in self.available_tiers}
        self.promotion_history.clear()
        self.episodes_in_current_level = 0
        print("Curriculum statistics reset")
    
    def export_data(self, filename: str):
        """Export curriculum data to JSON file."""
        data = {
            'config': self.config,
            'statistics': self.get_statistics(),
            'episode_history': list(self.episode_history)
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Curriculum data exported to: {filename}")
        except Exception as e:
            print(f"Error exporting curriculum data: {e}")
    
    def print_progress_report(self):
        """Print detailed progress report."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("CURRICULUM PROGRESS REPORT")
        print("="*60)
        print(f"Current Level: {stats['current_level']} ({stats['current_tier_index']+1}/{len(self.available_tiers)})")
        print(f"Episodes in Current Level: {stats['episodes_in_current_level']}")
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Current Success Rate: {stats['current_success_rate']:.2%}")
        print(f"Overall Success Rate: {stats['overall_success_rate']:.2%}")
        
        print("\nLevel Statistics:")
        for level, level_stats in stats['level_statistics'].items():
            if level_stats['episodes'] > 0:
                success_rate = level_stats['successes'] / level_stats['episodes']
                avg_reward = level_stats['total_reward'] / level_stats['episodes']
                marker = "👑" if level == self.current_level else "✓"
                print(f"  {marker} {level:12} | Episodes: {level_stats['episodes']:4} | "
                      f"Success: {success_rate:.2%} | Avg Reward: {avg_reward:.2f}")
        
        if stats['promotion_history']:
            print("\nPromotion History:")
            for i, promo in enumerate(stats['promotion_history']):
                print(f"  {i+1}. {promo['from_level']} → {promo['to_level']} "
                      f"(Episodes: {promo['episodes_in_level']}, "
                      f"Success: {promo['success_rate']:.2%})")
        
        print("="*60 + "\n")