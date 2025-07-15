#!/usr/bin/env python3
"""
Demo Script for AegisIntercept Phase 3 - 6DOF System.

This script demonstrates the complete 6-DOF missile intercept simulation
with real-time visualization, trajectory logging, and Unity export.

Usage:
    python demo_6dof_system.py [options]

Examples:
    # Run basic demo
    python demo_6dof_system.py
    
    # Run with trained model
    python demo_6dof_system.py --model-path trained_model.zip
    
    # Run with specific scenario
    python demo_6dof_system.py --scenario-file easy_scenario.json
    
    # Export to Unity
    python demo_6dof_system.py --export-unity --output-dir demo_exports
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv
from aegis_intercept.rendering.viewer3d import MatplotlibViewer
from aegis_intercept.logging import TrajectoryLogger, UnityExporter, ExportManager
from aegis_intercept.curriculum import ScenarioGenerator
from aegis_intercept.utils.physics3d import WindField

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: Stable-Baselines3 not available. Running with random policy.")
    SB3_AVAILABLE = False


class DemoController:
    """
    Demonstration controller for AegisIntercept Phase 3.
    
    This class manages the demo execution, integrating visualization,
    logging, and model inference for comprehensive demonstration.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize demo controller."""
        self.args = args
        self.current_episode = 0
        self.total_episodes = args.num_episodes
        
        # Initialize components
        self.env = None
        self.model = None
        self.viewer = None
        self.trajectory_logger = None
        self.export_manager = None
        self.scenario_generator = None
        
        # Demo state
        self.episode_metrics = []
        self.demo_start_time = time.time()
        
        # Setup logging
        self.setup_logging()
        
        print("="*60)
        print("AegisIntercept Phase 3 - 6DOF Demonstration")
        print("="*60)
        print(f"Episodes to run: {self.total_episodes}")
        print(f"Visualization: {'Enabled' if not args.headless else 'Disabled'}")
        print(f"Model: {'Trained' if args.model_path else 'Random Policy'}")
        print(f"Export Unity: {'Yes' if args.export_unity else 'No'}")
        print("="*60)
    
    def setup_logging(self):
        """Setup logging and export systems."""
        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trajectory logger
        self.trajectory_logger = TrajectoryLogger(
            enable_detailed_logging=True,
            enable_performance_mode=False,  # Full logging for demo
            log_frequency=1,
            max_trajectory_points=5000
        )
        
        # Initialize export manager
        if self.args.export_unity or self.args.export_csv:
            self.export_manager = ExportManager(
                output_directory=str(output_dir),
                auto_compress=True,
                max_episodes_per_file=10
            )
        
        print(f"Output directory: {output_dir}")
    
    def load_model(self) -> bool:
        """Load trained model if specified."""
        if not self.args.model_path or not SB3_AVAILABLE:
            return False
        
        model_path = Path(self.args.model_path)
        if not model_path.exists():
            print(f"Warning: Model file not found: {model_path}")
            return False
        
        try:
            print(f"Loading model: {model_path}")
            self.model = PPO.load(str(model_path))
            
            # Load normalization if available
            norm_path = model_path.parent / "vec_normalize.pkl"
            if norm_path.exists():
                print("Loading normalization statistics...")
                # Note: This would need environment integration for full support
            
            print("Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def create_environment(self) -> Aegis6DOFEnv:
        """Create demonstration environment."""
        # Determine scenario parameters
        if self.args.scenario_file:
            scenario_params = self.load_scenario_file()
        else:
            scenario_params = self.get_default_scenario()
        
        # Create environment
        env = Aegis6DOFEnv(
            render_mode="human" if not self.args.headless else None,
            max_episode_steps=scenario_params.get('max_episode_steps', 1000),
            time_step=self.args.time_step,
            world_scale=scenario_params.get('world_scale', 2000.0),
            wind_config=scenario_params.get('wind_config'),
            curriculum_level=scenario_params.get('difficulty', 'medium')
        )
        
        print(f"Environment created with scenario: {scenario_params.get('name', 'default')}")
        return env
    
    def load_scenario_file(self) -> Dict[str, Any]:
        """Load scenario from file."""
        try:
            with open(self.args.scenario_file, 'r') as f:
                scenario_data = json.load(f)
            
            print(f"Loaded scenario: {scenario_data.get('name', 'unnamed')}")
            return scenario_data
            
        except Exception as e:
            print(f"Error loading scenario file: {e}")
            return self.get_default_scenario()
    
    def get_default_scenario(self) -> Dict[str, Any]:
        """Get default demo scenario."""
        return {
            'name': 'default_demo',
            'description': 'Standard demonstration scenario',
            'difficulty': 'medium',
            'max_episode_steps': 1000,
            'world_scale': 2000.0,
            'wind_config': {
                'steady_wind': [5.0, 2.0, 0.0],
                'turbulence_intensity': 0.15,
                'gust_amplitude': 8.0,
                'wind_profile': 'logarithmic'
            }
        }
    
    def create_visualization(self) -> Optional[MatplotlibViewer]:
        """Create visualization system."""
        if self.args.headless:
            return None
        
        viewer = MatplotlibViewer(
            world_scale=2000.0,
            update_rate=20.0,
            trail_length=100,
            enable_wind_vectors=True,
            enable_performance_plots=True
        )
        
        print("Visualization system initialized")
        return viewer
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action from model or random policy."""
        if self.model:
            try:
                action, _ = self.model.predict(observation, deterministic=True)
                return action
            except Exception as e:
                print(f"Model prediction error: {e}")
        
        # Random policy fallback
        return self.env.action_space.sample()
    
    def run_single_episode(self) -> Dict[str, Any]:
        """Run a single demonstration episode."""
        print(f"\nStarting Episode {self.current_episode + 1}/{self.total_episodes}")
        
        # Reset environment
        observation, info = self.env.reset()
        
        # Start episode logging
        episode_metadata = {
            'episode_number': self.current_episode,
            'demo_mode': True,
            'model_used': self.args.model_path is not None,
            'scenario': 'demo'
        }
        self.trajectory_logger.start_episode(episode_metadata)
        
        # Episode loop
        step = 0
        total_reward = 0.0
        episode_start_time = time.time()
        
        while True:
            # Get action
            action = self.get_action(observation)
            
            # Step environment
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            step += 1
            
            # Log trajectory data
            self.log_step_data(step, info, reward, action)
            
            # Update visualization
            if self.viewer:
                self.update_visualization(info)
                
                # Small delay for real-time viewing
                if not self.args.fast_mode:
                    time.sleep(self.args.time_step * 0.5)  # Slow down for viewing
            
            # Check termination
            if terminated or truncated:
                break
            
            observation = next_observation
        
        # Episode completed
        episode_time = time.time() - episode_start_time
        
        # Finalize logging
        final_metadata = {
            'total_steps': step,
            'total_reward': total_reward,
            'episode_time': episode_time,
            'success': info.get('episode_metrics', {}).get('success', False),
            'final_distance': info.get('intercept_distance', 0.0)
        }
        self.trajectory_logger.end_episode(final_metadata)
        
        # Episode summary
        episode_summary = {
            'episode': self.current_episode,
            'steps': step,
            'reward': total_reward,
            'time': episode_time,
            'success': final_metadata['success'],
            'final_distance': final_metadata['final_distance']
        }
        
        print(f"Episode completed: {episode_summary}")
        
        # Add to export manager
        if self.export_manager:
            episode_id = f"demo_episode_{self.current_episode:03d}"
            self.export_manager.add_episode(episode_id, self.trajectory_logger)
        
        return episode_summary
    
    def log_step_data(self, step: int, info: Dict[str, Any], reward: float, action: np.ndarray):
        """Log detailed step data."""
        # Extract state information
        interceptor_state = {
            'position': info.get('interceptor_position', [0, 0, 0]),
            'velocity': info.get('interceptor_velocity', [0, 0, 0]),
            'orientation': info.get('interceptor_orientation', [1, 0, 0, 0]),
            'angular_velocity': info.get('interceptor_angular_velocity', [0, 0, 0])
        }
        
        adversary_state = {
            'position': info.get('adversary_position', [0, 0, 0]),
            'velocity': info.get('adversary_velocity', [0, 0, 0]),
            'orientation': info.get('adversary_orientation', [1, 0, 0, 0]),
            'angular_velocity': info.get('adversary_angular_velocity', [0, 0, 0])
        }
        
        environment_state = {
            'wind_velocity': info.get('wind_velocity', [0, 0, 0]),
            'atmospheric_density': info.get('atmospheric_density', 1.225)
        }
        
        control_inputs = {
            'interceptor_thrust': action[:3].tolist() if len(action) >= 3 else [0, 0, 0],
            'interceptor_torque': action[3:6].tolist() if len(action) >= 6 else [0, 0, 0],
            'adversary_thrust': info.get('adversary_thrust', [0, 0, 0]),
            'adversary_torque': info.get('adversary_torque', [0, 0, 0])
        }
        
        reward_info = {
            'step_reward': reward,
            'distance_reward': info.get('distance_reward', 0.0),
            'fuel_penalty': info.get('fuel_penalty', 0.0),
            'control_penalty': info.get('control_penalty', 0.0)
        }
        
        mission_status = {
            'fuel_remaining': info.get('fuel_remaining', 1.0),
            'episode_complete': False,
            'success': info.get('episode_metrics', {}).get('success', False)
        }
        
        # Log the step
        self.trajectory_logger.log_step(
            step=step,
            simulation_time=step * self.args.time_step,
            interceptor_state=interceptor_state,
            adversary_state=adversary_state,
            environment_state=environment_state,
            control_inputs=control_inputs,
            reward_info=reward_info,
            mission_status=mission_status
        )
    
    def update_visualization(self, info: Dict[str, Any]):
        """Update real-time visualization."""
        if not self.viewer:
            return
        
        # Extract positions and velocities
        interceptor_pos = np.array(info.get('interceptor_position', [0, 0, 0]))
        interceptor_vel = np.array(info.get('interceptor_velocity', [0, 0, 0]))
        adversary_pos = np.array(info.get('adversary_position', [0, 0, 0]))
        adversary_vel = np.array(info.get('adversary_velocity', [0, 0, 0]))
        target_pos = np.array([0, 0, 0])  # Target at origin
        
        # Performance metrics
        performance_metrics = {
            'intercept_distance': info.get('intercept_distance', 0.0),
            'step_reward': info.get('step_reward', 0.0),
            'fuel_remaining': info.get('fuel_remaining', 1.0)
        }
        
        # Update visualization
        self.viewer.update_visualization(
            interceptor_pos=interceptor_pos,
            interceptor_vel=interceptor_vel,
            adversary_pos=adversary_pos,
            adversary_vel=adversary_vel,
            target_pos=target_pos,
            wind_field=None,  # Could be added if needed
            performance_metrics=performance_metrics,
            simulation_time=info.get('episode_time', 0.0)
        )
    
    def run_demo(self):
        """Run the complete demonstration."""
        try:
            # Load model
            model_loaded = self.load_model()
            
            # Create environment
            self.env = self.create_environment()
            
            # Create visualization
            self.viewer = self.create_visualization()
            if self.viewer:
                self.viewer.start()
            
            # Run episodes
            for episode in range(self.total_episodes):
                self.current_episode = episode
                episode_summary = self.run_single_episode()
                self.episode_metrics.append(episode_summary)
                
                # Break if visualization closed
                if self.viewer and not self.viewer.is_running:
                    print("Visualization closed by user")
                    break
            
            # Demo completed
            self.finalize_demo()
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
            self.finalize_demo()
        
        except Exception as e:
            print(f"\nDemo failed with error: {e}")
            return 1
        
        finally:
            # Cleanup
            if self.viewer:
                self.viewer.stop()
            if self.env:
                self.env.close()
        
        return 0
    
    def finalize_demo(self):
        """Finalize demo and export results."""
        demo_time = time.time() - self.demo_start_time
        
        print(f"\nDemo completed in {demo_time:.2f} seconds")
        print(f"Episodes run: {len(self.episode_metrics)}")
        
        # Calculate summary statistics
        if self.episode_metrics:
            success_rate = sum(ep['success'] for ep in self.episode_metrics) / len(self.episode_metrics)
            avg_reward = np.mean([ep['reward'] for ep in self.episode_metrics])
            avg_steps = np.mean([ep['steps'] for ep in self.episode_metrics])
            
            print(f"Success rate: {success_rate:.2%}")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Average steps: {avg_steps:.1f}")
        
        # Export results
        self.export_results()
        
        # Save screenshot if visualization was used
        if self.viewer and not self.args.headless:
            screenshot_path = Path(self.args.output_dir) / "demo_screenshot.png"
            self.viewer.save_screenshot(str(screenshot_path))
            
            trajectory_path = Path(self.args.output_dir) / "demo_trajectory.png"
            self.viewer.export_trajectory_plot(str(trajectory_path))
    
    def export_results(self):
        """Export demonstration results."""
        output_dir = Path(self.args.output_dir)
        
        # Export episode summaries
        summaries_path = output_dir / "episode_summaries.json"
        with open(summaries_path, 'w') as f:
            json.dump(self.episode_metrics, f, indent=2)
        
        print(f"Episode summaries saved: {summaries_path}")
        
        # Export via export manager
        if self.export_manager:
            formats = []
            if self.args.export_csv:
                formats.append('csv')
            if self.args.export_unity:
                formats.append('unity')
            formats.append('json')  # Always export JSON
            
            exported_files = self.export_manager.export_all_episodes(
                formats=formats,
                filename_prefix='demo_results'
            )
            
            print(f"Exported files: {list(exported_files.keys())}")
            
            # Create analysis report
            analysis_path = output_dir / "demo_analysis.json"
            self.export_manager.create_analysis_report(str(analysis_path))
            print(f"Analysis report saved: {analysis_path}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='AegisIntercept Phase 3 Demonstration')
    
    # Demo parameters
    parser.add_argument('--num-episodes', type=int, default=1,
                       help='Number of episodes to run')
    parser.add_argument('--time-step', type=float, default=0.02,
                       help='Simulation time step')
    parser.add_argument('--headless', action='store_true',
                       help='Run without visualization')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Run at maximum speed (no delays)')
    
    # Model and scenario
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model')
    parser.add_argument('--scenario-file', type=str, default=None,
                       help='Path to scenario configuration file')
    
    # Export options
    parser.add_argument('--output-dir', type=str, default='demo_output',
                       help='Output directory for results')
    parser.add_argument('--export-unity', action='store_true',
                       help='Export Unity-compatible data')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export CSV trajectory data')
    
    args = parser.parse_args()
    
    # Create and run demo
    demo_controller = DemoController(args)
    exit_code = demo_controller.run_demo()
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)