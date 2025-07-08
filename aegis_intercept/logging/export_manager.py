"""
Export Manager for AegisIntercept Phase 3.

This module manages the aggregation and export of multiple episodes
for batch analysis and Unity visualization.
"""

import json
import csv
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time
import zipfile
import tempfile
import shutil

from .trajectory_logger import TrajectoryLogger, TrajectoryPoint
from .unity_exporter import UnityExporter


class ExportManager:
    """
    Manages batch export operations for multiple episodes.
    
    This class aggregates trajectory data from multiple episodes,
    provides statistical analysis, and exports data in various formats
    suitable for analysis and visualization.
    """
    
    def __init__(self,
                 output_directory: str = "exports",
                 auto_compress: bool = True,
                 max_episodes_per_file: int = 100):
        """
        Initialize export manager.
        
        Args:
            output_directory: Directory for exported files
            auto_compress: Automatically compress large exports
            max_episodes_per_file: Maximum episodes per export file
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.auto_compress = auto_compress
        self.max_episodes_per_file = max_episodes_per_file
        
        # Episode storage
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self.episode_summaries: List[Dict[str, Any]] = []
        
        # Export statistics
        self.export_stats = {
            'total_episodes': 0,
            'total_exports': 0,
            'last_export_time': None,
            'total_export_size_mb': 0.0
        }
        
        # Unity exporter
        self.unity_exporter = UnityExporter(
            coordinate_conversion=True,
            optimize_for_unity=True
        )
    
    def add_episode(self,
                   episode_id: str,
                   trajectory_logger: TrajectoryLogger,
                   additional_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an episode to the export manager.
        
        Args:
            episode_id: Unique identifier for the episode
            trajectory_logger: TrajectoryLogger with episode data
            additional_metadata: Optional additional metadata
            
        Returns:
            True if episode added successfully, False otherwise
        """
        try:
            if not trajectory_logger.current_trajectory:
                print(f"Warning: Episode {episode_id} has no trajectory data")
                return False
            
            # Create episode data
            episode_data = {
                'trajectory_data': trajectory_logger.current_trajectory.copy(),
                'metadata': trajectory_logger.trajectory_metadata.copy(),
                'summary': trajectory_logger.get_trajectory_summary(),
                'logging_stats': trajectory_logger.get_logging_performance(),
                'timestamp': time.time()
            }
            
            # Add additional metadata
            if additional_metadata:
                episode_data['metadata'].update(additional_metadata)
            
            # Store episode
            self.episodes[episode_id] = episode_data
            self.episode_summaries.append({
                'episode_id': episode_id,
                'success': episode_data['summary'].get('success', False),
                'final_distance': episode_data['summary'].get('final_distance', 0.0),
                'total_reward': episode_data['summary'].get('total_reward', 0.0),
                'duration': episode_data['summary'].get('duration', 0.0),
                'fuel_consumed': episode_data['summary'].get('fuel_consumed', 0.0),
                'timestamp': episode_data['timestamp']
            })
            
            self.export_stats['total_episodes'] += 1
            
            return True
            
        except Exception as e:
            print(f"Error adding episode {episode_id}: {e}")
            return False
    
    def export_all_episodes(self,
                           formats: List[str] = ['csv', 'json', 'unity'],
                           filename_prefix: str = 'aegis_export') -> Dict[str, str]:
        """
        Export all episodes in specified formats.
        
        Args:
            formats: List of export formats ('csv', 'json', 'unity')
            filename_prefix: Prefix for export filenames
            
        Returns:
            Dictionary mapping format to exported filename
        """
        exported_files = {}
        timestamp = int(time.time())
        
        try:
            # Export CSV format
            if 'csv' in formats:
                csv_filename = f"{filename_prefix}_{timestamp}_episodes.csv"
                csv_path = self.output_directory / csv_filename
                if self._export_episodes_csv(str(csv_path)):
                    exported_files['csv'] = str(csv_path)
            
            # Export JSON format
            if 'json' in formats:
                json_filename = f"{filename_prefix}_{timestamp}_episodes.json"
                json_path = self.output_directory / json_filename
                if self._export_episodes_json(str(json_path)):
                    exported_files['json'] = str(json_path)
            
            # Export Unity format
            if 'unity' in formats:
                unity_filename = f"{filename_prefix}_{timestamp}_unity.json"
                unity_path = self.output_directory / unity_filename
                if self._export_episodes_unity(str(unity_path)):
                    exported_files['unity'] = str(unity_path)
            
            # Create compressed archive if requested
            if self.auto_compress and exported_files:
                archive_filename = f"{filename_prefix}_{timestamp}_archive.zip"
                archive_path = self.output_directory / archive_filename
                if self._create_compressed_archive(list(exported_files.values()), str(archive_path)):
                    exported_files['archive'] = str(archive_path)
            
            self.export_stats['total_exports'] += 1
            self.export_stats['last_export_time'] = time.time()
            
            print(f"Episodes exported successfully: {len(exported_files)} files created")
            
        except Exception as e:
            print(f"Error during export: {e}")
        
        return exported_files
    
    def _export_episodes_csv(self, filename: str) -> bool:
        """Export all episodes to CSV format."""
        try:
            with open(filename, 'w', newline='') as csvfile:
                if not self.episode_summaries:
                    return False
                
                # Write episode summaries
                fieldnames = list(self.episode_summaries[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for summary in self.episode_summaries:
                    writer.writerow(summary)
            
            return True
            
        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return False
    
    def _export_episodes_json(self, filename: str) -> bool:
        """Export all episodes to JSON format."""
        try:
            # Create comprehensive export data
            export_data = {
                'export_info': {
                    'version': '1.0',
                    'export_time': time.time(),
                    'total_episodes': len(self.episodes),
                    'format': 'aegis_intercept_batch_export'
                },
                'statistics': self.get_batch_statistics(),
                'episode_summaries': self.episode_summaries,
                'episodes': {}
            }
            
            # Add episode data (potentially large)
            for episode_id, episode_data in self.episodes.items():
                export_data['episodes'][episode_id] = {
                    'metadata': episode_data['metadata'],
                    'summary': episode_data['summary'],
                    'trajectory_points': len(episode_data['trajectory_data'])
                    # Note: Full trajectory data could be very large
                    # Include only summary for batch exports
                }
            
            with open(filename, 'w') as jsonfile:
                json.dump(export_data, jsonfile, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting JSON: {e}")
            return False
    
    def _export_episodes_unity(self, filename: str) -> bool:
        """Export episodes to Unity-compatible format."""
        try:
            # Limit episodes for Unity export (performance)
            max_episodes = min(10, len(self.episodes))
            selected_episodes = list(self.episodes.items())[:max_episodes]
            
            unity_batch_data = {
                'version': '1.0',
                'format': 'unity_batch_episodes',
                'episode_count': len(selected_episodes),
                'export_time': time.time(),
                'episodes': {}
            }
            
            # Convert each episode to Unity format
            for episode_id, episode_data in selected_episodes:
                unity_data = self.unity_exporter.create_unity_trajectory(
                    episode_data['trajectory_data'],
                    episode_data['metadata']
                )
                unity_batch_data['episodes'][episode_id] = unity_data
            
            with open(filename, 'w') as jsonfile:
                json.dump(unity_batch_data, jsonfile, indent=2)
            
            # Also create Unity config file
            config_filename = filename.replace('.json', '_config.json')
            self.unity_exporter.create_unity_config(config_filename)
            
            return True
            
        except Exception as e:
            print(f"Error exporting Unity format: {e}")
            return False
    
    def _create_compressed_archive(self, file_paths: List[str], archive_path: str) -> bool:
        """Create compressed archive of exported files."""
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in file_paths:
                    if Path(file_path).exists():
                        zipf.write(file_path, Path(file_path).name)
            
            # Calculate archive size
            archive_size_mb = Path(archive_path).stat().st_size / (1024 * 1024)
            self.export_stats['total_export_size_mb'] += archive_size_mb
            
            return True
            
        except Exception as e:
            print(f"Error creating archive: {e}")
            return False
    
    def export_episode_batch(self,
                           episode_ids: List[str],
                           output_filename: str,
                           format_type: str = 'json') -> bool:
        """
        Export a specific batch of episodes.
        
        Args:
            episode_ids: List of episode IDs to export
            output_filename: Output filename
            format_type: Export format ('json', 'csv', 'unity')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Filter episodes
            batch_episodes = {eid: self.episodes[eid] for eid in episode_ids if eid in self.episodes}
            
            if not batch_episodes:
                print("No valid episodes found for batch export")
                return False
            
            if format_type == 'json':
                return self._export_batch_json(batch_episodes, output_filename)
            elif format_type == 'unity':
                return self._export_batch_unity(batch_episodes, output_filename)
            elif format_type == 'csv':
                return self._export_batch_csv(batch_episodes, output_filename)
            else:
                print(f"Unsupported format: {format_type}")
                return False
                
        except Exception as e:
            print(f"Error exporting episode batch: {e}")
            return False
    
    def _export_batch_json(self, episodes: Dict[str, Any], filename: str) -> bool:
        """Export episode batch to JSON."""
        try:
            batch_data = {
                'batch_info': {
                    'episode_count': len(episodes),
                    'export_time': time.time(),
                    'format': 'aegis_intercept_episode_batch'
                },
                'episodes': episodes
            }
            
            with open(filename, 'w') as f:
                json.dump(batch_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting batch JSON: {e}")
            return False
    
    def _export_batch_unity(self, episodes: Dict[str, Any], filename: str) -> bool:
        """Export episode batch to Unity format."""
        try:
            episode_data_list = [(episodes[eid], eid) for eid in episodes.keys()]
            
            # Create mock TrajectoryLogger objects for Unity export
            unity_episodes = []
            for episode_data, episode_id in episode_data_list:
                mock_logger = TrajectoryLogger()
                mock_logger.current_trajectory = episode_data['trajectory_data']
                mock_logger.trajectory_metadata = episode_data['metadata']
                unity_episodes.append((mock_logger, episode_id))
            
            return self.unity_exporter.export_multiple_episodes(unity_episodes, filename)
            
        except Exception as e:
            print(f"Error exporting batch Unity: {e}")
            return False
    
    def _export_batch_csv(self, episodes: Dict[str, Any], filename: str) -> bool:
        """Export episode batch summaries to CSV."""
        try:
            batch_summaries = []
            for episode_id, episode_data in episodes.items():
                summary = episode_data['summary'].copy()
                summary['episode_id'] = episode_id
                batch_summaries.append(summary)
            
            with open(filename, 'w', newline='') as csvfile:
                if batch_summaries:
                    fieldnames = list(batch_summaries[0].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for summary in batch_summaries:
                        writer.writerow(summary)
            
            return True
            
        except Exception as e:
            print(f"Error exporting batch CSV: {e}")
            return False
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all episodes."""
        if not self.episode_summaries:
            return {}
        
        # Extract statistics
        successes = [ep['success'] for ep in self.episode_summaries]
        distances = [ep['final_distance'] for ep in self.episode_summaries]
        rewards = [ep['total_reward'] for ep in self.episode_summaries]
        durations = [ep['duration'] for ep in self.episode_summaries]
        fuel_consumed = [ep['fuel_consumed'] for ep in self.episode_summaries]
        
        statistics = {
            'episode_count': len(self.episode_summaries),
            'success_rate': sum(successes) / len(successes) if successes else 0.0,
            'distance_statistics': {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'median': float(np.median(distances))
            },
            'reward_statistics': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'total': float(np.sum(rewards))
            },
            'duration_statistics': {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations))
            },
            'fuel_statistics': {
                'mean_consumed': float(np.mean(fuel_consumed)),
                'std_consumed': float(np.std(fuel_consumed)),
                'min_consumed': float(np.min(fuel_consumed)),
                'max_consumed': float(np.max(fuel_consumed))
            },
            'export_stats': self.export_stats.copy()
        }
        
        return statistics
    
    def clear_episodes(self):
        """Clear all stored episodes."""
        self.episodes.clear()
        self.episode_summaries.clear()
        self.export_stats['total_episodes'] = 0
    
    def get_episode_count(self) -> int:
        """Get number of stored episodes."""
        return len(self.episodes)
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export manager statistics."""
        return {
            'episodes_stored': len(self.episodes),
            'export_stats': self.export_stats.copy(),
            'batch_statistics': self.get_batch_statistics()
        }
    
    def create_analysis_report(self, output_filename: str) -> bool:
        """Create a comprehensive analysis report."""
        try:
            report_data = {
                'report_info': {
                    'creation_time': time.time(),
                    'report_type': 'aegis_intercept_analysis',
                    'version': '1.0'
                },
                'summary': self.get_batch_statistics(),
                'episode_details': self.episode_summaries,
                'recommendations': self._generate_recommendations()
            }
            
            with open(output_filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"Analysis report created: {output_filename}")
            return True
            
        except Exception as e:
            print(f"Error creating analysis report: {e}")
            return False
    
    def _generate_recommendations(self) -> List[str]:
        """Generate training recommendations based on episode data."""
        if not self.episode_summaries:
            return []
        
        recommendations = []
        stats = self.get_batch_statistics()
        
        # Success rate analysis
        success_rate = stats.get('success_rate', 0.0)
        if success_rate < 0.3:
            recommendations.append("Low success rate detected. Consider reducing curriculum difficulty.")
        elif success_rate > 0.9:
            recommendations.append("High success rate achieved. Consider advancing curriculum level.")
        
        # Distance analysis
        mean_distance = stats.get('distance_statistics', {}).get('mean', 0.0)
        if mean_distance > 50.0:
            recommendations.append("Large intercept distances observed. Focus on improving guidance accuracy.")
        
        # Fuel efficiency analysis
        mean_fuel = stats.get('fuel_statistics', {}).get('mean_consumed', 0.0)
        if mean_fuel > 0.8:
            recommendations.append("High fuel consumption detected. Optimize control efficiency.")
        
        # Training consistency
        reward_std = stats.get('reward_statistics', {}).get('std', 0.0)
        reward_mean = stats.get('reward_statistics', {}).get('mean', 0.0)
        if reward_mean > 0 and reward_std / reward_mean > 0.5:
            recommendations.append("High reward variance observed. Consider stabilizing training parameters.")
        
        return recommendations