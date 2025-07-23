"""
Test for diagnostics schema versioning and migration logic.

This test verifies that the diagnostics system properly adds schema_version: 1
to all JSON outputs and can handle migration from legacy (v0) files.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

try:
    from ..diagnostics import (
        Logger, export_to_json, export_to_csv, 
        load_episode_data, load_inference_results,
        _migrate_episode_data_v0_to_v1, _migrate_inference_results_v0_to_v1
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from diagnostics import (
        Logger, export_to_json, export_to_csv,
        load_episode_data, load_inference_results,
        _migrate_episode_data_v0_to_v1, _migrate_inference_results_v0_to_v1
    )


class TestDiagnosticsSchema:
    """Test suite for diagnostics schema versioning."""
    
    def test_logger_save_episode_includes_schema_version(self):
        """Test that Logger.save_episode includes schema_version: 1."""
        logger = Logger()
        logger.reset_episode()
        
        # Log some dummy data
        step_data = {
            'step': 0,
            'observation': [1.0, 2.0, 3.0],
            'action': [0.5],
            'reward': 1.0,
            'done': False,
            'info': {}
        }
        logger.log_step(step_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            logger.save_episode(temp_path)
            
            # Load and verify schema version
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert 'schema_version' in data, "Episode data should include schema_version"
            assert data['schema_version'] == 1, "Schema version should be 1"
            
            # Verify other expected fields are still present
            assert 'metrics' in data, "Episode data should include metrics"
            assert 'step_data' in data, "Episode data should include step_data"
            assert 'timestamp' in data, "Episode data should include timestamp"
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_export_to_json_includes_schema_version(self):
        """Test that export_to_json includes schema_version: 1."""
        # Create dummy results
        results = {
            'test_scenario': [
                {
                    'episode': 0,
                    'total_reward': 10.0,
                    'episode_length': 50,
                    'success': True,
                    'timestamp': 1234567890.0,
                    'metrics': {'test': 'data'}
                }
            ]
        }
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            export_to_json(results, temp_path)
            
            # Load and verify schema version
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert 'schema_version' in data, "Exported data should include schema_version"
            assert data['schema_version'] == 1, "Schema version should be 1"
            
            # Verify other expected fields are still present
            assert 'metadata' in data, "Exported data should include metadata"
            assert 'results' in data, "Exported data should include results"
            
            # Verify metadata structure
            metadata = data['metadata']
            assert 'export_timestamp' in metadata, "Metadata should include export_timestamp"
            assert 'total_scenarios' in metadata, "Metadata should include total_scenarios"
            assert 'total_episodes' in metadata, "Metadata should include total_episodes"
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_episode_data_v1_format(self):
        """Test loading v1 (current) format episode data."""
        # Create v1 format data
        v1_data = {
            'schema_version': 1,
            'metrics': {'test': 'metrics'},
            'step_data': [{'step': 0, 'reward': 1.0}],
            'timestamp': 1234567890.0
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            json.dump(v1_data, f)
        
        try:
            # Load using the new function
            loaded_data = load_episode_data(temp_path)
            
            # Should be identical to original
            assert loaded_data == v1_data, "V1 data should load without modification"
            assert loaded_data['schema_version'] == 1, "Schema version should remain 1"
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_episode_data_v0_migration(self):
        """Test loading and migrating v0 (legacy) format episode data."""
        # Create v0 format data (no schema_version)
        v0_data = {
            'metrics': {'test': 'metrics'},
            'step_data': [{'step': 0, 'reward': 1.0}],
            'timestamp': 1234567890.0
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            json.dump(v0_data, f)
        
        try:
            # Load using the new function
            loaded_data = load_episode_data(temp_path)
            
            # Should have schema_version added
            assert 'schema_version' in loaded_data, "Migrated data should include schema_version"
            assert loaded_data['schema_version'] == 1, "Migrated schema version should be 1"
            
            # Original fields should be preserved
            assert loaded_data['metrics'] == v0_data['metrics'], "Metrics should be preserved"
            assert loaded_data['step_data'] == v0_data['step_data'], "Step data should be preserved"
            assert loaded_data['timestamp'] == v0_data['timestamp'], "Timestamp should be preserved"
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_inference_results_v1_format(self):
        """Test loading v1 (current) format inference results."""
        # Create v1 format data
        v1_data = {
            'schema_version': 1,
            'metadata': {
                'export_timestamp': 1234567890.0,
                'total_scenarios': 1,
                'total_episodes': 1
            },
            'results': {'test': [{'episode': 0}]}
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            json.dump(v1_data, f)
        
        try:
            # Load using the new function
            loaded_data = load_inference_results(temp_path)
            
            # Should be identical to original
            assert loaded_data == v1_data, "V1 data should load without modification"
            assert loaded_data['schema_version'] == 1, "Schema version should remain 1"
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_inference_results_v0_migration(self):
        """Test loading and migrating v0 (legacy) format inference results."""
        # Create v0 format data (no schema_version, no metadata)
        v0_data = {
            'results': {
                'scenario1': [{'episode': 0, 'reward': 10.0}],
                'scenario2': [{'episode': 0, 'reward': 5.0}, {'episode': 1, 'reward': 15.0}]
            },
            'timestamp': 1234567890.0  # Legacy timestamp field
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            json.dump(v0_data, f)
        
        try:
            # Load using the new function
            loaded_data = load_inference_results(temp_path)
            
            # Should have schema_version added
            assert 'schema_version' in loaded_data, "Migrated data should include schema_version"
            assert loaded_data['schema_version'] == 1, "Migrated schema version should be 1"
            
            # Should have metadata added
            assert 'metadata' in loaded_data, "Migrated data should include metadata"
            metadata = loaded_data['metadata']
            assert metadata['total_scenarios'] == 2, "Should count 2 scenarios"
            assert metadata['total_episodes'] == 3, "Should count 3 total episodes"
            assert metadata['export_timestamp'] == 1234567890.0, "Should use legacy timestamp"
            
            # Original results should be preserved
            assert loaded_data['results'] == v0_data['results'], "Results should be preserved"
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_migration_functions_directly(self):
        """Test the migration functions directly."""
        # Test episode data migration
        v0_episode = {
            'metrics': {'reward': 10.0},
            'step_data': [],
            'timestamp': 1234567890.0
        }
        
        migrated_episode = _migrate_episode_data_v0_to_v1(v0_episode)
        assert migrated_episode['schema_version'] == 1, "Should add schema version"
        assert migrated_episode['metrics'] == v0_episode['metrics'], "Should preserve metrics"
        
        # Test inference results migration
        v0_results = {
            'results': {'test': [{'episode': 0}]},
            'timestamp': 1234567890.0
        }
        
        migrated_results = _migrate_inference_results_v0_to_v1(v0_results)
        assert migrated_results['schema_version'] == 1, "Should add schema version"
        assert 'metadata' in migrated_results, "Should add metadata"
        assert migrated_results['metadata']['total_scenarios'] == 1, "Should calculate scenarios"
    
    def test_future_version_warning(self):
        """Test that future schema versions trigger warnings."""
        # Create future version data
        future_data = {
            'schema_version': 99,
            'metrics': {'test': 'data'},
            'step_data': [],
            'timestamp': 1234567890.0
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            json.dump(future_data, f)
        
        try:
            # Should trigger warning but still load
            with pytest.warns(UserWarning, match="Unknown schema version 99"):
                loaded_data = load_episode_data(temp_path)
            
            # Data should load as-is
            assert loaded_data == future_data, "Future version should load unchanged"
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_backwards_compatibility_roundtrip(self):
        """Test full backwards compatibility by creating, saving, and loading data."""
        # Create logger and generate data
        logger = Logger()
        logger.reset_episode()
        
        # Add some test data
        for i in range(3):
            logger.log_step({
                'step': i,
                'observation': [float(i), float(i+1)],
                'action': [0.5],
                'reward': i * 0.1,
                'done': i == 2,
                'info': {'test': f'step_{i}'}
            })
        
        # Save episode
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            episode_path = f.name
        
        # Save inference results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            results_path = f.name
        
        try:
            # Save episode data
            logger.save_episode(episode_path)
            
            # Create and save inference results
            results = {
                'test_scenario': [
                    {
                        'episode': 0,
                        'total_reward': 0.3,
                        'episode_length': 3,
                        'success': True,
                        'timestamp': 1234567890.0,
                        'metrics': logger.get_episode_metrics()
                    }
                ]
            }
            export_to_json(results, results_path)
            
            # Load back using migration functions
            loaded_episode = load_episode_data(episode_path)
            loaded_results = load_inference_results(results_path)
            
            # Verify schema versions
            assert loaded_episode['schema_version'] == 1, "Episode should have schema version 1"
            assert loaded_results['schema_version'] == 1, "Results should have schema version 1"
            
            # Verify data integrity
            assert len(loaded_episode['step_data']) == 3, "Should have 3 steps"
            assert 'metrics' in loaded_episode, "Should have metrics"
            assert 'test_scenario' in loaded_results['results'], "Should have test scenario"
            
        finally:
            Path(episode_path).unlink(missing_ok=True)
            Path(results_path).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])