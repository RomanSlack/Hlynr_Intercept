# Test Patterns Research for Unity-RL Bridge

## Testing Framework and Structure

### Primary Testing Framework
- **pytest** is used as the primary testing framework
- Tests are organized in `src/phase4_rl/tests/` directory
- Individual test files follow `test_*.py` naming convention
- Test classes use `TestClassName` convention

### Test Organization Patterns

#### Class-based Test Organization
```python
class TestConfigLoader:
    """Test cases for ConfigLoader class."""
    
    def setup_method(self):
        """Setup for each test method."""
        reset_config()
    
    def test_default_config_loading(self):
        """Test loading default configuration."""
        # Test implementation
        pass
```

#### Module-level Test Functions
```python
def test_episode_logging():
    """Test the episode logging functionality."""
    # Direct function testing
    pass
```

## Common Testing Patterns

### Configuration Testing
```python
def test_custom_config_file(self):
    """Test loading from custom configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        custom_config = {
            'environment': {
                'num_missiles': 3,
                'num_interceptors': 2
            }
        }
        yaml.dump(custom_config, f)
        temp_path = f.name
    
    try:
        config = ConfigLoader(temp_path)
        assert config.get('environment.num_missiles') == 3
    finally:
        os.unlink(temp_path)
```

### Temporary File Handling
```python
# Pattern 1: Using tempfile.NamedTemporaryFile
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    temp_path = f.name

try:
    # Test operations
    pass
finally:
    Path(temp_path).unlink(missing_ok=True)

# Pattern 2: Direct pathlib usage
temp_path = Path("temp_test_file.json")
try:
    # Test operations
    pass
finally:
    temp_path.unlink(missing_ok=True)
```

### JSON Schema and Migration Testing
```python
def test_load_episode_data_v0_migration(self):
    """Test loading and migrating v0 (legacy) format episode data."""
    # Create v0 format data (no schema_version)
    v0_data = {
        'metrics': {'test': 'metrics'},
        'step_data': [{'step': 0, 'reward': 1.0}],
        'timestamp': 1234567890.0
    }
    
    # Save and load test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        json.dump(v0_data, f)
    
    try:
        loaded_data = load_episode_data(temp_path)
        
        # Validation assertions
        assert 'schema_version' in loaded_data
        assert loaded_data['schema_version'] == 1
        assert loaded_data['metrics'] == v0_data['metrics']
    finally:
        Path(temp_path).unlink(missing_ok=True)
```

### Error Handling and Edge Cases
```python
def test_nonexistent_config_file(self):
    """Test handling of nonexistent configuration file."""
    config = ConfigLoader('/nonexistent/path/config.yaml')
    
    # Should fall back to default configuration
    assert config.get('environment.num_missiles') == 1
    assert config.get('training.algorithm') == 'PPO'

def test_future_version_warning(self):
    """Test that future schema versions trigger warnings."""
    future_data = {'schema_version': 99, 'metrics': {'test': 'data'}}
    
    # Should trigger warning but still load
    with pytest.warns(UserWarning, match="Unknown schema version 99"):
        loaded_data = load_episode_data(temp_path)
```

### Integration Testing Patterns
```python
def test_episode_logging():
    """Test the episode logging functionality."""
    # Create environment with specific config
    config = {
        'environment': {
            'num_missiles': 1,
            'num_interceptors': 1,
            'max_episode_steps': 100
        }
    }
    
    env = FastSimEnv(
        config=config,
        scenario_name="easy",
        enable_episode_logging=True,
        episode_log_dir="test_runs"
    )
    
    # Run episode and validate outputs
    for episode in range(3):
        obs, info = env.reset(seed=42 + episode)
        # ... run episode
    
    # Validate generated files
    log_dir = Path("test_runs")
    manifest_path = latest_run / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert len(manifest['episodes']) == 3
```

### Import and Basic Functionality Testing
```python
# Test import
try:
    from bridge_server import BridgeServer
    from client_stub import BridgeClient
    print("✅ Bridge server imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test initialization
try:
    server = BridgeServer(
        checkpoint_path=checkpoint_path,
        scenario_name="easy",
        host="localhost",
        port=5001
    )
    print("✅ Bridge server initialization successful")
except Exception as e:
    print(f"❌ Bridge server initialization failed: {e}")
    sys.exit(1)
```

## Assertion Patterns

### Basic Assertions
```python
assert config.get('environment.num_missiles') == 1
assert config.get('training.algorithm') == 'PPO'
assert 'schema_version' in data
assert data['schema_version'] == 1
```

### Collection Assertions
```python
assert len(obs) == 30, f"Expected 30 dimensions, got {len(obs)}"
assert all(isinstance(x, (int, float)) for x in obs), "All values should be numeric"
assert len(manifest['episodes']) == 3
```

### Numerical Assertions with Tolerance
```python
import numpy as np
assert np.allclose(enu_position, enu_pos_back, atol=1e-10)
assert abs(q_norm - 1.0) < 0.001, "Quaternion must be normalized"
```

### Object Identity and Type Assertions
```python
assert config1 is config2  # Same instance
assert config3 is not config1  # Different instance
assert isinstance(result, dict)
```

## Mocking and Patching

### Warning Testing
```python
import pytest

with pytest.warns(UserWarning, match="Unknown schema version 99"):
    loaded_data = load_episode_data(temp_path)
```

### Exception Testing
```python
with pytest.raises(ValidationError):
    invalid_model = SomeModel(invalid_data)

with pytest.raises(ValueError, match="specific error message"):
    function_that_should_fail()
```

## Test Data Generation

### Configuration Data
```python
custom_config = {
    'environment': {
        'num_missiles': 3,
        'num_interceptors': 2
    },
    'training': {
        'learning_rate': 0.001
    }
}
```

### Episode Data
```python
step_data = {
    'step': 0,
    'observation': [1.0, 2.0, 3.0],
    'action': [0.5],
    'reward': 1.0,
    'done': False,
    'info': {}
}
```

### Schema Migration Data
```python
v0_data = {
    'metrics': {'test': 'metrics'},
    'step_data': [{'step': 0, 'reward': 1.0}],
    'timestamp': 1234567890.0
    # Note: no schema_version field
}

v1_data = {
    'schema_version': 1,
    'metrics': {'test': 'metrics'},
    'step_data': [{'step': 0, 'reward': 1.0}],
    'timestamp': 1234567890.0
}
```

## File and Directory Testing

### Directory Structure Validation
```python
log_dir = Path("test_runs")
if log_dir.exists():
    run_dirs = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    episode_files = list(latest_run.glob("ep_*.jsonl"))
    assert len(episode_files) > 0
```

### File Content Validation
```python
with open(episode_files[0]) as f:
    lines = f.readlines()
    header = json.loads(lines[0])
    
    if "meta" in header:
        assert header['meta']['ep_id'].startswith('ep_')
    
    if len(lines) > 10:
        sample = json.loads(lines[10])
        assert 't' in sample
        assert isinstance(sample['t'], (int, float))
```

## Key Testing Principles

### Test Isolation
- Use `setup_method()` and `teardown_method()` for test isolation
- Clean up temporary files and resources
- Reset global state between tests

### Comprehensive Coverage
- Test happy path scenarios
- Test error conditions and edge cases
- Test backwards compatibility and migration
- Test integration between components

### Clear Test Names and Documentation
```python
def test_load_episode_data_v0_migration(self):
    """Test loading and migrating v0 (legacy) format episode data."""
    
def test_nonexistent_config_file(self):
    """Test handling of nonexistent configuration file."""
```

### Validation Patterns
- Validate data structure and types
- Validate numerical ranges and constraints
- Validate file formats and schemas
- Validate error handling behavior

## Test Execution Patterns

### Running Tests
```bash
# Run specific test file
pytest test_config.py -v

# Run all tests in directory
pytest tests/ -v

# Run with specific markers or patterns
pytest -k "test_config" -v
```

### Test Organization for Unity-RL Bridge

Based on existing patterns, tests should be organized as:

```
tests/
├── test_schemas.py          # Pydantic schema validation
├── test_transforms.py       # ENU ↔ Unity coordinate transforms
├── test_normalize.py        # VecNormalize integration
├── test_bridge_server.py    # Flask API endpoints
├── test_episode_logger.py   # JSONL logging functionality
├── test_clamps.py          # Safety constraint validation
├── test_end_to_end.py      # Complete pipeline testing
└── fixtures/               # Test data and configuration files
```