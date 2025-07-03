# AegisIntercept 6DOF Test Suite

This directory contains a comprehensive test suite for validating the 6DOF (Six Degrees of Freedom) system implementation in Phase 3 of the AegisIntercept project.

## Test Structure

### Core Test Modules

1. **`test_physics_6dof_validation.py`** - Physics Engine Validation
   - Quaternion operations and consistency
   - Numerical integration accuracy and stability
   - Conservation laws validation
   - Aerodynamic modeling verification
   - Performance benchmarks

2. **`test_environment_6dof_validation.py`** - Environment Validation
   - State space consistency and bounds
   - Action space validation across different modes
   - Reward system correctness and stability
   - Episode boundary conditions
   - Observation/action space compatibility

3. **`test_curriculum_validation.py`** - Curriculum Learning Validation
   - Progressive difficulty verification
   - Phase transition logic validation
   - JSON configuration handling
   - Performance tracking accuracy
   - Advancement criteria validation

4. **`test_integration_6dof.py`** - Integration Tests
   - Backward compatibility with Phase 2 (3DOF mode)
   - End-to-end training pipeline functionality
   - Logging system integrity and data capture
   - Unity export validation
   - Component interaction validation

5. **`test_adversary_validation.py`** - Adversary Behavior Tests
   - Evasive pattern verification and execution
   - Threat assessment accuracy and responsiveness
   - Parameter sensitivity and configuration handling
   - Realistic flight dynamics and physics compliance

6. **`test_performance_regression.py`** - Performance & Regression Tests
   - Training convergence validation
   - Computational performance benchmarks
   - Memory usage monitoring and leak detection
   - Regression prevention for Phase 2 capabilities
   - Scalability and efficiency validation

7. **`test_realworld_validation.py`** - Real-World Validation
   - Physics accuracy against known aerospace parameters
   - Trajectory realism and believability
   - Unity visualization accuracy and correctness
   - Curriculum effectiveness in learning speed
   - Environmental condition accuracy

### Infrastructure

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`test_runners.py`** - Automated test runners for different scenarios
- **`pytest.ini`** - Pytest configuration file

## Running Tests

### Quick Start

```bash
# Run basic validation tests (fast)
python tests/test_runners.py quick

# Run complete validation suite
python tests/test_runners.py full

# Run performance benchmarks
python tests/test_runners.py performance

# Run regression prevention tests
python tests/test_runners.py regression
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_physics_6dof_validation.py

# Run tests with specific markers
pytest -m "not slow"  # Exclude slow tests
pytest -m "physics"   # Only physics tests
pytest -m "performance"  # Only performance tests

# Run with verbose output
pytest -v

# Run with coverage (if pytest-cov installed)
pytest --cov=aegis_intercept

# Run in parallel (if pytest-xdist installed)
pytest -n auto
```

### Test Categories

Tests are organized with the following markers:

- `slow` - Tests that may take more than 30 seconds
- `physics` - Physics accuracy and validation tests
- `performance` - Performance benchmarking tests
- `integration` - Component integration tests
- `regression` - Regression prevention tests
- `realworld` - Real-world applicability validation

### CI/CD Integration

```bash
# CI validation (fast, essential tests)
python tests/test_runners.py ci

# Nightly comprehensive test suite
python tests/test_runners.py nightly
```

## Test Results

Test results are automatically saved to the `tests/results/` directory with timestamps. Use the test runner to generate comprehensive reports:

```bash
# Generate test report from recent results
python tests/test_runners.py --generate-report
```

## Environment Requirements

The test suite requires:

- Python 3.8+
- pytest 6.0+
- numpy
- psutil (for performance monitoring)
- matplotlib (for trajectory analysis, optional)

Optional dependencies for enhanced features:
- pytest-cov (for coverage reporting)
- pytest-xdist (for parallel execution)
- pytest-timeout (for test timeouts)

## Key Test Scenarios

### Physics Validation
- Quaternion normalization and consistency
- Integration accuracy against analytical solutions
- Conservation of energy and momentum
- Aerodynamic force scaling verification
- Atmospheric model validation

### Environment Validation
- 3DOF vs 6DOF observation space compatibility
- Action space validation across all modes
- Reward system consistency and realism
- Deterministic behavior with fixed seeds
- Boundary condition handling

### Performance Validation
- Environment step performance (>1000 steps/sec)
- Physics engine performance (>50,000 steps/sec)
- Memory leak detection over extended runs
- Scalability with different world sizes
- Parallel environment performance

### Real-World Validation
- Missile parameters vs real-world systems
- Trajectory realism and intercept geometry
- Unity export data accuracy
- Curriculum learning effectiveness
- Environmental condition effects

## Expected Performance Benchmarks

### Computational Performance
- Environment steps: >1,000 steps/second
- Physics simulation: >50,000 steps/second
- Memory growth: <100MB over 1000 episodes
- Test suite completion: <10 minutes (excluding slow tests)

### Accuracy Requirements
- Quaternion normalization: <1e-10 error
- Position tracking: <1e-6 meter error
- Integration stability: Energy conservation within 10x factor
- Real-world parameter alignment: Within 2x of actual missile specs

## Troubleshooting

### Common Issues

1. **Tests timeout**: Some physics tests may be slow on older hardware
   - Use `-m "not slow"` to exclude long-running tests
   - Increase timeout with `--timeout` option

2. **Memory issues**: Performance tests monitor memory usage
   - Ensure sufficient RAM (4GB+ recommended)
   - Close other applications during testing

3. **Import errors**: Ensure aegis_intercept package is installed
   ```bash
   pip install -e .
   ```

4. **Random test failures**: Some tests use randomness
   - Tests should be deterministic with fixed seeds
   - Report persistent random failures as bugs

### Debugging Tests

```bash
# Run with debugging output
pytest -s -vv tests/test_specific_module.py

# Run single test function
pytest tests/test_physics_6dof_validation.py::TestQuaternionUtils::test_quaternion_normalization

# Stop on first failure
pytest -x

# Start debugger on failure
pytest --pdb
```

## Contributing

When adding new tests:

1. Follow the existing naming convention (`test_*.py`)
2. Use appropriate markers for categorization
3. Include docstrings explaining test purpose
4. Add performance benchmarks where relevant
5. Ensure tests are deterministic and repeatable
6. Update this README if adding new test categories

## Validation Criteria

For the 6DOF system to be considered validated, the test suite should achieve:

- [ ] 100% pass rate on physics validation tests
- [ ] 100% pass rate on environment validation tests
- [ ] 95%+ pass rate on integration tests
- [ ] Performance benchmarks within acceptable ranges
- [ ] No memory leaks detected
- [ ] Backward compatibility maintained
- [ ] Real-world parameter alignment verified

This comprehensive test suite ensures the 6DOF system maintains quality while adding new capabilities and provides confidence in the system's correctness and performance.