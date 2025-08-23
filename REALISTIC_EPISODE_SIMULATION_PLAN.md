# Realistic Episode Simulation Enhancement Plan

## Overview

Enhance the test inference client to run complete missile engagement scenarios at realistic update frequencies (0.5-1.0 Hz) until mission completion, providing production-ready validation of the inference API and comprehensive scenario data for Unity integration.

## Core Design Principles

- **Non-Destructive**: Preserve existing 10-step test functionality as default mode
- **Realistic Physics**: 1-2 Hz update rate matching real missile guidance systems
- **Production-Grade**: Robust error handling, comprehensive logging, and validation
- **Unity-Ready**: Output format optimized for visualization and replay

## Functional Requirements

### Episode Management
- **Scenario-Based**: Load predefined engagement scenarios (easy/medium/hard)
- **Completion Conditions**: Intercept success, miss, timeout, fuel depletion
- **State Persistence**: Track full trajectory and decision history
- **Multiple Episodes**: Run sequential scenarios with different parameters

### Update Frequency
- **Command Rate**: 0.5-1.0 second intervals (configurable)
- **Physics Integration**: Allow missile dynamics to evolve between commands
- **Realistic Timing**: Match operational missile guidance frequencies

### End Conditions
```python
COMPLETION_CONDITIONS = {
    "intercept_success": {"distance_threshold": 50.0, "outcome": "SUCCESS"},
    "miss_diverging": {"distance_threshold": 10000.0, "outcome": "MISS"},
    "fuel_depletion": {"fuel_threshold": 0.05, "outcome": "FUEL_OUT"},
    "timeout": {"max_duration": 120.0, "outcome": "TIMEOUT"}
}
```

## Technical Architecture

### Command Line Interface
```bash
# Existing functionality (preserved)
python test_inference_client.py

# New episode simulation mode
python test_inference_client.py --episode --scenario=easy --update-rate=1.0
python test_inference_client.py --episode --scenario=medium --duration=90
```

### Episode Configuration
- **Scenario Files**: JSON-based scenario definitions with initial conditions
- **Parameter Overrides**: Command-line configuration of key parameters
- **Reproducibility**: Deterministic scenarios with configurable random seeds

### Data Collection
- **Enhanced Logging**: Full trajectory data with physics interpolation
- **Performance Metrics**: Mission success rate, time-to-intercept, fuel efficiency
- **Unity Assets**: Generate Unity-compatible replay files automatically

## Implementation Strategy

### Phase 1: Core Episode Engine
- Add `--episode` flag with backward compatibility
- Implement realistic timing loop with configurable update rates
- Create robust completion condition checking
- Enhance JSON output with trajectory metadata

### Phase 2: Scenario System
- Define scenario file format and loading mechanism
- Implement configurable initial conditions (positions, velocities, fuel)
- Add difficulty progression (easy → hard scenarios)
- Create scenario validation and error handling

### Phase 3: Production Features
- Add comprehensive metrics and analytics
- Implement multi-episode batch testing
- Create Unity replay file generation
- Add performance benchmarking and reporting

## Expected Outputs

### Episode Data Files
```json
{
  "episode_metadata": {
    "scenario": "easy",
    "outcome": "SUCCESS", 
    "duration": 25.5,
    "total_commands": 26,
    "final_distance": 42.3
  },
  "trajectory_data": [
    {"t": 0.0, "blue_pos": [...], "red_pos": [...], "command": {...}},
    {"t": 1.0, "blue_pos": [...], "red_pos": [...], "command": {...}}
  ],
  "performance_metrics": {
    "intercept_time": 25.5,
    "fuel_consumed": 0.23,
    "average_latency": 2.1
  }
}
```

### Scenario Files
```json
{
  "scenario_name": "easy",
  "description": "Basic head-on intercept",
  "initial_conditions": {
    "blue": {"pos_m": [0, 0, 1000], "vel_mps": [200, 0, 0], "fuel_frac": 1.0},
    "red": {"pos_m": [8000, 0, 1000], "vel_mps": [-100, 0, 0]},
    "update_rate_hz": 1.0,
    "max_duration_s": 60.0
  }
}
```

## Integration Points

### Unity Compatibility
- **Coordinate Systems**: Automatic ENU↔Unity transformations preserved
- **Replay Format**: Direct import into Unity timeline systems
- **Interpolation**: Smooth trajectory data for 60fps visualization
- **Event Markers**: Mission phases and decision points clearly marked

### Validation Framework
- **API Testing**: Comprehensive endpoint validation maintained
- **Scenario Coverage**: Test suite covering engagement envelope
- **Performance Validation**: Latency and success rate benchmarks
- **Regression Testing**: Automated scenario replay for CI/CD

## Benefits

### For Development
- **Comprehensive Testing**: Full engagement scenario validation
- **Performance Insights**: Real-world latency and success metrics
- **Debug Capabilities**: Detailed trajectory analysis for policy tuning

### For Unity Integration
- **Ready-to-Use Data**: Complete trajectory files for immediate visualization
- **Realistic Timing**: Update rates matching actual operational requirements
- **Scenario Library**: Pre-validated engagement scenarios for demonstration

### For Production
- **Operational Validation**: Proof of end-to-end system performance
- **Benchmarking**: Quantifiable metrics for system evaluation
- **Documentation**: Comprehensive scenario data for stakeholder review