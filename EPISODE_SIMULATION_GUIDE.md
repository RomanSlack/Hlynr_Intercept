# Episode Simulation Guide

## Overview

The enhanced test client now supports realistic episode simulation with configurable scenarios, update rates, and completion conditions. This provides production-grade validation of the inference API and comprehensive data for Unity integration.

## Usage Examples

### Standard Testing (Default)
```bash
# Run standard 10-step test sequence
python test_inference_client.py
```

### Episode Simulation Examples
```bash
# Basic episode simulation (easy scenario, 1Hz)
python test_inference_client.py --episode

# Medium difficulty scenario with 0.5Hz updates
python test_inference_client.py --episode --scenario medium --update-rate 0.5

# Hard scenario with custom 90-second timeout
python test_inference_client.py --episode --scenario hard --duration 90

# Save results to specific file
python test_inference_client.py --episode --scenario easy --output easy_test_results.json
```

### Help and Options
```bash
# Show all available options
python test_inference_client.py --help
```

## Available Scenarios

### Easy Scenario
- **Description**: Basic head-on intercept
- **Initial Distance**: ~8000m
- **Default Update Rate**: 1.0 Hz (1-second intervals)
- **Expected Duration**: 30-60 seconds
- **Success Probability**: High
- **Fuel Consumption**: Low

### Medium Scenario
- **Description**: Crossing trajectory intercept
- **Initial Distance**: ~6300m (with lateral separation)
- **Default Update Rate**: 0.5 Hz (2-second intervals)
- **Expected Duration**: 60-90 seconds
- **Success Probability**: Moderate
- **Fuel Consumption**: Medium

### Hard Scenario
- **Description**: High-speed evasive target with 3D geometry
- **Initial Distance**: ~5200m (with altitude difference)
- **Default Update Rate**: 0.5 Hz (2-second intervals)
- **Expected Duration**: 90-120 seconds
- **Success Probability**: Variable
- **Fuel Consumption**: High

## Completion Conditions

Episodes automatically terminate when any condition is met:

1. **Intercept Success**: Distance ≤ 50m
2. **Miss**: Distance ≥ 10,000m (target diverging)
3. **Fuel Depletion**: Remaining fuel ≤ 5%
4. **Timeout**: Maximum scenario duration exceeded

## Output Files

### Standard Test Output
```
inference_test_responses_YYYYMMDD_HHMMSS.json
```

### Episode Simulation Output
```
episode_simulation_[outcome]_YYYYMMDD_HHMMSS.json
```
Examples:
- `episode_simulation_success_20240125_143022.json`
- `episode_simulation_miss_20240125_143155.json`
- `episode_simulation_timeout_20240125_143401.json`

## Output File Structure

### Episode Metadata
```json
{
  "episode_metadata": {
    "outcome": "SUCCESS",
    "duration": 25.5,
    "total_commands": 26,
    "final_distance": 42.3,
    "fuel_consumed": 0.23,
    "intercept_time": 25.5,
    "performance_metrics": {
      "average_latency_ms": 2.1,
      "max_latency_ms": 8.7,
      "success_rate": 1.0
    }
  }
}
```

### Trajectory Data (Unity-Ready)
```json
{
  "trajectory_data": [
    {
      "t": 0.0,
      "episode_step": 0,
      "blue_state": {
        "pos_m": [0.0, 0.0, 1000.0],
        "vel_mps": [200.0, 0.0, 0.0],
        "fuel_frac": 1.0
      },
      "red_state": {
        "pos_m": [8000.0, 0.0, 1000.0],
        "vel_mps": [-100.0, 0.0, 0.0]
      },
      "command": {
        "rate_cmd_radps": {"pitch": 0.12, "yaw": -0.08, "roll": 0.05},
        "thrust_cmd": 0.75
      },
      "diagnostics": {...},
      "safety": {...}
    }
  ],
  "unity_replay_format": {
    "coordinate_system": "ENU_to_Unity",
    "update_rate_hz": 1.0,
    "total_duration_s": 25.5
  }
}
```

## Performance Monitoring

The system tracks key performance metrics:

- **Latency**: Individual request latency (target: <35ms p95)
- **Update Rate**: Actual vs. target update frequency
- **Success Rate**: Episode completion statistics
- **Fuel Efficiency**: Fuel consumption patterns

## Unity Integration

The episode simulation output is designed for direct Unity import:

1. **Coordinate System**: ENU format with Unity transformation hints
2. **Timing**: Realistic 0.5-1.0 Hz update rates matching operational systems
3. **Interpolation**: Smooth trajectory data for 60fps visualization
4. **Replay**: Complete state history for timeline-based playback

## Troubleshooting

### Episode Fails Immediately
- Check server health with standard test first
- Verify scenario parameters are realistic
- Ensure adequate fuel for scenario duration

### High Latency Warnings
- First request is always slower (model initialization)
- Subsequent requests should be <35ms
- Check system resources and network connectivity

### Unexpected Outcomes
- Review trajectory data for guidance anomalies
- Check fuel consumption patterns
- Verify completion condition thresholds

### No Simulation State Data
- Server may not expose environment state
- Feature gracefully degrades to control commands only
- Check server logs for state extraction errors

## Development and Testing

### Custom Scenarios
Scenarios can be extended by:
1. Modifying the `SCENARIOS` dictionary in the code
2. Adding JSON scenario files in `scenarios/` directory
3. Implementing external scenario loading (future enhancement)

### Performance Benchmarking
Run multiple episodes for statistical analysis:
```bash
# Run easy scenario 10 times for success rate analysis
for i in {1..10}; do
  python test_inference_client.py --episode --scenario easy --output "easy_run_$i.json"
done
```

### Integration Testing
Combine with server startup for automated testing:
```bash
# Start server in background
./start_inference_server.sh &
SERVER_PID=$!

# Wait for startup then run episode
sleep 10
python test_inference_client.py --episode --scenario medium

# Clean up
kill $SERVER_PID
```

## Next Steps

The episode simulation system provides a foundation for:
1. **Batch Testing**: Multiple scenario execution
2. **Statistical Analysis**: Success rate and performance metrics
3. **Unity Visualization**: Real-time and replay capabilities
4. **Regression Testing**: Automated scenario validation
5. **Performance Benchmarking**: Latency and success rate baselines