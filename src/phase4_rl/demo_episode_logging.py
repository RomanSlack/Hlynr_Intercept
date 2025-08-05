#!/usr/bin/env python3
"""
Demo script showing how to use the episode logging system for Unity replay.

This demonstrates:
1. Training with episode logging enabled
2. Running inference with logging
3. Viewing the generated log files
"""

import argparse
from pathlib import Path
import json


def demo_training_with_logging():
    """Show how to train with episode logging enabled."""
    print("=" * 60)
    print("TRAINING WITH EPISODE LOGGING")
    print("=" * 60)
    
    print("\nTo train with episode logging enabled during evaluation:")
    print("python train_radar_ppo.py --scenario easy --timesteps 10000 --enable-episode-logging")
    
    print("\nOr configure in config.yaml:")
    print("""
episode_logging:
  enabled: true
  output_dir: "runs"
  log_during_training: false  # Don't log every training episode
  log_during_eval: true       # Log evaluation episodes
  log_interval: 100           # If training logging enabled, log every N episodes
  coordinate_frame: "ENU_RH"  # Right-handed East-North-Up for Unity
  sampling_rate_hz: 100       # 100 Hz sampling rate
""")


def demo_inference_with_logging():
    """Show how to run inference with logging."""
    print("\n" + "=" * 60)
    print("INFERENCE WITH EPISODE LOGGING")
    print("=" * 60)
    
    print("\nTo run inference with episode logging:")
    print("python run_inference.py --checkpoint checkpoints/phase4_easy_final.zip --episodes 5 --enable-episode-logging")


def examine_log_structure():
    """Show the structure of generated logs."""
    print("\n" + "=" * 60)
    print("EPISODE LOG STRUCTURE")
    print("=" * 60)
    
    print("\nGenerated file structure:")
    print("""
runs/
└── run_2025-08-04-123456/
    ├── manifest.json         # Index of all episodes
    ├── ep_000001.jsonl      # Episode 1 data (JSON Lines)
    ├── ep_000002.jsonl      # Episode 2 data
    └── ...
""")
    
    print("\nManifest.json structure:")
    manifest_example = {
        "schema_version": "1.0",
        "coord_frame": "ENU_RH",
        "units": {
            "pos": "m",
            "vel": "m/s",
            "ang": "rad",
            "time": "s"
        },
        "dt_nominal": 0.01,
        "gravity": [0.0, 0.0, -9.81],
        "episodes": [
            {
                "id": "ep_000001",
                "file": "ep_000001.jsonl",
                "seed": 12345,
                "scenario": "easy",
                "notes": "FastSimEnv training episode"
            }
        ]
    }
    print(json.dumps(manifest_example, indent=2))
    
    print("\nEpisode file structure (ep_XXXXXX.jsonl):")
    print("\nLine 1 - Header with metadata:")
    header = {
        "t": 0.0,
        "meta": {
            "ep_id": "ep_000001",
            "seed": 12345,
            "coord_frame": "ENU_RH",
            "dt_nominal": 0.01,
            "scenario": "easy"
        },
        "scene": {
            "interceptor_0": {
                "mass_kg": 10.0,
                "max_torque": [8000, 8000, 2000],
                "sensor_fov_deg": 30.0,
                "max_thrust_n": 1000.0
            },
            "threat_0": {
                "type": "ballistic",
                "mass_kg": 100.0,
                "aim_point": [0, 0, 0]
            }
        }
    }
    print(json.dumps(header, indent=2))
    
    print("\nLine 2+ - Timestep data:")
    timestep = {
        "t": 1.23,
        "agents": {
            "interceptor_0": {
                "p": [100.5, 200.3, 0.0],      # position (m)
                "q": [1.0, 0.0, 0.0, 0.0],     # quaternion
                "v": [10.2, 5.1, 0.0],         # velocity (m/s)
                "w": [0.0, 0.0, 0.1],          # angular velocity (rad/s)
                "u": [0.5, 0.3, 0.0, 0.0, 0.0, 0.8],  # action vector
                "fuel_kg": 85.3,
                "status": "active"
            },
            "threat_0": {
                "p": [500.1, 600.2, 0.0],
                "q": [0.924, 0.0, 0.0, 0.383],
                "v": [-50.0, -40.0, 0.0],
                "w": [0.0, 0.0, 0.0],
                "status": "active"
            }
        },
        "events": []
    }
    print(json.dumps(timestep, indent=2))
    
    print("\nFinal line - Summary:")
    summary = {
        "t": 12.37,
        "summary": {
            "outcome": "hit",
            "miss_distance_m": 2.1,
            "impact_time_s": 11.94,
            "episode_duration": 12.37,
            "notes": "Episode ended after 124 steps"
        }
    }
    print(json.dumps(summary, indent=2))


def unity_integration_notes():
    """Notes on Unity integration."""
    print("\n" + "=" * 60)
    print("UNITY INTEGRATION NOTES")
    print("=" * 60)
    
    print("""
Coordinate System Mapping:
- Python uses ENU_RH (East-North-Up, Right-Handed)
  - X = East
  - Y = North  
  - Z = Up
  
- Unity mapping from ENU:
  - Unity.X = Python.X (East)
  - Unity.Y = Python.Z (Up)
  - Unity.Z = Python.Y (North)

Key Features:
- Fixed timestep sampling (100 Hz default)
- Quaternion rotations (w,x,y,z format)
- All units in SI (meters, seconds, radians, kg, newtons)
- Supports multiple interceptors/threats (future extension)
- Events system for discrete occurrences (lock acquisition, impacts)

Unity Replayer should:
1. Parse manifest.json to get episode list
2. Load selected episode JSONL file
3. Read line-by-line for streaming playback
4. Interpolate between timesteps if needed
5. Map coordinates from ENU to Unity space
6. Render entities with trail visualization
7. Display HUD with fuel, velocity, status info
""")


def main():
    """Run the demo."""
    parser = argparse.ArgumentParser(description='Episode Logging Demo')
    parser.add_argument('--section', choices=['all', 'training', 'inference', 'structure', 'unity'],
                        default='all', help='Which section to show')
    args = parser.parse_args()
    
    sections = {
        'training': demo_training_with_logging,
        'inference': demo_inference_with_logging,
        'structure': examine_log_structure,
        'unity': unity_integration_notes
    }
    
    if args.section == 'all':
        for func in sections.values():
            func()
    else:
        sections[args.section]()
    
    print("\n" + "=" * 60)
    print("For more information, see episode_logger.py")
    print("=" * 60)


if __name__ == "__main__":
    main()