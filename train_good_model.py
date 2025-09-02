#!/usr/bin/env python3
"""
Train a GOOD 17-dim radar-only model for single interceptor vs single missile.
This will create a working model like the one that achieved 47m intercept distance.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "phase4_rl"))

# Force single entity mode for 17-dim observations
os.environ['FORCE_SINGLE_ENTITY'] = '1'

from train_radar_ppo import Phase4Trainer


def train_good_radar_model():
    """Train a 17-dim radar-only model."""
    
    print("🎯 Training GOOD 17-dim Radar-Only Model")
    print("=" * 50)
    print("Target: Single interceptor vs single missile (17-dim obs)")
    print("Goal: Achieve 47m intercept performance like successful results")
    print()
    
    # Create trainer with single-entity configuration
    trainer = Phase4Trainer(
        scenario_name="easy",
        config_path=None,  # Use default config
        checkpoint_dir="checkpoints_radar17_good",
        log_dir="logs_radar17_good"
    )
    
    # Override environment config to force single entity
    trainer.scenario_config.update({
        'environment': {
            'num_missiles': 1,        # Single missile
            'num_interceptors': 1,    # Single interceptor
            'max_episode_steps': 1000,
            'observation_type': 'radar_only'
        },
        'spawn': {
            'missile_spawn_positions': [[0, 0, 300]],      # Threat at origin, 300m up
            'interceptor_spawn_positions': [[500, 500, 100]], # Interceptor forward deployed
            'target_positions': [[1000, 1000, 5]]         # Ground target to defend
        }
    })
    
    print("🚀 Starting training...")
    print(f"Checkpoint directory: checkpoints_radar17_good")
    print(f"Log directory: logs_radar17_good")
    print()
    print("Monitor with: tensorboard --logdir logs_radar17_good")
    print()
    
    # Train the model
    trainer.train(
        total_timesteps=1000000,  # 1M steps for good convergence
        seed=42                   # Reproducible training
    )
    
    print()
    print("✅ Training completed!")
    print()
    print("🧪 Test the trained model:")
    print("cd src/phase4_rl")
    print("python run_inference.py ../../checkpoints_radar17_good/phase4_easy_final.zip --scenario easy --episodes 5")
    print()
    print("🎮 Generate Unity episodes:")
    print("python generate_unity_episodes_working.py \\")
    print("  --checkpoint ../../checkpoints_radar17_good/phase4_easy_final.zip \\")
    print("  --vecnorm ../../checkpoints_radar17_good/vec_normalize.pkl \\")
    print("  --episodes 10")


if __name__ == "__main__":
    train_good_radar_model()