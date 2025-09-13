#!/usr/bin/env python3
"""
Production inference script for 17-dimensional radar observations.
Generates Unity episodes with the trained 17D model.
"""

import os
import sys
sys.path.insert(0, '../phase4_rl')

from generate_unity_episodes_working import generate_unity_episodes

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Unity episodes with 17D radar model')
    parser.add_argument('--checkpoint', 
                        default='checkpoints_radar17_good/phase4_easy_final.zip',
                        help='Path to model checkpoint')
    parser.add_argument('--vecnorm', 
                        default='checkpoints_radar17_good/vec_normalize_final.pkl',
                        help='Path to VecNormalize stats')  
    parser.add_argument('--scenario', 
                        default='easy',
                        choices=['easy', 'medium', 'hard'],
                        help='Scenario difficulty')
    parser.add_argument('--episodes', 
                        type=int, 
                        default=5,
                        help='Number of episodes to generate')
    parser.add_argument('--output-dir', 
                        default='unity_episodes',
                        help='Output directory for episodes')
    
    args = parser.parse_args()
    
    print("🎯 Generating Unity episodes with 17D radar observations...")
    print(f"📊 Model: {args.checkpoint}")
    print(f"🎮 Scenario: {args.scenario}")
    print(f"📁 Output: {args.output_dir}")
    print()
    
    success = generate_unity_episodes(
        checkpoint_path=args.checkpoint,
        vecnorm_path=args.vecnorm,
        scenario=args.scenario,
        num_episodes=args.episodes,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n✅ Episode generation successful!")
    else:
        print("\n❌ Episode generation failed!")
        sys.exit(1)