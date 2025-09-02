#!/usr/bin/env python3
"""
Production training script for 17-dimensional radar observations.
Ensures consistent observation dimensions throughout training.
"""

import os
import sys
sys.path.insert(0, 'src/phase4_rl')

from train_radar_ppo import main

if __name__ == "__main__":
    # Run the main training with proper path
    main()