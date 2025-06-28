# AegisIntercept – Nuclear-Defense Path-Finding Simulator

A physically realizable Gymnasium-style environment for developing and testing reinforcement learning algorithms in missile interception scenarios. This project provides a sandbox for deterrence-focused RL research, enabling rapid prototyping and evaluation of AI decision-making systems under extreme time constraints.

## Overview

AegisIntercept simulates defensive missile interception scenarios where an AI agent must learn optimal trajectories to intercept incoming threats before they reach defended targets. The environment progresses from simple 2D kinematics to high-fidelity 6-DOF simulations across multiple phases, supporting research into hierarchical reinforcement learning, distributed training, and real-time decision making.

## Collaborators

* **Roman Slack** (RIT Rochester)
* **Quinn Hasse** (UW Madison)

## Phase Roadmap

| Phase | Scope | Primary Tools |
|-------|-------|---------------|
| **1** | 2-D prototype, flat PPO | Gymnasium, CleanRL, PyGame |
| **2** | 3-D kinematics & basic visual | Matplotlib, CleanRL, PyTorch |
| **3** | Hierarchical RL (manager/worker) | SB3 + HIRO or Tianshou |
| **4** | Distributed, high-fidelity 6-DOF, flexible body | RLlib, Isaac Gym, Optuna |

## Architecture

```
+-------------------------+
|   train_ppo_phase2.py   |
| (CleanRL, PyTorch)      |
+-----------+-------------+
            |
            v
+-----------+-------------+
|   Aegis3DInterceptEnv   |
| (Gymnasium)             |
+-----------+-------------+
            |
            v
+-----------+-------------+
|      physics3d.py       |
| (Numpy)                 |
+-------------------------+
```

## Phase 2: 3-DOF Simulation

* **3-DOF Expansion:** The environment is now a 3D simulation with simple kinematics and linear drag.
* **Adversary Evasion:** The adversary missile now has a basic sine-wave based evasion routine.
* **PPO-GPU Training:** The PPO agent is updated for the 3D environment and optimized for GPU training with PyTorch.

### GPU Optimization

The training script `train_ppo_phase2.py` is optimized for NVIDIA GPUs supporting CUDA. For RTX 40 series cards, you can expect significant performance gains. The script uses mixed-precision training (`torch.cuda.amp`) to accelerate training while maintaining model accuracy.

## Quickstart

See `RUNNING.md` for a quick start guide.

## Phase 1 Features

* **Environment**: 2D continuous missile interception with realistic kinematics
* **Agent**: PPO-based interceptor with continuous action space
* **Visualization**: Real-time PyGame rendering for debugging and analysis
* **Testing**: Comprehensive pytest suite for environment validation

## License & Citation

MIT License (see LICENSE file)

When using this work in research, please cite:
```
@misc{aegis_intercept_2025,
  title={AegisIntercept: Nuclear-Defense Path-Finding Simulator},
  author={Slack, Roman and Hasse, Quinn},
  year={2025},
  url={https://github.com/rslack/aegis_intercept}
}
```