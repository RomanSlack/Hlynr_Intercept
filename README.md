# AegisIntercept – Nuclear-Defense Path-Finding Simulator

A physically realizable Gymnasium-style environment for developing and testing reinforcement learning algorithms in missile interception scenarios. This project provides a sandbox for deterrence-focused RL research, enabling rapid prototyping and evaluation of AI decision-making systems under extreme time constraints.

## Overview

AegisIntercept simulates defensive missile interception scenarios where an AI agent must learn optimal trajectories to intercept incoming threats before they reach defended targets. The environment progresses from simple 2D kinematics to high-fidelity 6-DOF simulations across multiple phases, supporting research into hierarchical reinforcement learning, distributed training, and real-time decision making.

## Collaborators

* **Roman Slack** (RIT Rochester)
* **Quinn Hasse** (UW Madison)

## Phase Roadmap

| Phase | Scope                                                                                                 | Primary Tools                          |
|-------|-------------------------------------------------------------------------------------------------------|----------------------------------------|
| **1** | 2D missile interception with PPO and real-time visualization.                                         | Gymnasium, PyGame, CleanRL             |
| **2** | 3DOF physics, 3D kinematics, adversary movement, togglable headless mode, basic checkpointing.        | Gymnasium VectorEnv, TorchRL           |
| **3** | Full 6DOF, realistic physics (wind, drag, IRL constants), modular scenario randomization, curriculum learning, enhanced logging, refined reward shaping. | Gymnasium, Custom PPO Trainer, WandB   |
| **4** | Scalable distributed training with multiple interceptors, physical body flexibility, and transfer learning benchmarks. | RLlib, Isaac Gym, Optuna               |


## Repository Structure

```
.
├── README.md              # Project documentation
├── pyproject.toml         # Python package configuration
├── .gitignore            # Git ignore patterns
├── aegis_intercept/      # Main Python package
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── aegis_2d_env.py   # 2-D missile vs interceptor simulation
│   └── utils/
│       ├── __init__.py
│       └── maths.py          # Vector math utilities
├── scripts/
│   └── train_ppo_phase1.py   # CleanRL PPO training script
├── tests/
│   └── test_env_basic.py     # Pytest environment tests
├── docs/                     # Reserved for design documentation
└── models/                   # Saved model checkpoints
```

## Quickstart

1. **Setup Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -e .
   # or alternatively:
   # pip install -r requirements.txt
   ```

3. **Run Training**
   ```bash
   python scripts/train_ppo_phase1.py
   ```

4. **Run Tests**
   ```bash
   pytest tests/
   ```

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

---

## ⚖️ Legal and Ethical Disclaimer

This project is a purely academic and simulated environment intended for reinforcement learning research and experimentation. It does not interface with real-world defense systems, targeting software, or weaponized hardware.

The repository does **not** include any classified, restricted, or export-controlled materials. It is intended only for educational and non-military applications.

Use of this software must comply with applicable U.S. export control laws (ITAR, EAR) and international dual-use regulations. The authors explicitly prohibit any use of this project for real-world weaponization, targeting, or autonomous lethal systems.

By using this repository, you agree to use it solely for lawful, ethical, and research-related purposes.

