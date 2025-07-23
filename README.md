# Hlynr Intercept – Missile-Defense Path-Finding Simulator

A physically realizable Gymnasium-style environment for developing and testing reinforcement learning algorithms in missile interception scenarios. This project provides a sandbox for deterrence-focused RL research, enabling rapid prototyping and evaluation of AI decision-making systems under extreme time constraints.

## Overview

Hlynr Intercept simulates defensive missile interception scenarios where an AI agent must learn optimal trajectories to intercept incoming threats before they reach defended targets. The environment progresses from simple 2D kinematics to high-fidelity 6-DOF simulations across multiple phases, supporting research into hierarchical reinforcement learning, distributed training, and real-time decision making.

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


## Phase 4 Usage

Phase 4 provides a complete multi-entity radar-based RL system with deterministic environments, fast simulation, comprehensive diagnostics, and Unity bridge integration.

### Training

Train a new model using the fast simulation environment:

```bash
cd src/phase4_rl
python train_radar_ppo.py --scenario easy --timesteps 100000
```

Available scenarios: `easy`, `medium`, `hard`, `impossible`

For distributed training with multiple environments:
```bash
python train_radar_ppo.py --scenario easy --timesteps 500000 --checkpoint-dir checkpoints --log-dir logs
```

### Inference

Run inference on a trained model:

```bash
cd src/phase4_rl
python run_inference.py checkpoints/phase4_radar_baseline.zip --episodes 50 --scenario easy
```

Multi-scenario evaluation:
```bash
python run_inference.py checkpoints/phase4_radar_baseline.zip --multi-scenario --episodes 20
```

### Unity Bridge Server

Start the bridge server for Unity integration:

```bash
cd src/phase4_rl
python bridge_server.py --checkpoint checkpoints/phase4_radar_baseline.zip --port 5000
```

Test the bridge server:
```bash
python client_stub.py --test-type all --host localhost --port 5000
```

### Diagnostics and Visualization

Generate episode plots from diagnostics data:

```bash
cd src/phase4_rl
python plot_episode.py episode_data.json --plot-type dashboard --output results.png
```

Plot types available: `trajectories`, `rewards`, `distances`, `dashboard`, `all`

### Phase 4 Architecture

- **RadarEnv**: Multi-entity environment with deterministic seeding
- **FastSimEnv**: Headless wrapper for accelerated training  
- **Bridge Server**: HTTP API for Unity integration (`/act`, `/health`, `/stats`)
- **Diagnostics**: JSON logging with matplotlib visualization
- **Scenarios**: Configurable difficulty levels with entity variations

### Key Files

- `radar_env.py` - Core multi-entity radar environment
- `fast_sim_env.py` - Headless training wrapper
- `train_radar_ppo.py` - PPO training with scenario management
- `run_inference.py` - Multi-scenario evaluation
- `bridge_server.py` - Unity integration server
- `diagnostics.py` - Comprehensive logging and analysis
- `plot_episode.py` - Quick visualization helper


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

---

## ⚖️ Legal and Ethical Use Notice

This repository is released under the [Hippocratic License 2.1](./LICENSE), which permits use, modification, and distribution of this software for purposes that do not violate human rights or enable weaponized, military, or surveillance applications.

**This project is for academic, research, and peaceful experimentation only.**

Use of this code in autonomous weapons, missile guidance systems, surveillance infrastructure, or other military/defense-related applications is explicitly prohibited.

If you are unsure whether your intended use violates this principle, do not use this software.


---

## ⚖️ Legal and Ethical Disclaimer

This project is a purely academic and simulated environment intended for reinforcement learning research and experimentation. It does not interface with real-world defense systems, targeting software, or weaponized hardware.

The repository does **not** include any classified, restricted, or export-controlled materials. It is intended only for educational and non-military applications.

Use of this software must comply with applicable U.S. export control laws (ITAR, EAR) and international dual-use regulations. The authors explicitly prohibit any use of this project for real-world weaponization, targeting, or autonomous lethal systems.

By using this repository, you agree to use it solely for lawful, ethical, and research-related purposes.

