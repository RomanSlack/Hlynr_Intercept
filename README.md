# Hlynr Intercept – Radar-Based Missile Defense Simulator

A production-ready RL environment for training interceptor missiles using **realistic radar-only observations**. Based on PAC-3/THAAD interceptor specifications, this system trains agents that have **no direct knowledge** of incoming threats and must rely entirely on simulated radar sensors - just like real-world missile defense systems.

## Overview

Hlynr Intercept simulates authentic defensive missile interception scenarios where AI agents learn to:
- **Search and acquire** targets using realistic radar systems (5000m range, 60° beam width)
- **Track through noise** with range-dependent measurement uncertainty  
- **Intercept under constraints** with fuel limits, thrust vectoring, and 6-DOF physics
- **Handle detection failures** when radar loses lock or targets move outside sensor range

The system features a complete **17-dimensional radar observation space** that provides only sensor-realistic information, making trained policies directly transferable to real hardware without the "sim-to-real gap" of omniscient training environments.

## Collaborators

* **Roman Slack** (RIT Rochester)
* **Quinn Hasse** (UW Madison)

## Current System

**🎯 Production-Ready Implementation**: The **`rl_system/`** directory contains a complete, radar-based missile defense simulator ready for training and deployment.

**📡 Key Features**:
- **Authentic radar physics**: Range limits, beam width constraints, detection failures
- **PAC-3 interceptor modeling**: 500kg mass, 50 m/s² acceleration, realistic fuel consumption
- **17D observation space**: Radar-only with perfect self-state knowledge
- **Multi-scenario training**: Easy → Medium → Hard difficulty progression
- **Production deployment**: FastAPI server + offline batch evaluation


## Repository Structure

```
.
├── README.md                    # Project overview
├── SYSTEM_ARCHITECTURE_REPORT.md # Technical architecture analysis  
├── rl_system/                  # 🚀 PRODUCTION SYSTEM
│   ├── README.md              # Complete usage guide
│   ├── SYSTEM_DESIGN.md       # Technical specifications
│   ├── core.py                # 17D radar observations + safety
│   ├── environment.py         # 6DOF physics simulation
│   ├── train.py               # PPO training with adaptive features
│   ├── inference.py           # FastAPI server + offline evaluation
│   ├── logger.py              # Unified timestamped logging
│   ├── config.yaml            # Main configuration
│   ├── scenarios/             # Easy/Medium/Hard presets
│   ├── Images/                # Documentation assets
│   └── requirements.txt       # Dependencies
├── deprecated/                # Legacy implementations
├── training/                  # Research prototypes (cluttered)
├── hlynr_bridge/             # Unity integration components
└── utilities/                # Episode generation tools
```

## Quick Start

**Navigate to the production system:**
```bash
cd rl_system/
```

**See complete documentation:**
- [`rl_system/README.md`](rl_system/README.md) - Usage guide with examples
- [`rl_system/SYSTEM_DESIGN.md`](rl_system/SYSTEM_DESIGN.md) - Technical specifications

**Train a radar-based interceptor:**
```bash
# Install dependencies
pip install -r requirements.txt

# Train with easy scenario (wide radar beam)
python train.py

# Run inference server
python inference.py --model checkpoints/best --mode server

# Batch evaluation with JSON export
python inference.py --model checkpoints/best --mode offline --scenario hard
```

## System Highlights

* **🎯 Radar-Only Training**: No omniscient observations - interceptors learn to search, acquire, and track targets through realistic sensor limitations
* **🚀 PAC-3 Physics**: Authentic 6DOF dynamics with thrust vectoring, fuel consumption, and environmental effects
* **📡 Progressive Scenarios**: Easy (wide beam) → Medium (standard) → Hard (narrow beam, high noise)
* **⚡ Production Ready**: FastAPI deployment + comprehensive logging for real-world applications

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

