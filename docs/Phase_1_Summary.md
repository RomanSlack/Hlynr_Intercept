
# AegisIntercept: Phase 1 Project Summary

This document provides a summary of the AegisIntercept project at the completion of Phase 1.

## 1. Project Overview

AegisIntercept is a simulation environment designed for developing and testing reinforcement learning (RL) algorithms for missile interception tasks. The project aims to create a realistic sandbox for research in AI-driven defensive systems, focusing on optimal trajectory planning under severe time constraints.

The project is a collaboration between Roman Slack (RIT Rochester) and Quinn Hasse (UW Madison).

### Phase 1 Goals:
- Develop a 2D prototype of the interception environment.
- Implement a Proximal Policy Optimization (PPO) agent.
- Use Gymnasium for the environment and CleanRL for the RL agent.
- Include visualization capabilities using PyGame.

## 2. Key Components

### 2.1. Environment: `Aegis2DInterceptEnv`

- **File:** `aegis_intercept/envs/aegis_2d_env.py`
- **Description:** A 2D continuous control environment built with Gymnasium.
- **State Space:** An 8-dimensional vector representing the positions and velocities of the interceptor and the missile.
- **Action Space:** A 2-dimensional continuous vector for controlling the interceptor's velocity.
- **Reward System:**
    - `+1.0` for a successful interception.
    - `-1.0` if the missile reaches its target.
    - `-0.5` if the interceptor goes out of bounds or the episode times out.
    - A small negative reward per step to encourage efficiency.
    - A small positive reward for reducing the distance to the missile.

### 2.2. Training Script: `train_ppo_phase1.py`

- **File:** `scripts/train_ppo_phase1.py`
- **Description:** A script for training a PPO agent in the `Aegis2DInterceptEnv`.
- **Frameworks:** Uses `CleanRL` for the PPO implementation and `PyTorch` for the neural network.
- **Functionality:**
    - Handles environment creation and vectorization.
    - Defines the PPO agent architecture.
    - Manages the training loop, including data collection, advantage calculation, and policy updates.
    - Supports resuming training from checkpoints.
    - Includes optional real-time visualization.

### 2.3. Testing: `test_env_basic.py`

- **File:** `tests/test_env_basic.py`
- **Description:** A suite of `pytest` tests for the `Aegis2DInterceptEnv`.
- **Coverage:**
    - Environment creation and space validation.
    - `reset` and `step` function behavior.
    - Deterministic seeding.
    - Reward and action clipping.
    - Termination conditions.

## 3. How to Run

### 3.1. Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

### 3.2. Training

-   **Run the training script:**
    ```bash
    python scripts/train_ppo_phase1.py
    ```
-   **To enable visualization:**
    ```bash
    python scripts/train_ppo_phase1.py --visualize
    ```

### 3.3. Testing

-   **Run the test suite:**
    ```bash
    pytest
    ```

## 4. Project Dependencies

The project's dependencies are defined in `pyproject.toml` and `requirements.txt`.

- **Core:**
    - `gymnasium`
    - `numpy`
    - `pygame`
    - `cleanrl`
    - `torch`
    - `stable-baselines3`
- **Testing:**
    - `pytest`
    - `pytest-cov`
- **Development:**
    - `black`
    - `flake8`
    - `mypy`

## 5. Future Work (Phase 2 and Beyond)

The project roadmap outlines the following future phases:

-   **Phase 2:** Transition to a 3D environment using MuJoCo or Gazebo.
-   **Phase 3:** Implement hierarchical reinforcement learning.
-   **Phase 4:** Focus on distributed, high-fidelity simulations.
