# Hlynr Intercept System Architecture Report

## Executive Summary

This document captures the essential architecture and design patterns discovered in the Hlynr Intercept missile defense RL system. It serves as the foundation for building a clean, maintainable version that preserves the core innovations while eliminating technical debt and unnecessary complexity.

## Core System Purpose

A defensive missile interception simulator using reinforcement learning to train AI agents that learn optimal trajectories for intercepting incoming threats. The system bridges academic RL research with practical defensive applications through physically realistic 6DOF simulations.

## Essential Components for Clean Rebuild

### 1. The 17-Dimensional Radar Observation System

**Purpose**: Provides a consistent, normalized interface between the physical simulation and the RL policy, mimicking real-world radar-based sensing.

**Core Design**:
- Fixed 17-dimensional vector for 1v1 intercept scenarios
- All values normalized to [-1, 1] range for neural network stability
- Incorporates realistic radar noise and uncertainty
- Deterministic given same inputs and seed

**Observation Components**:
- Relative position to target (3D, range-normalized)
- Relative velocity estimate (3D, normalized)
- Interceptor velocity (3D, body knowledge)
- Interceptor orientation (3D, euler angles)
- Fuel fraction (1D)
- Time-to-intercept estimate (1D)
- Radar lock quality (1D)
- Closing rate (1D)
- Off-axis angle (1D)

**Key Insight**: This observation space abstracts away coordinate system complexities and provides all information needed for interception decisions in a format suitable for neural network processing.

### 2. PPO-Based Training Pipeline

**Purpose**: Train robust interception policies using proven deep RL algorithms.

**Core Architecture**:
- Proximal Policy Optimization (PPO) from Stable Baselines3
- Multi-environment parallel training (8 environments default)
- Continuous 6D action space for interceptor control

**Training Stability Features**:
- Entropy coefficient scheduling for exploration-exploitation balance
- Learning rate scheduling with plateau detection
- Adaptive clip range based on policy update statistics
- Best model checkpointing based on evaluation performance
- VecNormalize for observation and reward scaling

**Key Insight**: PPO provides the right balance of sample efficiency and training stability for this safety-critical domain.

### 3. Inference API System

**Purpose**: Serve trained policies for real-time decision making with guaranteed determinism and safety.

**Two-Layer Architecture**:

**Layer 1: FastAPI Bridge Server**
- RESTful API for external integration (Unity, other simulators)
- Request/response pattern with Pydantic validation
- Coordinate system transformations (ENU ↔ Unity)
- Versioned components for reproducibility
- Prometheus metrics for monitoring

**Layer 2: Core Inference Engine**
- Model loading and warmup
- VecNormalize statistics management
- Deterministic seed handling
- Post-policy safety clamping
- Performance diagnostics

**Key Insight**: Separation of API concerns from inference logic enables flexible deployment while maintaining determinism.

### 4. Coordinate System Management

**Purpose**: Handle transformations between different coordinate conventions while maintaining physical accuracy.

**Supported Systems**:
- ENU Right-Handed (East-North-Up): Standard aerospace convention
- Unity Left-Handed: Game engine convention
- NED Right-Handed: Alternative aerospace convention

**Transformation Pipeline**:
- Versioned transform functions
- Deterministic quaternion conversions
- Proper handling of angular velocities and accelerations
- Validation of physical constraints

**Key Insight**: Clean abstraction of coordinate transforms prevents bugs and enables integration with multiple simulators.

### 5. Safety and Determinism Layer

**Purpose**: Ensure safe, reproducible behavior in deployment.

**Safety Components**:
- Post-policy action clamping with configurable limits
- Fuel consumption constraints
- Maximum acceleration limits
- Gimbal angle restrictions
- Emergency abort conditions

**Determinism Components**:
- Seed management across all random number generators
- Versioned normalization statistics
- Versioned observation computation
- Versioned coordinate transforms
- Deterministic model inference

**Key Insight**: Safety and determinism must be built-in from the ground up, not added as afterthoughts.

### 6. Episode Logging System

**Purpose**: Enable replay, debugging, and analysis of agent behavior.

**Logging Architecture**:
- JSONL format for streaming writes
- Fixed-timestep sampling for uniform playback
- Hierarchical event system
- Manifest generation for episode sets

**Data Captured**:
- Agent states (position, orientation, velocity)
- Control inputs and actions
- Discrete events (interceptions, destructions)
- Performance metrics
- Version information

**Key Insight**: Comprehensive logging enables both debugging and knowledge extraction from trained policies.

### 7. Scenario Management System

**Purpose**: Provide structured difficulty progression and evaluation benchmarks.

**Scenario Components**:
- Spawn configurations
- Environmental conditions (wind, atmospheric density)
- Adversary behavior parameters
- Radar characteristics
- Success criteria

**Difficulty Progression**:
- Easy: Single predictable missile, ideal conditions
- Medium: Moderate evasion, environmental effects
- Hard: Advanced evasion, degraded sensors
- Impossible: Extreme scenarios for robustness testing

**Key Insight**: Structured scenarios enable curriculum learning and standardized evaluation.

## Clean Architecture Vision

### Proposed Module Structure

```
hlynr_clean/
├── core/
│   ├── observations.py      # 17D radar observation
│   ├── transforms.py        # Coordinate systems
│   ├── safety.py           # Safety constraints
│   └── types.py            # Shared type definitions
├── training/
│   ├── ppo_trainer.py      # PPO training loop
│   ├── environments.py     # Gym environment
│   ├── callbacks.py        # Training callbacks
│   └── config.py           # Training configuration
├── inference/
│   ├── engine.py           # Core inference logic
│   ├── api.py              # FastAPI server
│   ├── normalize.py        # VecNormalize management
│   └── metrics.py          # Performance monitoring
├── scenarios/
│   ├── loader.py           # Scenario management
│   └── definitions/        # JSON scenario files
├── logging/
│   ├── episode.py          # Episode logging
│   ├── metrics.py          # Metrics aggregation
│   └── replay.py           # Replay utilities
└── configs/
    ├── training.yaml       # Training hyperparameters
    ├── inference.yaml      # Inference settings
    └── logging.yaml        # Logging configuration
```

### Design Principles for Clean Rebuild

1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Injection**: Configuration and dependencies passed explicitly
3. **Version Everything**: All components that affect determinism are versioned
4. **Fail Fast**: Validate inputs early and explicitly
5. **Pure Functions**: Minimize stateful components, prefer functional design
6. **Type Safety**: Use type hints and Pydantic models throughout
7. **Testability**: Design for unit testing from the start
8. **Documentation**: Every public API fully documented

### Migration Strategy

1. **Phase 1: Core Extraction**
   - Extract 17D observation system
   - Extract coordinate transforms
   - Extract safety constraints
   - Create clean type definitions

2. **Phase 2: Training Pipeline**
   - Clean PPO training loop
   - Simplified environment wrapper
   - Streamlined configuration

3. **Phase 3: Inference System**
   - Minimal inference engine
   - Clean API implementation
   - Metrics and monitoring

4. **Phase 4: Enhancement**
   - Episode logging
   - Scenario management
   - Advanced diagnostics

## Critical Success Factors

### Must Preserve
- 17D radar observation design
- PPO training stability features
- Coordinate transform accuracy
- Safety constraint system
- Deterministic inference
- Episode logging format

### Must Improve
- Module organization and separation
- Configuration management
- Dependency management
- Error handling and validation
- Test coverage
- Documentation

### Can Simplify
- Unity-specific integrations (make pluggable)
- Complex diagnostic features (make optional)
- Multiple checkpoint formats (standardize on one)
- Redundant logging systems (unify)

## Next Steps

1. Create clean project structure
2. Extract and refactor core observation system
3. Implement minimal training loop with essential features
4. Build streamlined inference API
5. Add logging and diagnostics as separate concerns
6. Comprehensive testing and documentation

## Conclusion

The current system contains valuable innovations in RL-based missile defense, particularly the 17D radar observation system and the comprehensive safety/determinism layer. By extracting these core components and rebuilding with clean architecture principles, we can create a maintainable, extensible system suitable for both research and practical applications.

The key is to preserve the domain knowledge and algorithmic innovations while eliminating the technical debt that has accumulated from rapid prototyping. This document provides the blueprint for that transformation.