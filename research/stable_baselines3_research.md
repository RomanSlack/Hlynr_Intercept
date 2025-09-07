# Stable-Baselines3 Research for Unity-RL Bridge

## Overview

Stable-Baselines3 (SB3) is a comprehensive RL library providing reliable implementations of RL algorithms. Key characteristics:
- Unified structure for all algorithms
- PEP8 compliant with high code coverage
- Type hints throughout
- Tensorboard support
- Extensive documentation

## PPO Algorithm for Inference

### Model Loading
```python
from stable_baselines3 import PPO

# Load model from checkpoint
model = PPO.load(
    path="path/to/checkpoint.zip",
    env=env,  # Optional: new environment
    device="cpu",  # Specify device
    custom_objects=None,  # Skip problematic objects
    print_system_info=True  # Debug compatibility
)
```

### Inference Methods

#### predict() Method
```python
# Basic inference
action, _states = model.predict(
    observation,
    deterministic=True,  # Use deterministic policy
    state=None,         # Hidden states for recurrent policies
    episode_start=True  # For episode-dependent logic
)

# Batch inference (vectorized environment)
observations = np.array([obs1, obs2, obs3])
actions, _states = model.predict(observations, deterministic=True)
```

### Action/Observation Space Support

#### Supported Spaces
- ✅ Discrete actions
- ✅ Box actions (continuous)
- ✅ MultiDiscrete actions
- ✅ MultiBinary actions
- ❌ Dict actions (not supported for policy)

#### Policy Networks
1. **MlpPolicy**: Multi-layer perceptron (most common)
2. **CnnPolicy**: Convolutional neural network
3. **MultiInputPolicy**: Complex observation structures

### Performance Characteristics
- PPO is optimized for CPU inference
- Avoid GPU for simple MLP policies unless using CNN
- Deterministic inference is faster than stochastic

## VecNormalize for Observation Normalization

### Core Functionality
VecNormalize provides moving average normalization for observations and rewards:

```python
from stable_baselines3.common.vec_env import VecNormalize

# Create normalized environment
vec_env = VecNormalize(
    env,
    training=True,        # Update statistics during training
    norm_obs=True,        # Normalize observations
    norm_reward=True,     # Normalize rewards
    clip_obs=10.0,        # Clip observations to [-10, 10]
    clip_reward=10.0,     # Clip rewards to [-10, 10]
    gamma=0.99,           # Discount factor for reward normalization
    epsilon=1e-8          # Small value to avoid division by zero
)
```

### Training vs Evaluation Mode

#### Training Mode
```python
# Enable statistics updates
vec_env.training = True

# Observations are normalized AND statistics are updated
obs = vec_env.reset()
```

#### Evaluation Mode (Inference)
```python
# Disable statistics updates
vec_env.training = False

# Observations are normalized but statistics remain frozen
obs = vec_env.reset()
```

### Saving and Loading Statistics

#### Save Normalization Statistics
```python
# Save current normalization parameters
vec_env.save("vec_normalize.pkl")
```

#### Load Normalization Statistics
```python
# Load previously saved statistics
vec_env = VecNormalize.load(
    "vec_normalize.pkl", 
    venv=base_env  # Original environment
)

# Set to evaluation mode for inference
vec_env.training = False
vec_env.norm_reward = False  # Don't normalize rewards during inference
```

### Normalization Methods

#### Manual Normalization
```python
# Normalize observations without updating statistics
normalized_obs = vec_env.normalize_obs(observations)

# Get original (unnormalized) observations
original_obs = vec_env.get_original_obs()

# Get original rewards
original_rewards = vec_env.get_original_reward()
```

## Complete Inference Pipeline

### Basic Inference Setup
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

def setup_inference_pipeline(checkpoint_path, vec_normalize_path, env_creator):
    """Setup complete inference pipeline with normalization."""
    
    # Create environment
    env = DummyVecEnv([env_creator])
    
    # Load normalization statistics
    if vec_normalize_path:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False      # Freeze statistics
        env.norm_reward = False   # Don't normalize rewards
    
    # Load model
    model = PPO.load(checkpoint_path, env=env)
    
    return model, env

def inference_step(model, env, observation):
    """Single inference step with proper normalization."""
    
    # Reshape for vectorized environment
    obs = np.array(observation).reshape(1, -1)
    
    # Get action (deterministic)
    action, _states = model.predict(obs, deterministic=True)
    
    # Extract action from vectorized format
    action = action[0] if len(action.shape) > 1 else action
    
    return action
```

### Error Handling and Validation

#### Model Loading Validation
```python
def validate_model_compatibility(checkpoint_path):
    """Validate model can be loaded successfully."""
    try:
        # Load with system info for debugging
        model = PPO.load(checkpoint_path, print_system_info=True)
        return True, None
    except Exception as e:
        return False, str(e)

def validate_observation_shape(env, observation):
    """Validate observation matches expected shape."""
    expected_shape = env.observation_space.shape
    obs_array = np.array(observation)
    
    if obs_array.shape != expected_shape:
        raise ValueError(
            f"Observation shape {obs_array.shape} does not match "
            f"expected {expected_shape}"
        )
    
    return obs_array
```

## Performance Optimization

### Inference Speed Optimization
```python
# Use CPU for MLP policies
model = PPO.load(checkpoint_path, device="cpu")

# Use deterministic inference for consistent performance
action, _ = model.predict(obs, deterministic=True)

# Batch multiple observations when possible
batch_obs = np.array([obs1, obs2, obs3])
batch_actions, _ = model.predict(batch_obs, deterministic=True)
```

### Memory Management
```python
# Disable gradient computation for inference
import torch
with torch.no_grad():
    action, _ = model.predict(observation, deterministic=True)
```

## Common Patterns for Unity Integration

### Stateless Inference Server
```python
class InferenceEngine:
    def __init__(self, checkpoint_path, vec_normalize_path=None):
        self.model = None
        self.env = None
        self.load_model(checkpoint_path, vec_normalize_path)
    
    def load_model(self, checkpoint_path, vec_normalize_path):
        """Load model and normalization."""
        # Create dummy environment for loading
        dummy_env = DummyVecEnv([lambda: None])
        
        if vec_normalize_path:
            dummy_env = VecNormalize.load(vec_normalize_path, dummy_env)
            dummy_env.training = False
            dummy_env.norm_reward = False
        
        self.model = PPO.load(checkpoint_path, env=dummy_env)
        self.env = dummy_env
    
    def predict(self, observation):
        """Get action for observation."""
        obs = np.array(observation).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        return action[0] if len(action.shape) > 1 else action
```

## Key Takeaways for Unity-RL Bridge

1. **Use VecNormalize for consistent inference** - Load frozen statistics from training
2. **Set training=False during inference** - Prevent statistics updates
3. **Use deterministic inference** - Consistent, faster than stochastic
4. **Validate observation shapes** - Catch shape mismatches early
5. **Handle vectorized environments properly** - Reshape inputs/outputs correctly
6. **CPU is sufficient for MLP policies** - No GPU needed for simple networks
7. **Save/load normalization separately** - Keep statistics with model checkpoints
8. **Use error handling for model loading** - Gracefully handle compatibility issues