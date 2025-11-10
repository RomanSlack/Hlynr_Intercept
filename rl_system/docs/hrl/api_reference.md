# HRL API Reference

## Quick Links

- **Implementation**: See `/Users/quinnhasse/Hlynr_Intercept/rl_system/hrl/` for source code
- **Examples**: See existing HRL modules for usage patterns
- **Config Reference**: See `configs/hrl/*.yaml` for configuration structure

---

## hrl.manager

### class HierarchicalManager

High-level controller coordinating selector and specialists.

**Location**: `hrl/manager.py`

#### Constructor

```python
def __init__(
    self,
    selector_policy: SelectorPolicy,
    specialists: Dict[Option, SpecialistPolicy],
    decision_interval: int = 100,
    enable_forced_transitions: bool = True,
    default_option: Option = Option.SEARCH,
)
```

**Parameters**:
- `selector_policy`: High-level option selector (trained or untrained)
- `specialists`: Dict mapping {SEARCH: search_policy, TRACK: track_policy, TERMINAL: terminal_policy}
- `decision_interval`: Steps between high-level decisions (default: 100 = 1Hz @ 100Hz sim)
- `enable_forced_transitions`: Allow environment state to force option changes
- `default_option`: Initial option on reset

#### Method: select_action

```python
def select_action(
    self,
    full_obs: np.ndarray,
    env_state: Optional[Dict[str, Any]] = None,
    deterministic: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]
```

**Parameters**:
- `full_obs`: 26D or 104D frame-stacked observation
- `env_state`: Dict with keys: `lock_quality`, `distance`, `fuel`, `closing_rate`
- `deterministic`: Use deterministic action selection

**Returns**:
- `action` (np.ndarray): 6D continuous action [thrust_x, thrust_y, thrust_z, angular_x, angular_y, angular_z]
- `info` (Dict): Debug info with keys:
  - `option` (str): Current option name
  - `option_index` (int): 0-2
  - `steps_in_option` (int)
  - `option_switched` (bool)
  - `switch_reason` (str): "forced", "selector", or "continue"
  - `forced_transition` (bool)

**Example**:
```python
from hrl.manager import HierarchicalManager
from hrl.selector_policy import SelectorPolicy
from hrl.specialist_policies import SearchSpecialist, TrackSpecialist, TerminalSpecialist
from hrl.option_definitions import Option

# Load trained specialists
search = SearchSpecialist(model_path="checkpoints/hrl/specialists/search/best/best_model")
track = TrackSpecialist(model_path="checkpoints/hrl/specialists/track/best/best_model")
terminal = TerminalSpecialist(model_path="checkpoints/hrl/specialists/terminal/best/best_model")

# Create selector
selector = SelectorPolicy(obs_dim=7, model_path="checkpoints/hrl/selector/best/best_model")

# Create manager
manager = HierarchicalManager(
    selector_policy=selector,
    specialists={Option.SEARCH: search, Option.TRACK: track, Option.TERMINAL: terminal},
    decision_interval=100,
)

# Use in episode
obs, _ = env.reset()
env_state = {'lock_quality': 0.0, 'distance': 1000.0, 'fuel': 1.0, 'closing_rate': 0.0}
action, info = manager.select_action(obs, env_state, deterministic=True)
print(f"Option: {info['option']}, Action: {action}")
```

#### Method: reset

```python
def reset(self)
```

Reset manager state at episode start. Call this after `env.reset()`.

---

## hrl.selector_policy

### class SelectorPolicy

High-level discrete policy for option selection.

**Location**: `hrl/selector_policy.py`

#### Constructor

```python
def __init__(
    self,
    obs_dim: int,
    n_options: int = 3,
    model_path: Optional[str] = None,
    device: str = 'auto',
)
```

**Parameters**:
- `obs_dim`: Abstract state dimension (typically 7)
- `n_options`: Number of options (default: 3)
- `model_path`: Path to trained model (without .zip extension)
- `device`: 'cuda', 'cpu', or 'auto'

#### Method: predict

```python
def predict(
    self,
    abstract_obs: np.ndarray,
    deterministic: bool = True,
) -> int
```

**Parameters**:
- `abstract_obs`: 7D abstract state vector
- `deterministic`: Use deterministic policy (recommended for inference)

**Returns**:
- `option_index` (int): {0=SEARCH, 1=TRACK, 2=TERMINAL}

**Example**:
```python
from hrl.selector_policy import SelectorPolicy
from hrl.observation_abstraction import abstract_observation

selector = SelectorPolicy(obs_dim=7, model_path="checkpoints/hrl/selector/best/best_model")

full_obs = env.reset()[0]
abstract_state = abstract_observation(full_obs)
option = selector.predict(abstract_state, deterministic=True)
print(f"Selected option: {option}")  # 0, 1, or 2
```

---

## hrl.specialist_policies

### class SpecialistPolicy (Base Class)

Low-level continuous policy specialized for one option.

**Location**: `hrl/specialist_policies.py`

#### Constructor

```python
def __init__(
    self,
    option_type: Option,
    obs_dim: int = 104,
    action_dim: int = 6,
    use_lstm: bool = True,
    lstm_hidden_dim: int = 256,
    model_path: Optional[str] = None,
    device: str = 'auto',
)
```

#### Method: predict

```python
def predict(
    self,
    obs: np.ndarray,
    deterministic: bool = True,
) -> np.ndarray
```

**Returns**: 6D continuous action

#### Method: predict_with_lstm

```python
def predict_with_lstm(
    self,
    obs: np.ndarray,
    lstm_state: Optional[Tuple],
    deterministic: bool = True,
) -> Tuple[np.ndarray, Tuple]
```

**Returns**: (action, new_lstm_state)

### Concrete Specialists

```python
class SearchSpecialist(SpecialistPolicy):
    """Wide-area scanning for target acquisition"""

class TrackSpecialist(SpecialistPolicy):
    """Maintain lock and close distance"""

class TerminalSpecialist(SpecialistPolicy):
    """Final intercept guidance"""
```

**Example**:
```python
from hrl.specialist_policies import TrackSpecialist

track = TrackSpecialist(
    model_path="checkpoints/hrl/specialists/track/best/best_model",
    use_lstm=True,
)

obs = env.reset()[0]
lstm_state = None

for step in range(100):
    action, lstm_state = track.predict_with_lstm(obs, lstm_state, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
```

---

## hrl.observation_abstraction

### Function: abstract_observation

Convert 26D/104D observation to 7D abstract state.

**Location**: `hrl/observation_abstraction.py`

```python
def abstract_observation(full_obs: np.ndarray) -> np.ndarray
```

**Parameters**:
- `full_obs`: 26D base observation or 104D frame-stacked

**Returns**: 7D abstract state vector:
```
[0] distance_to_target (normalized 0-1, 5km max)
[1] closing_rate (normalized -1 to 1, ±500 m/s)
[2] radar_lock_quality (0-1)
[3] fuel_fraction (0-1)
[4] off_axis_angle (normalized -1 to 1, ±π rad)
[5] time_to_intercept (normalized 0-1, 0-10 sec)
[6] relative_altitude (normalized -1 to 1, ±1km)
```

**Example**:
```python
from hrl.observation_abstraction import abstract_observation

full_obs = env.reset()[0]  # 26D
abstract_state = abstract_observation(full_obs)
print(abstract_state.shape)  # (7,)
print(f"Distance: {abstract_state[0]:.2f}, Lock: {abstract_state[2]:.2f}")
```

### Function: extract_env_state_for_transitions

```python
def extract_env_state_for_transitions(full_obs: np.ndarray) -> dict
```

**Returns**: Dict with keys: `lock_quality`, `distance`, `fuel`, `closing_rate`

Used by `HierarchicalManager` for forced transition checks.

---

## hrl.reward_decomposition

### Function: compute_strategic_reward

Strategic reward for selector policy.

**Location**: `hrl/reward_decomposition.py`

```python
def compute_strategic_reward(
    env_state: Dict[str, Any],
    option: Option,
    next_env_state: Dict[str, Any],
    episode_done: bool,
    episode_info: Dict[str, Any],
) -> float
```

**Focus**: Long-term outcomes (intercept success, fuel efficiency, timeout penalties)

### Function: compute_tactical_reward

Tactical reward for specialist policies.

```python
def compute_tactical_reward(
    env_state: Dict[str, Any],
    action: np.ndarray,
    option: Option,
    next_env_state: Dict[str, Any],
    episode_done: bool,
) -> float
```

**Focus**: Option-specific objectives (lock acquisition, distance reduction, precision)

Internally dispatches to:
- `compute_search_reward()` - Lock acquisition bonus
- `compute_track_reward()` - Lock maintenance, distance reduction
- `compute_terminal_reward()` - Miss distance minimization

**Example**:
```python
from hrl.reward_decomposition import compute_tactical_reward
from hrl.option_definitions import Option

env_state = {'lock_quality': 0.5, 'distance': 1000.0, 'fuel': 0.8}
action = np.array([1.0, 0.0, 0.5, 0.1, 0.0, 0.0])
next_state = {'lock_quality': 0.7, 'distance': 950.0, 'fuel': 0.75}

reward = compute_tactical_reward(env_state, action, Option.TRACK, next_state, False)
print(f"Track reward: {reward:.2f}")
```

---

## hrl.option_definitions

### Enum: Option

```python
class Option(IntEnum):
    SEARCH = 0      # Wide-area scanning
    TRACK = 1       # Maintain lock and approach
    TERMINAL = 2    # Final intercept guidance
```

### Constants

```python
FORCED_TRANSITION_THRESHOLDS = {
    'radar_lock_quality_min': 0.3,      # Below: lose track
    'radar_lock_quality_search': 0.7,   # Above: exit search
    'close_range_threshold': 100.0,     # meters
    'terminal_fuel_min': 0.1,           # fraction
}

OPTION_METADATA = {
    Option.SEARCH: {
        'name': 'Search',
        'description': 'Wide-area scanning, large angular changes',
        'expected_duration': 200,  # steps
        'forced_exit_conditions': ['radar_lock_acquired'],
        'color': '#FF6B6B',
    },
    # ... TRACK and TERMINAL similarly
}
```

---

## hrl.hierarchical_env

### Function: create_hrl_env

Create HRL-wrapped environment.

**Location**: `hrl/hierarchical_env.py`

```python
def create_hrl_env(
    config: dict,
    manager: HierarchicalManager,
    return_hrl_info: bool = True,
) -> gym.Env
```

**Returns**: Gymnasium environment with HRL wrapper

**Example**:
```python
from hrl.hierarchical_env import create_hrl_env
from environment import InterceptEnvironment
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Create base environment
base_env = InterceptEnvironment(config)

# Create HRL manager (as shown earlier)
manager = ...

# Wrap with HRL
env = create_hrl_env(config, manager)

obs, info = env.reset()
# Now env.step() uses HRL manager automatically
```

---

## Configuration Structure

### HRL Base Config (`configs/hrl/hrl_base.yaml`)

```yaml
hrl:
  enabled: true
  decision_interval_steps: 100
  default_option: 0  # SEARCH
  enable_forced_transitions: true

  thresholds:
    radar_lock_quality_min: 0.3
    radar_lock_quality_search: 0.7
    close_range_threshold: 100.0
    terminal_fuel_min: 0.1

  selector:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    net_arch: [256, 256]

  specialists:
    search:
      learning_rate: 0.0003
      use_lstm: true
      lstm_hidden_dim: 256
      net_arch: [512, 512, 256]
      ent_coef: 0.02

    track:
      # ... similar structure

    terminal:
      # ... similar structure

  rewards:
    strategic:
      intercept_success: 1000.0
      fuel_efficiency_bonus: 100.0
      timeout_penalty: -100.0

    tactical:
      search:
        lock_acquisition_bonus: 50.0
        lock_improvement: 10.0

      track:
        lock_maintenance_bonus: 2.0
        distance_reduction: 1.0

      terminal:
        proximity_bonus_scale: 10.0
```

**Loading Config**:
```python
import yaml

with open('configs/hrl/hrl_base.yaml') as f:
    config = yaml.safe_load(f)

decision_interval = config['hrl']['decision_interval_steps']
thresholds = config['hrl']['thresholds']
```

---

## Common Patterns

### Full Inference Pipeline

```python
import yaml
import numpy as np
from hrl.manager import HierarchicalManager
from hrl.selector_policy import SelectorPolicy
from hrl.specialist_policies import SearchSpecialist, TrackSpecialist, TerminalSpecialist
from hrl.option_definitions import Option
from environment import InterceptEnvironment

# Load config
with open('configs/hrl/hrl_base.yaml') as f:
    config = yaml.safe_load(f)

# Load specialists
specialists = {
    Option.SEARCH: SearchSpecialist(model_path="checkpoints/hrl/specialists/search/best/best_model"),
    Option.TRACK: TrackSpecialist(model_path="checkpoints/hrl/specialists/track/best/best_model"),
    Option.TERMINAL: TerminalSpecialist(model_path="checkpoints/hrl/specialists/terminal/best/best_model"),
}

# Load selector
selector = SelectorPolicy(obs_dim=7, model_path="checkpoints/hrl/selector/best/best_model")

# Create manager
manager = HierarchicalManager(selector, specialists, decision_interval=100)

# Create environment
env = InterceptEnvironment(config)

# Run episode
obs, info = env.reset()
manager.reset()

total_reward = 0
done = False

while not done:
    # Extract environment state
    env_state = {
        'lock_quality': obs[14],
        'distance': np.linalg.norm(obs[0:3]),
        'fuel': obs[12],
        'closing_rate': obs[15],
    }

    # Get action from manager
    action, hrl_info = manager.select_action(obs, env_state, deterministic=True)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

    if hrl_info['option_switched']:
        print(f"Step {info.get('step', 0)}: Switched to {hrl_info['option']}")

print(f"Episode complete. Total reward: {total_reward:.2f}")
print(f"Intercept success: {info.get('intercept_success', False)}")
```

---

## Error Handling

### Common Errors

**ValueError: Observation shape mismatch**
```python
# Check observation dimension
assert obs.shape in [(26,), (104,)], f"Invalid obs shape: {obs.shape}"
```

**KeyError: Missing env_state key**
```python
# Ensure all required keys present
required_keys = ['lock_quality', 'distance', 'fuel', 'closing_rate']
assert all(k in env_state for k in required_keys)
```

**FileNotFoundError: Model checkpoint not found**
```python
# Check checkpoint path
import os
model_path = "checkpoints/hrl/selector/best/best_model"
assert os.path.exists(f"{model_path}.zip"), f"Model not found: {model_path}"
```

---

## Next Steps

- **Training**: See [training_guide.md](training_guide.md)
- **Architecture**: See [architecture.md](architecture.md)
- **Migration**: See [migration_guide.md](migration_guide.md)
- **Source Code**: See `hrl/*.py` for implementation details
