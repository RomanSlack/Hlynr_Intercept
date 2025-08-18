name: "Unity-RL Inference API Bridge - Production Implementation"
description: |
  Complete PRP for implementing Unity↔RL Inference API Bridge with deterministic transforms,
  episode logging, safety clamps, and sub-20ms latency. Extends existing bridge_server.py 
  infrastructure for Unity integration.

---

## Goal

Build a production-ready Unity↔RL Inference API Bridge that enables Unity simulation to control interceptor missiles using trained Phase 4 RL policies in real-time with deterministic, low-latency inference.

### End State
- **HTTP/JSON API v1.0** serving Unity requests with sub-20ms latency
- **Deterministic ENU↔Unity coordinate transforms** with versioned compatibility  
- **Complete ML pipeline**: Unity state → VecNormalize → PPO → safety clamps → Unity commands
- **Episode JSONL logging** compatible with Unity replay system
- **Production monitoring** with health checks, metrics, and error handling

## Why

- **Real-time Unity Integration**: Enable Unity simulation to use trained RL policies for missile guidance
- **Research Reproducibility**: Deterministic inference ensures consistent results across runs
- **Unity Replay System**: Episode logging enables visualization and analysis in Unity
- **Production Readiness**: Robust API with monitoring, validation, and error handling
- **Performance Requirements**: Sub-20ms latency enables real-time 100Hz control loops

## What

Unity sends missile state data via HTTP POST to Python bridge server, which:
1. Transforms Unity coordinates (left-handed) to ENU (right-handed)
2. Normalizes observations using frozen VecNormalize statistics  
3. Gets deterministic action from trained PPO policy
4. Applies safety clamps to rate and thrust commands
5. Transforms action back to Unity coordinates
6. Logs timestep data in JSONL format for Unity replay
7. Returns action with diagnostics and safety metadata

### Success Criteria
- [ ] API responds to Unity requests in <20ms p50, <35ms p95
- [ ] Deterministic: same request returns byte-identical response
- [ ] Safety: all commands clamped to safe ranges with logging
- [ ] Unity replay: episode JSONL files loadable in Unity visualization
- [ ] Testing: 100% pass rate on transform round-trips and validation
- [ ] Production: health/metrics endpoints, structured logging, error handling

## All Needed Context

### Documentation & References
```yaml
# MUST READ - All context needed for implementation

- docfile: research/flask_api_research.md
  why: Flask patterns for JSON APIs, CORS setup, error handling, request validation

- docfile: research/pydantic_validation_research.md  
  why: Schema validation, field validators, nested models, Flask integration

- docfile: research/stable_baselines3_research.md
  why: PPO inference, VecNormalize patterns, model loading, performance optimization

- docfile: research/jsonl_format_research.md
  why: Episode logging format, streaming patterns, Unity integration

- docfile: research/unity_coordinates_research.md
  why: ENU↔Unity transforms, quaternion handling, coordinate validation

- docfile: research/test_patterns_research.md
  why: Testing conventions, file handling, validation patterns

- file: examples/UNITY_DATA_REFERENCE.md
  why: Episode data contract, coordinate system, file structure specification

- file: examples/Unity_RL_API_Integration_Feasibility.md
  why: Communication protocol, system mapping, integration strategy  

- file: examples/Unity_RL_Inference_API_Feasibility.md
  why: Performance targets, existing infrastructure, extension patterns

- file: src/phase4_rl/bridge_server.py
  why: Existing Flask API patterns, model loading, inference pipeline to extend

- file: src/phase4_rl/episode_logger.py  
  why: JSONL logging patterns, episode structure, file organization

- url: https://flask.palletsprojects.com/en/3.0.x/quickstart/
  section: JSON handling, error handlers, request validation
  critical: request.get_json(), jsonify(), @app.errorhandler() patterns

- url: https://docs.pydantic.dev/latest/concepts/models/
  section: BaseModel, field validation, JSON serialization
  critical: @field_validator, model_validate(), model_dump_json()

- url: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
  section: PPO.predict(), deterministic inference, VecNormalize integration
  critical: predict(obs, deterministic=True), VecNormalize.load() patterns
```

### Current Codebase Structure
```bash
src/phase4_rl/
├── bridge_server.py          # Existing Flask API - EXTEND THIS
├── episode_logger.py         # JSONL logging - INTEGRATE  
├── config.py                 # Configuration management
├── scenarios/                # Scenario definitions
├── tests/                    # Testing patterns
│   ├── test_config.py        # Configuration testing
│   ├── test_diagnostics_*.py # Schema validation testing
│   └── test_*.py             # Various test patterns
├── my_runs/                  # Episode outputs
└── checkpoints/              # Model files

research/                     # Generated research files
├── flask_api_research.md
├── pydantic_validation_research.md
├── stable_baselines3_research.md
├── jsonl_format_research.md
├── unity_coordinates_research.md
└── test_patterns_research.md
```

### Desired Codebase Structure with New Files
```bash
src/hlynr_bridge/             # NEW MODULE - Unity bridge implementation
├── __init__.py               # Module initialization
├── server.py                 # Main Flask application with Unity endpoints
├── schemas.py                # Pydantic request/response schemas
├── transforms.py             # ENU↔Unity coordinate transformations  
├── normalize.py              # VecNormalize integration
├── policy.py                 # PPO policy inference wrapper
├── clamps.py                 # Safety constraint enforcement
├── episode_logger.py         # Unity-specific episode logging
├── config.py                 # Bridge-specific configuration
└── health.py                 # Health checks and metrics

tests/                        # NEW TEST SUITE
├── test_transforms.py        # Coordinate transformation tests
├── test_schemas.py           # Pydantic schema validation tests  
├── test_normalize.py         # VecNormalize integration tests
├── test_clamps.py            # Safety constraint tests
├── test_end_to_end.py        # Complete pipeline tests
└── fixtures/                 # Test data and configurations
    ├── sample_unity_request.json
    ├── sample_rl_observation.json
    └── test_configs.yaml
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: Unity coordinate system differences
# Unity: Left-handed, Y-up (X=right, Y=up, Z=forward)  
# ENU: Right-handed, Z-up (X=east, Y=north, Z=up)
# Mapping: Unity.X=ENU.X, Unity.Y=ENU.Z, Unity.Z=ENU.Y

# CRITICAL: Quaternion order differences
# Unity: [x,y,z,w] (vector first, scalar last)
# Standard: [w,x,y,z] (scalar first, vector last)

# CRITICAL: VecNormalize must be frozen during inference
vec_normalize.training = False  # Prevent statistics updates
vec_normalize.norm_reward = False  # Don't normalize rewards

# CRITICAL: Flask CORS must allow Unity client origins
CORS(app, resources={r"/api/*": {"origins": ["http://unity-client"]}})

# CRITICAL: Pydantic v2 syntax (not v1)
# Use @field_validator instead of @validator
# Use model_validate() instead of parse_obj()

# CRITICAL: Stable-Baselines3 vectorized environment handling
obs = observation.reshape(1, -1)  # Reshape for vectorized env
action, _ = model.predict(obs, deterministic=True)
action = action[0] if len(action.shape) > 1 else action  # Extract from vector
```

## Implementation Blueprint

### Data Models and Structure

Create comprehensive Pydantic schemas for type safety and validation:

```python
# schemas.py - Complete API contract
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional

class Position3D(BaseModel):
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate") 
    z: float = Field(..., description="Z coordinate")

class Quaternion(BaseModel):
    w: float = Field(..., description="Scalar component")
    x: float = Field(..., description="X vector component")
    y: float = Field(..., description="Y vector component")
    z: float = Field(..., description="Z vector component")
    
    @field_validator('*')
    @classmethod
    def validate_finite(cls, v):
        if not (-10.0 <= v <= 10.0):
            raise ValueError('Quaternion components must be finite')
        return v

class UnityMissileState(BaseModel):
    position: Position3D
    velocity: Position3D
    angular_velocity: Position3D
    orientation: Quaternion
    fuel_remaining: float = Field(ge=0.0, le=100.0)
    thrust_current: float = Field(ge=0.0)
    target_lock: bool
    target_position: Optional[Position3D] = None
    target_distance: Optional[float] = Field(ge=0.0, default=None)

class UnityAPIRequest(BaseModel):
    meta: Dict[str, Any] = Field(..., description="Episode metadata")
    frames: Dict[str, Any] = Field(..., description="Coordinate frame info")
    blue: UnityMissileState = Field(..., description="Interceptor state")
    red: Dict[str, Any] = Field(..., description="Threat state")
    guidance: Dict[str, Any] = Field(..., description="Guidance data")
    env: Dict[str, Any] = Field(..., description="Environment data")
    normalization: Dict[str, str] = Field(..., description="Normalization IDs")
    
    @field_validator('normalization')
    @classmethod
    def validate_normalization_ids(cls, v):
        required = ['obs_version', 'vecnorm_stats_id', 'transform_version']
        for field in required:
            if field not in v:
                raise ValueError(f'Missing required field: {field}')
        return v

class RateCommand(BaseModel):
    pitch: float = Field(..., ge=-5.0, le=5.0, description="Pitch rate rad/s")
    yaw: float = Field(..., ge=-5.0, le=5.0, description="Yaw rate rad/s") 
    roll: float = Field(..., ge=-5.0, le=5.0, description="Roll rate rad/s")

class ActionCommand(BaseModel):
    rate_cmd_radps: RateCommand
    thrust_cmd: float = Field(..., ge=0.0, le=1.0, description="Normalized thrust")
    aux: List[float] = Field(default_factory=list, description="Auxiliary outputs")

class DiagnosticsData(BaseModel):
    policy_latency_ms: float = Field(..., ge=0.0)
    obs_clip_fractions: Dict[str, float] = Field(default_factory=dict)
    value_estimate: Optional[float] = None

class SafetyData(BaseModel):
    clamped: bool = Field(..., description="Whether any values were clamped")
    clamp_reason: Optional[str] = Field(default=None, description="Reason for clamping")

class UnityAPIResponse(BaseModel):
    action: ActionCommand
    diagnostics: DiagnosticsData
    safety: SafetyData
    success: bool = True
    error: Optional[str] = None
    timestamp: float
```

### Task List for Implementation

```yaml
Task 1: Create Core Bridge Module Structure
CREATE src/hlynr_bridge/__init__.py:
  - EMPTY module initialization file

CREATE src/hlynr_bridge/schemas.py:
  - COPY pattern from: research/pydantic_validation_research.md
  - IMPLEMENT complete Pydantic models as shown above
  - ADD Unity-specific field validators for coordinate ranges
  - INCLUDE comprehensive error messages

Task 2: Implement Coordinate Transformation System  
CREATE src/hlynr_bridge/transforms.py:
  - COPY pattern from: research/unity_coordinates_research.md  
  - IMPLEMENT CoordinateTransformer class with all methods
  - ADD version tracking and validation
  - INCLUDE round-trip testing functions

Task 3: Create VecNormalize Integration
CREATE src/hlynr_bridge/normalize.py:
  - COPY pattern from: research/stable_baselines3_research.md
  - IMPLEMENT VecNormalize loader with frozen statistics
  - ADD observation preprocessing pipeline
  - INCLUDE shape validation and error handling

Task 4: Implement PPO Policy Wrapper
CREATE src/hlynr_bridge/policy.py:
  - COPY pattern from: research/stable_baselines3_research.md
  - IMPLEMENT PolicyEngine class for deterministic inference
  - ADD model loading validation and caching
  - INCLUDE performance monitoring

Task 5: Create Safety Constraint System
CREATE src/hlynr_bridge/clamps.py:
  - IMPLEMENT safety bounds checking for rates and thrust
  - ADD comprehensive logging of constraint violations
  - INCLUDE configuration for safety limits
  - ADD emergency stop mechanisms

Task 6: Extend Episode Logging for Unity
MODIFY src/phase4_rl/episode_logger.py:
  - ADD Unity-specific coordinate frame handling
  - IMPLEMENT versioned transform metadata in logs
  - ADD safety constraint logging
  - INCLUDE Unity replay compatibility validation

Task 7: Create Main Flask Application
CREATE src/hlynr_bridge/server.py:
  - EXTEND pattern from: src/phase4_rl/bridge_server.py
  - ADD Unity-specific endpoints (/api/v1/inference)
  - IMPLEMENT complete request/response pipeline
  - ADD comprehensive error handling and CORS

Task 8: Configuration and Health Monitoring
CREATE src/hlynr_bridge/config.py:
  - COPY pattern from: src/phase4_rl/config.py
  - ADD Unity bridge-specific configuration
  - INCLUDE environment variable loading

CREATE src/hlynr_bridge/health.py:
  - IMPLEMENT health check endpoints
  - ADD performance metrics collection
  - INCLUDE system status monitoring

Task 9: Comprehensive Test Suite
CREATE tests/test_schemas.py:
  - COPY pattern from: research/test_patterns_research.md
  - TEST all Pydantic schemas with valid/invalid data
  - VALIDATE error messages and edge cases

CREATE tests/test_transforms.py:
  - TEST coordinate round-trip accuracy
  - VALIDATE quaternion normalization
  - CHECK axis mapping correctness

CREATE tests/test_normalize.py:
  - TEST VecNormalize integration
  - VALIDATE observation preprocessing
  - CHECK frozen statistics behavior

CREATE tests/test_clamps.py:
  - TEST safety constraint enforcement
  - VALIDATE logging of violations
  - CHECK emergency stop behavior

CREATE tests/test_end_to_end.py:
  - TEST complete Unity request → RL response pipeline
  - VALIDATE latency requirements
  - CHECK deterministic behavior

Task 10: Integration and Deployment
MODIFY pyproject.toml:
  - ADD hlynr_bridge module dependencies
  - INCLUDE Flask, Pydantic, numpy requirements

CREATE docker-compose.yml:
  - SETUP containerized deployment
  - INCLUDE environment configuration
  - ADD health check monitoring
```

### Per-Task Pseudocode

#### Task 1: Core Module Structure
```python
# schemas.py implementation details
class UnityAPIRequest(BaseModel):
    # PATTERN: Always validate input at API boundary
    meta: Dict[str, Any] = Field(..., description="Episode metadata")
    
    @field_validator('meta')
    @classmethod
    def validate_meta_fields(cls, v):
        # CRITICAL: Episode tracking requires these fields
        required = ['episode_id', 't', 'dt', 'sim_tick']
        for field in required:
            if field not in v:
                raise ValueError(f'Meta missing required field: {field}')
        return v
    
    @model_validator(mode='after')
    def validate_coordinate_consistency(self):
        # PATTERN: Cross-field validation for data consistency
        if self.frames.get('frame') != 'ENU':
            raise ValueError('Only ENU coordinate frame supported')
        return self
```

#### Task 2: Coordinate Transformations
```python
# transforms.py implementation details
class CoordinateTransformer:
    def __init__(self, transform_version: str = "1.0"):
        self.transform_version = transform_version
        # GOTCHA: Version tracking is critical for determinism
    
    def unity_to_enu_state(self, unity_state: UnityMissileState) -> Dict[str, Any]:
        # PATTERN: Always validate before transformation
        self._validate_unity_state(unity_state)
        
        # CRITICAL: Coordinate mapping Unity→ENU
        enu_position = [unity_state.position.x, unity_state.position.z, unity_state.position.y]
        
        # CRITICAL: Quaternion order conversion Unity[x,y,z,w] → ENU[w,x,y,z]
        enu_quaternion = [unity_state.orientation.w, unity_state.orientation.x, 
                         unity_state.orientation.z, unity_state.orientation.y]
        
        # GOTCHA: Must normalize quaternions after transformation
        enu_quaternion = self.normalize_quaternion(enu_quaternion)
        
        return {
            'position': enu_position,
            'quaternion': enu_quaternion,
            'velocity': [unity_state.velocity.x, unity_state.velocity.z, unity_state.velocity.y],
            'angular_velocity': [unity_state.angular_velocity.x, unity_state.angular_velocity.z, unity_state.angular_velocity.y],
            'fuel_remaining': unity_state.fuel_remaining,
            'transform_version': self.transform_version
        }
```

#### Task 4: PPO Policy Integration
```python
# policy.py implementation details
class PolicyEngine:
    def __init__(self, checkpoint_path: str, vecnorm_path: str):
        # PATTERN: Validate files exist before loading
        self._validate_checkpoint_files(checkpoint_path, vecnorm_path)
        
        # CRITICAL: Load VecNormalize first, then model
        dummy_env = DummyVecEnv([lambda: None])
        self.vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
        self.vec_normalize.training = False  # FREEZE statistics
        self.vec_normalize.norm_reward = False  # Don't normalize rewards
        
        self.model = PPO.load(checkpoint_path, env=self.vec_normalize)
        
    def predict(self, enu_state: Dict[str, Any]) -> np.ndarray:
        start_time = time.monotonic()
        
        # PATTERN: Transform state to observation format
        observation = self._state_to_observation(enu_state)
        
        # CRITICAL: Reshape for vectorized environment
        obs = np.array(observation).reshape(1, -1)
        
        # GOTCHA: Always use deterministic=True for consistent results
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Extract from vectorized format
        action = action[0] if len(action.shape) > 1 else action
        
        # PERFORMANCE: Track inference latency
        self.last_inference_time = (time.monotonic() - start_time) * 1000
        
        return action
```

#### Task 5: Safety Constraints
```python
# clamps.py implementation details
class SafetyConstraints:
    def __init__(self, rate_max_radps: float = 5.0):
        self.rate_max_radps = rate_max_radps
        # PATTERN: Log all constraint parameters
        logger.info(f"Safety constraints: rate_max={rate_max_radps} rad/s")
    
    def apply_clamps(self, raw_action: np.ndarray) -> Tuple[ActionCommand, SafetyData]:
        clamped = False
        clamp_reasons = []
        
        # CRITICAL: Extract and validate action components
        pitch_rate = raw_action[0]
        yaw_rate = raw_action[1] 
        roll_rate = raw_action[2]
        thrust_cmd = raw_action[3]
        
        # Apply rate limits
        if abs(pitch_rate) > self.rate_max_radps:
            pitch_rate = np.clip(pitch_rate, -self.rate_max_radps, self.rate_max_radps)
            clamped = True
            clamp_reasons.append(f"pitch_rate exceeded ±{self.rate_max_radps}")
        
        # Apply thrust limits  
        if not (0.0 <= thrust_cmd <= 1.0):
            thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)
            clamped = True
            clamp_reasons.append("thrust_cmd outside [0,1]")
        
        # PATTERN: Log all safety violations
        if clamped:
            logger.warning(f"Safety clamps applied: {', '.join(clamp_reasons)}")
        
        return ActionCommand(
            rate_cmd_radps=RateCommand(pitch=pitch_rate, yaw=yaw_rate, roll=roll_rate),
            thrust_cmd=thrust_cmd,
            aux=raw_action[4:].tolist() if len(raw_action) > 4 else []
        ), SafetyData(clamped=clamped, clamp_reason=', '.join(clamp_reasons) if clamped else None)
```

#### Task 7: Main Flask Application
```python
# server.py implementation details
app = Flask(__name__)

# CRITICAL: Configure CORS for Unity integration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:*", "http://unity-client"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

@app.route('/api/v1/inference', methods=['POST'])
def unity_inference():
    """Main Unity inference endpoint."""
    start_time = time.monotonic()
    
    try:
        # PATTERN: Validate request format first
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Content-Type must be application/json'}), 400
        
        # PATTERN: Use Pydantic for request validation
        try:
            unity_request = UnityAPIRequest.model_validate(request.get_json())
        except ValidationError as e:
            return jsonify({
                'success': False, 
                'error': 'Request validation failed',
                'details': e.errors()
            }), 400
        
        # Transform Unity state to ENU
        enu_state = transformer.unity_to_enu_state(unity_request.blue)
        
        # Get policy action
        raw_action = policy_engine.predict(enu_state)
        
        # Apply safety constraints
        action_cmd, safety_data = safety_constraints.apply_clamps(raw_action)
        
        # Build response
        response = UnityAPIResponse(
            action=action_cmd,
            diagnostics=DiagnosticsData(
                policy_latency_ms=policy_engine.last_inference_time,
                obs_clip_fractions={},  # TODO: Implement clip tracking
                value_estimate=None
            ),
            safety=safety_data,
            timestamp=time.time()
        )
        
        # PERFORMANCE: Track total latency
        total_latency = (time.monotonic() - start_time) * 1000
        
        # PATTERN: Log for monitoring
        logger.info(f"Unity inference completed in {total_latency:.2f}ms")
        
        return jsonify(response.model_dump())
        
    except Exception as e:
        logger.error(f"Unity inference error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500
```

### Integration Points

```yaml
FLASK APPLICATION:
  - extend: src/phase4_rl/bridge_server.py patterns
  - add: Unity-specific endpoints and CORS configuration
  - pattern: Use existing error handling and logging structure

PYDANTIC SCHEMAS:
  - add to: src/hlynr_bridge/schemas.py
  - pattern: Field validators with clear error messages
  - integration: Flask request validation decorator

MODEL LOADING:
  - extend: existing PPO loading patterns from bridge_server.py
  - add: VecNormalize integration with frozen statistics
  - pattern: Comprehensive validation and error handling

EPISODE LOGGING:
  - extend: src/phase4_rl/episode_logger.py
  - add: Unity coordinate frame metadata
  - pattern: Versioned JSONL format with transform tracking

CONFIGURATION:
  - extend: src/phase4_rl/config.py patterns
  - add: Unity bridge-specific settings
  - pattern: Environment variable loading with defaults
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/hlynr_bridge/ --fix
mypy src/hlynr_bridge/
ruff check tests/ --fix
mypy tests/

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests - Each Component
```python
# CREATE tests/test_transforms.py with these test cases:
def test_coordinate_round_trip():
    """Test ENU↔Unity transformations maintain accuracy."""
    transformer = CoordinateTransformer()
    
    # Test position round-trip
    enu_pos = [100.0, 200.0, 50.0]
    unity_pos = transformer.enu_to_unity_position(enu_pos)
    enu_pos_back = transformer.unity_to_enu_position(unity_pos)
    
    assert np.allclose(enu_pos, enu_pos_back, atol=1e-10)

def test_quaternion_normalization():
    """Test quaternion normalization after transformation."""
    transformer = CoordinateTransformer()
    
    # Test with unnormalized quaternion
    unnormalized = [2.0, 0.0, 0.0, 0.0]
    normalized = transformer.normalize_quaternion(unnormalized)
    
    magnitude = sum(q**2 for q in normalized) ** 0.5
    assert abs(magnitude - 1.0) < 1e-10

def test_axis_mapping():
    """Test that coordinate axis mapping is correct."""
    transformer = CoordinateTransformer()
    
    # Test unit vectors
    east_enu = [1.0, 0.0, 0.0]
    north_enu = [0.0, 1.0, 0.0]
    up_enu = [0.0, 0.0, 1.0]
    
    # Should map to Unity axes correctly
    assert transformer.enu_to_unity_position(east_enu) == [1.0, 0.0, 0.0]   # East → X
    assert transformer.enu_to_unity_position(up_enu) == [0.0, 1.0, 0.0]     # Up → Y
    assert transformer.enu_to_unity_position(north_enu) == [0.0, 0.0, 1.0]  # North → Z

# CREATE tests/test_schemas.py with these test cases:
def test_unity_request_validation():
    """Test Unity request schema validation."""
    valid_request = {
        "meta": {"episode_id": "ep_001", "t": 1.0, "dt": 0.01, "sim_tick": 100},
        "frames": {"frame": "ENU", "unity_lh": True},
        "blue": {
            "position": {"x": 100.0, "y": 50.0, "z": 200.0},
            "velocity": {"x": 10.0, "y": 2.0, "z": 15.0},
            "angular_velocity": {"x": 0.1, "y": 0.2, "z": 0.0},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "fuel_remaining": 85.5,
            "thrust_current": 800.0,
            "target_lock": True
        },
        "red": {"position": {"x": 500.0, "y": 100.0, "z": 600.0}},
        "guidance": {"los": [0.1, 0.2], "range": 223.6},
        "env": {"wind": [0.0, 0.0, 0.0]},
        "normalization": {
            "obs_version": "1.0",
            "vecnorm_stats_id": "phase4_final",
            "transform_version": "1.0"
        }
    }
    
    # Should validate successfully
    request = UnityAPIRequest.model_validate(valid_request)
    assert request.blue.fuel_remaining == 85.5

def test_schema_validation_errors():
    """Test schema validation catches errors."""
    invalid_request = {
        "meta": {"episode_id": "ep_001"},  # Missing required fields
        # ... rest of invalid data
    }
    
    with pytest.raises(ValidationError):
        UnityAPIRequest.model_validate(invalid_request)

# CREATE tests/test_clamps.py with these test cases:
def test_safety_clamps_rates():
    """Test rate command safety clamping."""
    constraints = SafetyConstraints(rate_max_radps=5.0)
    
    # Test excessive rates are clamped
    excessive_action = np.array([10.0, -8.0, 3.0, 0.8, 0.0, 0.0])
    action_cmd, safety_data = constraints.apply_clamps(excessive_action)
    
    assert action_cmd.rate_cmd_radps.pitch == 5.0    # Clamped to max
    assert action_cmd.rate_cmd_radps.yaw == -5.0     # Clamped to min
    assert action_cmd.rate_cmd_radps.roll == 3.0     # Within limits
    assert safety_data.clamped == True
    assert "pitch_rate exceeded" in safety_data.clamp_reason

def test_safety_clamps_thrust():
    """Test thrust command safety clamping."""
    constraints = SafetyConstraints()
    
    # Test excessive thrust is clamped
    excessive_action = np.array([1.0, 0.0, 0.0, 1.5, 0.0, 0.0])
    action_cmd, safety_data = constraints.apply_clamps(excessive_action)
    
    assert action_cmd.thrust_cmd == 1.0  # Clamped to max
    assert safety_data.clamped == True
    assert "thrust_cmd outside" in safety_data.clamp_reason
```

```bash
# Run and iterate until passing:
cd src/hlynr_bridge && python -m pytest ../../tests/ -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Start the bridge server
export MODEL_CHECKPOINT=checkpoints/best_model.zip
export VECNORM_STATS_ID=phase4_final
export OBS_VERSION=1.0
export TRANSFORM_VERSION=1.0
python -m hlynr_bridge.server --host localhost --port 5000

# Test the Unity inference endpoint
curl -X POST http://localhost:5000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_unity_request.json

# Expected: {"action": {...}, "diagnostics": {...}, "safety": {...}, "success": true}
# If error: Check logs and validate request format
```

### Level 4: Performance Validation
```bash
# Test latency requirements (should be <20ms p50, <35ms p95)
python tests/test_performance.py --requests 1000 --target-p50 20 --target-p95 35

# Test deterministic behavior (same input = same output)
python tests/test_determinism.py --iterations 100

# Expected: All performance targets met, deterministic behavior confirmed
```

## Final Validation Checklist
- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] No linting errors: `ruff check src/hlynr_bridge/ tests/`
- [ ] No type errors: `mypy src/hlynr_bridge/ tests/`
- [ ] API responds to Unity requests: `curl test successful`
- [ ] Latency targets met: `p50 < 20ms, p95 < 35ms`
- [ ] Deterministic responses: `same input = same output`
- [ ] Safety clamps working: `excessive inputs clamped and logged`
- [ ] Episode logging functional: `JSONL files created and valid`
- [ ] Transform round-trips accurate: `ENU↔Unity accuracy < 1e-10`
- [ ] Health endpoints working: `/health`, `/stats` respond correctly

---

## Anti-Patterns to Avoid
- ❌ Don't modify existing bridge_server.py - extend it instead
- ❌ Don't skip coordinate transformation validation - Unity/ENU differences are critical
- ❌ Don't ignore safety constraints - all outputs must be clamped
- ❌ Don't allow VecNormalize training mode during inference - freeze statistics
- ❌ Don't return non-deterministic responses - same input must give same output
- ❌ Don't skip CORS configuration - Unity client needs proper headers
- ❌ Don't use hardcoded paths - use environment variables and configuration
- ❌ Don't skip latency monitoring - sub-20ms requirement is critical

## Confidence Score: 9/10

This PRP provides comprehensive context, specific implementation patterns, detailed validation, and addresses all known gotchas. The existing bridge_server.py infrastructure significantly reduces implementation risk. Success depends on careful attention to coordinate transformations and deterministic inference patterns.

**Key Success Factors:**
1. Follow existing patterns from bridge_server.py closely
2. Implement comprehensive coordinate transformation testing
3. Ensure VecNormalize statistics remain frozen
4. Apply safety constraints consistently with logging
5. Validate all schemas thoroughly with Pydantic
6. Monitor latency and determinism continuously