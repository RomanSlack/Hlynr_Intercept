"""
Pydantic schemas for Unity↔RL Inference API v1.0.

This module defines the exact request/response contracts for the Unity bridge API
as specified in the PRP requirements.

CRITICAL FIX: transform_version removed from request (server-side only per PRP requirements)
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
import time


class FrameInfo(BaseModel):
    """Coordinate frame information."""
    frame: str = Field(..., description="Coordinate frame identifier")
    unity_lh: bool = Field(..., description="Whether Unity uses left-handed coordinates")


class MetaInfo(BaseModel):
    """Request metadata."""
    episode_id: str = Field(..., description="Unique episode identifier")
    t: float = Field(..., description="Simulation time in seconds")
    dt: float = Field(..., description="Timestep duration in seconds")
    sim_tick: int = Field(..., description="Simulation tick counter")


class BlueState(BaseModel):
    """Blue force (interceptor) state."""
    pos_m: List[float] = Field(..., description="Position [x,y,z] in meters", min_items=3, max_items=3)
    vel_mps: List[float] = Field(..., description="Velocity [vx,vy,vz] in m/s", min_items=3, max_items=3)
    quat_wxyz: List[float] = Field(..., description="Quaternion [w,x,y,z]", min_items=4, max_items=4)
    ang_vel_radps: List[float] = Field(..., description="Angular velocity [wx,wy,wz] in rad/s", min_items=3, max_items=3)
    fuel_frac: float = Field(..., description="Fuel fraction remaining [0..1]", ge=0.0, le=1.0)
    
    @validator('quat_wxyz')
    def quaternion_normalized(cls, v):
        """Validate quaternion is approximately normalized."""
        magnitude = sum(x*x for x in v) ** 0.5
        if not (0.9 <= magnitude <= 1.1):  # Allow some tolerance
            raise ValueError(f"Quaternion magnitude {magnitude} not normalized")
        return v


class RedState(BaseModel):
    """Red force (threat) state."""
    pos_m: List[float] = Field(..., description="Position [x,y,z] in meters", min_items=3, max_items=3)
    vel_mps: List[float] = Field(..., description="Velocity [vx,vy,vz] in m/s", min_items=3, max_items=3)
    quat_wxyz: List[float] = Field(..., description="Quaternion [w,x,y,z]", min_items=4, max_items=4)
    
    @validator('quat_wxyz')
    def quaternion_normalized(cls, v):
        """Validate quaternion is approximately normalized."""
        magnitude = sum(x*x for x in v) ** 0.5
        if not (0.9 <= magnitude <= 1.1):  # Allow some tolerance
            raise ValueError(f"Quaternion magnitude {magnitude} not normalized")
        return v


class GuidanceInfo(BaseModel):
    """Guidance system information."""
    los_unit: List[float] = Field(..., description="Line-of-sight unit vector [x,y,z]", min_items=3, max_items=3)
    los_rate_radps: List[float] = Field(..., description="LOS rate [wx,wy,wz] in rad/s", min_items=3, max_items=3)
    range_m: float = Field(..., description="Range to target in meters", ge=0.0)
    closing_speed_mps: float = Field(..., description="Closing speed in m/s")
    fov_ok: bool = Field(..., description="Target in field of view")
    g_limit_ok: bool = Field(..., description="Within G-force limits")
    
    @validator('los_unit')
    def los_unit_normalized(cls, v):
        """Validate LOS unit vector is normalized."""
        magnitude = sum(x*x for x in v) ** 0.5
        if not (0.9 <= magnitude <= 1.1):  # Allow some tolerance
            raise ValueError(f"LOS unit vector magnitude {magnitude} not normalized")
        return v


class EnvironmentInfo(BaseModel):
    """Environment conditions."""
    wind_mps: Optional[List[float]] = Field(None, description="Wind velocity [x,y,z] in m/s", min_items=3, max_items=3)
    noise_std: Optional[float] = Field(None, description="Observation noise standard deviation", ge=0.0)
    episode_step: int = Field(..., description="Current episode step", ge=0)
    max_steps: int = Field(..., description="Maximum episode steps", gt=0)


class NormalizationInfo(BaseModel):
    """Observation normalization information."""
    obs_version: str = Field(..., description="Observation format version")
    vecnorm_stats_id: str = Field(..., description="VecNormalize statistics identifier")
    # CRITICAL FIX: transform_version removed (server-side only per PRP requirements)


class InferenceRequest(BaseModel):
    """V1.0 Inference API request schema."""
    meta: MetaInfo = Field(..., description="Request metadata")
    frames: FrameInfo = Field(..., description="Coordinate frame information")
    blue: BlueState = Field(..., description="Blue force state")
    red: RedState = Field(..., description="Red force state")
    guidance: GuidanceInfo = Field(..., description="Guidance information")
    env: EnvironmentInfo = Field(..., description="Environment conditions")
    normalization: NormalizationInfo = Field(..., description="Normalization parameters")
    
    @root_validator
    def validate_request(cls, values):
        """Cross-field validation."""
        # Validate coordinate frame consistency
        frames = values.get('frames')
        if frames and frames.frame != "ENU":
            raise ValueError(f"Unsupported coordinate frame: {frames.frame}")
        
        # Validate episode step limits
        env = values.get('env')
        if env and env.episode_step >= env.max_steps:
            raise ValueError(f"Episode step {env.episode_step} >= max_steps {env.max_steps}")
        
        return values


class RateCommand(BaseModel):
    """Body rate command."""
    pitch: float = Field(..., description="Pitch rate command in rad/s")
    yaw: float = Field(..., description="Yaw rate command in rad/s")
    roll: float = Field(..., description="Roll rate command in rad/s")


class ActionCommand(BaseModel):
    """Action command output."""
    rate_cmd_radps: RateCommand = Field(..., description="Body rate commands")
    thrust_cmd: float = Field(..., description="Thrust command [0..1]", ge=0.0, le=1.0)
    aux: List[float] = Field(default_factory=list, description="Auxiliary outputs")


class ClipFractions(BaseModel):
    """Observation clipping statistics."""
    low: float = Field(..., description="Fraction of observations clipped below", ge=0.0, le=1.0)
    high: float = Field(..., description="Fraction of observations clipped above", ge=0.0, le=1.0)


class DiagnosticsInfo(BaseModel):
    """Policy diagnostics information."""
    policy_latency_ms: float = Field(..., description="Policy inference latency in ms", ge=0.0)
    obs_clip_fractions: ClipFractions = Field(..., description="Observation clipping statistics")
    value_estimate: Optional[float] = Field(None, description="Policy value estimate")


class SafetyInfo(BaseModel):
    """Safety clamping information."""
    clamped: bool = Field(..., description="Whether outputs were clamped")
    clamp_reason: Optional[str] = Field(None, description="Reason for clamping if applicable")


class InferenceResponse(BaseModel):
    """V1.0 Inference API response schema."""
    action: ActionCommand = Field(..., description="Control action commands")
    diagnostics: DiagnosticsInfo = Field(..., description="Policy diagnostics")
    safety: SafetyInfo = Field(..., description="Safety information")
    
    # Response metadata
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    success: bool = Field(default=True, description="Request success status")
    error: Optional[str] = Field(None, description="Error message if unsuccessful")


class HealthResponse(BaseModel):
    """Health check response schema."""
    # CRITICAL: All 7 required keys per PRP requirements
    ok: bool = Field(..., description="Overall health status")
    policy_loaded: bool = Field(..., description="Whether policy model is loaded")
    policy_id: Optional[str] = Field(None, description="Active policy identifier")
    vecnorm_stats_id: Optional[str] = Field(None, description="Loaded VecNormalize stats ID")
    obs_version: Optional[str] = Field(None, description="Active observation version")
    transform_version: Optional[str] = Field(None, description="Active transform version")
    seed: Optional[int] = Field(None, description="Active random seed for determinism")
    
    # Legacy fields for backward compatibility
    status: str = Field(default="healthy", description="Health status string")
    model_loaded: bool = Field(default=False, description="Whether model is loaded")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    loaded_models: Dict[str, str] = Field(default_factory=dict, description="Loaded model information")


class MetricsResponse(BaseModel):
    """Metrics endpoint response schema."""
    # Request statistics
    requests_served: int = Field(..., description="Total requests served")
    requests_failed: int = Field(default=0, description="Total failed requests")
    
    # Latency statistics (in milliseconds)
    latency_p50_ms: float = Field(..., description="50th percentile latency")
    latency_p95_ms: float = Field(..., description="95th percentile latency")
    latency_p99_ms: Optional[float] = Field(None, description="99th percentile latency")
    latency_mean_ms: float = Field(..., description="Mean latency")
    
    # Throughput statistics
    requests_per_second: float = Field(..., description="Current requests per second")
    
    # Safety statistics
    safety_clamps_total: int = Field(default=0, description="Total safety clamps applied")
    safety_clamp_rate: float = Field(default=0.0, description="Safety clamp rate [0..1]")
    
    # Model information
    model_loaded_at: Optional[float] = Field(None, description="Model load timestamp")
    vecnorm_stats_id: Optional[str] = Field(None, description="Active VecNormalize stats ID")
    
    timestamp: float = Field(default_factory=time.time, description="Metrics timestamp")


# Validation constants
SUPPORTED_OBS_VERSIONS = ["obs_v1.0", "obs_v1.1"]
SUPPORTED_TRANSFORM_VERSIONS = ["tfm_v1.0"]
MAX_RATE_RADPS = 10.0  # Maximum angular rate limit
MIN_RATE_RADPS = -10.0  # Minimum angular rate limit

# CRITICAL: Canonical clamp reason strings per PRP requirements
CANONICAL_CLAMP_REASONS = [
    "rate_limit_pitch",
    "rate_limit_yaw", 
    "rate_limit_roll",
    "thrust_floor",
    "thrust_ceiling"
]


def validate_obs_version(obs_version: str) -> bool:
    """Validate observation version is supported."""
    return obs_version in SUPPORTED_OBS_VERSIONS


def validate_transform_version(transform_version: str) -> bool:
    """Validate transform version is supported."""
    return transform_version in SUPPORTED_TRANSFORM_VERSIONS


def validate_rate_commands(rate_cmd: RateCommand) -> RateCommand:
    """Validate and clamp rate commands to safety limits."""
    clamped = RateCommand(
        pitch=max(MIN_RATE_RADPS, min(MAX_RATE_RADPS, rate_cmd.pitch)),
        yaw=max(MIN_RATE_RADPS, min(MAX_RATE_RADPS, rate_cmd.yaw)),
        roll=max(MIN_RATE_RADPS, min(MAX_RATE_RADPS, rate_cmd.roll))
    )
    return clamped


def check_if_clamped(original: RateCommand, clamped: RateCommand) -> tuple[bool, str]:
    """Check if rate commands were clamped and return canonical reason."""
    reasons = []
    
    if original.pitch != clamped.pitch:
        reasons.append("rate_limit_pitch")
    if original.yaw != clamped.yaw:
        reasons.append("rate_limit_yaw")
    if original.roll != clamped.roll:
        reasons.append("rate_limit_roll")
    
    if reasons:
        return True, ", ".join(reasons)
    return False, ""


# Example usage and validation
if __name__ == "__main__":
    # Example request (FIXED: no transform_version in request)
    example_request = {
        "meta": {
            "episode_id": "ep_000001",
            "t": 1.23,
            "dt": 0.01,
            "sim_tick": 123
        },
        "frames": {
            "frame": "ENU",
            "unity_lh": True
        },
        "blue": {
            "pos_m": [100.0, 200.0, 50.0],
            "vel_mps": [150.0, 10.0, -5.0],
            "quat_wxyz": [0.995, 0.0, 0.1, 0.0],
            "ang_vel_radps": [0.1, 0.2, 0.05],
            "fuel_frac": 0.75
        },
        "red": {
            "pos_m": [500.0, 600.0, 100.0],
            "vel_mps": [-50.0, -40.0, -10.0],
            "quat_wxyz": [0.924, 0.0, 0.0, 0.383]
        },
        "guidance": {
            "los_unit": [0.8, 0.6, 0.0],
            "los_rate_radps": [0.01, -0.02, 0.0],
            "range_m": 500.0,
            "closing_speed_mps": 200.0,
            "fov_ok": True,
            "g_limit_ok": True
        },
        "env": {
            "wind_mps": [2.0, 1.0, 0.0],
            "noise_std": 0.01,
            "episode_step": 123,
            "max_steps": 1000
        },
        "normalization": {
            "obs_version": "obs_v1.0",
            "vecnorm_stats_id": "vecnorm_baseline_001"
            # FIXED: transform_version removed from request
        }
    }
    
    # Validate request
    try:
        request = InferenceRequest(**example_request)
        print("✅ Request validation passed")
        print(f"Request: {request.json(indent=2)}")
    except Exception as e:
        print(f"❌ Request validation failed: {e}")
    
    # Example health check with all 7 required keys
    example_health = {
        "ok": True,
        "policy_loaded": True,
        "policy_id": "policy_123",
        "vecnorm_stats_id": "vecnorm_001",
        "obs_version": "obs_v1.0",
        "transform_version": "tfm_v1.0",
        "seed": 42
    }
    
    try:
        health = HealthResponse(**example_health)
        print("✅ Health response validation passed")
        print(f"Health: {health.json(indent=2)}")
    except Exception as e:
        print(f"❌ Health response validation failed: {e}")
        
    # Example response
    example_response = {
        "action": {
            "rate_cmd_radps": {
                "pitch": 0.5,
                "yaw": -0.2,
                "roll": 0.1
            },
            "thrust_cmd": 0.8,
            "aux": [0.0, 0.0]
        },
        "diagnostics": {
            "policy_latency_ms": 15.3,
            "obs_clip_fractions": {
                "low": 0.02,
                "high": 0.01
            },
            "value_estimate": 0.75
        },
        "safety": {
            "clamped": False,
            "clamp_reason": None
        }
    }
    
    # Validate response
    try:
        response = InferenceResponse(**example_response)
        print("✅ Response validation passed")
        print(f"Response: {response.json(indent=2)}")
    except Exception as e:
        print(f"❌ Response validation failed: {e}")