"""
Unity↔RL Inference API Bridge Server v1.0

Enhanced bridge server implementing the full PRP specification with:
- v1.0 API schemas with Pydantic validation
- Deterministic ENU↔Unity coordinate transforms  
- VecNormalize with versioned statistics tracking
- Post-policy safety clamps with logging
- Episode JSONL logging with manifest generation
- Comprehensive latency monitoring and metrics
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional
import numpy as np
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import collections

# CRITICAL: Prometheus metrics support
try:
    from prometheus_client import Counter, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available - metrics will use JSON format")

# Import our custom modules
try:
    from .schemas import (
        InferenceRequest, InferenceResponse, HealthResponse, MetricsResponse,
        ActionCommand, RateCommand, DiagnosticsInfo, SafetyInfo, ClipFractions
    )
    from deprecated.hlynr_bridge.transforms import get_transform, validate_transform_version
    from .normalize import get_vecnorm_manager, load_vecnormalize_by_id
    from .seed_manager import set_deterministic_seeds, get_current_seed, validate_deterministic_setup
    from .clamps import get_safety_clamp_system, SafetyLimits
    from .episode_logger import get_inference_logger, reset_inference_logger
    from .config import get_config, reset_config
    from phase4_rl.scenarios import get_scenario_loader, reset_scenario_loader
    from phase4_rl.radar_env import RadarEnv
    from phase4_rl.fast_sim_env import FastSimEnv
    from phase4_rl.env_config import get_bridge_config, BridgeServerConfig
except ImportError:
    # Fallback for direct execution - temporary import shim from phase4_rl
    import sys
    from pathlib import Path
    phase4_path = Path(__file__).parent.parent / "phase4_rl"
    sys.path.insert(0, str(phase4_path))
    
    from schemas import (
        InferenceRequest, InferenceResponse, HealthResponse, MetricsResponse,
        ActionCommand, RateCommand, DiagnosticsInfo, SafetyInfo, ClipFractions
    )
    from transforms import get_transform, validate_transform_version
    from normalize import get_vecnorm_manager, load_vecnormalize_by_id
    from seed_manager import set_deterministic_seeds, get_current_seed, validate_deterministic_setup
    from .clamps import get_safety_clamp_system, SafetyLimits
    from .episode_logger import get_inference_logger, reset_inference_logger
    from .config import get_config, reset_config
    from phase4_rl.scenarios import get_scenario_loader, reset_scenario_loader
    from phase4_rl.radar_env import RadarEnv
    from phase4_rl.fast_sim_env import FastSimEnv
    from phase4_rl.env_config import get_bridge_config, BridgeServerConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables  
app = FastAPI(title="Hlynr Unity Bridge Server v1.0", version="1.0")

# Configure CORS at module level to avoid middleware conflict
from phase4_rl.env_config import BridgeServerConfig
try:
    env_config = BridgeServerConfig()
    if env_config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=env_config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info(f"CORS enabled for origins: {env_config.cors_origins}")
except Exception as e:
    logger.warning(f"CORS configuration failed: {e}")
model = None
env = None
vec_normalize = None
vec_normalize_info = None
transform = None
safety_clamp_system = None
inference_logger = None

# Server state
server_stats = {
    'requests_served': 0,
    'requests_failed': 0,
    'total_inference_time': 0.0,
    'average_inference_time': 0.0,
    'model_loaded_at': None,
    'checkpoint_path': None,
    'vecnorm_stats_id': None,
    'transform_version': None,
    'policy_id': None,
    'seed': None,  # CRITICAL: Add seed tracking
    'obs_version': None
}

# Latency tracking (for p50/p95 calculation)
latency_history = collections.deque(maxlen=10000)  # Keep last 10k latencies

# CRITICAL: Prometheus metrics (with fallback to JSON)
if PROMETHEUS_AVAILABLE:
    # Define Prometheus metrics
    hlynr_requests_total = Counter('hlynr_requests_total', 'Total number of inference requests', ['status'])
    hlynr_inference_latency_hist = Histogram('hlynr_inference_latency_ms', 'Inference latency histogram in milliseconds')
    hlynr_inference_latency_summary = Summary('hlynr_inference_latency_ms_quantiles', 'Inference latency quantiles in milliseconds')
    hlynr_safety_clamps_total = Counter('hlynr_safety_clamps_total', 'Total number of safety clamps applied', ['type'])
    hlynr_policy_latency_hist = Histogram('hlynr_policy_latency_ms', 'Policy inference latency histogram in milliseconds')

# Global server instance for startup
_server_instance = None

# Load model on first request
_model_loaded = False

def ensure_model_loaded():
    """Ensure model is loaded before processing requests."""
    global _model_loaded, _server_instance
    
    if not _model_loaded:
        logger.info("Loading model for first request...")
        try:
            _server_instance = HlynrBridgeServer()
            _server_instance.load_model()
            _model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e


class HlynrBridgeServer:
    """Unity↔RL Inference API Bridge Server v1.0."""
    
    def __init__(self, 
                 env_config: Optional[BridgeServerConfig] = None,
                 checkpoint_path: Optional[str] = None,
                 scenario_name: Optional[str] = None,
                 config_path: Optional[str] = None,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 obs_version: Optional[str] = None,
                 vecnorm_stats_id: Optional[str] = None,
                 transform_version: Optional[str] = None,
                 policy_id: Optional[str] = None,
                 rate_max_radps: Optional[float] = None):
        """
        Initialize Hlynr bridge server.
        
        Args:
            env_config: Environment configuration (if None, loads from environment variables)
            checkpoint_path: Path to trained model checkpoint (overrides env_config)
            scenario_name: Scenario name for environment configuration (overrides env_config)
            config_path: Path to configuration file (overrides env_config)
            host: Server host address (overrides env_config)
            port: Server port (overrides env_config)
            obs_version: Observation version (overrides env_config)
            vecnorm_stats_id: VecNormalize statistics ID (overrides env_config)
            transform_version: Transform version (overrides env_config)
            policy_id: Policy identifier (overrides env_config)
            rate_max_radps: Maximum angular rate for safety clamps (overrides env_config)
        """
        # Load environment configuration if not provided
        if env_config is None:
            env_config = get_bridge_config()
        self.env_config = env_config
        
        # Override with provided arguments
        self.checkpoint_path = Path(checkpoint_path or env_config.model_checkpoint)
        self.scenario_name = scenario_name or env_config.scenario_name
        self.config_path = config_path or env_config.config_path
        self.host = host or env_config.host
        self.port = port or env_config.port
        self.obs_version = obs_version or env_config.obs_version
        self.vecnorm_stats_id = vecnorm_stats_id or env_config.vecnorm_stats_id
        self.transform_version = transform_version or env_config.transform_version
        self.policy_id = policy_id or env_config.policy_id or f"policy_{int(time.time())}"
        self.rate_max_radps = rate_max_radps or env_config.rate_max_radps
        
        # CORS is now configured at module level
        
        # Reset global instances
        reset_config()
        reset_scenario_loader()
        reset_inference_logger()
        
        # Load configuration
        self.config_loader = get_config()
        self.config = self.config_loader
        
        # Load scenario
        self.scenario_loader = get_scenario_loader()
        self.scenario_config = self.scenario_loader.create_environment_config(
            self.scenario_name, self.config
        )
        
        # Setup safety limits
        self.safety_limits = SafetyLimits(rate_max_radps=self.rate_max_radps)
        
        logger.info(f"Hlynr bridge server initialized (policy_id: {self.policy_id})")
        
        # CRITICAL: Set deterministic seeds immediately during initialization
        seed = set_deterministic_seeds()
        server_stats['seed'] = seed
        server_stats['obs_version'] = self.obs_version
        
        # Validate deterministic setup
        if not validate_deterministic_setup():
            logger.warning("Deterministic setup validation failed - inference may not be reproducible")
        
        if env_config.debug_mode:
            logger.info("Server configuration:")
            logger.info(f"  Checkpoint: {self.checkpoint_path}")
            logger.info(f"  Host:Port: {self.host}:{self.port}")
            logger.info(f"  Scenario: {self.scenario_name}")
            logger.info(f"  Safety rate limit: {self.rate_max_radps} rad/s")
            logger.info(f"  Deterministic seed: {seed}")
    
    def load_model(self):
        """Load the trained model and setup all components."""
        global model, env, vec_normalize, vec_normalize_info, transform
        global safety_clamp_system, inference_logger, server_stats
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        logger.info(f"Loading model from: {self.checkpoint_path}")
        
        # Create environment
        def make_env():
            return FastSimEnv(
                config=self.scenario_config,
                scenario_name=self.scenario_name
            )
        
        env = DummyVecEnv([make_env])
        
        # Load VecNormalize by ID
        vecnorm_manager = get_vecnorm_manager()
        
        if self.vecnorm_stats_id:
            try:
                vec_normalize, vec_normalize_info = load_vecnormalize_by_id(
                    self.vecnorm_stats_id, env, self.obs_version
                )
                env = vec_normalize
                logger.info(f"Loaded VecNormalize: {self.vecnorm_stats_id}")
            except Exception as e:
                logger.warning(f"Failed to load VecNormalize by ID: {e}")
                logger.info("Attempting to load from legacy file...")
                self._load_legacy_vecnormalize(env, vecnorm_manager)
        else:
            logger.info("No vecnorm_stats_id provided, attempting legacy file...")
            self._load_legacy_vecnormalize(env, vecnorm_manager)
        
        # Load model
        model = PPO.load(self.checkpoint_path, env=env)
        
        # Setup coordinate transform
        if not validate_transform_version(self.transform_version):
            raise ValueError(f"Unsupported transform version: {self.transform_version}")
        transform = get_transform(self.transform_version)
        
        # Setup safety clamp system
        safety_clamp_system = get_safety_clamp_system(self.safety_limits)
        
        # Setup inference logger
        inference_logger = get_inference_logger(
            output_dir="inference_episodes",
            policy_id=self.policy_id
        )
        
        # Update server stats
        server_stats.update({
            'model_loaded_at': time.time(),
            'checkpoint_path': str(self.checkpoint_path),
            'vecnorm_stats_id': vec_normalize_info.stats_id if vec_normalize_info else None,
            'transform_version': self.transform_version,
            'policy_id': self.policy_id,
            'obs_version': self.obs_version
        })
        
        logger.info("Model loaded successfully!")
        logger.info(f"Environment: {self.scenario_name} scenario")
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        logger.info(f"Transform version: {self.transform_version}")
        if vec_normalize_info:
            logger.info(f"VecNormalize: {vec_normalize_info.stats_id} (v{vec_normalize_info.obs_version})")
    
    def _load_legacy_vecnormalize(self, env, vecnorm_manager):
        """Load VecNormalize from legacy file and register it."""
        global vec_normalize, vec_normalize_info
        
        # Look for legacy vec_normalize files
        vecnorm_files = [
            self.checkpoint_path.parent / "vec_normalize_final.pkl",
            self.checkpoint_path.parent / "vec_normalize.pkl"
        ]
        
        for vecnorm_path in vecnorm_files:
            if vecnorm_path.exists():
                logger.info(f"Loading legacy VecNormalize from: {vecnorm_path}")
                
                # Register the legacy file
                stats_id = vecnorm_manager.create_from_legacy_file(
                    legacy_path=str(vecnorm_path),
                    env=env,
                    obs_version=self.obs_version,
                    description=f"Auto-imported from {vecnorm_path.name}"
                )
                
                # Load the registered VecNormalize
                vec_normalize, vec_normalize_info = load_vecnormalize_by_id(
                    stats_id, env, self.obs_version
                )
                
                # Update server to use this stats_id
                self.vecnorm_stats_id = stats_id
                
                logger.info(f"Registered legacy VecNormalize as: {stats_id}")
                return
        
        logger.warning("No VecNormalize file found - running without normalization")
    
    def run(self):
        """Start the bridge server using uvicorn ASGI server."""
        # Load model before starting server
        self.load_model()
        
        logger.info(f"Starting Hlynr Bridge Server v1.0 on {self.host}:{self.port}")
        logger.info(f"API Endpoints:")
        logger.info(f"  POST /v1/inference - Unity RL Inference API v1.0")
        logger.info(f"  POST /act - Legacy inference endpoint")
        logger.info(f"  GET /healthz - Health check")
        logger.info(f"  GET /metrics - Performance metrics")
        
        # Note: FastAPI apps should be run with uvicorn
        # This method now loads the model but uvicorn handles the server
        logger.info("Model loaded. Use uvicorn to start the server:")
        logger.info(f"  uvicorn hlynr_bridge.server:app --host {self.host} --port {self.port}")
        
        # For compatibility, we'll raise an error suggesting uvicorn
        raise RuntimeError(
            f"FastAPI server must be run with uvicorn. "
            f"Use: uvicorn hlynr_bridge.server:app --host {self.host} --port {self.port}"
        )


@app.post("/v1/inference")
def v1_inference(inference_request: InferenceRequest):
    """
    Unity RL Inference API v1.0 endpoint.
    
    Implements the full PRP specification with:
    - Pydantic request/response validation
    - Coordinate transforms
    - Safety clamps
    - Comprehensive logging
    - Latency monitoring
    """
    global model, env, transform, safety_clamp_system, inference_logger, server_stats
    
    start_time = time.perf_counter()
    
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        
        # Check model is loaded
        if model is None:
            return _error_response("Model not loaded", 500)
        
        # Validate versioning compatibility
        validation_error = _validate_request_versions(inference_request)
        if validation_error:
            return _error_response(validation_error, 400)
        
        # Transform Unity state to RL observation
        try:
            observation = _unity_to_rl_observation(inference_request)
        except Exception as e:
            return _error_response(f"Observation conversion failed: {e}", 500)
        
        # Policy inference with deterministic enforcement
        policy_start = time.perf_counter()
        try:
            # CRITICAL: Enforce single-thread mode for each inference
            import torch
            torch.set_num_threads(1)
            
            action, _states = model.predict(observation, deterministic=True)
            action = action[0] if len(action.shape) > 1 else action
        except Exception as e:
            return _error_response(f"Policy inference failed: {e}", 500)
        
        policy_latency = (time.perf_counter() - policy_start) * 1000  # ms
        
        # CRITICAL: Record policy latency to Prometheus
        if PROMETHEUS_AVAILABLE:
            hlynr_policy_latency_hist.observe(policy_latency)
        
        # Convert RL action to Unity command
        try:
            action_command = _rl_to_unity_action(action)
        except Exception as e:
            return _error_response(f"Action conversion failed: {e}", 500)
        
        # Apply safety clamps
        clamped_action, safety_info = safety_clamp_system.apply_safety_clamps(action_command)
        
        # CRITICAL: Record safety clamp metrics to Prometheus
        if PROMETHEUS_AVAILABLE and safety_info.clamped:
            # Map clamp reasons to canonical PRP strings
            clamp_reasons = _parse_canonical_clamp_reasons(action_command, clamped_action)
            for reason in clamp_reasons:
                hlynr_safety_clamps_total.labels(reason=reason).inc()
        
        # Compute diagnostics
        diagnostics = _compute_diagnostics(observation, policy_latency)
        
        # Create response
        response = InferenceResponse(
            action=clamped_action,
            diagnostics=diagnostics,
            safety=safety_info
        )
        
        # Calculate total latency
        total_latency = (time.perf_counter() - start_time) * 1000  # ms
        
        # Update server statistics
        _update_server_stats(total_latency, success=True)
        
        # Log inference step
        _log_inference_step(inference_request, response, total_latency)
        
        logger.debug(f"v1/inference completed in {total_latency:.2f}ms")
        
        return response  # FastAPI auto-serializes Pydantic models
        
    except Exception as e:
        total_latency = (time.perf_counter() - start_time) * 1000
        _update_server_stats(total_latency, success=False)
        logger.error(f"v1/inference error: {e}")
        return _error_response(f"Internal server error: {e}", 500)


@app.get("/healthz")
def health_check():
    """CRITICAL: Enhanced health check endpoint with all 7 required keys."""
    global model, vec_normalize_info, transform, server_stats
    
    # Ensure model is loaded
    try:
        ensure_model_loaded()
    except Exception:
        pass  # Allow health check to return current state even if model fails to load
    
    # All 7 required keys per PRP requirements
    response = HealthResponse(
        ok=model is not None,
        policy_loaded=model is not None,
        policy_id=server_stats.get('policy_id'),
        vecnorm_stats_id=vec_normalize_info.stats_id if vec_normalize_info else None,
        obs_version=server_stats.get('obs_version'),
        transform_version=server_stats.get('transform_version'),
        seed=server_stats.get('seed'),
        
        # Legacy fields for backward compatibility
        status='healthy' if model is not None else 'not_ready',
        model_loaded=model is not None
    )
    
    return response  # FastAPI auto-serializes Pydantic models


@app.get("/metrics")
def get_metrics():
    """CRITICAL: Prometheus metrics endpoint with p50/p95 percentiles."""
    global server_stats, safety_clamp_system, latency_history
    
    if PROMETHEUS_AVAILABLE:
        # Return Prometheus exposition format
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    else:
        # Fallback to JSON format for backward compatibility
        return _get_json_metrics()


def _get_json_metrics():
    """Fallback JSON metrics when Prometheus is not available."""
    global server_stats, safety_clamp_system, latency_history
    
    # Calculate latency percentiles
    latencies = list(latency_history) if latency_history else [0.0]
    latencies_array = np.array(latencies)
    
    # Calculate requests per second (last minute)
    current_time = time.time()
    recent_latencies = [lat for lat in latency_history 
                       if hasattr(lat, 'timestamp') and current_time - lat.timestamp < 60]
    rps = len(recent_latencies) / 60.0 if recent_latencies else 0.0
    
    # Get safety clamp statistics
    clamp_stats = safety_clamp_system.get_clamp_statistics() if safety_clamp_system else {}
    
    response = MetricsResponse(
        requests_served=server_stats['requests_served'],
        requests_failed=server_stats['requests_failed'],
        latency_p50_ms=float(np.percentile(latencies_array, 50)),
        latency_p95_ms=float(np.percentile(latencies_array, 95)),
        latency_p99_ms=float(np.percentile(latencies_array, 99)),
        latency_mean_ms=float(np.mean(latencies_array)),
        requests_per_second=rps,
        safety_clamps_total=clamp_stats.get('total_clamped', 0),
        safety_clamp_rate=clamp_stats.get('overall_clamp_rate', 0.0),
        model_loaded_at=server_stats.get('model_loaded_at'),
        vecnorm_stats_id=server_stats.get('vecnorm_stats_id')
    )
    
    return response  # FastAPI auto-serializes Pydantic models


# Legacy endpoint compatibility
@app.post("/act")
def legacy_act(request: Request):
    """Legacy inference endpoint for backward compatibility."""
    # CRITICAL: Gate legacy endpoint behind environment variable
    enable_legacy = os.getenv('ENABLE_LEGACY_ACT', 'false').lower() in ('true', '1', 'yes')
    
    if not enable_legacy:
        # Return 404 with deprecation header when disabled
        raise HTTPException(
            status_code=404,
            detail='Legacy /act endpoint is disabled. Use /v1/inference instead.',
            headers={'X-Deprecated': 'Use /v1/inference'}
        )
    
    try:
        request_body = request.json()
        observation = np.array(request_body['observation'], dtype=np.float32)
        deterministic = request_body.get('deterministic', True)
        
        # Simple inference without full v1.0 pipeline
        action, _states = model.predict(observation.reshape(1, -1), deterministic=deterministic)
        action = action[0] if len(action.shape) > 1 else action
        
        return {
            'success': True,
            'action': action.tolist(),
            'inference_time': 0.0,  # Not tracked for legacy endpoint
            'error': None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'error': str(e),
                'action': None,
                'inference_time': 0.0
            }
        )


def _parse_canonical_clamp_reasons(original: ActionCommand, clamped: ActionCommand) -> List[str]:
    """Parse canonical clamp reasons from action comparison."""
    reasons = []
    
    # Check angular rate clamps
    if original.rate_cmd_radps.pitch != clamped.rate_cmd_radps.pitch:
        reasons.append("rate_limit_pitch")
    if original.rate_cmd_radps.yaw != clamped.rate_cmd_radps.yaw:
        reasons.append("rate_limit_yaw")
    if original.rate_cmd_radps.roll != clamped.rate_cmd_radps.roll:
        reasons.append("rate_limit_roll")
    
    # Check thrust clamps
    if original.thrust_cmd != clamped.thrust_cmd:
        if original.thrust_cmd < clamped.thrust_cmd:
            reasons.append("thrust_floor")
        else:
            reasons.append("thrust_ceiling")
    
    return reasons


def _error_response(message: str, status_code: int):
    """Create standardized error response."""
    raise HTTPException(
        status_code=status_code,
        detail={
            'success': False,
            'error': message,
            'timestamp': time.time()
        }
    )


def _validate_request_versions(request: InferenceRequest) -> Optional[str]:
    """Validate request versions are compatible."""
    global vec_normalize_info, transform
    
    # Note: transform_version will be removed from request schema - server-side only
    # For now, handle both cases for backward compatibility
    transform_version = getattr(request.normalization, 'transform_version', None)
    if transform_version and not validate_transform_version(transform_version):
        return f"Unsupported transform version: {transform_version}"
    
    # Validate VecNormalize compatibility
    if vec_normalize_info:
        if request.normalization.vecnorm_stats_id != vec_normalize_info.stats_id:
            return f"VecNormalize ID mismatch: requested {request.normalization.vecnorm_stats_id}, loaded {vec_normalize_info.stats_id}"
        
        if request.normalization.obs_version != vec_normalize_info.obs_version:
            return f"Observation version mismatch: requested {request.normalization.obs_version}, loaded {vec_normalize_info.obs_version}"
    
    return None


def _unity_to_rl_observation(request: InferenceRequest) -> np.ndarray:
    """Convert Unity request to RL radar-only observation format."""
    global transform
    
    # Transform Unity coordinates to ENU if needed
    if request.frames.unity_lh:
        # Transform blue state
        blue_enu = transform.transform_state_unity_to_enu({
            'pos_m': request.blue.pos_m,
            'vel_mps': request.blue.vel_mps,
            'quat_wxyz': request.blue.quat_wxyz,
            'ang_vel_radps': request.blue.ang_vel_radps
        })
        
        # Transform red state
        red_enu = transform.transform_state_unity_to_enu({
            'pos_m': request.red.pos_m,
            'vel_mps': request.red.vel_mps,
            'quat_wxyz': request.red.quat_wxyz
        })
    else:
        # Already in ENU
        blue_enu = {
            'pos_m': request.blue.pos_m,
            'vel_mps': request.blue.vel_mps,
            'quat_wxyz': request.blue.quat_wxyz,
            'ang_vel_radps': request.blue.ang_vel_radps
        }
        red_enu = {
            'pos_m': request.red.pos_m,
            'vel_mps': request.red.vel_mps,
            'quat_wxyz': request.red.quat_wxyz
        }
    
    # Create radar-only observation matching training format (17 dims for easy scenario)
    # For 1 interceptor, 1 missile configuration
    
    # 1. Interceptor essential (4 dims): fuel, status, target_id, distance
    interceptor_essential = [
        request.blue.fuel_frac,           # fuel
        1.0,                              # status (active=1.0, destroyed=0.0)
        0.0,                              # target_id (targeting missile 0)
        request.guidance.range_m / 1000.0 # distance to target (normalized)
    ]
    
    # 2. Ground radar returns (4 dims): detected_x, detected_y, confidence, range
    # Simulate noisy ground radar detection of the missile
    missile_pos = red_enu['pos_m']
    ground_radar = [
        missile_pos[0] / 1000.0,          # detected_x (normalized)
        missile_pos[1] / 1000.0,          # detected_y (normalized) 
        0.9,                              # confidence (high for ground radar)
        request.guidance.range_m / 1000.0 # range (normalized)
    ]
    
    # 3. Onboard radar returns (3 dims): detections, bearing, range
    # Convert LOS to bearing relative to interceptor
    los_unit = request.guidance.los_unit
    bearing = np.arctan2(los_unit[1], los_unit[0])  # bearing in radians
    onboard_radar = [
        1.0 if request.guidance.fov_ok else 0.0,     # detections (1 if detected)
        bearing / np.pi,                             # bearing (normalized to [-1,1])
        request.guidance.range_m / 1000.0            # range (normalized)
    ]
    
    # 4. Environmental data (6 dims): wind, atmosphere, etc.
    wind_mps = request.env.wind_mps or [0.0, 0.0, 0.0]
    env_data = [
        wind_mps[0] / 10.0,                          # wind_x (normalized)
        wind_mps[1] / 10.0,                          # wind_y (normalized)
        wind_mps[2] / 10.0,                          # wind_z (normalized)
        request.env.noise_std,                       # noise level
        request.env.episode_step / max(request.env.max_steps, 1),  # normalized step
        0.0                                          # reserved/atmosphere
    ]
    
    # Assemble radar-only observation (17 dims total)
    radar_obs = interceptor_essential + ground_radar + onboard_radar + env_data
    
    return np.array(radar_obs, dtype=np.float32)


def _rl_to_unity_action(action: np.ndarray) -> ActionCommand:
    """Convert RL action to Unity action command."""
    global transform
    
    # Extract action components (adjust based on your action space)
    if len(action) >= 4:
        rate_pitch = float(action[0])
        rate_yaw = float(action[1])
        rate_roll = float(action[2])
        thrust = float(action[3])
        aux = action[4:].tolist() if len(action) > 4 else []
    else:
        raise ValueError(f"Invalid action dimension: {len(action)}")
    
    # Transform angular rates from ENU to Unity if needed
    enu_rates = [rate_pitch, rate_yaw, rate_roll]
    unity_rates = transform.enu_to_unity_angular_velocity(enu_rates)
    
    # Apply basic clamps to ensure Pydantic validation passes
    # (Full safety clamping happens later in the pipeline)
    thrust_clamped = max(0.0, min(1.0, thrust))
    aux_clamped = [max(-1.0, min(1.0, a)) for a in aux]
    
    return ActionCommand(
        rate_cmd_radps=RateCommand(
            pitch=unity_rates[0],
            yaw=unity_rates[1],
            roll=unity_rates[2]
        ),
        thrust_cmd=thrust_clamped,
        aux=aux_clamped
    )


def _compute_diagnostics(observation: np.ndarray, policy_latency_ms: float) -> DiagnosticsInfo:
    """Compute inference diagnostics."""
    global vec_normalize, vecnorm_manager
    
    # Compute observation clipping fractions
    clip_fractions = {'low': 0.0, 'high': 0.0}
    if vec_normalize:
        vecnorm_manager = get_vecnorm_manager()
        clip_fractions = vecnorm_manager.compute_clip_fractions(observation)
    
    return DiagnosticsInfo(
        policy_latency_ms=policy_latency_ms,
        obs_clip_fractions=ClipFractions(
            low=clip_fractions['low'],
            high=clip_fractions['high']
        ),
        value_estimate=None  # Could be computed if needed
    )


def _update_server_stats(latency_ms: float, success: bool):
    """Update server statistics and Prometheus metrics."""
    global server_stats, latency_history
    
    server_stats['requests_served'] += 1
    if not success:
        server_stats['requests_failed'] += 1
    
    # CRITICAL: Record Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        status = 'success' if success else 'error'
        hlynr_requests_total.labels(status=status).inc()
        
        if success:
            # Record latency to both histogram and summary for p50/p95
            hlynr_inference_latency_hist.observe(latency_ms)
            hlynr_inference_latency_summary.observe(latency_ms)
    
    if success:
        server_stats['total_inference_time'] += latency_ms
        server_stats['average_inference_time'] = (
            server_stats['total_inference_time'] / server_stats['requests_served']
        )
        
        # Add to latency history for percentile calculation
        latency_history.append(latency_ms)


def _log_inference_step(request: InferenceRequest, response: InferenceResponse, latency_ms: float):
    """Log inference step to episode logger."""
    global inference_logger, server_stats
    
    if inference_logger and inference_logger.enable_logging:
        try:
            # Check if we need to end the previous episode (episode_id changed)
            current_episode_id = getattr(inference_logger, 'current_episode_id', None)
            if current_episode_id and current_episode_id != request.meta.episode_id:
                # End the previous episode
                inference_logger.end_inference_episode(
                    outcome="completed",  # Default outcome for inference episodes
                    final_time=getattr(inference_logger, 'last_t', 0.0),
                    total_steps=getattr(inference_logger, 'step_count', 0),
                    notes="Episode ended due to new episode_id"
                )
            
            # Check if we need to start a new episode
            if not hasattr(inference_logger, 'current_episode_file') or \
               inference_logger.current_episode_file is None or \
               current_episode_id != request.meta.episode_id:
                inference_logger.begin_inference_episode(
                    episode_id=request.meta.episode_id,
                    seed=server_stats.get('seed'),  # CRITICAL: Get seed from server stats
                    scenario_name="inference",  # Mark as inference episode
                    obs_version=request.normalization.obs_version,
                    vecnorm_stats_id=request.normalization.vecnorm_stats_id,
                    transform_version=server_stats.get('transform_version')  # CRITICAL: Server-side transform_version
                )
                # Track episode
                inference_logger.current_episode_id = request.meta.episode_id
                inference_logger.step_count = 0
            
            # Log the inference step
            inference_logger.log_inference_step(
                t=request.meta.t,
                request=request,
                response=response,
                latency_ms=latency_ms
            )
            
            # Update episode tracking
            inference_logger.last_t = request.meta.t
            inference_logger.step_count = getattr(inference_logger, 'step_count', 0) + 1
            
            # CRITICAL: Check for episode completion (at max steps)
            if request.env.episode_step >= request.env.max_steps - 1:
                inference_logger.end_inference_episode(
                    outcome="timeout",  # Completed all steps
                    final_time=request.meta.t,
                    total_steps=inference_logger.step_count,
                    notes=f"Episode completed after {request.env.max_steps} steps"
                )
                
        except Exception as e:
            logger.warning(f"Failed to log inference step: {e}")


def verify_manifest(runs_dir: str = "runs") -> Dict[str, Any]:
    """
    Verify manifest.json structure and episode file integrity.
    
    Args:
        runs_dir: Directory containing runs and manifest.json
        
    Returns:
        Verification result dictionary with status and details
    """
    runs_path = Path(runs_dir)
    manifest_path = runs_path / "manifest.json"
    
    result = {
        "manifest_exists": False,
        "manifest_valid": False,
        "episodes_found": 0,
        "episodes_verified": 0,
        "missing_files": [],
        "invalid_episodes": [],
        "summary_check": {
            "episodes_with_summary": 0,
            "episodes_missing_summary": 0
        },
        "version_fields_check": {
            "episodes_with_version_fields": 0,
            "episodes_missing_version_fields": 0
        }
    }
    
    # Check if manifest exists
    if not manifest_path.exists():
        result["error"] = f"Manifest file not found: {manifest_path}"
        return result
    
    result["manifest_exists"] = True
    
    try:
        # Load and validate manifest structure
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Check required manifest fields
        required_fields = ["schema_version", "coord_frame", "units", "episodes"]
        missing_fields = [field for field in required_fields if field not in manifest]
        if missing_fields:
            result["error"] = f"Manifest missing required fields: {missing_fields}"
            return result
        
        result["manifest_valid"] = True
        episodes = manifest.get("episodes", [])
        result["episodes_found"] = len(episodes)
        
        # Verify each episode
        for episode in episodes:
            episode_id = episode.get("id", "unknown")
            episode_file = episode.get("file")
            
            if not episode_file:
                result["invalid_episodes"].append(f"{episode_id}: Missing 'file' field")
                continue
                
            episode_path = runs_path / episode_file
            
            # Check if episode file exists
            if not episode_path.exists():
                result["missing_files"].append(str(episode_path))
                continue
            
            # Verify episode file content
            try:
                episode_valid = True
                has_summary = False
                has_version_fields = False
                
                with open(episode_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                            
                        try:
                            record = json.loads(line)
                            
                            # Check for summary line
                            if "summary" in record:
                                has_summary = True
                                summary = record["summary"]
                                # Verify required summary fields
                                required_summary_fields = ["outcome", "episode_duration"]
                                missing_summary_fields = [f for f in required_summary_fields if f not in summary]
                                if missing_summary_fields:
                                    result["invalid_episodes"].append(
                                        f"{episode_id}: Summary missing fields: {missing_summary_fields}"
                                    )
                                    episode_valid = False
                            
                            # Check for version fields in timesteps (not summary lines)
                            elif "t" in record and "summary" not in record:
                                # Check for version fields in inference section or top level
                                inference_section = record.get("inference", {})
                                if (
                                    inference_section.get("obs_version") and
                                    inference_section.get("vecnorm_stats_id") and
                                    inference_section.get("transform_version") and
                                    inference_section.get("policy_id")
                                ):
                                    has_version_fields = True
                                    
                        except json.JSONDecodeError as e:
                            result["invalid_episodes"].append(
                                f"{episode_id}:{line_num}: JSON decode error: {e}"
                            )
                            episode_valid = False
                            break
                
                if episode_valid:
                    result["episodes_verified"] += 1
                    
                if has_summary:
                    result["summary_check"]["episodes_with_summary"] += 1
                else:
                    result["summary_check"]["episodes_missing_summary"] += 1
                    
                if has_version_fields:
                    result["version_fields_check"]["episodes_with_version_fields"] += 1
                else:
                    result["version_fields_check"]["episodes_missing_version_fields"] += 1
                    
            except Exception as e:
                result["invalid_episodes"].append(f"{episode_id}: File read error: {e}")
        
        # Add success status
        result["success"] = (
            result["manifest_valid"] and
            len(result["missing_files"]) == 0 and
            len(result["invalid_episodes"]) == 0
        )
        
        return result
        
    except Exception as e:
        result["error"] = f"Failed to process manifest: {e}"
        return result


def main():
    """Main entry point for Hlynr bridge server."""
    parser = argparse.ArgumentParser(description='Hlynr Unity Bridge Server v1.0')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='easy',
        help='Scenario name for environment configuration'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Server host address'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Server port'
    )
    
    parser.add_argument(
        '--obs-version',
        type=str,
        default='obs_v1.0',
        help='Observation version'
    )
    
    parser.add_argument(
        '--vecnorm-stats-id',
        type=str,
        default=None,
        help='VecNormalize statistics ID'
    )
    
    parser.add_argument(
        '--transform-version',
        type=str,
        default='tfm_v1.0',
        help='Coordinate transform version'
    )
    
    parser.add_argument(
        '--policy-id',
        type=str,
        default=None,
        help='Policy identifier'
    )
    
    parser.add_argument(
        '--rate-max-radps',
        type=float,
        default=10.0,
        help='Maximum angular rate for safety clamps (rad/s)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    try:
        # Create and start server
        server = HlynrBridgeServer(
            checkpoint_path=args.checkpoint,
            scenario_name=args.scenario,
            config_path=args.config,
            host=args.host,
            port=args.port,
            obs_version=args.obs_version,
            vecnorm_stats_id=args.vecnorm_stats_id,
            transform_version=args.transform_version,
            policy_id=args.policy_id,
            rate_max_radps=args.rate_max_radps
        )
        
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()