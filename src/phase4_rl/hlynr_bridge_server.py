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
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import collections

# Import our custom modules
try:
    from .schemas import (
        InferenceRequest, InferenceResponse, HealthResponse, MetricsResponse,
        ActionCommand, RateCommand, DiagnosticsInfo, SafetyInfo, ClipFractions
    )
    from .transforms import get_transform, validate_transform_version
    from .normalize import get_vecnorm_manager, load_vecnormalize_by_id
    from .clamps import get_safety_clamp_system, SafetyLimits
    from .inference_logger import get_inference_logger, reset_inference_logger
    from .config import get_config, reset_config
    from .scenarios import get_scenario_loader, reset_scenario_loader
    from .radar_env import RadarEnv
    from .fast_sim_env import FastSimEnv
    from .env_config import get_bridge_config, BridgeServerConfig
except ImportError:
    # Fallback for direct execution
    from schemas import (
        InferenceRequest, InferenceResponse, HealthResponse, MetricsResponse,
        ActionCommand, RateCommand, DiagnosticsInfo, SafetyInfo, ClipFractions
    )
    from transforms import get_transform, validate_transform_version
    from normalize import get_vecnorm_manager, load_vecnormalize_by_id
    from clamps import get_safety_clamp_system, SafetyLimits
    from inference_logger import get_inference_logger, reset_inference_logger
    from config import get_config, reset_config
    from scenarios import get_scenario_loader, reset_scenario_loader
    from radar_env import RadarEnv
    from fast_sim_env import FastSimEnv
    from env_config import get_bridge_config, BridgeServerConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables  
app = Flask(__name__)
# CORS will be configured based on environment variables
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
    'policy_id': None
}

# Latency tracking (for p50/p95 calculation)
latency_history = collections.deque(maxlen=10000)  # Keep last 10k latencies


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
        
        # Configure CORS if enabled
        if env_config.enable_cors:
            CORS(app, origins=env_config.cors_origins)
            logger.info(f"CORS enabled for origins: {env_config.cors_origins}")
        
        # Reset global instances
        reset_config()
        reset_scenario_loader()
        reset_inference_logger()
        
        # Load configuration
        self.config_loader = get_config(self.config_path)
        self.config = self.config_loader._config
        
        # Load scenario
        self.scenario_loader = get_scenario_loader()
        self.scenario_config = self.scenario_loader.create_environment_config(
            self.scenario_name, self.config
        )
        
        # Setup safety limits
        self.safety_limits = SafetyLimits(rate_max_radps=self.rate_max_radps)
        
        logger.info(f"Hlynr bridge server initialized (policy_id: {self.policy_id})")
        if env_config.debug_mode:
            logger.info("Server configuration:")
            logger.info(f"  Checkpoint: {self.checkpoint_path}")
            logger.info(f"  Host:Port: {self.host}:{self.port}")
            logger.info(f"  Scenario: {self.scenario_name}")
            logger.info(f"  Safety rate limit: {self.rate_max_radps} rad/s")
    
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
        vecnorm_manager = get_vecnorm_manager()\n        \n        if self.vecnorm_stats_id:\n            try:\n                vec_normalize, vec_normalize_info = load_vecnormalize_by_id(\n                    self.vecnorm_stats_id, env, self.obs_version\n                )\n                env = vec_normalize\n                logger.info(f\"Loaded VecNormalize: {self.vecnorm_stats_id}\")\n            except Exception as e:\n                logger.warning(f\"Failed to load VecNormalize by ID: {e}\")\n                logger.info(\"Attempting to load from legacy file...\")\n                self._load_legacy_vecnormalize(env, vecnorm_manager)\n        else:\n            logger.info(\"No vecnorm_stats_id provided, attempting legacy file...\")\n            self._load_legacy_vecnormalize(env, vecnorm_manager)\n        \n        # Load model\n        model = PPO.load(self.checkpoint_path, env=env)\n        \n        # Setup coordinate transform\n        if not validate_transform_version(self.transform_version):\n            raise ValueError(f\"Unsupported transform version: {self.transform_version}\")\n        transform = get_transform(self.transform_version)\n        \n        # Setup safety clamp system\n        safety_clamp_system = get_safety_clamp_system(self.safety_limits)\n        \n        # Setup inference logger\n        inference_logger = get_inference_logger(\n            output_dir=\"inference_episodes\",\n            policy_id=self.policy_id\n        )\n        \n        # Update server stats\n        server_stats.update({\n            'model_loaded_at': time.time(),\n            'checkpoint_path': str(self.checkpoint_path),\n            'vecnorm_stats_id': vec_normalize_info.stats_id if vec_normalize_info else None,\n            'transform_version': self.transform_version,\n            'policy_id': self.policy_id\n        })\n        \n        logger.info(\"Model loaded successfully!\")\n        logger.info(f\"Environment: {self.scenario_name} scenario\")\n        logger.info(f\"Observation space: {env.observation_space}\")\n        logger.info(f\"Action space: {env.action_space}\")\n        logger.info(f\"Transform version: {self.transform_version}\")\n        if vec_normalize_info:\n            logger.info(f\"VecNormalize: {vec_normalize_info.stats_id} (v{vec_normalize_info.obs_version})\")\n    \n    def _load_legacy_vecnormalize(self, env, vecnorm_manager):\n        \"\"\"Load VecNormalize from legacy file and register it.\"\"\"\n        global vec_normalize, vec_normalize_info\n        \n        # Look for legacy vec_normalize files\n        vecnorm_files = [\n            self.checkpoint_path.parent / \"vec_normalize_final.pkl\",\n            self.checkpoint_path.parent / \"vec_normalize.pkl\"\n        ]\n        \n        for vecnorm_path in vecnorm_files:\n            if vecnorm_path.exists():\n                logger.info(f\"Loading legacy VecNormalize from: {vecnorm_path}\")\n                \n                # Register the legacy file\n                stats_id = vecnorm_manager.create_from_legacy_file(\n                    legacy_path=str(vecnorm_path),\n                    env=env,\n                    obs_version=self.obs_version,\n                    description=f\"Auto-imported from {vecnorm_path.name}\"\n                )\n                \n                # Load the registered VecNormalize\n                vec_normalize, vec_normalize_info = load_vecnormalize_by_id(\n                    stats_id, env, self.obs_version\n                )\n                \n                # Update server to use this stats_id\n                self.vecnorm_stats_id = stats_id\n                \n                logger.info(f\"Registered legacy VecNormalize as: {stats_id}\")\n                return\n        \n        logger.warning(\"No VecNormalize file found - running without normalization\")\n    \n    def run(self):\n        \"\"\"Start the bridge server.\"\"\"\n        # Load model before starting server\n        self.load_model()\n        \n        logger.info(f\"Starting Hlynr Bridge Server v1.0 on {self.host}:{self.port}\")\n        logger.info(f\"API Endpoints:\")\n        logger.info(f\"  POST /v1/inference - Unity RL Inference API v1.0\")\n        logger.info(f\"  POST /act - Legacy inference endpoint\")\n        logger.info(f\"  GET /healthz - Health check\")\n        logger.info(f\"  GET /metrics - Performance metrics\")\n        \n        app.run(host=self.host, port=self.port, debug=False)\n\n\n@app.route('/v1/inference', methods=['POST'])\ndef v1_inference():\n    \"\"\"\n    Unity RL Inference API v1.0 endpoint.\n    \n    Implements the full PRP specification with:\n    - Pydantic request/response validation\n    - Coordinate transforms\n    - Safety clamps\n    - Comprehensive logging\n    - Latency monitoring\n    \"\"\"\n    global model, env, transform, safety_clamp_system, inference_logger, server_stats\n    \n    start_time = time.perf_counter()\n    \n    try:\n        # Validate request\n        if not request.is_json:\n            return _error_response(\"Request must be JSON\", 400)\n        \n        # Parse and validate request schema\n        try:\n            inference_request = InferenceRequest(**request.get_json())\n        except Exception as e:\n            return _error_response(f\"Invalid request schema: {e}\", 400)\n        \n        # Check model is loaded\n        if model is None:\n            return _error_response(\"Model not loaded\", 500)\n        \n        # Validate versioning compatibility\n        validation_error = _validate_request_versions(inference_request)\n        if validation_error:\n            return _error_response(validation_error, 400)\n        \n        # Transform Unity state to RL observation\n        try:\n            observation = _unity_to_rl_observation(inference_request)\n        except Exception as e:\n            return _error_response(f\"Observation conversion failed: {e}\", 500)\n        \n        # Policy inference\n        policy_start = time.perf_counter()\n        try:\n            action, _states = model.predict(observation, deterministic=True)\n            action = action[0] if len(action.shape) > 1 else action\n        except Exception as e:\n            return _error_response(f\"Policy inference failed: {e}\", 500)\n        \n        policy_latency = (time.perf_counter() - policy_start) * 1000  # ms\n        \n        # Convert RL action to Unity command\n        try:\n            action_command = _rl_to_unity_action(action)\n        except Exception as e:\n            return _error_response(f\"Action conversion failed: {e}\", 500)\n        \n        # Apply safety clamps\n        clamped_action, safety_info = safety_clamp_system.apply_safety_clamps(action_command)\n        \n        # Compute diagnostics\n        diagnostics = _compute_diagnostics(observation, policy_latency)\n        \n        # Create response\n        response = InferenceResponse(\n            action=clamped_action,\n            diagnostics=diagnostics,\n            safety=safety_info\n        )\n        \n        # Calculate total latency\n        total_latency = (time.perf_counter() - start_time) * 1000  # ms\n        \n        # Update server statistics\n        _update_server_stats(total_latency, success=True)\n        \n        # Log inference step\n        _log_inference_step(inference_request, response, total_latency)\n        \n        logger.debug(f\"v1/inference completed in {total_latency:.2f}ms\")\n        \n        return jsonify(response.dict())\n        \n    except Exception as e:\n        total_latency = (time.perf_counter() - start_time) * 1000\n        _update_server_stats(total_latency, success=False)\n        logger.error(f\"v1/inference error: {e}\")\n        return _error_response(f\"Internal server error: {e}\", 500)\n\n\n@app.route('/healthz', methods=['GET'])\ndef health_check():\n    \"\"\"Enhanced health check endpoint.\"\"\"\n    global model, vec_normalize_info, transform, server_stats\n    \n    loaded_models = {}\n    if model is not None:\n        loaded_models['ppo_policy'] = str(server_stats.get('checkpoint_path', 'unknown'))\n    \n    response = HealthResponse(\n        status='healthy' if model is not None else 'not_ready',\n        model_loaded=model is not None,\n        loaded_models=loaded_models,\n        vecnorm_stats_loaded=vec_normalize_info.stats_id if vec_normalize_info else None,\n        transform_version=server_stats.get('transform_version')\n    )\n    \n    return jsonify(response.dict())\n\n\n@app.route('/metrics', methods=['GET'])\ndef get_metrics():\n    \"\"\"Enhanced metrics endpoint with latency percentiles.\"\"\"\n    global server_stats, safety_clamp_system, latency_history\n    \n    # Calculate latency percentiles\n    latencies = list(latency_history) if latency_history else [0.0]\n    latencies_array = np.array(latencies)\n    \n    # Calculate requests per second (last minute)\n    current_time = time.time()\n    recent_latencies = [lat for lat in latency_history \n                       if hasattr(lat, 'timestamp') and current_time - lat.timestamp < 60]\n    rps = len(recent_latencies) / 60.0 if recent_latencies else 0.0\n    \n    # Get safety clamp statistics\n    clamp_stats = safety_clamp_system.get_clamp_statistics() if safety_clamp_system else {}\n    \n    response = MetricsResponse(\n        requests_served=server_stats['requests_served'],\n        requests_failed=server_stats['requests_failed'],\n        latency_p50_ms=float(np.percentile(latencies_array, 50)),\n        latency_p95_ms=float(np.percentile(latencies_array, 95)),\n        latency_p99_ms=float(np.percentile(latencies_array, 99)),\n        latency_mean_ms=float(np.mean(latencies_array)),\n        requests_per_second=rps,\n        safety_clamps_total=clamp_stats.get('total_clamped', 0),\n        safety_clamp_rate=clamp_stats.get('overall_clamp_rate', 0.0),\n        model_loaded_at=server_stats.get('model_loaded_at'),\n        vecnorm_stats_id=server_stats.get('vecnorm_stats_id')\n    )\n    \n    return jsonify(response.dict())\n\n\n# Legacy endpoint compatibility\n@app.route('/act', methods=['POST'])\ndef legacy_act():\n    \"\"\"Legacy inference endpoint for backward compatibility.\"\"\"\n    try:\n        data = request.get_json()\n        observation = np.array(data['observation'], dtype=np.float32)\n        deterministic = data.get('deterministic', True)\n        \n        # Simple inference without full v1.0 pipeline\n        action, _states = model.predict(observation.reshape(1, -1), deterministic=deterministic)\n        action = action[0] if len(action.shape) > 1 else action\n        \n        return jsonify({\n            'success': True,\n            'action': action.tolist(),\n            'inference_time': 0.0,  # Not tracked for legacy endpoint\n            'error': None\n        })\n        \n    except Exception as e:\n        return jsonify({\n            'success': False,\n            'error': str(e),\n            'action': None,\n            'inference_time': 0.0\n        }), 500\n\n\ndef _error_response(message: str, status_code: int):\n    \"\"\"Create standardized error response.\"\"\"\n    return jsonify({\n        'success': False,\n        'error': message,\n        'timestamp': time.time()\n    }), status_code\n\n\ndef _validate_request_versions(request: InferenceRequest) -> Optional[str]:\n    \"\"\"Validate request versions are compatible.\"\"\"\n    global vec_normalize_info, transform\n    \n    # Validate transform version\n    if not validate_transform_version(request.normalization.transform_version):\n        return f\"Unsupported transform version: {request.normalization.transform_version}\"\n    \n    # Validate VecNormalize compatibility\n    if vec_normalize_info:\n        if request.normalization.vecnorm_stats_id != vec_normalize_info.stats_id:\n            return f\"VecNormalize ID mismatch: requested {request.normalization.vecnorm_stats_id}, loaded {vec_normalize_info.stats_id}\"\n        \n        if request.normalization.obs_version != vec_normalize_info.obs_version:\n            return f\"Observation version mismatch: requested {request.normalization.obs_version}, loaded {vec_normalize_info.obs_version}\"\n    \n    return None\n\n\ndef _unity_to_rl_observation(request: InferenceRequest) -> np.ndarray:\n    \"\"\"Convert Unity request to RL observation.\"\"\"\n    global transform\n    \n    # Transform Unity coordinates to ENU if needed\n    if request.frames.unity_lh:\n        # Transform blue state\n        blue_enu = transform.transform_state_unity_to_enu({\n            'pos_m': request.blue.pos_m,\n            'vel_mps': request.blue.vel_mps,\n            'quat_wxyz': request.blue.quat_wxyz,\n            'ang_vel_radps': request.blue.ang_vel_radps\n        })\n        \n        # Transform red state\n        red_enu = transform.transform_state_unity_to_enu({\n            'pos_m': request.red.pos_m,\n            'vel_mps': request.red.vel_mps,\n            'quat_wxyz': request.red.quat_wxyz\n        })\n    else:\n        # Already in ENU\n        blue_enu = {\n            'pos_m': request.blue.pos_m,\n            'vel_mps': request.blue.vel_mps,\n            'quat_wxyz': request.blue.quat_wxyz,\n            'ang_vel_radps': request.blue.ang_vel_radps\n        }\n        red_enu = {\n            'pos_m': request.red.pos_m,\n            'vel_mps': request.red.vel_mps,\n            'quat_wxyz': request.red.quat_wxyz\n        }\n    \n    # Assemble observation vector (this should match the training observation format)\n    # Note: This is a simplified example - actual implementation depends on your RL environment\n    obs_vector = [\n        # Blue state\n        *blue_enu['pos_m'],\n        *blue_enu['vel_mps'],\n        *blue_enu['quat_wxyz'],\n        *blue_enu['ang_vel_radps'],\n        request.blue.fuel_frac,\n        \n        # Red state\n        *red_enu['pos_m'],\n        *red_enu['vel_mps'],\n        *red_enu['quat_wxyz'],\n        \n        # Guidance\n        *request.guidance.los_unit,\n        *request.guidance.los_rate_radps,\n        request.guidance.range_m,\n        request.guidance.closing_speed_mps,\n        float(request.guidance.fov_ok),\n        float(request.guidance.g_limit_ok),\n        \n        # Environment\n        request.env.episode_step / max(request.env.max_steps, 1),  # Normalized step\n        *(request.env.wind_mps or [0.0, 0.0, 0.0])\n    ]\n    \n    return np.array(obs_vector, dtype=np.float32).reshape(1, -1)\n\n\ndef _rl_to_unity_action(action: np.ndarray) -> ActionCommand:\n    \"\"\"Convert RL action to Unity action command.\"\"\"\n    global transform\n    \n    # Extract action components (adjust based on your action space)\n    if len(action) >= 4:\n        rate_pitch = float(action[0])\n        rate_yaw = float(action[1])\n        rate_roll = float(action[2])\n        thrust = float(action[3])\n        aux = action[4:].tolist() if len(action) > 4 else []\n    else:\n        raise ValueError(f\"Invalid action dimension: {len(action)}\")\n    \n    # Transform angular rates from ENU to Unity if needed\n    enu_rates = [rate_pitch, rate_yaw, rate_roll]\n    unity_rates = transform.enu_to_unity_angular_velocity(enu_rates)\n    \n    return ActionCommand(\n        rate_cmd_radps=RateCommand(\n            pitch=unity_rates[0],\n            yaw=unity_rates[1],\n            roll=unity_rates[2]\n        ),\n        thrust_cmd=thrust,\n        aux=aux\n    )\n\n\ndef _compute_diagnostics(observation: np.ndarray, policy_latency_ms: float) -> DiagnosticsInfo:\n    \"\"\"Compute inference diagnostics.\"\"\"\n    global vec_normalize, vecnorm_manager\n    \n    # Compute observation clipping fractions\n    clip_fractions = {'low': 0.0, 'high': 0.0}\n    if vec_normalize:\n        vecnorm_manager = get_vecnorm_manager()\n        clip_fractions = vecnorm_manager.compute_clip_fractions(observation)\n    \n    return DiagnosticsInfo(\n        policy_latency_ms=policy_latency_ms,\n        obs_clip_fractions=ClipFractions(\n            low=clip_fractions['low'],\n            high=clip_fractions['high']\n        ),\n        value_estimate=None  # Could be computed if needed\n    )\n\n\ndef _update_server_stats(latency_ms: float, success: bool):\n    \"\"\"Update server statistics.\"\"\"\n    global server_stats, latency_history\n    \n    server_stats['requests_served'] += 1\n    if not success:\n        server_stats['requests_failed'] += 1\n    \n    if success:\n        server_stats['total_inference_time'] += latency_ms\n        server_stats['average_inference_time'] = (\n            server_stats['total_inference_time'] / server_stats['requests_served']\n        )\n        \n        # Add to latency history for percentile calculation\n        latency_history.append(latency_ms)\n\n\ndef _log_inference_step(request: InferenceRequest, response: InferenceResponse, latency_ms: float):\n    \"\"\"Log inference step to episode logger.\"\"\"\n    global inference_logger\n    \n    if inference_logger and inference_logger.enable_logging:\n        try:\n            # Check if we need to start a new episode\n            if not hasattr(inference_logger, 'current_episode_file') or \\\n               inference_logger.current_episode_file is None:\n                inference_logger.begin_inference_episode(\n                    episode_id=request.meta.episode_id,\n                    seed=None,  # Could extract from request if available\n                    scenario_name=None,  # Could extract from request if available\n                    obs_version=request.normalization.obs_version,\n                    vecnorm_stats_id=request.normalization.vecnorm_stats_id,\n                    transform_version=request.normalization.transform_version\n                )\n            \n            # Log the inference step\n            inference_logger.log_inference_step(\n                t=request.meta.t,\n                request=request,\n                response=response,\n                latency_ms=latency_ms\n            )\n        except Exception as e:\n            logger.warning(f\"Failed to log inference step: {e}\")\n\n\ndef main():\n    \"\"\"Main entry point for Hlynr bridge server.\"\"\"\n    parser = argparse.ArgumentParser(description='Hlynr Unity Bridge Server v1.0')\n    \n    parser.add_argument(\n        '--checkpoint',\n        type=str,\n        required=True,\n        help='Path to model checkpoint'\n    )\n    \n    parser.add_argument(\n        '--scenario',\n        type=str,\n        default='easy',\n        help='Scenario name for environment configuration'\n    )\n    \n    parser.add_argument(\n        '--config',\n        type=str,\n        default=None,\n        help='Path to configuration file'\n    )\n    \n    parser.add_argument(\n        '--host',\n        type=str,\n        default='0.0.0.0',\n        help='Server host address'\n    )\n    \n    parser.add_argument(\n        '--port',\n        type=int,\n        default=5000,\n        help='Server port'\n    )\n    \n    parser.add_argument(\n        '--obs-version',\n        type=str,\n        default='obs_v1.0',\n        help='Observation version'\n    )\n    \n    parser.add_argument(\n        '--vecnorm-stats-id',\n        type=str,\n        default=None,\n        help='VecNormalize statistics ID'\n    )\n    \n    parser.add_argument(\n        '--transform-version',\n        type=str,\n        default='tfm_v1.0',\n        help='Coordinate transform version'\n    )\n    \n    parser.add_argument(\n        '--policy-id',\n        type=str,\n        default=None,\n        help='Policy identifier'\n    )\n    \n    parser.add_argument(\n        '--rate-max-radps',\n        type=float,\n        default=10.0,\n        help='Maximum angular rate for safety clamps (rad/s)'\n    )\n    \n    parser.add_argument(\n        '--verbose',\n        action='store_true',\n        help='Enable verbose logging'\n    )\n    \n    args = parser.parse_args()\n    \n    # Setup logging level\n    if args.verbose:\n        logging.getLogger().setLevel(logging.DEBUG)\n    \n    # Validate checkpoint\n    checkpoint_path = Path(args.checkpoint)\n    if not checkpoint_path.exists():\n        print(f\"Error: Checkpoint not found: {checkpoint_path}\")\n        sys.exit(1)\n    \n    try:\n        # Create and start server\n        server = HlynrBridgeServer(\n            checkpoint_path=args.checkpoint,\n            scenario_name=args.scenario,\n            config_path=args.config,\n            host=args.host,\n            port=args.port,\n            obs_version=args.obs_version,\n            vecnorm_stats_id=args.vecnorm_stats_id,\n            transform_version=args.transform_version,\n            policy_id=args.policy_id,\n            rate_max_radps=args.rate_max_radps\n        )\n        \n        server.run()\n        \n    except KeyboardInterrupt:\n        logger.info(\"Server stopped by user\")\n    except Exception as e:\n        logger.error(f\"Server failed: {e}\")\n        sys.exit(1)\n\n\nif __name__ == '__main__':\n    main()