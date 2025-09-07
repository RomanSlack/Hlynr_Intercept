"""
Enhanced episode logging for Unity RL Inference API v1.0.

This module extends the existing episode logging with deterministic inference logging
that includes all versioning information required by the PRP.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

# Import schema types - handle different import contexts
try:
    from ..hlynr_bridge.schemas import InferenceRequest, InferenceResponse
except ImportError:
    try:
        from hlynr_bridge.schemas import InferenceRequest, InferenceResponse
    except ImportError:
        # Fallback for testing - create mock types
        from typing import Any
        InferenceRequest = Any
        InferenceResponse = Any
from .episode_logger import EpisodeLogger, Event

# Import centralized path resolver
try:
    from hlynr_bridge.paths import logs_inference
except ImportError:
    # Fallback for development/testing
    def logs_inference():
        return Path("inference_episodes")

logger = logging.getLogger(__name__)


@dataclass
class InferenceLogEntry:
    """Single inference request/response log entry."""
    timestamp: float
    request: Dict[str, Any]
    response: Dict[str, Any]
    latency_ms: float
    safety_clamped: bool
    obs_version: str
    vecnorm_stats_id: str
    transform_version: str
    policy_id: str


class InferenceEpisodeLogger(EpisodeLogger):
    """
    Enhanced episode logger for Unity RL Inference API v1.0.
    
    Extends the base EpisodeLogger with deterministic inference logging
    that includes all required versioning information.
    """
    
    def __init__(self,
                 output_dir: Optional[str] = None,
                 coord_frame: str = "ENU_RH",
                 dt_nominal: float = 0.01,
                 enable_logging: bool = True,
                 policy_id: Optional[str] = None):
        """
        Initialize inference episode logger.
        
        Args:
            output_dir: Base directory for output files (deprecated - uses centralized logging)
            coord_frame: Coordinate frame
            dt_nominal: Nominal timestep
            enable_logging: Whether logging is enabled
            policy_id: Policy identifier for this session
        """
        # Use centralized inference logging directory instead of custom output_dir
        if output_dir is not None and output_dir != "inference_episodes":
            logger.warning(f"InferenceEpisodeLogger output_dir parameter is deprecated. "
                         f"Using centralized logging under logs/inference/ instead of {output_dir}")
        
        # Call parent constructor (it will use centralized paths automatically)
        super().__init__(output_dir, coord_frame, dt_nominal, enable_logging)
        
        # Override the run directory to use inference-specific location
        if self.enable_logging:
            from hlynr_bridge.paths import generate_run_timestamp
            run_timestamp = generate_run_timestamp()
            self.run_dir = logs_inference() / run_timestamp
            self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.policy_id = policy_id or f"policy_{int(time.time())}"
        self.session_id = str(uuid.uuid4())[:8]
        
        # Inference-specific tracking
        self.current_vecnorm_stats_id: Optional[str] = None
        self.current_obs_version: Optional[str] = None
        self.current_transform_version: Optional[str] = None
        
        # Per-request logging
        self.inference_log_entries: List[InferenceLogEntry] = []
        self.max_inference_entries = 10000  # Keep last 10k entries
        
        logger.info(f"Inference episode logger initialized (policy_id: {self.policy_id})")
    
    def begin_inference_episode(self,
                               episode_id: str,
                               seed: Optional[int] = None,
                               scenario_name: Optional[str] = None,
                               obs_version: str = "obs_v1.0",
                               vecnorm_stats_id: str = "unknown",
                               transform_version: str = "tfm_v1.0",
                               interceptor_config: Optional[Dict[str, Any]] = None,
                               threat_config: Optional[Dict[str, Any]] = None,
                               notes: Optional[str] = None):
        """
        Begin logging an inference episode with full versioning.
        
        Args:
            episode_id: Unique episode identifier
            seed: Random seed
            scenario_name: Scenario name
            obs_version: Observation version
            vecnorm_stats_id: VecNormalize statistics ID
            transform_version: Coordinate transform version
            interceptor_config: Interceptor configuration
            threat_config: Threat configuration
            notes: Optional notes
        """
        if not self.enable_logging:
            return
        
        # Update current versioning
        self.current_obs_version = obs_version
        self.current_vecnorm_stats_id = vecnorm_stats_id
        self.current_transform_version = transform_version
        
        # Set episode metadata with versioning
        self.episode_count += 1
        filename = f"{episode_id}.jsonl"
        
        self.current_episode_file = self.run_dir / filename
        self.current_episode_data = []
        self.episode_start_time = time.time()
        
        # Store episode metadata with full versioning
        self.episode_metadata = {
            "ep_id": episode_id,
            "seed": seed,
            "scenario": scenario_name,
            "obs_version": obs_version,
            "vecnorm_stats_id": vecnorm_stats_id,
            "transform_version": transform_version,
            "policy_id": self.policy_id,
            "session_id": self.session_id,
            "notes": notes or ""
        }
        
        # Add to manifest with versioning
        self.manifest["episodes"].append({
            "id": episode_id,
            "file": filename,
            "seed": seed,
            "scenario": scenario_name,
            "obs_version": obs_version,
            "vecnorm_stats_id": vecnorm_stats_id,
            "transform_version": transform_version,
            "policy_id": self.policy_id,
            "session_id": self.session_id,
            "notes": notes or ""
        })
        # Write header record with full versioning
        header = {
            "t": 0.0,
            "meta": {
                "ep_id": episode_id,
                "seed": seed,
                "coord_frame": self.coord_frame,
                "dt_nominal": self.dt_nominal,
                "scenario": scenario_name,
                "obs_version": obs_version,
                "vecnorm_stats_id": vecnorm_stats_id,
                "transform_version": transform_version,
                "policy_id": self.policy_id,
                "session_id": self.session_id
            },
            "scene": {}
        }

        # Add interceptor configuration
        if interceptor_config:
            header["scene"]["interceptor_0"] = {
                "mass_kg": interceptor_config.get("mass", 10.0),
                "max_torque": interceptor_config.get("max_torque", [8000, 8000, 2000]),
                "sensor_fov_deg": interceptor_config.get("sensor_fov", 30.0),
                "max_thrust_n": interceptor_config.get("max_thrust", 1000.0)
            }

        # Add threat configuration
        if threat_config:
            header["scene"]["threat_0"] = {
                "type": threat_config.get("type", "ballistic"),
                "mass_kg": threat_config.get("mass", 100.0),
                "aim_point": threat_config.get("aim_point", [0, 0, 0])
            }

        self._write_record(header)
        logger.info(f"Started inference episode: {episode_id} (v{obs_version}, {vecnorm_stats_id})")

    def log_inference_step(
        self,
        t: float,
        request: InferenceRequest,
        response: InferenceResponse,
        latency_ms: float,
        interceptor_state: Optional[Dict[str, Any]] = None,
        threat_state: Optional[Dict[str, Any]] = None,
        events: Optional[List[Event]] = None
    ):
        """Log a single inference timestep with full versioning."""
        if not self.enable_logging or self.current_episode_file is None:
            return

        # Convert states to agent format if provided
        agents: Dict[str, Any] = {}

        if interceptor_state:
            agents["interceptor_0"] = self._process_agent_state(interceptor_state)
        else:
            agents["interceptor_0"] = {
                "p": request.blue.pos_m,
                "q": request.blue.quat_wxyz,
                "v": request.blue.vel_mps,
                "w": request.blue.ang_vel_radps,
                "fuel_frac": request.blue.fuel_frac,
                "status": "active",
            }

        if threat_state:
            agents["threat_0"] = self._process_agent_state(threat_state)
        else:
            agents["threat_0"] = {
                "p": request.red.pos_m,
                "q": request.red.quat_wxyz,
                "v": request.red.vel_mps,
                "status": "active",
            }

        # Add action command to interceptor state
        if "interceptor_0" in agents:
            agents["interceptor_0"]["u"] = [
                response.action.rate_cmd_radps.pitch,
                response.action.rate_cmd_radps.yaw,
                response.action.rate_cmd_radps.roll,
                response.action.thrust_cmd,
            ] + response.action.aux

        # Build record with versioning (top-level per spec)
        record = {
            "t": round(t, 3),
            "agents": agents,
            # REQUIRED versioning fields at top level:
            "obs_version": request.normalization.obs_version,
            "vecnorm_stats_id": request.normalization.vecnorm_stats_id,
            # transform_version is server-side; do NOT trust request:
            "transform_version": self.current_transform_version,
            "policy_id": self.policy_id,
            "inference": {
                "latency_ms": round(latency_ms, 2),
                "safety_clamped": response.safety.clamped,
                "clamp_reason": response.safety.clamp_reason,
                "value_estimate": response.diagnostics.value_estimate,
                "obs_clip_fractions": {
                    "low": response.diagnostics.obs_clip_fractions.low,
                    "high": response.diagnostics.obs_clip_fractions.high,
                },
            },
            "guidance": {
                "los_unit": request.guidance.los_unit,
                "los_rate_radps": request.guidance.los_rate_radps,
                "range_m": request.guidance.range_m,
                "closing_speed_mps": request.guidance.closing_speed_mps,
                "fov_ok": request.guidance.fov_ok,
                "g_limit_ok": request.guidance.g_limit_ok,
            },
        }

        if events:
            record["events"] = [asdict(e) if isinstance(e, Event) else e for e in events]

        self._write_record(record)
        self._log_inference_entry(request, response, latency_ms)

    def end_inference_episode(
        self,
        outcome: str,
        final_time: float,
        miss_distance: Optional[float] = None,
        impact_time: Optional[float] = None,
        total_steps: Optional[int] = None,
        notes: Optional[str] = None,
    ):
        """End the current inference episode with full summary."""
        if not self.enable_logging or self.current_episode_file is None:
            return

        summary = {
            "t": round(final_time, 3),
            "summary": {
                "outcome": outcome,                          # "hit"|"miss"|"timeout"
                "episode_duration_s": round(final_time, 3),  # use *_s per spec
                "notes": notes or "",
                "obs_version": self.current_obs_version,
                "vecnorm_stats_id": self.current_vecnorm_stats_id,
                "transform_version": self.current_transform_version,
                "policy_id": self.policy_id,
                "session_id": self.session_id,
            },
        }
        if miss_distance is not None:
            summary["summary"]["miss_distance_m"] = round(miss_distance, 3)
        if impact_time is not None:
            summary["summary"]["impact_time_s"] = round(impact_time, 3)
        if total_steps is not None:
            summary["summary"]["steps"] = total_steps
        if "seed" in self.episode_metadata:
            summary["summary"]["seed"] = self.episode_metadata["seed"]

        self._write_record(summary)

        # Close episode & update manifest
        self.current_episode_file = None
        self.current_episode_data = []
        self._save_manifest()

        logger.info(f"Ended inference episode: {outcome} (duration: {final_time:.2f}s)")