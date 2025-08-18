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

from .schemas import InferenceRequest, InferenceResponse
from .episode_logger import EpisodeLogger, Event

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
                 output_dir: str = "inference_episodes",
                 coord_frame: str = "ENU_RH",
                 dt_nominal: float = 0.01,
                 enable_logging: bool = True,
                 policy_id: Optional[str] = None):
        """
        Initialize inference episode logger.
        
        Args:
            output_dir: Base directory for output files
            coord_frame: Coordinate frame
            dt_nominal: Nominal timestep
            enable_logging: Whether logging is enabled
            policy_id: Policy identifier for this session
        """
        super().__init__(output_dir, coord_frame, dt_nominal, enable_logging)
        
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
        })\n        \n        # Write header record with full versioning\n        header = {\n            \"t\": 0.0,\n            \"meta\": {\n                \"ep_id\": episode_id,\n                \"seed\": seed,\n                \"coord_frame\": self.coord_frame,\n                \"dt_nominal\": self.dt_nominal,\n                \"scenario\": scenario_name,\n                \"obs_version\": obs_version,\n                \"vecnorm_stats_id\": vecnorm_stats_id,\n                \"transform_version\": transform_version,\n                \"policy_id\": self.policy_id,\n                \"session_id\": self.session_id\n            },\n            \"scene\": {}\n        }\n        \n        # Add interceptor configuration\n        if interceptor_config:\n            header[\"scene\"][\"interceptor_0\"] = {\n                \"mass_kg\": interceptor_config.get(\"mass\", 10.0),\n                \"max_torque\": interceptor_config.get(\"max_torque\", [8000, 8000, 2000]),\n                \"sensor_fov_deg\": interceptor_config.get(\"sensor_fov\", 30.0),\n                \"max_thrust_n\": interceptor_config.get(\"max_thrust\", 1000.0)\n            }\n        \n        # Add threat configuration\n        if threat_config:\n            header[\"scene\"][\"threat_0\"] = {\n                \"type\": threat_config.get(\"type\", \"ballistic\"),\n                \"mass_kg\": threat_config.get(\"mass\", 100.0),\n                \"aim_point\": threat_config.get(\"aim_point\", [0, 0, 0])\n            }\n        \n        self._write_record(header)\n        \n        logger.info(f\"Started inference episode: {episode_id} (v{obs_version}, {vecnorm_stats_id})\")\n    \n    def log_inference_step(self,\n                          t: float,\n                          request: InferenceRequest,\n                          response: InferenceResponse,\n                          latency_ms: float,\n                          interceptor_state: Optional[Dict[str, Any]] = None,\n                          threat_state: Optional[Dict[str, Any]] = None,\n                          events: Optional[List[Event]] = None):\n        \"\"\"\n        Log a single inference timestep with full versioning.\n        \n        Args:\n            t: Time in seconds from episode start\n            request: Inference request\n            response: Inference response\n            latency_ms: Inference latency in milliseconds\n            interceptor_state: Optional interceptor state\n            threat_state: Optional threat state\n            events: Optional events\n        \"\"\"\n        if not self.enable_logging or self.current_episode_file is None:\n            return\n        \n        # Convert states to agent format if provided\n        agents = {}\n        \n        if interceptor_state:\n            agents[\"interceptor_0\"] = self._process_agent_state(interceptor_state)\n        else:\n            # Extract from request if no explicit state provided\n            agents[\"interceptor_0\"] = {\n                \"p\": request.blue.pos_m,\n                \"q\": request.blue.quat_wxyz,\n                \"v\": request.blue.vel_mps,\n                \"w\": request.blue.ang_vel_radps,\n                \"fuel_frac\": request.blue.fuel_frac,\n                \"status\": \"active\"\n            }\n        \n        if threat_state:\n            agents[\"threat_0\"] = self._process_agent_state(threat_state)\n        else:\n            # Extract from request if no explicit state provided\n            agents[\"threat_0\"] = {\n                \"p\": request.red.pos_m,\n                \"q\": request.red.quat_wxyz,\n                \"v\": request.red.vel_mps,\n                \"status\": \"active\"\n            }\n        \n        # Add action command to interceptor state\n        if \"interceptor_0\" in agents:\n            agents[\"interceptor_0\"][\"u\"] = [\n                response.action.rate_cmd_radps.pitch,\n                response.action.rate_cmd_radps.yaw,\n                response.action.rate_cmd_radps.roll,\n                response.action.thrust_cmd\n            ] + response.action.aux\n        \n        # Build record with full versioning\n        record = {\n            \"t\": round(t, 3),\n            \"agents\": agents,\n            \"inference\": {\n                \"latency_ms\": round(latency_ms, 2),\n                \"obs_version\": request.normalization.obs_version,\n                \"vecnorm_stats_id\": request.normalization.vecnorm_stats_id,\n                \"transform_version\": request.normalization.transform_version,\n                \"policy_id\": self.policy_id,\n                \"safety_clamped\": response.safety.clamped,\n                \"clamp_reason\": response.safety.clamp_reason,\n                \"value_estimate\": response.diagnostics.value_estimate,\n                \"obs_clip_fractions\": {\n                    \"low\": response.diagnostics.obs_clip_fractions.low,\n                    \"high\": response.diagnostics.obs_clip_fractions.high\n                }\n            },\n            \"guidance\": {\n                \"los_unit\": request.guidance.los_unit,\n                \"los_rate_radps\": request.guidance.los_rate_radps,\n                \"range_m\": request.guidance.range_m,\n                \"closing_speed_mps\": request.guidance.closing_speed_mps,\n                \"fov_ok\": request.guidance.fov_ok,\n                \"g_limit_ok\": request.guidance.g_limit_ok\n            }\n        }\n        \n        # Add events if any\n        if events:\n            record[\"events\"] = [asdict(e) if isinstance(e, Event) else e for e in events]\n        \n        self._write_record(record)\n        \n        # Also log the inference entry for statistics\n        self._log_inference_entry(request, response, latency_ms)\n    \n    def end_inference_episode(self,\n                             outcome: str,\n                             final_time: float,\n                             miss_distance: Optional[float] = None,\n                             impact_time: Optional[float] = None,\n                             total_steps: Optional[int] = None,\n                             notes: Optional[str] = None):\n        \"\"\"\n        End the current inference episode with full summary.\n        \n        Args:\n            outcome: \"hit\" | \"miss\" | \"timeout\"\n            final_time: Final simulation time\n            miss_distance: Miss distance in meters\n            impact_time: Time of impact/closest approach\n            total_steps: Total number of steps\n            notes: Optional notes\n        \"\"\"\n        if not self.enable_logging or self.current_episode_file is None:\n            return\n        \n        # Write footer/summary record with full versioning\n        summary = {\n            \"t\": round(final_time, 3),\n            \"summary\": {\n                \"outcome\": outcome,\n                \"episode_duration\": final_time,\n                \"notes\": notes or \"\",\n                \"obs_version\": self.current_obs_version,\n                \"vecnorm_stats_id\": self.current_vecnorm_stats_id,\n                \"transform_version\": self.current_transform_version,\n                \"policy_id\": self.policy_id,\n                \"session_id\": self.session_id\n            }\n        }\n        \n        if miss_distance is not None:\n            summary[\"summary\"][\"miss_distance_m\"] = round(miss_distance, 3)\n        \n        if impact_time is not None:\n            summary[\"summary\"][\"impact_time_s\"] = round(impact_time, 3)\n        \n        if total_steps is not None:\n            summary[\"summary\"][\"steps\"] = total_steps\n        \n        # Add random seed if available\n        if \"seed\" in self.episode_metadata:\n            summary[\"summary\"][\"seed\"] = self.episode_metadata[\"seed\"]\n        \n        self._write_record(summary)\n        \n        # Close episode\n        self.current_episode_file = None\n        self.current_episode_data = []\n        \n        # Update manifest\n        self._save_manifest()\n        \n        logger.info(f\"Ended inference episode: {outcome} (duration: {final_time:.2f}s)\")\n    \n    def _log_inference_entry(self, \n                           request: InferenceRequest,\n                           response: InferenceResponse,\n                           latency_ms: float):\n        \"\"\"\n        Log individual inference entry for statistics.\n        \n        Args:\n            request: Inference request\n            response: Inference response\n            latency_ms: Inference latency\n        \"\"\"\n        entry = InferenceLogEntry(\n            timestamp=time.time(),\n            request=request.dict(),\n            response=response.dict(),\n            latency_ms=latency_ms,\n            safety_clamped=response.safety.clamped,\n            obs_version=request.normalization.obs_version,\n            vecnorm_stats_id=request.normalization.vecnorm_stats_id,\n            transform_version=request.normalization.transform_version,\n            policy_id=self.policy_id\n        )\n        \n        self.inference_log_entries.append(entry)\n        \n        # Trim if too many entries\n        if len(self.inference_log_entries) > self.max_inference_entries:\n            self.inference_log_entries = self.inference_log_entries[-self.max_inference_entries:]\n    \n    def get_inference_statistics(self) -> Dict[str, Any]:\n        \"\"\"\n        Get comprehensive inference statistics.\n        \n        Returns:\n            Dictionary with inference statistics\n        \"\"\"\n        if not self.inference_log_entries:\n            return {}\n        \n        # Overall statistics\n        total_requests = len(self.inference_log_entries)\n        latencies = [entry.latency_ms for entry in self.inference_log_entries]\n        safety_clamps = sum(1 for entry in self.inference_log_entries if entry.safety_clamped)\n        \n        # Recent statistics (last 100)\n        recent_entries = self.inference_log_entries[-100:]\n        recent_latencies = [entry.latency_ms for entry in recent_entries] if recent_entries else []\n        recent_clamps = sum(1 for entry in recent_entries if entry.safety_clamped)\n        \n        import numpy as np\n        \n        stats = {\n            \"total_requests\": total_requests,\n            \"safety_clamps_total\": safety_clamps,\n            \"safety_clamp_rate\": safety_clamps / max(total_requests, 1),\n            \"latency_stats\": {\n                \"mean_ms\": float(np.mean(latencies)) if latencies else 0.0,\n                \"p50_ms\": float(np.percentile(latencies, 50)) if latencies else 0.0,\n                \"p95_ms\": float(np.percentile(latencies, 95)) if latencies else 0.0,\n                \"p99_ms\": float(np.percentile(latencies, 99)) if latencies else 0.0,\n                \"min_ms\": float(np.min(latencies)) if latencies else 0.0,\n                \"max_ms\": float(np.max(latencies)) if latencies else 0.0\n            },\n            \"recent_stats\": {\n                \"requests\": len(recent_entries),\n                \"safety_clamps\": recent_clamps,\n                \"clamp_rate\": recent_clamps / max(len(recent_entries), 1),\n                \"mean_latency_ms\": float(np.mean(recent_latencies)) if recent_latencies else 0.0\n            },\n            \"versioning\": {\n                \"current_obs_version\": self.current_obs_version,\n                \"current_vecnorm_stats_id\": self.current_vecnorm_stats_id,\n                \"current_transform_version\": self.current_transform_version,\n                \"policy_id\": self.policy_id,\n                \"session_id\": self.session_id\n            }\n        }\n        \n        return stats\n    \n    def export_inference_log(self, output_path: str):\n        \"\"\"\n        Export inference log entries to JSONL file.\n        \n        Args:\n            output_path: Output file path\n        \"\"\"\n        output_file = Path(output_path)\n        output_file.parent.mkdir(parents=True, exist_ok=True)\n        \n        with open(output_file, 'w') as f:\n            for entry in self.inference_log_entries:\n                f.write(json.dumps(asdict(entry)) + '\\n')\n        \n        logger.info(f\"Exported {len(self.inference_log_entries)} inference entries to {output_path}\")\n\n\n# Global inference logger instance\n_inference_logger = None\n\n\ndef get_inference_logger(output_dir: str = \"inference_episodes\",\n                        policy_id: Optional[str] = None) -> InferenceEpisodeLogger:\n    \"\"\"Get global inference episode logger instance.\"\"\"\n    global _inference_logger\n    \n    if _inference_logger is None:\n        _inference_logger = InferenceEpisodeLogger(\n            output_dir=output_dir,\n            policy_id=policy_id\n        )\n    \n    return _inference_logger\n\n\ndef reset_inference_logger():\n    \"\"\"Reset global inference logger.\"\"\"\n    global _inference_logger\n    _inference_logger = None\n\n\n# Example usage\nif __name__ == \"__main__\":\n    from .schemas import *\n    \n    # Test inference episode logger\n    logger = InferenceEpisodeLogger()\n    \n    print(\"Inference Episode Logger Test\")\n    print(\"=\" * 40)\n    \n    # Start episode\n    logger.begin_inference_episode(\n        episode_id=\"test_ep_001\",\n        seed=12345,\n        scenario_name=\"test\",\n        obs_version=\"obs_v1.0\",\n        vecnorm_stats_id=\"test_vecnorm_001\",\n        transform_version=\"tfm_v1.0\"\n    )\n    \n    # Mock request/response\n    mock_request = InferenceRequest(\n        meta=MetaInfo(episode_id=\"test_ep_001\", t=1.0, dt=0.01, sim_tick=100),\n        frames=FrameInfo(frame=\"ENU\", unity_lh=True),\n        blue=BlueState(\n            pos_m=[100.0, 200.0, 50.0],\n            vel_mps=[150.0, 10.0, -5.0],\n            quat_wxyz=[0.995, 0.0, 0.1, 0.0],\n            ang_vel_radps=[0.1, 0.2, 0.05],\n            fuel_frac=0.75\n        ),\n        red=RedState(\n            pos_m=[500.0, 600.0, 100.0],\n            vel_mps=[-50.0, -40.0, -10.0],\n            quat_wxyz=[0.924, 0.0, 0.0, 0.383]\n        ),\n        guidance=GuidanceInfo(\n            los_unit=[0.8, 0.6, 0.0],\n            los_rate_radps=[0.01, -0.02, 0.0],\n            range_m=500.0,\n            closing_speed_mps=200.0,\n            fov_ok=True,\n            g_limit_ok=True\n        ),\n        env=EnvironmentInfo(\n            episode_step=100,\n            max_steps=1000\n        ),\n        normalization=NormalizationInfo(\n            obs_version=\"obs_v1.0\",\n            vecnorm_stats_id=\"test_vecnorm_001\",\n            transform_version=\"tfm_v1.0\"\n        )\n    )\n    \n    mock_response = InferenceResponse(\n        action=ActionCommand(\n            rate_cmd_radps=RateCommand(pitch=0.5, yaw=-0.2, roll=0.1),\n            thrust_cmd=0.8,\n            aux=[0.0, 0.0]\n        ),\n        diagnostics=DiagnosticsInfo(\n            policy_latency_ms=15.3,\n            obs_clip_fractions=ClipFractions(low=0.02, high=0.01),\n            value_estimate=0.75\n        ),\n        safety=SafetyInfo(clamped=False, clamp_reason=None)\n    )\n    \n    # Log step\n    logger.log_inference_step(\n        t=1.0,\n        request=mock_request,\n        response=mock_response,\n        latency_ms=15.3\n    )\n    \n    # End episode\n    logger.end_inference_episode(\n        outcome=\"hit\",\n        final_time=10.5,\n        miss_distance=2.1,\n        impact_time=10.2,\n        total_steps=105\n    )\n    \n    # Show statistics\n    stats = logger.get_inference_statistics()\n    print(f\"Statistics: {stats}\")\n    \n    print(\"Test completed successfully!\")