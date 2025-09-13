"""
FastAPI inference server with deterministic execution and comprehensive logging.
"""

import os
import yaml
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import InterceptEnvironment
from core import Radar17DObservation, CoordinateTransform, SafetyClamp
from logger import UnifiedLogger, EpisodeEvent


# Pydantic models for API
class ObservationData(BaseModel):
    """Observation data from client."""
    interceptor: Dict[str, List[float]] = Field(..., description="Interceptor state")
    missile: Dict[str, List[float]] = Field(..., description="Missile state")
    radar_quality: float = Field(1.0, ge=0.0, le=1.0)
    radar_noise: float = Field(0.05, ge=0.0, le=1.0)


class InferenceRequest(BaseModel):
    """Inference request schema."""
    observation: ObservationData
    coordinate_system: str = Field("ENU", description="Coordinate system: ENU or UNITY")
    episode_id: Optional[str] = None
    seed: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ActionCommand(BaseModel):
    """Action command output."""
    thrust: List[float] = Field(..., description="Thrust vector [x,y,z]")
    angular: List[float] = Field(..., description="Angular rates [roll,pitch,yaw]")
    
    
class SafetyInfo(BaseModel):
    """Safety clamping information."""
    clamped: bool
    reasons: List[str]
    original_action: Optional[List[float]] = None
    clamped_action: Optional[List[float]] = None


class InferenceResponse(BaseModel):
    """Inference response schema."""
    action: ActionCommand
    raw_action: List[float]
    safety: SafetyInfo
    processing_time_ms: float
    episode_id: Optional[str] = None
    timestamp: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str = "1.0.0"
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Metrics response."""
    total_requests: int
    avg_latency_ms: float
    success_rate: float
    active_episodes: int


class InferenceServer:
    """Main inference server class."""
    
    def __init__(self, model_path: str, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize logger
        self.logger = UnifiedLogger(
            log_dir=self.config.get('logging', {}).get('log_dir', 'logs'),
            run_name="inference"
        )
        
        # Load model
        self.model_path = Path(model_path)
        self.model = None
        self.vec_normalize = None
        self._load_model()
        
        # Initialize components
        self.observation_generator = Radar17DObservation(
            max_range=self.config['environment'].get('max_range', 10000.0),
            max_velocity=self.config['environment'].get('max_velocity', 1000.0)
        )
        self.coordinate_transform = CoordinateTransform()
        self.safety_clamp = SafetyClamp()
        
        # Metrics tracking
        self.start_time = time.time()
        self.total_requests = 0
        self.total_latency = 0.0
        self.successful_requests = 0
        self.active_episodes = {}
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Hlynr Inference API",
            version="1.0.0",
            description="Missile interception inference server"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
        
        self.logger.logger.info("Inference server initialized")
    
    def _load_model(self):
        """Load trained model and normalization stats."""
        try:
            # Load PPO model
            model_file = self.model_path / "model.zip"
            if not model_file.exists():
                # Try alternate paths
                model_file = self.model_path / "best" / "best_model.zip"
                if not model_file.exists():
                    model_file = self.model_path / "final" / "model.zip"
            
            self.model = PPO.load(str(model_file))
            self.logger.logger.info(f"Loaded model from {model_file}")
            
            # Load VecNormalize stats
            vecnorm_file = self.model_path / "vec_normalize.pkl"
            if not vecnorm_file.exists():
                vecnorm_file = self.model_path / "final" / "vec_normalize.pkl"
            
            if vecnorm_file.exists():
                # Create dummy env for VecNormalize
                env = DummyVecEnv([lambda: InterceptEnvironment(self.config.get('environment', {}))])
                self.vec_normalize = VecNormalize.load(str(vecnorm_file), env)
                self.vec_normalize.training = False
                self.logger.logger.info(f"Loaded VecNormalize from {vecnorm_file}")
            else:
                self.logger.logger.warning("VecNormalize stats not found")
                
        except Exception as e:
            self.logger.logger.error(f"Failed to load model: {e}")
            raise
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model_loaded=self.model is not None,
                uptime_seconds=time.time() - self.start_time
            )
        
        @self.app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics():
            """Get server metrics."""
            avg_latency = self.total_latency / max(self.total_requests, 1)
            success_rate = self.successful_requests / max(self.total_requests, 1)
            
            return MetricsResponse(
                total_requests=self.total_requests,
                avg_latency_ms=avg_latency,
                success_rate=success_rate,
                active_episodes=len(self.active_episodes)
            )
        
        @self.app.post("/infer", response_model=InferenceResponse)
        async def infer(request: InferenceRequest):
            """Main inference endpoint."""
            start_time = time.time()
            self.total_requests += 1
            
            try:
                # Set seed if provided
                if request.seed is not None:
                    np.random.seed(request.seed)
                    torch.manual_seed(request.seed)
                    self.observation_generator.seed(request.seed)
                
                # Handle coordinate transformation if needed
                interceptor_state = request.observation.interceptor
                missile_state = request.observation.missile
                
                if request.coordinate_system == "UNITY":
                    # Transform from Unity to ENU
                    int_pos = np.array(interceptor_state['position'])
                    int_quat = np.array(interceptor_state['orientation'])
                    int_pos, int_quat = self.coordinate_transform.unity_to_enu(int_pos, int_quat)
                    interceptor_state['position'] = int_pos.tolist()
                    interceptor_state['orientation'] = int_quat.tolist()
                    
                    mis_pos = np.array(missile_state['position'])
                    mis_quat = np.array(missile_state.get('orientation', [1, 0, 0, 0]))
                    mis_pos, mis_quat = self.coordinate_transform.unity_to_enu(mis_pos, mis_quat)
                    missile_state['position'] = mis_pos.tolist()
                    missile_state['orientation'] = mis_quat.tolist()
                
                # Generate observation
                obs = self.observation_generator.compute(
                    interceptor_state,
                    missile_state,
                    request.observation.radar_quality,
                    request.observation.radar_noise
                )
                
                # Normalize observation if VecNormalize is available
                if self.vec_normalize:
                    obs = self.vec_normalize.normalize_obs(obs.reshape(1, -1))[0]
                
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Apply safety constraints
                current_fuel = interceptor_state.get('fuel', 100.0)
                clamped_action, safety_info = self.safety_clamp.apply(action, current_fuel)
                
                # Split action into thrust and angular components
                thrust = clamped_action[0:3].tolist()
                angular = clamped_action[3:6].tolist()
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000
                
                # Log inference
                self.logger.log_inference(
                    request.dict(),
                    {'action': {'thrust': thrust, 'angular': angular}},
                    processing_time
                )
                
                # Track episode if provided
                if request.episode_id:
                    if request.episode_id not in self.active_episodes:
                        self.active_episodes[request.episode_id] = {
                            'start_time': time.time(),
                            'request_count': 0
                        }
                        self.logger.begin_episode(request.episode_id, request.metadata)
                    
                    self.active_episodes[request.episode_id]['request_count'] += 1
                    
                    # Log state
                    self.logger.log_state('interceptor', interceptor_state, time.time())
                    self.logger.log_state('missile', missile_state, time.time())
                
                self.successful_requests += 1
                self.total_latency += processing_time
                
                return InferenceResponse(
                    action=ActionCommand(thrust=thrust, angular=angular),
                    raw_action=action.tolist(),
                    safety=SafetyInfo(
                        clamped=safety_info['clamped'],
                        reasons=safety_info['reasons'],
                        original_action=action.tolist() if safety_info['clamped'] else None,
                        clamped_action=clamped_action.tolist() if safety_info['clamped'] else None
                    ),
                    processing_time_ms=processing_time,
                    episode_id=request.episode_id,
                    timestamp=time.time()
                )
                
            except Exception as e:
                self.logger.logger.error(f"Inference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/episode/end")
        async def end_episode(episode_id: str, outcome: str, metrics: Optional[Dict[str, Any]] = None):
            """End an active episode."""
            if episode_id in self.active_episodes:
                self.logger.end_episode(outcome, metrics or {})
                del self.active_episodes[episode_id]
                return {"status": "success", "episode_id": episode_id}
            else:
                raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
        
        @self.app.post("/reset")
        async def reset_server():
            """Reset server state."""
            self.active_episodes.clear()
            self.total_requests = 0
            self.total_latency = 0.0
            self.successful_requests = 0
            self.logger.create_manifest()
            return {"status": "reset complete"}
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server."""
        import uvicorn
        self.logger.logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Inference server for missile interception")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    
    args = parser.parse_args()
    
    server = InferenceServer(args.model, args.config)
    server.run(args.host, args.port)


if __name__ == "__main__":
    main()