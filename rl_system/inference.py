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
from logger import make_json_serializable

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

from environment import InterceptEnvironment
from core import Radar26DObservation, CoordinateTransform, SafetyClamp
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
        radar_config = self.config.get('radar', self.config['environment'])
        ground_radar_config = self.config['environment'].get('ground_radar', {})
        self.observation_generator = Radar26DObservation(
            max_range=self.config['environment'].get('max_range', 10000.0),
            max_velocity=self.config['environment'].get('max_velocity', 1000.0),
            radar_range=radar_config.get('radar_range', 5000.0),
            min_detection_range=radar_config.get('min_detection_range', 50.0),
            ground_radar_config=ground_radar_config
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
                
                # Generate observation using radar detection simulation (with ground radar)
                obs = self.observation_generator.compute_radar_detection(
                    interceptor_state,
                    missile_state,
                    request.observation.radar_quality,
                    request.observation.radar_noise,
                    weather_factor=1.0  # Could be made configurable
                )
                
                # Normalize observation if VecNormalize is available
                if self.vec_normalize:
                    obs = self.vec_normalize.normalize_obs(obs.reshape(1, -1))[0]
                else:
                    # Fallback: Simple clipping to prevent extreme values
                    # Observation space is defined as [-1, 1], ensure compliance
                    obs = np.clip(obs, -1.0, 1.0)

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


def run_offline_inference(model_path: str, config_path: str, num_episodes: int = 100,
                         output_dir: str = "inference_results", scenario: Optional[str] = None,
                         volley_mode: bool = False, volley_size: int = 1):
    """Run offline inference and save results to JSON files."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"offline_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = UnifiedLogger(log_dir=str(run_dir), run_name="offline_inference")
    
    # Load model
    mp = Path(model_path)
    candidates = [
        mp if mp.suffix == ".zip" else None,
        mp / "model.zip",
        mp / "best_model.zip",
        mp / "best" / "best_model.zip",
        mp / "final" / "model.zip",
    ]
    candidates = [p for p in candidates if p and p.exists()]
    if not candidates:
        raise FileNotFoundError(f"Could not find a model zip under {mp}. "
                                f"Tried: model.zip, best_model.zip, best/best_model.zip, final/model.zip, or a direct .zip path.")
    # Load scenario if specified first to get correct config
    env_config = config.get('environment', {})
    if scenario:
        scenario_path = Path("scenarios") / f"{scenario}.yaml"
        if scenario_path.exists():
            with open(scenario_path, 'r') as f:
                scenario_config = yaml.safe_load(f)
                env_config.update(scenario_config.get('environment', {}))

    # Add volley mode to environment config if enabled
    if volley_mode:
        env_config['volley_mode'] = True
        env_config['volley_size'] = volley_size
        logger.logger.info(f"Volley mode enabled: {volley_size} missiles per episode")

    # Create environment with same preprocessing as training
    env = InterceptEnvironment(env_config)
    base_obs_dim = env.observation_space.shape[0]

    # Apply frame-stacking if configured (must match training)
    frame_stack = config['training'].get('frame_stack', 1)
    if frame_stack > 1:
        logger.logger.info(f"Applying frame-stacking: {frame_stack} frames ({base_obs_dim}D → {base_obs_dim * frame_stack}D)")
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=frame_stack)
        current_obs_dim = base_obs_dim * frame_stack
    else:
        current_obs_dim = base_obs_dim

    # Load model and check compatibility
    logger.logger.info(f"Loading model from {candidates[0]}")
    model = PPO.load(str(candidates[0]))
    model_obs_dim = model.observation_space.shape[0]

    if model_obs_dim != current_obs_dim:
        logger.logger.error(
            f"❌ Model/Environment dimension mismatch: "
            f"Model expects {model_obs_dim}D observations, environment provides {current_obs_dim}D"
        )
        if model_obs_dim == 17 and current_obs_dim == 26:
            logger.logger.error(
                f"❌ This model was trained BEFORE ground radar was added (17D observations)."
            )
            logger.logger.error(
                f"❌ You need to retrain with the new 26D observation space."
            )
            logger.logger.error(
                f"❌ Or temporarily disable ground radar in config.yaml to use old models:"
            )
            logger.logger.error(
                f"   ground_radar:"
            )
            logger.logger.error(
                f"     enabled: false"
            )
        raise ValueError(
            f"Model expects {model_obs_dim}D observations but environment provides {current_obs_dim}D. "
            f"Retrain model with current observation space or adjust config."
        )

    # Load VecNormalize if available (must apply AFTER frame-stacking)
    # Try multiple locations
    model_dir = Path(model_path).parent if Path(model_path).suffix == '.zip' else Path(model_path)
    vecnorm_candidates = [
        model_dir / "vec_normalize.pkl",  # Same dir as model
        model_dir.parent / "final" / "vec_normalize.pkl",  # Training final dir
        model_dir.parent / "vec_normalize.pkl",  # Parent training dir
    ]

    vecnorm_path = None
    for candidate in vecnorm_candidates:
        if candidate.exists():
            vecnorm_path = candidate
            break

    if vecnorm_path:
        try:
            logger.logger.info(f"Loading VecNormalize from {vecnorm_path}")
            env = VecNormalize.load(str(vecnorm_path), env)
            env.training = False  # Freeze normalization stats for inference
            env.norm_reward = False  # Don't normalize rewards during inference
            logger.logger.info(f"✓ Loaded VecNormalize (training=False)")
        except Exception as e:
            logger.logger.warning(f"⚠️  Failed to load VecNormalize: {e}")
            logger.logger.warning(f"⚠️  Continuing without normalization")
    else:
        logger.logger.warning(f"⚠️  VecNormalize not found in: {[str(c) for c in vecnorm_candidates]}")
        logger.logger.warning(f"⚠️  Model may have been trained with VecNormalize - results may be inaccurate!")
    
    # Run episodes
    results = []
    volley_stats = {
        'total_missiles_spawned': 0,
        'total_missiles_intercepted': 0,
        'episodes_all_intercepted': 0,
        'episodes_partial_interception': 0,
        'episodes_no_interception': 0
    } if volley_mode else None

    for ep in range(num_episodes):
        episode_data = {
            'episode_id': f"ep_{ep:04d}",
            'states': [],
            'actions': [],
            'rewards': [],
            'info': []
        }

        # Reset environment (VecEnv returns array)
        # Volley mode is already configured in environment config
        obs = env.reset()
        if isinstance(obs, tuple):  # Handle (obs, info) tuple from newer gym
            obs = obs[0]
        logger.begin_episode(episode_data['episode_id'])

        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Get action (obs already normalized by VecNormalize wrapper)
            action, _ = model.predict(obs, deterministic=True)
            
            # Store data (extract from VecEnv array format)
            obs_array = obs[0] if len(obs.shape) > 1 else obs
            episode_data['states'].append(obs_array.tolist())
            episode_data['actions'].append(action[0].tolist() if len(action.shape) > 1 else action.tolist())

            # Step environment (VecEnv returns arrays)
            obs, reward, done_array, info_array = env.step(action)

            # Extract from VecEnv arrays
            reward_val = reward[0] if isinstance(reward, np.ndarray) else reward
            done = done_array[0] if isinstance(done_array, np.ndarray) else done_array
            info = info_array[0] if isinstance(info_array, list) else info_array

            episode_data['rewards'].append(float(reward_val))
            episode_data['info'].append(info)

            total_reward += reward_val
            steps += 1

            # Log state with actions
            if 'interceptor_pos' in info:
                logger.log_state('interceptor', {
                    'position': info['interceptor_pos'].tolist(),
                    'fuel': info.get('fuel_remaining', 0),
                    'action': action[0].tolist() if len(action.shape) > 1 else action.tolist()
                })
            if 'missile_pos' in info:
                logger.log_state('missile', {
                    'position': info['missile_pos'].tolist()
                })

        # Episode complete
        # Determine outcome based on mode
        if volley_mode:
            missiles_intercepted = info.get('missiles_intercepted', 0)
            volley_sz = info.get('volley_size', volley_size)
            missile_min_distances = info.get('missile_min_distances', [])

            # Update volley statistics
            volley_stats['total_missiles_spawned'] += volley_sz
            volley_stats['total_missiles_intercepted'] += missiles_intercepted

            if missiles_intercepted == volley_sz:
                outcome = "all_intercepted"
                volley_stats['episodes_all_intercepted'] += 1
            elif missiles_intercepted > 0:
                outcome = "partial_interception"
                volley_stats['episodes_partial_interception'] += 1
            else:
                outcome = "failed"
                volley_stats['episodes_no_interception'] += 1

            # Add volley-specific episode data
            episode_data['volley_mode'] = True
            episode_data['volley_size'] = volley_sz
            episode_data['missiles_intercepted'] = missiles_intercepted
            episode_data['missiles_remaining'] = info.get('missiles_remaining', 0)
            episode_data['missile_min_distances'] = missile_min_distances
        else:
            outcome = "intercepted" if info.get('intercepted', False) else "failed"

        episode_data['outcome'] = outcome
        episode_data['total_reward'] = float(total_reward)
        episode_data['steps'] = int(steps)
        episode_data['final_distance'] = float(info.get('distance', 0))

        results.append(episode_data)

        logger.end_episode(outcome, {
            'total_reward': total_reward,
            'steps': steps,
            'final_distance': info['distance'],
            'fuel_used': info.get('fuel_used', 0),
            'volley_mode': volley_mode,
            'missiles_intercepted': info.get('missiles_intercepted', 0) if volley_mode else None,
            'volley_size': info.get('volley_size', 1) if volley_mode else None
        })

        if volley_mode:
            print(f"Episode {ep+1}/{num_episodes}: {outcome}, missiles={missiles_intercepted}/{volley_sz}, reward={total_reward:.2f}, steps={steps}")
        else:
            print(f"Episode {ep+1}/{num_episodes}: {outcome}, reward={total_reward:.2f}, steps={steps}")
    
    # Save aggregated results
    summary = {
        'run_id': f"offline_{timestamp}",
        'model_path': str(model_path),
        'num_episodes': num_episodes,
        'scenario': scenario,
        'volley_mode': volley_mode,
        'avg_reward': np.mean([r['total_reward'] for r in results]),
        'avg_steps': np.mean([r['steps'] for r in results]),
        'avg_final_distance': np.mean([r['final_distance'] for r in results]),
        'episodes': results
    }

    # Add mode-specific metrics
    if volley_mode:
        summary['volley_size'] = volley_size
        summary['volley_statistics'] = volley_stats
        summary['success_rate_all_intercepted'] = volley_stats['episodes_all_intercepted'] / num_episodes
        summary['success_rate_partial'] = volley_stats['episodes_partial_interception'] / num_episodes
        summary['failure_rate'] = volley_stats['episodes_no_interception'] / num_episodes
        summary['avg_missiles_intercepted_per_episode'] = volley_stats['total_missiles_intercepted'] / num_episodes
        summary['overall_interception_rate'] = volley_stats['total_missiles_intercepted'] / volley_stats['total_missiles_spawned'] if volley_stats['total_missiles_spawned'] > 0 else 0.0
    else:
        summary['success_rate'] = sum(1 for r in results if r['outcome'] == 'intercepted') / num_episodes
    
    # Write summary JSON
    summary_file = run_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(make_json_serializable(summary), f, indent=2)
    
    # Write detailed episodes JSONL
    episodes_file = run_dir / "episodes.jsonl"
    with open(episodes_file, 'w') as f:
        for episode in results:
            f.write(json.dumps(make_json_serializable(episode)) + '\n')
    
    logger.create_manifest()
    print(f"\nResults saved to: {run_dir}")

    if volley_mode:
        print(f"\n=== Volley Mode Results (Volley Size: {volley_size}) ===")
        print(f"All missiles intercepted: {summary['success_rate_all_intercepted']:.2%} ({volley_stats['episodes_all_intercepted']}/{num_episodes} episodes)")
        print(f"Partial interception: {summary['success_rate_partial']:.2%} ({volley_stats['episodes_partial_interception']}/{num_episodes} episodes)")
        print(f"No interception: {summary['failure_rate']:.2%} ({volley_stats['episodes_no_interception']}/{num_episodes} episodes)")
        print(f"Overall interception rate: {summary['overall_interception_rate']:.2%} ({volley_stats['total_missiles_intercepted']}/{volley_stats['total_missiles_spawned']} missiles)")
        print(f"Average missiles intercepted per episode: {summary['avg_missiles_intercepted_per_episode']:.2f}")
    else:
        print(f"Success rate: {summary['success_rate']:.2%}")

    print(f"Average reward: {summary['avg_reward']:.2f}")

    return summary


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
        "--mode",
        type=str,
        choices=["server", "offline"],
        default="server",
        help="Run mode: server (API) or offline (batch inference)"
    )
    
    # Server mode arguments
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (server mode)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (server mode)"
    )
    
    # Offline mode arguments
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run (offline mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results",
        help="Output directory for results (offline mode)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["easy", "medium", "hard"],
        help="Scenario to use (offline mode)"
    )

    # Volley mode arguments
    parser.add_argument(
        "--volley",
        action="store_true",
        help="Enable volley mode (multiple missiles per episode)"
    )
    parser.add_argument(
        "--volley-size",
        type=int,
        default=3,
        help="Number of missiles in each volley (default: 3)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "server":
        server = InferenceServer(args.model, args.config)
        server.run(args.host, args.port)
    else:
        # Offline mode
        run_offline_inference(
            model_path=args.model,
            config_path=args.config,
            num_episodes=args.episodes,
            output_dir=args.output,
            scenario=args.scenario,
            volley_mode=args.volley,
            volley_size=args.volley_size
        )


if __name__ == "__main__":
    main()