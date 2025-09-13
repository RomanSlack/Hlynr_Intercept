#!/usr/bin/env python3
"""
Unity bridge server for Phase 4 RL inference.

This server provides a REST API for Unity to communicate with trained RL models,
allowing real-time inference through HTTP requests.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from .config import get_config, reset_config
    from .scenarios import get_scenario_loader, reset_scenario_loader
    from .radar_env import RadarEnv
    from .fast_sim_env import FastSimEnv
except ImportError:
    # Fallback for direct execution
    from config import get_config, reset_config
    from scenarios import get_scenario_loader, reset_scenario_loader
    from radar_env import RadarEnv
    from fast_sim_env import FastSimEnv


# Global variables for the loaded model and environment
app = Flask(__name__)
CORS(app)  # Enable CORS for Unity integration
model = None
env = None
vec_normalize = None
server_stats = {
    'requests_served': 0,
    'total_inference_time': 0.0,
    'average_inference_time': 0.0,
    'model_loaded_at': None,
    'checkpoint_path': None
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BridgeServer:
    """Unity bridge server for RL model inference."""
    
    def __init__(self, checkpoint_path: str, 
                 scenario_name: str = "easy",
                 config_path: Optional[str] = None,
                 host: str = "0.0.0.0",
                 port: int = 5000):
        """
        Initialize bridge server.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            scenario_name: Scenario name for environment configuration
            config_path: Path to configuration file
            host: Server host address
            port: Server port
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.scenario_name = scenario_name
        self.config_path = config_path
        self.host = host
        self.port = port
        
        # Reset global instances
        reset_config()
        reset_scenario_loader()
        
        # Load configuration
        self.config_loader = get_config(config_path)
        self.config = self.config_loader._config
        
        # Load scenario
        self.scenario_loader = get_scenario_loader()
        self.scenario_config = self.scenario_loader.create_environment_config(
            scenario_name, self.config
        )
        
    def load_model(self):
        """Load the trained model and setup environment."""
        global model, env, vec_normalize, server_stats
        
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
        
        # Load normalization if available
        vec_normalize_path = self.checkpoint_path.parent / "vec_normalize.pkl"
        if vec_normalize_path.exists():
            logger.info(f"Loading normalization parameters from: {vec_normalize_path}")
            vec_normalize = VecNormalize.load(vec_normalize_path, env)
            vec_normalize.training = False  # Set to evaluation mode
            vec_normalize.norm_reward = False  # Don't normalize rewards during inference
            env = vec_normalize
        
        # Load model
        model = PPO.load(self.checkpoint_path, env=env)
        
        # Update server stats
        server_stats['model_loaded_at'] = time.time()
        server_stats['checkpoint_path'] = str(self.checkpoint_path)
        
        logger.info("Model loaded successfully!")
        logger.info(f"Environment: {self.scenario_name} scenario")
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
    def run(self):
        """Start the bridge server."""
        # Load model before starting server
        self.load_model()
        
        logger.info(f"Starting Unity Bridge Server on {self.host}:{self.port}")
        logger.info(f"Endpoints:")
        logger.info(f"  POST /act - Get action for observation")
        logger.info(f"  GET /health - Server health check")
        logger.info(f"  GET /stats - Server statistics")
        
        app.run(host=self.host, port=self.port, debug=False)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global model
    
    return jsonify({
        'status': 'healthy' if model is not None else 'not_ready',
        'model_loaded': model is not None,
        'timestamp': time.time()
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get server statistics."""
    global server_stats
    
    return jsonify(server_stats)


@app.route('/act', methods=['POST'])
def get_action():
    """
    Get action for given observation.
    
    Expected JSON format:
    {
        "observation": [float, ...],  # Observation array
        "deterministic": true,        # Optional: use deterministic policy
        "scenario": "easy"            # Optional: scenario name
    }
    
    Returns:
    {
        "action": [float, ...],       # Action array
        "inference_time": float,      # Inference time in seconds
        "success": true,              # Whether inference succeeded
        "error": null                 # Error message if any
    }
    """
    global model, env, server_stats
    
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded',
                'action': None,
                'inference_time': 0.0
            }), 500
        
        # Parse request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON',
                'action': None,
                'inference_time': 0.0
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'observation' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing observation field',
                'action': None,
                'inference_time': 0.0
            }), 400
        
        # Extract parameters
        observation = np.array(data['observation'], dtype=np.float32)
        deterministic = data.get('deterministic', True)
        
        # Validate observation shape
        expected_shape = env.observation_space.shape
        if observation.shape != expected_shape:
            return jsonify({
                'success': False,
                'error': f'Observation shape {observation.shape} does not match expected {expected_shape}',
                'action': None,
                'inference_time': 0.0
            }), 400
        
        # Reshape observation for vectorized environment
        obs = observation.reshape(1, -1)
        
        # Get action from model
        action, _states = model.predict(obs, deterministic=deterministic)
        
        # Extract action from vectorized format
        action = action[0] if len(action.shape) > 1 else action
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Update server stats
        server_stats['requests_served'] += 1
        server_stats['total_inference_time'] += inference_time
        server_stats['average_inference_time'] = (
            server_stats['total_inference_time'] / server_stats['requests_served']
        )
        
        # Return response
        response = {
            'success': True,
            'action': action.tolist(),
            'inference_time': inference_time,
            'error': None
        }
        
        logger.debug(f"Inference completed in {inference_time:.4f}s")
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Inference error: {str(e)}"
        logger.error(error_msg)
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'action': None,
            'inference_time': time.time() - start_time
        }), 500


@app.route('/reset', methods=['POST'])
def reset_environment():
    """
    Reset environment (optional endpoint for debugging).
    
    Returns:
    {
        "observation": [float, ...],  # Initial observation
        "success": true,              # Whether reset succeeded
        "error": null                 # Error message if any
    }
    """
    global env
    
    try:
        if env is None:
            return jsonify({
                'success': False,
                'error': 'Environment not initialized',
                'observation': None
            }), 500
        
        # Reset environment
        obs = env.reset()
        
        # Extract observation from vectorized format
        observation = obs[0] if len(obs.shape) > 1 else obs
        
        return jsonify({
            'success': True,
            'observation': observation.tolist(),
            'error': None
        })
        
    except Exception as e:
        error_msg = f"Reset error: {str(e)}"
        logger.error(error_msg)
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'observation': None
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/act', '/health', '/stats', '/reset']
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(error)
    }), 500


def main():
    """Main entry point for bridge server."""
    parser = argparse.ArgumentParser(description='Unity Bridge Server for Phase 4 RL')
    
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
        server = BridgeServer(
            checkpoint_path=args.checkpoint,
            scenario_name=args.scenario,
            config_path=args.config,
            host=args.host,
            port=args.port
        )
        
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()