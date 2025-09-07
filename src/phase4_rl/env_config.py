"""
Environment variable configuration for Hlynr Bridge Server.

This module provides environment variable loading and management for the
Unity↔RL Inference API Bridge Server, implementing the PRP specification
for configuration management.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

# Import centralized path resolver
try:
    from hlynr_bridge.paths import logs_server
except ImportError:
    # Fallback for development/testing
    def logs_server():
        return Path("inference_episodes")

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class BridgeServerConfig:
    """Environment configuration for Hlynr Bridge Server."""
    
    # Model and inference
    model_checkpoint: str
    vecnorm_stats_id: Optional[str] = None
    obs_version: str = "obs_v1.0"
    transform_version: str = "tfm_v1.0"
    policy_id: Optional[str] = None
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 5000
    cors_origins: List[str] = None
    
    # Logging and output
    log_dir: str = "inference_episodes"  # Deprecated - individual components use centralized paths
    log_level: str = "INFO"
    
    # Safety and performance
    rate_max_radps: float = 10.0
    latency_slo_p50_ms: float = 20.0
    latency_slo_p95_ms: float = 35.0
    
    # Environment and scenario
    scenario_name: str = "easy"
    config_path: Optional[str] = None
    
    # Advanced configuration
    enable_logging: bool = True
    enable_metrics: bool = True
    enable_cors: bool = True
    debug_mode: bool = False
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.cors_origins is None:
            self.cors_origins = ["*"]  # Allow all origins by default
        
        # Ensure log directory exists
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate required fields
        if not self.model_checkpoint:
            raise ValueError("MODEL_CHECKPOINT environment variable is required")
        
        if not Path(self.model_checkpoint).exists():
            logger.warning(f"Model checkpoint path does not exist: {self.model_checkpoint}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_env(cls) -> 'BridgeServerConfig':
        """
        Create configuration from environment variables.
        
        Environment variables (with fallback defaults):
        - MODEL_CHECKPOINT: Required, path to trained model
        - VECNORM_STATS_ID: Optional, VecNormalize statistics ID
        - OBS_VERSION: Observation version (default: obs_v1.0)
        - TRANSFORM_VERSION: Coordinate transform version (default: tfm_v1.0)
        - POLICY_ID: Optional policy identifier
        - HOST: Server host (default: 0.0.0.0)
        - PORT: Server port (default: 5000)
        - CORS_ORIGINS: Comma-separated CORS origins (default: *)
        - HLYNR_LOG_DIR: Root log directory (default: logs)
        - LOG_DIR: Legacy logging directory (deprecated, default: inference_episodes)
        - LOG_LEVEL: Logging level (default: INFO)
        - RATE_MAX_RADPS: Max angular rate for safety (default: 10.0)
        - SCENARIO_NAME: Scenario name (default: easy)
        - CONFIG_PATH: Optional path to YAML config
        - ENABLE_LOGGING: Enable episode logging (default: true)
        - ENABLE_METRICS: Enable metrics endpoint (default: true)
        - ENABLE_CORS: Enable CORS (default: true)
        - DEBUG_MODE: Enable debug mode (default: false)
        
        Returns:
            BridgeServerConfig instance
        """
        # Parse CORS origins
        cors_origins_str = os.getenv("CORS_ORIGINS", "*")
        cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]
        
        # Parse boolean environment variables
        def parse_bool(value: str, default: bool = False) -> bool:
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes", "on")
        
        return cls(
            model_checkpoint=os.getenv("MODEL_CHECKPOINT", ""),
            vecnorm_stats_id=os.getenv("VECNORM_STATS_ID"),
            obs_version=os.getenv("OBS_VERSION", "obs_v1.0"),
            transform_version=os.getenv("TRANSFORM_VERSION", "tfm_v1.0"),
            policy_id=os.getenv("POLICY_ID"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "5000")),
            cors_origins=cors_origins,
            log_dir=os.getenv("LOG_DIR", "inference_episodes"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            rate_max_radps=float(os.getenv("RATE_MAX_RADPS", "10.0")),
            latency_slo_p50_ms=float(os.getenv("LATENCY_SLO_P50_MS", "20.0")),
            latency_slo_p95_ms=float(os.getenv("LATENCY_SLO_P95_MS", "35.0")),
            scenario_name=os.getenv("SCENARIO_NAME", "easy"),
            config_path=os.getenv("CONFIG_PATH"),
            enable_logging=parse_bool(os.getenv("ENABLE_LOGGING"), True),
            enable_metrics=parse_bool(os.getenv("ENABLE_METRICS"), True),
            enable_cors=parse_bool(os.getenv("ENABLE_CORS"), True),
            debug_mode=parse_bool(os.getenv("DEBUG_MODE"), False)
        )
    
    def setup_logging(self):
        """Setup logging configuration based on environment settings."""
        numeric_level = getattr(logging, self.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {self.log_level}')
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(logs_server() / "bridge_server.log")
            ]
        )
        
        if self.debug_mode:
            logger.info("Debug mode enabled")
            logging.getLogger().setLevel(logging.DEBUG)
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List of validation messages
        """
        issues = []
        
        # Check required fields
        if not self.model_checkpoint:
            issues.append("ERROR: MODEL_CHECKPOINT is required")
        
        # Check file paths
        if self.model_checkpoint and not Path(self.model_checkpoint).exists():
            issues.append(f"WARNING: Model checkpoint not found: {self.model_checkpoint}")
        
        if self.config_path and not Path(self.config_path).exists():
            issues.append(f"WARNING: Config file not found: {self.config_path}")
        
        # Check numeric ranges
        if self.port < 1 or self.port > 65535:
            issues.append(f"ERROR: Invalid port number: {self.port}")
        
        if self.rate_max_radps <= 0:
            issues.append(f"ERROR: rate_max_radps must be positive: {self.rate_max_radps}")
        
        if self.latency_slo_p50_ms <= 0 or self.latency_slo_p95_ms <= 0:
            issues.append("ERROR: Latency SLO values must be positive")
        
        if self.latency_slo_p95_ms <= self.latency_slo_p50_ms:
            issues.append("WARNING: p95 latency SLO should be higher than p50")
        
        # Check version formats
        valid_obs_versions = ["obs_v1.0"]
        if self.obs_version not in valid_obs_versions:
            issues.append(f"WARNING: Unknown obs_version: {self.obs_version}")
        
        valid_transform_versions = ["tfm_v1.0"]
        if self.transform_version not in valid_transform_versions:
            issues.append(f"WARNING: Unknown transform_version: {self.transform_version}")
        
        return issues
    
    def print_configuration(self):
        """Print current configuration for debugging."""
        logger.info("Hlynr Bridge Server Configuration:")
        logger.info(f"  Model Checkpoint: {self.model_checkpoint}")
        logger.info(f"  VecNorm Stats ID: {self.vecnorm_stats_id}")
        logger.info(f"  Observation Version: {self.obs_version}")
        logger.info(f"  Transform Version: {self.transform_version}")
        logger.info(f"  Policy ID: {self.policy_id}")
        logger.info(f"  Server: {self.host}:{self.port}")
        logger.info(f"  CORS Origins: {self.cors_origins}")
        logger.info(f"  Log Directory: {self.log_dir}")
        logger.info(f"  Log Level: {self.log_level}")
        logger.info(f"  Safety Rate Limit: {self.rate_max_radps} rad/s")
        logger.info(f"  Latency SLOs: p50<{self.latency_slo_p50_ms}ms, p95<{self.latency_slo_p95_ms}ms")
        logger.info(f"  Scenario: {self.scenario_name}")
        logger.info(f"  Logging Enabled: {self.enable_logging}")
        logger.info(f"  Metrics Enabled: {self.enable_metrics}")
        logger.info(f"  CORS Enabled: {self.enable_cors}")
        logger.info(f"  Debug Mode: {self.debug_mode}")


def load_dotenv_if_available():
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
        else:
            logger.debug("No .env file found")
    except ImportError:
        logger.debug("python-dotenv not available, skipping .env file loading")


def get_bridge_config() -> BridgeServerConfig:
    """
    Get bridge server configuration from environment variables.
    
    This function automatically loads .env files if python-dotenv is available,
    then creates configuration from environment variables.
    
    Returns:
        BridgeServerConfig instance
    """
    # Load .env file if available
    load_dotenv_if_available()
    
    # Create configuration from environment
    config = BridgeServerConfig.from_env()
    
    # Setup logging
    config.setup_logging()
    
    # Validate configuration
    issues = config.validate_configuration()
    if issues:
        logger.warning("Configuration validation issues:")
        for issue in issues:
            if issue.startswith("ERROR"):
                logger.error(f"  {issue}")
            else:
                logger.warning(f"  {issue}")
        
        # Exit if there are any errors
        errors = [issue for issue in issues if issue.startswith("ERROR")]
        if errors:
            raise ValueError(f"Configuration errors found: {len(errors)} errors")
    
    # Print configuration if debug mode
    if config.debug_mode:
        config.print_configuration()
    
    return config


# Global configuration cache
_bridge_config_cache = None

def get_cached_bridge_config() -> BridgeServerConfig:
    """Get cached bridge configuration (loads once per process)."""
    global _bridge_config_cache
    if _bridge_config_cache is None:
        _bridge_config_cache = get_bridge_config()
    return _bridge_config_cache

def reset_bridge_config_cache():
    """Reset cached configuration (mainly for testing)."""
    global _bridge_config_cache
    _bridge_config_cache = None


if __name__ == "__main__":
    """Command-line interface for testing configuration."""
    import sys
    
    try:
        config = get_bridge_config()
        print("Configuration loaded successfully!")
        config.print_configuration()
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)