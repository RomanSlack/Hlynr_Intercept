"""
Hlynr Bridge - Unity↔RL Inference API Bridge v1.0

Canonical package structure for the Hlynr Intercept inference API bridge.
"""

# Use absolute imports to avoid circular import issues
try:
    from hlynr_bridge.server import HlynrBridgeServer
    from hlynr_bridge.schemas import InferenceRequest, InferenceResponse, HealthResponse
    from hlynr_bridge.transforms import get_transform, validate_transform_version
    from hlynr_bridge.normalize import load_vecnorm, set_deterministic_inference_mode
    from hlynr_bridge.seed_manager import set_deterministic_seeds, get_current_seed
except ImportError:
    # Fallback to relative imports if absolute don't work
    from .server import HlynrBridgeServer
    from .schemas import InferenceRequest, InferenceResponse, HealthResponse
    from .transforms import get_transform, validate_transform_version
    from .normalize import load_vecnorm, set_deterministic_inference_mode
    from .seed_manager import set_deterministic_seeds, get_current_seed

__version__ = "1.0.0"
__all__ = [
    "HlynrBridgeServer",
    "InferenceRequest", 
    "InferenceResponse",
    "HealthResponse",
    "get_transform",
    "validate_transform_version", 
    "load_vecnorm",
    "set_deterministic_inference_mode",
    "set_deterministic_seeds",
    "get_current_seed"
]