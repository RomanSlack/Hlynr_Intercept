"""Episode logger module - shim to phase4_rl implementation."""

from phase4_rl.episode_logger import *
from phase4_rl.inference_logger import InferenceEpisodeLogger

# Global episode logger instance 
_inference_logger = None

def get_inference_logger(output_dir: str = None, policy_id: str = None):
    """
    Get the global inference logger instance with configuration.
    
    Args:
        output_dir: Deprecated - now uses centralized logging paths
        policy_id: Policy identifier for logging
        
    Returns:
        InferenceEpisodeLogger instance
    """
    global _inference_logger
    if _inference_logger is None:
        _inference_logger = InferenceEpisodeLogger(output_dir=output_dir, policy_id=policy_id)
    return _inference_logger

def reset_inference_logger():
    """Reset the global inference logger instance."""
    global _inference_logger
    _inference_logger = None