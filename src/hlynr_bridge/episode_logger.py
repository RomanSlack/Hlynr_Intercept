"""Episode logger module - shim to phase4_rl implementation."""

from phase4_rl.episode_logger import *

# Global episode logger instance 
_inference_logger = None

def get_inference_logger(output_dir: str = "inference_episodes", policy_id: str = None):
    """Get the global inference logger instance with configuration."""
    global _inference_logger
    if _inference_logger is None:
        _inference_logger = EpisodeLogger(output_dir=output_dir)
        # Store policy_id as attribute for potential use in logging
        if policy_id:
            _inference_logger.policy_id = policy_id
    return _inference_logger

def reset_inference_logger():
    """Reset the global inference logger instance."""
    global _inference_logger
    _inference_logger = None