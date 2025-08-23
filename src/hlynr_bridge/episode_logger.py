"""Episode logger module - shim to phase4_rl implementation."""

from phase4_rl.episode_logger import *

# Global episode logger instance 
_inference_logger = None

def get_inference_logger():
    """Get the global inference logger instance."""
    global _inference_logger
    if _inference_logger is None:
        _inference_logger = EpisodeLogger()
    return _inference_logger

def reset_inference_logger():
    """Reset the global inference logger instance."""
    global _inference_logger
    _inference_logger = None