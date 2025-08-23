"""Config module - shim to phase4_rl implementation."""

from phase4_rl.config import *

# Global config instance
_config = None

def get_config():
    """Get the global config instance."""
    global _config
    if _config is None:
        # Return a basic config dict for now
        _config = {}
    return _config

def reset_config():
    """Reset the global config instance.""" 
    global _config
    _config = None