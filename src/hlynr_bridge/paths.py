"""
Centralized logging path resolver for Hlynr Intercept.

Provides single source of truth for all log directory paths with:
- Environment variable configuration (HLYNR_LOG_DIR)
- Deterministic path generation
- Automatic directory creation
- Legacy compatibility support
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Global cache for log directory
_log_dir_cache: Optional[Path] = None

def get_log_dir() -> Path:
    """
    Get the root logging directory.
    
    Returns:
        Resolved absolute path to root log directory
        
    Environment Variables:
        HLYNR_LOG_DIR: Custom log directory (default: "logs")
    """
    global _log_dir_cache
    if _log_dir_cache is None:
        log_dir_str = os.getenv("HLYNR_LOG_DIR", "logs")
        _log_dir_cache = Path(log_dir_str).resolve()
    
    return _log_dir_cache

def reset_log_dir_cache():
    """Reset the log directory cache. Used for testing."""
    global _log_dir_cache
    _log_dir_cache = None

def subdir(*parts: Union[str, Path]) -> Path:
    """
    Create and return a subdirectory under the log root.
    
    Args:
        *parts: Path components to join under log root
        
    Returns:
        Path to subdirectory (created if not exists)
    """
    path = get_log_dir()
    for part in parts:
        path = path / str(part)
    
    # Ensure directory exists
    path.mkdir(parents=True, exist_ok=True)
    return path

def logs_training(variant: Optional[str] = None) -> Path:
    """
    Get training logs directory.
    
    Args:
        variant: Training variant name (e.g. "radar17_good")
        
    Returns:
        Path to training logs directory
    """
    if variant:
        return subdir("training", variant)
    else:
        return subdir("training")

def logs_episodes(run_stamp: Optional[str] = None) -> Path:
    """
    Get episode logs directory with optional run timestamp.
    
    Args:
        run_stamp: Run timestamp (e.g. "run_2025-01-15-143022")
                  If None, returns base episodes directory
        
    Returns:
        Path to episodes directory
    """
    if run_stamp:
        return subdir("episodes", run_stamp)
    else:
        return subdir("episodes")

def logs_tensorboard(variant: Optional[str] = None) -> Path:
    """
    Get TensorBoard logs directory.
    
    Args:
        variant: Optional variant subdirectory
        
    Returns:
        Path to TensorBoard logs directory
    """
    if variant:
        return subdir("tensorboard", variant)
    else:
        return subdir("tensorboard")

def logs_inference() -> Path:
    """
    Get inference logs directory.
    
    Returns:
        Path to inference logs directory
    """
    return subdir("inference")

def logs_diagnostics() -> Path:
    """
    Get diagnostics logs directory.
    
    Returns:
        Path to diagnostics logs directory
    """
    return subdir("diagnostics")

def logs_server() -> Path:
    """
    Get server logs directory.
    
    Returns:
        Path to server logs directory
    """
    return subdir("server")

def generate_run_timestamp() -> str:
    """
    Generate consistent run timestamp string.
    
    Returns:
        Timestamp string in format "run_YYYY-MM-DD-HHMMSS"
    """
    return f"run_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"

def create_legacy_symlinks() -> dict:
    """
    Create symlinks from legacy log directories to new centralized structure.
    
    Returns:
        Dictionary with symlink creation results:
        {
            "created": ["path1", "path2"],
            "failed": [{"path": "path3", "error": "reason"}],
            "skipped": ["path4"]  # already exists
        }
    """
    results = {
        "created": [],
        "failed": [],
        "skipped": []
    }
    
    # Define legacy -> new mappings
    symlinks = [
        ("logs_radar17_good", logs_training("radar17_good")),
        ("logs_radar17_fixed", logs_training("radar17_fixed")),
        ("unity_episodes", logs_episodes()),
        ("src/phase4_rl/unity_episodes", logs_episodes()),
    ]
    
    for legacy_path, new_path in symlinks:
        legacy = Path(legacy_path)
        
        # Skip if legacy path already exists
        if legacy.exists():
            if legacy.is_symlink():
                results["skipped"].append(str(legacy))
            else:
                results["skipped"].append(f"{legacy} (exists, not symlink)")
            continue
        
        try:
            # Ensure target directory exists
            new_path.mkdir(parents=True, exist_ok=True)
            
            # Create relative symlink for cleaner links
            if legacy.parent.exists():
                relative_target = os.path.relpath(new_path, legacy.parent)
                legacy.symlink_to(relative_target)
            else:
                # Parent doesn't exist, create absolute symlink
                legacy.parent.mkdir(parents=True, exist_ok=True)
                legacy.symlink_to(new_path)
                
            results["created"].append(str(legacy))
            logger.info(f"Created symlink: {legacy} -> {new_path}")
            
        except (OSError, NotImplementedError) as e:
            results["failed"].append({
                "path": str(legacy),
                "error": str(e)
            })
            logger.warning(f"Failed to create symlink {legacy}: {e}")
    
    return results

def get_log_layout_report() -> dict:
    """
    Generate a comprehensive report of the current logging layout.
    
    Returns:
        Dictionary with logging layout information
    """
    log_dir = get_log_dir()
    
    report = {
        "log_root": str(log_dir),
        "log_root_absolute": str(log_dir.resolve()),
        "environment_var": os.getenv("HLYNR_LOG_DIR", "logs (default)"),
        "directories": {
            "training": str(logs_training()),
            "episodes": str(logs_episodes()),
            "tensorboard": str(logs_tensorboard()),
            "inference": str(logs_inference()),
            "diagnostics": str(logs_diagnostics()),
            "server": str(logs_server()),
        },
        "directory_exists": {
            "log_root": log_dir.exists(),
            "training": logs_training().exists(),
            "episodes": logs_episodes().exists(),
            "tensorboard": logs_tensorboard().exists(),
            "inference": logs_inference().exists(),
            "diagnostics": logs_diagnostics().exists(),
            "server": logs_server().exists(),
        }
    }
    
    return report

def ensure_log_directories():
    """
    Ensure all standard log directories exist.
    Called during application startup.
    """
    dirs_to_create = [
        logs_training(),
        logs_episodes(),
        logs_tensorboard(),
        logs_inference(),
        logs_diagnostics(),
        logs_server(),
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    logger.info(f"Initialized log directories under {get_log_dir()}")