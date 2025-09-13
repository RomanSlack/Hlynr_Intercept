"""
Unified logging system for training, inference, and episode replay.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class AgentState:
    """State snapshot for logging."""
    timestamp: float
    position: List[float]
    velocity: List[float]
    orientation: List[float]  # Quaternion [w,x,y,z]
    angular_velocity: List[float]
    fuel: Optional[float] = None
    action: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, handling numpy arrays."""
        d = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
            else:
                d[key] = value
        return d


@dataclass
class EpisodeEvent:
    """Discrete event during episode."""
    timestamp: float
    event_type: str  # 'interception', 'destruction', 'spawn', 'fuel_depleted'
    source: Optional[str] = None
    target: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class UnifiedLogger:
    """Central logging system with timestamped outputs."""
    
    def __init__(self, log_dir: str = "logs", run_name: Optional[str] = None):
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            self.run_id = f"{run_name}_{timestamp}"
        else:
            self.run_id = f"run_{timestamp}"
        
        self.log_dir = Path(log_dir) / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logger
        self._setup_python_logger()
        
        # Episode tracking
        self.current_episode = None
        self.episode_count = 0
        self.episode_file = None
        self.episode_buffer = []
        
        # Metrics tracking
        self.metrics_buffer = []
        self.metrics_file = self.log_dir / "metrics.jsonl"
        
        # Training tracking
        self.training_metrics = {}
        self.training_file = self.log_dir / "training.jsonl"
        
        self.logger.info(f"Unified logger initialized: {self.log_dir}")
    
    def _setup_python_logger(self):
        """Configure Python logging to file and console."""
        self.logger = logging.getLogger(self.run_id)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_dir / "system.log")
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def begin_episode(self, episode_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """Start logging a new episode."""
        self.episode_count += 1
        episode_id = episode_id or f"ep_{self.episode_count:06d}"
        
        # Create episode file
        self.episode_file = self.log_dir / "episodes" / f"{episode_id}.jsonl"
        self.episode_file.parent.mkdir(exist_ok=True)
        
        # Write episode header
        header = {
            'type': 'header',
            'episode_id': episode_id,
            'start_time': time.time(),
            'metadata': metadata or {}
        }
        
        with open(self.episode_file, 'w') as f:
            f.write(json.dumps(header) + '\n')
        
        self.current_episode = episode_id
        self.episode_buffer = []
        
        self.logger.info(f"Episode {episode_id} started")
    
    def log_state(self, entity_id: str, state: Dict[str, Any], timestamp: Optional[float] = None):
        """Log entity state during episode."""
        if not self.current_episode:
            return
        
        timestamp = timestamp or time.time()
        
        # Convert numpy arrays to lists
        clean_state = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                clean_state[key] = value.tolist()
            else:
                clean_state[key] = value
        
        entry = {
            'type': 'state',
            'timestamp': timestamp,
            'entity_id': entity_id,
            'state': clean_state
        }
        
        self.episode_buffer.append(entry)
        
        # Flush periodically
        if len(self.episode_buffer) >= 100:
            self._flush_episode_buffer()
    
    def log_event(self, event: EpisodeEvent):
        """Log discrete event during episode."""
        if not self.current_episode:
            return
        
        entry = {
            'type': 'event',
            'timestamp': event.timestamp,
            'event_type': event.event_type,
            'source': event.source,
            'target': event.target,
            'data': event.data
        }
        
        self.episode_buffer.append(entry)
        self.logger.info(f"Event: {event.event_type} at {event.timestamp:.3f}")
    
    def end_episode(self, outcome: str, metrics: Dict[str, Any]):
        """Finish logging current episode."""
        if not self.current_episode:
            return
        
        # Flush remaining buffer
        self._flush_episode_buffer()
        
        # Write episode footer
        footer = {
            'type': 'footer',
            'episode_id': self.current_episode,
            'end_time': time.time(),
            'outcome': outcome,
            'metrics': metrics
        }
        
        with open(self.episode_file, 'a') as f:
            f.write(json.dumps(footer) + '\n')
        
        # Log metrics
        self.log_metrics({
            'episode': self.current_episode,
            'outcome': outcome,
            **metrics
        })
        
        self.logger.info(f"Episode {self.current_episode} ended: {outcome}")
        self.current_episode = None
        self.episode_file = None
    
    def _flush_episode_buffer(self):
        """Write buffered episode data to file."""
        if not self.episode_buffer or not self.episode_file:
            return
        
        with open(self.episode_file, 'a') as f:
            for entry in self.episode_buffer:
                f.write(json.dumps(entry) + '\n')
        
        self.episode_buffer = []
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics with timestamp."""
        entry = {
            'timestamp': time.time(),
            **metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        self.metrics_buffer.append(entry)
    
    def log_training_step(self, step: int, metrics: Dict[str, Any]):
        """Log training metrics."""
        entry = {
            'step': step,
            'timestamp': time.time(),
            **metrics
        }
        
        with open(self.training_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Keep recent metrics in memory
        self.training_metrics[step] = entry
        
        # Log subset to console
        key_metrics = ['mean_reward', 'loss', 'learning_rate']
        display = {k: v for k, v in metrics.items() if k in key_metrics}
        if display:
            self.logger.info(f"Step {step}: {display}")
    
    def log_inference(self, request: Dict, response: Dict, latency_ms: float):
        """Log inference request/response."""
        entry = {
            'timestamp': time.time(),
            'request': request,
            'response': response,
            'latency_ms': latency_ms
        }
        
        inference_file = self.log_dir / "inference.jsonl"
        with open(inference_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def create_manifest(self):
        """Create manifest file for the run."""
        manifest = {
            'run_id': self.run_id,
            'created': datetime.now().isoformat(),
            'log_dir': str(self.log_dir),
            'episode_count': self.episode_count,
            'files': {
                'system_log': 'system.log',
                'metrics': 'metrics.jsonl',
                'training': 'training.jsonl',
                'inference': 'inference.jsonl',
                'episodes_dir': 'episodes/'
            }
        }
        
        manifest_file = self.log_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"Manifest created: {manifest_file}")
        return manifest
    
    def get_tensorboard_writer(self):
        """Get TensorBoard SummaryWriter for this run."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.log_dir / "tensorboard"
            return SummaryWriter(str(tb_dir))
        except ImportError:
            self.logger.warning("TensorBoard not available")
            return None