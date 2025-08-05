"""
Episode logger for Unity replay system.

This module provides comprehensive logging of RL episodes in a format
that Unity can read to reconstruct and visualize the simulation.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime


@dataclass
class AgentState:
    """State snapshot for a single agent."""
    p: List[float]  # position [x, y, z] in meters
    q: List[float]  # quaternion [w, x, y, z]
    v: List[float]  # velocity [vx, vy, vz] in m/s
    w: List[float]  # angular velocity [wx, wy, wz] in rad/s
    u: Optional[List[float]] = None  # action/control input
    fuel_kg: Optional[float] = None
    thrust_n: Optional[float] = None
    status: str = "active"  # active | destroyed | finished


@dataclass
class Event:
    """Discrete event during simulation."""
    type: str
    src: Optional[str] = None
    dst: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class EpisodeLogger:
    """
    Logs episode data for Unity replay visualization.
    
    Follows the contract specification:
    - Right-handed ENU coordinates (x=East, y=North, z=Up)
    - Units: meters, seconds, radians, newtons, kg
    - Fixed timestep sampling
    - JSONL format for episodes
    """
    
    def __init__(self, 
                 output_dir: str = "runs",
                 coord_frame: str = "ENU_RH",
                 dt_nominal: float = 0.01,
                 enable_logging: bool = True):
        """
        Initialize episode logger.
        
        Args:
            output_dir: Base directory for output files
            coord_frame: Coordinate frame ("ENU_RH" or "UNITY_LH")
            dt_nominal: Nominal timestep for sampling
            enable_logging: Whether logging is enabled
        """
        self.enable_logging = enable_logging
        if not self.enable_logging:
            return
            
        self.output_dir = Path(output_dir)
        self.coord_frame = coord_frame
        self.dt_nominal = dt_nominal
        
        # Create run directory with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Episode tracking
        self.episode_count = 0
        self.current_episode_file = None
        self.current_episode_data = []
        self.episode_start_time = None
        self.episode_metadata = {}
        
        # Manifest data
        self.manifest = {
            "schema_version": "1.0",
            "coord_frame": coord_frame,
            "units": {
                "pos": "m",
                "vel": "m/s", 
                "ang": "rad",
                "time": "s"
            },
            "dt_nominal": dt_nominal,
            "gravity": [0.0, 0.0, -9.81] if coord_frame == "ENU_RH" else [0.0, -9.81, 0.0],
            "episodes": []
        }
    
    def begin_episode(self, 
                      seed: Optional[int] = None,
                      scenario_name: Optional[str] = None,
                      interceptor_config: Optional[Dict[str, Any]] = None,
                      threat_config: Optional[Dict[str, Any]] = None,
                      notes: Optional[str] = None):
        """Begin logging a new episode."""
        if not self.enable_logging:
            return
            
        self.episode_count += 1
        episode_id = f"ep_{self.episode_count:06d}"
        filename = f"{episode_id}.jsonl"
        
        self.current_episode_file = self.run_dir / filename
        self.current_episode_data = []
        self.episode_start_time = time.time()
        
        # Store episode metadata
        self.episode_metadata = {
            "ep_id": episode_id,
            "seed": seed,
            "scenario": scenario_name,
            "notes": notes or ""
        }
        
        # Add to manifest
        self.manifest["episodes"].append({
            "id": episode_id,
            "file": filename,
            "seed": seed,
            "scenario": scenario_name,
            "notes": notes or ""
        })
        
        # Write header record
        header = {
            "t": 0.0,
            "meta": {
                "ep_id": episode_id,
                "seed": seed,
                "coord_frame": self.coord_frame,
                "dt_nominal": self.dt_nominal,
                "scenario": scenario_name
            },
            "scene": {}
        }
        
        # Add interceptor configuration
        if interceptor_config:
            header["scene"]["interceptor_0"] = {
                "mass_kg": interceptor_config.get("mass", 10.0),
                "max_torque": interceptor_config.get("max_torque", [8000, 8000, 2000]),
                "sensor_fov_deg": interceptor_config.get("sensor_fov", 30.0),
                "max_thrust_n": interceptor_config.get("max_thrust", 1000.0)
            }
        
        # Add threat configuration  
        if threat_config:
            header["scene"]["threat_0"] = {
                "type": threat_config.get("type", "ballistic"),
                "mass_kg": threat_config.get("mass", 100.0),
                "aim_point": threat_config.get("aim_point", [0, 0, 0])
            }
        
        self._write_record(header)
    
    def log_step(self,
                 t: float,
                 interceptor_state: Dict[str, Any],
                 threat_state: Dict[str, Any],
                 events: Optional[List[Event]] = None):
        """
        Log a single timestep.
        
        Args:
            t: Time in seconds from episode start
            interceptor_state: State dict with p, q, v, w, etc.
            threat_state: State dict with p, q, v, w, etc.
            events: Optional list of events at this timestep
        """
        if not self.enable_logging or self.current_episode_file is None:
            return
        
        # Convert states to agent format
        agents = {}
        
        # Process interceptor
        if interceptor_state:
            agents["interceptor_0"] = self._process_agent_state(interceptor_state)
        
        # Process threat
        if threat_state:
            agents["threat_0"] = self._process_agent_state(threat_state)
        
        # Build record
        record = {
            "t": round(t, 3),  # Round to milliseconds
            "agents": agents
        }
        
        # Add events if any
        if events:
            record["events"] = [asdict(e) if isinstance(e, Event) else e for e in events]
        
        self._write_record(record)
    
    def end_episode(self,
                    outcome: str,
                    final_time: float,
                    miss_distance: Optional[float] = None,
                    impact_time: Optional[float] = None,
                    notes: Optional[str] = None):
        """
        End the current episode and write summary.
        
        Args:
            outcome: "hit" | "miss" | "timeout"
            final_time: Final simulation time
            miss_distance: Miss distance in meters (if applicable)
            impact_time: Time of impact/closest approach
            notes: Optional notes about the episode
        """
        if not self.enable_logging or self.current_episode_file is None:
            return
        
        # Write footer/summary record
        summary = {
            "t": round(final_time, 3),
            "summary": {
                "outcome": outcome,
                "episode_duration": final_time,
                "notes": notes or ""
            }
        }
        
        if miss_distance is not None:
            summary["summary"]["miss_distance_m"] = round(miss_distance, 3)
        
        if impact_time is not None:
            summary["summary"]["impact_time_s"] = round(impact_time, 3)
        
        self._write_record(summary)
        
        # Close episode
        self.current_episode_file = None
        self.current_episode_data = []
        
        # Update manifest
        self._save_manifest()
    
    def _process_agent_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw state dict to agent state format."""
        # Extract position (handle both 2D and 3D)
        if "position" in state:
            p = state["position"]
            if len(p) == 2:
                p = [p[0], p[1], 0.0]  # Add z=0 for 2D
        elif "p" in state:
            p = state["p"]
        else:
            p = [0.0, 0.0, 0.0]
        
        # Extract quaternion (convert from euler if needed)
        if "quaternion" in state:
            q = state["quaternion"]
        elif "q" in state:
            q = state["q"]
        elif "orientation" in state or "heading" in state:
            # Convert heading/euler to quaternion
            heading = state.get("orientation", state.get("heading", 0.0))
            q = self._heading_to_quaternion(heading)
        else:
            q = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
        
        # Extract velocity
        if "velocity" in state:
            v = state["velocity"]
            if len(v) == 2:
                v = [v[0], v[1], 0.0]
        elif "v" in state:
            v = state["v"]
        else:
            v = [0.0, 0.0, 0.0]
        
        # Extract angular velocity
        w = state.get("w", state.get("angular_velocity", [0.0, 0.0, 0.0]))
        if isinstance(w, (int, float)):
            w = [0.0, 0.0, w]  # Assume rotation about z-axis
        
        # Build agent state
        agent = {
            "p": [round(x, 3) for x in p],
            "q": [round(x, 4) for x in q],
            "v": [round(x, 3) for x in v],
            "w": [round(x, 4) for x in w],
            "status": state.get("status", "active")
        }
        
        # Add optional fields
        if "action" in state or "u" in state:
            agent["u"] = state.get("action", state.get("u"))
        
        if "fuel" in state or "fuel_kg" in state:
            agent["fuel_kg"] = round(state.get("fuel", state.get("fuel_kg", 0.0)), 3)
        
        if "thrust" in state or "thrust_n" in state:
            agent["thrust_n"] = round(state.get("thrust", state.get("thrust_n", 0.0)), 1)
        
        return agent
    
    def _heading_to_quaternion(self, heading: float) -> List[float]:
        """Convert heading angle (radians) to quaternion."""
        # Assuming rotation about z-axis (up)
        half_angle = heading / 2.0
        w = np.cos(half_angle)
        x = 0.0
        y = 0.0
        z = np.sin(half_angle)
        return [w, x, y, z]
    
    def _write_record(self, record: Dict[str, Any]):
        """Write a record to the current episode file."""
        if self.current_episode_file is None:
            return
        
        # Append to file (create if first write)
        with open(self.current_episode_file, 'a') as f:
            json.dump(record, f, separators=(',', ':'))
            f.write('\n')
    
    def _save_manifest(self):
        """Save the manifest file."""
        manifest_path = self.run_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def add_event(self, t: float, event_type: str, **kwargs):
        """
        Add a discrete event at the current timestep.
        
        Args:
            t: Time of event
            event_type: Type of event (e.g., "lock_acquired", "impact")
            **kwargs: Additional event data
        """
        if not self.enable_logging:
            return
        
        event = Event(type=event_type, **kwargs)
        # Events will be included in the next log_step call
        # For now, just return the event for the caller to include
        return event