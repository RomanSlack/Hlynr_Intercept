# JSON Lines (JSONL) Format Research

## Format Specification

JSON Lines is a streaming data format where each line is a valid JSON value:

### Core Rules
- **UTF-8 encoding required**
- Each line must be a valid JSON value (object, array, null, etc.)
- Line terminator must be `\n` (newline)
- Recommended file extension: `.jsonl`
- MIME type: `application/jsonl` (not yet standardized)

### Example Format
```jsonl
{"name": "John", "age": 30}
{"name": "Jane", "age": 25}
{"event": "user_login", "timestamp": 1632123456}
```

## Key Characteristics

### Streaming Capabilities
- Designed for **processing one record at a time**
- Ideal for log files and inter-process messaging
- Supports continuous reading/writing without loading entire file
- Works well with Unix-style text processing tools and shell pipelines

### Performance Benefits
- Lightweight text-based format
- Low overhead for parsing individual lines
- Easy to read and process incrementally
- Compressible with standard tools (gzip, bzip2)

### File Organization
- First line is "value 1"
- No array wrapper - each line is independent
- Easy concatenation of multiple files
- Supports append-only operations

## Reading Patterns

### Python Implementation
```python
import json
from typing import Iterator, Dict, Any

def read_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
    """Read JSONL file line by line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                yield json.loads(line)

def read_jsonl_batch(file_path: str, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
    """Read JSONL in batches for memory efficiency."""
    batch = []
    for record in read_jsonl(file_path):
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:  # Yield remaining records
        yield batch
```

### Streaming with Error Handling
```python
import logging

def safe_read_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
    """Read JSONL with error handling for malformed lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping malformed JSON on line {line_num}: {e}")
                continue
```

## Writing Patterns

### Basic Writing
```python
import json
from typing import Dict, Any, List

def write_jsonl(file_path: str, records: List[Dict[str, Any]]):
    """Write records to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, separators=(',', ':'))
            f.write('\n')

def append_jsonl(file_path: str, record: Dict[str, Any]):
    """Append single record to JSONL file."""
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(record, f, separators=(',', ':'))
        f.write('\n')
```

### Streaming Writer
```python
class JSONLWriter:
    """Context manager for streaming JSONL writing."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file = None
    
    def __enter__(self):
        self.file = open(self.file_path, 'w', encoding='utf-8')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
    
    def write_record(self, record: Dict[str, Any]):
        """Write a single record."""
        json.dump(record, self.file, separators=(',', ':'))
        self.file.write('\n')
        self.file.flush()  # Ensure immediate write

# Usage
with JSONLWriter('output.jsonl') as writer:
    writer.write_record({"step": 1, "value": 42})
    writer.write_record({"step": 2, "value": 43})
```

## Unity Episode Logging Application

### Episode Structure
Based on the Unity Data Reference, episodes follow this JSONL pattern:

```jsonl
{"t": 0.0, "meta": {"ep_id": "ep_000001", "seed": 12345}, "scene": {...}}
{"t": 0.01, "agents": {"interceptor_0": {...}, "threat_0": {...}}}
{"t": 0.02, "agents": {"interceptor_0": {...}, "threat_0": {...}}}
...
{"t": 12.37, "summary": {"outcome": "hit", "miss_distance_m": 2.1}}
```

### Performance Considerations for Unity
```python
class EpisodeJSONLWriter:
    """Optimized JSONL writer for Unity episodes."""
    
    def __init__(self, file_path: str, buffer_size: int = 8192):
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.file = None
    
    def __enter__(self):
        self.file = open(self.file_path, 'w', encoding='utf-8', buffering=self.buffer_size)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
    
    def write_timestep(self, t: float, agents: Dict[str, Any], events: List[Dict] = None):
        """Write timestep record with consistent formatting."""
        record = {
            "t": round(t, 3),  # Millisecond precision
            "agents": agents
        }
        if events:
            record["events"] = events
        
        # Use compact JSON for space efficiency
        json.dump(record, self.file, separators=(',', ':'), ensure_ascii=False)
        self.file.write('\n')
    
    def write_header(self, meta: Dict[str, Any], scene: Dict[str, Any]):
        """Write episode header."""
        header = {"t": 0.0, "meta": meta, "scene": scene}
        json.dump(header, self.file, separators=(',', ':'))
        self.file.write('\n')
    
    def write_summary(self, t: float, summary: Dict[str, Any]):
        """Write episode summary."""
        record = {"t": round(t, 3), "summary": summary}
        json.dump(record, self.file, separators=(',', ':'))
        self.file.write('\n')
```

## Best Practices for Unity Integration

### File Organization
```python
def organize_episode_files(run_dir: str, max_episodes_per_manifest: int = 100):
    """Organize episodes with proper manifest structure."""
    
    # Create manifest for episode tracking
    manifest = {
        "schema_version": "1.0",
        "coord_frame": "ENU_RH",
        "units": {"pos": "m", "vel": "m/s", "ang": "rad", "time": "s"},
        "dt_nominal": 0.01,
        "episodes": []
    }
    
    # Track episodes
    for episode_file in Path(run_dir).glob("ep_*.jsonl"):
        episode_id = episode_file.stem
        manifest["episodes"].append({
            "id": episode_id,
            "file": episode_file.name,
            "size_bytes": episode_file.stat().st_size
        })
    
    # Save manifest
    with open(Path(run_dir) / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
```

### Memory-Efficient Reading for Unity
```python
def stream_episode_for_unity(episode_path: str) -> Iterator[Dict[str, Any]]:
    """Stream episode data optimized for Unity consumption."""
    
    with open(episode_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                record = json.loads(line.strip())
                
                # Add line number for debugging
                record['_line'] = line_num + 1
                
                yield record
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num + 1}: {e}")
                continue
```

### Validation
```python
def validate_episode_jsonl(file_path: str) -> List[str]:
    """Validate JSONL episode file format."""
    errors = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                # Validate required fields
                if 't' not in record:
                    errors.append(f"Line {line_num}: Missing 't' field")
                
                if line_num == 1 and 'meta' not in record:
                    errors.append(f"Line {line_num}: Header missing 'meta' field")
                
                if 'agents' in record:
                    for agent_id, agent_data in record['agents'].items():
                        required_fields = ['p', 'q', 'v', 'w']
                        for field in required_fields:
                            if field not in agent_data:
                                errors.append(f"Line {line_num}: Agent {agent_id} missing '{field}'")
                
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON decode error: {e}")
    
    return errors
```

## Key Takeaways for Unity-RL Bridge

1. **Use append-only writes** - Perfect for real-time episode logging
2. **Stream processing** - Don't load entire files into memory
3. **Error handling** - Skip malformed lines gracefully
4. **Consistent formatting** - Use separators=(',', ':') for compact output
5. **UTF-8 encoding** - Ensure proper character handling
6. **Line buffering** - Use appropriate buffer sizes for performance
7. **Validation** - Check format compliance before Unity consumption