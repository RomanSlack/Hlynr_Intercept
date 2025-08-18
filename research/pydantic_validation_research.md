# Pydantic Schema Validation Research for Unity-RL Bridge

## Overview

Pydantic is a high-performance data validation library for Python 3.9+ that uses type hints for schema validation and serialization. It's powered by Rust for performance and is widely adopted (360M+ downloads per month).

## Core BaseModel Patterns

### Basic Model Definition
```python
from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime

class UnityStateRequest(BaseModel):
    timestamp: float
    missile_id: str
    state: dict
    meta: Optional[dict] = None
    
    model_config = ConfigDict(
        str_max_length=100,
        frozen=False,  # Allow mutations
        extra='forbid'  # Reject extra fields
    )
```

### JSON Serialization/Deserialization
```python
# From JSON
request_data = UnityStateRequest.model_validate(json_data)

# To JSON
response_json = request_data.model_dump_json()

# To dictionary
response_dict = request_data.model_dump()
```

## Validation Patterns

### Field Validators

#### After Validators (Most Common)
```python
from pydantic import field_validator, ValidationError

class ActionResponse(BaseModel):
    thrust_command: float
    rate_cmd_radps: dict
    
    @field_validator('thrust_command', mode='after')
    @classmethod
    def validate_thrust_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError('thrust_command must be between 0.0 and 1.0')
        return value
    
    @field_validator('rate_cmd_radps', mode='after')
    @classmethod
    def validate_rate_commands(cls, value: dict) -> dict:
        required_keys = ['pitch', 'yaw', 'roll']
        for key in required_keys:
            if key not in value:
                raise ValueError(f'Missing required rate command: {key}')
            if not isinstance(value[key], (int, float)):
                raise ValueError(f'Rate command {key} must be numeric')
        return value
```

#### Before Validators (Input Transformation)
```python
class ObservationRequest(BaseModel):
    observation: list
    
    @field_validator('observation', mode='before')
    @classmethod
    def ensure_list(cls, value) -> list:
        if not isinstance(value, list):
            return [value]
        return value
    
    @field_validator('observation', mode='after')
    @classmethod
    def validate_observation_length(cls, value: list) -> list:
        expected_length = 34  # From radar environment
        if len(value) != expected_length:
            raise ValueError(f'Observation must have {expected_length} elements, got {len(value)}')
        return value
```

### Model Validators

#### Model-Level Validation
```python
from pydantic import model_validator

class EpisodeState(BaseModel):
    position: list
    velocity: list
    quaternion: list
    
    @model_validator(mode='after')
    def validate_coordinate_system(self):
        # Validate position is 3D
        if len(self.position) != 3:
            raise ValueError('Position must be 3D [x, y, z]')
        
        # Validate quaternion is normalized
        if len(self.quaternion) != 4:
            raise ValueError('Quaternion must be [w, x, y, z]')
        
        q_norm = sum(q**2 for q in self.quaternion) ** 0.5
        if abs(q_norm - 1.0) > 0.001:
            raise ValueError('Quaternion must be normalized')
        
        return self
```

### Error Handling

#### Custom Error Types
```python
from pydantic import PydanticCustomError

class SafetyConstraints(BaseModel):
    rate_max_radps: float = 5.0
    
    @field_validator('rate_max_radps')
    @classmethod
    def validate_safety_limit(cls, value: float) -> float:
        if value <= 0:
            raise PydanticCustomError(
                'safety_violation',
                'Rate limit must be positive, got {value}',
                {'value': value}
            )
        return value
```

#### Handling Validation Errors
```python
try:
    request = UnityStateRequest.model_validate(request_data)
except ValidationError as e:
    # Get structured error information
    for error in e.errors():
        field = error['loc']
        message = error['msg']
        error_type = error['type']
        print(f"Validation error in {field}: {message} (type: {error_type})")
```

## Advanced Features

### Configuration Options
```python
class APIRequest(BaseModel):
    model_config = ConfigDict(
        frozen=True,           # Immutable after creation
        extra='allow',         # Allow extra fields
        str_strip_whitespace=True,  # Strip whitespace from strings
        validate_default=True, # Validate default values
        from_attributes=True   # Allow creation from object attributes
    )
```

### Nested Models
```python
class Position3D(BaseModel):
    x: float
    y: float
    z: float

class Quaternion(BaseModel):
    w: float
    x: float
    y: float
    z: float
    
    @model_validator(mode='after')
    def normalize_quaternion(self):
        norm = (self.w**2 + self.x**2 + self.y**2 + self.z**2) ** 0.5
        if norm > 0:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
        return self

class MissileState(BaseModel):
    position: Position3D
    quaternion: Quaternion
    fuel_remaining: float
    timestamp: float
```

### Dynamic Model Creation
```python
from pydantic import create_model

# Create models at runtime based on scenario configuration
def create_observation_model(obs_length: int):
    return create_model(
        'ObservationModel',
        observation=(list, ...),  # Required list field
        __validators__={
            'validate_length': field_validator('observation')(
                lambda cls, v: v if len(v) == obs_length else 
                ValueError(f'Expected {obs_length} elements')
            )
        }
    )
```

## JSON Schema Generation

### Basic Schema Generation
```python
# Generate JSON schema for API documentation
schema = UnityStateRequest.model_json_schema()

# Schema includes:
# - Field types and constraints
# - Validation rules
# - Default values
# - Documentation strings
```

### Custom Schema Information
```python
from pydantic import Field

class ActionCommand(BaseModel):
    thrust_cmd: float = Field(
        ge=0.0, le=1.0,
        description="Normalized thrust command (0.0 to 1.0)",
        json_schema_extra={"units": "normalized"}
    )
    rate_cmd_radps: dict = Field(
        description="Angular rate commands in rad/s",
        json_schema_extra={
            "required_keys": ["pitch", "yaw", "roll"],
            "units": "rad/s"
        }
    )
```

## Flask Integration Patterns

### Request Validation Decorator
```python
from functools import wraps
from flask import request, jsonify

def validate_json(model_class):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Validate incoming JSON against Pydantic model
                validated_data = model_class.model_validate(request.get_json())
                return f(validated_data, *args, **kwargs)
            except ValidationError as e:
                return jsonify({
                    'error': 'Validation failed',
                    'details': e.errors()
                }), 400
        return wrapper
    return decorator

# Usage
@app.route('/api/action', methods=['POST'])
@validate_json(UnityStateRequest)
def get_action(validated_request: UnityStateRequest):
    # Process validated request
    return jsonify(response_data)
```

## Key Takeaways for Unity-RL Bridge

1. **Use BaseModel for all API schemas** - Ensures type safety and validation
2. **Implement comprehensive field validators** - Catch invalid data early
3. **Use model validators for cross-field validation** - Ensure data consistency
4. **Generate JSON schemas for documentation** - Self-documenting APIs
5. **Handle ValidationError gracefully** - Return structured error responses
6. **Use nested models for complex data** - Better organization and reusability
7. **Leverage configuration options** - Control behavior for different use cases