"""
Comprehensive tests for Pydantic schemas.

Tests request/response validation, edge cases, and schema compliance
as required by the PRP.
"""

import pytest
import json
import time
from typing import Dict, Any

from ..schemas import (
    InferenceRequest, InferenceResponse, HealthResponse, MetricsResponse,
    FrameInfo, MetaInfo, BlueState, RedState, GuidanceInfo, EnvironmentInfo,
    NormalizationInfo, ActionCommand, RateCommand, DiagnosticsInfo, 
    SafetyInfo, ClipFractions, validate_obs_version, validate_transform_version,
    validate_rate_commands, check_if_clamped, SUPPORTED_OBS_VERSIONS,
    SUPPORTED_TRANSFORM_VERSIONS
)


class TestSchemaValidation:
    """Test Pydantic schema validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_request_data = {
            "meta": {
                "episode_id": "ep_000001",
                "t": 1.23,
                "dt": 0.01,
                "sim_tick": 123
            },
            "frames": {
                "frame": "ENU",
                "unity_lh": True
            },
            "blue": {
                "pos_m": [100.0, 200.0, 50.0],
                "vel_mps": [150.0, 10.0, -5.0],
                "quat_wxyz": [0.995, 0.0, 0.1, 0.0],
                "ang_vel_radps": [0.1, 0.2, 0.05],
                "fuel_frac": 0.75
            },
            "red": {
                "pos_m": [500.0, 600.0, 100.0],
                "vel_mps": [-50.0, -40.0, -10.0],
                "quat_wxyz": [0.924, 0.0, 0.0, 0.383]
            },
            "guidance": {
                "los_unit": [0.8, 0.6, 0.0],
                "los_rate_radps": [0.01, -0.02, 0.0],
                "range_m": 500.0,
                "closing_speed_mps": 200.0,
                "fov_ok": True,
                "g_limit_ok": True
            },
            "env": {
                "wind_mps": [2.0, 1.0, 0.0],
                "noise_std": 0.01,
                "episode_step": 123,
                "max_steps": 1000
            },
            "normalization": {
                "obs_version": "obs_v1.0",
                "vecnorm_stats_id": "vecnorm_baseline_001",
                "transform_version": "tfm_v1.0"
            }
        }
        
        self.valid_response_data = {
            "action": {
                "rate_cmd_radps": {
                    "pitch": 0.5,
                    "yaw": -0.2,
                    "roll": 0.1
                },
                "thrust_cmd": 0.8,
                "aux": [0.0, 0.0]
            },
            "diagnostics": {
                "policy_latency_ms": 15.3,
                "obs_clip_fractions": {
                    "low": 0.02,
                    "high": 0.01
                },
                "value_estimate": 0.75
            },
            "safety": {
                "clamped": False,
                "clamp_reason": None
            }
        }
    
    def test_valid_inference_request(self):
        """Test valid inference request validation."""
        request = InferenceRequest(**self.valid_request_data)
        
        # Check all fields are parsed correctly
        assert request.meta.episode_id == "ep_000001"
        assert request.meta.t == 1.23
        assert request.frames.frame == "ENU"
        assert request.blue.fuel_frac == 0.75
        assert len(request.blue.pos_m) == 3
        assert len(request.blue.quat_wxyz) == 4
        assert request.guidance.fov_ok is True
        assert request.normalization.obs_version == "obs_v1.0"
    
    def test_valid_inference_response(self):
        """Test valid inference response validation."""
        response = InferenceResponse(**self.valid_response_data)
        
        # Check all fields are parsed correctly
        assert response.action.rate_cmd_radps.pitch == 0.5
        assert response.action.thrust_cmd == 0.8
        assert response.diagnostics.policy_latency_ms == 15.3
        assert response.safety.clamped is False
        assert response.success is True
    
    def test_request_validation_errors(self):
        """Test request validation errors."""
        # Test missing required fields
        invalid_data = self.valid_request_data.copy()
        del invalid_data["meta"]
        
        with pytest.raises(Exception):  # Pydantic ValidationError
            InferenceRequest(**invalid_data)
        
        # Test invalid array dimensions
        invalid_data = self.valid_request_data.copy()
        invalid_data["blue"]["pos_m"] = [1.0, 2.0]  # Should be 3D
        
        with pytest.raises(Exception):
            InferenceRequest(**invalid_data)
        
        # Test invalid fuel fraction range
        invalid_data = self.valid_request_data.copy()
        invalid_data["blue"]["fuel_frac"] = 1.5  # Should be [0..1]
        
        with pytest.raises(Exception):
            InferenceRequest(**invalid_data)
        
        # Test invalid quaternion
        invalid_data = self.valid_request_data.copy()
        invalid_data["blue"]["quat_wxyz"] = [10.0, 0.0, 0.0, 0.0]  # Not normalized
        
        with pytest.raises(Exception):
            InferenceRequest(**invalid_data)
    
    def test_response_validation_errors(self):
        """Test response validation errors."""
        # Test invalid thrust command range
        invalid_data = self.valid_response_data.copy()
        invalid_data["action"]["thrust_cmd"] = 1.5  # Should be [0..1]
        
        with pytest.raises(Exception):
            InferenceResponse(**invalid_data)
        
        # Test invalid clip fractions\n        invalid_data = self.valid_response_data.copy()\n        invalid_data[\"diagnostics\"][\"obs_clip_fractions\"][\"low\"] = -0.1  # Should be [0..1]\n        \n        with pytest.raises(Exception):\n            InferenceResponse(**invalid_data)\n    \n    def test_quaternion_validation(self):\n        \"\"\"Test quaternion normalization validation.\"\"\"\n        # Valid normalized quaternion\n        valid_data = self.valid_request_data.copy()\n        valid_data[\"blue\"][\"quat_wxyz\"] = [1.0, 0.0, 0.0, 0.0]\n        request = InferenceRequest(**valid_data)\n        assert request.blue.quat_wxyz == [1.0, 0.0, 0.0, 0.0]\n        \n        # Approximately normalized quaternion (should pass)\n        valid_data[\"blue\"][\"quat_wxyz\"] = [0.995, 0.0, 0.1, 0.0]  # ~normalized\n        request = InferenceRequest(**valid_data)\n        \n        # Severely unnormalized quaternion (should fail)\n        with pytest.raises(Exception):\n            valid_data[\"blue\"][\"quat_wxyz\"] = [10.0, 0.0, 0.0, 0.0]\n            InferenceRequest(**valid_data)\n    \n    def test_los_unit_vector_validation(self):\n        \"\"\"Test line-of-sight unit vector validation.\"\"\"\n        # Valid normalized LOS vector\n        valid_data = self.valid_request_data.copy()\n        valid_data[\"guidance\"][\"los_unit\"] = [1.0, 0.0, 0.0]\n        request = InferenceRequest(**valid_data)\n        \n        # Unnormalized LOS vector (should fail)\n        with pytest.raises(Exception):\n            valid_data[\"guidance\"][\"los_unit\"] = [10.0, 0.0, 0.0]\n            InferenceRequest(**valid_data)\n    \n    def test_cross_field_validation(self):\n        \"\"\"Test cross-field validation rules.\"\"\"\n        # Test unsupported coordinate frame\n        invalid_data = self.valid_request_data.copy()\n        invalid_data[\"frames\"][\"frame\"] = \"INVALID_FRAME\"\n        \n        with pytest.raises(Exception):\n            InferenceRequest(**invalid_data)\n        \n        # Test episode step >= max_steps\n        invalid_data = self.valid_request_data.copy()\n        invalid_data[\"env\"][\"episode_step\"] = 1000\n        invalid_data[\"env\"][\"max_steps\"] = 1000\n        \n        with pytest.raises(Exception):\n            InferenceRequest(**invalid_data)\n    \n    def test_json_serialization(self):\n        \"\"\"Test JSON serialization/deserialization.\"\"\"\n        # Request serialization\n        request = InferenceRequest(**self.valid_request_data)\n        json_str = request.json()\n        parsed_data = json.loads(json_str)\n        \n        # Should be able to recreate from JSON\n        request2 = InferenceRequest(**parsed_data)\n        assert request2.meta.episode_id == request.meta.episode_id\n        \n        # Response serialization\n        response = InferenceResponse(**self.valid_response_data)\n        json_str = response.json()\n        parsed_data = json.loads(json_str)\n        \n        response2 = InferenceResponse(**parsed_data)\n        assert response2.action.thrust_cmd == response.action.thrust_cmd\n    \n    def test_optional_fields(self):\n        \"\"\"Test optional fields handling.\"\"\"\n        # Request with minimal required fields\n        minimal_data = self.valid_request_data.copy()\n        del minimal_data[\"env\"][\"wind_mps\"]  # Optional field\n        del minimal_data[\"env\"][\"noise_std\"]  # Optional field\n        \n        request = InferenceRequest(**minimal_data)\n        assert request.env.wind_mps is None\n        assert request.env.noise_std is None\n        \n        # Response with minimal fields\n        minimal_response = self.valid_response_data.copy()\n        del minimal_response[\"diagnostics\"][\"value_estimate\"]  # Optional\n        minimal_response[\"safety\"][\"clamp_reason\"] = None  # Optional\n        \n        response = InferenceResponse(**minimal_response)\n        assert response.diagnostics.value_estimate is None\n        assert response.safety.clamp_reason is None\n    \n    def test_health_response_schema(self):\n        \"\"\"Test health response schema.\"\"\"\n        health_data = {\n            \"status\": \"healthy\",\n            \"model_loaded\": True,\n            \"loaded_models\": {\"ppo_policy\": \"/path/to/model.zip\"},\n            \"vecnorm_stats_loaded\": \"vecnorm_001\",\n            \"transform_version\": \"tfm_v1.0\"\n        }\n        \n        response = HealthResponse(**health_data)\n        assert response.status == \"healthy\"\n        assert response.model_loaded is True\n        assert response.transform_version == \"tfm_v1.0\"\n        \n        # Test with defaults\n        minimal_health = {\"status\": \"not_ready\", \"model_loaded\": False}\n        response = HealthResponse(**minimal_health)\n        assert response.loaded_models == {}\n    \n    def test_metrics_response_schema(self):\n        \"\"\"Test metrics response schema.\"\"\"\n        metrics_data = {\n            \"requests_served\": 1000,\n            \"requests_failed\": 5,\n            \"latency_p50_ms\": 15.2,\n            \"latency_p95_ms\": 28.5,\n            \"latency_mean_ms\": 16.8,\n            \"requests_per_second\": 25.5,\n            \"safety_clamps_total\": 10,\n            \"safety_clamp_rate\": 0.01\n        }\n        \n        response = MetricsResponse(**metrics_data)\n        assert response.requests_served == 1000\n        assert response.latency_p50_ms == 15.2\n        assert response.safety_clamp_rate == 0.01\n    \n    def test_version_validation_functions(self):\n        \"\"\"Test version validation helper functions.\"\"\"\n        # Test obs version validation\n        assert validate_obs_version(\"obs_v1.0\") is True\n        assert validate_obs_version(\"invalid\") is False\n        \n        # Test transform version validation\n        assert validate_transform_version(\"tfm_v1.0\") is True\n        assert validate_transform_version(\"invalid\") is False\n        \n        # Test supported versions lists\n        assert \"obs_v1.0\" in SUPPORTED_OBS_VERSIONS\n        assert \"tfm_v1.0\" in SUPPORTED_TRANSFORM_VERSIONS\n    \n    def test_rate_command_validation(self):\n        \"\"\"Test rate command validation and clamping.\"\"\"\n        # Valid rate command\n        rate_cmd = RateCommand(pitch=0.5, yaw=-0.2, roll=0.1)\n        clamped = validate_rate_commands(rate_cmd)\n        assert clamped.pitch == 0.5\n        assert clamped.yaw == -0.2\n        assert clamped.roll == 0.1\n        \n        # Rate command needing clamping\n        rate_cmd = RateCommand(pitch=15.0, yaw=-12.0, roll=8.0)\n        clamped = validate_rate_commands(rate_cmd)\n        assert clamped.pitch == 10.0  # Clamped to max\n        assert clamped.yaw == -10.0   # Clamped to min\n        assert clamped.roll == 8.0    # Within limits\n    \n    def test_clamp_detection(self):\n        \"\"\"Test clamp detection functionality.\"\"\"\n        # No clamping needed\n        original = RateCommand(pitch=0.5, yaw=-0.2, roll=0.1)\n        clamped = RateCommand(pitch=0.5, yaw=-0.2, roll=0.1)\n        is_clamped, reason = check_if_clamped(original, clamped)\n        assert is_clamped is False\n        assert reason == \"\"\n        \n        # Clamping occurred\n        original = RateCommand(pitch=15.0, yaw=-12.0, roll=0.1)\n        clamped = RateCommand(pitch=10.0, yaw=-10.0, roll=0.1)\n        is_clamped, reason = check_if_clamped(original, clamped)\n        assert is_clamped is True\n        assert \"pitch\" in reason\n        assert \"yaw\" in reason\n        assert \"roll\" not in reason\n    \n    def test_deterministic_serialization(self):\n        \"\"\"Test deterministic JSON serialization.\"\"\"\n        request = InferenceRequest(**self.valid_request_data)\n        \n        # Multiple serializations should be identical\n        json1 = request.json(sort_keys=True)\n        json2 = request.json(sort_keys=True)\n        json3 = request.json(sort_keys=True)\n        \n        assert json1 == json2 == json3\n        \n        # Test response determinism\n        response = InferenceResponse(**self.valid_response_data)\n        json1 = response.json(sort_keys=True)\n        json2 = response.json(sort_keys=True)\n        \n        assert json1 == json2\n    \n    def test_schema_backwards_compatibility(self):\n        \"\"\"Test schema backwards compatibility.\"\"\"\n        # Test that old valid requests still work\n        legacy_request = {\n            \"meta\": {\"episode_id\": \"ep_001\", \"t\": 1.0, \"dt\": 0.01, \"sim_tick\": 100},\n            \"frames\": {\"frame\": \"ENU\", \"unity_lh\": True},\n            \"blue\": {\n                \"pos_m\": [0.0, 0.0, 0.0],\n                \"vel_mps\": [0.0, 0.0, 0.0],\n                \"quat_wxyz\": [1.0, 0.0, 0.0, 0.0],\n                \"ang_vel_radps\": [0.0, 0.0, 0.0],\n                \"fuel_frac\": 1.0\n            },\n            \"red\": {\n                \"pos_m\": [100.0, 0.0, 0.0],\n                \"vel_mps\": [0.0, 0.0, 0.0],\n                \"quat_wxyz\": [1.0, 0.0, 0.0, 0.0]\n            },\n            \"guidance\": {\n                \"los_unit\": [1.0, 0.0, 0.0],\n                \"los_rate_radps\": [0.0, 0.0, 0.0],\n                \"range_m\": 100.0,\n                \"closing_speed_mps\": 0.0,\n                \"fov_ok\": True,\n                \"g_limit_ok\": True\n            },\n            \"env\": {\"episode_step\": 0, \"max_steps\": 1000},\n            \"normalization\": {\n                \"obs_version\": \"obs_v1.0\",\n                \"vecnorm_stats_id\": \"baseline\",\n                \"transform_version\": \"tfm_v1.0\"\n            }\n        }\n        \n        # Should parse without errors\n        request = InferenceRequest(**legacy_request)\n        assert request.meta.episode_id == \"ep_001\"\n    \n    def test_field_constraints(self):\n        \"\"\"Test field constraints and edge cases.\"\"\"\n        # Test range constraints\n        test_cases = [\n            # (field_path, invalid_value, should_raise)\n            (\"blue.fuel_frac\", -0.1, True),     # Below 0\n            (\"blue.fuel_frac\", 1.1, True),      # Above 1\n            (\"blue.fuel_frac\", 0.0, False),     # Valid boundary\n            (\"blue.fuel_frac\", 1.0, False),     # Valid boundary\n            (\"guidance.range_m\", -1.0, True),   # Negative range\n            (\"guidance.range_m\", 0.0, False),   # Valid boundary\n            (\"env.episode_step\", -1, True),     # Negative step\n            (\"env.max_steps\", 0, True),         # Zero max steps\n        ]\n        \n        for field_path, value, should_raise in test_cases:\n            test_data = self.valid_request_data.copy()\n            \n            # Navigate to nested field\n            parts = field_path.split('.')\n            target = test_data\n            for part in parts[:-1]:\n                target = target[part]\n            target[parts[-1]] = value\n            \n            if should_raise:\n                with pytest.raises(Exception):\n                    InferenceRequest(**test_data)\n            else:\n                # Should not raise\n                request = InferenceRequest(**test_data)\n                assert request is not None\n    \n    def test_performance_large_requests(self):\n        \"\"\"Test performance with large requests.\"\"\"\n        import time\n        \n        # Create request with maximum aux array\n        large_request_data = self.valid_request_data.copy()\n        \n        # Create large response with big aux array\n        large_response_data = self.valid_response_data.copy()\n        large_response_data[\"action\"][\"aux\"] = [0.0] * 100  # Large aux array\n        \n        # Benchmark parsing performance\n        start_time = time.perf_counter()\n        for _ in range(1000):\n            request = InferenceRequest(**large_request_data)\n            response = InferenceResponse(**large_response_data)\n        parse_time = time.perf_counter() - start_time\n        \n        # Should be reasonably fast\n        assert parse_time < 5.0, f\"Schema parsing too slow: {parse_time:.3f}s\"\n        print(f\"Schema parsing rate: {2000 / parse_time:.0f} schemas/sec\")\n\n\nif __name__ == \"__main__\":\n    pytest.main([__file__])