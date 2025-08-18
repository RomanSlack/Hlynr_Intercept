"""
Tests for safety clamping system.

Tests safety limits enforcement, logging, and statistics tracking.
"""

import pytest
import time
import numpy as np

from ..clamps import (
    SafetyClampSystem, SafetyLimits, ClampResult,
    get_safety_clamp_system, apply_safety_clamps
)
from ..schemas import ActionCommand, RateCommand, SafetyInfo


class TestSafetyClamps:
    """Test safety clamping functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.safety_limits = SafetyLimits(
            rate_max_radps=10.0,
            rate_min_radps=-10.0,
            thrust_max=1.0,
            thrust_min=0.0,
            aux_max=1.0,
            aux_min=-1.0
        )
        self.clamp_system = SafetyClampSystem(self.safety_limits)
    
    def test_no_clamping_needed(self):
        """Test actions within safety limits."""
        action = ActionCommand(
            rate_cmd_radps=RateCommand(pitch=0.5, yaw=-0.2, roll=0.1),
            thrust_cmd=0.8,
            aux=[0.0, 0.5]
        )
        
        clamped_action, safety_info = self.clamp_system.apply_safety_clamps(action)
        
        # Should be unchanged
        assert clamped_action.rate_cmd_radps.pitch == 0.5
        assert clamped_action.rate_cmd_radps.yaw == -0.2
        assert clamped_action.rate_cmd_radps.roll == 0.1
        assert clamped_action.thrust_cmd == 0.8
        assert clamped_action.aux == [0.0, 0.5]
        
        # No clamping should be reported
        assert safety_info.clamped is False
        assert safety_info.clamp_reason is None
    
    def test_rate_clamping(self):
        """Test angular rate clamping."""
        action = ActionCommand(
            rate_cmd_radps=RateCommand(pitch=15.0, yaw=-12.0, roll=8.0),
            thrust_cmd=0.8,
            aux=[]
        )
        
        clamped_action, safety_info = self.clamp_system.apply_safety_clamps(action)
        
        # Rates should be clamped
        assert clamped_action.rate_cmd_radps.pitch == 10.0    # Clamped to max
        assert clamped_action.rate_cmd_radps.yaw == -10.0     # Clamped to min
        assert clamped_action.rate_cmd_radps.roll == 8.0      # Within limits
        assert clamped_action.thrust_cmd == 0.8               # Unchanged
        
        # Clamping should be reported
        assert safety_info.clamped is True
        assert "angular_rates" in safety_info.clamp_reason
    
    def test_thrust_clamping(self):
        """Test thrust command clamping."""
        # Test thrust too high
        action_high = ActionCommand(
            rate_cmd_radps=RateCommand(pitch=0.0, yaw=0.0, roll=0.0),
            thrust_cmd=1.5,
            aux=[]
        )
        
        clamped_action, safety_info = self.clamp_system.apply_safety_clamps(action_high)
        
        assert clamped_action.thrust_cmd == 1.0  # Clamped to max
        assert safety_info.clamped is True
        assert "thrust" in safety_info.clamp_reason
        
        # Test thrust too low
        action_low = ActionCommand(
            rate_cmd_radps=RateCommand(pitch=0.0, yaw=0.0, roll=0.0),
            thrust_cmd=-0.1,
            aux=[]
        )
        
        clamped_action, safety_info = self.clamp_system.apply_safety_clamps(action_low)
        
        assert clamped_action.thrust_cmd == 0.0  # Clamped to min
        assert safety_info.clamped is True
    
    def test_auxiliary_clamping(self):
        """Test auxiliary command clamping."""
        action = ActionCommand(
            rate_cmd_radps=RateCommand(pitch=0.0, yaw=0.0, roll=0.0),
            thrust_cmd=0.5,
            aux=[2.0, -2.0, 0.5]  # First two need clamping
        )
        
        clamped_action, safety_info = self.clamp_system.apply_safety_clamps(action)
        
        assert clamped_action.aux[0] == 1.0   # Clamped to max\n        assert clamped_action.aux[1] == -1.0  # Clamped to min\n        assert clamped_action.aux[2] == 0.5   # Within limits\n        \n        assert safety_info.clamped is True\n        assert \"auxiliary\" in safety_info.clamp_reason\n    \n    def test_multiple_clamps(self):\n        \"\"\"Test multiple types of clamping simultaneously.\"\"\"\n        action = ActionCommand(\n            rate_cmd_radps=RateCommand(pitch=15.0, yaw=0.0, roll=0.0),\n            thrust_cmd=1.5,\n            aux=[2.0]\n        )\n        \n        clamped_action, safety_info = self.clamp_system.apply_safety_clamps(action)\n        \n        # All should be clamped\n        assert clamped_action.rate_cmd_radps.pitch == 10.0\n        assert clamped_action.thrust_cmd == 1.0\n        assert clamped_action.aux[0] == 1.0\n        \n        # All clamp types should be reported\n        assert safety_info.clamped is True\n        assert \"angular_rates\" in safety_info.clamp_reason\n        assert \"thrust\" in safety_info.clamp_reason\n        assert \"auxiliary\" in safety_info.clamp_reason\n    \n    def test_clamp_statistics(self):\n        \"\"\"Test clamp statistics tracking.\"\"\"\n        # Apply several clamps\n        test_actions = [\n            ActionCommand(rate_cmd_radps=RateCommand(0.0, 0.0, 0.0), thrust_cmd=0.5, aux=[]),  # No clamp\n            ActionCommand(rate_cmd_radps=RateCommand(15.0, 0.0, 0.0), thrust_cmd=0.5, aux=[]), # Pitch clamp\n            ActionCommand(rate_cmd_radps=RateCommand(0.0, -15.0, 0.0), thrust_cmd=0.5, aux=[]), # Yaw clamp\n            ActionCommand(rate_cmd_radps=RateCommand(0.0, 0.0, 0.0), thrust_cmd=1.5, aux=[]),  # Thrust clamp\n            ActionCommand(rate_cmd_radps=RateCommand(0.0, 0.0, 0.0), thrust_cmd=0.5, aux=[2.0]), # Aux clamp\n        ]\n        \n        for action in test_actions:\n            self.clamp_system.apply_safety_clamps(action)\n        \n        stats = self.clamp_system.get_clamp_statistics()\n        \n        assert stats['total_commands'] == 5\n        assert stats['total_clamped'] == 4  # All except first\n        assert stats['overall_clamp_rate'] == 0.8  # 4/5\n        assert stats['clamps_by_axis']['pitch'] == 1\n        assert stats['clamps_by_axis']['yaw'] == 1\n        assert stats['clamps_by_axis']['thrust'] == 1\n        assert stats['clamps_by_axis']['aux'] == 1\n    \n    def test_action_validation(self):\n        \"\"\"Test action validation without clamping.\"\"\"\n        # Valid action\n        valid_action = ActionCommand(\n            rate_cmd_radps=RateCommand(pitch=0.5, yaw=-0.2, roll=0.1),\n            thrust_cmd=0.8,\n            aux=[0.0]\n        )\n        \n        violations = self.clamp_system.validate_action(valid_action)\n        assert len(violations) == 0\n        \n        # Invalid action\n        invalid_action = ActionCommand(\n            rate_cmd_radps=RateCommand(pitch=15.0, yaw=-12.0, roll=0.1),\n            thrust_cmd=1.5,\n            aux=[2.0]\n        )\n        \n        violations = self.clamp_system.validate_action(invalid_action)\n        assert len(violations) > 0\n        assert any(\"Pitch rate\" in v for v in violations)\n        assert any(\"Yaw rate\" in v for v in violations)\n        assert any(\"Thrust\" in v for v in violations)\n        assert any(\"Aux\" in v for v in violations)\n    \n    def test_recent_clamps_tracking(self):\n        \"\"\"Test recent clamps tracking for diagnostics.\"\"\"\n        # Generate some clamps\n        for i in range(15):\n            action = ActionCommand(\n                rate_cmd_radps=RateCommand(pitch=15.0, yaw=0.0, roll=0.0),\n                thrust_cmd=0.5,\n                aux=[]\n            )\n            self.clamp_system.apply_safety_clamps(action)\n            time.sleep(0.001)  # Small delay to differentiate timestamps\n        \n        # Get recent clamps\n        recent = self.clamp_system.get_recent_clamps(10)\n        \n        assert len(recent) == 10  # Should limit to requested count\n        \n        # Should be in chronological order (most recent first)\n        for i in range(1, len(recent)):\n            assert recent[i-1]['timestamp'] >= recent[i]['timestamp']\n        \n        # All should have pitch clamping\n        for clamp in recent:\n            assert \"Pitch rate\" in clamp['clamp_reason'] or \"angular_rates\" in clamp['clamp_reason']\n    \n    def test_limits_update(self):\n        \"\"\"Test updating safety limits.\"\"\"\n        original_limits = self.clamp_system.limits\n        \n        # Update limits\n        new_limits = SafetyLimits(\n            rate_max_radps=5.0,  # More restrictive\n            rate_min_radps=-5.0,\n            thrust_max=0.9,\n            thrust_min=0.1,\n            aux_max=0.8,\n            aux_min=-0.8\n        )\n        \n        self.clamp_system.update_limits(new_limits)\n        \n        # Test that new limits are applied\n        action = ActionCommand(\n            rate_cmd_radps=RateCommand(pitch=8.0, yaw=0.0, roll=0.0),  # Would be OK with old limits\n            thrust_cmd=0.95,  # Would be OK with old limits\n            aux=[]\n        )\n        \n        clamped_action, safety_info = self.clamp_system.apply_safety_clamps(action)\n        \n        assert clamped_action.rate_cmd_radps.pitch == 5.0  # Clamped with new limit\n        assert clamped_action.thrust_cmd == 0.9  # Clamped with new limit\n        assert safety_info.clamped is True\n    \n    def test_statistics_reset(self):\n        \"\"\"Test statistics reset functionality.\"\"\"\n        # Generate some statistics\n        action = ActionCommand(\n            rate_cmd_radps=RateCommand(pitch=15.0, yaw=0.0, roll=0.0),\n            thrust_cmd=0.5,\n            aux=[]\n        )\n        \n        for _ in range(5):\n            self.clamp_system.apply_safety_clamps(action)\n        \n        # Verify statistics exist\n        stats = self.clamp_system.get_clamp_statistics()\n        assert stats['total_commands'] == 5\n        assert stats['total_clamped'] == 5\n        \n        # Reset statistics\n        self.clamp_system.reset_statistics()\n        \n        # Verify reset\n        stats = self.clamp_system.get_clamp_statistics()\n        assert stats['total_commands'] == 0\n        assert stats['total_clamped'] == 0\n        assert stats['clamps_by_axis']['pitch'] == 0\n    \n    def test_clamp_result_recording(self):\n        \"\"\"Test detailed clamp result recording.\"\"\"\n        action = ActionCommand(\n            rate_cmd_radps=RateCommand(pitch=15.0, yaw=-12.0, roll=0.1),\n            thrust_cmd=1.5,\n            aux=[2.0, -2.0]\n        )\n        \n        clamped_action, safety_info = self.clamp_system.apply_safety_clamps(action)\n        \n        # Get recent clamps for detailed analysis\n        recent = self.clamp_system.get_recent_clamps(1)\n        assert len(recent) == 1\n        \n        clamp_event = recent[0]\n        \n        # Should record original and clamped values\n        assert clamp_event['original_action']['rate_cmd_radps']['pitch'] == 15.0\n        assert clamp_event['clamped_action']['rate_cmd_radps']['pitch'] == 10.0\n        \n        assert clamp_event['original_action']['thrust_cmd'] == 1.5\n        assert clamp_event['clamped_action']['thrust_cmd'] == 1.0\n        \n        # Should have detailed clamp information\n        assert 'pitch' in clamp_event['clamp_details']\n        assert 'yaw' in clamp_event['clamp_details']\n        assert 'thrust' in clamp_event['clamp_details']\n        assert 'aux_0' in clamp_event['clamp_details']\n        assert 'aux_1' in clamp_event['clamp_details']\n    \n    def test_global_clamp_system(self):\n        \"\"\"Test global clamp system management.\"\"\"\n        # Test getting global instance\n        system1 = get_safety_clamp_system()\n        system2 = get_safety_clamp_system()\n        \n        # Should return same instance\n        assert system1 is system2\n        \n        # Test apply_safety_clamps convenience function\n        action = ActionCommand(\n            rate_cmd_radps=RateCommand(pitch=15.0, yaw=0.0, roll=0.0),\n            thrust_cmd=0.5,\n            aux=[]\n        )\n        \n        clamped_action, safety_info = apply_safety_clamps(action)\n        assert clamped_action.rate_cmd_radps.pitch == 10.0\n        assert safety_info.clamped is True\n    \n    def test_deterministic_clamping(self):\n        \"\"\"Test that clamping is deterministic.\"\"\"\n        action = ActionCommand(\n            rate_cmd_radps=RateCommand(pitch=15.0, yaw=-12.0, roll=8.0),\n            thrust_cmd=1.5,\n            aux=[2.0, -2.0, 0.5]\n        )\n        \n        # Apply clamping multiple times\n        results = []\n        for _ in range(10):\n            clamped_action, safety_info = self.clamp_system.apply_safety_clamps(action)\n            \n            result = {\n                'pitch': clamped_action.rate_cmd_radps.pitch,\n                'yaw': clamped_action.rate_cmd_radps.yaw,\n                'roll': clamped_action.rate_cmd_radps.roll,\n                'thrust': clamped_action.thrust_cmd,\n                'aux': clamped_action.aux,\n                'clamped': safety_info.clamped\n            }\n            results.append(result)\n        \n        # All results should be identical\n        for i in range(1, len(results)):\n            assert results[0] == results[i], \"Clamping not deterministic\"\n    \n    def test_edge_cases(self):\n        \"\"\"Test edge cases and boundary conditions.\"\"\"\n        # Exactly at limits (should not be clamped)\n        action = ActionCommand(\n            rate_cmd_radps=RateCommand(\n                pitch=self.safety_limits.rate_max_radps,\n                yaw=self.safety_limits.rate_min_radps,\n                roll=0.0\n            ),\n            thrust_cmd=self.safety_limits.thrust_max,\n            aux=[self.safety_limits.aux_max, self.safety_limits.aux_min]\n        )\n        \n        clamped_action, safety_info = self.clamp_system.apply_safety_clamps(action)\n        \n        # Should not be clamped\n        assert safety_info.clamped is False\n        assert clamped_action.rate_cmd_radps.pitch == self.safety_limits.rate_max_radps\n        assert clamped_action.thrust_cmd == self.safety_limits.thrust_max\n        \n        # Empty aux array\n        action_empty_aux = ActionCommand(\n            rate_cmd_radps=RateCommand(pitch=0.0, yaw=0.0, roll=0.0),\n            thrust_cmd=0.5,\n            aux=[]\n        )\n        \n        clamped_action, safety_info = self.clamp_system.apply_safety_clamps(action_empty_aux)\n        assert safety_info.clamped is False\n        assert clamped_action.aux == []\n    \n    def test_performance_benchmarks(self):\n        \"\"\"Test clamping performance.\"\"\"\n        import time\n        \n        action = ActionCommand(\n            rate_cmd_radps=RateCommand(pitch=15.0, yaw=-12.0, roll=8.0),\n            thrust_cmd=1.5,\n            aux=[2.0, -2.0]\n        )\n        \n        # Benchmark clamping performance\n        start_time = time.perf_counter()\n        for _ in range(10000):\n            self.clamp_system.apply_safety_clamps(action)\n        clamp_time = time.perf_counter() - start_time\n        \n        # Should be very fast\n        assert clamp_time < 1.0, f\"Clamping too slow: {clamp_time:.3f}s\"\n        \n        rate = 10000 / clamp_time\n        print(f\"Clamping rate: {rate:.0f} ops/sec\")\n        \n        # Should easily meet real-time requirements\n        assert rate > 1000, f\"Clamping rate too low: {rate:.0f} ops/sec\"\n\n\nif __name__ == \"__main__\":\n    pytest.main([__file__])