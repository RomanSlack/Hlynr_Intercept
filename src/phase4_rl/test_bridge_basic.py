#!/usr/bin/env python3
"""
Basic test for bridge server functionality.
"""

import sys
import traceback
from pathlib import Path

# Test import
try:
    from bridge_server import BridgeServer
    from client_stub import BridgeClient, generate_dummy_observation
    print("✅ Bridge server imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test bridge server initialization
try:
    checkpoint_path = "../../checkpoints/best_model.zip"
    server = BridgeServer(
        checkpoint_path=checkpoint_path,
        scenario_name="easy",
        host="localhost",
        port=5001
    )
    print("✅ Bridge server initialization successful")
except Exception as e:
    print(f"❌ Bridge server initialization failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test dummy observation generation
try:
    obs = generate_dummy_observation(30)
    print(f"✅ Dummy observation generated: shape={len(obs)}, sample={obs[:3]}")
    
    # Validate observation format
    assert len(obs) == 30, f"Expected 30 dimensions, got {len(obs)}"
    assert all(isinstance(x, (int, float)) for x in obs), "All values should be numeric"
    
    print("✅ Observation validation passed")
except Exception as e:
    print(f"❌ Dummy observation generation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test client initialization
try:
    client = BridgeClient(host="localhost", port=5001)
    print("✅ Bridge client initialization successful")
except Exception as e:
    print(f"❌ Bridge client initialization failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All basic bridge tests passed!")
print("✅ Bridge server can be initialized")
print("✅ Dummy observations can be generated")
print("✅ Bridge client can be created")
print("✅ All imports are working")

print("\nTo test the full functionality:")
print("1. Start server: python bridge_server.py --checkpoint ../../checkpoints/best_model.zip")
print("2. In another terminal: python client_stub.py --test-type all")