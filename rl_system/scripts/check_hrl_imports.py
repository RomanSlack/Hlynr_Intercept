#!/usr/bin/env python3
"""
Safety check: Verify all HRL modules can be imported without errors.

Phase 1 smoke test to ensure scaffolding is properly set up.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_hrl_imports():
    """Test all HRL module imports."""
    print("=" * 60)
    print("HRL Import Safety Check")
    print("=" * 60)

    modules_to_test = [
        "hrl.option_definitions",
        "hrl.observation_abstraction",
        "hrl.option_manager",
        "hrl.specialist_policies",
        "hrl.selector_policy",
        "hrl.manager",
        "hrl.wrappers",
        "hrl.hierarchical_env",
    ]

    failed_imports = []

    for module_name in modules_to_test:
        try:
            print(f"\nTesting: {module_name}...", end=" ")
            __import__(module_name)
            print("✓ OK")
        except Exception as e:
            print(f"✗ FAILED")
            print(f"  Error: {e}")
            failed_imports.append((module_name, str(e)))

    print("\n" + "=" * 60)

    if failed_imports:
        print(f"❌ FAILED: {len(failed_imports)}/{len(modules_to_test)} imports failed")
        for module, error in failed_imports:
            print(f"  - {module}: {error}")
        return False
    else:
        print(f"✅ SUCCESS: All {len(modules_to_test)} HRL modules imported successfully")
        return True


def test_hrl_basic_usage():
    """Test basic HRL usage patterns."""
    print("\n" + "=" * 60)
    print("HRL Basic Usage Test")
    print("=" * 60)

    try:
        # Test Option enum
        from hrl.option_definitions import Option, get_option_name
        print(f"\n✓ Option enum works: {Option.SEARCH} = '{get_option_name(Option.SEARCH)}'")

        # Test observation abstraction
        import numpy as np
        from hrl.observation_abstraction import abstract_observation
        dummy_obs = np.random.randn(26)
        abstract_state = abstract_observation(dummy_obs)
        print(f"✓ Observation abstraction works: 26D → {len(abstract_state)}D")

        # Test manager creation
        from hrl.manager import HierarchicalManager
        manager = HierarchicalManager(action_dim=6)
        print(f"✓ HierarchicalManager created: default_option={manager.default_option.name}")

        # Test action selection
        action, info = manager.select_action(dummy_obs)
        print(f"✓ Action selection works: action shape={action.shape}, current_option={info['hrl/option']}")

        print("\n✅ All basic usage tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Basic usage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all safety checks."""
    print("\nRunning HRL Safety Checks (Phase 1)\n")

    import_success = test_hrl_imports()
    usage_success = test_hrl_basic_usage()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Import Check: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"Usage Check:  {'✅ PASS' if usage_success else '❌ FAIL'}")

    if import_success and usage_success:
        print("\n🎉 HRL scaffolding is ready for Phase 2!")
        return 0
    else:
        print("\n⚠️  HRL scaffolding has issues - see errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
