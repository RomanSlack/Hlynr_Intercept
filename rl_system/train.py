"""
PPO Training Entry Point - Backward Compatible Wrapper

This script maintains backward compatibility with existing training workflows
while supporting future HRL training modes.

For new development, see:
- scripts/train_flat_ppo.py - Flat PPO training implementation
- scripts/train_hrl_full.py - HRL training pipeline (Phase 4+)
"""

import sys
import warnings
import argparse


def main():
    """
    Main entry point for training.

    Maintains full backward compatibility with existing workflows.
    Automatically detects mode and delegates to appropriate training script.
    """
    parser = argparse.ArgumentParser(
        description="Train PPO agent for missile interception",
        epilog="""
Examples:
  # Standard flat PPO training (default)
  python train.py --config config.yaml

  # HRL training (future - not yet implemented)
  python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--hrl",
        action="store_true",
        help="[DEPRECATED] Use scripts/train_hrl_full.py instead"
    )

    args = parser.parse_args()

    # Detect HRL mode
    if args.hrl or "--hrl" in sys.argv:
        warnings.warn(
            "The --hrl flag is deprecated. "
            "Please use scripts/train_hrl_full.py for HRL training instead.",
            DeprecationWarning,
            stacklevel=2
        )
        print("\n" + "=" * 60)
        print("HRL Training Not Yet Fully Implemented")
        print("=" * 60)
        print("\nPlease use one of the following:")
        print("  python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml")
        print("\nOr for testing individual components:")
        print("  python scripts/train_hrl_pretrain.py --agent search --config configs/hrl/search_specialist.yaml")
        print("  python scripts/train_hrl_selector.py --config configs/hrl/selector_config.yaml")
        print("\n" + "=" * 60 + "\n")
        return 1
    else:
        # Standard flat PPO training (backward compatible)
        from scripts.train_flat_ppo import train
        train(args.config)
        return 0


if __name__ == "__main__":
    sys.exit(main())
