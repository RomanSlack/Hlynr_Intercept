#!/usr/bin/env python3
"""
HRL Full Training Pipeline Orchestrator

Orchestrates the complete HRL training pipeline:
1. Pre-train specialists (search, track, terminal)
2. Train selector with frozen specialists
3. Optional: Joint fine-tuning

This script manages the entire workflow and can resume from any stage.

Usage:
    # Full pipeline from scratch
    python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml

    # Resume from specific stage
    python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml --resume-stage selector

    # Skip pretraining (use existing specialists)
    python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml \\
        --specialist-dir checkpoints/hrl/specialists/ --skip-pretrain
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load parent config if specified
    if 'parent_config' in config:
        parent_path = config_path.parent / config['parent_config']
        if parent_path.exists():
            with open(parent_path, 'r') as f:
                parent_config = yaml.safe_load(f)

            # Merge configs
            merged = parent_config.copy()
            for key in config:
                if key != 'parent_config':
                    if isinstance(config[key], dict) and key in merged:
                        merged[key].update(config[key])
                    else:
                        merged[key] = config[key]
            config = merged

    return config


def save_pipeline_metadata(
    output_dir: Path,
    stage: str,
    config: Dict[str, Any],
    specialist_paths: Optional[Dict[str, str]] = None,
    selector_path: Optional[str] = None,
):
    """
    Save pipeline training metadata for reproducibility.

    Args:
        output_dir: Directory to save metadata
        stage: Current training stage
        config: Configuration dictionary
        specialist_paths: Paths to specialist checkpoints
        selector_path: Path to selector checkpoint
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'stage': stage,
        'config_file': str(config.get('_config_path', 'unknown')),
        'specialists': specialist_paths or {},
        'selector': selector_path,
        'training_config': {
            'specialist_steps': config.get('training', {}).get('specialist_timesteps', 100000),
            'selector_steps': config.get('training', {}).get('selector_timesteps', 50000),
        }
    }

    metadata_file = output_dir / 'pipeline_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved pipeline metadata to {metadata_file}")


def run_command(cmd: List[str], description: str) -> int:
    """
    Run a command and stream output.

    Args:
        cmd: Command to run as list of strings
        description: Human-readable description

    Returns:
        Return code
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode


def pretrain_specialists(
    config: Dict[str, Any],
    output_dir: Path,
    specialists: List[str] = ["search", "track", "terminal"],
    steps_override: Optional[int] = None,
) -> Dict[str, str]:
    """
    Pre-train all specialist policies.

    Args:
        config: Configuration dictionary
        output_dir: Base output directory
        specialists: List of specialists to train
        steps_override: Override training steps

    Returns:
        Dictionary mapping specialist name to checkpoint path
    """
    print("\n" + "="*60)
    print("STAGE 1: Pre-Training Specialists")
    print("="*60)

    specialist_paths = {}
    config_dir = Path(config.get('_config_path', 'configs/hrl/hrl_curriculum.yaml')).parent

    for specialist in specialists:
        print(f"\n>>> Training {specialist} specialist...")

        # Construct command
        specialist_config = config_dir / f"{specialist}_specialist.yaml"
        if not specialist_config.exists():
            print(f"⚠️  Config not found: {specialist_config}, using base config")
            specialist_config = Path(config.get('_config_path', 'configs/hrl/hrl_curriculum.yaml'))

        checkpoint_dir = output_dir / "specialists" / specialist

        cmd = [
            sys.executable,
            "scripts/train_hrl_pretrain.py",
            "--agent", specialist,
            "--config", str(specialist_config),
            "--checkpoint-dir", str(checkpoint_dir),
        ]

        if steps_override:
            cmd.extend(["--steps", str(steps_override)])

        # Run training
        returncode = run_command(cmd, f"{specialist} specialist pre-training")

        if returncode != 0:
            raise RuntimeError(f"Failed to train {specialist} specialist")

        # Find the checkpoint
        final_model = checkpoint_dir / "final" / "model.zip"
        if not final_model.exists():
            raise FileNotFoundError(f"Specialist checkpoint not found: {final_model}")

        specialist_paths[specialist] = str(checkpoint_dir)
        print(f"✓ {specialist} specialist trained: {checkpoint_dir}")

    print("\n✅ All specialists pre-trained successfully")
    return specialist_paths


def train_selector(
    config: Dict[str, Any],
    output_dir: Path,
    specialist_dir: Path,
    resume_path: Optional[str] = None,
    steps_override: Optional[int] = None,
) -> str:
    """
    Train selector policy with frozen specialists.

    Args:
        config: Configuration dictionary
        output_dir: Base output directory
        specialist_dir: Directory containing specialist checkpoints
        resume_path: Path to resume from (optional)
        steps_override: Override training steps

    Returns:
        Path to trained selector checkpoint
    """
    print("\n" + "="*60)
    print("STAGE 2: Training Selector Policy")
    print("="*60)

    # Construct command
    selector_config = Path(config.get('_config_path', 'configs/hrl/hrl_curriculum.yaml')).parent / "selector_config.yaml"
    if not selector_config.exists():
        print(f"⚠️  Selector config not found: {selector_config}, using base config")
        selector_config = Path(config.get('_config_path', 'configs/hrl/hrl_curriculum.yaml'))

    checkpoint_dir = output_dir / "selector"

    cmd = [
        sys.executable,
        "scripts/train_hrl_selector.py",
        "--config", str(selector_config),
        "--specialist-dir", str(specialist_dir),
        "--checkpoint-dir", str(checkpoint_dir),
    ]

    if resume_path:
        cmd.extend(["--resume", resume_path])

    if steps_override:
        cmd.extend(["--steps", str(steps_override)])

    # Run training
    returncode = run_command(cmd, "Selector policy training")

    if returncode != 0:
        raise RuntimeError("Failed to train selector policy")

    # Find the checkpoint
    final_model = checkpoint_dir / "final" / "model.zip"
    if not final_model.exists():
        raise FileNotFoundError(f"Selector checkpoint not found: {final_model}")

    print(f"✓ Selector trained: {checkpoint_dir}")
    return str(checkpoint_dir)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HRL Full Training Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline from scratch
  python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml

  # Resume from selector stage
  python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml --resume-stage selector

  # Use existing specialists
  python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml \\
      --specialist-dir checkpoints/hrl/specialists/ --skip-pretrain

  # Override training steps
  python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml \\
      --specialist-steps 50000 --selector-steps 25000
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to HRL curriculum config YAML"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory for all checkpoints (default: checkpoints/hrl/pipeline_TIMESTAMP)"
    )

    parser.add_argument(
        "--resume-stage",
        type=str,
        choices=["pretrain", "selector", "finetune"],
        default=None,
        help="Resume from specific stage"
    )

    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip specialist pre-training (use existing specialists)"
    )

    parser.add_argument(
        "--specialist-dir",
        type=str,
        default=None,
        help="Directory containing pre-trained specialists (required if --skip-pretrain)"
    )

    parser.add_argument(
        "--specialist-steps",
        type=int,
        default=None,
        help="Override specialist training steps"
    )

    parser.add_argument(
        "--selector-steps",
        type=int,
        default=None,
        help="Override selector training steps"
    )

    parser.add_argument(
        "--enable-finetune",
        action="store_true",
        help="Enable joint fine-tuning stage (experimental)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("\n" + "="*60)
    print("HRL FULL TRAINING PIPELINE")
    print("="*60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load config
    try:
        config = load_config(args.config)
        config['_config_path'] = args.config  # Store for reference
    except Exception as e:
        print(f"\n❌ Error loading config: {e}")
        return 1

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("checkpoints/hrl") / f"pipeline_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Track progress
    specialist_paths = {}
    selector_path = None

    try:
        # STAGE 1: Pre-train specialists
        if not args.skip_pretrain and (not args.resume_stage or args.resume_stage == "pretrain"):
            specialist_paths = pretrain_specialists(
                config=config,
                output_dir=output_dir,
                steps_override=args.specialist_steps
            )

            # Save metadata after specialist training
            save_pipeline_metadata(
                output_dir=output_dir,
                stage="pretrain_complete",
                config=config,
                specialist_paths=specialist_paths,
            )
        else:
            if args.specialist_dir:
                specialist_dir = Path(args.specialist_dir)
                if not specialist_dir.exists():
                    raise FileNotFoundError(f"Specialist directory not found: {specialist_dir}")

                # Map specialists
                for specialist in ["search", "track", "terminal"]:
                    spec_path = specialist_dir / specialist
                    if not spec_path.exists():
                        raise FileNotFoundError(f"Specialist not found: {spec_path}")
                    specialist_paths[specialist] = str(spec_path)

                print(f"\n✓ Using existing specialists from: {specialist_dir}")
            elif args.resume_stage:
                # Try to load from previous pipeline run
                specialist_dir = output_dir / "specialists"
                if specialist_dir.exists():
                    for specialist in ["search", "track", "terminal"]:
                        specialist_paths[specialist] = str(specialist_dir / specialist)
                    print(f"\n✓ Resuming with specialists from: {specialist_dir}")
                else:
                    raise ValueError("Cannot resume: no specialist directory specified and none found in output dir")
            else:
                raise ValueError("Must provide --specialist-dir if skipping pre-training")

        # STAGE 2: Train selector
        if not args.resume_stage or args.resume_stage in ["pretrain", "selector"]:
            specialist_dir = output_dir / "specialists" if not args.specialist_dir else Path(args.specialist_dir)

            # Copy or link specialists to output dir if needed
            if args.specialist_dir and specialist_dir != output_dir / "specialists":
                (output_dir / "specialists").mkdir(exist_ok=True)
                print(f"\nℹ️  Note: Using specialists from {args.specialist_dir}")
                specialist_dir = Path(args.specialist_dir)

            selector_path = train_selector(
                config=config,
                output_dir=output_dir,
                specialist_dir=specialist_dir,
                steps_override=args.selector_steps
            )

            # Save metadata after selector training
            save_pipeline_metadata(
                output_dir=output_dir,
                stage="selector_complete",
                config=config,
                specialist_paths=specialist_paths,
                selector_path=selector_path,
            )

        # STAGE 3: Joint fine-tuning (optional, experimental)
        if args.enable_finetune:
            print("\n" + "="*60)
            print("STAGE 3: Joint Fine-Tuning (Experimental)")
            print("="*60)
            print("\n⚠️  Joint fine-tuning not yet implemented")
            print("This would train selector and specialists together for final optimization")

        # Final summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"\nOutput directory: {output_dir}")
        print("\nCheckpoints:")
        if specialist_paths:
            print("  Specialists:")
            for name, path in specialist_paths.items():
                print(f"    - {name}: {path}")
        if selector_path:
            print(f"  Selector: {selector_path}")

        print(f"\nMetadata: {output_dir / 'pipeline_metadata.json'}")
        print("\n✅ HRL training pipeline completed successfully")

        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

        # Save failure metadata
        try:
            save_pipeline_metadata(
                output_dir=output_dir,
                stage="failed",
                config=config,
                specialist_paths=specialist_paths,
                selector_path=selector_path,
            )
        except:
            pass

        return 1


if __name__ == "__main__":
    sys.exit(main())
