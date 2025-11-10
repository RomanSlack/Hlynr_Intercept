#!/usr/bin/env python3
"""
Checkpoint Migration Script

Scan for existing flat PPO checkpoints and migrate them to the new structure:
- Move to checkpoints/flat_ppo/
- Create backup if requested
- Verify all checkpoints load correctly

Usage:
    # Dry run (preview only)
    python migrate_checkpoints.py --dry-run

    # Migrate with backup
    python migrate_checkpoints.py --backup

    # Migrate without backup
    python migrate_checkpoints.py

    # Specify custom checkpoint directory
    python migrate_checkpoints.py --checkpoint-dir /path/to/checkpoints --backup

Example:
    # Preview what would be migrated
    python migrate_checkpoints.py --dry-run

    # Actually migrate with backup
    python migrate_checkpoints.py --backup
"""
import argparse
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import PPO


class CheckpointMigrator:
    """Migrate existing checkpoints to new HRL-aware structure."""

    def __init__(
        self,
        checkpoint_dir: Path,
        backup: bool = False,
        dry_run: bool = False,
    ):
        """
        Initialize checkpoint migrator.

        Args:
            checkpoint_dir: Root checkpoint directory
            backup: Create backup before migrating
            dry_run: Preview changes without executing
        """
        self.logger = logging.getLogger("CheckpointMigrator")
        self.logger.setLevel(logging.INFO)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.backup = backup
        self.dry_run = dry_run

        # Target directories
        self.flat_ppo_dir = self.checkpoint_dir / "flat_ppo"
        self.backup_dir = self.checkpoint_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.info(f"Dry run: {self.dry_run}")
        self.logger.info(f"Backup: {self.backup}")

    def scan_checkpoints(self) -> Dict[str, List[Path]]:
        """
        Scan checkpoint directory for existing checkpoints.

        Returns:
            Dictionary categorizing found checkpoints
        """
        self.logger.info(f"Scanning {self.checkpoint_dir}...")

        checkpoints = {
            'flat_ppo': [],  # Already in flat_ppo/
            'hrl': [],       # HRL checkpoints
            'legacy': [],    # Needs migration
            'other': [],     # Other files
        }

        if not self.checkpoint_dir.exists():
            self.logger.warning(f"Checkpoint directory does not exist: {self.checkpoint_dir}")
            return checkpoints

        # Scan for .zip files (SB3 checkpoint format)
        for item in self.checkpoint_dir.rglob("*.zip"):
            # Skip if already in structured directories
            if "flat_ppo" in item.parts or "hrl" in item.parts:
                if "flat_ppo" in item.parts:
                    checkpoints['flat_ppo'].append(item)
                elif "hrl" in item.parts:
                    checkpoints['hrl'].append(item)
            else:
                # Legacy checkpoint that needs migration
                checkpoints['legacy'].append(item)

        # Report findings
        self.logger.info(f"Found {len(checkpoints['flat_ppo'])} checkpoints already in flat_ppo/")
        self.logger.info(f"Found {len(checkpoints['hrl'])} HRL checkpoints")
        self.logger.info(f"Found {len(checkpoints['legacy'])} legacy checkpoints to migrate")

        return checkpoints

    def migrate(self) -> Dict[str, Any]:
        """
        Execute checkpoint migration.

        Returns:
            Migration summary dictionary
        """
        # Scan for checkpoints
        checkpoints = self.scan_checkpoints()

        if len(checkpoints['legacy']) == 0:
            self.logger.info("No legacy checkpoints to migrate")
            return {
                'success': True,
                'migrated': 0,
                'backed_up': 0,
                'verified': 0,
                'errors': [],
            }

        # Create backup if requested
        if self.backup and not self.dry_run:
            self._create_backup(checkpoints['legacy'])

        # Migrate legacy checkpoints
        migration_results = {
            'success': True,
            'migrated': 0,
            'backed_up': len(checkpoints['legacy']) if self.backup else 0,
            'verified': 0,
            'errors': [],
        }

        for checkpoint in checkpoints['legacy']:
            try:
                self._migrate_checkpoint(checkpoint)
                migration_results['migrated'] += 1

                # Verify checkpoint loads
                if not self.dry_run:
                    if self._verify_checkpoint(self._get_target_path(checkpoint)):
                        migration_results['verified'] += 1
                    else:
                        migration_results['errors'].append(f"Verification failed: {checkpoint}")

            except Exception as e:
                error_msg = f"Failed to migrate {checkpoint}: {e}"
                self.logger.error(error_msg)
                migration_results['errors'].append(error_msg)
                migration_results['success'] = False

        # Print summary
        self._print_summary(migration_results)

        return migration_results

    def _create_backup(self, checkpoints: List[Path]):
        """Create backup of checkpoints before migration."""
        self.logger.info(f"Creating backup in {self.backup_dir}")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        for checkpoint in checkpoints:
            # Preserve relative structure in backup
            rel_path = checkpoint.relative_to(self.checkpoint_dir)
            backup_path = self.backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(checkpoint, backup_path)
            self.logger.debug(f"Backed up: {checkpoint.name}")

        self.logger.info(f"Backup complete: {len(checkpoints)} files")

    def _get_target_path(self, checkpoint: Path) -> Path:
        """Determine target path for checkpoint migration."""
        # Preserve subdirectory structure if it exists
        rel_path = checkpoint.relative_to(self.checkpoint_dir)

        # If checkpoint is in a subdirectory (e.g., training_*/), preserve that
        if len(rel_path.parts) > 1:
            # Keep subdirectory structure under flat_ppo/
            target_path = self.flat_ppo_dir / rel_path
        else:
            # Direct child of checkpoint_dir, move to flat_ppo/
            target_path = self.flat_ppo_dir / checkpoint.name

        return target_path

    def _migrate_checkpoint(self, checkpoint: Path):
        """
        Migrate single checkpoint to new structure.

        Args:
            checkpoint: Path to checkpoint file
        """
        target_path = self._get_target_path(checkpoint)

        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would migrate: {checkpoint} -> {target_path}")
            return

        # Create target directory
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Move checkpoint
        self.logger.info(f"Migrating: {checkpoint.name} -> {target_path.relative_to(self.checkpoint_dir)}")
        shutil.move(str(checkpoint), str(target_path))

        # Also migrate associated files (vec_normalize.pkl, etc.)
        base_name = checkpoint.stem  # Without .zip
        for associated_file in checkpoint.parent.glob(f"{base_name}*"):
            if associated_file != checkpoint and associated_file.suffix != '.zip':
                target_associated = target_path.parent / associated_file.name
                if not self.dry_run:
                    shutil.move(str(associated_file), str(target_associated))
                    self.logger.debug(f"  Also migrated: {associated_file.name}")

    def _verify_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Verify that checkpoint loads correctly.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if checkpoint loads successfully
        """
        try:
            # Remove .zip extension for loading
            model_path = checkpoint_path.with_suffix('')
            model = PPO.load(model_path)
            self.logger.debug(f"✓ Verified: {checkpoint_path.name}")
            return True
        except Exception as e:
            self.logger.error(f"✗ Verification failed for {checkpoint_path.name}: {e}")
            return False

    def _print_summary(self, results: Dict[str, Any]):
        """Print migration summary."""
        print("\n" + "=" * 80)
        print("CHECKPOINT MIGRATION SUMMARY")
        print("=" * 80)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'EXECUTE'}")
        print(f"Checkpoint Directory: {self.checkpoint_dir}")
        print()

        print("MIGRATION RESULTS:")
        print(f"  Checkpoints Migrated:  {results['migrated']}")
        print(f"  Checkpoints Verified:  {results['verified']}")
        print(f"  Checkpoints Backed Up: {results['backed_up']}")
        print(f"  Errors:                {len(results['errors'])}")
        print()

        if results['errors']:
            print("ERRORS:")
            for error in results['errors']:
                print(f"  - {error}")
            print()

        print(f"Status: {'SUCCESS' if results['success'] else 'FAILED WITH ERRORS'}")

        if self.dry_run:
            print("\nThis was a DRY RUN. No changes were made.")
            print("Run without --dry-run to execute migration.")

        if self.backup and not self.dry_run:
            print(f"\nBackup created at: {self.backup_dir}")

        print("=" * 80)


def main():
    """Main migration script."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Migrate existing checkpoints to new HRL-aware structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory to migrate (default: checkpoints)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before migration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing checkpoints without migration",
    )

    args = parser.parse_args()

    # Convert to absolute path
    checkpoint_dir = Path(args.checkpoint_dir).resolve()

    # Create migrator
    migrator = CheckpointMigrator(
        checkpoint_dir=checkpoint_dir,
        backup=args.backup,
        dry_run=args.dry_run,
    )

    if args.verify_only:
        # Only verify existing checkpoints
        print("Verifying existing checkpoints...")
        checkpoints = migrator.scan_checkpoints()

        all_checkpoints = (
            checkpoints['flat_ppo'] +
            checkpoints['hrl'] +
            checkpoints['legacy']
        )

        verified = 0
        failed = 0

        for checkpoint in all_checkpoints:
            if migrator._verify_checkpoint(checkpoint):
                verified += 1
            else:
                failed += 1

        print(f"\nVerification complete: {verified} OK, {failed} FAILED")
        return 0 if failed == 0 else 1

    # Run migration
    results = migrator.migrate()

    return 0 if results['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
