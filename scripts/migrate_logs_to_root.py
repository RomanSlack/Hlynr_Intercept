#!/usr/bin/env python3
"""
Migration script to move existing logs into centralized logging structure.

This script safely moves existing log directories to the new centralized
logging layout under logs/ while preserving all timestamps and structure.
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from hlynr_bridge.paths import (
        get_log_dir, logs_training, logs_episodes, logs_tensorboard,
        get_log_layout_report
    )
except ImportError:
    print("ERROR: Could not import centralized paths module.")
    print("Please ensure you're running from the project root.")
    sys.exit(1)


class LogMigrator:
    """Handles migration of legacy log directories to centralized structure."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.migration_report = []
        self.errors = []
        
        # Get centralized log root
        self.log_root = get_log_dir()
        self.archives_dir = self.log_root / "archives"
        
        # Migration mappings: legacy_path -> (new_path, description)
        self.migrations = [
            # Training logs
            ("logs_radar17_good", logs_training("radar17_good"), "Training logs (radar17_good)"),
            ("logs_radar17_fixed", logs_training("radar17_fixed"), "Training logs (radar17_fixed)"),
            
            # Episode logs
            ("unity_episodes", logs_episodes(), "Unity episode logs (root)"),
            ("src/phase4_rl/unity_episodes", logs_episodes(), "Unity episode logs (phase4_rl)"),
            
            # Other potential log directories
            ("inference_episodes", logs_episodes(), "Inference episode logs"),
            ("runs", logs_episodes(), "Generic run logs"),
        ]
    
    def check_migrations_needed(self) -> List[Tuple[Path, Path, str]]:
        """Check which migrations are needed."""
        needed = []
        
        for legacy_str, new_path, description in self.migrations:
            legacy_path = Path(legacy_str)
            
            # Skip if legacy doesn't exist
            if not legacy_path.exists():
                continue
                
            # Skip if it's already a symlink (handled by symlink system)
            if legacy_path.is_symlink():
                print(f"⏭️  Skipping {legacy_path} (already symlinked)")
                continue
            
            # Skip if new path already exists and is not empty
            if new_path.exists() and any(new_path.iterdir()):
                print(f"⚠️  Skipping {legacy_path} -> {new_path} (target exists and is not empty)")
                continue
            
            needed.append((legacy_path, new_path, description))
        
        return needed
    
    def migrate_directory(self, source: Path, target: Path, description: str) -> bool:
        """Migrate a single directory."""
        try:
            print(f"📁 {description}")
            print(f"   {source} -> {target}")
            
            if self.dry_run:
                print(f"   [DRY RUN] Would move directory")
                self.migration_report.append({
                    "action": "move_directory",
                    "source": str(source),
                    "target": str(target),
                    "description": description,
                    "dry_run": True
                })
                return True
            
            # Ensure target parent exists
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the directory
            shutil.move(str(source), str(target))
            
            print(f"   ✅ Successfully moved")
            self.migration_report.append({
                "action": "move_directory",
                "source": str(source),
                "target": str(target),
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            return True
            
        except Exception as e:
            error_msg = f"Failed to move {source} -> {target}: {e}"
            print(f"   ❌ {error_msg}")
            self.errors.append(error_msg)
            self.migration_report.append({
                "action": "move_directory",
                "source": str(source),
                "target": str(target),
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            return False
    
    def create_migration_report(self) -> Path:
        """Create a migration report file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.archives_dir / f"migration_report_{timestamp}.json"
        
        # Ensure archives directory exists
        self.archives_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            "migration_timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "log_root": str(self.log_root),
            "migrations": self.migration_report,
            "errors": self.errors,
            "summary": {
                "total_migrations": len(self.migration_report),
                "successful": len([m for m in self.migration_report if m.get("success", False)]),
                "failed": len([m for m in self.migration_report if not m.get("success", True)]),
                "errors": len(self.errors)
            }
        }
        
        if not self.dry_run:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📋 Migration report saved: {report_file}")
        else:
            print(f"\n📋 [DRY RUN] Would save migration report: {report_file}")
        
        return report_file
    
    def run_migration(self) -> bool:
        """Run the complete migration process."""
        print("🔄 Hlynr Intercept Log Migration")
        print("=" * 50)
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will be moved")
            print()
        
        # Show current centralized logging layout
        layout = get_log_layout_report()
        print(f"📂 Centralized log root: {layout['log_root']}")
        print(f"   Environment variable: {layout['environment_var']}")
        print()
        
        # Check what needs migration
        migrations_needed = self.check_migrations_needed()
        
        if not migrations_needed:
            print("✨ No migrations needed - all legacy logs already moved or don't exist")
            return True
        
        print(f"📋 Found {len(migrations_needed)} directories to migrate:")
        for source, target, description in migrations_needed:
            print(f"   • {description}: {source} -> {target}")
        print()
        
        if not self.dry_run:
            response = input("Proceed with migration? [y/N]: ")
            if response.lower() not in ('y', 'yes'):
                print("Migration cancelled by user")
                return False
        
        # Perform migrations
        success_count = 0
        for source, target, description in migrations_needed:
            if self.migrate_directory(source, target, description):
                success_count += 1
        
        print()
        print(f"📊 Migration Summary:")
        print(f"   Total directories: {len(migrations_needed)}")
        print(f"   Successfully migrated: {success_count}")
        print(f"   Failed: {len(migrations_needed) - success_count}")
        print(f"   Errors: {len(self.errors)}")
        
        if self.errors:
            print("\n❌ Errors encountered:")
            for error in self.errors:
                print(f"   • {error}")
        
        # Create migration report
        self.create_migration_report()
        
        # Show final layout
        if not self.dry_run and success_count > 0:
            print(f"\n🎉 Migration complete! New centralized layout:")
            final_layout = get_log_layout_report()
            for name, path in final_layout['directories'].items():
                exists = "✓" if final_layout['directory_exists'][name] else "✗"
                print(f"   {exists} {name}: {path}")
        
        return len(self.errors) == 0


def main():
    """Main migration script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate legacy log directories to centralized logging structure"
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help="Show what would be migrated without actually moving files"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Skip confirmation prompts"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}")
    print()
    
    # Run migration
    migrator = LogMigrator(dry_run=args.dry_run)
    success = migrator.run_migration()
    
    if not success:
        print("\n💥 Migration completed with errors")
        sys.exit(1)
    else:
        print("\n✅ Migration completed successfully")


if __name__ == "__main__":
    main()