#!/usr/bin/env python3
"""Debug config override logic."""

from config import get_config

# Test the config override logic that the training script uses
config_loader = get_config(None)  # Use default config
print("Original config:")
print("episode_logging enabled:", config_loader._config.get('episode_logging', {}).get('enabled', False))

# Override like the training script does
if 'episode_logging' not in config_loader._config:
    config_loader._config['episode_logging'] = {}
config_loader._config['episode_logging']['enabled'] = True
config_loader._config['episode_logging']['output_dir'] = 'my_runs'

print("\nAfter override:")
print("episode_logging enabled:", config_loader._config.get('episode_logging', {}).get('enabled', False))
print("episode_logging output_dir:", config_loader._config.get('episode_logging', {}).get('output_dir', 'default'))
print("episode_logging log_during_eval:", config_loader._config.get('episode_logging', {}).get('log_during_eval', True))

print("\nFull episode_logging config:")
print(config_loader._config.get('episode_logging', {}))