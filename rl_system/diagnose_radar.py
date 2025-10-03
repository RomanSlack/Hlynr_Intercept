"""
Diagnostic script to analyze radar detection issues.
Runs a few episodes and logs detailed radar detection statistics.
"""

import numpy as np
import yaml
from environment import InterceptEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pathlib import Path

def diagnose_radar_issues(model_path: str = "checkpoints/best", config_path: str = "config.yaml"):
    """Run diagnostic on radar detection behavior."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create environment
    env = InterceptEnvironment(config['environment'])
    vec_env = DummyVecEnv([lambda: env])

    # Try to load VecNormalize if exists
    model_path = Path(model_path)
    vec_normalize_path = model_path / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        vec_env = VecNormalize.load(str(vec_normalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"✓ Loaded VecNormalize")

    # Load model
    if model_path.is_dir():
        model_file = model_path / "best_model.zip"
    else:
        model_file = model_path

    model = PPO.load(str(model_file), env=vec_env)
    print(f"✓ Loaded model from {model_file}\n")

    # Run diagnostic episodes
    n_episodes = 5

    for ep in range(n_episodes):
        print(f"{'='*80}")
        print(f"EPISODE {ep+1}/{n_episodes}")
        print(f"{'='*80}")

        obs = vec_env.reset()
        done = False
        step = 0

        # Detection statistics
        onboard_detections = 0
        ground_detections = 0
        total_steps = 0

        # Initial conditions
        int_state = env.interceptor_state
        mis_state = env.missile_state
        initial_range = np.linalg.norm(int_state['position'] - mis_state['position'])

        print(f"\nInitial Conditions:")
        print(f"  Interceptor pos: {int_state['position']}")
        print(f"  Missile pos: {mis_state['position']}")
        print(f"  Initial range: {initial_range:.1f}m")
        print(f"  Interceptor orientation (quat): {int_state['orientation']}")

        # Check forward vector
        from core import get_forward_vector
        int_forward = get_forward_vector(int_state['orientation'])
        to_missile = (mis_state['position'] - int_state['position']) / initial_range
        beam_angle_deg = np.degrees(np.arccos(np.clip(np.dot(int_forward, to_missile), -1, 1)))

        print(f"  Interceptor forward vector: {int_forward}")
        print(f"  Direction to missile: {to_missile}")
        print(f"  Initial beam angle: {beam_angle_deg:.1f}° (beam width: 60°)")
        print(f"  Missile in beam? {'YES' if beam_angle_deg < 60 else 'NO'}")

        # Sample first few steps
        print(f"\nFirst 10 Steps:")
        print(f"{'Step':>4} | {'Range':>8} | {'Onboard':>8} | {'Ground':>8} | {'Beam°':>6} | {'DataLink':>8} | {'Fusion':>8}")
        print(f"{'-'*80}")

        while not done and step < 500:
            # Get action
            action, _ = model.predict(obs, deterministic=True)

            # Extract observation values (26D)
            # [14] = onboard radar quality
            # [23] = ground radar quality
            # [24] = datalink quality
            # [25] = fusion confidence

            onboard_quality = obs[0][14]
            ground_quality = obs[0][23]
            datalink_quality = obs[0][24]
            fusion_confidence = obs[0][25]

            # Track detections
            if onboard_quality > 0.1:
                onboard_detections += 1
            if ground_quality > 0.1:
                ground_detections += 1
            total_steps += 1

            # Get current range and beam angle
            int_state = env.interceptor_state
            mis_state = env.missile_state
            current_range = np.linalg.norm(int_state['position'] - mis_state['position'])

            int_forward = get_forward_vector(int_state['orientation'])
            rel_pos = mis_state['position'] - int_state['position']
            if current_range > 1e-6:
                to_missile = rel_pos / current_range
                beam_angle_deg = np.degrees(np.arccos(np.clip(np.dot(int_forward, to_missile), -1, 1)))
            else:
                beam_angle_deg = 0.0

            # Print first 10 steps
            if step < 10:
                print(f"{step:4d} | {current_range:8.1f} | {onboard_quality:8.3f} | {ground_quality:8.3f} | {beam_angle_deg:6.1f} | {datalink_quality:8.3f} | {fusion_confidence:8.3f}")

            # Step
            obs, reward, done, info = vec_env.step(action)
            step += 1

        # Episode summary
        onboard_rate = (onboard_detections / total_steps) * 100 if total_steps > 0 else 0
        ground_rate = (ground_detections / total_steps) * 100 if total_steps > 0 else 0

        print(f"\nEpisode Summary:")
        print(f"  Total steps: {total_steps}")
        print(f"  Onboard detections: {onboard_detections}/{total_steps} ({onboard_rate:.1f}%)")
        print(f"  Ground detections: {ground_detections}/{total_steps} ({ground_rate:.1f}%)")
        print(f"  Final distance: {info[0].get('distance', 0):.1f}m")
        print(f"  Intercepted: {info[0].get('intercepted', False)}")
        print()

if __name__ == "__main__":
    diagnose_radar_issues()
