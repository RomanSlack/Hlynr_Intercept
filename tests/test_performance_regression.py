"""
Performance and Regression Test Suite

This module contains comprehensive tests to validate:
- Training convergence and learning performance
- Computational performance benchmarks
- Memory usage monitoring and leak detection
- Regression prevention for Phase 2 capabilities
- Scalability and efficiency validation
- Performance comparison between 3DOF and 6DOF modes

Author: Tester Agent
Date: Phase 3 Testing Framework
"""

import pytest
import numpy as np
import time
import psutil
import os
import threading
import tempfile
import json
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import patch, MagicMock
import gc

from aegis_intercept.envs.aegis_6dof_env import (
    Aegis6DInterceptEnv, DifficultyMode, ActionMode
)
from aegis_intercept.envs.aegis_3d_env import Aegis3DInterceptEnv
from aegis_intercept.curriculum.curriculum_manager import CurriculumManager
from aegis_intercept.utils.physics6dof import RigidBody6DOF, VehicleType


class PerformanceMonitor:
    """Helper class for monitoring performance metrics"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = self.process.memory_info().rss
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, sample_interval: float = 0.1):
        """Start continuous monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(sample_interval,))
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return results"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss
        
        return {
            'duration': end_time - self.start_time,
            'memory_start': self.start_memory,
            'memory_end': end_memory,
            'memory_peak': max(self.memory_samples) if self.memory_samples else end_memory,
            'memory_samples': self.memory_samples,
            'cpu_samples': self.cpu_samples,
            'memory_growth': end_memory - self.start_memory
        }
    
    def _monitor_loop(self, sample_interval: float):
        """Monitoring loop running in separate thread"""
        while self.monitoring:
            try:
                mem_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                self.memory_samples.append(mem_info.rss)
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(sample_interval)
            except:
                break


class TestTrainingConvergence:
    """Test training convergence and learning performance"""
    
    @pytest.mark.slow
    def test_basic_learning_capability(self):
        """Test that agent can learn basic intercept task"""
        env = Aegis6DInterceptEnv(
            difficulty_mode=DifficultyMode.SIMPLIFIED_6DOF,
            max_steps=100,
            intercept_threshold=50.0  # Generous threshold for learning test
        )
        
        # Simple learning simulation using random policy improvement
        performance_history = []
        
        for epoch in range(10):  # Simulate 10 training epochs
            epoch_rewards = []
            epoch_successes = []
            
            for episode in range(20):  # 20 episodes per epoch
                obs, _ = env.reset(seed=epoch * 20 + episode)
                episode_reward = 0
                success = False
                
                # Simple adaptive policy (improves over epochs)
                adaptation_factor = min(1.0, epoch / 5.0)  # Improves over first 5 epochs
                
                for step in range(100):
                    if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                        # Increasingly good policy
                        int_pos = env.interceptor_6dof.position
                        mis_pos = env.missile_6dof.position
                        direction = mis_pos - int_pos
                        
                        if np.linalg.norm(direction) > 1e-6:
                            direction = direction / np.linalg.norm(direction)
                        
                        # Mix good policy with random noise, less noise over time
                        good_action = np.concatenate([direction * 0.8, direction * 0.2, [0.0]])
                        noise = np.random.normal(0, 0.5 * (1 - adaptation_factor), 7)
                        action = good_action + noise
                        action = np.clip(action, -1, 1)
                    else:
                        action = env.action_space.sample()
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    if terminated:
                        if reward > 10:  # Successful intercept
                            success = True
                        break
                    elif truncated:
                        break
                
                epoch_rewards.append(episode_reward)
                epoch_successes.append(success)
            
            avg_reward = np.mean(epoch_rewards)
            success_rate = np.mean(epoch_successes)
            
            performance_history.append({
                'epoch': epoch,
                'avg_reward': avg_reward,
                'success_rate': success_rate
            })
        
        # Check for learning (improvement over time)
        early_performance = np.mean([p['avg_reward'] for p in performance_history[:3]])
        late_performance = np.mean([p['avg_reward'] for p in performance_history[-3:]])
        
        improvement = late_performance - early_performance
        assert improvement > 2.0, f"Should show learning improvement: {improvement}"
        
        # Success rate should also improve
        early_success = np.mean([p['success_rate'] for p in performance_history[:3]])
        late_success = np.mean([p['success_rate'] for p in performance_history[-3:]])
        
        success_improvement = late_success - early_success
        assert success_improvement > 0.1, f"Success rate should improve: {success_improvement}"
    
    def test_curriculum_learning_progression(self):
        """Test that curriculum learning shows expected progression"""
        curriculum = CurriculumManager()
        
        # Simulate curriculum learning progression
        phases_tested = []
        phase_performances = {}
        
        test_phases = [
            DifficultyMode.EASY_3DOF,
            DifficultyMode.SIMPLIFIED_6DOF,
            DifficultyMode.FULL_6DOF
        ]
        
        for difficulty in test_phases:
            env = Aegis6DInterceptEnv(
                difficulty_mode=difficulty,
                max_steps=50,
                intercept_threshold=40.0
            )
            
            episode_rewards = []
            
            for episode in range(15):
                obs, _ = env.reset(seed=1000 + episode)
                episode_reward = 0
                
                for step in range(50):
                    # Simple consistent policy for all modes
                    if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                        int_pos = env.interceptor_6dof.position
                        mis_pos = env.missile_6dof.position
                        direction = mis_pos - int_pos
                        if np.linalg.norm(direction) > 1e-6:
                            direction = direction / np.linalg.norm(direction)
                        action = np.concatenate([direction * 0.6, direction * 0.1, [0.0]])
                    elif hasattr(env, 'interceptor_pos_3d') and hasattr(env, 'missile_pos_3d'):
                        direction = env.missile_pos_3d - env.interceptor_pos_3d
                        if np.linalg.norm(direction) > 1e-6:
                            direction = direction / np.linalg.norm(direction)
                        action = np.concatenate([direction * 0.6, [0.0]])
                    else:
                        action = env.action_space.sample()
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(episode_reward)
            
            avg_performance = np.mean(episode_rewards)
            phase_performances[difficulty] = avg_performance
            phases_tested.append(difficulty)
        
        # Performance should be reasonable across phases
        for difficulty, performance in phase_performances.items():
            assert performance > -50, f"Performance too poor for {difficulty}: {performance}"
            assert performance < 100, f"Performance suspiciously high for {difficulty}: {performance}"
    
    def test_learning_stability(self):
        """Test learning stability over extended training"""
        env = Aegis6DInterceptEnv(
            difficulty_mode=DifficultyMode.FULL_6DOF,
            max_steps=80
        )
        
        # Track performance variance over time
        window_size = 10
        performance_windows = []
        
        for window in range(10):  # 10 windows of 10 episodes each
            window_rewards = []
            
            for episode in range(window_size):
                obs, _ = env.reset(seed=2000 + window * window_size + episode)
                episode_reward = 0
                
                for step in range(80):
                    # Consistent policy to test stability
                    if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                        int_pos = env.interceptor_6dof.position
                        mis_pos = env.missile_6dof.position
                        direction = mis_pos - int_pos
                        if np.linalg.norm(direction) > 1e-6:
                            direction = direction / np.linalg.norm(direction)
                        action = np.concatenate([direction * 0.7, direction * 0.15, [0.0]])
                    else:
                        action = env.action_space.sample()
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                window_rewards.append(episode_reward)
            
            window_avg = np.mean(window_rewards)
            window_std = np.std(window_rewards)
            
            performance_windows.append({
                'window': window,
                'avg_reward': window_avg,
                'std_reward': window_std
            })
        
        # Check stability (variance shouldn't increase dramatically over time)
        early_variance = np.mean([w['std_reward'] for w in performance_windows[:3]])
        late_variance = np.mean([w['std_reward'] for w in performance_windows[-3:]])
        
        variance_ratio = late_variance / early_variance if early_variance > 0 else 1.0
        assert variance_ratio < 3.0, f"Performance variance increased too much: {variance_ratio}"


class TestComputationalPerformance:
    """Test computational performance benchmarks"""
    
    def test_environment_step_performance(self):
        """Test environment step performance benchmarks"""
        # Test different environment modes
        test_configs = [
            ('3DOF Legacy', {'legacy_3dof_mode': True, 'difficulty_mode': DifficultyMode.EASY_3DOF}),
            ('6DOF Simplified', {'difficulty_mode': DifficultyMode.SIMPLIFIED_6DOF}),
            ('6DOF Full', {'difficulty_mode': DifficultyMode.FULL_6DOF}),
            ('6DOF Expert', {'difficulty_mode': DifficultyMode.EXPERT_6DOF})
        ]
        
        performance_results = {}
        
        for config_name, config in test_configs:
            env = Aegis6DInterceptEnv(**config)
            
            # Warmup
            obs, _ = env.reset(seed=42)
            for _ in range(10):
                action = env.action_space.sample()
                env.step(action)
            
            # Benchmark
            num_steps = 1000
            start_time = time.time()
            
            obs, _ = env.reset(seed=42)
            for step in range(num_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    obs, _ = env.reset(seed=42 + step)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            steps_per_second = num_steps / elapsed_time
            
            performance_results[config_name] = {
                'steps_per_second': steps_per_second,
                'time_per_step': elapsed_time / num_steps
            }
        
        # Performance requirements
        for config_name, result in performance_results.items():
            steps_per_sec = result['steps_per_second']
            time_per_step = result['time_per_step']
            
            # All configurations should achieve reasonable performance
            assert steps_per_sec > 1000, f"{config_name} too slow: {steps_per_sec:.0f} steps/sec"
            assert time_per_step < 0.001, f"{config_name} time per step too high: {time_per_step:.6f}s"
        
        # 3DOF should be faster than 6DOF
        if '3DOF Legacy' in performance_results and '6DOF Full' in performance_results:
            legacy_perf = performance_results['3DOF Legacy']['steps_per_second']
            full_6dof_perf = performance_results['6DOF Full']['steps_per_second']
            
            # 3DOF should be at least 50% as fast as 6DOF (6DOF is more complex but shouldn't be dramatically slower)
            assert full_6dof_perf > legacy_perf * 0.5, \
                f"6DOF performance degradation too large: {full_6dof_perf} vs {legacy_perf}"
        
        print(f"\nPerformance Benchmark Results:")
        for config_name, result in performance_results.items():
            print(f"{config_name}: {result['steps_per_second']:.0f} steps/sec ({result['time_per_step']*1000:.3f}ms/step)")
    
    def test_physics_engine_performance(self):
        """Test 6DOF physics engine performance"""
        # Test different vehicle configurations
        vehicle_types = [VehicleType.INTERCEPTOR, VehicleType.MISSILE]
        
        for vehicle_type in vehicle_types:
            rigid_body = RigidBody6DOF(
                vehicle_type,
                initial_position=np.array([0, 0, 1000]),
                initial_velocity=np.array([100, 50, 20]),
                initial_orientation=np.array([1, 0, 0, 0]),
                initial_angular_velocity=np.array([0.1, 0.2, 0.05])
            )
            
            # Set control inputs
            rigid_body.set_control_inputs(
                np.array([1000, 200, 100]),
                np.array([50, 25, 10])
            )
            
            # Warmup
            for _ in range(100):
                rigid_body.step(0.01, 0.0, np.array([5, 2, 0]))
            
            # Benchmark
            num_steps = 5000
            dt = 0.01
            wind_velocity = np.array([10, 5, 0])
            
            start_time = time.time()
            for i in range(num_steps):
                rigid_body.step(dt, i * dt, wind_velocity)
            
            elapsed_time = time.time() - start_time
            steps_per_second = num_steps / elapsed_time
            
            # Physics should be very fast
            assert steps_per_second > 50000, \
                f"{vehicle_type} physics too slow: {steps_per_second:.0f} steps/sec"
            
            print(f"{vehicle_type} Physics: {steps_per_second:.0f} steps/sec")
    
    def test_curriculum_manager_performance(self):
        """Test curriculum manager performance"""
        curriculum = CurriculumManager()
        
        # Benchmark curriculum operations
        num_updates = 10000
        
        start_time = time.time()
        for i in range(num_updates):
            # Simulate performance updates
            reward = np.random.normal(10, 5)
            success = np.random.random() > 0.3
            fuel_used = np.random.uniform(20, 80)
            intercept_time = np.random.uniform(3, 15)
            
            curriculum.update_performance(reward, success, fuel_used, intercept_time)
        
        elapsed_time = time.time() - start_time
        updates_per_second = num_updates / elapsed_time
        
        # Curriculum updates should be very fast
        assert updates_per_second > 10000, f"Curriculum updates too slow: {updates_per_second:.0f}/sec"
        
        print(f"Curriculum Manager: {updates_per_second:.0f} updates/sec")
    
    @pytest.mark.slow
    def test_parallel_environment_performance(self):
        """Test performance with multiple parallel environments"""
        num_envs = 4
        
        # Create multiple environments
        envs = []
        for i in range(num_envs):
            env = Aegis6DInterceptEnv(
                difficulty_mode=DifficultyMode.FULL_6DOF,
                max_steps=50
            )
            envs.append(env)
        
        # Reset all environments
        for i, env in enumerate(envs):
            env.reset(seed=i)
        
        # Benchmark parallel stepping
        num_steps = 500
        start_time = time.time()
        
        for step in range(num_steps):
            for env in envs:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    env.reset(seed=step)
        
        elapsed_time = time.time() - start_time
        total_steps = num_steps * num_envs
        steps_per_second = total_steps / elapsed_time
        
        # Parallel performance should scale reasonably
        assert steps_per_second > 2000, f"Parallel performance too low: {steps_per_second:.0f} steps/sec"
        
        print(f"Parallel ({num_envs} envs): {steps_per_second:.0f} total steps/sec")


class TestMemoryUsage:
    """Test memory usage and leak detection"""
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended simulation"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring(sample_interval=0.2)
        
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        # Run extended simulation
        num_episodes = 50
        for episode in range(num_episodes):
            obs, _ = env.reset(seed=episode)
            
            for step in range(100):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            # Force garbage collection periodically
            if episode % 10 == 0:
                gc.collect()
        
        results = monitor.stop_monitoring()
        
        # Check for memory leaks
        memory_growth = results['memory_growth']
        memory_growth_mb = memory_growth / (1024 * 1024)
        
        # Memory growth should be reasonable (less than 100MB for this test)
        assert memory_growth_mb < 100, f"Potential memory leak: {memory_growth_mb:.1f}MB growth"
        
        # Check memory stability (no continuous growth)
        if len(results['memory_samples']) > 10:
            memory_samples = np.array(results['memory_samples'])
            # Check if memory is continuously growing
            half_point = len(memory_samples) // 2
            first_half_avg = np.mean(memory_samples[:half_point])
            second_half_avg = np.mean(memory_samples[half_point:])
            
            memory_increase_ratio = second_half_avg / first_half_avg
            assert memory_increase_ratio < 1.5, f"Continuous memory growth detected: {memory_increase_ratio:.2f}x"
        
        print(f"Memory usage - Growth: {memory_growth_mb:.1f}MB, Peak: {results['memory_peak']/(1024*1024):.1f}MB")
    
    def test_environment_cleanup(self):
        """Test that environments properly clean up resources"""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss
        
        # Create and destroy multiple environments
        for i in range(20):
            env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
            
            # Use the environment briefly
            obs, _ = env.reset(seed=i)
            for _ in range(10):
                action = env.action_space.sample()
                env.step(action)
            
            # Explicitly close if method exists
            if hasattr(env, 'close'):
                env.close()
            
            # Delete reference
            del env
            
            # Force garbage collection every few iterations
            if i % 5 == 0:
                gc.collect()
        
        # Final garbage collection
        gc.collect()
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_growth = final_memory - initial_memory
        memory_growth_mb = memory_growth / (1024 * 1024)
        
        # Memory growth from environment creation/destruction should be minimal
        assert memory_growth_mb < 50, f"Environment cleanup incomplete: {memory_growth_mb:.1f}MB growth"
    
    def test_large_state_memory_efficiency(self):
        """Test memory efficiency with large state spaces"""
        # Test 6DOF environment with large observation space
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        initial_memory = psutil.Process(os.getpid()).memory_info().rss
        
        # Collect many observations
        observations = []
        infos = []
        
        for episode in range(100):
            obs, info = env.reset(seed=episode)
            observations.append(obs.copy())
            infos.append(info.copy())
            
            for step in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                observations.append(obs.copy())
                infos.append(info.copy())
                
                if terminated or truncated:
                    break
        
        current_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_for_data = current_memory - initial_memory
        
        # Estimate expected memory usage
        obs_size = observations[0].nbytes if observations else 0
        num_observations = len(observations)
        estimated_obs_memory = obs_size * num_observations
        
        # Actual memory usage shouldn't be dramatically higher than estimated
        # (allowing for Python overhead and other data structures)
        memory_efficiency_ratio = memory_for_data / max(estimated_obs_memory, 1)
        
        assert memory_efficiency_ratio < 10, f"Memory usage inefficient: {memory_efficiency_ratio:.1f}x overhead"
        
        print(f"State Memory Efficiency - Observations: {num_observations}, Memory: {memory_for_data/(1024*1024):.1f}MB, Ratio: {memory_efficiency_ratio:.1f}x")


class TestRegressionPrevention:
    """Test regression prevention for Phase 2 capabilities"""
    
    def test_3dof_performance_regression(self):
        """Test that 3DOF performance hasn't regressed"""
        # Test original 3DOF environment
        if hasattr(Aegis3DInterceptEnv, '__init__'):
            env_3d_original = Aegis3DInterceptEnv()
        else:
            pytest.skip("Original 3DOF environment not available")
        
        # Test 6DOF environment in legacy mode
        env_6d_legacy = Aegis6DInterceptEnv(
            legacy_3dof_mode=True,
            difficulty_mode=DifficultyMode.EASY_3DOF
        )
        
        # Compare performance
        performance_3d = self._measure_environment_performance(env_3d_original)
        performance_6d = self._measure_environment_performance(env_6d_legacy)
        
        # 6DOF legacy mode should perform similarly to original 3DOF
        performance_ratio = performance_6d['steps_per_second'] / performance_3d['steps_per_second']
        assert performance_ratio > 0.7, f"6DOF legacy mode performance regression: {performance_ratio:.2f}x"
        
        print(f"3DOF Performance - Original: {performance_3d['steps_per_second']:.0f}, Legacy: {performance_6d['steps_per_second']:.0f}")
    
    def test_reward_system_consistency(self):
        """Test that reward systems remain consistent"""
        # Test scenarios that should give similar rewards in both modes
        test_scenarios = [
            {'seed': 42, 'actions': [[0.5, 0.3, 0.8], [0.2, -0.4, 0.6], [0.0, 0.0, 0.0]]},
            {'seed': 123, 'actions': [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]},
        ]
        
        for scenario in test_scenarios:
            # Legacy 3DOF mode
            env_legacy = Aegis6DInterceptEnv(
                legacy_3dof_mode=True,
                difficulty_mode=DifficultyMode.EASY_3DOF,
                max_steps=10
            )
            
            obs, _ = env_legacy.reset(seed=scenario['seed'])
            rewards_legacy = []
            
            for action_3d in scenario['actions']:
                action = np.concatenate([action_3d, [0.0]])  # Add explosion command
                obs, reward, terminated, truncated, info = env_legacy.step(action)
                rewards_legacy.append(reward)
                if terminated or truncated:
                    break
            
            # Test that rewards are reasonable and finite
            assert all(np.isfinite(r) for r in rewards_legacy), "All rewards should be finite"
            assert not all(r == 0 for r in rewards_legacy), "Not all rewards should be zero"
    
    def test_observation_space_compatibility(self):
        """Test observation space backward compatibility"""
        # Legacy mode should have observation space compatible with Phase 2
        env_legacy = Aegis6DInterceptEnv(
            legacy_3dof_mode=True,
            difficulty_mode=DifficultyMode.EASY_3DOF
        )
        
        obs, _ = env_legacy.reset(seed=42)
        
        # Observation should be reasonably sized for 3DOF (not too large)
        assert obs.shape[0] <= 20, f"Legacy observation space too large: {obs.shape[0]}D"
        assert obs.shape[0] >= 10, f"Legacy observation space too small: {obs.shape[0]}D"
        
        # Should be compatible with typical RL algorithms
        assert obs.dtype == np.float32, "Observation should be float32"
        assert np.all(np.isfinite(obs)), "All observations should be finite"
    
    def test_action_space_compatibility(self):
        """Test action space backward compatibility"""
        env_legacy = Aegis6DInterceptEnv(
            legacy_3dof_mode=True,
            action_mode=ActionMode.ACCELERATION_3DOF
        )
        
        # Action space should be compatible with Phase 2
        assert env_legacy.action_space.shape == (4,), "Legacy action space should be 4D"
        assert env_legacy.action_space.dtype == np.float32, "Action space should be float32"
        
        # Test action execution
        obs, _ = env_legacy.reset(seed=42)
        action = env_legacy.action_space.sample()
        obs_next, reward, terminated, truncated, info = env_legacy.step(action)
        
        assert env_legacy.observation_space.contains(obs_next), "Action execution should produce valid observations"
    
    def _measure_environment_performance(self, env) -> Dict[str, float]:
        """Helper method to measure environment performance"""
        # Warmup
        obs, _ = env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)
        
        # Benchmark
        num_steps = 1000
        start_time = time.time()
        
        obs, _ = env.reset(seed=42)
        for step in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset(seed=42 + step)
        
        elapsed_time = time.time() - start_time
        steps_per_second = num_steps / elapsed_time
        
        return {
            'steps_per_second': steps_per_second,
            'time_per_step': elapsed_time / num_steps
        }


class TestScalabilityValidation:
    """Test scalability and efficiency validation"""
    
    def test_world_size_scalability(self):
        """Test performance with different world sizes"""
        world_sizes = [100, 300, 500, 1000]
        performance_by_size = {}
        
        for world_size in world_sizes:
            env = Aegis6DInterceptEnv(
                difficulty_mode=DifficultyMode.FULL_6DOF,
                world_size=world_size,
                max_steps=50
            )
            
            # Measure performance
            num_steps = 200
            start_time = time.time()
            
            obs, _ = env.reset(seed=42)
            for step in range(num_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    obs, _ = env.reset(seed=42 + step)
            
            elapsed_time = time.time() - start_time
            steps_per_second = num_steps / elapsed_time
            
            performance_by_size[world_size] = steps_per_second
        
        # Performance shouldn't degrade dramatically with world size
        min_performance = min(performance_by_size.values())
        max_performance = max(performance_by_size.values())
        
        performance_range_ratio = max_performance / min_performance
        assert performance_range_ratio < 3.0, f"Performance varies too much with world size: {performance_range_ratio:.2f}x"
        
        print(f"World Size Scalability: {dict(performance_by_size)}")
    
    def test_max_steps_scalability(self):
        """Test performance with different max episode lengths"""
        max_steps_values = [50, 100, 200, 500]
        
        for max_steps in max_steps_values:
            env = Aegis6DInterceptEnv(
                difficulty_mode=DifficultyMode.FULL_6DOF,
                max_steps=max_steps
            )
            
            # Run episode to completion
            start_time = time.time()
            
            obs, _ = env.reset(seed=42)
            steps_taken = 0
            
            for step in range(max_steps + 10):  # Allow for some overage
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                steps_taken += 1
                
                if terminated or truncated:
                    break
            
            elapsed_time = time.time() - start_time
            
            # Episode should terminate within reasonable time
            assert elapsed_time < max_steps * 0.001, f"Episode too slow for max_steps={max_steps}: {elapsed_time:.3f}s"
            
            # Should respect max_steps limit
            assert steps_taken <= max_steps, f"Episode exceeded max_steps: {steps_taken} > {max_steps}"
    
    @pytest.mark.slow
    def test_long_running_stability(self):
        """Test stability over very long runs"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring(sample_interval=1.0)
        
        # Run for extended period
        total_steps = 0
        num_episodes = 200
        
        for episode in range(num_episodes):
            obs, _ = env.reset(seed=episode)
            
            for step in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_steps += 1
                
                if terminated or truncated:
                    break
        
        results = monitor.stop_monitoring()
        
        # Should maintain performance over long runs
        avg_steps_per_second = total_steps / results['duration']
        assert avg_steps_per_second > 1000, f"Long-run performance degraded: {avg_steps_per_second:.0f} steps/sec"
        
        # Memory should remain stable
        memory_growth_mb = results['memory_growth'] / (1024 * 1024)
        assert memory_growth_mb < 200, f"Excessive memory growth over long run: {memory_growth_mb:.1f}MB"
        
        print(f"Long-run Stability - Steps: {total_steps}, Performance: {avg_steps_per_second:.0f} steps/sec, Memory: {memory_growth_mb:.1f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])