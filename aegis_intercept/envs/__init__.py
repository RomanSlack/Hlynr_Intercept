"""
Gymnasium environments for AegisIntercept.

This package contains the missile interception environments for training
reinforcement learning agents in 2D, 3D, and 6-DOF scenarios.
"""

from gymnasium.envs.registration import register

# Register legacy environments with Gymnasium
register(
    id='Aegis2DIntercept-v0',
    entry_point='aegis_intercept.envs:Aegis2DInterceptEnv',
    max_episode_steps=300,
)

register(
    id='Aegis2DInterceptQuinn-v0',
    entry_point='aegis_intercept.envs:Aegis2DInterceptEnvQuinn',
    max_episode_steps=300,
)

register(
    id='Aegis3D-v0',
    entry_point='aegis_intercept.envs:Aegis3DEnv',
    max_episode_steps=300,
)

# Register Phase 3 environments
register(
    id='Aegis6DOF-v0',
    entry_point='aegis_intercept.envs:Aegis6DOFEnv',
    max_episode_steps=1000,
)

register(
    id='AegisMultiInterceptor-v0',
    entry_point='aegis_intercept.envs:MultiInterceptorEnv',
    max_episode_steps=1000,
)

# Import environments for direct access
from .aegis_2d_env import Aegis2DInterceptEnv
from .aegis_2d_env_quinn import Aegis2DInterceptEnvQuinn
from .aegis_3d_env import Aegis3DEnv
from .aegis_6dof_env import Aegis6DOFEnv
from .multi_interceptor_env import MultiInterceptorEnv

__all__ = [
    'Aegis2DInterceptEnv', 
    'Aegis2DInterceptEnvQuinn', 
    'Aegis3DEnv',
    'Aegis6DOFEnv',
    'MultiInterceptorEnv'
]