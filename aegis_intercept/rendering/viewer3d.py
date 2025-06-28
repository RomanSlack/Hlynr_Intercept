"""3D Viewer for AegisIntercept using Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np

class Viewer3D:
    def __init__(self, world_size: float):
        self.world_size = world_size
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-world_size, world_size])
        self.ax.set_ylim([-world_size, world_size])
        self.ax.set_zlim([-world_size, world_size])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("AegisIntercept 3D")
        plt.ion()

    def render(self, interceptor_pos: np.ndarray, missile_pos: np.ndarray, target_pos: np.ndarray):
        self.ax.cla()
        self.ax.set_xlim([-self.world_size, self.world_size])
        self.ax.set_ylim([-self.world_size, self.world_size])
        self.ax.set_zlim([-self.world_size, self.world_size])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("AegisIntercept 3D")

        # Draw target
        self.ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='g', marker='o', s=100, label="Target")

        # Draw interceptor
        self.ax.scatter(interceptor_pos[0], interceptor_pos[1], interceptor_pos[2], c='b', marker='^', s=100, label="Interceptor")

        # Draw missile
        self.ax.scatter(missile_pos[0], missile_pos[1], missile_pos[2], c='r', marker='x', s=100, label="Missile")

        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close()
