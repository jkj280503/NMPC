#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terrain-aware /global_path publisher for NMPC dynamic version 3.

This script is designed for the three SDF worlds previously provided:
  1) uniform_slope10_only.world.sdf
  2) uniform_cross5_only.world.sdf
  3) uniform_slope10_cross5.world.sdf

It publishes nav_msgs/Path on /global_path. Each PoseStamped contains:
  position.x, position.y, position.z  -> used by NMPC to compute longitudinal grade alpha
  orientation roll                    -> used by NMPC as cross-slope beta
  orientation yaw                     -> used as reference heading theta_ref

Coordinate convention used here:
  - +x direction uphill for positive alpha_world_deg
  - +y direction uphill for positive beta_world_deg
  - Terrain plane is approximated as z = tan(alpha_world)*x + tan(beta_world)*y + z0

For a curved path, the path-direction longitudinal slope and lateral cross-slope vary with yaw:
  alpha_path = atan(grad_z dot tangent)
  beta_path  = atan(grad_z dot left_normal)
The script writes beta_path into pose.orientation roll.
"""

import math
from typing import Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert roll-pitch-yaw to geometry_msgs/Quaternion."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class TerrainPathPublisher(Node):
    def __init__(self):
        super().__init__('publish_terrain_path_for_v3')

        # World/path selection.
        self.declare_parameter('world_mode', 'slope10_cross5')
        self.declare_parameter('path_type', 'straight')  # straight, s_curve

        # If alpha_deg / beta_deg are left at 9999.0, defaults are taken from world_mode.
        self.declare_parameter('alpha_deg', 9999.0)
        self.declare_parameter('beta_deg', 9999.0)

        # Reference trajectory parameters.
        self.declare_parameter('v_ref', 0.4)
        self.declare_parameter('duration', 20.0)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('length', -1.0)  # if <=0, length = v_ref * duration
        self.declare_parameter('x0', 0.0)
        self.declare_parameter('y0', 0.0)
        self.declare_parameter('z0', 0.05)

        # S-curve parameters.
        self.declare_parameter('s_curve_amp', 0.8)
        self.declare_parameter('s_curve_wavelength', 12.0)

        # Publishing behavior.
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('topic', '/global_path')
        self.declare_parameter('publish_period', 1.0)
        self.declare_parameter('publish_once', False)

        self.world_mode = self.get_parameter('world_mode').value
        self.path_type = self.get_parameter('path_type').value
        self.v_ref = float(self.get_parameter('v_ref').value)
        self.duration = float(self.get_parameter('duration').value)
        self.dt = float(self.get_parameter('dt').value)
        self.length = float(self.get_parameter('length').value)
        self.x0 = float(self.get_parameter('x0').value)
        self.y0 = float(self.get_parameter('y0').value)
        self.z0 = float(self.get_parameter('z0').value)
        self.s_curve_amp = float(self.get_parameter('s_curve_amp').value)
        self.s_curve_wavelength = float(self.get_parameter('s_curve_wavelength').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.topic = self.get_parameter('topic').value
        self.publish_period = float(self.get_parameter('publish_period').value)
        self.publish_once = bool(self.get_parameter('publish_once').value)

        alpha_deg_param = float(self.get_parameter('alpha_deg').value)
        beta_deg_param = float(self.get_parameter('beta_deg').value)
        alpha_default, beta_default = self.default_slopes_from_world(self.world_mode)
        self.alpha_world_deg = alpha_default if abs(alpha_deg_param - 9999.0) < 1e-6 else alpha_deg_param
        self.beta_world_deg = beta_default if abs(beta_deg_param - 9999.0) < 1e-6 else beta_deg_param

        if self.v_ref <= 1e-6:
            raise ValueError('v_ref must be positive.')
        if self.dt <= 1e-6:
            raise ValueError('dt must be positive.')
        if self.duration <= 0.0:
            raise ValueError('duration must be positive.')
        if self.length <= 0.0:
            self.length = self.v_ref * self.duration

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.path_pub = self.create_publisher(Path, self.topic, qos)

        self.path_msg = self.build_path()
        self.timer = self.create_timer(self.publish_period, self.publish_path)
        self.published_count = 0

        self.get_logger().info(
            f"Terrain path publisher started. topic={self.topic}, world_mode={self.world_mode}, "
            f"path_type={self.path_type}, alpha_world={self.alpha_world_deg:.3f} deg, "
            f"beta_world={self.beta_world_deg:.3f} deg, length={self.length:.2f} m, "
            f"v_ref={self.v_ref:.2f} m/s, points={len(self.path_msg.poses)}"
        )
        self.publish_path()

    @staticmethod
    def default_slopes_from_world(world_mode: str) -> Tuple[float, float]:
        mode = world_mode.lower()
        if mode in ['flat', 'plane', 'level']:
            return 0.0, 0.0
        if mode in ['slope10', 'slope10_only', 'uniform_slope10_only']:
            return 10.0, 0.0
        if mode in ['cross5', 'cross5_only', 'uniform_cross5_only']:
            return 0.0, 5.0
        if mode in ['slope10_cross5', 'uniform_slope10_cross5', 'slope_cross']:
            return 10.0, 5.0
        # Unknown mode: safe default, can still be overridden by alpha_deg/beta_deg.
        return 0.0, 0.0

    def generate_xy(self, x_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return x, y, yaw arrays."""
        x = self.x0 + x_samples

        if self.path_type == 'straight':
            y = np.full_like(x, self.y0, dtype=float)
            yaw = np.zeros_like(x, dtype=float)
            return x, y, yaw

        if self.path_type == 's_curve':
            k = 2.0 * math.pi / max(self.s_curve_wavelength, 1e-6)
            y = self.y0 + self.s_curve_amp * np.sin(k * x_samples)
            dy_dx = self.s_curve_amp * k * np.cos(k * x_samples)
            yaw = np.arctan2(dy_dx, np.ones_like(dy_dx))
            return x, y, yaw

        raise ValueError(f'Unsupported path_type: {self.path_type}. Use straight or s_curve.')

    def build_path(self) -> Path:
        alpha_world = math.radians(self.alpha_world_deg)
        beta_world = math.radians(self.beta_world_deg)

        # Plane gradient. With the SDF worlds given earlier, positive alpha means +x uphill,
        # positive beta means +y uphill.
        dz_dx = math.tan(alpha_world)
        dz_dy = math.tan(beta_world)

        # Use approximately duration/dt points. x length is specified by length; timestamps are based on arc length/v_ref.
        n = max(2, int(math.ceil(self.duration / self.dt)) + 1)
        x_samples = np.linspace(0.0, self.length, n)
        x, y, yaw = self.generate_xy(x_samples)

        # Terrain plane height.
        z = self.z0 + dz_dx * (x - self.x0) + dz_dy * (y - self.y0)

        # Compute cumulative path length and timestamps.
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx * dx + dy * dy)
        s_cum = np.zeros_like(x)
        s_cum[1:] = np.cumsum(ds)
        t = s_cum / self.v_ref

        path_msg = Path()
        path_msg.header.frame_id = self.frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()

        start_time = self.get_clock().now()

        alpha_list = []
        beta_list = []

        for i in range(n):
            theta = float(yaw[i])

            # Unit tangent and left-normal in xy plane.
            tx = math.cos(theta)
            ty = math.sin(theta)
            nx = -math.sin(theta)
            ny = math.cos(theta)

            # Directional slopes along path tangent and left-normal.
            grade_t = dz_dx * tx + dz_dy * ty
            grade_n = dz_dx * nx + dz_dy * ny

            alpha_path = math.atan(grade_t)
            beta_path = math.atan(grade_n)

            alpha_list.append(alpha_path)
            beta_list.append(beta_path)

            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = (start_time + Duration(seconds=float(t[i]))).to_msg()
            pose.pose.position.x = float(x[i])
            pose.pose.position.y = float(y[i])
            pose.pose.position.z = float(z[i])

            # NMPC v3 reads beta from roll and yaw from theta. Keep pitch zero to avoid ambiguity.
            pose.pose.orientation = rpy_to_quaternion(beta_path, 0.0, theta)

            path_msg.poses.append(pose)

        self.alpha_min = min(alpha_list)
        self.alpha_max = max(alpha_list)
        self.beta_min = min(beta_list)
        self.beta_max = max(beta_list)
        self.t_final = float(t[-1])
        return path_msg

    def publish_path(self):
        now = self.get_clock().now()
        self.path_msg.header.stamp = now.to_msg()
        # Refresh pose timestamps so the first pose starts from current publication time.
        if self.path_msg.poses:
            first_stamp = self.path_msg.poses[0].header.stamp
            # Rebuild is simpler and keeps stamps consistent with current time.
            self.path_msg = self.build_path()
            self.path_msg.header.stamp = now.to_msg()

        self.path_pub.publish(self.path_msg)
        self.published_count += 1

        self.get_logger().info(
            f"Published /global_path #{self.published_count}: "
            f"t_final={self.t_final:.2f}s, "
            f"alpha=[{self.alpha_min:.4f}, {self.alpha_max:.4f}] rad, "
            f"beta=[{self.beta_min:.4f}, {self.beta_max:.4f}] rad"
        )

        if self.publish_once and self.published_count >= 1:
            self.get_logger().info('publish_once=true, shutting down after one publication.')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TerrainPathPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
