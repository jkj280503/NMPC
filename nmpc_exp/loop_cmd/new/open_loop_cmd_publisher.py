#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 open-loop command publisher for skid-steer / differential-drive model identification.

Publishes:
  /cmd_vel                       geometry_msgs/Twist
  /openloop_debug/alpha          std_msgs/Float64   metadata for longitudinal slope [rad]
  /openloop_debug/beta           std_msgs/Float64   metadata for cross slope [rad]
  /openloop_debug/mu             std_msgs/Float64   metadata for nominal friction
  /openloop_debug/case           std_msgs/String
  /openloop_debug/phase          std_msgs/String

The alpha/beta/mu topics do not affect Gazebo. They are recorded for later MATLAB/offline
model simulation, so make sure they match the world you launched.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, String


@dataclass(frozen=True)
class CaseConfig:
    v: float
    w: float
    duration: float
    alpha_deg: float = 0.0
    beta_deg: float = 0.0
    mu: float = 1.0
    note: str = ""


# Default cases follow the staged identification plan.
# You can override any field with --v, --w, --duration, --alpha-deg, --beta-deg, --mu.
CASES: Dict[str, CaseConfig] = {
    # Stage 1: flat longitudinal response, kv/tau_v
    "A1": CaseConfig(0.3, 0.0, 12.0, note="flat longitudinal step, v=0.3"),
    "A2": CaseConfig(0.5, 0.0, 12.0, note="flat longitudinal step, v=0.5"),
    "A3": CaseConfig(0.8, 0.0, 12.0, note="flat longitudinal step, v=0.8"),
    "A4": CaseConfig(0.5, 0.0, 8.0, note="flat accel then natural stop in post-stop phase"),

    # Stage 2: flat yaw response, kr/tau_r
    "B1": CaseConfig(0.0, 0.3, 12.0, note="flat in-place yaw step, w=0.3"),
    "B2": CaseConfig(0.0, 0.5, 12.0, note="flat in-place yaw step, w=0.5"),
    "B3": CaseConfig(0.0, -0.3, 12.0, note="flat in-place yaw step, w=-0.3"),
    "B4": CaseConfig(0.4, 0.3, 12.0, note="flat arc, v=0.4 w=0.3"),
    "B5": CaseConfig(0.4, -0.3, 12.0, note="flat arc, v=0.4 w=-0.3"),

    # Stage 3: flat arc validation
    "C1": CaseConfig(0.3, 0.2, 15.0, note="flat arc validation"),
    "C2": CaseConfig(0.5, 0.2, 15.0, note="flat arc validation"),
    "C3": CaseConfig(0.5, 0.4, 15.0, note="flat arc validation"),
    "C4": CaseConfig(0.5, -0.4, 15.0, note="flat arc validation"),

    # Stage 4: longitudinal slope response, slope_gain/c_rr. Launch matching slope world manually.
    "D1": CaseConfig(0.3, 0.0, 12.0, alpha_deg=10.0, note="10deg uphill, v=0.3"),
    "D2": CaseConfig(0.5, 0.0, 12.0, alpha_deg=10.0, note="10deg uphill, v=0.5"),
    "D3": CaseConfig(0.3, 0.0, 12.0, alpha_deg=-10.0, note="10deg downhill, v=0.3"),
    "D4": CaseConfig(0.5, 0.0, 12.0, alpha_deg=-10.0, note="10deg downhill, v=0.5"),

    # Stage 5: cross-slope response, cross_slope_gain/tau_y. Launch matching cross-slope world manually.
    "E1": CaseConfig(0.3, 0.0, 12.0, beta_deg=5.0, note="5deg cross slope, v=0.3"),
    "E2": CaseConfig(0.5, 0.0, 12.0, beta_deg=5.0, note="5deg cross slope, v=0.5"),
    "E3": CaseConfig(0.3, 0.0, 12.0, beta_deg=10.0, note="10deg cross slope, v=0.3"),
    "E4": CaseConfig(0.5, 0.0, 12.0, beta_deg=10.0, note="10deg cross slope, v=0.5"),

    # Stage 6: cross-slope arc response, k_vxr/ky_vr/C_yr if needed.
    "F1": CaseConfig(0.5, 0.3, 15.0, beta_deg=0.0, note="flat arc for turn-slip check"),
    "F2": CaseConfig(0.5, -0.3, 15.0, beta_deg=0.0, note="flat arc for turn-slip check"),
    "F3": CaseConfig(0.5, 0.3, 15.0, beta_deg=5.0, note="5deg cross-slope arc"),
    "F4": CaseConfig(0.5, -0.3, 15.0, beta_deg=5.0, note="5deg cross-slope arc"),

    # Stage 7: saturation/friction validation. Usually do this only after stages 1-6 are stable.
    "G1": CaseConfig(0.8, 0.0, 8.0, mu=1.0, note="large longitudinal command, friction/saturation check"),
    "G2": CaseConfig(0.5, 0.8, 8.0, mu=1.0, note="large yaw command, friction/saturation check"),
}


class OpenLoopCmdPublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('open_loop_cmd_publisher')
        self.args = args
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.alpha_pub = self.create_publisher(Float64, '/openloop_debug/alpha', 10)
        self.beta_pub = self.create_publisher(Float64, '/openloop_debug/beta', 10)
        self.mu_pub = self.create_publisher(Float64, '/openloop_debug/mu', 10)
        self.case_pub = self.create_publisher(String, '/openloop_debug/case', 10)
        self.phase_pub = self.create_publisher(String, '/openloop_debug/phase', 10)

        self.dt = 1.0 / args.hz
        self.start_time = time.monotonic()
        self.finished = False
        self.csv_file = None
        self.csv_writer = None

        if args.csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
            self.csv_file = open(args.csv, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                't_wall_rel', 'phase', 'case', 'v_cmd', 'omega_cmd',
                'alpha_rad', 'beta_rad', 'mu', 'note'
            ])

        self.total_time = args.warmup + args.duration + args.poststop
        self.timer = self.create_timer(self.dt, self.on_timer)

        self.get_logger().info(
            f"case={args.case}, note={args.note}, v={args.v:.3f}, w={args.w:.3f}, "
            f"duration={args.duration:.2f}s, warmup={args.warmup:.2f}s, poststop={args.poststop:.2f}s, "
            f"alpha={args.alpha_deg:.2f}deg, beta={args.beta_deg:.2f}deg, mu={args.mu:.2f}, hz={args.hz:.1f}"
        )

    def publish_stop(self) -> None:
        msg = Twist()
        self.cmd_pub.publish(msg)

    def on_timer(self) -> None:
        t = time.monotonic() - self.start_time

        if t < self.args.warmup:
            phase = 'warmup_zero'
            v_cmd = 0.0
            w_cmd = 0.0
        elif t < self.args.warmup + self.args.duration:
            phase = 'command'
            v_cmd = self.args.v
            w_cmd = self.args.w
        elif t < self.total_time:
            phase = 'poststop_zero'
            v_cmd = 0.0
            w_cmd = 0.0
        else:
            self.publish_stop()
            if not self.finished:
                self.get_logger().info('Finished. Published final zero cmd_vel.')
                self.finished = True
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
                self.csv_file = None
            rclpy.shutdown()
            return

        alpha_rad = math.radians(self.args.alpha_deg)
        beta_rad = math.radians(self.args.beta_deg)

        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.cmd_pub.publish(cmd)

        a = Float64(); a.data = alpha_rad; self.alpha_pub.publish(a)
        b = Float64(); b.data = beta_rad; self.beta_pub.publish(b)
        m = Float64(); m.data = float(self.args.mu); self.mu_pub.publish(m)
        c = String(); c.data = str(self.args.case); self.case_pub.publish(c)
        p = String(); p.data = phase; self.phase_pub.publish(p)

        if self.csv_writer:
            self.csv_writer.writerow([
                f'{t:.6f}', phase, self.args.case, f'{v_cmd:.8f}', f'{w_cmd:.8f}',
                f'{alpha_rad:.8f}', f'{beta_rad:.8f}', f'{self.args.mu:.8f}', self.args.note
            ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Open-loop /cmd_vel publisher for model identification')
    parser.add_argument('--case', default='A1', help=f'case id, e.g. A1/B1/C1/D1/E1/F1/G1, or custom. Known: {sorted(CASES)}')
    parser.add_argument('--v', type=float, default=None, help='override linear velocity command [m/s]')
    parser.add_argument('--w', type=float, default=None, help='override angular velocity command [rad/s]')
    parser.add_argument('--duration', type=float, default=None, help='override command duration [s]')
    parser.add_argument('--warmup', type=float, default=2.0, help='zero-command time before step [s]')
    parser.add_argument('--poststop', type=float, default=4.0, help='zero-command time after step [s]')
    parser.add_argument('--hz', type=float, default=50.0, help='publish frequency [Hz]')
    parser.add_argument('--alpha-deg', type=float, default=None, help='metadata longitudinal slope angle [deg]')
    parser.add_argument('--beta-deg', type=float, default=None, help='metadata cross-slope angle [deg]')
    parser.add_argument('--mu', type=float, default=None, help='metadata nominal friction')
    parser.add_argument('--csv', default='', help='optional local CSV log of the published commands')
    return parser


def apply_case_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cfg = CASES.get(args.case, CaseConfig(0.0, 0.0, 10.0, note='custom or unknown case'))
    args.v = cfg.v if args.v is None else args.v
    args.w = cfg.w if args.w is None else args.w
    args.duration = cfg.duration if args.duration is None else args.duration
    args.alpha_deg = cfg.alpha_deg if args.alpha_deg is None else args.alpha_deg
    args.beta_deg = cfg.beta_deg if args.beta_deg is None else args.beta_deg
    args.mu = cfg.mu if args.mu is None else args.mu
    args.note = cfg.note
    return args


def main() -> None:
    parser = build_parser()
    # remove_ros_args keeps this script compatible with `ros2 run ... --ros-args ...`.
    non_ros_argv = remove_ros_args(args=sys.argv)[1:]
    args = apply_case_defaults(parser.parse_args(non_ros_argv))

    rclpy.init(args=sys.argv)
    node = OpenLoopCmdPublisher(args)

    def handle_signal(signum, frame):  # noqa: ARG001
        node.get_logger().warn('Interrupted. Publishing zero cmd_vel before exit.')
        node.publish_stop()
        if node.csv_file:
            node.csv_file.flush()
            node.csv_file.close()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.publish_stop()
            rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
