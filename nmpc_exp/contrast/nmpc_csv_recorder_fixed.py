#!/usr/bin/env python3
"""
NMPC CSV recorder for comparing multiple controller versions.

Features:
- Records selected /nmpc_debug/* Float64 topics into one CSV.
- Skips invalid blank rows before controller starts publishing.
- Uses controller t_ref as the aligned plot time column `t`.
- Also estimates wall time since controller start from first valid t_ref sample.
- Optional --t_final makes arrival_time_error computation reliable.
- Clean Ctrl+C shutdown under ROS 2 Jazzy.
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime
from typing import Callable, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class NmpcCsvRecorder(Node):
    def __init__(
        self,
        exp_name: str,
        out_dir: str,
        rate_hz: float,
        goal_tolerance: float,
        t_final: Optional[float],
        auto_stop: bool,
        post_arrival_sec: float,
    ):
        super().__init__('nmpc_csv_recorder')

        self.exp_name = exp_name
        self.out_dir = os.path.expanduser(out_dir)
        self.rate_hz = float(rate_hz)
        self.goal_tolerance = float(goal_tolerance)
        self.t_final_user = float(t_final) if t_final is not None else None
        self.auto_stop = bool(auto_stop)
        self.post_arrival_sec = float(post_arrival_sec)

        os.makedirs(self.out_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.out_dir, f'{stamp}_{self.exp_name}.csv')

        # Latest topic values. None means not received yet.
        self.values: Dict[str, Optional[float]] = {
            'e_t': None,
            'e_n': None,
            'e_theta': None,
            't_ref': None,
            'goal_dist': None,
            'solve_ms': None,
            'v_cmd': None,
            'omega_cmd': None,
            'v_sat_ratio': None,
            'omega_sat_ratio': None,
            # Optional dynamic-model topics.
            'alpha': None,
            'beta': None,
            'mu': None,
            'vy': None,
        }

        # Wall-clock alignment. When first valid t_ref arrives, infer the controller start time.
        self.valid_started = False
        self.controller_start_wall: Optional[float] = None
        self.max_t_ref_seen = 0.0

        self.arrived = False
        self.arrival_wall_time: Optional[float] = None
        self.arrival_time_error = math.nan
        self.arrival_detect_wall: Optional[float] = None

        self._create_subscriptions()

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            # `t` should be used as PlotJuggler x-axis; it is controller t_ref, so runs align.
            't',
            # Wall time since inferred controller start, useful for arrival-time error.
            'elapsed_wall',
            't_ref',
            'e_t',
            'e_n',
            'e_theta',
            'goal_dist',
            'solve_ms',
            'v_cmd',
            'omega_cmd',
            'v_sat_ratio',
            'omega_sat_ratio',
            'arrival_time_error',
            'arrived',
            'alpha',
            'beta',
            'mu',
            'vy',
        ])

        period = 1.0 / max(self.rate_hz, 1e-6)
        self.create_timer(period, self.write_row)

        self.get_logger().info(f'CSV recorder started: {self.csv_path}')
        if self.t_final_user is None:
            self.get_logger().warn(
                'No --t_final provided. Arrival error will use max observed t_ref; '
                'for strict comparison, pass --t_final 12.118867 or your trajectory final time.'
            )

    def _create_subscriptions(self) -> None:
        topics = {
            'e_t': '/nmpc_debug/e_t',
            'e_n': '/nmpc_debug/e_n',
            'e_theta': '/nmpc_debug/e_theta',
            't_ref': '/nmpc_debug/t_ref',
            'goal_dist': '/nmpc_debug/goal_dist',
            'solve_ms': '/nmpc_debug/solve_ms',
            'v_cmd': '/nmpc_debug/v_cmd',
            'omega_cmd': '/nmpc_debug/omega_cmd',
            'v_sat_ratio': '/nmpc_debug/v_sat_ratio',
            'omega_sat_ratio': '/nmpc_debug/omega_sat_ratio',
            'alpha': '/nmpc_debug/alpha',
            'beta': '/nmpc_debug/beta',
            'mu': '/nmpc_debug/mu',
            'vy': '/nmpc_debug/vy',
        }
        for key, topic in topics.items():
            self.create_subscription(Float64, topic, self._callback_for(key), 20)

    def _callback_for(self, key: str) -> Callable[[Float64], None]:
        def _callback(msg: Float64) -> None:
            try:
                self.values[key] = float(msg.data)
            except Exception:
                self.values[key] = None
        return _callback

    def _has_minimum_valid_data(self) -> bool:
        # Do not write rows before controller publishes usable debug data.
        required = ['t_ref', 'e_t', 'e_n', 'e_theta', 'goal_dist', 'solve_ms', 'v_cmd', 'omega_cmd']
        return all(self.values.get(k) is not None for k in required)

    def _current_t_final(self) -> float:
        if self.t_final_user is not None:
            return self.t_final_user
        return self.max_t_ref_seen

    def write_row(self) -> None:
        if not self._has_minimum_valid_data():
            return

        now = time.time()
        t_ref = float(self.values['t_ref'])
        goal_dist = float(self.values['goal_dist'])

        if not self.valid_started:
            # Infer wall-clock controller start from first t_ref sample.
            self.controller_start_wall = now - t_ref
            self.valid_started = True

        self.max_t_ref_seen = max(self.max_t_ref_seen, t_ref)
        elapsed_wall = now - self.controller_start_wall if self.controller_start_wall is not None else math.nan
        t_final = self._current_t_final()

        # Arrival detection. For best accuracy, provide --t_final.
        if not self.arrived and t_final > 0.1:
            # With user-provided t_final, this is strict. Without it, max_t_ref_seen may be less reliable.
            at_final_time = t_ref >= (t_final - 1e-3)
            in_goal = goal_dist <= self.goal_tolerance
            if at_final_time and in_goal:
                self.arrived = True
                self.arrival_wall_time = elapsed_wall
                self.arrival_time_error = self.arrival_wall_time - t_final
                self.arrival_detect_wall = now
                self.get_logger().info(
                    f'Arrived: arrival_time={self.arrival_wall_time:.3f}s, '
                    f't_final={t_final:.3f}s, error={self.arrival_time_error:.3f}s, '
                    f'goal_dist={goal_dist:.3f}m'
                )

        self.writer.writerow([
            t_ref,                         # t: x-axis aligned by controller reference time
            elapsed_wall,
            t_ref,
            self._num(self.values['e_t']),
            self._num(self.values['e_n']),
            self._num(self.values['e_theta']),
            self._num(self.values['goal_dist']),
            self._num(self.values['solve_ms']),
            self._num(self.values['v_cmd']),
            self._num(self.values['omega_cmd']),
            self._num(self.values['v_sat_ratio']),
            self._num(self.values['omega_sat_ratio']),
            self.arrival_time_error if not math.isnan(self.arrival_time_error) else '',
            1.0 if self.arrived else 0.0,
            self._num(self.values['alpha']),
            self._num(self.values['beta']),
            self._num(self.values['mu']),
            self._num(self.values['vy']),
        ])
        self.csv_file.flush()

        if self.auto_stop and self.arrived and self.arrival_detect_wall is not None:
            if (now - self.arrival_detect_wall) >= self.post_arrival_sec:
                self.get_logger().info('Auto-stop after arrival.')
                rclpy.shutdown()

    @staticmethod
    def _num(x: Optional[float]):
        # Empty optional columns are easier for PlotJuggler/CSV than text placeholders.
        return '' if x is None else x

    def destroy_node(self) -> bool:
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='nmpc_exp')
    parser.add_argument('--out', type=str, default='~/nmpc/nmpc_exp/contrast/csv_logs')
    parser.add_argument('--rate', type=float, default=20.0)
    parser.add_argument('--goal_tolerance', type=float, default=0.10)
    parser.add_argument('--t_final', type=float, default=None,
                        help='Reference final time. Recommended for accurate arrival_time_error, e.g. 12.118867')
    parser.add_argument('--auto_stop', action='store_true',
                        help='Stop recorder automatically after arrival is detected.')
    parser.add_argument('--post_arrival_sec', type=float, default=0.5,
                        help='Seconds to keep recording after arrival when --auto_stop is enabled.')
    args = parser.parse_args()

    rclpy.init()
    node = NmpcCsvRecorder(
        exp_name=args.name,
        out_dir=args.out,
        rate_hz=args.rate,
        goal_tolerance=args.goal_tolerance,
        t_final=args.t_final,
        auto_stop=args.auto_stop,
        post_arrival_sec=args.post_arrival_sec,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('CSV recorder stopped by Ctrl+C')
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
