#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a ROS2 bag recorded during open-loop identification to CSV files.

Compared with the old version, this version fixes the time-base problem:
  - default: use /clock simulation time when /clock exists;
  - optional: use /odom.header.stamp for odom rows;
  - fallback: use bag record time only when sim time is unavailable.

Dependency:
  pip install rosbags pandas numpy

Usage:
  python3 rosbag_to_csv_openloop_simtime.py --bag /path/to/bag_dir --out /path/to/csv_dir
  python3 rosbag_to_csv_openloop_simtime.py --bag /path/to/bag_dir --out /path/to/csv_dir --time-source clock
  python3 rosbag_to_csv_openloop_simtime.py --bag /path/to/bag_dir --out /path/to/csv_dir --time-source odom_header
  python3 rosbag_to_csv_openloop_simtime.py --bag /path/to/bag_dir --out /path/to/csv_dir --time-source bag

Outputs:
  odom.csv
  cmd_vel.csv
  openloop_debug.csv
  clock.csv              if /clock exists
  merged_openloop.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_float(stamp) -> float:
    """Convert ROS builtin_interfaces/Time-like stamp to seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def bag_rel_time(timestamp_ns: int, t0_ns: int) -> float:
    return (timestamp_ns - t0_ns) * 1e-9


def build_clock_mapper(clock_rows: List[Dict]) -> Tuple[Optional[callable], Optional[pd.DataFrame]]:
    """Return function bag_rel -> sim_rel using /clock, and clock dataframe."""
    if not clock_rows:
        return None, None

    clock = pd.DataFrame(clock_rows).sort_values('t_bag').drop_duplicates('t_bag')
    if len(clock) < 2:
        return None, clock

    # Make simulation time start at zero.
    sim0 = float(clock['t_clock_abs'].iloc[0])
    clock['t_clock'] = clock['t_clock_abs'] - sim0

    bag_t = clock['t_bag'].to_numpy(dtype=float)
    sim_t = clock['t_clock'].to_numpy(dtype=float)

    def mapper(t_bag: float) -> float:
        return float(np.interp(float(t_bag), bag_t, sim_t))

    return mapper, clock


def finite_difference_pose_velocity(merged: pd.DataFrame) -> pd.DataFrame:
    """Add vx_pose/vy_pose/r_pose based on pose derivative under the current time column t."""
    if merged.empty or len(merged) < 3:
        return merged

    dt = merged['t'].diff()
    dx = merged['x'].diff()
    dy = merged['y'].diff()

    # Avoid inf caused by repeated timestamps.
    dt = dt.replace(0.0, np.nan)
    xdot = dx / dt
    ydot = dy / dt

    # Use unwrapped theta for r_pose, but use raw theta direction for vx/vy projection.
    theta_raw = merged['theta']
    theta_unwrap = np.unwrap(theta_raw.to_numpy(dtype=float))
    merged['theta_unwrap'] = theta_unwrap

    merged['vx_pose'] = xdot * np.cos(theta_raw) + ydot * np.sin(theta_raw)
    merged['vy_pose'] = -xdot * np.sin(theta_raw) + ydot * np.cos(theta_raw)
    merged['r_pose'] = pd.Series(theta_unwrap).diff().to_numpy() / dt.to_numpy()
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert ROS2 open-loop identification bag to CSV with sim-time support')
    parser.add_argument('--bag', required=True, help='ROS2 bag directory, the folder containing metadata.yaml')
    parser.add_argument('--out', required=True, help='output CSV directory')
    parser.add_argument(
        '--time-source',
        choices=['clock', 'odom_header', 'bag'],
        default='clock',
        help=(
            'time source for CSV t column. clock: use /clock sim time if available; '
            'odom_header: use odom.header.stamp for odom and /clock-mapped time for other topics; '
            'bag: old behavior, use rosbag record timestamps.'
        ),
    )
    parser.add_argument('--merge-tolerance', type=float, default=0.10, help='merge_asof nearest tolerance [s]')
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    odom_rows: List[Dict] = []
    cmd_rows: List[Dict] = []
    debug_rows_raw: List[Dict] = []
    clock_rows: List[Dict] = []

    wanted_topics = {
        '/clock',
        '/odom',
        '/cmd_vel',
        '/openloop_debug/alpha',
        '/openloop_debug/beta',
        '/openloop_debug/mu',
        '/openloop_debug/case',
        '/openloop_debug/phase',
    }

    # First pass: buffer messages and collect /clock for bag-time -> sim-time mapping.
    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic in wanted_topics]
        if not connections:
            raise RuntimeError(f'No wanted topics found in {bag_path}. Available: {[c.topic for c in reader.connections]}')

        t0_ns: Optional[int] = None
        buffered = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if t0_ns is None:
                t0_ns = timestamp
            t_bag = bag_rel_time(timestamp, t0_ns)
            msg = reader.deserialize(rawdata, connection.msgtype)
            buffered.append((connection.topic, connection.msgtype, timestamp, t_bag, msg))
            if connection.topic == '/clock':
                clock_rows.append({
                    't_bag': t_bag,
                    't_clock_abs': stamp_to_float(msg.clock),
                })

    if t0_ns is None:
        raise RuntimeError('Bag contains no messages on wanted topics.')

    clock_mapper, clock_df = build_clock_mapper(clock_rows)
    has_clock = clock_mapper is not None
    if args.time_source == 'clock' and not has_clock:
        print('[WARN] /clock not found or insufficient. Falling back to bag time.')
    if args.time_source == 'odom_header' and not has_clock:
        print('[WARN] /clock not found. cmd_vel/debug topics will use bag time; odom still uses header stamp.')

    # For odom_header relative time origin.
    odom_stamp_origin: Optional[float] = None

    def topic_time(t_bag: float) -> float:
        if args.time_source in ('clock', 'odom_header') and has_clock:
            return clock_mapper(t_bag)  # type: ignore[misc]
        return t_bag

    for topic, msgtype, timestamp, t_bag, msg in buffered:  # noqa: ARG001
        if topic == '/clock':
            continue

        # Default time for non-header topics.
        t = topic_time(t_bag)

        if topic == '/odom':
            q = msg.pose.pose.orientation
            pos = msg.pose.pose.position
            tw = msg.twist.twist

            # For odom rows, allow direct odom.header.stamp time.
            t_odom_header_abs = None
            try:
                t_odom_header_abs = stamp_to_float(msg.header.stamp)
            except Exception:
                t_odom_header_abs = None

            if args.time_source == 'odom_header' and t_odom_header_abs is not None:
                if odom_stamp_origin is None:
                    odom_stamp_origin = t_odom_header_abs
                t = t_odom_header_abs - odom_stamp_origin

            odom_rows.append({
                't': t,
                't_bag': t_bag,
                't_odom_header_abs': t_odom_header_abs,
                'x': float(pos.x),
                'y': float(pos.y),
                'z': float(pos.z),
                'qx': float(q.x),
                'qy': float(q.y),
                'qz': float(q.z),
                'qw': float(q.w),
                'theta': yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w)),
                'vx_odom': float(tw.linear.x),
                'vy_odom': float(tw.linear.y),
                'vz_odom': float(tw.linear.z),
                'r_odom': float(tw.angular.z),
            })
        elif topic == '/cmd_vel':
            cmd_rows.append({
                't': t,
                't_bag': t_bag,
                'cmd_v': float(msg.linear.x),
                'cmd_w': float(msg.angular.z),
            })
        elif topic in ('/openloop_debug/alpha', '/openloop_debug/beta', '/openloop_debug/mu'):
            debug_rows_raw.append({
                't': t,
                't_bag': t_bag,
                'name': topic.rsplit('/', 1)[-1],
                'value': float(msg.data),
            })
        elif topic in ('/openloop_debug/case', '/openloop_debug/phase'):
            debug_rows_raw.append({
                't': t,
                't_bag': t_bag,
                'name': topic.rsplit('/', 1)[-1],
                'value': str(msg.data),
            })

    odom = pd.DataFrame(odom_rows).sort_values('t') if odom_rows else pd.DataFrame()
    cmd = pd.DataFrame(cmd_rows).sort_values('t') if cmd_rows else pd.DataFrame()

    if debug_rows_raw:
        dbg_raw = pd.DataFrame(debug_rows_raw).sort_values('t')
        debug_frames = []
        for name, group in dbg_raw.groupby('name'):
            sub = group[['t', 'value']].rename(columns={'value': name}).sort_values('t')
            debug_frames.append(sub)
        debug = debug_frames[0]
        for sub in debug_frames[1:]:
            debug = pd.merge_asof(
                debug.sort_values('t'),
                sub.sort_values('t'),
                on='t',
                direction='nearest',
                tolerance=args.merge_tolerance,
            )
    else:
        debug = pd.DataFrame()

    if clock_df is not None:
        clock_df.to_csv(out_dir / 'clock.csv', index=False)
    if not odom.empty:
        odom.to_csv(out_dir / 'odom.csv', index=False)
    if not cmd.empty:
        cmd.to_csv(out_dir / 'cmd_vel.csv', index=False)
    if not debug.empty:
        debug.to_csv(out_dir / 'openloop_debug.csv', index=False)

    # Make a merged table sampled on odom timestamps.
    merged = odom.copy()
    if not merged.empty and not cmd.empty:
        merged = pd.merge_asof(
            merged.sort_values('t'),
            cmd.sort_values('t'),
            on='t',
            direction='nearest',
            tolerance=args.merge_tolerance,
            suffixes=('', '_cmd'),
        )
    if not merged.empty and not debug.empty:
        merged = pd.merge_asof(
            merged.sort_values('t'),
            debug.sort_values('t'),
            on='t',
            direction='nearest',
            tolerance=args.merge_tolerance,
        )

    merged = finite_difference_pose_velocity(merged)

    if not merged.empty:
        merged.to_csv(out_dir / 'merged_openloop.csv', index=False)

    print(f'Wrote CSV files to: {out_dir}')
    print(f'  time_source requested: {args.time_source}')
    print(f'  /clock available: {has_clock}')
    print(f'  odom rows: {len(odom)}')
    print(f'  cmd rows: {len(cmd)}')
    print(f'  debug rows: {len(debug)}')
    print(f'  merged rows: {len(merged)}')
    if not merged.empty and 'cmd_v' in merged.columns:
        valid = merged['cmd_v'].notna() & merged['cmd_w'].notna()
        print(f'  merged rows with cmd: {int(valid.sum())}')
        if len(merged) > 2:
            duration = float(merged['t'].iloc[-1] - merged['t'].iloc[0])
            print(f'  merged t range: {merged["t"].iloc[0]:.6f} -> {merged["t"].iloc[-1]:.6f} s, duration={duration:.6f} s')


if __name__ == '__main__':
    main()
