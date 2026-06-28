#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a ROS2 bag recorded during open-loop identification to CSV files.

Dependency:
  pip install rosbags pandas

Usage:
  python3 rosbag_to_csv_openloop.py --bag /path/to/bag_dir --out /path/to/csv_dir

Outputs:
  odom.csv
  cmd_vel.csv
  openloop_debug.csv
  merged_openloop.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rosbags.highlevel import AnyReader


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    # yaw around z axis, ROS quaternion convention x,y,z,w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def msg_time_to_float_from_bag(timestamp_ns: int, t0_ns: int) -> float:
    return (timestamp_ns - t0_ns) * 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert ROS2 open-loop identification bag to CSV')
    parser.add_argument('--bag', required=True, help='ROS2 bag directory, the folder containing metadata.yaml')
    parser.add_argument('--out', required=True, help='output CSV directory')
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    odom_rows: List[Dict] = []
    cmd_rows: List[Dict] = []
    debug_rows_raw: List[Dict] = []

    wanted_topics = {
        '/odom',
        '/cmd_vel',
        '/openloop_debug/alpha',
        '/openloop_debug/beta',
        '/openloop_debug/mu',
        '/openloop_debug/case',
        '/openloop_debug/phase',
    }

    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic in wanted_topics]
        if not connections:
            raise RuntimeError(f'No wanted topics found in {bag_path}. Available: {[c.topic for c in reader.connections]}')

        # Find earliest timestamp among wanted messages.
        t0_ns: Optional[int] = None
        buffered = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if t0_ns is None:
                t0_ns = timestamp
            buffered.append((connection, timestamp, rawdata))
        if t0_ns is None:
            raise RuntimeError('Bag contains no messages on wanted topics.')

        for connection, timestamp, rawdata in buffered:
            msg = reader.deserialize(rawdata, connection.msgtype)
            t = msg_time_to_float_from_bag(timestamp, t0_ns)
            topic = connection.topic

            if topic == '/odom':
                q = msg.pose.pose.orientation
                pos = msg.pose.pose.position
                tw = msg.twist.twist
                odom_rows.append({
                    't': t,
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
                    'cmd_v': float(msg.linear.x),
                    'cmd_w': float(msg.angular.z),
                })
            elif topic in ('/openloop_debug/alpha', '/openloop_debug/beta', '/openloop_debug/mu'):
                debug_rows_raw.append({
                    't': t,
                    'name': topic.rsplit('/', 1)[-1],
                    'value': float(msg.data),
                })
            elif topic in ('/openloop_debug/case', '/openloop_debug/phase'):
                debug_rows_raw.append({
                    't': t,
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
            debug = pd.merge_asof(debug.sort_values('t'), sub.sort_values('t'), on='t', direction='nearest', tolerance=0.1)
    else:
        debug = pd.DataFrame()

    if not odom.empty:
        odom.to_csv(out_dir / 'odom.csv', index=False)
    if not cmd.empty:
        cmd.to_csv(out_dir / 'cmd_vel.csv', index=False)
    if not debug.empty:
        debug.to_csv(out_dir / 'openloop_debug.csv', index=False)

    # Make a merged table sampled on odom timestamps.
    merged = odom.copy()
    if not merged.empty and not cmd.empty:
        merged = pd.merge_asof(merged.sort_values('t'), cmd.sort_values('t'), on='t', direction='nearest', tolerance=0.1)
    if not merged.empty and not debug.empty:
        merged = pd.merge_asof(merged.sort_values('t'), debug.sort_values('t'), on='t', direction='nearest', tolerance=0.1)

    # Add pose-differentiated velocities for validation, using central-ish finite differences.
    if not merged.empty and len(merged) >= 3:
        dt = merged['t'].diff()
        dx = merged['x'].diff()
        dy = merged['y'].diff()
        xdot = dx / dt
        ydot = dy / dt
        th = merged['theta']
        merged['vx_pose'] = xdot * th.apply(math.cos) + ydot * th.apply(math.sin)
        merged['vy_pose'] = -xdot * th.apply(math.sin) + ydot * th.apply(math.cos)
        merged['r_pose'] = merged['theta'].diff() / dt

    if not merged.empty:
        merged.to_csv(out_dir / 'merged_openloop.csv', index=False)

    print(f'Wrote CSV files to: {out_dir}')
    print(f'  odom rows: {len(odom)}')
    print(f'  cmd rows: {len(cmd)}')
    print(f'  debug rows: {len(debug)}')
    print(f'  merged rows: {len(merged)}')


if __name__ == '__main__':
    main()
