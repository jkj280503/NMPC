#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a ROS 2 rosbag2 MCAP bag containing /cmd_vel and /odom into CSV.

Output columns:
    t, x, y, theta, v_act, omega_act, v_cmd, omega_cmd

Usage:
    source /opt/ros/jazzy/setup.bash
    python3 bag_to_csv.py --bag /path/to/rosbag_folder --out gazebo_openloop_v1.csv

Notes:
    - --bag should be the folder containing metadata.yaml, not the .mcap file itself.
    - The script uses bag receive timestamps for both /cmd_vel and /odom because /cmd_vel has no header stamp.
    - Commands are merged to odom timestamps using zero-order hold: each odom row uses the latest cmd_vel message before that odom timestamp.
"""

import argparse
import csv
import math
import os
from bisect import bisect_right

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def read_bag(bag_dir, cmd_topic='/cmd_vel', odom_topic='/odom'):
    if os.path.isfile(bag_dir):
        raise ValueError(
            f"--bag should be a rosbag folder containing metadata.yaml, not a file: {bag_dir}"
        )
    if not os.path.exists(os.path.join(bag_dir, 'metadata.yaml')):
        raise FileNotFoundError(f"metadata.yaml not found in bag folder: {bag_dir}")

    storage_options = StorageOptions(uri=bag_dir, storage_id='mcap')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    if cmd_topic not in type_map:
        raise RuntimeError(f"Topic {cmd_topic} not found. Available topics: {list(type_map.keys())}")
    if odom_topic not in type_map:
        raise RuntimeError(f"Topic {odom_topic} not found. Available topics: {list(type_map.keys())}")

    cmd_msg_type = get_message(type_map[cmd_topic])
    odom_msg_type = get_message(type_map[odom_topic])

    cmd_rows = []
    odom_rows = []
    first_stamp_ns = None

    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        if first_stamp_ns is None:
            first_stamp_ns = stamp_ns
        t = (stamp_ns - first_stamp_ns) * 1e-9

        if topic == cmd_topic:
            msg = deserialize_message(data, cmd_msg_type)
            cmd_rows.append((t, float(msg.linear.x), float(msg.angular.z)))

        elif topic == odom_topic:
            msg = deserialize_message(data, odom_msg_type)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            tw = msg.twist.twist
            theta = yaw_from_quaternion(q)
            odom_rows.append((
                t,
                float(p.x),
                float(p.y),
                float(theta),
                float(tw.linear.x),
                float(tw.angular.z),
            ))

    return cmd_rows, odom_rows, type_map


def merge_cmd_to_odom(cmd_rows, odom_rows):
    if len(cmd_rows) == 0:
        raise RuntimeError('No /cmd_vel messages were read from bag.')
    if len(odom_rows) == 0:
        raise RuntimeError('No /odom messages were read from bag.')

    cmd_t = [r[0] for r in cmd_rows]
    merged = []

    for od in odom_rows:
        t = od[0]
        idx = bisect_right(cmd_t, t) - 1
        if idx < 0:
            v_cmd, w_cmd = 0.0, 0.0
        else:
            v_cmd, w_cmd = cmd_rows[idx][1], cmd_rows[idx][2]

        merged.append((
            od[0],   # t
            od[1],   # x
            od[2],   # y
            od[3],   # theta
            od[4],   # v_act
            od[5],   # omega_act
            v_cmd,
            w_cmd,
        ))

    return merged


def write_csv(rows, out_csv):
    header = ['t', 'x', 'y', 'theta', 'v_act', 'omega_act', 'v_cmd', 'omega_cmd']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True, help='rosbag folder containing metadata.yaml')
    parser.add_argument('--out', default='gazebo_openloop.csv', help='output csv path')
    parser.add_argument('--cmd-topic', default='/cmd_vel')
    parser.add_argument('--odom-topic', default='/odom')
    args = parser.parse_args()

    rclpy.init(args=None)
    try:
        cmd_rows, odom_rows, type_map = read_bag(args.bag, args.cmd_topic, args.odom_topic)
        merged = merge_cmd_to_odom(cmd_rows, odom_rows)
        write_csv(merged, args.out)

        duration = merged[-1][0] - merged[0][0] if len(merged) > 1 else 0.0
        print('Topics in bag:')
        for name, typ in type_map.items():
            print(f'  {name}: {typ}')
        print(f'Read cmd messages : {len(cmd_rows)}')
        print(f'Read odom messages: {len(odom_rows)}')
        print(f'CSV rows          : {len(merged)}')
        print(f'Duration          : {duration:.3f} s')
        print(f'Wrote             : {args.out}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
