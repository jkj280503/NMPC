#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy


class PathPublisher(Node):
    def __init__(self):
        super().__init__('test_path_publisher')
        qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub = self.create_publisher(Path, '/global_path', qos_profile)
        self.publish_straight_variable_speed_path()

    def velocity_profile(self, t: float) -> float:
        """
        直线变速速度规划，总时长约 12s。

        阶段：
        0.0~2.0s   : 0 -> 0.5 m/s 匀加速
        2.0~4.0s   : 0.5 m/s 匀速
        4.0~5.5s   : 0.5 -> 0.3 m/s 匀减速
        5.5~7.5s   : 0.3 m/s 匀速
        7.5~9.0s   : 0.3 -> 0.6 m/s 匀加速
        9.0~12.0s  : 0.6 -> 0 m/s 匀减速，到终点停车
        """
        if t < 0.0:
            return 0.0
        elif t <= 2.0:
            return 0.5 / 2.0 * t
        elif t <= 4.0:
            return 0.5
        elif t <= 5.5:
            return 0.5 + (0.3 - 0.5) / (5.5 - 4.0) * (t - 4.0)
        elif t <= 7.5:
            return 0.3
        elif t <= 9.0:
            return 0.3 + (0.6 - 0.3) / (9.0 - 7.5) * (t - 7.5)
        elif t <= 12.0:
            return 0.6 + (0.0 - 0.6) / (12.0 - 9.0) * (t - 9.0)
        else:
            return 0.0

    def publish_straight_variable_speed_path(self):
        msg = Path()
        base_stamp = self.get_clock().now()
        msg.header.stamp = base_stamp.to_msg()
        msg.header.frame_id = "odom"

        total_time = 12.0
        dt_sample = 0.10
        n_points = int(total_time / dt_sample) + 1

        ts = [i * dt_sample for i in range(n_points)]
        # 确保最后一个点精确为 total_time
        ts[-1] = total_time

        vs = [self.velocity_profile(t) for t in ts]

        # 用梯形积分根据速度生成直线位置 x(t)
        xs = [0.0]
        for i in range(1, len(ts)):
            dt = ts[i] - ts[i - 1]
            ds = 0.5 * (vs[i - 1] + vs[i]) * dt
            xs.append(xs[-1] + ds)

        y = 0.0
        theta = 0.0

        for t, x in zip(ts, xs):
            pose = PoseStamped()
            pose.header.stamp = (base_stamp + rclpy.time.Duration(nanoseconds=int(t * 1e9))).to_msg()
            pose.header.frame_id = "odom"

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            # 直线轨迹，航向角为 0
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)

            msg.poses.append(pose)

        self.pub.publish(msg)
        self.get_logger().info(
            f"✅ 直线变速全局路径已发送！总时间 {total_time:.2f}s，"
            f"总长度约 {xs[-1]:.2f}m，速度序列: "
            f"0 -> 0.5 -> 0.3 -> 0.6 -> 0 m/s。"
        )


if __name__ == '__main__':
    rclpy.init()
    node = PathPublisher()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()
