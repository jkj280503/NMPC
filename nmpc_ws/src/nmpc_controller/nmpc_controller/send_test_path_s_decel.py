#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy
import math

class PathPublisher(Node):
    def __init__(self):
        super().__init__('test_path_publisher')
        qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub = self.create_publisher(Path, '/global_path', qos_profile)
        self.publish_sine_wave_path()

    def publish_sine_wave_path(self):
        msg = Path()
        base_stamp = self.get_clock().now()
        msg.header.stamp = base_stamp.to_msg()
        msg.header.frame_id = "odom"

        target_velocity = 1.0

        # ========== 末端匀减速参数 ==========
        # 只修改最后一段速度规划：前面保持匀速，最后 decel_distance 米匀减速到终点停车
        decel_distance = 2.0       # 末端减速距离，单位 m；可以按需要调成 1.5~3.0
        v_end = 0.0                # 终点参考速度

        # ========== 先生成几何路径，不直接赋时间 ==========
        xs = []
        ys = []
        thetas = []
        s_list = []

        cumulative_dist = 0.0
        prev_x, prev_y = 0.0, 0.0

        # 生成一段 10米长 的 S型 (正弦) 轨迹
        for i in range(100):
            x = i * 0.1  # 每隔 0.1m 一个点
            y = 1.0 * math.sin(x) # 振幅为 1m 的正弦波

            if i > 0:
                dist = math.hypot(x - prev_x, y - prev_y)
                cumulative_dist += dist
            prev_x, prev_y = x, y

            # 计算轨迹切向角 (导数)
            theta = math.atan(1.0 * math.cos(x))

            xs.append(x)
            ys.append(y)
            thetas.append(theta)
            s_list.append(cumulative_dist)

        total_length = s_list[-1]
        decel_distance = min(decel_distance, 0.5 * total_length)
        s_decel_start = total_length - decel_distance

        # 匀减速：v_end^2 = v0^2 - 2*a*D
        # 这里 v_end = 0，所以 a_abs = v0^2/(2D)
        v0 = target_velocity
        a_abs = (v0 * v0 - v_end * v_end) / (2.0 * max(decel_distance, 1e-6))
        t_decel_start = s_decel_start / target_velocity

        # ========== 根据弧长分配时间戳：前段匀速，末段匀减速 ==========
        for x, y, theta, s in zip(xs, ys, thetas, s_list):
            if s <= s_decel_start:
                t_seconds = s / target_velocity
            else:
                ds = s - s_decel_start
                # s = v0*tau - 0.5*a_abs*tau^2
                # tau = (v0 - sqrt(v0^2 - 2*a_abs*ds)) / a_abs
                v_sq = max(v0 * v0 - 2.0 * a_abs * ds, v_end * v_end)
                v_s = math.sqrt(v_sq)
                tau = (v0 - v_s) / max(a_abs, 1e-6)
                t_seconds = t_decel_start + tau

            pose = PoseStamped()
            pose.header.stamp = (base_stamp + rclpy.time.Duration(nanoseconds=int(t_seconds * 1e9))).to_msg()
            pose.header.frame_id = "odom"
            pose.pose.position.x = x
            pose.pose.position.y = y

            # 简单的偏航角转四元数 (只绕 Z 轴旋转)
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)

            msg.poses.append(pose)

        self.pub.publish(msg)
        self.get_logger().info(
            f"✅ 测试 S型全局路径已发送！长约 {total_length:.2f}m，"
            f"末端 {decel_distance:.2f}m 匀减速到 0，不添加终点保持段。"
        )

if __name__ == '__main__':
    rclpy.init()
    node = PathPublisher()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()
