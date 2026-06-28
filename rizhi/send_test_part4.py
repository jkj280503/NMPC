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

        # ========== 轨迹参数 ==========
        n_points = 120
        dx_sample = 0.1
        amp = 1.0

        # ========== 速度规划参数 ==========
        v_max = 1.0      # 中段最大速度
        a_acc = 0.8      # 起步加速度
        a_dec = 0.8      # 末端减速度

        # 先生成空间路径
        xs = []
        ys = []
        thetas = []

        for i in range(n_points):
            x = i * dx_sample
            y = amp * math.sin(x)
            theta = math.atan(amp * math.cos(x))

            xs.append(x)
            ys.append(y)
            thetas.append(theta)

        # 计算累计弧长
        s_list = [0.0]
        for i in range(1, n_points):
            ds = math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1])
            s_list.append(s_list[-1] + ds)

        total_s = s_list[-1]

        # ========== 根据弧长生成梯形/三角速度时间戳 ==========
        s_acc = v_max ** 2 / (2.0 * a_acc)
        s_dec = v_max ** 2 / (2.0 * a_dec)

        # 如果路径不够长，自动改成三角速度
        if s_acc + s_dec > total_s:
            v_peak = math.sqrt(
                2.0 * a_acc * a_dec * total_s / (a_acc + a_dec)
            )
            s_acc = v_peak ** 2 / (2.0 * a_acc)
            s_dec = v_peak ** 2 / (2.0 * a_dec)
            v_cruise = v_peak
        else:
            v_cruise = v_max

        s_cruise_start = s_acc
        s_cruise_end = total_s - s_dec

        t_acc = v_cruise / a_acc
        t_dec = v_cruise / a_dec
        t_cruise = max(0.0, (s_cruise_end - s_cruise_start) / v_cruise)

        def time_from_s(s):
            # 加速段
            if s <= s_cruise_start:
                return math.sqrt(2.0 * s / a_acc)

            # 匀速段
            elif s <= s_cruise_end:
                return t_acc + (s - s_cruise_start) / v_cruise

            # 减速段
            else:
                s_remain = total_s - s
                return t_acc + t_cruise + t_dec - math.sqrt(
                    max(0.0, 2.0 * s_remain / a_dec)
                )

        # 生成 Path
        for i in range(n_points):
            t_seconds = time_from_s(s_list[i])

            pose = PoseStamped()
            pose.header.stamp = (
                base_stamp + rclpy.time.Duration(
                    nanoseconds=int(t_seconds * 1e9)
                )
            ).to_msg()
            pose.header.frame_id = "odom"

            pose.pose.position.x = xs[i]
            pose.pose.position.y = ys[i]

            theta = thetas[i]
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)

            msg.poses.append(pose)

        # ========== 终点保持段 ==========
        # 如果你的 NMPC path_callback 不会过滤重复点，可以加入终点保持
        # 这会让参考轨迹末端显式保持在终点，速度为 0
        hold_time = 1.0
        final_t = time_from_s(total_s)
        final_x = xs[-1]
        final_y = ys[-1]
        final_theta = thetas[-1]

        for j in range(1, 6):
            pose = PoseStamped()
            t_hold = final_t + hold_time * j / 5.0

            pose.header.stamp = (
                base_stamp + rclpy.time.Duration(
                    nanoseconds=int(t_hold * 1e9)
                )
            ).to_msg()
            pose.header.frame_id = "odom"

            pose.pose.position.x = final_x
            pose.pose.position.y = final_y

            pose.pose.orientation.z = math.sin(final_theta / 2.0)
            pose.pose.orientation.w = math.cos(final_theta / 2.0)

            msg.poses.append(pose)

        self.pub.publish(msg)
        self.get_logger().info(
            f"✅ S型全局路径已发送：总弧长 {total_s:.2f} m，"
            f"总时间 {final_t:.2f} s，末端保持 {hold_time:.1f} s"
        )

if __name__ == '__main__':
    rclpy.init()
    node = PathPublisher()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()