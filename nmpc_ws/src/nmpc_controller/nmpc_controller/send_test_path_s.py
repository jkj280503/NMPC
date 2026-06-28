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
            t_seconds = cumulative_dist / target_velocity
            # 计算轨迹切向角 (导数)
            theta = math.atan(1.0 * math.cos(x))
            
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
        self.get_logger().info("✅ 测试 S型全局路径已发送！长 10m。")

if __name__ == '__main__':
    rclpy.init()
    node = PathPublisher()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()