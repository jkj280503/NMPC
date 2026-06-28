import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import math
import argparse


class SlopePathPublisher(Node):
    def __init__(self, mode):
        super().__init__('slope_path_publisher')
        self.mode = mode
        self.pub = self.create_publisher(Path, '/global_path', 10)
        self.timer = self.create_timer(1.0, self.publish_path)
        self.published = False

    def yaw_to_quat(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qz, qw

    def publish_path(self):
        if self.published:
            return

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'

        N = 201
        total_time = 20.0
        length = 8.0
        
        slope_deg = 10.0
        slope_rad = math.radians(slope_deg)
        height = length * math.tan(slope_rad)
        
        for i in range(N):
            s = i / (N - 1)
            x = length * s
            y = 0.0

            if self.mode == 'flat':
                z = 0.0
            elif self.mode == 'uphill':
                z = height * s       
            elif self.mode == 'downhill':
                z = height * (1.0 - s)
            else:
                z = 0.0

            yaw = 0.0

            pose = PoseStamped()
            pose.header.frame_id = 'odom'

            # 给每个路径点写时间戳，NMPC 会用这个时间戳插值参考轨迹
            t_i = total_time * s
            sec = int(t_i)
            nanosec = int((t_i - sec) * 1e9)
            pose.header.stamp.sec = sec
            pose.header.stamp.nanosec = nanosec

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z

            qz, qw = self.yaw_to_quat(yaw)
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            path.poses.append(pose)

        self.pub.publish(path)
        self.published = True
        self.get_logger().info(f'Published {self.mode} path with {N} points.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='uphill',
                        choices=['flat', 'uphill', 'downhill'])
    args = parser.parse_args()

    rclpy.init()
    node = SlopePathPublisher(args.mode)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
