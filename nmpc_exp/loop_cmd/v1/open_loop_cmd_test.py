import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import argparse


class OpenLoopCmdTest(Node):
    def __init__(self, mode):
        super().__init__('open_loop_cmd_test')
        self.mode = mode
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info(f'Open-loop cmd test started, mode = {self.mode}')

    def get_cmd(self, t):
        if self.mode == 'straight':
            # 纵向速度阶跃：用于验证 kv、tau_v、slope_gain
            if t < 2.0:
                return 0.0, 0.0
            elif t < 10.0:
                return 0.5, 0.0
            elif t < 16.0:
                return 0.8, 0.0
            elif t < 22.0:
                return 0.3, 0.0
            else:
                return 0.0, 0.0

        elif self.mode == 'yaw':
            # 原地转向阶跃：主要用于验证 kr、tau_r
            if t < 2.0:
                return 0.0, 0.0
            elif t < 8.0:
                return 0.0, 0.4
            elif t < 14.0:
                return 0.0, -0.4
            elif t < 20.0:
                return 0.0, 0.8
            else:
                return 0.0, 0.0

        elif self.mode == 'combined':
            # 直线 + 转弯联合验证
            if t < 2.0:
                return 0.0, 0.0
            elif t < 8.0:
                return 0.5, 0.0
            elif t < 14.0:
                return 0.5, 0.4
            elif t < 20.0:
                return 0.5, -0.4
            elif t < 26.0:
                return 0.8, 0.3
            else:
                return 0.0, 0.0

        else:
            return 0.0, 0.0

    def timer_callback(self):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds * 1e-9

        v_cmd, w_cmd = self.get_cmd(t)

        msg = Twist()
        msg.linear.x = float(v_cmd)
        msg.angular.z = float(w_cmd)
        self.pub.publish(msg)

        if t > 30.0:
            stop = Twist()
            self.pub.publish(stop)
            self.get_logger().info('Open-loop test finished.')
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='straight',
                        choices=['straight', 'yaw', 'combined'])
    args = parser.parse_args()

    rclpy.init()
    node = OpenLoopCmdTest(args.mode)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
