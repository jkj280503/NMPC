import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class OpenLoopCmdTest(Node):
    def __init__(self):
        super().__init__('open_loop_cmd_test')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.05, self.timer_callback)

    def get_cmd(self, t):
        if t < 2.0:
            return 0.0, 0.0
        elif t < 8.0:
            return 0.5, 0.0
        elif t < 14.0:
            return 0.5, 0.4
        elif t < 20.0:
            return 0.5, -0.4
        elif t < 26.0:
            return 1.0, 0.3
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

        if t > 32.0:
            stop = Twist()
            self.pub.publish(stop)
            self.get_logger().info('Open-loop test finished.')
            rclpy.shutdown()

def main():
    rclpy.init()
    node = OpenLoopCmdTest()
    rclpy.spin(node)

if __name__ == '__main__':
    main()