import rclpy
from rclpy.node import Node
import numpy as np

from nmpc_interfaces.msg import CpsTrajectory, CpsTrajectoryPoint


class CpsTxtPublisher(Node):
    def __init__(self):
        super().__init__('cps_txt_publisher')

        self.declare_parameter('txt_path', '')
        self.declare_parameter('publish_hz', 2.0)
        self.declare_parameter('repeat_count', 3)
        self.declare_parameter('frame_id', 'gps')

        self.txt_path = self.get_parameter('txt_path').value
        self.publish_hz = float(self.get_parameter('publish_hz').value)
        self.repeat_count = int(self.get_parameter('repeat_count').value)
        self.frame_id = self.get_parameter('frame_id').value

        if self.txt_path == '':
            raise RuntimeError('必须通过参数 txt_path 指定 CPS 轨迹 txt 文件路径')

        self.pub = self.create_publisher(CpsTrajectory, '/cps_trajectory', 10)

        self.msg = self.load_txt_as_msg(self.txt_path)
        self.publish_counter = 0

        period = 1.0 / max(self.publish_hz, 1e-6)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f'已读取 CPS txt: {self.txt_path}, 点数={len(self.msg.points)}, '
            f'即将发布到 /cps_trajectory'
        )

    def load_txt_as_msg(self, txt_path):
        data = np.loadtxt(txt_path)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 5:
            raise RuntimeError(
                f'CPS txt 至少需要5列: lat_rad lon_rad altitude_m speed_kmh time_s，'
                f'当前列数={data.shape[1]}'
            )

        lat = data[:, 0]
        lon = data[:, 1]
        alt = data[:, 2]
        speed_kmh = data[:, 3]
        time_s = data[:, 4]

        # 时间从0开始，防止上层给的是绝对时间
        time_s = time_s - time_s[0]

        msg = CpsTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        for i in range(data.shape[0]):
            p = CpsTrajectoryPoint()
            p.latitude_rad = float(lat[i])
            p.longitude_rad = float(lon[i])
            p.altitude_m = float(alt[i])
            p.speed_kmh = float(speed_kmh[i])
            p.time_s = float(time_s[i])
            msg.points.append(p)

        return msg

    def timer_callback(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)
        self.publish_counter += 1

        self.get_logger().info(
            f'已发布 CPS 轨迹 {self.publish_counter}/{self.repeat_count}',
            throttle_duration_sec=0.5
        )

        if self.publish_counter >= self.repeat_count:
            self.get_logger().info('CPS 轨迹发布完成，发布器退出')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CpsTxtPublisher()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
