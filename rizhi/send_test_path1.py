#straightpath
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy

class StraightPathPublisher(Node):
    def __init__(self):
        super().__init__('straight_path_publisher')
        
        qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub = self.create_publisher(Path, '/global_path', qos_profile)
        self.publish_straight_path()

    def publish_straight_path(self):
        msg = Path()
        base_stamp = self.get_clock().now()
        msg.header.stamp = base_stamp.to_msg()
        msg.header.frame_id = "odom"

        target_velocity = 1.0  
        
        for i in range(100):
            x = i * 0.1  
            y = 0.0     
            theta = 0.0 
            
            t_seconds = x / target_velocity
            
            pose = PoseStamped()
            pose.header.stamp = (base_stamp + rclpy.time.Duration(nanoseconds=int(t_seconds * 1e9))).to_msg()
            pose.header.frame_id = "odom"
            pose.pose.position.x = x
            pose.pose.position.y = y
            
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            
            msg.poses.append(pose)

        self.pub.publish(msg)
        self.get_logger().info("✅ 直线全局路径已发送！长 10m，期望速度 1.0m/s。")

if __name__ == '__main__':
    rclpy.init()
    node = StraightPathPublisher()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()