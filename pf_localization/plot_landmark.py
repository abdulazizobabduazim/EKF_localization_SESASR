import rclpy
from rclpy.node import Node
from landmark_msgs.msg import LandmarkArray
from std_msgs.msg import Float32

import numpy as np

class LandmarkPublisher(Node):
    def __init__(self):
        super().__init__('landmark_publisher')
        self.subscription = self.create_subscription(
            LandmarkArray,
            '/camera/landmarks',
            self.landmark_callback,
            10
        )
        self.pub_x = self.create_publisher(Float32, '/landmark_x', 10)
        self.pub_y = self.create_publisher(Float32, '/landmark_y', 10)

    def landmark_callback(self, msg):
        for landmark in msg.landmarks:
            # Calculate x, y from range and bearing
            landmark_x = landmark.range * np.cos(landmark.bearing)
            landmark_y = landmark.range * np.sin(landmark.bearing)

            # Publish x and y separately
            self.pub_x.publish(Float32(data=landmark_x))
            self.pub_y.publish(Float32(data=landmark_y))

            # Optional: Log the results for debugging
            self.get_logger().info(f"Landmark ID: {landmark.id}, X: {landmark_x}, Y: {landmark_y}")

def main():
    rclpy.init()
    node = LandmarkPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
