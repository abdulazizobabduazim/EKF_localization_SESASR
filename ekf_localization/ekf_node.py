import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from landmark_msgs.msg import LandmarkArray  # Ensure this package is available
import numpy as np
from .probabilistic_models import sample_velocity_motion_model,velocity_mm_simpy
from .ekf import RobotEKF
import yaml


class EKF_NODE(Node):
    def __init__(self):
        # Initialize the parent Node class with the name 'controller'
        super().__init__('ekf_node')

        # Create a publisher for the /cmd_topic with message type Twist
        self.ekf_pub = self.create_publisher(Odometry, '/ekf',10)
        self.odom_sub=self.create_subscription(Twist,'/cmd_vel',self.odom_callback,10)
        self.landmarks_sub = self.create_subscription(LandmarkArray, '/landmarks', self.landmarks_callback, 10)
        
        # Timer for the prediction step at 20 Hz
        self.timer = self.create_timer(1 / 20.0, self.prediction_step)

        # EKF State: position (x, y) and orientation (theta)
        self.state = np.array([0.0, 0.0, 0.0])  # Initial state estimate
        self.covariance = np.eye(3)  # Initial covariance matrix
        self.noise_params = np.array([0.05, 0.1, 0.05, 0.1, 0.025, 0.025])  # Motion model noise parameters
        _, eval_Gt, eval_Vt = velocity_mm_simpy()
        # EKF Initialization
        self.ekf = RobotEKF(
            dim_x=3,
            dim_u=2,
            eval_gux=sample_velocity_motion_model,
            eval_Gt=eval_Gt,
            eval_Vt=eval_Vt,
        )

        # Control inputs and last odometry data
        self.velocity = 1e-10
        self.angular_velocity = 1e-10    
        self.dt=0.05    
        # Load landmarks from the YAML file
        self.landmarks = self.load_landmarks('/home/ubuntu2204/ros2_ws/src/turtlebot3_perception/turtlebot3_perception/config/landmarks.yaml')

    def load_landmarks(self, yaml_file_path):
        """Load landmark positions from the YAML file."""
        with open(yaml_file_path, 'r') as file:
            landmarks = yaml.safe_load(file)['landmarks']
            return {
                id_: (x, y) for id_, x, y in zip(landmarks['id'], landmarks['x'], landmarks['y'])
            }


    def odom_callback(self, msg1):

        self.velocity = msg1.linear.x
        self.angular_velocity = msg1.angular.z
        if self.velocity <= 1e-10 or self.angular_velocity<=1e-10:
            self.velocity=1e-10
            self.angular_velocity=1e-10
        

 
    def prediction_step(self):
        # Perform the prediction step of the EKF
        control = np.array([self.velocity, self.angular_velocity])  # Use v and ω from /odom
        self.ekf.predict(control, self.noise_params, (self.dt,))  # Predict the new state
        self.state = self.ekf.mu  # Update the EKF state
        self.covariance = self.ekf.Sigma  # Update the covariance
        self.publish_state()  # Publish the predicted state to /ekf

    def landmarks_callback(self, msg2):
        # Perform the update step of the EKF using landmark measurements
        for landmark in msg2.landmarks:
            z = np.array([landmark.range, landmark.bearing])
            if landmark.id in self.landmarks:
                global_position = self.landmarks[landmark.id]
                #Ht_args = (self.state, global_position)
                self.ekf.update(
                    z,
                    eval_hx=self.measurement_model,
                    eval_Ht=self.measurement_jacobian,
                    Qt=np.diag([0.1, 0.1]),  # Measurement noise covariance
                    Ht_args = (self.state, global_position),
                    hx_args=(self.state, global_position),

                )
        self.state = self.ekf.mu
        self.covariance = self.ekf.Sigma
        self.publish_state()

    def publish_state(self):
        # Publish the EKF estimated state to /ekf
        ekf_msg = Odometry()
        ekf_msg.header.stamp = self.get_clock().now().to_msg()
        ekf_msg.pose.pose.position.x = self.state[0]
        ekf_msg.pose.pose.position.y = self.state[1]
        ekf_msg.pose.pose.orientation.z = np.sin(self.state[2] / 2)
        ekf_msg.pose.pose.orientation.w = np.cos(self.state[2] / 2)
        print(ekf_msg.pose.pose.position.x)
        self.ekf_pub.publish(ekf_msg)
     
    def measurement_model(self, x, landmark_position):
        # Expected measurement based on the current state and a given landmark
        dx = landmark_position[0] - x[0]
        dy = landmark_position[1] - x[1]
        range_ = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx) - x[2]
        return np.array([range_, bearing])

    def measurement_jacobian(self, x, landmark_position):
        # Jacobian of the measurement model with respect to the state
        dx = landmark_position[0] - x[0]
        dy = landmark_position[1] - x[1]
        q = dx**2 + dy**2
        sqrt_q = np.sqrt(q)

        Ht = np.zeros((2, 3))
        Ht[0, 0] = -dx / sqrt_q
        Ht[0, 1] = -dy / sqrt_q
        Ht[1, 0] = dy / q
        Ht[1, 1] = -dx / q
        Ht[1, 2] = -1
        return Ht 
    
def main(args=None):
    rclpy.init(args=args)
    node = EKF_NODE()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()