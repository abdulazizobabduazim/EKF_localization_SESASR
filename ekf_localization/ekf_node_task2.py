import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray  # Ensure this package is available
import numpy as np
from .probabilistic_models import sample_velocity_motion_model,velocity_mm_simpy

from .ekf import RobotEKF
import yaml


class EKF_NODE(Node):
    def __init__(self):
        # Initialize the parent Node class with the name 
        super().__init__('ekf_node_2')
        
        # Define the standard deviation (uncertainty) values for odometry and IMU
        self.std_odom = 0.01  # Example: Standard deviation for odometry (linear and angular velocity)
        self.std_imu = 0.01  # Example: Standard deviation for IMU (angular velocity)

        # Create a publisher for the /cmd_topic with message type Twist
        self.ekf_pub = self.create_publisher(Odometry, '/ekf',10)
        self.odom_sub=self.create_subscription(Odometry,'/odom',self.odom_callback,10)
        self.landmarks_sub = self.create_subscription(LandmarkArray, '/landmarks', self.landmarks_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu',self.imu_callback,10)
        # Timer for the prediction step at 20 Hz
        self.timer = self.create_timer(1 / 20.0, self.prediction_step)

        # EKF State: position (x, y) and orientation (theta), velocity (v), angular velocity(w)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Initial state estimate
        self.covariance = np.eye(5)  # Initial covariance matrix for 5 states
        self.noise_params = np.array([0.05, 0.1, 0.05, 0.1, 0.025, 0.025] ) # Motion model noise parameters
        _, eval_Gt, eval_Vt = velocity_mm_simpy()
        
        # EKF Initialization
        self.ekf = RobotEKF(
            dim_x=5, #updated state dimension to 5
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
        
    def ht_odom(self, x):
        """Jacobian for odometry measurements (v, ω)."""
        Ht = np.zeros((2, 5))
        Ht[0, 3] = 1  # v depends on v (linear velocity)
        Ht[1, 4] = 1  # ω depends on ω (angular velocity)
        return Ht

    def ht_imu(self, x):
        """Jacobian for IMU measurements (only ω)."""
        Ht = np.zeros((1, 5))
        Ht[0, 4] = 1  # ω depends on ω (angular velocity)
        return Ht

    def odom_callback(self, msg1):
        """Callback to get odometry data."""
        self.velocity = msg1.twist.twist.linear.x
        self.angular_velocity = msg1.twist.twist.angular.z
        if self.velocity <= 1e-10 :
            self.velocity = 1e-10
        if self.angular_velocity <= 1e-10:
            self.angular_velocity = 1e-10

        # Use odometry data to update velocity estimates (v and ω)
        Q_odom = np.diag([self.std_odom**2, self.std_odom**2])  # Covariance matrix for odometry
        
        # Perform the update step
        self.ekf.update(
            np.array([self.velocity, self.angular_velocity]),  # Odometry velocity data
            eval_hx=self.odometry_measurement_model,
            eval_Ht=self.ht_odom,
            Qt=Q_odom,  # Measurement noise covariance for odometry
            Ht_args=(self.state,),  # Pass state for Jacobian computation
            hx_args=(self.state,),  # Pass state for measurement model computation
        )
        # Update the state and covariance
        self.state = self.ekf.mu
        self.covariance = self.ekf.Sigma
        self.publish_state()  # Publish the updated state    
        

    def imu_callback(self, msg3):
        """Callback to get IMU data (angular velocity)."""
        imu_angular_velocity = msg3.angular_velocity.z
        
        # IMU data to update only the angular velocity (ω)
        Q_imu = np.diag([self.std_imu**2])  # Covariance matrix for IMU (only angular velocity)

        # Perform the update step
        self.ekf.update(
            np.array([imu_angular_velocity]),  # IMU only updates ω
            eval_hx=self.imu_measurement_model,
            eval_Ht=self.ht_imu,
            Qt=Q_imu,  # Measurement noise covariance for IMU
            Ht_args=(self.state,),  # Pass state for Jacobian computation
            hx_args=(self.state,),  # Pass state for measurement model computation
        )
        # Update the state and covariance
        self.state = self.ekf.mu
        self.covariance = self.ekf.Sigma
        self.publish_state()  # Publish the updated state
        

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
        ekf_msg.twist.twist.linear.x = self.state[3] # linear velocity
        ekf_msg.twist.twist.angular.z = self.state[4] # angular velocity
        print(ekf_msg.pose.pose.position.x)
        self.ekf_pub.publish(ekf_msg)

    def odometry_measurement_model(self,x):
        """Measurement model for odometry (corrects v and ω)."""
        return np.array([x[3], x[4]])  # Return velocity and angular velocity from state
    
    def imu_measurement_model(self, x):
        """Measurement model for IMU (corrects ω)."""
        return np.array([x[4]])  # Only update ω using IMU data 
      
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

        Ht = np.zeros((2, 5))
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