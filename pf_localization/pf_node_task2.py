import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from landmark_msgs.msg import LandmarkArray  # Ensure this package is available
import numpy as np
from .probabilistic_models import sample_velocity_motion_model, landmark_range_bearing_model, landmark_range_bearing_sensor
from .pf import RobotPF
from .utils import simple_resample, residual, state_mean, stratified_resample, systematic_resample, residual_resample
import yaml
import matplotlib.pyplot as plt

class PF_NODE(Node):
    def __init__(self):
        # Initialize the parent Node class with the name 'controller'
        super().__init__('pf_node')

        # Create a publisher for the /cmd_topic with message type Twist
        self.pf_pub = self.create_publisher(Odometry, '/pf',10)
        self.cmd_sub=self.create_subscription(Twist,'/cmd_vel',self.cmd_vel_callback,10)
        self.landmarks_sub = self.create_subscription(LandmarkArray, '/camera/landmarks', self.update_step, 10)
        #sensor param
        
        self.dt=0.05
        # Timer for the prediction step at 20 Hz
        self.timer = self.create_timer(1 / 20.0, self.prediction_step)

        # general noise parameters
        std_lin_vel = 0.3  # [m/s]
        std_ang_vel = np.deg2rad(3.0)  # [rad/s]
        self.sigma_u = np.array([std_lin_vel, std_ang_vel])

        # Define noise params and Q for landmark sensor model
        std_range = 0.225  # [m]
        std_bearing = np.deg2rad(3.12)  # [rad]
        self.sigma_z = np.array([std_range, std_bearing])
        
        
        # PF Initialization
        self.pf = RobotPF(
            dim_x=3,
            dim_u=2,
            eval_gux=sample_velocity_motion_model,
            resampling_fn=stratified_resample,
            boundaries=[(-3., 3.),(-3., 3.),(-np.pi, np.pi)],
            N=2000
        )
        
        self.pf.initialize_particles()
        
        # Control inputs and last odometry data
        self.velocity = 1e-10
        self.angular_velocity = 1e-10    
           
        # Load landmarks from the YAML file
        self.landmarks = self.load_landmarks('/home/ubuntu2204/ros2_ws/src/pf_localization/pf_localization/landmarks1.yaml')

    def load_landmarks(self, yaml_file_path):
        """Load landmark positions from the YAML file."""
        with open(yaml_file_path, 'r') as file:
            landmarks = yaml.safe_load(file)['landmarks']
            return {
                id_: (x, y) for id_, x, y in zip(landmarks['id'], landmarks['x'], landmarks['y'])
            }

    def cmd_vel_callback(self,msg1):
        self.get_logger().info(f"cmd vel callback")
        self.velocity = msg1.linear.x
        self.angular_velocity = msg1.angular.z
        self.get_logger().info(f"cmd vel: v={self.velocity}, w={self.angular_velocity}")
        if self.velocity <= 1e-10:
            self.velocity=1e-10
        if self.angular_velocity<=1e-10:
            self.angular_velocity=1e-10

    def prediction_step(self):
        # Perform the prediction step of the PF
        self.control = np.array([self.velocity, self.angular_velocity])  # Use v and ω from /odom
        self.pf.predict(self.control, self.sigma_u, (self.dt,))  # Predict the new state
        self.pf.estimate(mean_fn=state_mean,residual_fn=residual,angle_idx=2)
        self.publish_state()  # Publish the predicted state to /pf
    

    def update_step(self, msg2):
        # for each landmark simulate the measurement of the landmark
        for landmark in msg2.landmarks:
            # Get landmark coordinates using the ID from the message
            landmark_coords = self.landmarks.get(landmark.id)            
            landmark_x, landmark_y = landmark_coords
            
            landmark_coords = np.array([landmark_x, landmark_y])
            z = np.array([landmark.range, landmark.bearing])
            # landmarks out of the sensor's FOV will be not detected
            # if any landmark detected by the sensor, update the PF
            if z is not None: 
                # run the correction step of the PF
                self.get_logger().info(f"Landmark {landmark.id} associated with measurement {z}")

                self.pf.update(
                    z,
                    self.sigma_z,
                    eval_hx=landmark_range_bearing_model,
                    hx_args=(landmark_coords,self.sigma_z)
                    )

            else:
                self.get_logger().info(f"Landmark {landmark.id} out of sensor range or FOV")

        # after the update of the weights with the measurements, we normalize the weights to make them probabilities
        self.pf.normalize_weights()

        # resample if too few effective particles
        neff = self.pf.neff()
        if neff < self.pf.N / 2:
            self.get_logger().info(f"Effective particle count: {neff}")

            self.pf.resampling(
                resampling_fn=self.pf.resampling_fn,  # simple, residual, stratified, systematic
                resampling_args=(self.pf.weights,),  # tuple: only pf.weights if using pre-defined functions
            )
            assert np.allclose(self.pf.weights, 1 / self.pf.N)
        # estimate robot mean and covariance from particles

    def publish_state(self):
        # Publish the PF estimated state to /pf
        pf_msg = Odometry()
        pf_msg.header.stamp = self.get_clock().now().to_msg()
        pf_msg.pose.pose.position.x = self.pf.mu[0]
        pf_msg.pose.pose.position.y = self.pf.mu[1]
        pf_msg.pose.pose.orientation.z = np.sin(self.pf.mu[2] / 2)
        pf_msg.pose.pose.orientation.w = np.cos(self.pf.mu[2] / 2)
        self.pf_pub.publish(pf_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PF_NODE()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()