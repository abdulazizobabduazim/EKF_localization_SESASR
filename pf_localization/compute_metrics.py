import numpy as np
from scipy.interpolate import interp1d
from .rosbag2_reader_py import Rosbag2Reader
from .utils_rosbag import mse, rmse, mae
import matplotlib.pyplot as plt
import yaml

def load_landmarks(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        landmarks = yaml.safe_load(file)['landmarks']
        return {
            id_: (x, y) for id_, x, y in zip(landmarks['id'], landmarks['x'], landmarks['y'])
        }
landmark_file_path = "/home/ubuntu2204/ros2_ws/src/turtlebot3_perception/turtlebot3_perception/config/landmarks.yaml"
landmarks = load_landmarks(landmark_file_path)

def compute_metrics(path_to_bag):
    reader = Rosbag2Reader(path_to_bag)
    reader.set_filter(['/odom', '/ground_truth', '/pf'])

    # Data collection
    data = {'/odom': [], '/ground_truth': [], '/pf': []}
    timestamps = {'/odom': [], '/ground_truth': [], '/pf': []}

    for topic_name, msg, t in reader:
        if topic_name in data:
            timestamps[topic_name].append(t)
            data[topic_name].append((msg.pose.pose.position.x, msg.pose.pose.position.y))

    # Convert lists to numpy arrays for interpolation
    for topic in data:
        timestamps[topic] = np.array(timestamps[topic], dtype=float)
        data[topic] = np.array(data[topic])

    gt_interp_func = interp1d(timestamps['/ground_truth'], data['/ground_truth'], axis=0, kind='linear', bounds_error=False, fill_value=(data['/ground_truth'][0], data['/ground_truth'][-1]))
    gt_interpolated_odom = gt_interp_func(timestamps['/odom'])
    gt_interpolated_pf = gt_interp_func(timestamps['/pf'])

    # Check for NaN or Inf in the interpolated ground truth data
    print("NaN in Interpolated Ground Truth for Odom:", np.any(np.isnan(gt_interpolated_odom)))
    print("Inf in Interpolated Ground Truth for Odom:", np.any(np.isinf(gt_interpolated_odom)))
    print("NaN in Interpolated Ground Truth for PF:", np.any(np.isnan(gt_interpolated_pf)))
    print("Inf in Interpolated Ground Truth for PF:", np.any(np.isinf(gt_interpolated_pf)))
    diff_odom = data['/odom'] - gt_interpolated_odom
    print("NaN in Differences (Odom):", np.any(np.isnan(diff_odom)))
    print("Inf in Differences (Odom):", np.any(np.isinf(diff_odom)))
    
    diff_odom = data['/pf'] - gt_interpolated_ekf
    print("NaN in Differences (pf):", np.any(np.isnan(diff_odom)))
    print("Inf in Differences (pf):", np.any(np.isinf(diff_odom)))

    # Computing Metrics
    metrics = {}
    for topic in ['/odom', '/pf']:
        diff = data[topic] - gt_interp_func(timestamps[topic])
        metrics[topic] = {
            'RMSE': np.sqrt(np.mean(np.sum(diff**2, axis=1))),
            'MAE': np.mean(np.abs(diff))
    }
    
    print("Interpolated Ground Truth (Odometry):", gt_interpolated_odom)
    print("Interpolated Ground Truth (PF):", gt_interpolated_pf)

    plt.figure(figsize=(10, 6))
    for topic in ['/odom', '/pf']:
        plt.plot(data[topic][:, 0], data[topic][:, 1], label=f'{topic} Path')
    plt.plot(data['/ground_truth'][:, 0], data['/ground_truth'][:, 1], 'k--', label='Ground Truth')
    # Plot the landmarks
    landmark_x = [coord[0] for coord in landmarks.values()]
    landmark_y = [coord[1] for coord in landmarks.values()]
    plt.scatter(landmark_x, landmark_y, c='red', marker='x', label='Landmarks')
    
    plt.legend()
    plt.title("Path Comparison with Landmarks")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()

    return metrics

# Usage
path_to_bag = "/home/ubuntu2204/ros2_ws/src/pf_localization/pf_localization/rosbag01"
metrics = compute_metrics(path_to_bag)
print(metrics)

    


