import numpy as np
from scipy.interpolate import interp1d
from .rosbag2_reader_py import Rosbag2Reader
from .utils import mse, rmse, mae
import matplotlib.pyplot as plt

def compute_metrics(path_to_bag):
    reader = Rosbag2Reader(path_to_bag)
    reader.set_filter(['/odom', '/ground_truth', '/ekf'])

    # Data collection
    data = {'/odom': [], '/ground_truth': [], '/ekf': []}
    timestamps = {'/odom': [], '/ground_truth': [], '/ekf': []}

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
    gt_interpolated_ekf = gt_interp_func(timestamps['/ekf'])

    # Check for NaN or Inf in the interpolated ground truth data
    print("NaN in Interpolated Ground Truth for Odom:", np.any(np.isnan(gt_interpolated_odom)))
    print("Inf in Interpolated Ground Truth for Odom:", np.any(np.isinf(gt_interpolated_odom)))
    print("NaN in Interpolated Ground Truth for EKF:", np.any(np.isnan(gt_interpolated_ekf)))
    print("Inf in Interpolated Ground Truth for EKF:", np.any(np.isinf(gt_interpolated_ekf)))
    diff_odom = data['/odom'] - gt_interpolated_odom
    print("NaN in Differences (Odom):", np.any(np.isnan(diff_odom)))
    print("Inf in Differences (Odom):", np.any(np.isinf(diff_odom)))
    
    diff_odom = data['/ekf'] - gt_interpolated_ekf
    print("NaN in Differences (ekf):", np.any(np.isnan(diff_odom)))
    print("Inf in Differences (ekf):", np.any(np.isinf(diff_odom)))

    # Computing Metrics
    metrics = {}
    for topic in ['/odom', '/ekf']:
        diff = data[topic] - gt_interp_func(timestamps[topic])
        metrics[topic] = {
            'RMSE': np.sqrt(np.mean(np.sum(diff**2, axis=1))),
            'MAE': np.mean(np.abs(diff))
    }
    
    print("Interpolated Ground Truth (Odometry):", gt_interpolated_odom)
    print("Interpolated Ground Truth (EKF):", gt_interpolated_ekf)

    plt.figure(figsize=(10, 6))
    for topic in ['/odom', '/ekf']:
        plt.plot(data[topic][:, 0], data[topic][:, 1], label=f'{topic} Path')
    plt.plot(data['/ground_truth'][:, 0], data['/ground_truth'][:, 1], 'k--', label='Ground Truth')
    plt.legend()
    plt.title("Path Comparison")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()

    return metrics

# Usage
path_to_bag = "/home/ubuntu2204/ros2_ws/src/ekf_localization/ekf_localization/rosbagtask3"
metrics = compute_metrics(path_to_bag)
print(metrics)

    


