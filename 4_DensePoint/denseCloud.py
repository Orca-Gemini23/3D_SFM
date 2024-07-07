import cv2
import numpy as np

# Define your actual camera intrinsic parameters
fx = 1000.0  # Replace with your focal length in pixels
fy = 1000.0  # Replace with your focal length in pixels
cx = 320.0   # Replace with your principal point x-coordinate in pixels
cy = 240.0   # Replace with your principal point y-coordinate in pixels

def generate_dense_point_cloud(camera_poses_file, sparse_3d_points_file, fx, fy, cx, cy):
    # Load camera poses and sparse points
    camera_poses = np.load(camera_poses_file, allow_pickle=True)
    sparse_points = np.load(sparse_3d_points_file)

    # Define camera intrinsic parameters
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    dense_point_cloud = []

    # Iterate over each camera pose
    for pose in camera_poses:
        R = pose['rotation']
        t = pose['translation']

        # Project sparse points into 3D using the current pose
        projected_points, _ = cv2.projectPoints(sparse_points, R, t, K, distCoeffs=None)
        projected_points = projected_points.squeeze()

        # Add projected points to the dense point cloud
        dense_point_cloud.extend(projected_points)

    # Convert to numpy array
    dense_point_cloud = np.array(dense_point_cloud)

    return dense_point_cloud

# Example usage with placeholders for intrinsic parameters
camera_poses_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\3_SFM\camera_poses.npy'
sparse_3d_points_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\3_SFM\sparse.npy'


# Generate dense point cloud
dense_point_cloud = generate_dense_point_cloud(camera_poses_file, sparse_3d_points_file, fx, fy, cx, cy)

# Save dense point cloud to numpy file
output_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\4_DensePoint\dp.npy'
np.save(output_file, dense_point_cloud)

print(f"Dense point cloud saved to {output_file}")
