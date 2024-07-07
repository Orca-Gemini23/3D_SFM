import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paths to the input files
sparse_point_cloud_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\3_SFM\sparse.npy'
camera_poses_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\3_SFM\camera_poses.npy'

# Load the sparse point cloud and camera poses
sparse_point_cloud = np.load(sparse_point_cloud_file)
camera_poses = np.load(camera_poses_file, allow_pickle=True)

# Extract camera positions from the poses
camera_positions = []
for pose in camera_poses:
    R = pose['rotation']
    t = pose['translation']
    camera_position = -R.T @ t
    camera_positions.append(camera_position.flatten())

camera_positions = np.array(camera_positions)

# Plotting the sparse point cloud and camera positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D points
ax.scatter(sparse_point_cloud[:, 0], sparse_point_cloud[:, 1], sparse_point_cloud[:, 2], c='b', marker='o', s=1)

# Plot the camera positions
ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='r', marker='^', s=50)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set title
ax.set_title('Sparse 3D Point Cloud and Camera Poses')

# Show plot
plt.show()
