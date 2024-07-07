import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load dense point cloud from npy file
dense_point_cloud_file =  r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\4_DensePoint\dp.npy'
dense_point_cloud = np.load(dense_point_cloud_file)

# Assuming dense_point_cloud is already flattened, reshape it to (N, 3) format
dense_point_cloud = dense_point_cloud.reshape(-1, 3)

# Extract x, y, z coordinates
x = dense_point_cloud[:, 0]
y = dense_point_cloud[:, 1]
z = dense_point_cloud[:, 2]

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', marker='.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Dense Point Cloud Visualization')

plt.show()
