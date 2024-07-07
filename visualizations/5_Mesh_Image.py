import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
import cv2
import os

# Define input paths
sparse_point_cloud_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3d_Construction\visualizations\sparse_point_cloud.npy'
input_images_folder = r'C:\Users\csyas\OneDrive\Desktop\projects\3d_Construction\images'

# Load sparse point cloud
points = np.load(sparse_point_cloud_file)

# Perform Delaunay tetrahedralization
tetra = Delaunay(points)

# Extract the surface triangles
triangles = []
for simplex in tetra.simplices:
    for i in range(4):
        triangle = [simplex[j] for j in range(4) if j != i]
        triangle.sort()
        if triangle not in triangles:
            triangles.append(triangle)

# Function to plot the mesh
def plot_mesh(points, triangles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(points[triangles], alpha=0.3))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o', color='r', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Visualize the surface mesh
plot_mesh(points, triangles)

# Optional: Apply textures from images (complex step, requires additional processing)
# This part would involve mapping the textures from the images onto the 3D mesh,
# typically done using UV mapping techniques and more advanced 3D libraries.
