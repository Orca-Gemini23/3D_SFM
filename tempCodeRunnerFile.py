import numpy as np
from scipy.spatial import Delaunay
import open3d as o3d
import os

# Load the sparse point cloud (assuming it's stored as a numpy array)
sparse_point_cloud_file = 'C:\\Users\\csyas\\OneDrive\\Desktop\\projects\\3d_Construction\\sparse_point_cloud.npy'
sparse_point_cloud = np.load(sparse_point_cloud_file)

# Perform Delaunay tetrahedralization
tri = Delaunay(sparse_point_cloud)

# Extract vertices and tetrahedra
vertices = sparse_point_cloud
tetrahedra = tri.simplices

# Create an Open3D mesh from the tetrahedra
mesh = o3d.geometry.TriangleMesh()

# Add vertices to the mesh
mesh.vertices = o3d.utility.Vector3dVector(vertices)

# Convert tetrahedra to triangles for visualization (Open3D doesn't support tetrahedra directly)
triangles = []
for tet in tetrahedra:
    triangles.append([tet[0], tet[1], tet[2]])
    triangles.append([tet[0], tet[1], tet[3]])
    triangles.append([tet[0], tet[2], tet[3]])
    triangles.append([tet[1], tet[2], tet[3]])

mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

# Optionally, compute vertex normals
mesh.compute_vertex_normals()

# Save the mesh to a file
output_folder = 'C:\\Users\\csyas\\OneDrive\\Desktop\\projects\\3d_Construction\\mesh_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
mesh_file = os.path.join(output_folder, 'mesh.ply')
o3d.io.write_triangle_mesh(mesh_file, mesh)

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])

print('Mesh construction completed and saved to:', mesh_file)
