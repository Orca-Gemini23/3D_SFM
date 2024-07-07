import numpy as np
from scipy.spatial import Delaunay

# Load sparse point cloud
sparse_point_cloud_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3d_Construction\visualizations\sparse_point_cloud.npy'
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

# Prepare vertices and faces for OBJ file
vertices = points
faces = np.array(triangles)

# Save as OBJ file
output_obj_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3d_Construction\visualizations\surface_mesh.obj'

with open(output_obj_file, 'w') as f:
    # Write vertices
    for vertex in vertices:
        f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
    
    # Write faces (1-based indexing for OBJ format)
    for face in faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

print(f"Surface mesh saved as OBJ file: {output_obj_file}")
