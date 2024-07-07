import numpy as np
import trimesh

def generate_mesh_from_dense_point_cloud(dense_point_cloud_file):
    # Load dense point cloud data
    dense_point_cloud = np.load(dense_point_cloud_file)
    
    # Reshape to (N, 3) where N is the number of points
    vertices = dense_point_cloud.reshape(-1, 3)
    
    # Perform Delaunay triangulation to generate mesh
    mesh = trimesh.Trimesh(vertices=vertices, process=False)  # Disable automatic processing for direct control
    
    return mesh

# Example usage
dense_point_cloud_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\4_DensePoint\dp.npy'

# Generate mesh from dense point cloud
mesh = generate_mesh_from_dense_point_cloud(dense_point_cloud_file)

# Save mesh as OBJ file
output_obj_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\5_MeshGeneration\denseMesh.obj'
mesh.export(output_obj_file, file_type='obj')

print(f"Mesh saved as OBJ file: {output_obj_file}")
