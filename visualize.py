import numpy as np
import open3d as o3d

def load_off(file_path):
    """
    Load a .off file and extract the vertices as a point cloud.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip the first line (it should be "OFF")
    assert lines[0].strip() == "OFF", "The file does not start with 'OFF'"
    
    # Read number of vertices and faces
    num_vertices, num_faces, num_edges = map(int, lines[1].split())
    
    # Read vertices
    vertices = []
    for i in range(2, 2 + num_vertices):
        x, y, z = map(float, lines[i].split())
        vertices.append([x, y, z])
    
    # Convert to numpy array for easy manipulation
    vertices = np.array(vertices)
    print(vertices.shape)
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    
    return point_cloud

def visualize_point_cloud(point_cloud):
    """
    Visualize a point cloud using Open3D.
    """
    o3d.visualization.draw_geometries([point_cloud])

file_path = "ModelNet40/person/train/person_0084.off"  # Replace with your .off file path
point_cloud = load_off(file_path)

# Visualize the point cloud
mesh = o3d.io.read_triangle_mesh(file_path)
sampled_point_cloud = mesh.sample_points_uniformly(number_of_points=10000)
visualize_point_cloud(sampled_point_cloud)