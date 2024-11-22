import numpy as np
import open3d as o3d

def get_off_vertices(file_path):
    """
    Load a .off file and extract the vertices
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
    return  num_vertices, vertices

def get_point_cloud(vertices):
     # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    return point_cloud

def get_uniform_point_cloud(file_path, num_points=10000):
    mesh = o3d.io.read_triangle_mesh(file_path)
    sampled_point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
    return sampled_point_cloud

def visualize_point_cloud(point_cloud):
    """
    Visualize a point cloud using Open3D.
    """
    o3d.visualization.draw_geometries([point_cloud])

file_path = "../ModelNet40/cup/train/cup_0001.off"  # Replace with your .off file path
num_vertices, vertices = get_off_vertices(file_path)
point_cloud = get_point_cloud(vertices)

# Visualize the point cloud
visualize_point_cloud(point_cloud) 
