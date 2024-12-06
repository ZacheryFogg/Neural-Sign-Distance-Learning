import os
from tqdm import tqdm
import numpy as np
import open3d as o3d
import subprocess

def get_file_paths(base_folder):
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            file_path = os.path.join(root, file)
            yield file_path

def sdf_command(file_path):
    write_file_path = "./vdbs/" + file_path[13:-4] + ".vdb"
    directory_path = os.path.dirname(write_file_path)
    os.makedirs(directory_path, exist_ok=True)
    command = ["vdb_tool", "--read", "./temp.ply", "--points2ls", "--write", "codec=blosc", write_file_path]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file_path}: {e}")

file_path_gen = get_file_paths('./ModelNet40')
total_files = sum([len(files) for _, _, files in os.walk('./ModelNet40')])
print('Total Files: ', total_files)
for file_path in tqdm(file_path_gen, total=total_files, desc="Processing Files"):
    mesh = o3d.io.read_triangle_mesh(file_path)    
    sampled_point_cloud = mesh.sample_points_uniformly(number_of_points=1000000)
    sampled_point_cloud.normals = o3d.utility.Vector3dVector([])
    o3d.io.write_point_cloud('./temp.ply', sampled_point_cloud)
    sdf_command(file_path)