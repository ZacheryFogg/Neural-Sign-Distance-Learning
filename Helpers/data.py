import os
import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d

class PointCloudDataset(Dataset):
    def __init__(self,base_dir, point_cloud_size = 5000, split = 'train', object_classes = None, order_points = True):
        
        self.point_cloud_size = point_cloud_size
        self.point_clouds = None
        self.split = split
        self.base_dir = base_dir
        self.object_classes = object_classes
        self.order_points = order_points

        self.files = self.get_file_paths()
        self.point_clouds = self.get_uniform_point_clouds()


    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        return {
            "points": self.point_clouds[idx],
            "filename" : self.files[idx]
        }
    
    def get_file_paths(self):
        '''
        Return list of all filepaths in ModelNet40 that are part of split (train or test)
        If self.object_classes is populated with class names, then only files in those classes will be returned
        '''
        file_paths = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.off'):
                    full_path = os.path.join(root, file)
                    if f'{self.split}' in full_path:
                        if self.object_classes is not None:
                            if file.split('_')[0] in self.object_classes:
                                file_paths.append(full_path)
                        else: 
                            file_paths.append(full_path)
        return file_paths
    
    def norm(self, x):
        x_min = x.min()
        x_max = x.max()

        x_norm = (x - x_min) / (x_max - x_min)
        
        return x_norm
    
    def interleave_bits(self, x, y, z, num_bits=10):
        morton_code = 0
        for i in range(num_bits):
            morton_code |= ((x >> i) & 1) << (3 * i)
            morton_code |= ((y >> i) & 1) << (3 * i + 1)
            morton_code |= ((z >> i) & 1) << (3 * i + 2)
        return morton_code

    def encode_point_cloud(self, points, num_bits=10):
        morton_codes = []
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        range_coords = max_coords - min_coords
        scaled_points = ((points - min_coords) / range_coords) #scales points on range [0, 1]
        scaled_points = (scaled_points * (2**num_bits - 1)).astype(int) # scales points to [0, 2^num_bits]
        
        for p in scaled_points:
            x, y, z = p
            morton_code = self.interleave_bits(x, y, z, num_bits)
            morton_codes.append(morton_code)
        
        sorted_indices = np.argsort(morton_codes)
        sorted_points = points[sorted_indices]
        
        return sorted_points

    def get_uniform_point_clouds(self):
        '''
        Return a tensor that is all point clouds of fixed size
        '''
        
        point_clouds_list = []
        for file in self.files: 
            mesh = o3d.io.read_triangle_mesh(file)
            try: 
                sampled_point_cloud = mesh.sample_points_uniformly(number_of_points = self.point_cloud_size)
                points = np.asanyarray(sampled_point_cloud.points)
                if self.order_points:
                    points = self.encode_point_cloud(points, num_bits=10)
                point_clouds_list.append(self.norm(torch.tensor(points ,dtype = torch.float32)))
            
            except RuntimeError: # Some .OFF files are damaged, run repair script
                print(f'Damaged file: {file}')

        return point_clouds_list
        