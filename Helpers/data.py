import os
import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d

class PointCloudDataset(Dataset):
    def __init__(self,base_dir, point_cloud_size = 5000, split = 'train'):
        
        self.point_cloud_size = point_cloud_size
        self.point_clouds = None
        self.split = split
        self.base_dir = base_dir

        self.files = self.get_file_paths(self.split, self.base_dir)
        self.point_clouds = self.get_uniform_point_clouds()


    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        return {
            "points": self.point_clouds[idx],
            "filename" : self.files[idx]
        }
    
    def get_file_paths(self, split, base_dir):
        '''
        Return list of all filepaths in ModelNet40 that are part of split (train or test)
        '''
        file_paths = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.off'):
                    full_path = os.path.join(root, file)
                    if f'{split}' in full_path:
                        file_paths.append(full_path)
        print(len(file_paths))
        return file_paths
    
    
    def get_uniform_point_clouds(self):
        '''
        Return a tensor that is all point clouds of fixed size
        '''
        
        point_clouds_list = []
        for file in self.files: 
            mesh = o3d.io.read_triangle_mesh(file)
            try: 
                sampled_point_cloud = mesh.sample_points_uniformly(number_of_points = self.point_cloud_size)
                point_clouds_list.append(torch.tensor(np.asanyarray(sampled_point_cloud.points),dtype = torch.float32))
            except RuntimeError: # Some .OFF files are damaged, run repair script
                print(f'Damaged file: {file}')

        return point_clouds_list
        