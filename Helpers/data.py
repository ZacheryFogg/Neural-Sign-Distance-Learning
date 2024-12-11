import os
import math
import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from AutoEncoder.encoder import PointCloudAutoEncoder

class PointCloudDataset(Dataset):
    def __init__(self,base_dir, point_cloud_size = 3072, split = 'train', object_classes = None, order_points = True):
        
        self.point_cloud_size = point_cloud_size
        self.point_clouds = None
        self.split = split
        self.base_dir = base_dir
        self.object_classes = object_classes
        self.order_points = order_points
        self.files_paths = self.get_file_paths()
        self.file_names = [file.split('\\')[-1].split('.')[0] for file in self.files_paths]
        self.point_clouds = self.get_uniform_point_clouds()

    def get_point_cloud_by_name(self, name):
        idx = self.file_names.index(name)
        return self.point_clouds[idx] 
    
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        return {
            "points": self.point_clouds[idx],
            "filename" : self.file_names[idx]
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
                    # print(full_path, self.split)
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
        for file in self.files_paths: 
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
        

class SDDataset(Dataset):
    def __init__(self,model_net_dir, sdf_dir, device, point_cloud_size = 3072, split = 'train', object_classes = None, order_points = True):
        
        self.split = split 
        self.device = device
        self.model_net_dir = model_net_dir
        self.sdf_dir = sdf_dir
        self.point_size = point_cloud_size
        self.object_classes = object_classes
        self.order_points = order_points
        self.num_xyzs = 10000
        self.point_clouds_ds = PointCloudDataset(self.model_net_dir, self.point_size, self.split, self.object_classes , self.order_points)
        # self.point_clouds_ds = torch.load('../Data/point_cloud_dataset_full_1024_train.pt', weights_only = False)
        self.file_names = self.point_clouds_ds.file_names

        self.latent_reps = self.compute_latent_reps() 
        self.sdfs = self.retrieve_sdfs()

    def __len__(self):
        return len(self.file_names) * self.num_xyzs

    def __getitem__(self, idx):
        file_idx = math.ceil((idx + 1) / self.num_xyzs) - 1 # Index of specific point cloud: Range 0 - Number of Pointclouds in this plit
        name = self.file_names[file_idx] # File name of the point cloud

        sd_idx = idx % self.num_xyzs # Index of the specific (xyz, sd). Range 0 - 10000

        return {
            'latent_rep' : self.latent_reps[name],
            'xyz' : self.sdfs[name][0][sd_idx,:],
            'sd' : self.sdfs[name][1][sd_idx]
        }

    def compute_latent_reps(self):
        '''
        Return a dictionary of: {filepath: latent_rep}
        ''' 
        encoder = PointCloudAutoEncoder(model_type = '800T', point_size= str(self.point_size), path_to_weight_dir='../AutoEncoder/')
        encoder.to(self.device)
        encoder.eval()

        latent_reps = {}

        for name in self.file_names:
            with torch.no_grad():
                point_cloud = self.point_clouds_ds.get_point_cloud_by_name(name)
                point_cloud = point_cloud.T.unsqueeze(0).to(self.device)

                latent_rep = encoder(point_cloud).squeeze().to('cpu')

                latent_reps[name] = latent_rep

        return latent_reps
    
    def retrieve_sdfs(self):
        '''
        For each file 
        '''
        sdfs = {}

        sdf_file_paths = []

        for root, _, files in os.walk(self.sdf_dir):
            for file in files:
                if file.endswith('.txt'):
                    full_path = os.path.join(root, file)
                    if f'{self.split}' in full_path:
                        if self.object_classes is not None:
                            if file.split('_')[0] in self.object_classes:
                                sdf_file_paths.append(full_path)
                        else: 
                            sdf_file_paths.append(full_path)

        for path in sdf_file_paths:
            name = path.split('\\')[-1].replace('.vdb', '').split('.')[0]

            if name in self.file_names:

                line = np.loadtxt(path, skiprows=1, delimiter = ' ', dtype=np.float32)
                points = torch.tensor(line[:self.num_xyzs, 0:3])
                sds = torch.tensor(line[:self.num_xyzs,3])

                sdfs[name] = (points, sds)    
                
        return sdfs
    