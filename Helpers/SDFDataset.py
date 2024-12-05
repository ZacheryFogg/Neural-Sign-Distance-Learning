import os
import torch
# from PointCloudDataset import PointCloudDataset
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d

class SDFDataset(Dataset):
    def __init__(self,base_dir, latent_rep, split = 'train', object_classes = None):
        
        self.latent_rep = latent_rep
        self.base_dir=base_dir
        self.split = split
        self.train_split_pct=0.8
        self.test_split_pct=0.2
        self.object_classes = object_classes
        self.class_to_paths = self.get_class_to_file_paths_map()
        self.files = self.get_file_paths()
        self.points, self.labels = self.get_point_clouds_and_labels()

    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        return {
            "points": self.points[idx],
            "labels": self.labels[idx],
            "filename" : self.files[idx]
        }

    def get_class_to_file_paths_map(self):
        '''
        Return dictionary of object class to list of file paths.
        If self.object_classes is populated with class names, then only files in those classes will be returned
        '''
        file_paths = {}
        for object_class in self.object_classes:
            file_paths[object_class] = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.txt'):
                    full_path = os.path.join(root, file)
                    object_class = file.split('_')[0] 
                    if self.object_classes is not None:
                        if object_class in self.object_classes:
                            file_paths[object_class].append(full_path)
                    else: 
                        file_paths[object_class].append(full_path)
        return file_paths
    
    def train_test_split(self):
        for object_class, file_paths in self.class_to_paths.items():
            if self.split == 'train':
                self.file_paths[object_class] = file_paths[0:self.train_split_pct*len(file_paths)]
            elif self.split == 'test':
                self.file_paths[object_class] = file_paths[0:self.test_split_pct*len(file_paths)]

    def get_file_paths(self):
        return [file_path for _, file_paths in self.class_to_paths.items() for file_path in file_paths]    
    
    def get_point_clouds_and_labels(self):
        '''
        Return two tensors for point clouds and sdfs
        '''
        point_clouds_list = []
        sdf_list = []
        file = self.files[1]
        for file in self.files: 
            point_cloud_data_and_labels = np.loadtxt(file, skiprows=1,delimiter=' ', dtype=np.float32) 
            point_cloud_data = point_cloud_data_and_labels[:,0:3]
            point_cloud_labels = point_cloud_data_and_labels[:,3]
            point_clouds_list.append(torch.tensor(point_cloud_data ,dtype = torch.float32))
            sdf_list.append(torch.tensor(point_cloud_labels ,dtype = torch.float32))
            # PointCloudOpen3d.visualize_point_cloud(get_point_cloud(point_cloud_data))
            # print(points_and_labels)
        return point_clouds_list, sdf_list
    