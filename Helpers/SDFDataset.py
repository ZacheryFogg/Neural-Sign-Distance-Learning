import os
import torch

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from Helpers.PointCloudDataset import PointCloudDataset
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d

'''PyTorch Dataset class that stores SDF testing points (x,y,z) and their corresponding labels.
Labels are the signed distances between (x,y,z) and the corresponding point cloud in off file.'''
class SDFDataset(Dataset):
    def __init__(self,sdf_base_dir, point_cloud_base_dir, point_cloud_size, split='train', object_classes = None):
        self.point_cloud_base_dir = point_cloud_base_dir
        self.point_cloud_size = point_cloud_size
        self.sdf_base_dir=sdf_base_dir
        self.split = split
        self.train_split_pct=0.8
        self.test_split_pct=0.2
        self.object_classes = object_classes

        self.pc_dataset = PointCloudDataset(self.point_cloud_base_dir, self.point_cloud_size, split, object_classes)
        # self.pc_dataset = torch.load('train_dataset_3072_airplanes', weights_only = False)
        self.pc_dataset_file_names = [file.split('/')[-1] for file in self.pc_dataset.files] #get off file names
        self.class_to_paths = self.get_class_to_sdf_file_paths_map()
        self.sdf_files = self.get_file_paths()
        self.sdf_points, self.sdf_labels = self.get_sdf_points_and_labels()

    def __len__(self):
        return len(self.sdf_points)
    
    def __getitem__(self, idx):
        return {
            "sdf_points": self.sdf_points[idx],
            "sdf_labels": self.sdf_labels[idx],
            "sdf_filename": self.sdf_files[idx],
            "pc_filename" : self.pc_dataset.files[idx],
            "point_clouds": self.pc_dataset.point_clouds[idx]
        }   

    def get_class_to_sdf_file_paths_map(self):
        '''
        Return dictionary of object class to list of file paths.
        If self.object_classes is populated with class names, then only files in those classes will be returned.
        Otherwise, if self.object_classes is None, files in all classes will be returned.
        '''
        print('classtopaths')
        file_paths = {}
        for object_class in self.object_classes:
            file_paths[object_class] = []
        for root, _, files in os.walk(self.sdf_base_dir):
            for file in files:
                off_file = file.replace('.vdb.txt', '.off')
                if file.endswith('.txt'):
                    full_path = os.path.join(root, file)
                    object_class = file.split('_')[0]
                    if off_file in self.pc_dataset_file_names:
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
    
    def get_sdf_points_and_labels(self):
        '''
        Return two tensors for sdf point and sdf labels
        '''
        point_clouds_list = []
        sdf_list = []
        for file in self.sdf_files: 
            point_data_and_labels = np.loadtxt(file, skiprows=1,delimiter=' ', dtype=np.float32) 
            point_data = point_data_and_labels[:,0:3]
            point_cloud_labels = point_data_and_labels[:,3]
            point_clouds_list.append(torch.tensor(point_data ,dtype = torch.float32))
            sdf_list.append(torch.tensor(point_cloud_labels ,dtype = torch.float32))
            # PointCloudOpen3d.visualize_point_cloud(get_point_cloud(point_cloud_data))
            # print(points_and_labels)
        return point_clouds_list, sdf_list   