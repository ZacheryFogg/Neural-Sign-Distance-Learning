import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class ReadDataset(Dataset):
    def __init__(self,  source):
     
        self.data = torch.from_numpy(source).float()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def RandomSplit(datasets, train_set_percentage):
    lengths = [int(len(datasets)*train_set_percentage), len(datasets)-int(len(datasets)*train_set_percentage)]
    return random_split(datasets, lengths)

def GetDataLoaders(npArray, batch_size, train_set_percentage = 0.9, shuffle=True, num_workers=0, pin_memory=True):
    
    
    pc = ReadDataset(npArray)

    train_set, test_set = RandomSplit(pc, train_set_percentage)

    train_loader = DataLoader(train_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    
    return train_loader, test_loader

def interleave_bits(x, y, z, num_bits=10):
    morton_code = 0
    for i in range(num_bits):
        morton_code |= ((x >> i) & 1) << (3 * i)
        morton_code |= ((y >> i) & 1) << (3 * i + 1)
        morton_code |= ((z >> i) & 1) << (3 * i + 2)
    return morton_code

def encode_point_cloud(points, num_bits=10):
    morton_codes = []
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    range_coords = max_coords - min_coords
    scaled_points = ((points - min_coords) / range_coords) #scales points on range [0, 1]
    scaled_points = (scaled_points * (2**num_bits - 1)).astype(int) # scales points to [0, 2^num_bits]
    
    for p in scaled_points:
        x, y, z = p
        morton_code = interleave_bits(x, y, z, num_bits)
        morton_codes.append(morton_code)
    
    sorted_indices = np.argsort(morton_codes)
    sorted_points = points[sorted_indices]
    
    return sorted_points