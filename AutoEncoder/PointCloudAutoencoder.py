import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

"""Current up-to-date Autoencoder Architecture"""
class PointCloudAutoEncoder(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAutoEncoder, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        
        # Archictectures I trained: 
        # 1. Changed conv1 to (3, 64, 1). Kept Conv2 same as PointCloudAE repo - (64, 128, 1). 
        # 2. Changed conv1 to (6, 64, 1). Changed Conv2 to (64, 128, 2).
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 2)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,self.point_size*3)

    def encoder(self, x): 
        # print(f'input: {x.shape}')
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f'relu(bn1(conv1)): {x.shape}')
        x = F.relu(self.bn2(self.conv2(x)))
        # print(f'relu(bn2(conv2)): {x.shape}')
        x = self.bn3(self.conv3(x))
        # print(f'bn3: {x.shape}')
        x = torch.max(x, 2, keepdim=True)[0]
        # print('max: {x.shape}')
        x = x.view(-1, self.latent_size)
        # print(f'end of enc x.shape: {x.shape}')
        return x
    
    def decoder(self, x):
        # print(f'start dec x.shape: {x.shape}')
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        # print(f'end dec x.shape: {x.shape}')
        return x.view(-1, self.point_size, 3)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


