import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import Helpers.PointCloudOpen3d as pc
from old_auto_encoders import PointCloudAutoencoder
import numpy as np



file_path = "../../ModelNet40/cup/train/cup_0001.off" 
num_vertices, vertices = pc.get_off_vertices(file_path)
# Convert vertices to a NumPy array
# TODO: Add padding to point clouds with less vertices less than input_dim
vertices = np.array(vertices, dtype=np.float32).reshape((3, num_vertices)) # Assuming each vertex has 3 coordinates (x, y, z)

# Example usage
input_dim = num_vertices  # TODO: Change to some constant (e.g. 2000)
hidden_dim = 128 # TODO: Maximize hidden dim but ensure reasonable training time
latent_dim = 32

model = PointCloudAutoencoder(input_dim, hidden_dim, latent_dim)

# Create a DataLoader
batch_size = 32
dataset = TensorDataset(torch.tensor(vertices, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

file_path = "../../ModelNet40/cup/train/cup_0001.off"  # Replace with your .off file path
num_vertices, vertices = pc.get_off_vertices(file_path)
point_cloud = pc.get_point_cloud(vertices)

# Visualize the point cloud
pc.visualize_point_cloud(point_cloud) 
