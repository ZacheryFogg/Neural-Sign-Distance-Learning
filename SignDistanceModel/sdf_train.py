import torch 
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import ConcatDataset, DataLoader, Subset
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from AutoEncoder.encoder import PointCloudAutoEncoder
from Helpers.SDFDataset import SDFDataset
from Helpers.data import PointCloudDataset
from sdf_models import SDFRegressionModel

# Hyperparameters
input_dim = 3
latent_dim = 512
hidden_dim = 256
learning_rate = 0.001
num_epochs = 100
# Device
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f'Using: {device}')

# Example of a batch of point clouds being encoded to latent reps
point_size = 3072
latent_shape = 512
encoder = PointCloudAutoEncoder(model_type= '800T', point_size= str(point_size), path_to_weight_dir= '../AutoEncoder/')
encoder.eval() # Let torch know that it doesn't need to store activations as there will be no backward pass

# Combined PC and SDF Dataset and DataLoader
object_classes=['airplane']
# object_classes=['airplane','bathtub','bed','bench','bookshelf','bottle','car']
sdf_base_dir='sampled_vdbs/sampled_vdbs'
point_cloud_base_dir= "../../Data/ModelNet40"
dataset = SDFDataset(sdf_base_dir, point_cloud_base_dir, 3072, 'test', object_classes)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Save Dataset and Load DataLoader
# torch.save(dataset, 'combined_dataset')
# dataloader=torch.load('combined_dataset', weights_only=False)

# Model, Loss, Optimizer
model = SDFRegressionModel(input_dim, latent_dim, hidden_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    latent_reps = []
    for data in dataloader:
        point_cloud_filenames = [] # Removes filename directory and extension
        # This loop ensures that the point cloud filename corresponds to the sdf filename, with dir and ext info removed
        for path_to_pc, path_to_sdf in zip(data['pc_filename'], data['sdf_filename']):
            pc_filename = path_to_pc.split('/')[-1].split('.')[0]
            sdf_filename = path_to_sdf.split('/')[-1].split('.')[0]
            # print(f'{pc_filename} {sdf_filename}')
            if pc_filename!= sdf_filename:
                print(f"PointCloudDataset {pc_filename} and SDFDataset {sdf_filename} filenames need to correspond")
                break
            
        sample_sdf, point_cloud, sdf_labels = data['sdf_points'], data['point_clouds'], data['sdf_labels']
        if point_cloud.shape[0] == 64:
            point_cloud = point_cloud.permute(0,2,1)
            latent_rep = encoder(point_cloud)
            latent_rep = latent_rep.unsqueeze(1).repeat(1, 10000, 1).to(device)
            # print(latent_rep.shape)
            sdf_point=sample_sdf.to(device)
            # print(sdf_point.shape)
            labels = sdf_labels.to(device) # Shape (10000,1)
            # print(labels.shape)# shape (64,10000)

            optimizer.zero_grad()
            outputs = model(sdf_point, latent_rep) 
            # print(outputs.shape) # shape (64,10000,1)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")
