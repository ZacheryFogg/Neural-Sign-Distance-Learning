import torch 
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from AutoEncoder.encoder import PointCloudAutoEncoder
from Helpers.SDFDataset import SDFDataset
from Helpers.PointCloudDataset import PointCloudDataset
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

# DataLoaders
object_classes=['airplane']

base_dir='sampled_vdbs/sampled_vdbs'
train_sdf_dataset = SDFDataset(base_dir, object_classes=object_classes)
sdf_train_loader= DataLoader(train_sdf_dataset, batch_size = 64, shuffle = False)

train_dataset_3072 = PointCloudDataset("../../Data/ModelNet40", 3072, 'train', object_classes = object_classes)
train_loader_3072 = DataLoader(train_dataset_3072, batch_size = 64, shuffle = False)

# Load And Save DataLoaders
# object_classes=['airplane','bathtub','bed','bench','bookshelf','bottle','car']
# torch.save(train_sdf_dataset, 'train_sdf_dataset_airplanes')   
# train_sdf_dataset = torch.load('train_sdf_dataset_airplanes', weights_only = False)
# test_sdf_dataset = SDFDataset(base_dir, split='test',object_classes=object_classes)
# sdf_test_loader = DataLoader(test_sdf_dataset, batch_size = 128, shuffle = False)

# train_dataset_3072 = torch.load('train_dataset_3072_airplanes', weights_only = False)
# torch.save(train_dataset_3072, 'train_dataset_3072_airplanes')   
# test_dataset_3072 = PointCloudDataset("../Data/ModelNet40", 3072, 'test', object_classes = object_classes)

# Model, Loss, Optimizer
model = SDFRegressionModel(input_dim, latent_dim, hidden_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    latent_reps = []
    for sample_sdf, point_cloud in zip(sdf_train_loader, train_loader_3072):
        point_cloud_filenames = [] # Removes filename directory and extension
        # This loop ensures that the point cloud filename corresponds to the sdf filename, with dir and ext info removed
        for path_to_pc, path_to_sdf in zip(point_cloud['filename'], sample_sdf['filename']):
            pc_filename = path_to_pc.split('/')[-1].split('.')[0]
            sdf_filename = path_to_sdf.split('/')[-1].split('.')[0]
            # print(f'{pc_filename} {sdf_filename}')
            if pc_filename!= sdf_filename:
                print(f"PointCloudDataset {pc_filename} and SDFDataset {sdf_filename} filenames need to correspond")
                break
            
        if point_cloud['points'].shape[0] == 64:
            point_cloud = point_cloud['points'].permute(0,2,1)
            latent_rep = encoder(point_cloud)
            latent_rep = latent_rep.unsqueeze(1).repeat(1, 10000, 1).to(device)
            # print(latent_rep.shape)
            sdf_point=sample_sdf['points'].to(device)
            # print(sdf_point.shape)
            labels = sample_sdf['labels'].to(device) # Shape (10000,1)
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
