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

# Random batch of examples
x = torch.rand(64, 3072, 3).permute(0,2,1) # Cloud must be permuted to be (batch_size, xyz, point_size)
object_classes=['airplane']
# object_classes=['airplane','bathtub','bed','bench','bookshelf','bottle','car']
train_dataset_3072 = PointCloudDataset("../../Data/ModelNet40", 3072, 'train', object_classes = object_classes)
# test_dataset_3072 = PointCloudDataset("../Data/ModelNet40", 3072, 'test', object_classes = object_classes)
train_loader_3072 = DataLoader(train_dataset_3072, batch_size = 64, shuffle = False)

# Example of a batch of point clouds being encoded to latent reps
point_size = 3072
latent_shape = 512
encoder = PointCloudAutoEncoder(model_type= '800T', point_size= str(point_size), path_to_weight_dir= '../AutoEncoder/')
encoder.eval() # Let torch know that it doesn't need to store activations as there will be no backward pass

# DataLoaders
base_dir='sampled_vdbs/sampled_vdbs'
train_sdf_dataset = SDFDataset(base_dir, object_classes=object_classes)
# test_sdf_dataset = SDFDataset(base_dir, split='test',object_classes=object_classes)
sdf_train_loader= DataLoader(train_sdf_dataset, batch_size = 64, shuffle = True)
# sdf_test_loader = DataLoader(test_sdf_dataset, batch_size = 128, shuffle = False)

# Model, Loss, Optimizer
model = SDFRegressionModel(input_dim, latent_dim, hidden_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    latent_reps = []
    for sample_sdf, point_cloud in zip(sdf_train_loader, train_loader_3072):
        if point_cloud['points'].shape[0] == 64:
            point_cloud = point_cloud['points'].permute(0,2,1)
            latent_rep = encoder(point_cloud)
            latent_rep = latent_rep.unsqueeze(1).repeat(1, 10000, 1).to(device)
            print(latent_rep.shape)
            sdf_point=sample_sdf['points'].to(device)
            print(sdf_point.shape)
            labels = sample_sdf['labels'].to(device) # Shape (10000,1)
            # print(labels.shape)# shape (64,10000)

            optimizer.zero_grad()
            outputs = model(sdf_point, latent_rep) 
            print(outputs.shape) # shape (64,10000,1)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")
