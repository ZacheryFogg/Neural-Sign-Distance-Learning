import torch 
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from AutoEncoder.encoder import PointCloudAutoEncoder
from Helpers.SDFDataset import SDFDataset
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

# Example of a batch of point clouds being encoded to latent reps
point_size = 3072
latent_shape = 512
encoder = PointCloudAutoEncoder(model_type= '800T', point_size= str(point_size), path_to_weight_dir= '../AutoEncoder/')
encoder.eval() # Let torch know that it doesn't need to store activations as there will be no backward pass
latent_rep = encoder(x).to(device)
latent_rep = latent_rep.unsqueeze(1).expand(-1, x.size(1), -1)
print(latent_rep.shape)

# DataLoaders
base_dir='sampled_vdbs/sampled_vdbs'
# object_classes=['airplane','bathtub','bed','bench','bookshelf','bottle','car']
object_classes=['airplane']
train_sdf_dataset = SDFDataset(base_dir,latent_rep, object_classes=object_classes)
test_sdf_dataset = SDFDataset(base_dir,latent_rep, split='test',object_classes=object_classes)

train_loader= DataLoader(train_sdf_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_sdf_dataset, batch_size = 128, shuffle = False)

# Model, Loss, Optimizer
model = SDFRegressionModel(input_dim, latent_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for data in train_loader:
        points = data['points'].to(device).permute(0,2,1) # Reshape (10000,3) to (3,10000)
        labels = data['labels'].to(device) # Shape (10000,1)
        # model.train()
        print(points.shape)
        print(labels.shape)

        optimizer.zero_grad()
        outputs = model(points, latent_rep)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")
