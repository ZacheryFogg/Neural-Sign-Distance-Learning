##############################
# Imports & Dataset Creation #
##############################

import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from torch.utils.data import Dataset, DataLoader, Subset
from dataclasses import dataclass
import sys
from pathlib import Path
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
import autoencoders as ae
import json


sys.path.append(str(Path.cwd().parent))

from Helpers.data import PointCloudDataset
import Helpers.PointCloudOpen3d as pc

if torch.cuda.is_available():
    device = "cuda"

elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f'Using: {device}')


train_dataset_3072 = PointCloudDataset("../Data/ModelNet40", 3072, 'train', object_classes = None )
test_dataset_3072 = PointCloudDataset("../Data/ModelNet40", 3072, 'test', object_classes = None)

test_size = len(test_dataset_3072)
split_idx = test_size // 2
indices = list(range(test_size))
val_dataset_3072 = Subset(test_dataset_3072, indices[:split_idx])
test_dataset_3072 = Subset(test_dataset_3072, indices[split_idx:])

train_loader_3072 = DataLoader(train_dataset_3072, batch_size = 64, shuffle = True)
val_loader_3072 = DataLoader(val_dataset_3072, batch_size = 128, shuffle = False)
test_loader_3072 = DataLoader(test_dataset_3072, batch_size = 128, shuffle = False)

train_dataset_1024 = PointCloudDataset("../Data/ModelNet40", 1024, 'train', object_classes =  None )
test_dataset_1024 = PointCloudDataset("../Data/ModelNet40", 1024, 'test', object_classes =  None)

test_size = len(test_dataset_1024)
split_idx = test_size // 2
indices = list(range(test_size))
val_dataset_1024 = Subset(test_dataset_1024, indices[:split_idx])
test_dataset_1024 = Subset(test_dataset_1024, indices[split_idx:])

train_loader_1024= DataLoader(train_dataset_1024, batch_size = 64, shuffle = True)
val_loader_1024 = DataLoader(val_dataset_1024, batch_size = 128, shuffle = False)
test_loader_1024 = DataLoader(test_dataset_1024, batch_size = 128, shuffle = False)





##############################
#    Training Function       #
##############################


def train_model(key, model, num_epochs, train_loader, val_loader, size):

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    min_val_loss = np.inf

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):

        # Train one epoch
        train_loss = 0 

        for data in train_loader:
            
            x = data['points'].to(device)

            reconstructed_x = model(x.permute(0,2,1)) # Model expects point clouds to be (3, num_points)
            optimizer.zero_grad()
            loss, _ = chamfer_distance(x, reconstructed_x)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Calculate validation loss

        val_loss = 0 

        for data in val_loader:

            x = data['points'].to(device)

            with torch.no_grad():
                reconstructed_x = model(x.permute(0,2,1))
                loss, _ = chamfer_distance(x, reconstructed_x)
                val_loss+= loss.item()

        val_loss /= len(val_loader)

        # print(f'\nEpoch {epoch+1} \t Train Loss: {train_loss:.5f} \t Val Loss: {val_loss:.5f}')

        # Save best model
        if val_loss < min_val_loss:
            # print(f'Val Loss Decreased({min_val_loss:.6f} ---> {val_loss:.6f}) \t Saving The Model')
            min_val_loss = val_loss

            torch.save(model.state_dict(), f'./trained_autoencoders/{size}/{key}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses


##############################
#         3072 & 512         #
##############################

num_epochs = 300
point_size = 3072 
latent_size = 512

models = {
    'ConvMax' : ae.ConvMax_AE,
    'ConvOnly' : ae.ConvOnly_AE,
    'MLP' : ae.MLP_AE,
    'Conv_7500T' : ae.ConvAE_7500T,
    'Conv_6800T' : ae.ConvAE_6800T,
    'Conv_6300T' : ae.ConvAE_6300T,
    'Conv_1600T' : ae.ConvAE_1600T,
    'Conv_800T' : ae.ConvAE_800T,
    'Conv_270T' : ae.ConvAE_270T,
}

training_results_3072 = {}


for key in models.keys(): 

    print(f'Training {key} model')
    
    model = models[key](point_size, latent_size).to(device)

    train_losses, val_losses = train_model(key, model, num_epochs, train_loader_3072, val_loader_3072, '3072_512')
    
    training_results_3072[key] = {'train_losses' : train_losses, 'val_losses' : val_losses}


with open('./results/3072_512.json', 'w') as f:
    json.dump(training_results_3072, f)


##############################
#         1024 & 256         #
##############################


num_epochs = 300
point_size = 1024
latent_size = 256

models = {
    'ConvMax' : ae.ConvMax_AE,
    'ConvOnly' : ae.ConvOnly_AE,
    'MLP' : ae.MLP_AE,
    'Conv_7500T' : ae.ConvAE_7500T,
    'Conv_6800T' : ae.ConvAE_6800T,
    'Conv_6300T' : ae.ConvAE_6300T,
    'Conv_1600T' : ae.ConvAE_1600T,
    'Conv_800T' : ae.ConvAE_800T,
    'Conv_270T' : ae.ConvAE_270T,
}

training_results_1024= {}


for key in models.keys(): 

    print(f'Training {key} model')
    
    model = models[key](point_size, latent_size).to(device)

    train_losses, val_losses = train_model(key, model, num_epochs, train_loader_1024, val_loader_1024, '1024_256')
    
    training_results_1024[key] = {'train_losses' : train_losses, 'val_losses' : val_losses}

with open('./results/1024_256.json', 'w') as f:
    json.dump(training_results_1024, f)
    


##############################
#         Plot Results       #
##############################


epochs = list(range(num_epochs -2))

for key in training_results_3072.keys():
    plt.plot(epochs, training_results_3072[key]['train_losses'][2:], label = key)

plt.title('Training Loss 3072')
plt.legend()
plt.show()


for key in training_results_3072.keys():
    plt.plot(epochs, training_results_3072[key]['val_losses'][2:], label = key)

plt.title('Validation Loss 3072')
plt.legend()
plt.show()

for key in training_results_1024.keys():
    plt.plot(epochs, training_results_1024[key]['train_losses'][2:], label = key)

plt.title('Training Loss 1024')
plt.legend()
plt.show()


for key in training_results_1024.keys():
    plt.plot(epochs, training_results_1024[key]['val_losses'][2:], label = key)

plt.title('Validation Loss 1024')
plt.legend()
plt.show()

