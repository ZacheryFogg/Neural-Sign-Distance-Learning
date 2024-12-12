import numpy as np
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
from tqdm import tqdm
import sdf_models as sd
import json


sys.path.append(str(Path.cwd().parent))

from Helpers.data import SDDataset
import Helpers.PointCloudOpen3d as pc

if torch.cuda.is_available():
    device = "cuda"

elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f'Using: {device}')


# train_dataset = SDDataset('../Data/ModelNet40', '../Data/sampled_points', device, 1024, 'train', object_classes= None)
# test_dataset = SDDataset('../Data/ModelNet40', '../Data/sampled_points', device, 1024, 'test', object_classes= None)

# test_size = len(test_dataset)
# split_idx = test_size // 2
# indices = list(range(test_size))
# val_dataset = Subset(test_dataset, indices[:split_idx])
# test_dataset = Subset(test_dataset, indices[split_idx:])

train_dataset = torch.load('../Data/sd_dataset_train.pt', weights_only= False)
val_dataset = torch.load('../Data/sd_dataset_val.pt', weights_only= False)
test_dataset = torch.load('../Data/sd_dataset_test.pt', weights_only= False)

batch_size = 1024

train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle= True)
test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle= True)


val_report_rate = 30000 # How often should we calculate the validation metrics.

def calculate_acc(logits, targets): 
    preds = torch.round(torch.sigmoid(logits))
    correct = (preds == targets).float() 
    accuracy = torch.sum(correct) / (targets.shape[0])
    return accuracy.item()


def calculate_val_metrics(model, device, val_loader):
    # Calculate validation loss
    val_loss = 0 
    val_acc = 0

    for data in val_loader:

        latent_rep = data['latent_rep'].to(device)
        xyz = data['xyz'].to(device)
        sd = (data['sd'] >  0).float().to(device)

        with torch.no_grad():
            logits = model(latent_rep, xyz)
            logits = logits.squeeze()

            loss = F.binary_cross_entropy_with_logits(logits, sd)
            val_loss += loss.item()
            val_acc += calculate_acc(logits, sd)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    return val_loss, val_acc

def train_model(key, model, num_epochs, train_loader, val_loader, lr = 1e-3):

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    max_val_acc = -np.inf

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):

        # Train one epoch
        model.train()

        train_loss = 0 
        train_acc = 0
        
        batch = 0
        for data in tqdm(train_loader):
 
            latent_rep = data['latent_rep'].to(device)
            xyz = data['xyz'].to(device)
            sd = (data['sd'] > 0).float().to(device)

            logits = model(latent_rep, xyz)
            logits = logits.squeeze()

            optimizer.zero_grad()
            
            loss = F.binary_cross_entropy_with_logits(logits, sd)
        
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_acc = calculate_acc(logits, sd)

            # There are 120,000,000 examples per epoch, and so validation metrics should be calculated more frequently than 1 epoch
            if batch % val_report_rate == 0:
                model.eval()
                val_loss, val_acc = calculate_val_metrics(model, device, val_loader)
                model.train()

                val_losses.append(val_loss)
                val_accs.append(val_acc)

                print(f'\nBatch Num {batch+1} \t Val Loss: {val_loss:.5f} \t Val Acc: {val_acc:.5f}')

                # Save best model
                if val_acc > max_val_acc:
                    print(f'Val Acc Increased({max_val_acc:.6f} ---> {val_acc:.6f}) \t Saving The Model')
                    max_val_acc = val_acc

                    torch.save(model.state_dict(), f'./trained_sdf_models/{key}')

            batch+=1 

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            train_losses.append(train_loss)
            train_accs.append(train_acc) 

    # model.eval()
    val_loss, val_acc = calculate_val_metrics(model, device, val_loader)
    # model.train()
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    results = {
        "train_losses" : train_losses, 
        "train_accs" : train_accs,
        "val_losses" : val_losses, 
        "val_accs" : val_accs
    }

    return results


num_epochs = 2
latent_size = 256
lr = 1e-3

overall_results = {}

models = {
    'SD_3L' : {'class' : sd.SD_3L , 'lr' : lr},
    'SD_3L_Upscale32' : {'class' : sd.SD_3L_Upscale32 , 'lr' : lr},
    'SD_4L_Upscale128' : {'class' : sd.SD_4L_Upscale128 , 'lr' : lr},

    'SD_3L_LR_1e-4' : {'class' : sd.SD_3L , 'lr' : 1e-4},
    'SD_4L_Upscale128_LR_1e-4' : {'class' : sd.SD_4L_Upscale128 , 'lr' : 1e-4},

    'SD_5L_Upscale256' : {'class' : sd.SD_5L_Upscale256 , 'lr' : lr},
    'SD_4L_Upscale256_LatentEncode' : {'class' : sd.SD_4L_Upscale256_LatentEncode , 'lr' : lr},
}

for key in models.keys():
    print(f'Training {key} model')
    
    model = models[key]['class'](latent_size).to(device)

    results = train_model(key, model, num_epochs, train_loader, val_loader, models[key]['lr'])

    with open(f'./results/{key}.json', 'w') as f:
        json.dump(results, f)

    overall_results[key] = results

with open(f'./results/overall_results.json', 'w') as f: 
    json.dump(overall_results, f)
