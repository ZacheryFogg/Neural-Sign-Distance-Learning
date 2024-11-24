import torch
import numpy as np
import sys
from PointCloudAutoencoder import PointCloudAutoEncoder
from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance
from torch.utils.data import DataLoader

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Helpers.dataset_helpers import GetDataLoaders
from Helpers.data import PointCloudDataset
import Helpers.PointCloudOpen3d as pc


if torch.cuda.is_available():
    device = "cuda"

elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f'Using: {device}')
point_cloud_size = 5000 
model = PointCloudAutoEncoder(1024,768).to(device)
optim = torch.optim.AdamW(model.parameters(), lr= 0.0005)
epochs = 1000
report_rate = 600
train_dataset = PointCloudDataset("../../ModelNet40", 5000, 'train', object_classes= ['cup', 'bowl', 'bottle', 'cone', 'flower_pot'])
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = False)
pc_array = np.load("../../chair_set.npy")

train_loader, test_loader = GetDataLoaders(npArray=pc_array, batch_size= 64)

# cloud = pc.get_point_cloud(pc_array[0])
# pc.visualize_point_cloud(cloud)

for epoch in range(epochs):
    running_loss = 0 
    batch_count = 0 

    for i, data in enumerate(train_loader):
        x = data.to(device).permute(0,2,1)
        # x = data['points'].to(device).permute(0,2,1)
        # x = F.normalize(x, dim = 2)      
        optim.zero_grad()

        pred = model(x)
        pred = pred.permute(0,2,1)
        # print(x.shape)
        # print(pred.shape)
        # loss = F.mse_loss(pred, x)
        # print(x.shape)
        # print(pred.shape)
        loss, _ = chamfer_distance(x, pred) 

        # cloud = pc.get_point_cloud(x[1].T.to('cpu'))
        # pc.visualize_point_cloud(cloud)
        # p = pred[1].T.to('cpu').detach()

        # cloud = pc.get_point_cloud(p)
        # pc.visualize_point_cloud(cloud)

        loss.backward()
        optim.step()
        running_loss += loss.item()
        batch_count +=1

    if epoch % 10 == 9:
        print(f'Epoch {epoch:<3} Epoch Loss: {running_loss / batch_count}')
        

        # if i % report_rate == report_rate - 1:
        #     print(f'Batch {i:<3} Running Loss: {running_loss / report_rate}')
        #     running_loss = 0
    
def test_trained_point_cloud():
    x = next(iter(train_loader))[1]
    x = F.normalize(x, dim = 0)

    print(type(x))
    cloud = pc.get_point_cloud(x)
    pc.visualize_point_cloud(cloud)


    with torch.no_grad():
        x = x.T.unsqueeze(0).to(device)
        rec_x = np.array(model(x)[0].to('cpu'))
        cloud = pc.get_point_cloud(rec_x)
        pc.visualize_point_cloud(cloud)
