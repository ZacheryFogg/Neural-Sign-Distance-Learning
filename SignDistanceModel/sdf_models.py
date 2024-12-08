import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFRegressionModel(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(SDFRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, latent):
        x = torch.cat((x, latent), dim=2)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
