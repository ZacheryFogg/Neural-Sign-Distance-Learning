import torch.nn as nn
import torch.nn.functional as F

"""Previous Autoencoder Architectures"""

class PointCloudAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(PointCloudAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    
class MLPEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim1)
        self.fc2 = nn.Linear(config.hidden_dim1, config.hidden_dim2)
        self.fc3 = nn.Linear(config.hidden_dim2, config.latent_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)

        return x

class MLPDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.latent_dim, config.hidden_dim2)
        self.fc2 = nn.Linear(config.hidden_dim2, config.hidden_dim1)
        self.fc3 = nn.Linear(config.hidden_dim1, config.input_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)

        return x


class Autoencoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = MLPEncoder(config)
        self.decoder = MLPDecoder(config)

    def forward(self,x):
        latent_rep = self.encoder(x)
        out = self.decoder(latent_rep)
        return out


# class Autoencoder(nn.Module):

#     def __init__(self, config):
#         super().__init__()

#         self.fc = nn.Linear(config.input_dim, config.input_dim)

#     def forward(self,x):
#         y = self.fc(x)
#         return x + (.00001 * y)
    
