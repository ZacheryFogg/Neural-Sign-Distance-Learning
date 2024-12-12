import torch
import torch.nn as nn
import torch.nn.functional as F


############################################
######  BINARY CLASSIFICATION MODELS  ######
############################################
class SD_3L(nn.Module):
    def __init__(self, latent_dim, input_dim = 3, hidden_dim = 256):
        super(SD_3L, self).__init__()
        self.fc1 = nn.Linear(input_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, latent, x ):
        x = torch.cat((x, latent), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class SD_3L_Upscale32(nn.Module):
    def __init__(self, latent_dim, input_dim = 3, hidden_dim = 256):
        super(SD_3L_Upscale32, self).__init__()
        # Increase point representation from 3 to 64
        self.fc0 = nn.Linear(input_dim, 32)
        self.fc1 = nn.Linear(32, 64)

        # Concat with latent dim to predict SDF value
        self.fc2 = nn.Linear(64 + latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, latent, x):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = torch.cat((x, latent), dim=1)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

class SD_3L_Upscale128(nn.Module):
    def __init__(self, latent_dim, input_dim = 3, hidden_dim = 256):
        super(SD_3L_Upscale128, self).__init__()
        self.fc0 = nn.Linear(input_dim, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)

        self.fc3 = nn.Linear(128 + latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, latent, x):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        x = torch.cat((x, latent), dim=1)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
# Extra Downsampling layer
class SD_4L_Upscale128(nn.Module):
    def __init__(self, latent_dim, input_dim = 3, hidden_dim = 256):
        super(SD_4L_Upscale128, self).__init__()
        self.fc0 = nn.Linear(input_dim, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)

        self.fc4 = nn.Linear(128 + latent_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc6 = nn.Linear(hidden_dim, int(hidden_dim/2)) # (256,128)
        self.fc7 = nn.Linear(int(hidden_dim/2), 1)
        self.relu = nn.ReLU()

    def forward(self, latent, x):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        x = torch.cat((x, latent), dim=1)
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return x
    

class SD_5L_Upscale256(nn.Module):

    def __init__(self, latent_rep_size):
        
        super().__init__()

        self.up1 = nn.Linear(3, 64)
        self.up2 = nn.Linear(64, latent_rep_size)

        self.l1 = nn.Linear(latent_rep_size * 2, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 1)

        self.up_bn1 = nn.BatchNorm1d(64)
        
        self.l_bn1 = nn.BatchNorm1d(1024)
        self.l_bn2 = nn.BatchNorm1d(512)
        self.l_bn3 = nn.BatchNorm1d(256)
        self.l_bn4 = nn.BatchNorm1d(128)


    def forward(self, latent_rep, xyz):
        xyz = F.gelu(self.up_bn1(self.up1(xyz)))
        xyz = self.up2(xyz)

        x = torch.concat((latent_rep, xyz), dim = 1)

        x = F.gelu(self.l_bn1(self.l1(x)))
        x = F.gelu(self.l_bn2(self.l2(x)))
        x = F.gelu(self.l_bn3(self.l3(x)))
        x = F.gelu(self.l_bn4(self.l4(x)))
        x = self.l5(x)

        return x

class SD_6L_Upscale256(nn.Module):

    def __init__(self, latent_rep_size):
        
        super().__init__()

        self.up1 = nn.Linear(3, 64)
        self.up2 = nn.Linear(64, 128)
        self.up3 = nn.Linear(128, latent_rep_size)

        self.l1 = nn.Linear(latent_rep_size * 2, 2048)
        self.l2 = nn.Linear(2048, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.l4 = nn.Linear(512, 256)
        self.l5 = nn.Linear(256, 64)
        self.l6 = nn.Linear(64, 1)

        self.up_bn1 = nn.BatchNorm1d(64)
        self.up_bn2 = nn.BatchNorm1d(128)
        self.up_bn3 = nn.BatchNorm1d(latent_rep_size)
        
        self.l_bn1 = nn.BatchNorm1d(2048)
        self.l_bn2 = nn.BatchNorm1d(1024)
        self.l_bn3 = nn.BatchNorm1d(512)
        self.l_bn4 = nn.BatchNorm1d(256)
        self.l_bn5 = nn.BatchNorm1d(64)


    def forward(self, latent_rep, xyz):
        xyz = F.gelu(self.up_bn1(self.up1(xyz)))
        xyz = F.gelu(self.up_bn2(self.up2(xyz)))
        xyz = self.up3(xyz)

        x = torch.concat((latent_rep, xyz), dim = 1)

        x = F.gelu(self.l_bn1(self.l1(x)))
        x = F.gelu(self.l_bn2(self.l2(x)))
        x = F.gelu(self.l_bn3(self.l3(x)))
        x = F.gelu(self.l_bn4(self.l4(x)))
        x = F.gelu(self.l_bn5(self.l5(x)))
        x = self.l6(x)

        return x
    


class SD_4L_Upscale256_LatentEncode(nn.Module):
    def __init__(self, latent_dim, input_dim = 3, hidden_dim = 256):
        super(SD_4L_Upscale256_LatentEncode, self).__init__()
        self.x_fc = nn.Linear(input_dim, hidden_dim)
        self.laten_fc = nn.Linear(latent_dim, hidden_dim) 
        self.batch_norm1 = nn.BatchNorm1d(2* hidden_dim, affine=True)
        self.fc1 = nn.Linear(2* hidden_dim, 2*hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(2*hidden_dim, affine=True)
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim, affine=True)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm4 = nn.BatchNorm1d(hidden_dim, affine=True)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()


    def forward(self, latent_encoding, x):
        x = self.relu(self.x_fc(x))
        latent_encoding = self.relu(self.laten_fc(latent_encoding))
        x = torch.cat((x, latent_encoding), dim=-1)
        x = self.batch_norm1(x)
        x = self.batch_norm2(self.relu(self.fc1(x)))
        x = self.batch_norm3(self.relu(self.fc2(x)))
        x = self.batch_norm4(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x