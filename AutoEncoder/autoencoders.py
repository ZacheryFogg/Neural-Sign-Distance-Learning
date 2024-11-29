import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################################
#  Common Decoder that is shared amongst all models #
#####################################################
class ConvDecoder(nn.Module):
    
    def __init__(self, point_size, latent_dim):
        super().__init__()
        
        self.point_size = point_size

        self.l1 = nn.Linear(latent_dim, 1024)
        self.l2 = nn.Linear(1024, 2048)
        self.l3 = nn.Linear(2048, 3072)
        self.l4 = nn.Linear(3072, point_size * 3)

    def forward(self, x):
        x = F.gelu(self.l1(x))
        x = F.gelu(self.l2(x))
        x = F.gelu(self.l3(x))
        x = self.l4(x)
        x = x.view(-1, self.point_size, 3)
        return x    
    
#######################################
# Medium | 5.4 million param encoder  #
#######################################
    
class ConvEncoder_5400T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.point_size = point_size 

        # Blowup point representation from 3 to 32
        self.conv1 = nn.Conv1d(3, 16, 1)
        self.conv2 = nn.Conv1d(16, 32, 1)

        # Points talk to each other wo/ downsampling 
        self.conv3 = nn.Conv1d(32, 32, kernel_size = 9, stride= 1, padding= 4)
        self.conv4 = nn.Conv1d(32, 32, kernel_size = 9, stride = 1, padding = 4)
        self.conv5 = nn.Conv1d(32, 32, kernel_size = 9, stride = 1, padding = 4)

        # Downsampling 
        self.conv6 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv7 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv8 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv9 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)

        # Linear 
        self.lin1 = nn.Linear(point_size * 2 , 1024)
        self.lin2 = nn.Linear(1024, 768)
        self.lin3 = nn.Linear(768, latent_size)

    def forward(self, x):
        
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))

        x = F.gelu(self.conv6(x))
        x = F.gelu(self.conv7(x))
        x = F.gelu(self.conv8(x))
        x = F.gelu(self.conv9(x))

        x = x.view(-1, self.point_size * 2)

        x = F.gelu(self.lin1(x))
        x = F.gelu(self.lin2(x))
        x = self.lin3(x)

        return x


class ConvAE_5400T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_5400T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud
    


######################################
# Small | 4.7 million param encoder  #
######################################
    
class ConvEncoder_4700T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.point_size = point_size 

        # Blowup point representation from 3 to 32
        self.conv1 = nn.Conv1d(3, 32, 1)

        # Points talk to each other wo/ downsampling 
        self.conv3 = nn.Conv1d(32, 32, kernel_size = 9, stride= 1, padding= 4)
        self.conv4 = nn.Conv1d(32, 32, kernel_size = 9, stride = 1, padding = 4)

        # Downsampling 
        self.conv6 = nn.Conv1d(32, 16, kernel_size = 8, stride = 2, padding = 3)
        self.conv7 = nn.Conv1d(16, 16, kernel_size = 8, stride = 2, padding = 3)
        self.conv8 = nn.Conv1d(16, 16, kernel_size = 8, stride = 2, padding = 3)

        # Linear 
        self.lin1 = nn.Linear(point_size * 2 , 1024)
        self.lin3 = nn.Linear(1024, latent_size)

    def forward(self, x):
        
        x = F.gelu(self.conv1(x))

        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))

        x = F.gelu(self.conv6(x))
        x = F.gelu(self.conv7(x))
        x = F.gelu(self.conv8(x))

        x = x.view(-1, self.point_size * 2)

        x = F.gelu(self.lin1(x))
        x = self.lin3(x)

        return x


class ConvAE_4700T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_4700T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud
    
######################################
# Small | 3.2 million param encoder  #
######################################

class ConvEncoder_3200T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.point_size = point_size 

        # Blowup point representation from 3 to 32
        self.conv1 = nn.Conv1d(3, 32, 1)

        # Points talk to each other wo/ downsampling 
        self.conv3 = nn.Conv1d(32, 32, kernel_size = 9, stride= 1, padding= 4)
        self.conv4 = nn.Conv1d(32, 32, kernel_size = 9, stride = 1, padding = 4)

        # Downsampling 
        self.conv6 = nn.Conv1d(32, 16, kernel_size = 8, stride = 2, padding = 3)
        self.conv7 = nn.Conv1d(16, 16, kernel_size = 8, stride = 2, padding = 3)

        # Linear 
        self.lin1 = nn.Linear(point_size * 4 , latent_size)

    def forward(self, x):
        
        x = F.gelu(self.conv1(x))

        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))

        x = F.gelu(self.conv6(x))
        x = F.gelu(self.conv7(x))

        x = x.view(-1, self.point_size * 4)

        x = self.lin1(x)

        return x


class ConvAE_3200T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_3200T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud
    

######################################
# Mini | 3.2 million param encoder  #
######################################
    
class ConvEncoder_2800T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.point_size = point_size 

        # Points talk to each other wo/ downsampling 
        self.conv1 = nn.Conv1d(3, 32, kernel_size = 9, stride= 1, padding= 4)
        self.conv2 = nn.Conv1d(32, 32, kernel_size = 9, stride = 1, padding = 4)

        # Downsampling 
        self.conv3 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv4 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv5 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv6 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv7 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)

        # Linear 
        self.lin1 = nn.Linear(point_size , latent_size)

    def forward(self, x):
        
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))
        x = F.gelu(self.conv7(x))

        x = x.view(-1, self.point_size)

        x = self.lin1(x)

        return x

class ConvAE_2800T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_2800T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud
    
######################################
# Tiny | .57 million param encoder  #
######################################

class ConvEncoder_570T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.point_size = point_size 

        # Points talk to each other wo/ downsampling 
        self.conv1 = nn.Conv1d(3, 32, kernel_size = 9, stride= 1, padding= 4)
        self.conv2 = nn.Conv1d(32, 32, kernel_size = 9, stride = 1, padding = 4)

        # Downsampling 
        self.conv3 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv4 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv5 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv6 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv7 = nn.Conv1d(32, 16, kernel_size = 8, stride = 2, padding = 3)

        # Linear 
        self.lin1 = nn.Linear(int(point_size / 2) , latent_size)

    def forward(self, x):
        
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))
        x = F.gelu(self.conv7(x))

        x = x.view(-1, int(self.point_size / 2))

        x = self.lin1(x)

        return x


class ConvAE_570T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_570T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud
    

class ConvEncoder_5800T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.point_size = point_size 

        # Blowup point representation from 3 to 32
        self.conv1 = nn.Conv1d(3, 16, 1)
        self.conv2 = nn.Conv1d(16, 32, 1)


        # Points talk to each other wo/ downsampling 
        self.conv3 = nn.Conv1d(32, 64, kernel_size = 9, stride= 1, padding= 4)
        self.conv4 = nn.Conv1d(64, 128, kernel_size = 9, stride = 1, padding = 4)
        self.conv5 = nn.Conv1d(128, 256, kernel_size = 9, stride = 1, padding = 4)

        # Downsampling 
        self.conv6 = nn.Conv1d(256, 128, kernel_size = 8, stride = 2, padding = 3)
        self.conv7 = nn.Conv1d(128, 64, kernel_size = 8, stride = 2, padding = 3)
        self.conv8 = nn.Conv1d(64, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv9 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)


        # Linear 
        self.lin1 = nn.Linear(point_size * 2, 768)
        self.lin2 = nn.Linear(768, latent_size)

    def forward(self, x):
        
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))

        x = F.gelu(self.conv6(x))
        x = F.gelu(self.conv7(x))
        x = F.gelu(self.conv8(x))
        x = F.gelu(self.conv9(x))

        x = x.view(-1, self.point_size * 2)

        x = F.gelu(self.lin1(x))
        x = self.lin2(x)

        return x


class ConvAE_5800T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_5800T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud