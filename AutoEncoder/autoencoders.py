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
    
######################################
#     7.5 million param encoder      #
######################################
    
class ConvEncoder_7500T(nn.Module):
    
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


class ConvAE_7500T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_7500T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud
    


######################################
#     6.8 million param encoder      #
######################################
    
class ConvEncoder_6800T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.point_size = point_size 

        # Blowup point representation from 3 to 32
        self.conv1 = nn.Conv1d(3, 32, 1)

        # Points talk to each other wo/ downsampling 
        self.conv2 = nn.Conv1d(32, 32, kernel_size = 9, stride= 1, padding= 4)
        self.conv3 = nn.Conv1d(32, 32, kernel_size = 9, stride = 1, padding = 4)

        # Downsampling 
        self.conv4 = nn.Conv1d(32, 16, kernel_size = 8, stride = 2, padding = 3)
        self.conv5 = nn.Conv1d(16, 16, kernel_size = 8, stride = 2, padding = 3)
        self.conv6 = nn.Conv1d(16, 16, kernel_size = 8, stride = 2, padding = 3)

        # Linear 
        self.lin1 = nn.Linear(point_size * 2 , 1024)
        self.lin2 = nn.Linear(1024, latent_size)

    def forward(self, x):
        
        x = F.gelu(self.conv1(x))

        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))

        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))

        x = x.view(-1, self.point_size * 2)

        x = F.gelu(self.lin1(x))
        x = self.lin2(x)

        return x


class ConvAE_6800T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_6800T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud
    
######################################
#     6.3 million param encoder      #
######################################

class ConvEncoder_6300T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.point_size = point_size 

        # Blowup point representation from 3 to 32
        self.conv1 = nn.Conv1d(3, 32, 1)

        # Points talk to each other wo/ downsampling 
        self.conv2 = nn.Conv1d(32, 32, kernel_size = 9, stride= 1, padding= 4)
        self.conv3 = nn.Conv1d(32, 32, kernel_size = 9, stride = 1, padding = 4)

        # Downsampling 
        self.conv4 = nn.Conv1d(32, 16, kernel_size = 8, stride = 2, padding = 3)
        self.conv5 = nn.Conv1d(16, 16, kernel_size = 8, stride = 2, padding = 3)

        # Linear 
        self.lin1 = nn.Linear(point_size * 4 , latent_size)

    def forward(self, x):
        
        x = F.gelu(self.conv1(x))

        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))

        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))

        x = x.view(-1, self.point_size * 4)

        x = self.lin1(x)

        return x


class ConvAE_6300T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_6300T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud
    
######################################
#     1.6 million param encoder      #
######################################


class ConvEncoder_1600T(nn.Module):
    
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


class ConvAE_1600T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_1600T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud
    

######################################
#     .8 million param encoder       #
######################################


class ConvEncoder_800T(nn.Module):
    
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


class ConvAE_800T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_800T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud
    

######################################
#     .27 million param encoder      #
######################################


class ConvEncoder_270T(nn.Module):
    
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
        self.conv8 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv9 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)
        self.conv10 = nn.Conv1d(32, 32, kernel_size = 8, stride = 2, padding = 3)


        # Linear 
        self.lin1 = nn.Linear(int(point_size / 8) , latent_size)

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
        x = F.gelu(self.conv10(x))

        x = x.view(-1, int(self.point_size / 8))

        x = self.lin1(x)

        return x


class ConvAE_270T(nn.Module):
    
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = ConvEncoder_270T(point_size, latent_size)
        self.decoder = ConvDecoder(point_size, latent_size)

    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud
    
    
###################
#   MLP Only AE   #
###################

class MLPEncoder(nn.Module):

    def __init__(self, point_size, latent_size):
        super().__init__()

        self.point_size = point_size

        self.fc1 = nn.Linear(point_size * 3, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, latent_size)

    def forward(self, x):

        x = torch.reshape(x, (x.shape[0], -1))
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)

        return x

class MLPDecoder(nn.Module):

    def __init__(self, point_size, latent_size):
        super().__init__()
        
        self.point_size = point_size

        self.fc1 = nn.Linear(latent_size, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, point_size * 3)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.point_size, 3)

        return x


class MLP_AE(nn.Module):

    def __init__(self, point_size, latent_size):
        super().__init__()

        self.encoder = MLPEncoder(point_size, latent_size)
        self.decoder = MLPDecoder(point_size, latent_size)

    def forward(self,x):
        latent_rep = self.encoder(x)
        reconstructed_cloud = self.decoder(latent_rep)
        return reconstructed_cloud


#######################
# Conv w/ Max pooling #
#######################

class ConvMax_Encoder(nn.Module):
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.latent_size = latent_size
        self.point_size = point_size
        
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 2)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 6, int(self.point_size / 2)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x

class ConvMax_Decoder(nn.Module):
    def __init__(self, point_size, latent_size):
        super().__init__()

        self.latent_size = latent_size
        self.point_size = point_size

        self.dec1 = nn.Linear(self.latent_size,768)
        self.dec2 = nn.Linear(768,1024)
        self.dec3 = nn.Linear(1024,self.point_size*3)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)
    
class ConvMax_AE(nn.Module):
    def __init__(self, point_size, latent_size):
        super().__init__()
        
        self.encoder = ConvMax_Encoder(point_size, latent_size)
        self.decoder = ConvMax_Decoder(point_size, latent_size)
    
    def forward(self, x):
        latent_rep = self.encoder(x)
        reconstructed_x = self.decoder(latent_rep)
        return reconstructed_x
    

#######################
#     Conv Only AE    #
#######################

class ConvOnlyEncoder(nn.Module): 
    
    def __init__(self, point_size, latent_dim):
        super().__init__()

        self.point_size = point_size

        self.c1 = nn.Conv1d(3, 64,1) # (16, point_size) - each point has been blown up from 3 to 16 numbers  
        self.c2 = nn.Conv1d(64, 128,1) # (32, point_size) - each point has been blown up from 16 to 32 numbers
        self.c3 = nn.Conv1d(128, 32, 7, stride=1, padding=3) # (32, point_size) - each activation is now an amalgamation of 7 neighboring points 
        self.c4 = nn.Conv1d(32, 32, 7, stride=2, padding=3) # (32, point_size / 2) - each activation is now an amalgamation of roughly 49 neighboring points (is this right? )
        self.c5 = nn.Conv1d(32, 32, 7, stride=2, padding=3) # (32, point_size / 4)
        self.c6 = nn.Conv1d(32, 32, 7, stride=2, padding=3) # (32, point_size / 8)
        self.c7 = nn.Conv1d(32, 32, 7, stride=2, padding=3)

        self.lin7 = nn.Linear(point_size * 2, 1024)
        self.lin8 = nn.Linear(1024, latent_dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(32)
        self.bn7 = nn.BatchNorm1d(32)
        self.bn8 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.gelu(self.bn1(self.c1(x)))
        x = F.gelu(self.bn2(self.c2(x)))
        x = F.gelu(self.bn3(self.c3(x)))
        x = F.gelu(self.bn4(self.c4(x)))
        x = F.gelu(self.bn5(self.c5(x)))
        x = F.gelu(self.bn6(self.c6(x)))
        x = F.gelu(self.bn6(self.c7(x)))

        x = x.view(-1, self.point_size * 2 )
        x = F.gelu(self.bn8(self.lin7(x)))
        x = self.lin8(x)

        return x

class ConvOnlyDecoder(nn.Module):
    
    def __init__(self, point_size, latent_dim):
        super().__init__()

        self.point_size = point_size

        self.dl1 = nn.Linear(latent_dim, 1024) 
        self.dl2 = nn.Linear(1024, point_size * 2) 

        self.dc3 = nn.ConvTranspose1d(32, 32, 8, stride=2, padding = 3)
        self.dc4 = nn.ConvTranspose1d(32, 32, 8, stride=2, padding = 3) 
        self.dc5 = nn.ConvTranspose1d(32, 32, 8, stride=2, padding= 3) 
        self.dc6 = nn.ConvTranspose1d(32, 128, 7, stride=1, padding= 3) 
        self.dc7 = nn.ConvTranspose1d(128, 64, 1) 
        self.dc8 = nn.ConvTranspose1d(64 ,3, 1) 

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(self.point_size * 2)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(64)

    def forward(self, x):
        
        x = F.gelu(self.bn1(self.dl1(x)))
        x = F.gelu(self.bn2(self.dl2(x)))

        x = x.view(x.shape[0], 32, -1)
        
        x = F.gelu(self.bn3(self.dc3(x)))
        x = F.gelu(self.bn4(self.dc4(x)))
        x = F.gelu(self.bn5(self.dc5(x)))
        x = F.gelu(self.bn6(self.dc6(x)))
        x = F.gelu(self.bn7(self.dc7(x)))
        x = F.gelu(self.dc8(x))

        return x

class ConvOnly_AE(nn.Module):

    def __init__(self, point_size, latent_dim):
        super().__init__()

        self.encoder = ConvOnlyEncoder(point_size, latent_dim)
        self.decoder = ConvOnlyDecoder(point_size, latent_dim)

    def forward(self, x):
        latent_rep = self.encoder(x)
        recontructed_cloud = self.decoder(latent_rep)

        recontructed_cloud = recontructed_cloud.permute(0,2,1)
        return recontructed_cloud