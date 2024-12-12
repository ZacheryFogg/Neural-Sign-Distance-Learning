import torch
import torch.nn as nn
import torch.nn.functional as F



#####################################
#######   REGRESSION MODELS  ########
#####################################
class SDFRegressionModel_3L(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(SDFRegressionModel_3L, self).__init__()
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
    

class SDFRegressionModel_5L(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(SDFRegressionModel_5L, self).__init__()
        # Increase point representation from 3 to 64
        self.fc0 = nn.Linear(input_dim, 32)
        self.fc1 = nn.Linear(32, 64)

        # Concat with latent dim to predict SDF value
        self.fc2 = nn.Linear(64 + latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, latent):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = torch.cat((x, latent), dim=2)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

class SDFRegressionModel_7L(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(SDFRegressionModel_7L, self).__init__()
        self.fc0 = nn.Linear(input_dim, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)

        self.fc3 = nn.Linear(128 + latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, latent):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        x = torch.cat((x, latent), dim=2)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
# Extra Downsampling layer
class SDFRegressionModel_9L(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(SDFRegressionModel_9L, self).__init__()
        self.fc0 = nn.Linear(input_dim, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)

        self.fc4 = nn.Linear(128 + latent_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc6 = nn.Linear(hidden_dim, int(hidden_dim/2)) # (256,128)
        self.fc7 = nn.Linear(int(hidden_dim/2), 1)
        self.relu = nn.ReLU()

    def forward(self, x, latent):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        x = torch.cat((x, latent), dim=2)
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return x


############################################
######  BINARY CLASSIFICATION MODELS  ######
############################################
class SDFClassificationModel_3L(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(SDFClassificationModel_3L, self).__init__()
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
    

class SDFClassificationModel_5L(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(SDFClassificationModel_5L, self).__init__()
        # Increase point representation from 3 to 64
        self.fc0 = nn.Linear(input_dim, 32)
        self.fc1 = nn.Linear(32, 64)

        # Concat with latent dim to predict SDF value
        self.fc2 = nn.Linear(64 + latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, latent):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = torch.cat((x, latent), dim=2)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

class SDFClassificationModel_7L(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(SDFClassificationModel_7L, self).__init__()
        self.fc0 = nn.Linear(input_dim, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)

        self.fc3 = nn.Linear(128 + latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, latent):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        x = torch.cat((x, latent), dim=2)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
# Extra Downsampling layer
class SDFClassificationModel_9L(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(SDFClassificationModel_9L, self).__init__()
        self.fc0 = nn.Linear(input_dim, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)

        self.fc4 = nn.Linear(128 + latent_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc6 = nn.Linear(hidden_dim, int(hidden_dim/2)) # (256,128)
        self.fc7 = nn.Linear(int(hidden_dim/2), 1)
        self.relu = nn.ReLU()

    def forward(self, x, latent):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        x = torch.cat((x, latent), dim=2)
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return x
