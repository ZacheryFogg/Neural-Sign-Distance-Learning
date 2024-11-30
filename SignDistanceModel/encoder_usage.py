import torch 
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from AutoEncoder.encoder import PointCloudAutoEncoder

# Random batch of examples
x = torch.rand(64, 3072, 3).permute(0,2,1) # Cloud must be permuted to be (batch_size, xyz, point_size)

# Example of a batch of point clouds being encoded to latent reps
point_size = 3072
latent_shape = 512


encoder = PointCloudAutoEncoder(model_type= '800T', point_size= str(point_size), path_to_weight_dir= '../AutoEncoder/')
encoder.eval() # Let torch know that it doesn't need to store activations as there will be no backward pass

latent_rep = encoder(x)

print(latent_rep.shape)

