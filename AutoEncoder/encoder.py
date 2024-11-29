import torch 
import models

available_models = {
    '4700T' : {
        'path': './trained_models/ConvEnc_LinDec/ConvAutoEncoder_ModelNet40_4700T',
        'point_size' : 2048,
        'latent_size' : 512
    },
    '3200T' : {
        'path': './trained_models/ConvEnc_LinDec/ConvAutoEncoder_ModelNet40_3200T',
        'point_size' : 2048,
        'latent_size' : 512
    },
}

class PointCloudAutoEncoder(torch.nn.Module):

    def __init__(self, model_type):
        super().__init__()

        self.encoder = None 
        self.set_encoder(model_type)

    def forward(self, x):
        return self.encoder(x)

    def load_weights_from_pretrained(self, encoder, weight_path):
        '''
        Copy only the encoder weights from pretrained autoencoder into our encoder only model 
        '''
        
        enc_sd = encoder.state_dict()

        full_model_sd = torch.load(weight_path, weights_only= True)

        full_model_sd_keys = [key for key in full_model_sd.keys() if 'encoder' in key]

        for keys in zip(full_model_sd_keys, enc_sd.keys()): 
    
            assert(keys[0].split('encoder.')[1] == keys[1])
            assert(enc_sd[keys[1]].shape == full_model_sd[keys[0]].shape)

            with torch.no_grad():
                enc_sd[keys[1]].copy_(full_model_sd[keys[0]])


    def set_encoder(self, model_type):

        point_size = available_models[model_type]['point_size']
        latent_size = available_models[model_type]['latent_size']
        weight_path = available_models[model_type]['path']

        if model_type == '4700T':
            self.encoder = models.ConvEncoder_4700T(point_size, latent_size)
        elif model_type == '3200T':
            self.encoder = models.ConvEncoder_3200T(point_size=point_size, latent_size= latent_size)
        
        self.load_weights_from_pretrained(self.encoder, weight_path)