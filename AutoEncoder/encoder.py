import torch 
import AutoEncoder.autoencoders as ae

available_models = {
    '3072' : {
        '6800T' : {
            'path': '3072_512/Conv_6800T',
            'point_size' : 3072,
            'latent_size' : 512
        },
        'MLP' : {
            'path': '3072_512/MLP',
            'point_size' : 3072,
            'latent_size' : 512
        },
        '800T' : {
            'path': '3072_512/Conv_800T',
            'point_size' : 3072,
            'latent_size' : 512
        },
    },
    '1024' : {
        '6800T' : {
            'path': '1024_256/Conv_6800T',
            'point_size' : 1024,
            'latent_size' : 256
        },
        'MLP' : {
            'path': '1024_256/MLP',
            'point_size' : 1024,
            'latent_size' : 256
        },
        '800T' : {
            'path': '1024_256/Conv_800T',
            'point_size' : 1024,
            'latent_size' : 256
        },
    }
}

class PointCloudAutoEncoder(torch.nn.Module):

    def __init__(self, model_type, point_size, path_to_weight_dir = './'):
        super().__init__()

        self.encoder = None 
        self.point_size = point_size
        self.model_type = model_type
        self.path_to_weight_dir = path_to_weight_dir
        self.set_encoder()

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


    def set_encoder(self):


        point_size = available_models[self.point_size][self.model_type]['point_size']
        latent_size = available_models[self.point_size][self.model_type]['latent_size']
        path = available_models[self.point_size][self.model_type]['path']

        if self.model_type == '6800T':
            self.encoder = ae.ConvEncoder_6800T(point_size, latent_size)
        elif self.model_type == 'MLP':
            self.encoder = ae.MLPEncoder(point_size, latent_size)
        elif self.model_type == '800T':
            self.encoder = ae.ConvEncoder_800T(point_size, latent_size)
        else: 
            self.encoder = ae.ConvEncoder_800T(point_size, latent_size)

        weight_path = self.path_to_weight_dir + 'trained_autoencoders/' + path
        print(weight_path)
        self.load_weights_from_pretrained(self.encoder, weight_path)