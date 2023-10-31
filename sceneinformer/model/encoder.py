import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from sceneinformer.model.utils import MLPPointEncoder, PointEncoder, count_parameters


class Encoder(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super(Encoder, self).__init__()
        self.config = config

        self.hidden_dim = config['d_model']

        if 'point_enc' in config.keys():
            if config['point_enc'] == 'mlp':
                self.veh_encoder = MLPPointEncoder(config['vehicle_encoder'])
                self.ped_encoder = MLPPointEncoder(config['pedestrian_encoder'])
                self.bike_encoder = MLPPointEncoder(config['bike_encoder'])
            elif config['point_enc'] == 'pointnet':
                self.veh_encoder = PointEncoder(config['vehicle_encoder'])
                self.ped_encoder = PointEncoder(config['pedestrian_encoder'])
                self.bike_encoder = PointEncoder(config['bike_encoder'])
        else:
            self.veh_encoder = MLPPointEncoder(config['vehicle_encoder'])
            self.ped_encoder = MLPPointEncoder(config['pedestrian_encoder'])
            self.bike_encoder = MLPPointEncoder(config['bike_encoder'])

        self.poly_encoder = PointEncoder(config['map_encoder'])

        encoder_layer = nn.TransformerEncoderLayer( \
            d_model=config['d_model'], \
            nhead=config['nhead'],  \
            dim_feedforward=config['dim_feedforward'], \
            batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder( \
            encoder_layer, \
            num_layers=config['num_layers'])
        
        if config['compile_transformer']:
            self.transformer_encoder = torch.compile(self.transformer_encoder, mode="reduce-overhead")

    def _get_the_mask(self, objects: torch.Tensor) -> torch.Tensor:
        B, N, T, D = objects.shape
        objects = objects[:,:,:,:2].reshape(B, N, T*2) # (x,y)
        nan_objects_ind = torch.argwhere(torch.isnan(objects).all(2))
        nan_objects_mask = torch.zeros((B,N)).to(self.device).bool()
        nan_objects_mask[nan_objects_ind[:,0], nan_objects_ind[:,1]] = True 
        nan_objects_mask = nan_objects_mask.reshape(B, N) 
        return nan_objects_mask


    def forward(self, sample: dict) -> torch.Tensor:
        objects = sample['observed_trajectories']  # (B, Na+Np+Nc, T, D)
        polylines = sample['polylines']  # (B, Nm, n, D)

        # Generate the masks to ignore NaN values
        objects_mask = self._get_the_mask(objects)
        polylines_mask = self._get_the_mask(polylines)
        src_key_padding_mask = torch.cat([objects_mask, polylines_mask], dim=1)

        # Reshape the objects tensor and extract the object types
        B, N, T, D = objects.shape
        objects = objects.reshape(B * N, T, D)
        objects_types = objects[:, 0, -1]
        objects = objects[:, :, :-1]

        # Generate masks for each object type
        veh_ind_mask = objects_types == 0
        ped_ind_mask = objects_types == 1
        bike_ind_mask = objects_types == 2

        objects = torch.nan_to_num(objects, nan=0) # -99?

        vehs = objects[veh_ind_mask]
        peds = objects[ped_ind_mask]
        bike = objects[bike_ind_mask]

        vehs = vehs.permute(0, 2, 1) if vehs.shape[0] > 0 else torch.empty(0, 11, T, device=self.device, dtype=vehs.dtype)
        peds = peds.permute(0, 2, 1) if peds.shape[0] > 0 else torch.empty(0, 11, T, device=self.device, dtype=peds.dtype)
        bike = bike.permute(0, 2, 1) if bike.shape[0] > 0 else torch.empty(0, 11, T, device=self.device, dtype=bike.dtype)

        # Encode the objects using the appropriate encoder for each object type
        vehs = self.veh_encoder(vehs) if vehs.shape[0] > 0 else torch.empty(0, self.hidden_dim, device=self.device)
        peds = self.ped_encoder(peds) if peds.shape[0] > 0 else torch.empty(0, self.hidden_dim, device=self.device)
        bike = self.bike_encoder(bike) if bike.shape[0] > 0 else torch.empty(0, self.hidden_dim, device=self.device)

        peds = peds.type(vehs.dtype)
        bike = bike.type(vehs.dtype)

        processed_objects = torch.zeros(B * N, self.hidden_dim, device=self.device, dtype=vehs.dtype)

        processed_objects[veh_ind_mask] = vehs
        processed_objects[ped_ind_mask] = peds
        processed_objects[bike_ind_mask] = bike
        processed_objects = processed_objects.reshape(B, N, -1)  # (B, Na+Np+Nc, D)

        polylines = torch.nan_to_num(polylines, nan=0)
        B, Nm, Np, D = polylines.shape 
        polylines = polylines.reshape(B*Nm, Np, D)
        polylines = polylines.permute(0, 2, 1)
        processed_polylines = self.poly_encoder(polylines) #(B, Nm, D)
        processed_polylines = processed_polylines.reshape(B, Nm, -1) #(B, Nm, D)

        obs_tokens = torch.cat([processed_objects, processed_polylines], dim=1)
        encoded_obs = self.transformer_encoder(obs_tokens, src_key_padding_mask=src_key_padding_mask) #CHECK

        assert not torch.isnan(encoded_obs).any(), 'NaNs in the encoded observations!'

        return {
            'encoded_obs': encoded_obs,
            'src_key_padding_mask': src_key_padding_mask
        }