import torch
import torch.nn as nn
import lightning.pytorch as pl

from sceneinformer.model.utils import MLP

class Decoder(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super(Decoder, self).__init__()
        self.config = config

        self.anchor_encoding = MLP(config['anchor_encoding'])

        decoder_layer = nn.TransformerDecoderLayer( \
            d_model=config['d_model'], \
            nhead=config['nhead'],  \
            dim_feedforward=config['dim_feedforward'], \
            batch_first=True)

        self.transformer_decoder = nn.TransformerDecoder( \
            decoder_layer, \
            num_layers=config['num_layers'])

        if config['compile_transformer']:
            self.transformer_decoder = torch.compile(self.transformer_decoder, mode="reduce-overhead")

        self.token_decoder = MLP(config['token_decoder'])
        self.classifier_traj = MLP(config['classifier_traj'])
        self.classifier_occ = MLP(config['classifier_occ'])
        self.predictor = MLP(config['predictor'])

    def forward(self, anchors: torch.Tensor, memory_tokens: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = anchors.shape

        invalid_anchors = torch.argwhere(torch.isnan(anchors))

        invalid_anchors_mask = torch.ones(anchors.shape[:2]).to(anchors.device)
        invalid_anchors_mask[invalid_anchors[:,0], invalid_anchors[:,1]] = 0

        bool_tgt_anchor_mask = torch.zeros(anchors.shape[:2]).to(anchors.device).bool()
        bool_tgt_anchor_mask[invalid_anchors[:,0], invalid_anchors[:,1]] = True

        anchors = torch.nan_to_num(anchors, nan=0)

        # Encode anchors with MLP.
        anchors = anchors.reshape(B * N, D)
        anchor_tokens = self.anchor_encoding(anchors)
        anchor_tokens = anchor_tokens.reshape(B, N, -1)

        decoded_obs = self.transformer_decoder(anchor_tokens, 
                                               memory_tokens, 
                                               tgt_key_padding_mask=bool_tgt_anchor_mask, 
                                               memory_key_padding_mask=memory_mask) 
        decoded_obs = decoded_obs.reshape(B * N, -1)

        decoded_obs = self.token_decoder(decoded_obs)
        logits_traj = self.classifier_traj(decoded_obs) 
        logits_occ = self.classifier_occ(decoded_obs)
        predictions = self.predictor(decoded_obs)

        predictions = predictions.reshape(B, N, -1)
        predictions = predictions.reshape(B, N, self.config['num_modes'], -1, self.step_dim)
        anchors = anchors.reshape(B, N, 1, 1, 2)

        predictions[:,:,:,:,:2] = predictions[:,:,:,:,:2] + anchors 

        logits_traj = logits_traj.reshape(B, N, -1)
        logits_occ = logits_occ.reshape(B, N, -1)

        logits_traj = logits_traj * invalid_anchors_mask.reshape(B,N,1)
        logits_occ = logits_occ * invalid_anchors_mask.reshape(B,N,1)
        predictions = predictions * invalid_anchors_mask.reshape(B, N, 1, 1, 1)

        return {
            'logits_traj': logits_traj,
            'logits_occ': logits_occ,
            'predictions': predictions,
        }
