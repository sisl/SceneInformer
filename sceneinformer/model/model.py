import torch
import lightning.pytorch as pl
from sceneinformer.model.encoder import Encoder
from sceneinformer.model.decoder import Decoder
from sceneinformer.model.loss import compute_loss

class SceneInformer(pl.LightningModule):
    def __init__(self, config):
        super(SceneInformer, self).__init__()
        config.decoder.num_modes = config.k_modes
        config.decoder.predictor.out_dim = config.k_modes * (config.n_future_steps) * config.step_dim
        config.decoder.classifier_traj.out_dim = config.k_modes

        self.config = config
        self.learning_rate = config.learning_rate
        self.loss_config = config.loss
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)
        self.decoder.step_dim = config.step_dim
        self.batch = None

    def forward(self, sample):
        encoder_dict = self.encoder(sample)
        decoder_dict = self.decoder(sample['anchors'], encoder_dict['encoded_obs'], encoder_dict['src_key_padding_mask'])
        return decoder_dict
    
    def training_step(self, batch, batch_idx):
        prediction_dict = self(batch)
        loss, metrics = compute_loss(prediction_dict, batch, self.loss_config)
        self.convert_and_log_metrics(metrics, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        prediction_dict = self(batch)
        loss, metrics = compute_loss(prediction_dict, batch, self.loss_config)
        if self.trainer.global_rank == 0:        
            self.convert_and_log_metrics(metrics, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        prediction_dict = self(batch)
        loss, metrics = compute_loss(prediction_dict, batch, self.loss_config)
        self.convert_and_log_metrics(metrics, 'test')
        return loss
    
    def predict_step(self, batch, batch_idx):
        decoder_dict = self(batch)
        return decoder_dict
    
    def convert_and_log_metrics(self, metrics_dict, prefix):
        for key in metrics_dict.keys():
            self.log(f'{prefix}_{key}', metrics_dict[key], logger=True, on_step=True, on_epoch=False) #, sync_dist=True, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        lr_inital = self.learning_rate
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr_inital)

        def custom_lr_lambda(step, lr_up_steps=10000, lr_start=0, lr_end=1e-4, lr_final=0, lr_down_steps=int(1e6)):
            if step <= lr_up_steps:
                new_lr = lr_start + (lr_end - lr_start) * step / lr_up_steps
            elif step <= lr_up_steps + lr_down_steps:
                new_lr = lr_end + (lr_final - lr_end) * (step - lr_up_steps) / (lr_down_steps)
            else:
                new_lr = lr_final
            new_lr = new_lr / lr_inital
            return new_lr
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: custom_lr_lambda(step))

        return {
            "optimizer": optimizer,
            "lr_scheduler":  {
                "scheduler": scheduler, 
                "interval": "step"
                },
            } 
    
    