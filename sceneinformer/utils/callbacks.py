import os

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback
except:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

from functools import partial

from omegaconf import OmegaConf
from sceneinformer.utils.utils import (WrappedDataset, instantiate_from_config,
                             my_collate, my_collate_multi_occlusion)
from torch.utils.data import DataLoader


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        init_fn = None
        train_dataset = self.datasets["train"]
        return DataLoader(train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, # IMPORTANT: It is shuffled in the sampler. Hence, not needed here. It leads to error.
                          worker_init_fn=init_fn,
                          collate_fn=my_collate,
                          #sampler=train_sampler,
                          pin_memory=True, #False, #True,
                          persistent_workers=True,
                          )

    def _val_dataloader(self, shuffle=False):
        init_fn = None
        val_dataset = self.datasets["validation"]
        return DataLoader(val_dataset, batch_size=self.batch_size,
                    num_workers=self.num_workers, shuffle=True,
                    worker_init_fn=init_fn,
                    collate_fn=my_collate,
                   # sampler=val_sampler,
                    pin_memory=True, #False, #True,
                    persistent_workers=True,
                    )

    def _test_dataloader(self, shuffle=False):
        init_fn = None
        return None
        # return DataLoader(self.datasets["test"], batch_size=self.batch_size,
        #                   num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

class DataModuleFromConfigMultiOcclusion(DataModuleFromConfig):
    def _train_dataloader(self):
        init_fn = None
        train_dataset = self.datasets["train"]
        return DataLoader(train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, # IMPORTANT: It is shuffled in the sampler. Hence, not needed here. It leads to error.
                          worker_init_fn=init_fn,
                          collate_fn=my_collate_multi_occlusion,
                          #sampler=train_sampler,
                          pin_memory=True, #False, #True,
                          persistent_workers=True,
                          )
    def _val_dataloader(self, shuffle=False):
        init_fn = None
        val_dataset = self.datasets["validation"]
        return DataLoader(val_dataset, batch_size=self.batch_size,
                    num_workers=self.num_workers, shuffle=True,
                    worker_init_fn=init_fn,
                    collate_fn=my_collate_multi_occlusion,
                   # sampler=val_sampler,
                    pin_memory=True, #False, #True,
                    persistent_workers=True,
                    )

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        print("Setup callback")

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0 and isinstance(exception, KeyboardInterrupt):
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            print(f'Creating directories: {self.logdir}, {self.ckptdir}, {self.cfgdir}')
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            # print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass