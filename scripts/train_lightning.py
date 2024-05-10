import argparse
import datetime
import glob
import os
import pathlib
import sys

import lightning as L
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.trainer import Trainer
from omegaconf import OmegaConf

sys.path.append(str(pathlib.Path(__file__).parents[1]))

from sceneinformer.utils.utils import instantiate_from_config


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        default=["configs/scene_informer.yaml"],
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="./outputs/", #/raid/blange/occlusion_logs",
        help="directory for logging",
    )
    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
        os.makedirs(logdir, exist_ok=True)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        model = instantiate_from_config(config.model)
        model.save_logdir = logdir
        trainer_kwargs = dict()

        # default logger configs
        # logger_cfgs = {
        #     "wandb": {
        #         "target": "lightning.pytorch.loggers.WandbLogger",
        #         "params": {
        #             "project": "OcclusionInformer",
        #             "name": nowname,
        #             "save_dir": logdir,
        #             "offline": opt.debug,
        #             "id": nowname,
        #             "log_model": "all",  
        #             "dir": logdir,
        #             "config": dict(config),
        #         }
        #     },
        # }
        # if "logger" in lightning_config:
        #     logger_cfg = lightning_config.logger
        # else:
        #     logger_cfg = OmegaConf.create()
        # logger_cfg = OmegaConf.merge(logger_cfgs, logger_cfg)

        # trainer_kwargs["logger"] = [instantiate_from_config(logger_cfgs["wandb"])]
        # (trainer_kwargs["logger"][0]).watch(model)    

        default_callbacks_cfg = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "lightning.pytorch.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": ckptdir,
                    "filename": "interval-{epoch:06}-{step:09}",
                    "verbose": True,
                    'save_top_k': -1,
                    'every_n_train_steps': 20000,
                }
            },
            "setup_callback": {
                "target": "sceneinformer.utils.callbacks.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "learning_rate_logger": {
                "target": "lightning.pytorch.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                }
            },
        }

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer(
                          precision=trainer_config["precision"],
                          benchmark=trainer_config["benchmark"],
                          accumulate_grad_batches=trainer_config["accumulate_grad_batches"],
                          val_check_interval=trainer_config["val_check_interval"],
                          limit_val_batches=trainer_config["limit_val_batches"],
                          gradient_clip_val=trainer_config["gradient_clip_norm"],
                          #logger=trainer_kwargs["logger"],
                          devices=trainer_config["devices"],
                          accelerator=trainer_config["accelerator"],
                          callbacks=trainer_kwargs["callbacks"]) 

        trainer.logdir = logdir  

        # data
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")


        print(f'Using {lightning_config.trainer.devices} GPUs.')
        if isinstance(lightning_config.trainer.devices, list):
            ngpu = len(lightning_config.trainer.devices) #len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = lightning_config.trainer.devices

        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        if opt.resume:
            trainer.fit(model, data, ckpt_path=opt.resume_from_checkpoint)
        else:
            trainer.fit(model, data)

        # run
        if opt.train:
            try:
                if opt.resume:
                    trainer.fit(model, data, ckpt_path=opt.resume_from_checkpoint)
                else:
                    trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            pass
            # trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        pass