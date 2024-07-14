# %%
import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger
from copy import deepcopy

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_loftr import PL_LoFTR

loguru_logger = get_rank_zero_only_logger(loguru_logger)

TRAIN_IMG_SIZE = 640
n_nodes = 1
n_gpus_per_node = 4
torch_num_workers = 4
batch_size = 1
pin_memory = True
exp_name = f"outdoor-ds-{TRAIN_IMG_SIZE}-bs=1"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_cfg_path', type=str, default=f"configs/data/megadepth_trainval_{TRAIN_IMG_SIZE}.py")
parser.add_argument(
    '--main_cfg_path', type=str, default="configs/loftr/outdoor/loftr_ds_dense.py")
parser.add_argument('--exp_name', type=str, default=exp_name)
parser.add_argument(
    '--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument(
    '--num_workers', type=int, default=4)
parser.add_argument(
    '--pin_memory', type=lambda x: bool(strtobool(x)),
    nargs='?', default=True, help='whether loading data to pinned memory or not')
parser.add_argument(
    '--ckpt_path', type=str, default=None,
    help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
parser.add_argument(
    '--disable_ckpt', action='store_true',
    help='disable checkpoint saving (useful for debugging).')
parser.add_argument(
    '--profiler_name', type=str, default=None,
    help='options: [inference, pytorch], or leave it unset')
parser.add_argument(
    '--parallel_load_data', action='store_true',
    help='load datasets in with multiple processes.')

args = pl.Trainer.add_argparse_args(parser).parse_args([])
rank_zero_only(pprint.pprint)(vars(args))

# init default-cfg and merge it with the main- and data-cfg
config = get_cfg_defaults()
config.merge_from_file(args.main_cfg_path)
config.merge_from_file(args.data_cfg_path)
pl.seed_everything(config.TRAINER.SEED)  # reproducibility

# scale lr and warmup-step automatically
args.gpus = 1
config.TRAINER.WORLD_SIZE = 1
config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
_scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
config.TRAINER.SCALING = _scaling
config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

# %%
profiler = build_profiler(args.profiler_name)
model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
loguru_logger.info(f"LoFTR LightningModule initialized!")

data_module = MultiSceneDataModule(args, config)
loguru_logger.info(f"LoFTR DataModule initialized!")

logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
ckpt_dir = Path(logger.log_dir) / 'checkpoints'

ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=5, mode='max',
                                save_last=True,
                                dirpath=str(ckpt_dir),
                                filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [lr_monitor]
if not args.disable_ckpt:
    callbacks.append(ckpt_callback)

plugins = DDPPlugin(
    find_unused_parameters=False,
    num_nodes=args.num_nodes,
    sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
)
trainer = pl.Trainer.from_argparse_args(
    args,
    plugins=plugins,
    gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
    callbacks=callbacks,
    logger=logger,
    sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
    replace_sampler_ddp=False,  # use custom sampler
    reload_dataloaders_every_epoch=False,  # avoid repeated samples!
    weights_summary='full',
    profiler=profiler,
)

# %%
loguru_logger.info(f"Trainer initialized!")
loguru_logger.info(f"Start training!")
trainer.fit(model, datamodule=data_module)

# %%
# MVP of inference for debugging and understanding.

def to_device(v, device):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    elif isinstance(v, list):
        return [to_device(item, device) for item in v]
    elif isinstance(v, dict):
        return {k: to_device(item, device) for k, item in v.items()}
    else:
        return v

data_module.prepare_data()
data_module.setup(stage='fit')

train_dataloader = data_module.train_dataloader()
batch = next(iter(train_dataloader))
batch.keys()

# %%
batch2 = to_device(deepcopy(batch), model.device)
model._trainval_inference(batch2)
batch2.keys()

# %%
