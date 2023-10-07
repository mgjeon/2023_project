import argparse
import json
import os
import torch

from nnf2.train.module import NF2Module, save
from nnf2.train.data_loader import ISEEDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LambdaCallback, ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True, help="config file for the simulation"
)
args = parser.parse_args()


with open(args.config) as config:
    info = json.load(config)

for key, value in info.items():
    args.__dict__[key] = value


base_path = args.base_path
os.makedirs(base_path, exist_ok=True)

wandb_id = args.logging["wandb_id"] if "wandb_id" in args.logging else None
log_model = (
    args.logging["wandb_log_model"] if "wandb_log_model" in args.logging else False
)
wandb_logger = WandbLogger(
    project=args.logging["wandb_project"],
    name=args.logging["wandb_name"],
    offline=False,
    entity=args.logging["wandb_entity"],
    id=wandb_id,
    dir=base_path,
    log_model=log_model,
)
wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

if "work_directory" not in args.data or args.data["work_directory"] is None:
    args.data["work_directory"] = base_path

if args.data["type"] == "isee":
    data_module = ISEEDataModule(**args.data)

validation_settings = {
    "cube_shape": data_module.cube_dataset.coords_shape,
    "gauss_per_dB": args.data["b_norm"],
    "Mm_per_ds": data_module.Mm_per_pixel * args.data["spatial_norm"],
}

try:
    meta_path = args.meta_path if args.meta_path else None
except AttributeError:
    meta_path = None

nf2 = NF2Module(validation_settings, meta_path=meta_path, **args.model, **args.training)

n_gpus = torch.cuda.device_count()

save_path = os.path.join(base_path, "extrapolation_result.nf2")
config = {"data": args.data, "model": args.model, "training": args.training}

save_callback = LambdaCallback(
    on_validation_end=lambda *args: save(save_path, nf2.model, data_module, config)
)

checkpoint_callback = ModelCheckpoint(
    dirpath=base_path,
    every_n_train_steps=int(args.training["validation_interval"]),
    save_last=True,
)

trainer = Trainer(
    max_epochs=1,
    logger=wandb_logger,
    devices=n_gpus,
    accelerator="gpu",
    strategy="dp",
    num_sanity_val_steps=0,
    val_check_interval=int(args.training["validation_interval"]),
    gradient_clip_val=0.1,
    callbacks=[checkpoint_callback, save_callback],
)

trainer.fit(nf2, data_module, ckpt_path="last")
