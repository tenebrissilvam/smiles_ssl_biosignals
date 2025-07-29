import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import torch
from ecg_data_feature_extraction_v3 import *

# from ECG_features_extracting import augment_ecg_channels
from ECG_features_extracting_fin_v3 import augment_ecg_channels
from ecg_jepa_feature_extracting_v3 import ecg_jepa  # ###Егор добавил признаки
from scipy.signal import resample
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader


def downsample_waves(waves, new_size):
    return np.array([resample(wave, new_size, axis=1) for wave in waves])


# Argument parser
parser = argparse.ArgumentParser(description="Pretrain the JEPA model with ECG data")
parser.add_argument(
    "--mask_scale", type=float, nargs=2, default=[0.175, 0.225], help="Scale of masking"
)
parser.add_argument(
    "--batch_size", type=int, default=8, help="Batch size"
)  ###Egor modify
parser.add_argument(
    "--lr", type=float, default=5e-20, help="Learning rate"
)  ###Egor modify
parser.add_argument(
    "--mask_type", type=str, default="block", help="Type of masking"
)  # 'block' or 'random'
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument(
    "--wd", type=float, default=5e-10, help="Weight decay"
)  ###Egor modify
parser.add_argument(
    "--data_dir_shao",
    type=str,
    default="C:/Users/padin/Desktop/Scoltech/BIOSIGNALS/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords_cut/",
    help="Directory for Shaoxing data",
)


parser.add_argument(
    "--data_dir_code15",
    type=str,
    default="C:/Users/padin/Desktop/Scoltech/BIOSIGNALS/code15",
    help="Directory for Code15 data",
)

args = parser.parse_args()

# Access the arguments like this
mask_scale = tuple(args.mask_scale)
batch_size = args.batch_size
lr = args.lr
mask_type = args.mask_type
epochs = args.epochs
wd = args.wd
data_dir_shao = args.data_dir_shao
data_dir_code15 = args.data_dir_code15

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create logs directory if it doesn't exist
save_dir = f"./weights/ecg_jepa_{timestamp}_{mask_scale}"
os.makedirs(save_dir, exist_ok=True)
log_file = os.path.join(save_dir, f"training_{timestamp}.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_params(params_dict):
    for key, value in params_dict.items():
        logging.info(f"{key}: {value}")
        print(f"{key}: {value}")


os.makedirs(save_dir, exist_ok=True)

start_time = time.time()


#### Egorjust for test####
# Shaoxing (Ningbo + Chapman)
waves_shaoxing = waves_shao(data_dir_shao)
waves_shaoxing = downsample_waves(waves_shaoxing, 2500)
print(f"Shao waves shape: {waves_shaoxing.shape}")
logging.info(f"Shao waves shape: {waves_shaoxing.shape}")
waves_shaoxing = augment_ecg_channels(waves_shaoxing)


dataset = ECGDataset_pretrain(waves_shaoxing)
# dataset = []
#### Egorjust for test####


# Code15
# dataset_code15 = Code15Dataset(data_dir_code15)

#### Egorjust for test####

# print(f'Code15 waves shape: ({len(dataset_code15.file_indices)}, 8, 2500)')
# logging.info(f'Code15 waves shape: ({len(dataset_code15.file_indices)}, 8, 2500)')

# loading_time = time.time() - start_time
# print(f'Data loading time: {loading_time:.2f}s')

# dataset = ConcatDataset([dataset, dataset_code15])
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#### Egorjust for test####

#### Egorjust for test####
del waves_shaoxing
#### Egorjust for test####
# model = ecg_jepa(encoder_embed_dim=768,
#                 encoder_depth=12,
#                 encoder_num_heads=16,
#                 predictor_embed_dim=384,
#                 predictor_depth=6,
#                 predictor_num_heads=12,
#                 drop_path_rate=0.1,
#                 mask_scale=mask_scale,
#                 mask_type=mask_type,
#                 pos_type='sincos',
#                 c=8,
#                 p=50,
#                 t=50).to('cuda')

model = ecg_jepa(
    encoder_embed_dim=384,
    encoder_depth=6,
    encoder_num_heads=16,
    predictor_embed_dim=192,
    predictor_depth=6,
    predictor_num_heads=12,
    drop_path_rate=0.1,
    mask_scale=mask_scale,
    mask_type=mask_type,
    pos_type="sincos",
    c=19,
    p=50,
    t=50,
)


total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params_million = total_params / 1000000
logging.info(
    f"Total number of learnable parameters: {total_params_million:.2f} million"
)

param_groups = [
    {
        "params": (
            p
            for n, p in model.named_parameters()
            if (p.requires_grad) and ("bias" not in n) and (len(p.shape) != 1)
        )
    },
    {
        "params": (
            p
            for n, p in model.named_parameters()
            if (p.requires_grad) and (("bias" in n) or (len(p.shape) == 1))
        ),
        "WD_exclude": True,
        "weight_decay": 0,
    },
]

iterations_per_epoch = len(train_loader)
# iterations_per_epoch = 50
optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
scheduler = CosineLRScheduler(
    optimizer,
    t_initial=iterations_per_epoch * epochs,
    cycle_mul=1,
    lr_min=1e-8,  ##Egor modify
    cycle_decay=0.1,
    warmup_lr_init=1e-7,
    warmup_t=10 * iterations_per_epoch,  ##Egor modify
    cycle_limit=1,
    t_in_epochs=True,
)

ema = [0.996, 1.0]
momentum_target_encoder_scheduler = (
    ema[0] + i * (ema[1] - ema[0]) / (iterations_per_epoch * epochs)
    for i in range(int(iterations_per_epoch * epochs) + 1)
)

# Log hyperparameters and model arguments
hyperparameters = vars(args)
log_params(hyperparameters)

###Egor commenting########
scaler = GradScaler()

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    total_loss = 0.0
    for minibatch, wave in enumerate(train_loader):
        scheduler.step(epoch * iterations_per_epoch + minibatch)
        bs, c, t = wave.shape

        ###egor_masking
        ##wave = wave.to('cuda')

        optimizer.zero_grad()
        with autocast():  # Enable mixed precision
            loss = model(wave)

        # Scale the loss and backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        print("total_loss=", total_loss)
        with torch.no_grad():
            m = next(momentum_target_encoder_scheduler)
            for param_q, param_k in zip(
                model.encoder.parameters(), model.target_encoder.parameters()
            ):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    total_loss /= len(train_loader)
    epoch_time = time.time() - start_time
    print(
        f"epoch={epoch:04d}/{epochs:04d}  loss={total_loss:.4f}  time={epoch_time:.2f}s"
    )
    logging.info(
        f"epoch={epoch:04d}/{epochs:04d}  loss={total_loss:.4f}  time={epoch_time:.2f}s"
    )
    ##Egor comment####

    if epoch > 1 and (epoch + 1) % 5 == 0:
        print("epoch=", epoch)
        model.to("cpu")
        torch.save(
            {
                "encoder": model.encoder.state_dict(),
                "epoch": epoch,
            },
            f"{save_dir}/epoch{epoch + 1}.pth",
        )
        model.to("cpu")
