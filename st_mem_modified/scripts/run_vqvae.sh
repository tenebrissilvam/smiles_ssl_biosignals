#!/bin/bash

# Run VQ-VAE training
python main_vqvae.py \
    --config_path ./configs/pretrain/vqvae.yaml \
    --output_dir ./output \
    --exp_name vqvae_ecg
