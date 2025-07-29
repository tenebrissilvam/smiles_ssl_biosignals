#!/bin/bash

# Set paths
OUTPUT_DIR="./outs"
EXP_NAME="triplet_linprobe"
ENCODER_PATH="/Users/tenebrissilvam/Code/smiles_2025/ST-MEM/outs/triplet_exp/best_auroc_model.pth"  # Path to your pretrained ST-MEM encoder

# Run linear probing
python main_downstream.py \
    --config_path /Users/tenebrissilvam/Code/smiles_2025/ST-MEM/configs/downstream/triplet_linprobe.yaml \
    --output_dir ${OUTPUT_DIR} \
    --exp_name ${EXP_NAME} \
    --encoder_path ${ENCODER_PATH}
