#!/bin/bash

# Set paths
PTB_XL_DIR="/path/to/ptb-xl"  # Path to PTB-XL dataset
OUTPUT_DIR="/path/to/processed/ptb-xl"  # Path to store processed ECG files
INDEX_DIR="/path/to/ptb-xl/index"  # Path to store index files

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${INDEX_DIR}

# Process PTB-XL dataset using the data processing script
python data/process_ecg.py \
    --input_dir ${PTB_XL_DIR}/records500 \
    --output_dir ${OUTPUT_DIR} \
    --index_path ${INDEX_DIR}/ptb_xl_all.csv \
    --database_type ptb-xl

# Prepare train/valid/test splits
python prepare_ptb_xl.py \
    --ptb_xl_dir ${PTB_XL_DIR} \
    --output_dir ${INDEX_DIR} \
    --valid_ratio 0.1 \
    --test_ratio 0.1 \
    --random_seed 42
