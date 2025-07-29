#!/bin/bash

# Example script for running reconstruction inference with pretrained ST-MEM model

# Configuration
MODEL_TYPE="full"
MODEL_PATH="./weights/st_mem_vit_base_full.pth"
CONFIG_PATH="./configs/pretrain/st_mem.yaml"  # Use pretrain config for full model
OUTPUT_DIR="./inference_results/reconstruction"
MASK_RATIO=0.75
DEVICE="auto"  # auto, cpu, cuda, or mps

# Option 1: Use dummy data from config (default behavior)
echo "Running reconstruction inference on dummy data..."
python inference.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --mask_ratio $MASK_RATIO \
    --device $DEVICE \
    --batch_size 4

echo "Done! Results saved to: $OUTPUT_DIR"

# Option 2: Use specific data files (uncomment to use)
# echo "Running reconstruction inference on specific data..."
# python inference.py \
#     --model_type $MODEL_TYPE \
#     --model_path $MODEL_PATH \
#     --config_path $CONFIG_PATH \
#     --data_csv "./data/dummy/index.csv" \
#     --data_path "./data/dummy/ecgs" \
#     --output_dir $OUTPUT_DIR \
#     --mask_ratio $MASK_RATIO \
#     --device $DEVICE

# Option 3: Use single file (uncomment to use)
# echo "Running reconstruction inference on single file..."
# python inference.py \
#     --model_type $MODEL_TYPE \
#     --model_path $MODEL_PATH \
#     --config_path $CONFIG_PATH \
#     --data_path "./data/dummy/ecgs/000.pkl" \
#     --output_dir $OUTPUT_DIR \
#     --mask_ratio $MASK_RATIO \
#     --device $DEVICE
