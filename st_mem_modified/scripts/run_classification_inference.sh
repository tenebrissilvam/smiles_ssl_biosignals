#!/bin/bash

# Example script for running classification inference with pretrained ST-MEM encoder

# Configuration
MODEL_TYPE="encoder"
MODEL_PATH="./weights/st_mem_vit_base_encoder.pth"
CONFIG_PATH="./configs/downstream/st_mem.yaml"  # Use downstream config for encoder
OUTPUT_DIR="./inference_results/classification"
DEVICE="auto"  # auto, cpu, cuda, or mps

# Option 1: Use dummy data from config (default behavior)
echo "Running classification inference on dummy data..."
python inference.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    --batch_size 8

echo "Done! Results saved to: $OUTPUT_DIR"
echo "Check classification_results.csv for detailed results"

# Option 2: Use specific data files (uncomment to use)
# echo "Running classification inference on specific data..."
# python inference.py \
#     --model_type $MODEL_TYPE \
#     --model_path $MODEL_PATH \
#     --config_path $CONFIG_PATH \
#     --data_csv "./data/dummy/index.csv" \
#     --data_path "./data/dummy/ecgs" \
#     --output_dir $OUTPUT_DIR \
#     --device $DEVICE

# Option 3: Use single file (uncomment to use)
# echo "Running classification inference on single file..."
# python inference.py \
#     --model_type $MODEL_TYPE \
#     --model_path $MODEL_PATH \
#     --config_path $CONFIG_PATH \
#     --data_path "./data/dummy/ecgs/000.pkl" \
#     --output_dir $OUTPUT_DIR \
#     --device $DEVICE
