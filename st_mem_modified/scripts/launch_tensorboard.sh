#!/bin/bash
# Launch TensorBoard to view training logs

# Default log directory
LOG_DIR="./output"

# Check if a custom log directory was provided
if [ "$#" -eq 1 ]; then
    LOG_DIR="$1"
fi

echo "Starting TensorBoard with logs from: $LOG_DIR"
tensorboard --logdir="$LOG_DIR" --port=6006
