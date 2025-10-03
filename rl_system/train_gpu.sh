#!/bin/bash
# Training script with proper CUDA initialization

# Clean CUDA environment
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Clear any Python CUDA cache
pkill -9 python 2>/dev/null || true
sleep 1

# Run training
python train.py --config config.yaml "$@"
