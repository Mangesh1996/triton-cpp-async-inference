#!/bin/bash
set -e

echo "[SYSTEM] Initializing Triton for Model Inference..."

# Check if NVIDIA Management Library is available (GPU check)
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU] Detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    echo "[WARNING] No GPU detected. Triton will run on CPU, which is slow for YOLOv8."
fi

# Optimization: Set environment variable to prevent memory fragmentation on some GPUs
export CUDA_MODULE_LOADING=LAZY

# Execute the CMD arguments (the tritonserver command)
exec "$@"