#!/bin/bash
# =============================================================================
# Wan2.2-I2V-A14B Training Script for VBVR Dataset
# =============================================================================
# This script trains LoRA adapters for the Wan2.2-I2V-A14B model using the
# VBVR-Dataset.
#
# Wan2.2-I2V-A14B uses a dual-DiT architecture:
#   - High Noise Model (dit):  timestep range [0, 0.358)
#   - Low Noise Model (dit2):  timestep range [0.358, 1.0]
# Each model is trained separately with its own LoRA adapter.
#
# Usage:
#   bash scripts/Wan2.2-I2V-14B_vbvr_dataset.sh
#
# Prerequisites:
#   1. Download VBVR-Dataset from:
#      https://huggingface.co/datasets/Video-Reason/VBVR-Dataset
#      and extract into ./data/
#   2. Install dependencies: pip install -e .
#   3. Set up accelerate config for multi-GPU training
# =============================================================================

# ---- User Configuration (modify as needed) ----
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_CONFIG_PATH="${REPO_DIR}/configs/vbvr_dataset.json"
HEIGHT=384
WIDTH=384
NUM_FRAMES=209
DATASET_REPEAT=1
LEARNING_RATE=1e-4
NUM_EPOCHS=1
LORA_RANK=32
SAVE_STEPS=10000

# Output directories for trained LoRA weights
HIGH_NOISE_OUTPUT_PATH="./outputs/Wan2.2-I2V-14B_vbvr/high_noise"
LOW_NOISE_OUTPUT_PATH="./outputs/Wan2.2-I2V-14B_vbvr/low_noise"

# ---- Distributed Training Configuration ----
NUM_GPUS=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}
NUM_PROCESSES=$((NUM_GPUS * NUM_NODES))

REMOVE_PREFIX_IN_CKPT="pipe.dit."

echo "=================================="
echo "Wan2.2-I2V-A14B VBVR Training"
echo "=================================="
echo "REPO_DIR:          ${REPO_DIR}"
echo "DATASET_CONFIG:    ${DATASET_CONFIG_PATH}"
echo "Resolution:        ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
echo "NUM_PROCESSES:     ${NUM_PROCESSES} (${NUM_GPUS} GPUs x ${NUM_NODES} nodes)"
echo "Learning Rate:     ${LEARNING_RATE}"
echo "LoRA Rank:         ${LORA_RANK}"
echo "High Noise Output: ${HIGH_NOISE_OUTPUT_PATH}"
echo "Low Noise Output:  ${LOW_NOISE_OUTPUT_PATH}"
echo "=================================="

# ---- Train High Noise Model LoRA ----
echo "[Step 1/2] Training High Noise Model LoRA (timestep: 0 ~ 0.358)"

cd ${REPO_DIR} && \
export DIFFSYNTH_DOWNLOAD_SOURCE="huggingface" && \
accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_NODES} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${NODE_RANK} \
    ${REPO_DIR}/examples/wanvideo/model_training/train.py \
    --dataset_config_path ${DATASET_CONFIG_PATH} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --dataset_repeat ${DATASET_REPEAT} \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --remove_prefix_in_ckpt ${REMOVE_PREFIX_IN_CKPT} \
    --output_path ${HIGH_NOISE_OUTPUT_PATH} \
    --lora_base_model 'dit' \
    --lora_target_modules 'q,k,v,o,ffn.0,ffn.2' \
    --lora_rank ${LORA_RANK} \
    --extra_inputs 'input_image' \
    --max_timestep_boundary 0.358 \
    --min_timestep_boundary 0 \
    --data_file_keys 'clip_path' \
    --save_steps ${SAVE_STEPS}

echo "[Step 1/2] High Noise Model training complete."

# ---- Train Low Noise Model LoRA ----
echo "[Step 2/2] Training Low Noise Model LoRA (timestep: 0.358 ~ 1.0)"

cd ${REPO_DIR} && \
export DIFFSYNTH_DOWNLOAD_SOURCE="huggingface" && \
accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_NODES} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${NODE_RANK} \
    ${REPO_DIR}/examples/wanvideo/model_training/train.py \
    --dataset_config_path ${DATASET_CONFIG_PATH} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --dataset_repeat ${DATASET_REPEAT} \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --remove_prefix_in_ckpt ${REMOVE_PREFIX_IN_CKPT} \
    --output_path ${LOW_NOISE_OUTPUT_PATH} \
    --lora_base_model 'dit' \
    --lora_target_modules 'q,k,v,o,ffn.0,ffn.2' \
    --lora_rank ${LORA_RANK} \
    --extra_inputs 'input_image' \
    --max_timestep_boundary 1 \
    --min_timestep_boundary 0.358 \
    --data_file_keys 'clip_path' \
    --save_steps ${SAVE_STEPS}

echo "[Step 2/2] Low Noise Model training complete."
echo "All training done!"
