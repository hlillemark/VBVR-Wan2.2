#!/bin/bash
# =============================================================================
# LTX-2.3 I2AV Training Script for VBVR Dataset
# =============================================================================
# This script trains a LoRA adapter for the LTX-2.3 Image-to-Audio-Video model
# using the VBVR-Dataset.
#
# Training is split into two stages:
#   Stage 1 (Data Processing): Encodes text/image/audio with frozen encoders
#   Stage 2 (Training):        Trains the DiT transformer with LoRA
#
# Usage:
#   bash scripts/LTX2.3-I2AV_vbvr_dataset.sh
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
HEIGHT=512
WIDTH=512
NUM_FRAMES=209
DATASET_REPEAT=1
LEARNING_RATE=1e-4
NUM_EPOCHS=1
LORA_RANK=32
SAVE_STEPS=10000

# Output directories
DATA_PROCESS_OUTPUT_PATH="./outputs/LTX2.3-I2AV_vbvr/data_process"
MODEL_OUTPUT_PATH="./outputs/LTX2.3-I2AV_vbvr/model"

# ---- Distributed Training Configuration ----
NUM_GPUS=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}
NUM_PROCESSES=$((NUM_GPUS * NUM_NODES))

REMOVE_PREFIX_IN_CKPT="pipe.dit."

echo "=================================="
echo "LTX-2.3 I2AV VBVR Training"
echo "=================================="
echo "REPO_DIR:          ${REPO_DIR}"
echo "DATASET_CONFIG:    ${DATASET_CONFIG_PATH}"
echo "Resolution:        ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
echo "NUM_PROCESSES:     ${NUM_PROCESSES} (${NUM_GPUS} GPUs x ${NUM_NODES} nodes)"
echo "Learning Rate:     ${LEARNING_RATE}"
echo "LoRA Rank:         ${LORA_RANK}"
echo "Data Process Path: ${DATA_PROCESS_OUTPUT_PATH}"
echo "Model Output Path: ${MODEL_OUTPUT_PATH}"
echo "=================================="

# ---- Stage 1: Data Processing ----
echo "[Stage 1/2] Data Processing (encoding text, images, audio with frozen models)"

cd ${REPO_DIR} && \
accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_NODES} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${NODE_RANK} \
    ${REPO_DIR}/examples/ltx2/model_training/train.py \
    --dataset_config_path ${DATASET_CONFIG_PATH} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --dataset_repeat ${DATASET_REPEAT} \
    --model_id_with_origin_paths "DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors" \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --remove_prefix_in_ckpt ${REMOVE_PREFIX_IN_CKPT} \
    --output_path ${DATA_PROCESS_OUTPUT_PATH} \
    --lora_base_model "dit" \
    --lora_target_modules "to_k,to_q,to_v,to_out.0" \
    --lora_rank ${LORA_RANK} \
    --extra_inputs "input_image" \
    --data_file_keys 'clip_path' \
    --save_steps ${SAVE_STEPS} \
    --task "sft:data_process"

echo "[Stage 1/2] Data processing complete."

# ---- Stage 2: LoRA Training ----
echo "[Stage 2/2] Training DiT LoRA (using pre-processed data)"

cd ${REPO_DIR} && \
accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_NODES} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${NODE_RANK} \
    ${REPO_DIR}/examples/ltx2/model_training/train.py \
    --dataset_base_path ${DATA_PROCESS_OUTPUT_PATH} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --dataset_repeat ${DATASET_REPEAT} \
    --model_id_with_origin_paths "DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors" \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --remove_prefix_in_ckpt ${REMOVE_PREFIX_IN_CKPT} \
    --output_path ${MODEL_OUTPUT_PATH} \
    --lora_base_model "dit" \
    --lora_target_modules "to_k,to_q,to_v,to_out.0" \
    --lora_rank ${LORA_RANK} \
    --extra_inputs "input_image" \
    --data_file_keys 'clip_path' \
    --save_steps ${SAVE_STEPS} \
    --find_unused_parameters \
    --task "sft:train"

echo "[Stage 2/2] LoRA training complete."
echo "All training done! Model saved to: ${MODEL_OUTPUT_PATH}"
