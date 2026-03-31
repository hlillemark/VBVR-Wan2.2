"""
Evaluate Wan2.2-I2V-A14B model on VBVR-Bench.

This script generates videos from the VBVR-Bench evaluation dataset using
the Wan2.2-I2V-A14B pipeline, optionally with trained LoRA adapters for
both high-noise and low-noise DiT models.

By default it evaluates on both In-Domain_50 and Out-of-Domain_50 splits,
iterating over all task directories found under each split.

Usage:
    # Evaluate base model (no LoRA) on all splits and tasks
    python examples/wanvideo/model_training/validate_lora/eval_vbvr_bench.py \
        --eval_root ./data/VBVR-Bench \
        --output_root ./outputs/eval/Wan2.2

    # Evaluate with LoRA (both high and low noise models)
    python examples/wanvideo/model_training/validate_lora/eval_vbvr_bench.py \
        --eval_root ./data/VBVR-Bench \
        --output_root ./outputs/eval/Wan2.2_lora \
        --high_noise_lora_path ./outputs/wan2.2-I2V-14B_vbvr/high_noise/step-15000.safetensors \
        --low_noise_lora_path ./outputs/wan2.2-I2V-14B_vbvr/low_noise/step-15000.safetensors

Prerequisites:
    Download VBVR-Bench-Data from:
    https://huggingface.co/datasets/Video-Reason/VBVR-Bench-Data
"""

import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import os
import subprocess
import argparse

EVAL_SPLITS = ["In-Domain_50", "Out-of-Domain_50"]


def get_video_frame_count(video_path):
    """Get the number of frames in a video using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_packets', '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(result.stdout.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Wan2.2-I2V-A14B on VBVR-Bench")
    parser.add_argument('--eval_root', type=str, required=True,
                        help='Root directory of VBVR-Bench evaluation data (e.g., ./data/VBVR-Bench)')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root directory for output videos')
    parser.add_argument('--high_noise_lora_path', type=str, default=None,
                        help='Path to trained high-noise LoRA weights (optional)')
    parser.add_argument('--low_noise_lora_path', type=str, default=None,
                        help='Path to trained low-noise LoRA weights (optional)')
    parser.add_argument('--lora_alpha', type=float, default=1.0,
                        help='LoRA alpha scaling factor (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for generation (default: 1)')
    parser.add_argument('--fps', type=int, default=16,
                        help='Output video FPS (default: 16)')
    return parser.parse_args()


def main():
    args = parse_args()

    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    # Initialize the Wan2.2-I2V-A14B pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="google/umt5-xxl/"),
    )

    # Optionally load LoRA weights
    if args.high_noise_lora_path is not None:
        print(f"Loading high-noise LoRA from: {args.high_noise_lora_path} (alpha={args.lora_alpha})")
        pipe.load_lora(pipe.dit, args.high_noise_lora_path, alpha=args.lora_alpha)
    if args.low_noise_lora_path is not None:
        print(f"Loading low-noise LoRA from: {args.low_noise_lora_path} (alpha={args.lora_alpha})")
        pipe.load_lora(pipe.dit2, args.low_noise_lora_path, alpha=args.lora_alpha)
    if args.high_noise_lora_path is None and args.low_noise_lora_path is None:
        print("Running base model evaluation (no LoRA)")

    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )

    os.makedirs(args.output_root, exist_ok=True)

    # Iterate over evaluation splits (In-Domain_50, Out-of-Domain_50)
    for split in EVAL_SPLITS:
        split_path = os.path.join(args.eval_root, split)
        split_output_dir = os.path.join(args.output_root, split)

        if not os.path.isdir(split_path):
            print(f"Split directory {split_path} does not exist, skipping")
            continue

        # Auto-discover all task directories under this split
        task_dirs = sorted([
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        ])
        print(f"\n{'='*60}")
        print(f"Split: {split} ({len(task_dirs)} tasks)")
        print(f"{'='*60}")

        for task_dir in task_dirs:
            task_path = os.path.join(split_path, task_dir)
            task_output_dir = os.path.join(split_output_dir, task_dir)
            os.makedirs(task_output_dir, exist_ok=True)

            # Get all sample directories
            sample_dirs = sorted([d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))])

            for sample_dir in sample_dirs:
                sample_path = os.path.join(task_path, sample_dir)

                # Define paths for input files
                first_frame_path = os.path.join(sample_path, "first_frame.png")
                ground_truth_path = os.path.join(sample_path, "ground_truth.mp4")
                prompt_path = os.path.join(sample_path, "prompt.txt")
                output_video_path = os.path.join(task_output_dir, f"{sample_dir}.mp4")

                # Skip if output already exists
                if os.path.exists(output_video_path):
                    print(f"Skipping {split}/{task_dir}/{sample_dir} - already exists")
                    continue

                # Check if all required files exist
                if not all(os.path.exists(p) for p in [first_frame_path, ground_truth_path, prompt_path]):
                    print(f"Skipping {split}/{task_dir}/{sample_dir} - missing required files")
                    continue

                try:
                    # Load input image
                    input_image = Image.open(first_frame_path)

                    # Get frame count from ground truth video
                    num_frames = get_video_frame_count(ground_truth_path)

                    # Read prompt
                    with open(prompt_path, 'r') as f:
                        prompt = f.read().strip()

                    print(f"Processing {split}/{task_dir}/{sample_dir}: {num_frames} frames")

                    # Generate video
                    video = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        input_image=input_image,
                        num_frames=num_frames,
                        seed=args.seed,
                        tiled=True,
                        height=input_image.height,
                        width=input_image.width,
                    )

                    # Save video
                    save_video(video, output_video_path, fps=args.fps, quality=5)
                    print(f"Saved: {output_video_path}")

                except Exception as e:
                    print(f"Error processing {split}/{task_dir}/{sample_dir}: {e}")
                    continue

            print(f"Done processing task: {split}/{task_dir}")

    print("\nAll splits and tasks done!")


if __name__ == "__main__":
    main()
