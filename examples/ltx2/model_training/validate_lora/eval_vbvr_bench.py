"""
Evaluate LTX-2.3 I2AV model on VBVR-Bench.

This script generates videos from the VBVR-Bench evaluation dataset using
the LTX-2.3 Image-to-Audio-Video pipeline, optionally with a trained LoRA.

By default it evaluates on both In-Domain_50 and Out-of-Domain_50 splits,
iterating over all task directories found under each split.

Usage:
    # Evaluate base model (no LoRA) on all splits and tasks
    python examples/ltx2/model_training/validate_lora/eval_vbvr_bench.py \
        --eval_root ./data/VBVR-Bench \
        --output_root ./outputs/eval/LTX2.3

    # Evaluate with LoRA
    python examples/ltx2/model_training/validate_lora/eval_vbvr_bench.py \
        --eval_root ./data/VBVR-Bench \
        --output_root ./outputs/eval/LTX2.3_lora \
        --lora_path ./outputs/LTX2.3-I2AV_vbvr/model/step-15469.safetensors

Prerequisites:
    Download VBVR-Bench-Data from:
    https://huggingface.co/datasets/Video-Reason/VBVR-Bench-Data
"""

import torch
from PIL import Image
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
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
    parser = argparse.ArgumentParser(description="Evaluate LTX-2.3 I2AV on VBVR-Bench")
    parser.add_argument('--eval_root', type=str, required=True,
                        help='Root directory of VBVR-Bench evaluation data (e.g., ./data/VBVR-Bench)')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root directory for output videos')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to trained LoRA weights (optional, omit for base model evaluation)')
    parser.add_argument('--lora_alpha', type=float, default=1.0,
                        help='LoRA alpha scaling factor (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for generation (default: 1)')
    parser.add_argument('--num_inference_steps', type=int, default=40,
                        help='Number of inference steps (default: 40)')
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

    # Initialize the LTX-2.3 pipeline
    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="transformer.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="video_vae_decoder.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vae_decoder.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vocoder.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="video_vae_encoder.safetensors", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
    )

    # Optionally load LoRA weights
    if args.lora_path is not None:
        print(f"Loading LoRA from: {args.lora_path} (alpha={args.lora_alpha})")
        pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)
    else:
        print("Running base model evaluation (no LoRA)")

    negative_prompt = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
        "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, "
        "unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, "
        "extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, "
        "camera shake, incorrect depth of field, background too sharp, background clutter, "
        "distracting reflections, harsh shadows, inconsistent lighting direction, color banding, "
        "cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, "
        "incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
        "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, "
        "background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, "
        "jittery movement, awkward pauses, incorrect timing, unnatural transitions, "
        "inconsistent framing, tilted camera, flat lighting, inconsistent tone, "
        "cinematic oversaturation, stylized filters, or AI artifacts."
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

                    # Generate video (and audio)
                    video, audio = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        input_images=[input_image],
                        input_images_indexes=[0],
                        input_images_strength=1.0,
                        num_frames=num_frames,
                        seed=args.seed,
                        tiled=True,
                        height=input_image.height,
                        width=input_image.width,
                        num_inference_steps=args.num_inference_steps,
                    )

                    # Save video with audio
                    write_video_audio_ltx2(
                        video=video,
                        audio=audio,
                        output_path=output_video_path,
                        fps=args.fps,
                        audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
                    )
                    print(f"Saved: {output_video_path}")

                except Exception as e:
                    print(f"Error processing {split}/{task_dir}/{sample_dir}: {e}")
                    continue

            print(f"Done processing task: {split}/{task_dir}")

    print("\nAll splits and tasks done!")


if __name__ == "__main__":
    main()
