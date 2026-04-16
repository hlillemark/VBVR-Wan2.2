#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import accelerate

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from diffsynth.evaluation import Wan22TI2V5BVBVRBenchRunner
from diffsynth.evaluation.vbvr_bench import log_eval_summary_to_wandb, parse_csv_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wan2.2-TI2V-5B inference in VBVR-Bench format and evaluate with VBVR-EvalKit by default."
    )
    parser.add_argument(
        "--bench-root",
        type=Path,
        required=True,
        help="Path to the downloaded VBVR-Bench ground truth directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_DIR / "models" / "Wan-AI" / "Wan2.2-TI2V-5B",
        help="Local Wan2.2-TI2V-5B model directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_DIR / "outputs" / "wan22_ti2v5b_vbvr_bench",
        help="Output directory that will contain In-Domain_50/ and Out-of-Domain_50/ folders.",
    )
    parser.add_argument(
        "--dit-checkpoint",
        type=Path,
        default=None,
        help="Optional full-finetune DiT checkpoint to load before inference.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="In-Domain_50,Out-of-Domain_50",
        help="Comma-separated benchmark splits to generate.",
    )
    parser.add_argument(
        "--finetuning-mode",
        type=str,
        default="auto",
        choices=["auto", "flow", "eqf"],
        help="Checkpoint conditioning mode. 'auto' reads the training metadata sidecar when available.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps used by the Wan sampler.",
    )
    parser.add_argument(
        "--inference-schedule",
        type=str,
        default="auto",
        choices=["auto", "linear", "sigma_shift", "sd3", "c_function"],
        help="Inference schedule family used to map solver noise levels to model timesteps. 'auto' uses the checkpoint's recommended mode-specific default.",
    )
    parser.add_argument(
        "--sigma-shift",
        type=float,
        default=5.0,
        help="Legacy Wan sigma-shift parameter used when --inference-schedule=sigma_shift.",
    )
    parser.add_argument(
        "--schedule-sd3-r",
        type=float,
        default=6.0,
        help="EqF-style SD3 warp parameter used when --inference-schedule=sd3.",
    )
    parser.add_argument(
        "--schedule-c-interp",
        type=float,
        default=0.8,
        help="EqF c-function breakpoint used when --inference-schedule=c_function.",
    )
    parser.add_argument(
        "--schedule-c-start",
        type=float,
        default=1.0,
        help="EqF c-function initial scale used when --inference-schedule=c_function.",
    )
    parser.add_argument(
        "--schedule-c-t-end",
        type=float,
        default=0.999,
        help="EqF c-function terminal native time used when --inference-schedule=c_function.",
    )
    parser.add_argument(
        "--schedule-c-grid-size",
        type=int,
        default=4096,
        help="Lookup-table resolution used when --inference-schedule=c_function.",
    )
    parser.add_argument(
        "--task-names",
        type=str,
        default=None,
        help="Optional comma-separated task names to restrict generation.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional maximum number of videos to generate, balanced across tasks.",
    )
    parser.add_argument(
        "--videos-per-task",
        type=int,
        default=1,
        help="Number of videos to generate per task before applying --max-videos.",
    )
    parser.add_argument(
        "--target-num-frames",
        type=int,
        default=209,
        help="Training-matched target frame count before 4n+1 adjustment.",
    )
    parser.add_argument(
        "--use-ground-truth-frame-count",
        action="store_true",
        help="Use the raw ground-truth frame count instead of training-matched clipping.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="FPS for saved MP4s.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base generation seed.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=1234,
        help="Seed for selecting which benchmark samples to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated videos.",
    )
    parser.add_argument(
        "--run-evalkit",
        action="store_true",
        default=True,
        help="Run VBVR-EvalKit immediately after generation. Enabled by default.",
    )
    parser.add_argument(
        "--no-run-evalkit",
        action="store_false",
        dest="run_evalkit",
        help="Skip VBVR-EvalKit and only generate benchmark videos.",
    )
    parser.add_argument(
        "--evalkit-path",
        type=Path,
        default=Path("/home/ubuntu/projects/VBVR-EvalKit"),
        help="Path to the VBVR-EvalKit repository.",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=None,
        help="Directory for EvalKit result JSONs. Defaults to <output-root>/evalkit_results.",
    )
    parser.add_argument(
        "--eval-name",
        type=str,
        default=None,
        help="Optional EvalKit run name.",
    )
    parser.add_argument(
        "--task-specific-only",
        action="store_true",
        default=True,
        help="Use EvalKit task-specific-only scoring, matching run_evaluation.py.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="eqforcing",
        help="Weights & Biases entity.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="eqf",
        help="Weights & Biases project.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name.",
    )
    return parser.parse_args()


def init_wandb(args):
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without wandb logging.")
        return None

    run_name = args.wandb_run_name or f"wan22-ti2v5b-vbvr-bench-{int(time.time())}"
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        config={
            "bench_root": str(args.bench_root),
            "model_dir": str(args.model_dir),
            "output_root": str(args.output_root),
            "dit_checkpoint": None if args.dit_checkpoint is None else str(args.dit_checkpoint),
            "splits": parse_csv_list(args.splits),
            "task_names": parse_csv_list(args.task_names),
            "max_videos": args.max_videos,
            "videos_per_task": args.videos_per_task,
            "finetuning_mode": args.finetuning_mode,
            "num_inference_steps": args.num_inference_steps,
            "inference_schedule": args.inference_schedule,
            "sigma_shift": args.sigma_shift,
            "schedule_sd3_r": args.schedule_sd3_r,
            "schedule_c_interp": args.schedule_c_interp,
            "schedule_c_start": args.schedule_c_start,
            "schedule_c_t_end": args.schedule_c_t_end,
            "schedule_c_grid_size": args.schedule_c_grid_size,
            "target_num_frames": args.target_num_frames,
            "use_ground_truth_frame_count": args.use_ground_truth_frame_count,
            "fps": args.fps,
            "seed": args.seed,
            "sample_seed": args.sample_seed,
            "run_evalkit": args.run_evalkit,
            "task_specific_only": args.task_specific_only,
        },
    )
    return run


def main():
    args = parse_args()
    if args.run_evalkit and not args.evalkit_path.exists():
        raise FileNotFoundError(
            f"EvalKit path does not exist: {args.evalkit_path}. "
            "Pass --no-run-evalkit to skip numeric evaluation."
        )
    accelerator_handle = accelerate.Accelerator()
    wandb_run = init_wandb(args) if accelerator_handle.is_main_process else None
    runner = Wan22TI2V5BVBVRBenchRunner(
        model_dir=args.model_dir,
        bench_root=args.bench_root,
        fps=args.fps,
        device=str(accelerator_handle.device),
        target_num_frames=args.target_num_frames,
        use_ground_truth_frame_count=args.use_ground_truth_frame_count,
    )

    if args.overwrite and accelerator_handle.is_main_process:
        runner.reset_generation_output(args.output_root)
    accelerator_handle.wait_for_everyone()

    runner.generate(
        output_root=args.output_root,
        checkpoint_path=args.dit_checkpoint,
        splits=args.splits,
        task_names=args.task_names,
        max_videos=args.max_videos,
        videos_per_task=args.videos_per_task,
        overwrite=args.overwrite,
        seed=args.seed,
        sample_seed=args.sample_seed,
        num_inference_steps=args.num_inference_steps,
        finetuning_mode=args.finetuning_mode,
        sigma_shift=args.sigma_shift,
        inference_schedule=args.inference_schedule,
        schedule_sd3_r=args.schedule_sd3_r,
        schedule_c_interp=args.schedule_c_interp,
        schedule_c_start=args.schedule_c_start,
        schedule_c_t_end=args.schedule_c_t_end,
        schedule_c_grid_size=args.schedule_c_grid_size,
        wandb_run=wandb_run,
        wandb_step=0,
        wandb_prefix="vbvr_bench",
        process_index=accelerator_handle.process_index,
        num_processes=accelerator_handle.num_processes,
    )

    accelerator_handle.wait_for_everyone()

    if accelerator_handle.is_main_process:
        generation = runner.finalize_generation(
            output_root=args.output_root,
            num_processes=accelerator_handle.num_processes,
        )
        print(f"Generated videos written to {generation['output_root']}")
        print(f"Manifest: {generation['manifest_path']}")

        if wandb_run is not None:
            wandb_run.save(generation["manifest_path"], base_path=str(args.output_root))

        if args.run_evalkit:
            eval_output_dir = args.eval_output_dir or (args.output_root / "evalkit_results")
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            eval_name = args.eval_name or (args.wandb_run_name or args.output_root.name)
            results = runner.run_evalkit(
                videos_path=args.output_root,
                evalkit_path=args.evalkit_path,
                eval_output_path=eval_output_dir,
                name=eval_name,
                task_specific_only=args.task_specific_only,
            )
            print(f"EvalKit overall mean score: {results['overall']['mean_score']:.4f}")
            log_eval_summary_to_wandb(wandb_run, results, 0, prefix="vbvr_bench/eval")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
