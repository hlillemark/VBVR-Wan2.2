import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps

from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline, normalize_finetuning_mode
from diffsynth.utils.data import VideoData, save_video


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

DEFAULT_SPLITS = ("In-Domain_50", "Out-of-Domain_50")
LANCZOS_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
CHECKPOINT_METADATA_SUFFIX = ".metadata.json"


@dataclass
class VBVRSample:
    split: str
    task_name: str
    sample_id: str
    sample_dir: Path
    first_frame_path: Path
    ground_truth_path: Path
    prompt_path: Path
    prompt: str


def parse_csv_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [item for item in value if item]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def default_inference_config_for_mode(finetuning_mode):
    finetuning_mode = normalize_finetuning_mode(finetuning_mode)
    if finetuning_mode == "eqf":
        return {
            "inference_schedule": "c_function",
            "sigma_shift": 5.0,
            "schedule_sd3_r": 6.0,
            "schedule_c_interp": 0.8,
            "schedule_c_start": 1.0,
            "schedule_c_t_end": 0.999,
            "schedule_c_grid_size": 4096,
        }
    return {
        "inference_schedule": "sigma_shift",
        "sigma_shift": 5.0,
        "schedule_sd3_r": 6.0,
        "schedule_c_interp": 0.8,
        "schedule_c_start": 1.0,
        "schedule_c_t_end": 0.999,
        "schedule_c_grid_size": 4096,
    }


def checkpoint_metadata_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    candidate_paths = [checkpoint_path]
    try:
        candidate_paths.append(checkpoint_path.resolve(strict=True))
    except FileNotFoundError:
        pass
    for candidate in candidate_paths:
        metadata_path = candidate.with_name(f"{candidate.name}{CHECKPOINT_METADATA_SUFFIX}")
        if metadata_path.is_file():
            return metadata_path
    return checkpoint_path.with_name(f"{checkpoint_path.name}{CHECKPOINT_METADATA_SUFFIX}")


def load_checkpoint_metadata(checkpoint_path):
    if checkpoint_path is None:
        return None
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    if not metadata_path.is_file():
        return None
    return json.loads(metadata_path.read_text())


def resolve_finetuning_mode(checkpoint_path=None, finetuning_mode="auto"):
    if finetuning_mode != "auto":
        return normalize_finetuning_mode(finetuning_mode)
    metadata = load_checkpoint_metadata(checkpoint_path)
    if metadata is not None:
        return normalize_finetuning_mode(metadata.get("finetuning_mode", "flow"))
    return "flow"


def resolve_inference_config(
    checkpoint_path,
    finetuning_mode,
    inference_schedule="auto",
    sigma_shift=5.0,
    schedule_sd3_r=6.0,
    schedule_c_interp=0.8,
    schedule_c_start=1.0,
    schedule_c_t_end=0.999,
    schedule_c_grid_size=4096,
):
    config = {
        "inference_schedule": inference_schedule,
        "sigma_shift": sigma_shift,
        "schedule_sd3_r": schedule_sd3_r,
        "schedule_c_interp": schedule_c_interp,
        "schedule_c_start": schedule_c_start,
        "schedule_c_t_end": schedule_c_t_end,
        "schedule_c_grid_size": schedule_c_grid_size,
    }
    if inference_schedule == "auto":
        metadata = load_checkpoint_metadata(checkpoint_path)
        if metadata is not None:
            config["inference_schedule"] = metadata.get(
                "recommended_inference_schedule",
                default_inference_config_for_mode(finetuning_mode)["inference_schedule"],
            )
            config.update(metadata.get("recommended_inference_kwargs", {}))
        else:
            config.update(default_inference_config_for_mode(finetuning_mode))
    return config


def adjust_num_frames(
    raw_frame_count,
    target_num_frames=None,
    time_division_factor=4,
    time_division_remainder=1,
    use_ground_truth_frame_count=False,
):
    if raw_frame_count <= 0:
        return 1
    if use_ground_truth_frame_count or target_num_frames is None:
        num_frames = raw_frame_count
    else:
        num_frames = min(raw_frame_count, target_num_frames)
    while num_frames > 1 and num_frames % time_division_factor != time_division_remainder:
        num_frames -= 1
    return max(1, num_frames)


def get_frame_count(video_path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def collect_benchmark_samples(bench_root, splits=None, task_names=None):
    bench_root = Path(bench_root)
    splits = parse_csv_list(splits) or list(DEFAULT_SPLITS)
    task_name_filter = set(parse_csv_list(task_names) or [])
    samples = []

    for split in splits:
        split_dir = bench_root / split
        if not split_dir.is_dir():
            continue
        for task_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if task_name_filter and task_dir.name not in task_name_filter:
                continue
            for sample_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                prompt_path = sample_dir / "prompt.txt"
                first_frame_path = sample_dir / "first_frame.png"
                ground_truth_path = sample_dir / "ground_truth.mp4"
                if not (prompt_path.is_file() and first_frame_path.is_file() and ground_truth_path.is_file()):
                    continue
                samples.append(
                    VBVRSample(
                        split=split,
                        task_name=task_dir.name,
                        sample_id=sample_dir.name,
                        sample_dir=sample_dir,
                        first_frame_path=first_frame_path,
                        ground_truth_path=ground_truth_path,
                        prompt_path=prompt_path,
                        prompt=prompt_path.read_text().strip(),
                    )
                )
    return samples


def select_samples(samples, max_videos=None, videos_per_task=None, sample_seed=1234):
    if videos_per_task is None and max_videos is None:
        return samples

    rng = random.Random(sample_seed)
    grouped = {}
    for sample in samples:
        grouped.setdefault((sample.split, sample.task_name), []).append(sample)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda s: s.sample_id)
        rng.shuffle(grouped[key])

    selected = []
    if videos_per_task is not None:
        for key in sorted(grouped):
            selected.extend(grouped[key][:videos_per_task])
    else:
        keys = sorted(grouped)
        idx = 0
        while len(selected) < max_videos and keys:
            key = keys[idx % len(keys)]
            if grouped[key]:
                selected.append(grouped[key].pop(0))
            else:
                keys.remove(key)
                continue
            idx += 1

    if max_videos is not None:
        selected = selected[:max_videos]
    return sorted(selected, key=lambda s: (s.split, s.task_name, s.sample_id))


def shard_samples(samples, process_index=0, num_processes=1):
    if num_processes is None or num_processes <= 1:
        return list(samples)
    return list(samples)[process_index::num_processes]


def _wandb_log(run, payload, step):
    if run is None:
        return
    run.log(payload, step=step)


def _wandb_video(path, fps):
    import wandb

    return wandb.Video(str(path), fps=fps, format="mp4")


def _to_rgb_image(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    return Image.fromarray(frame).convert("RGB")


def _normalize_frame_to_size(frame, width, height):
    frame = _to_rgb_image(frame)
    if frame.size != (width, height):
        frame = ImageOps.fit(frame, (width, height), method=LANCZOS_RESAMPLE)
    return frame


def _build_side_by_side_comparison_video(
    generated_frames,
    ground_truth_path,
    save_path,
    fps,
    border_px=2,
    border_color=(255, 255, 0),
    background_color=(0, 0, 0),
    quality=5,
):
    generated_frames = list(generated_frames)
    if len(generated_frames) == 0:
        raise ValueError("Cannot build comparison video without generated frames.")

    first_generated_frame = _to_rgb_image(generated_frames[0])
    frame_width, frame_height = first_generated_frame.size
    generated_frames = [
        _normalize_frame_to_size(frame, frame_width, frame_height)
        for frame in generated_frames
    ]

    ground_truth_video = VideoData(
        video_file=str(ground_truth_path),
        height=frame_height,
        width=frame_width,
    )
    ground_truth_length = len(ground_truth_video)
    if ground_truth_length == 0:
        raise ValueError(f"Ground-truth video has no frames: {ground_truth_path}")

    ground_truth_last_frame = _normalize_frame_to_size(
        ground_truth_video[ground_truth_length - 1],
        frame_width,
        frame_height,
    )
    stacked_frames = []
    canvas_width = (frame_width * 2) + (border_px * 2)
    canvas_height = frame_height + (border_px * 2)
    right_x = frame_width

    for frame_index, generated_frame in enumerate(generated_frames):
        ground_truth_frame = (
            _normalize_frame_to_size(ground_truth_video[frame_index], frame_width, frame_height)
            if frame_index < ground_truth_length
            else ground_truth_last_frame
        )
        ground_truth_with_border = ImageOps.expand(
            ground_truth_frame,
            border=border_px,
            fill=border_color,
        )
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=background_color)
        canvas.paste(generated_frame, (0, border_px))
        canvas.paste(ground_truth_with_border, (right_x, 0))
        stacked_frames.append(canvas)

    save_video(stacked_frames, str(save_path), fps=fps, quality=quality)
    return save_path


def log_eval_summary_to_wandb(run, results, step, prefix="vbvr_eval"):
    if run is None or results is None:
        return
    summary = results.get("overall", {})
    payload = {
        f"{prefix}/overall_mean_score": summary.get("mean_score", 0.0),
        f"{prefix}/overall_num_videos": summary.get("num_videos", 0),
    }
    if "In_Domain" in results:
        payload[f"{prefix}/in_domain_mean_score"] = results["In_Domain"].get("mean_score", 0.0)
        payload[f"{prefix}/in_domain_num_videos"] = results["In_Domain"].get("num_videos", 0)
    if "Out_of_Domain" in results:
        payload[f"{prefix}/out_of_domain_mean_score"] = results["Out_of_Domain"].get("mean_score", 0.0)
        payload[f"{prefix}/out_of_domain_num_videos"] = results["Out_of_Domain"].get("num_videos", 0)
    for category, score in results.get("overall", {}).get("by_category", {}).items():
        key = category.lower().replace(" ", "_")
        payload[f"{prefix}/category/{key}"] = score
    _wandb_log(run, payload, step)


class Wan22TI2V5BVBVRBenchRunner:
    def __init__(
        self,
        model_dir,
        bench_root,
        fps=16,
        device="cuda",
        target_num_frames=209,
        use_ground_truth_frame_count=False,
        negative_prompt=NEGATIVE_PROMPT,
    ):
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        self.fps = fps
        self.device = device
        self.target_num_frames = target_num_frames
        self.use_ground_truth_frame_count = use_ground_truth_frame_count
        self.negative_prompt = negative_prompt
        self.pipe = None

    def validate_paths(self):
        required = [
            self.model_dir / "diffusion_pytorch_model-00001-of-00003.safetensors",
            self.model_dir / "diffusion_pytorch_model-00002-of-00003.safetensors",
            self.model_dir / "diffusion_pytorch_model-00003-of-00003.safetensors",
            self.model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            self.model_dir / "Wan2.2_VAE.pth",
            self.model_dir / "google" / "umt5-xxl",
            self.bench_root,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Missing required path: {path}")

    def build_pipeline(self):
        if self.pipe is not None:
            return self.pipe
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.device,
            model_configs=[
                ModelConfig(
                    path=[
                        str(self.model_dir / "diffusion_pytorch_model-00001-of-00003.safetensors"),
                        str(self.model_dir / "diffusion_pytorch_model-00002-of-00003.safetensors"),
                        str(self.model_dir / "diffusion_pytorch_model-00003-of-00003.safetensors"),
                    ]
                ),
                ModelConfig(path=str(self.model_dir / "models_t5_umt5-xxl-enc-bf16.pth")),
                ModelConfig(path=str(self.model_dir / "Wan2.2_VAE.pth")),
            ],
            tokenizer_config=ModelConfig(path=str(self.model_dir / "google" / "umt5-xxl")),
        )
        return self.pipe

    def apply_checkpoint(self, checkpoint_path=None, checkpoint_state_dict=None, finetuning_mode="auto"):
        pipe = self.build_pipeline()
        resolved_finetuning_mode = resolve_finetuning_mode(checkpoint_path=checkpoint_path, finetuning_mode=finetuning_mode)
        pipe.finetuning_mode = resolved_finetuning_mode
        if checkpoint_path is None and checkpoint_state_dict is None:
            return pipe, resolved_finetuning_mode
        if checkpoint_state_dict is None:
            checkpoint_state_dict = load_state_dict(str(checkpoint_path))
        pipe.dit.load_state_dict(checkpoint_state_dict, strict=True)
        return pipe, resolved_finetuning_mode

    @staticmethod
    def _sort_manifest_record(record):
        return (
            record.get("split", ""),
            record.get("task_name", ""),
            record.get("sample_id", ""),
            record.get("generated_path", ""),
        )

    @staticmethod
    def rank_manifest_name(process_index):
        return f"manifest-rank-{process_index:05d}.json"

    def reset_generation_output(self, output_root):
        output_root = Path(output_root)
        if not output_root.exists():
            return
        for video_path in output_root.rglob("*.mp4"):
            video_path.unlink()
        for manifest_path in output_root.glob("manifest*.json"):
            manifest_path.unlink()

    def finalize_generation(self, output_root, num_processes=1):
        output_root = Path(output_root)
        if num_processes <= 1:
            manifest_path = output_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else []
            return {
                "manifest": manifest,
                "manifest_path": str(manifest_path),
                "output_root": str(output_root),
                "num_videos": len(manifest),
                "num_processes": num_processes,
            }

        manifest = []
        for process_index in range(num_processes):
            rank_manifest_path = output_root / self.rank_manifest_name(process_index)
            if not rank_manifest_path.is_file():
                raise FileNotFoundError(f"Missing distributed manifest: {rank_manifest_path}")
            manifest.extend(json.loads(rank_manifest_path.read_text()))

        manifest = sorted(manifest, key=self._sort_manifest_record)
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return {
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "output_root": str(output_root),
            "num_videos": len(manifest),
            "num_processes": num_processes,
        }

    def generate(
        self,
        output_root,
        checkpoint_path=None,
        checkpoint_state_dict=None,
        splits=None,
        task_names=None,
        max_videos=None,
        videos_per_task=None,
        overwrite=False,
        seed=1,
        sample_seed=1234,
        wandb_run=None,
        wandb_step=None,
        wandb_prefix="vbvr_eval",
        log_ground_truth_video=True,
        process_index=0,
        num_processes=1,
        num_inference_steps=50,
        sigma_shift=5.0,
        inference_schedule="auto",
        schedule_sd3_r=6.0,
        schedule_c_interp=0.8,
        schedule_c_start=1.0,
        schedule_c_t_end=0.999,
        schedule_c_grid_size=4096,
        finetuning_mode="auto",
    ):
        self.validate_paths()
        pipe, resolved_finetuning_mode = self.apply_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_state_dict=checkpoint_state_dict,
            finetuning_mode=finetuning_mode,
        )
        inference_config = resolve_inference_config(
            checkpoint_path,
            resolved_finetuning_mode,
            inference_schedule=inference_schedule,
            sigma_shift=sigma_shift,
            schedule_sd3_r=schedule_sd3_r,
            schedule_c_interp=schedule_c_interp,
            schedule_c_start=schedule_c_start,
            schedule_c_t_end=schedule_c_t_end,
            schedule_c_grid_size=schedule_c_grid_size,
        )
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        comparison_root = None
        if wandb_run is not None and log_ground_truth_video:
            comparison_root = Path(tempfile.mkdtemp(prefix="vbvr-wandb-comparisons-"))

        samples = collect_benchmark_samples(self.bench_root, splits=splits, task_names=task_names)
        samples = select_samples(
            samples,
            max_videos=max_videos,
            videos_per_task=videos_per_task,
            sample_seed=sample_seed,
        )
        local_samples = shard_samples(samples, process_index=process_index, num_processes=num_processes)

        print(
            f"Selected {len(samples)} benchmark sample(s) total; "
            f"process {process_index}/{num_processes - 1 if num_processes > 0 else 0} "
            f"will generate {len(local_samples)} sample(s) on device {self.device}."
        )

        manifest = []
        for idx, sample in enumerate(local_samples):
            output_video_path = output_root / sample.split / sample.task_name / f"{sample.sample_id}.mp4"
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            if output_video_path.exists() and not overwrite:
                manifest.append(
                    {
                        "split": sample.split,
                        "task_name": sample.task_name,
                        "sample_id": sample.sample_id,
                        "prompt": sample.prompt,
                        "first_frame_path": str(sample.first_frame_path),
                        "ground_truth_path": str(sample.ground_truth_path),
                        "generated_path": str(output_video_path),
                        "skipped": True,
                        "process_index": process_index,
                    }
                )
                continue

            raw_frame_count = get_frame_count(sample.ground_truth_path)
            num_frames = adjust_num_frames(
                raw_frame_count=raw_frame_count,
                target_num_frames=self.target_num_frames,
                use_ground_truth_frame_count=self.use_ground_truth_frame_count,
            )
            input_image = Image.open(sample.first_frame_path).convert("RGB")
            sample_seed_value = seed + idx

            print(
                f"Generating {sample.split}/{sample.task_name}/{sample.sample_id} "
                f"with {num_frames} frame(s), seed={sample_seed_value}"
            )
            started_at = time.time()
            video = pipe(
                prompt=sample.prompt,
                negative_prompt=self.negative_prompt,
                input_image=input_image,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                sigma_shift=inference_config["sigma_shift"],
                inference_schedule=inference_config["inference_schedule"],
                schedule_sd3_r=inference_config["schedule_sd3_r"],
                schedule_c_interp=inference_config["schedule_c_interp"],
                schedule_c_start=inference_config["schedule_c_start"],
                schedule_c_t_end=inference_config["schedule_c_t_end"],
                schedule_c_grid_size=inference_config["schedule_c_grid_size"],
                finetuning_mode=resolved_finetuning_mode,
                seed=sample_seed_value,
                tiled=True,
                height=input_image.height,
                width=input_image.width,
            )
            video = list(video)
            generation_seconds = time.time() - started_at
            save_video(video, str(output_video_path), fps=self.fps, quality=5)

            record = {
                "split": sample.split,
                "task_name": sample.task_name,
                "sample_id": sample.sample_id,
                "prompt": sample.prompt,
                "first_frame_path": str(sample.first_frame_path),
                "ground_truth_path": str(sample.ground_truth_path),
                "generated_path": str(output_video_path),
                "num_frames": num_frames,
                "raw_ground_truth_frame_count": raw_frame_count,
                "height": input_image.height,
                "width": input_image.width,
                "seed": sample_seed_value,
                "generation_seconds": generation_seconds,
                "num_inference_steps": num_inference_steps,
                "sigma_shift": inference_config["sigma_shift"],
                "inference_schedule": inference_config["inference_schedule"],
                "finetuning_mode": resolved_finetuning_mode,
                "process_index": process_index,
            }
            manifest.append(record)

            if wandb_run is not None:
                sample_key = f"{sample.split}/{sample.task_name}/{sample.sample_id}"
                payload_video_path = output_video_path
                if log_ground_truth_video:
                    comparison_video_path = comparison_root / sample.split / sample.task_name / f"{sample.sample_id}.mp4"
                    comparison_video_path.parent.mkdir(parents=True, exist_ok=True)
                    payload_video_path = _build_side_by_side_comparison_video(
                        generated_frames=video,
                        ground_truth_path=sample.ground_truth_path,
                        save_path=comparison_video_path,
                        fps=self.fps,
                    )
                payload = {
                    f"{wandb_prefix}/videos/{sample_key}/generated": _wandb_video(payload_video_path, self.fps),
                    f"{wandb_prefix}/metrics/{sample_key}/num_frames": num_frames,
                    f"{wandb_prefix}/metrics/{sample_key}/generation_seconds": generation_seconds,
                    f"{wandb_prefix}/metrics/{sample_key}/seed": sample_seed_value,
                    f"{wandb_prefix}/text/{sample_key}/prompt": sample.prompt,
                }
                _wandb_log(wandb_run, payload, wandb_step)

        manifest_path = output_root / (
            self.rank_manifest_name(process_index) if num_processes > 1 else "manifest.json"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return {
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "output_root": str(output_root),
            "num_selected_samples": len(samples),
            "num_local_samples": len(local_samples),
            "process_index": process_index,
            "num_processes": num_processes,
        }

    def run_evalkit(
        self,
        videos_path,
        evalkit_path,
        eval_output_path,
        name,
        task_specific_only=True,
        split=None,
        task_list=None,
        device=None,
        save_detailed=True,
    ):
        evalkit_path = Path(evalkit_path).resolve()
        if str(evalkit_path) not in sys.path:
            sys.path.insert(0, str(evalkit_path))
        from vbvr_bench import VBVRBench

        bench = VBVRBench(
            gt_base_path=str(self.bench_root),
            output_path=str(eval_output_path),
            device=device or self.device,
        )
        return bench.evaluate(
            videos_path=str(videos_path),
            name=name,
            task_list=parse_csv_list(task_list),
            split=split,
            save_detailed=save_detailed,
            task_specific_only=task_specific_only,
        )


class TrainingVBVREvalHook:
    def __init__(
        self,
        model_dir,
        bench_root,
        evalkit_path=None,
        eval_steps=None,
        max_videos=None,
        videos_per_task=1,
        splits=None,
        task_names=None,
        target_num_frames=209,
        use_ground_truth_frame_count=False,
        run_numeric_eval=False,
        task_specific_only=True,
        fps=16,
        seed=1,
        sample_seed=1234,
        finetuning_mode="flow",
        inference_schedule="auto",
        sigma_shift=5.0,
        schedule_sd3_r=6.0,
        schedule_c_interp=0.8,
        schedule_c_start=1.0,
        schedule_c_t_end=0.999,
        schedule_c_grid_size=4096,
    ):
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        self.evalkit_path = None if evalkit_path is None else Path(evalkit_path)
        self.eval_steps = eval_steps
        self.max_videos = max_videos
        self.videos_per_task = videos_per_task
        self.splits = parse_csv_list(splits)
        self.task_names = parse_csv_list(task_names)
        self.target_num_frames = target_num_frames
        self.use_ground_truth_frame_count = use_ground_truth_frame_count
        self.run_numeric_eval = run_numeric_eval
        self.task_specific_only = task_specific_only
        self.fps = fps
        self.seed = seed
        self.sample_seed = sample_seed
        self.finetuning_mode = finetuning_mode
        self.inference_schedule = inference_schedule
        self.sigma_shift = sigma_shift
        self.schedule_sd3_r = schedule_sd3_r
        self.schedule_c_interp = schedule_c_interp
        self.schedule_c_start = schedule_c_start
        self.schedule_c_t_end = schedule_c_t_end
        self.schedule_c_grid_size = schedule_c_grid_size
        self.runner = None

    def should_run(self, step):
        return self.eval_steps is not None and self.eval_steps > 0 and step % self.eval_steps == 0

    def get_runner(self, device=None):
        if self.runner is None:
            self.runner = Wan22TI2V5BVBVRBenchRunner(
                model_dir=self.model_dir,
                bench_root=self.bench_root,
                fps=self.fps,
                device=device or "cuda",
                target_num_frames=self.target_num_frames,
                use_ground_truth_frame_count=self.use_ground_truth_frame_count,
            )
        return self.runner

    def run(self, accelerator, model, model_logger, step):
        if not self.should_run(step):
            return None

        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        exported_state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
            state_dict,
            remove_prefix=model_logger.remove_prefix_in_ckpt,
        )
        exported_state_dict = model_logger.state_dict_converter(exported_state_dict)

        runner = self.get_runner(device=str(accelerator.device))
        eval_root = Path(model_logger.output_path) / "vbvr_eval" / f"step-{step}"
        videos_root = eval_root / "videos"
        if accelerator.is_main_process:
            runner.reset_generation_output(videos_root)
        accelerator.wait_for_everyone()
        runner.generate(
            output_root=eval_root / "videos",
            checkpoint_state_dict=exported_state_dict,
            splits=self.splits,
            task_names=self.task_names,
            max_videos=self.max_videos,
            videos_per_task=self.videos_per_task,
            overwrite=True,
            seed=self.seed,
            sample_seed=self.sample_seed,
            wandb_run=model_logger.wandb_run if accelerator.is_main_process else None,
            wandb_step=step,
            wandb_prefix="vbvr_eval",
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
            finetuning_mode=self.finetuning_mode,
            inference_schedule=self.inference_schedule,
            sigma_shift=self.sigma_shift,
            schedule_sd3_r=self.schedule_sd3_r,
            schedule_c_interp=self.schedule_c_interp,
            schedule_c_start=self.schedule_c_start,
            schedule_c_t_end=self.schedule_c_t_end,
            schedule_c_grid_size=self.schedule_c_grid_size,
        )

        accelerator.wait_for_everyone()

        results_path = eval_root / "results.json"
        if accelerator.is_main_process:
            inference_result = runner.finalize_generation(
                output_root=videos_root,
                num_processes=accelerator.num_processes,
            )
            results = {
                "inference": inference_result,
                "evaluation": None,
            }
            if self.run_numeric_eval:
                if self.evalkit_path is None:
                    raise ValueError("evalkit_path is required when run_numeric_eval=True")
                eval_output = eval_root / "evalkit_results"
                evaluation = runner.run_evalkit(
                    videos_path=videos_root,
                    evalkit_path=self.evalkit_path,
                    eval_output_path=eval_output,
                    name=f"step-{step}",
                    task_specific_only=self.task_specific_only,
                )
                log_eval_summary_to_wandb(model_logger.wandb_run, evaluation, step, prefix="vbvr_eval")
                results["evaluation"] = evaluation
            results_path.write_text(json.dumps(results, indent=2))

        accelerator.wait_for_everyone()
        if not results_path.is_file():
            raise FileNotFoundError(f"Missing distributed eval results file: {results_path}")
        return json.loads(results_path.read_text())
