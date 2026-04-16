import math
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from ..core import load_state_dict
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def collate_data(batch):
    if len(batch) == 1:
        return batch[0]
    sample = batch[0]
    if isinstance(sample, dict):
        return {key: collate_data([item[key] for item in batch]) for key in sample}
    return batch


def build_dataloader_kwargs(args, num_workers):
    kwargs = {
        "num_workers": num_workers,
    }
    if args is not None:
        kwargs["pin_memory"] = args.dataloader_pin_memory
        if num_workers > 0:
            kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
            kwargs["persistent_workers"] = args.dataloader_persistent_workers
    return kwargs


def set_training_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def resolve_training_state_path(args, output_path):
    explicit_path = None if args is None else args.resume_from_training_state
    if explicit_path is not None:
        return explicit_path
    if args is not None and args.no_resume_training_state:
        return None
    auto_path = os.path.join(output_path, "latest-training-state.pt")
    if os.path.exists(auto_path):
        return auto_path
    return None


def infer_checkpoint_path_from_training_state(training_state_path, training_state):
    model_file_name = training_state.get("model_file_name")
    if model_file_name is not None:
        return os.path.join(os.path.dirname(training_state_path), model_file_name)
    file_name = os.path.basename(training_state_path)
    if file_name.endswith("-training-state.pt"):
        model_file_name = file_name[:-len("-training-state.pt")] + ".safetensors"
        return os.path.join(os.path.dirname(training_state_path), model_file_name)
    raise ValueError(f"Unable to infer checkpoint path from training state: {training_state_path}")


def load_training_state(training_state_path):
    try:
        return torch.load(training_state_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(training_state_path, map_location="cpu")


def resolve_finetuning_mode(training_state, fallback="flow"):
    if training_state is None:
        return fallback
    logger_state = training_state.get("logger", {})
    return str(
        training_state.get(
            "finetuning_mode",
            logger_state.get("finetuning_mode", fallback),
        )
    )


def resolve_global_sample_in_epoch(training_state, fallback_num_replicas=1):
    """Return the number of global samples consumed in the current epoch.

    Preferentially reads ``global_sample_in_epoch`` from the saved training
    state. For older states that only stored per-rank cursors, the value is
    reconstructed from ``batch_in_epoch``/``sample_in_epoch`` using the saved
    ``num_replicas`` when available, or ``fallback_num_replicas`` otherwise.
    """
    if training_state is None:
        return 0
    if "global_sample_in_epoch" in training_state:
        return int(training_state.get("global_sample_in_epoch", 0))
    saved_num_replicas = int(training_state.get("num_replicas", fallback_num_replicas))
    batch_size = int(training_state.get("batch_size", 1))
    if "sample_in_epoch" in training_state:
        return int(training_state.get("sample_in_epoch", 0)) * saved_num_replicas
    batch_in_epoch = int(training_state.get("batch_in_epoch", 0))
    return batch_in_epoch * batch_size * saved_num_replicas


class DeterministicDistributedSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        num_replicas=1,
        rank=0,
        seed=1234,
        epoch=0,
        batch_size=1,
        start_sample_global=0,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = epoch
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) / float(self.num_replicas)))
        self.total_size = self.num_samples * self.num_replicas
        self.start_sample_global = max(0, min(int(start_sample_global), self.total_size))

    def _ordered_indices(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        if self.start_sample_global > 0:
            indices = indices[self.start_sample_global:]
        # Trim so every rank receives the same number of samples, which keeps
        # DDP collectives in sync even after a mid-epoch global skip.
        trimmed_length = (len(indices) // self.num_replicas) * self.num_replicas
        indices = indices[:trimmed_length]
        indices = indices[self.rank::self.num_replicas]
        return indices

    def __iter__(self):
        return iter(self._ordered_indices())

    def __len__(self):
        return len(self._ordered_indices())


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        max_steps = args.max_steps
        batch_size = args.batch_size
        training_seed = args.training_seed
        dataloader_seed = args.training_seed if args.dataloader_seed is None else args.dataloader_seed
        deterministic_dataloader = args.deterministic_dataloader
    else:
        batch_size = 1
        max_steps = None
        training_seed = 1234
        dataloader_seed = 1234
        deterministic_dataloader = True

    set_training_seed(training_seed)

    resume_state = None
    resume_state_path = resolve_training_state_path(args, model_logger.output_path)
    if resume_state_path is not None:
        if not os.path.exists(resume_state_path):
            raise FileNotFoundError(f"Training state not found: {resume_state_path}")
        if not deterministic_dataloader:
            raise ValueError("Resuming training requires deterministic dataloader ordering. Leave deterministic mode enabled.")
        resume_state = load_training_state(resume_state_path)
        checkpoint_path = infer_checkpoint_path_from_training_state(resume_state_path, resume_state)
        checkpoint_state = load_state_dict(checkpoint_path, device="cpu")
        model.load_trainable_state_dict(
            checkpoint_state,
            remove_prefix=resume_state.get("remove_prefix_in_ckpt", model_logger.remove_prefix_in_ckpt),
            strict=True,
        )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader_kwargs = build_dataloader_kwargs(args, num_workers)
    model.to(device=accelerator.device)
    if deterministic_dataloader:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_data,
            **dataloader_kwargs,
        )
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        model_logger.load_state_dict(resume_state.get("logger", {}))
        explicit_wandb_resume_id = None if args is None else getattr(args, "resume_wandb_run", None)
        if explicit_wandb_resume_id is not None:
            model_logger.wandb_resume_id = explicit_wandb_resume_id
        resume_finetuning_mode = resolve_finetuning_mode(resume_state, fallback=model_logger.finetuning_mode)
        model_finetuning_mode = getattr(model, "finetuning_mode", resume_finetuning_mode)
        if model_finetuning_mode != resume_finetuning_mode:
            raise ValueError(
                f"Resume finetuning_mode mismatch: checkpoint uses '{resume_finetuning_mode}' "
                f"but current run requested '{model_finetuning_mode}'."
            )
        if hasattr(model, "set_finetuning_mode"):
            model.set_finetuning_mode(resume_finetuning_mode)
        model_logger.finetuning_mode = resume_finetuning_mode
        restore_rng_state(resume_state.get("rng"))

    initialize_deepspeed_gradient_checkpointing(accelerator)

    num_replicas = max(1, int(getattr(accelerator, "num_processes", 1)))
    gradient_accumulation_steps = max(1, int(getattr(accelerator, "gradient_accumulation_steps", 1)))

    next_epoch = 0
    next_global_sample_in_epoch = 0
    if resume_state is not None:
        next_epoch = int(resume_state.get("epoch", 0))
        next_global_sample_in_epoch = resolve_global_sample_in_epoch(
            resume_state, fallback_num_replicas=num_replicas
        )
        saved_num_replicas = resume_state.get("num_replicas")
        if (
            saved_num_replicas is not None
            and int(saved_num_replicas) != num_replicas
            and accelerator.is_main_process
        ):
            print(
                f"[runner] Resuming with num_replicas={num_replicas} (was {int(saved_num_replicas)}). "
                "Global sample cursor will be preserved across the device-count change."
            )
        saved_grad_accum = resume_state.get("gradient_accumulation_steps")
        if (
            saved_grad_accum is not None
            and int(saved_grad_accum) != gradient_accumulation_steps
            and accelerator.is_main_process
        ):
            print(
                f"[runner] Resuming with gradient_accumulation_steps={gradient_accumulation_steps} "
                f"(was {int(saved_grad_accum)}). Optimizer step count continues from the saved value."
            )

    training_progress = {
        "epoch": next_epoch,
        "global_sample_in_epoch": next_global_sample_in_epoch,
        "training_seed": training_seed,
        "dataloader_seed": dataloader_seed,
        "deterministic_dataloader": deterministic_dataloader,
        "batch_size": batch_size,
        "num_replicas": num_replicas,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs": num_epochs,
        "max_steps": max_steps,
    }

    def build_training_state():
        return {
            "format_version": 3,
            "epoch": training_progress["epoch"],
            "global_sample_in_epoch": training_progress["global_sample_in_epoch"],
            "training_seed": training_progress["training_seed"],
            "dataloader_seed": training_progress["dataloader_seed"],
            "deterministic_dataloader": training_progress["deterministic_dataloader"],
            "batch_size": training_progress["batch_size"],
            "num_replicas": training_progress["num_replicas"],
            "gradient_accumulation_steps": training_progress["gradient_accumulation_steps"],
            "num_epochs": training_progress["num_epochs"],
            "max_steps": training_progress["max_steps"],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "logger": model_logger.state_dict(),
            "rng": capture_rng_state(),
            "remove_prefix_in_ckpt": model_logger.remove_prefix_in_ckpt,
            "finetuning_mode": model_logger.finetuning_mode,
            "recommended_inference_schedule": model_logger.recommended_inference_schedule,
            "recommended_inference_kwargs": dict(model_logger.recommended_inference_kwargs),
        }

    last_loss = None
    reached_max_steps = max_steps is not None and model_logger.num_steps >= max_steps
    for epoch_id in range(next_epoch, num_epochs):
        if reached_max_steps:
            break
        sample_start_global = next_global_sample_in_epoch if epoch_id == next_epoch else 0
        if deterministic_dataloader:
            sampler = DeterministicDistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=accelerator.process_index,
                seed=dataloader_seed,
                epoch=epoch_id,
                batch_size=batch_size,
                start_sample_global=sample_start_global,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                shuffle=False,
                batch_size=batch_size,
                collate_fn=collate_data,
                **dataloader_kwargs,
            )
            total_samples_this_epoch = sampler.total_size
        else:
            total_samples_this_epoch = None
        batches_this_epoch = len(dataloader)
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch_offset, data in enumerate(progress_bar):
            if max_steps is not None and model_logger.num_steps >= max_steps:
                reached_max_steps = True
                break
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                last_loss = loss.detach()
                if accelerator.sync_gradients:
                    samples_consumed_this_epoch = (
                        sample_start_global + (batch_offset + 1) * batch_size * num_replicas
                    )
                    if total_samples_this_epoch is not None:
                        samples_consumed_this_epoch = min(
                            samples_consumed_this_epoch, total_samples_this_epoch
                        )
                    is_last_batch_of_epoch = batch_offset + 1 >= batches_this_epoch
                    if is_last_batch_of_epoch:
                        training_progress["epoch"] = epoch_id + 1
                        training_progress["global_sample_in_epoch"] = 0
                    else:
                        training_progress["epoch"] = epoch_id
                        training_progress["global_sample_in_epoch"] = samples_consumed_this_epoch
                    model_logger.on_step_end(
                        accelerator,
                        model,
                        save_steps,
                        loss=loss,
                        training_state_fn=build_training_state,
                    )
                    if max_steps is not None and model_logger.num_steps >= max_steps:
                        reached_max_steps = True
                        break
        if reached_max_steps:
            break
        if save_steps is None:
            model_logger.on_epoch_end(
                accelerator,
                model,
                epoch_id,
                loss=last_loss,
                training_state_fn=build_training_state,
            )
        next_global_sample_in_epoch = 0
    model_logger.on_training_end(
        accelerator,
        model,
        save_steps,
        loss=last_loss,
        training_state_fn=build_training_state,
    )


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader_kwargs = build_dataloader_kwargs(args, num_workers)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=lambda x: x[0],
        **dataloader_kwargs,
    )
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
