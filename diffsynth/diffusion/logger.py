import math
import os
import torch
from accelerate import Accelerator


def _to_cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_state(item) for item in value)
    return value


class ModelLogger:
    ALIAS_FILE_NAMES = {"latest.safetensors", "best.safetensors"}

    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        evaluation_callback=None,
        forced_save_steps=None,
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.evaluation_callback = evaluation_callback
        self.forced_save_steps = self.parse_step_set(forced_save_steps)
        self.num_steps = 0
        self.last_archive_step = None
        self.best_loss = None
        self.latest_checkpoint_file = None
        self.best_checkpoint_file = None
        self.force_checkpoint_files = set()
        self.wandb_run = None
        self.wandb_import_failed = False

    @staticmethod
    def parse_step_set(step_values):
        if step_values is None:
            return set()
        if isinstance(step_values, (set, list, tuple)):
            return {int(step) for step in step_values}
        return {
            int(step.strip())
            for step in str(step_values).split(",")
            if step.strip() != ""
        }

    @staticmethod
    def _normalize_loss(loss):
        if loss is None:
            return None
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().float().item()
        return float(loss)

    @staticmethod
    def _synchronize_loss(accelerator: Accelerator, loss):
        loss = ModelLogger._normalize_loss(loss)
        if loss is None:
            return None
        loss_tensor = torch.tensor(loss, device=accelerator.device, dtype=torch.float32)
        return accelerator.reduce(loss_tensor, reduction="mean").item()

    @staticmethod
    def _format_loss_for_file(loss):
        if loss is None:
            return "na"
        if math.isnan(loss):
            return "nan"
        if math.isinf(loss):
            return "posinf" if loss > 0 else "neginf"
        return f"{loss:.6f}".replace("-", "neg")

    def _checkpoint_file_name(self, loss, forced=False):
        force_suffix = "-force" if forced else ""
        return f"step-{self.num_steps:08d}-loss-{self._format_loss_for_file(loss)}{force_suffix}.safetensors"

    def _latest_training_state_path(self):
        return os.path.join(self.output_path, "latest-training-state.pt")

    def state_dict(self):
        return {
            "num_steps": self.num_steps,
            "last_archive_step": self.last_archive_step,
            "best_loss": self.best_loss,
            "latest_checkpoint_file": self.latest_checkpoint_file,
            "best_checkpoint_file": self.best_checkpoint_file,
            "force_checkpoint_files": sorted(self.force_checkpoint_files),
        }

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        self.num_steps = state_dict.get("num_steps", self.num_steps)
        self.last_archive_step = state_dict.get("last_archive_step", self.last_archive_step)
        self.best_loss = state_dict.get("best_loss", self.best_loss)
        self.latest_checkpoint_file = state_dict.get("latest_checkpoint_file", self.latest_checkpoint_file)
        self.best_checkpoint_file = state_dict.get("best_checkpoint_file", self.best_checkpoint_file)
        self.force_checkpoint_files = set(state_dict.get("force_checkpoint_files", self.force_checkpoint_files))
        legacy_best_metric = state_dict.get("best_metric")
        if self.best_loss is None and isinstance(legacy_best_metric, dict):
            legacy_loss = legacy_best_metric.get("value")
            if legacy_best_metric.get("source") == "loss" and legacy_loss is not None:
                self.best_loss = float(legacy_loss)

    def init_wandb(self, accelerator: Accelerator):
        if self.wandb_run is not None or self.wandb_import_failed or not accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError:
            print("wandb is not installed; skipping wandb logging.")
            self.wandb_import_failed = True
            return
        run_name = os.path.basename(os.path.normpath(self.output_path)) or "training"
        self.wandb_run = wandb.init(
            entity="eqforcing",
            project="eqf",
            name=run_name,
            config={"output_path": self.output_path},
            reinit=True,
        )

    def _build_training_state_payload(self, training_state_fn, model_file_name):
        if training_state_fn is None:
            return None
        payload = dict(training_state_fn())
        payload["model_file_name"] = model_file_name
        return _to_cpu_state(payload)

    def _save_latest_training_state(self, training_state_fn, model_file_name):
        training_state_payload = self._build_training_state_payload(training_state_fn, model_file_name)
        if training_state_payload is None:
            return
        target_path = self._latest_training_state_path()
        temp_path = f"{target_path}.tmp"
        if os.path.lexists(temp_path):
            os.remove(temp_path)
        torch.save(training_state_payload, temp_path)
        os.replace(temp_path, target_path)

    def _update_symlink(self, alias_name, target_file_name):
        alias_path = os.path.join(self.output_path, alias_name)
        if target_file_name is None:
            if os.path.lexists(alias_path):
                os.remove(alias_path)
            return
        temp_path = f"{alias_path}.tmp"
        if os.path.lexists(temp_path):
            os.remove(temp_path)
        if os.path.lexists(alias_path):
            os.remove(alias_path)
        os.symlink(target_file_name, temp_path)
        os.replace(temp_path, alias_path)

    def _is_checkpoint_file(self, file_name):
        return file_name.endswith(".safetensors") and file_name not in self.ALIAS_FILE_NAMES

    def _is_forced_checkpoint_file(self, file_name):
        return self._is_checkpoint_file(file_name) and "-force.safetensors" in file_name

    def _cleanup_output_dir(self):
        if not os.path.isdir(self.output_path):
            return
        self.force_checkpoint_files.update(
            file_name
            for file_name in os.listdir(self.output_path)
            if self._is_forced_checkpoint_file(file_name)
        )
        kept_checkpoint_files = set(self.force_checkpoint_files)
        if self.latest_checkpoint_file is not None:
            kept_checkpoint_files.add(self.latest_checkpoint_file)
        if self.best_checkpoint_file is not None:
            kept_checkpoint_files.add(self.best_checkpoint_file)
        for file_name in os.listdir(self.output_path):
            file_path = os.path.join(self.output_path, file_name)
            if file_name in self.ALIAS_FILE_NAMES or file_name == "latest-training-state.pt":
                continue
            if file_name.endswith("-training-state.pt"):
                if os.path.lexists(file_path):
                    os.remove(file_path)
                continue
            if not self._is_checkpoint_file(file_name):
                continue
            if file_name in kept_checkpoint_files:
                continue
            if os.path.lexists(file_path):
                os.remove(file_path)
        existing_checkpoint_files = {
            file_name
            for file_name in os.listdir(self.output_path)
            if self._is_checkpoint_file(file_name)
        }
        self.force_checkpoint_files.intersection_update(existing_checkpoint_files)

    def save_checkpoint(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        loss=None,
        update_latest=False,
        update_best=False,
        training_state_fn=None,
        forced=False,
    ):
        if not update_latest and not update_best:
            return
        checkpoint_file_name = self._checkpoint_file_name(loss, forced=forced)

        if forced:
            self.force_checkpoint_files.add(checkpoint_file_name)
        if update_latest:
            self.latest_checkpoint_file = checkpoint_file_name
            self.last_archive_step = self.num_steps
        if update_best:
            self.best_checkpoint_file = checkpoint_file_name
            self.best_loss = loss

        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if not accelerator.is_main_process:
            return

        exported_state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
            state_dict,
            remove_prefix=self.remove_prefix_in_ckpt,
        )
        exported_state_dict = self.state_dict_converter(exported_state_dict)
        os.makedirs(self.output_path, exist_ok=True)

        checkpoint_path = os.path.join(self.output_path, checkpoint_file_name)
        accelerator.save(exported_state_dict, checkpoint_path, safe_serialization=True)

        self._update_symlink("latest.safetensors", self.latest_checkpoint_file)
        self._update_symlink("best.safetensors", self.best_checkpoint_file)
        if update_latest:
            self._save_latest_training_state(training_state_fn, checkpoint_file_name)
        self._cleanup_output_dir()

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, training_state_fn=None, **kwargs):
        self.num_steps += 1
        self.init_wandb(accelerator)

        loss = self._synchronize_loss(accelerator, kwargs.get("loss"))
        if loss is not None and self.wandb_run is not None:
            self.wandb_run.log({"loss": loss, "step": self.num_steps}, step=self.num_steps)

        if self.evaluation_callback is not None:
            self.evaluation_callback.run(accelerator, model, self, self.num_steps)

        forced_save = self.num_steps in self.forced_save_steps
        should_save_latest = (
            forced_save
            or (save_steps is not None and save_steps > 0 and self.num_steps % save_steps == 0)
        )
        should_save_best = loss is not None and (
            self.best_loss is None or loss < self.best_loss
        )
        self.save_checkpoint(
            accelerator,
            model,
            loss=loss,
            update_latest=should_save_latest,
            update_best=should_save_best,
            training_state_fn=training_state_fn,
            forced=forced_save,
        )

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id, loss=None, training_state_fn=None):
        del epoch_id
        loss = self._synchronize_loss(accelerator, loss)
        should_save_best = loss is not None and (
            self.best_loss is None or loss < self.best_loss
        )
        self.save_checkpoint(
            accelerator,
            model,
            loss=loss,
            update_latest=True,
            update_best=should_save_best,
            training_state_fn=training_state_fn,
            forced=False,
        )

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, loss=None, training_state_fn=None):
        should_save_final_step = (
            save_steps is not None
            and self.num_steps > 0
            and self.last_archive_step != self.num_steps
        )
        if should_save_final_step:
            loss = self._synchronize_loss(accelerator, loss)
            should_save_best = loss is not None and (
                self.best_loss is None or loss < self.best_loss
            )
            self.save_checkpoint(
                accelerator,
                model,
                loss=loss,
                update_latest=True,
                update_best=should_save_best,
                training_state_fn=training_state_fn,
                forced=False,
            )
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None
