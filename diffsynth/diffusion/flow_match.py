import math
import torch
from typing_extensions import Literal
from .inference_schedules import InferenceScheduleResolver


class FlowMatchScheduler():

    def __init__(self, template: Literal["FLUX.1", "Wan", "Qwen-Image", "FLUX.2", "Z-Image", "LTX-2", "Qwen-Image-Lightning"] = "FLUX.1"):
        self.set_timesteps_fn = {
            "FLUX.1": FlowMatchScheduler.set_timesteps_flux,
            "Wan": FlowMatchScheduler.set_timesteps_wan,
            "Qwen-Image": FlowMatchScheduler.set_timesteps_qwen_image,
            "FLUX.2": FlowMatchScheduler.set_timesteps_flux2,
            "Z-Image": FlowMatchScheduler.set_timesteps_z_image,
            "LTX-2": FlowMatchScheduler.set_timesteps_ltx2,
            "Qwen-Image-Lightning": FlowMatchScheduler.set_timesteps_qwen_image_lightning,
        }.get(template, FlowMatchScheduler.set_timesteps_flux)
        self.num_train_timesteps = 1000
        self.schedule_name = None
        self.solver_sigmas = None
        self.solver_timesteps = None
        self.step_sizes = None
        self.step_scales = None

    @staticmethod
    def _format_schedule_metadata(sigmas, timesteps, solver_sigmas=None, step_sizes=None, step_scales=None, schedule_name=None):
        sigmas = sigmas.to(dtype=torch.float32)
        timesteps = timesteps.to(dtype=torch.float32)
        solver_sigmas = sigmas.clone() if solver_sigmas is None else solver_sigmas.to(dtype=torch.float32)
        solver_timesteps = solver_sigmas * 1000

        if step_sizes is None:
            next_sigmas = torch.cat([sigmas[1:], torch.zeros(1, dtype=sigmas.dtype)])
            step_sizes = next_sigmas - sigmas
        else:
            step_sizes = step_sizes.to(dtype=torch.float32)

        if step_scales is None:
            next_solver_sigmas = torch.cat([solver_sigmas[1:], torch.zeros(1, dtype=solver_sigmas.dtype)])
            solver_step_sizes = next_solver_sigmas - solver_sigmas
            denom = torch.sign(solver_step_sizes) * torch.clamp(
                solver_step_sizes.abs(),
                min=torch.finfo(solver_step_sizes.dtype).eps,
            )
            step_scales = step_sizes / denom
        else:
            step_scales = step_scales.to(dtype=torch.float32)

        return {
            "sigmas": sigmas,
            "timesteps": timesteps,
            "solver_sigmas": solver_sigmas,
            "solver_timesteps": solver_timesteps,
            "step_sizes": step_sizes,
            "step_scales": step_scales,
            "schedule_name": schedule_name,
        }

    @staticmethod
    def set_timesteps_flux(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.003/1.002
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def set_timesteps_wan(
        num_inference_steps=100,
        denoising_strength=1.0,
        shift=None,
        inference_schedule: Literal["linear", "sigma_shift", "sd3", "c_function"] = "sigma_shift",
        sd3_r: float = 6.0,
        c_interp: float = 0.8,
        c_start: float = 1.0,
        c_t_end: float = 0.999,
        c_grid_size: int = 4096,
    ):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        solver_sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]

        if inference_schedule == "linear":
            sigmas = solver_sigmas.clone()
            timesteps = sigmas * num_train_timesteps
            return FlowMatchScheduler._format_schedule_metadata(
                sigmas,
                timesteps,
                solver_sigmas=solver_sigmas,
                schedule_name="linear",
            )

        if inference_schedule == "sigma_shift":
            sigmas = shift * solver_sigmas / (1 + (shift - 1) * solver_sigmas)
            timesteps = sigmas * num_train_timesteps
            return FlowMatchScheduler._format_schedule_metadata(
                sigmas,
                timesteps,
                solver_sigmas=solver_sigmas,
                schedule_name="sigma_shift",
            )

        if sigma_start <= 0:
            sigmas = solver_sigmas.clone()
            timesteps = sigmas * num_train_timesteps
            step_sizes = torch.zeros_like(sigmas)
            step_scales = torch.zeros_like(sigmas)
            return FlowMatchScheduler._format_schedule_metadata(
                sigmas,
                timesteps,
                solver_sigmas=solver_sigmas,
                step_sizes=step_sizes,
                step_scales=step_scales,
                schedule_name=inference_schedule,
            )

        solver_time = torch.linspace(0.0, 1.0, num_inference_steps + 1, dtype=solver_sigmas.dtype)[:-1]
        resolver = InferenceScheduleResolver(
            sample_with_warp=True,
            function_type=inference_schedule,
            sd3_r=sd3_r,
            c_interp=c_interp,
            c_start=c_start,
            c_t_end=c_t_end,
            c_grid_size=c_grid_size,
            device=solver_time.device,
            dtype=solver_time.dtype,
        )
        native_progress, step_scales = resolver.resolve(solver_time)
        sigmas = sigma_start * (1.0 - native_progress)
        timesteps = sigmas * num_train_timesteps

        next_solver_time = torch.cat([solver_time[1:], torch.ones(1, dtype=solver_time.dtype)])
        solver_step_size = next_solver_time - solver_time
        step_sizes = -sigma_start * step_scales * solver_step_size
        return FlowMatchScheduler._format_schedule_metadata(
            sigmas,
            timesteps,
            solver_sigmas=solver_sigmas,
            step_sizes=step_sizes,
            step_scales=step_scales,
            schedule_name=inference_schedule,
        )
    
    @staticmethod
    def _calculate_shift_qwen_image(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
    
    @staticmethod
    def set_timesteps_qwen_image(num_inference_steps=100, denoising_strength=1.0, exponential_shift_mu=None, dynamic_shift_len=None):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        shift_terminal = 0.02
        # Sigmas
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        # Mu
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = FlowMatchScheduler._calculate_shift_qwen_image(dynamic_shift_len)
        else:
            mu = 0.8
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        # Shift terminal
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - shift_terminal)
        sigmas = 1 - (one_minus_z / scale_factor)
        # Timesteps
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def set_timesteps_qwen_image_lightning(num_inference_steps=100, denoising_strength=1.0, exponential_shift_mu=None, dynamic_shift_len=None):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        base_shift = math.log(3)
        max_shift = math.log(3)
        # Sigmas
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        # Mu
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = FlowMatchScheduler._calculate_shift_qwen_image(dynamic_shift_len, base_shift=base_shift, max_shift=max_shift)
        else:
            mu = 0.8
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        # Timesteps
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def compute_empirical_mu(image_seq_len, num_steps):
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)
    
    @staticmethod
    def set_timesteps_flux2(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=None):
        sigma_min = 1 / num_inference_steps
        sigma_max = 1.0
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        if dynamic_shift_len is None:
            # If you ask me why I set mu=0.8,
            # I can only say that it yields better training results.
            mu = 0.8
        else:
            mu = FlowMatchScheduler.compute_empirical_mu(dynamic_shift_len, num_inference_steps)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_z_image(num_inference_steps=100, denoising_strength=1.0, shift=None, target_timesteps=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        if target_timesteps is not None:
            target_timesteps = target_timesteps.to(dtype=timesteps.dtype, device=timesteps.device)
            for timestep in target_timesteps:
                timestep_id = torch.argmin((timesteps - timestep).abs())
                timesteps[timestep_id] = timestep
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_ltx2(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=None, terminal=0.1, special_case=None):
        num_train_timesteps = 1000
        if special_case == "stage2":
            sigmas = torch.Tensor([0.909375, 0.725, 0.421875])
        elif special_case == "ditilled_stage1":
            sigmas = torch.Tensor([1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875])
        else:
            dynamic_shift_len = dynamic_shift_len or 4096
            sigma_shift = FlowMatchScheduler._calculate_shift_qwen_image(
                image_seq_len=dynamic_shift_len,
                base_seq_len=1024,
                max_seq_len=4096,
                base_shift=0.95,
                max_shift=2.05,
            )
            sigma_min = 0.0
            sigma_max = 1.0
            sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
            sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
            sigmas = math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1))
            # Shift terminal
            one_minus_z = 1.0 - sigmas
            scale_factor = one_minus_z[-1] / (1 - terminal)
            sigmas = 1.0 - (one_minus_z / scale_factor)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_training_weight(self):
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing
        
    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        results = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )
        if isinstance(results, dict):
            self.sigmas = results["sigmas"]
            self.timesteps = results["timesteps"]
            self.solver_sigmas = results["solver_sigmas"]
            self.solver_timesteps = results["solver_timesteps"]
            self.step_sizes = results["step_sizes"]
            self.step_scales = results["step_scales"]
            self.schedule_name = results["schedule_name"]
        else:
            self.sigmas, self.timesteps = results
            metadata = self._format_schedule_metadata(self.sigmas, self.timesteps)
            self.solver_sigmas = metadata["solver_sigmas"]
            self.solver_timesteps = metadata["solver_timesteps"]
            self.step_sizes = metadata["step_sizes"]
            self.step_scales = metadata["step_scales"]
            self.schedule_name = metadata["schedule_name"]
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, progress_id=None, **kwargs):
        if progress_id is not None and self.step_sizes is not None:
            step_size = self.step_sizes[progress_id].to(device=sample.device, dtype=sample.dtype)
            return sample + model_output * step_size
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    
    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    
    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    
    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
