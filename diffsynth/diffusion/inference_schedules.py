from __future__ import annotations

from dataclasses import dataclass

import torch


def truncated_c_function(
    t: torch.Tensor,
    interp: float = 0.8,
    start: float = 1.0,
) -> torch.Tensor:
    """Piecewise-linear c(t) schedule used by the EqF inference warp."""
    if not 0.0 < interp < 1.0:
        raise ValueError(f"interp must lie in (0, 1), got {interp}.")

    first_branch = start - ((start - 1.0) / interp) * t
    second_branch = (1.0 / (1.0 - interp)) - (t / (1.0 - interp))
    return torch.minimum(first_branch, second_branch)


def _interp_1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Linear interpolation over a monotone lookup table."""
    x = x.to(device=xp.device, dtype=xp.dtype)
    x = x.clamp(xp[0], xp[-1])

    idx = torch.searchsorted(xp, x, right=False)
    idx = idx.clamp(1, xp.numel() - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    denom = (x1 - x0).clamp_min(torch.finfo(xp.dtype).eps)
    weight = (x - x0) / denom
    return y0 + weight * (y1 - y0)


@dataclass
class InferenceScheduleResolver:
    sample_with_warp: bool
    function_type: str = "c_function"
    sd3_r: float = 6.0
    c_interp: float = 0.8
    c_start: float = 1.0
    c_t_end: float = 0.999
    c_grid_size: int = 4096
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.c_normalization = None
        self.c_solver_grid = None
        self.c_native_grid = None

        if not self.sample_with_warp:
            return

        if self.function_type not in {"c_function", "sd3"}:
            raise ValueError(
                f"Unsupported function_type '{self.function_type}'. Expected 'c_function' or 'sd3'."
            )

        if self.function_type == "sd3":
            if self.sd3_r <= 0.0:
                raise ValueError(f"sd3_r must be positive, got {self.sd3_r}.")
            return

        if self.c_grid_size < 2:
            raise ValueError(f"c_grid_size must be at least 2, got {self.c_grid_size}.")
        if not 0.0 < self.c_t_end < 1.0:
            raise ValueError(f"c_t_end must lie in (0, 1), got {self.c_t_end}.")

        native_t = torch.linspace(
            0.0,
            self.c_t_end,
            self.c_grid_size,
            device=self.device,
            dtype=self.dtype,
        )
        c_values = truncated_c_function(
            native_t,
            interp=self.c_interp,
            start=self.c_start,
        ).clamp_min(torch.finfo(self.dtype).eps)
        inv_c = 1.0 / c_values

        cumulative = torch.zeros_like(native_t)
        delta_t = native_t[1:] - native_t[:-1]
        trap = 0.5 * (inv_c[:-1] + inv_c[1:]) * delta_t
        cumulative[1:] = torch.cumsum(trap, dim=0)

        self.c_normalization = cumulative[-1]
        self.c_solver_grid = cumulative / self.c_normalization
        self.c_native_grid = native_t

    def resolve(self, solver_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        solver_time = solver_time.to(device=self.device, dtype=self.dtype)

        if not self.sample_with_warp:
            return solver_time, torch.ones_like(solver_time)

        solver_time = solver_time.clamp(0.0, 1.0)

        if self.function_type == "sd3":
            denom = 1.0 + (self.sd3_r - 1.0) * solver_time
            native_t = self.sd3_r * solver_time / denom
            step_scale = self.sd3_r / denom.square()
            return native_t, step_scale

        native_t = _interp_1d(solver_time, self.c_solver_grid, self.c_native_grid)
        step_scale = self.c_normalization * truncated_c_function(
            native_t,
            interp=self.c_interp,
            start=self.c_start,
        )
        return native_t, step_scale
