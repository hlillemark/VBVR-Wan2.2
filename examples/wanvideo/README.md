English Document: https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Wan.html

中文文档：https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Wan.html

The `Wan2.2-TI2V-5B.py` example scripts now expose the same inference schedule kwargs used by the benchmark runner:

- `inference_schedule="linear"` for an unwarped schedule
- `inference_schedule="sigma_shift"` with `sigma_shift=...` for the legacy Wan schedule
- `inference_schedule="sd3"` with `schedule_sd3_r=...`
- `inference_schedule="c_function"` with `schedule_c_interp`, `schedule_c_start`, `schedule_c_t_end`, and `schedule_c_grid_size`

For a stable benchmark-oriented entrypoint and a schedule visualization command, see the root [`README.md`](../../README.md).
