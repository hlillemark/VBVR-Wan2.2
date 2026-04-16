PYTHONPATH=/home/ubuntu/projects/VBVR-Wan2.2:${PYTHONPATH:-} 


# Training command
accelerate launch --config_file ./examples/wanvideo/model_training/full/accelerate_config_14B.yaml --num_processes 1 --main_process_port 29500 ./examples/wanvideo/model_training/train.py --dataset_config_path ./configs/vbvr_dataset.json --height 384 --width 384 --num_frames 209 --batch_size 1 --gradient_accumulation_steps 1 --dataset_num_workers 4 --dataloader_prefetch_factor 2 --dataloader_pin_memory --dataloader_persistent_workers --dataset_repeat 1 --data_file_keys clip_path --model_paths '[["./models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors","./models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors","./models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],"./models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth","./models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]' --learning_rate 1e-5 --num_epochs 999 --training_seed 123456789 --output_path ./outputs/Wan2.2-TI2V-5B-test-run --remove_prefix_in_ckpt pipe.dit. --trainable_models dit --extra_inputs input_image

 --use_gradient_checkpointing


# Example inference command
cd /home/ubuntu/projects/VBVR-Wan2.2 && PYTHONPATH=/home/ubuntu/projects/VBVR-Wan2.2:${PYTHONPATH:-} DIFFSYNTH_SKIP_DOWNLOAD=True CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 ./scripts/run_wan22_ti2v5b_vbvr_bench.py --bench-root /path/to/VBVR-Bench --model-dir ./models/Wan-AI/Wan2.2-TI2V-5B --output-root ./outputs/wan22_ti2v5b_vbvr_bench_test --splits In-Domain_50 --max-videos 2 --videos-per-task 1 --num-inference-steps 50 --inference-schedule sd3 --schedule-sd3-r 6.0 --target-num-frames 209 --fps 16 --seed 1 --sample-seed 1234 --overwrite --no-run-evalkit


# Schedule visualization command
cd /home/ubuntu/projects/VBVR-Wan2.2 && PYTHONPATH=/home/ubuntu/projects/VBVR-Wan2.2:${PYTHONPATH:-} python ./scripts/plot_wan_inference_schedules.py --output-path ./docs/assets/wan_inference_schedules.png


