CUDA_VISIBLE_DEVICES=4 python3 noise_prior_inference.py --dataset cityscapes \
                                 --sample_size 50000  --diffusion_steps 799 --seed 4 \
                                 --save_dir ./trash --prior_type spatial-class-joint \
                                 --spatial_scale 1.00 --class_scale 1.00 --batch_size 5 \
                                 --resolution 512 1024 --ckpt /path/to/your/ckpt \
                                 --stat_dir ./statistics --stat_name cityscapes_hr --scale 4.0