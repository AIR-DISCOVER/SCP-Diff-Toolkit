CUDA_VISIBLE_DEVICES=3 \
    python3 noise_prior.py --dataset coco-stuff --sample_size 10 --save_name coco-stuff_10 \
    --ckpt '/path/to/your/ckpt' \
    --resolution 384 512 --save_dir ./trash/statistics