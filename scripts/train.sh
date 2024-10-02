CUDA_VISIBLE_DEVICES=3,4 \
    python tutorial_train.py --batch_size 2 --dataset coco-stuff --default_root_dir trash --gpus 0 1 --resume_path /path/to/your/ckpt