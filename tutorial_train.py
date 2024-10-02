from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import ArgumentParser

import os


def load_dataset(dataset_name, args):
    if dataset_name == 'ade20k':
        from datasets.ade20k import ADE20KDataset
        return ADE20KDataset()
    elif dataset_name == 'cityscapes':
        from datasets.cityscapes import CityscapesDataset
        return CityscapesDataset()
    elif dataset_name == 'coco-stuff':
        from datasets.cocostuff import CocostuffDataset
        return CocostuffDataset()
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--resume_path', type=str, default='./models/control_sd21_ini.ckpt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--logger_freq', type=int, default=300)
    parser.add_argument('--sd_locked', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='./models/cldm_v21.yaml')
    parser.add_argument('--dataset', type=str, default='ade20k')
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--default_root_dir', type=str, default='work_dir')
    parser.add_argument('--gpus', nargs='+', type=int, help='List of integer values', default=[0])

    args = parser.parse_args()

    resume_path = args.resume_path
    batch_size = args.batch_size
    logger_freq = 300
    learning_rate = 1e-5
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = only_mid_control

    
    os.makedirs(args.default_root_dir, exist_ok=True)
    dataset = load_dataset(args.dataset, args)
    dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=args.gpus, max_steps=100000, default_root_dir=os.path.join(args.default_root_dir, args.dataset),
                        callbacks=[logger, 
                                    ModelCheckpoint(dirpath=os.path.join(args.default_root_dir, args.dataset, 'ckpt'),
                                    save_last=True, every_n_train_steps=5000, save_top_k=-1)],
                        enable_progress_bar=True
                        )

    # Train!
    trainer.fit(model, dataloader)