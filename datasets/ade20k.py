import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils import gen_ade20k_color_map, convert_gray_to_color

class ADE20KDataset(Dataset):
    BASE = '/path/to/your/dataset'
    RESOLUTION = (512, 512) 

    def __init__(self, split='training'):
        super().__init__()
        self.data, self.labels = [], []

        self.split = split

        for root, _, files in os.walk(osp.join(self.BASE, 'images', self.split)):
            for file in files:
                self.data.append(osp.join(self.BASE, 'images', self.split, file))
        
        for root, _, files in os.walk(osp.join(self.BASE, 'annotations', self.split)):
            for file in files:
                self.labels.append(osp.join(self.BASE, 'annotations', self.split, file))
        
        self.labels = sorted(self.labels)
        self.data = sorted(self.data)
        
        assert len(self.data) == len(self.labels), "Number of images and labels do not match."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = osp.join(self.BASE, self.data[idx])
        label_path = osp.join(self.BASE, self.labels[idx])
        img_arr = np.array(Image.open(img_path))
        label_arr = np.array(Image.open(label_path))

        color_map = gen_ade20k_color_map()
        color_label_arr = convert_gray_to_color(color_map, label_arr)

        hint = color_label_arr.astype(np.float32) / 255.0  # (H, W, 3)
        target = img_arr.astype(np.float32) / 127.5 - 1.0  # (H, W, 3)

        # Resize
        T = Resize(self.RESOLUTION, antialias=True)

        resized_hint = T(torch.tensor(hint).permute(2, 0, 1)).permute(1, 2, 0)
        resized_target = T(torch.tensor(target).permute(2, 0, 1)).permute(1, 2, 0)

        prompt = "Photorealistic and diverse images depicting various scenes"
        return dict(jpg=resized_target.float(), txt=prompt, hint=resized_hint.float(), label=T(torch.tensor(label_arr)[None, ...])[0], orig_img_path=img_path, orig_label_path=label_path)
