import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize

import glob

import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils import create_coco_stuff_label_colormap, convert_gray_to_color_official

class CocostuffDataset(Dataset):
    BASE = '/path/to/your/dataset'
    RESOLUTION = (384, 512) 

    def __init__(self):
        self.data, self.labels = [], []

        for file in tqdm(os.scandir(os.path.join(self.BASE, 'annotations', 'train2017'))):
            self.labels.append(os.path.join(self.BASE, 'annotations', 'train2017', file))
            
        for file in tqdm(os.scandir(os.path.join(self.BASE, 'images', 'train2017'))):
            self.data.append(os.path.join(self.BASE, 'images', 'train2017', file))
        
        self.data = sorted(self.data)
        self.labels = sorted(self.labels)

        assert len(self.data) == len(self.labels), "Data len is not equal as label len."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_arr = np.array(Image.open(self.data[idx]))
        gray_label = np.array(Image.open(self.labels[idx]).convert('L'))
        if len(img_arr.shape) != 3:
            img_arr = np.expand_dims(img_arr,2).repeat(3,axis=2)    # (H,W)->(H,W,3)

        color_map = create_coco_stuff_label_colormap()
        color_label_arr = convert_gray_to_color_official(color_map, gray_label)

        hint = color_label_arr.astype(np.float32) / 255.0  # (H, W, 3)
        target = img_arr.astype(np.float32) / 127.5 - 1.0  # (H, W, 3)
        T = Resize(self.RESOLUTION, antialias=True)

        resized_hint = T(torch.tensor(hint).permute(2, 0, 1)).permute(1, 2, 0)
        resized_target = T(torch.tensor(target).permute(2, 0, 1)).permute(1, 2, 0)

        prompt = "high quality, detailed."
        return dict(jpg=resized_target.float(), txt=prompt, hint=resized_hint.float(), label=T(torch.tensor(gray_label)[None, ...])[0], orig_img_path=self.data[idx], orig_label_path=self.labels[idx])
