import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import Resize

import os

class CityscapesDataset(Dataset):
    RESOLUTION = (512, 1024)
    BASE = '/path/to/your/dataset'

    def __init__(self):
        super().__init__()
        self.data, self.label, self.labelIds = [], [], []

        for root, dirs, files in os.walk(os.path.join(self.BASE, 'leftImg8bit')):
            if 'test' in dirs:
                dirs.remove('test')
                continue
            
            for file in files:
                if file.endswith("_leftImg8bit.png"):
                    file_path = os.path.join(root, file)
                    self.data.append(file_path)
        
        for root, dirs, files in os.walk(os.path.join(self.BASE, 'gtFine')):
            if 'test' in dirs:
                dirs.remove('test')
                continue

            for file in files:
                if file.endswith("_gtFine_color.png"):
                    file_path = os.path.join(root, file)
                    self.label.append(file_path)
                elif file.endswith("_gtFine_labelTrainIds.png"):
                    file_path = os.path.join(root, file)
                    self.labelIds.append(file_path)
        
        self.data = sorted(self.data)
        self.label = sorted(self.label)
        self.labelIds = sorted(self.labelIds)

        assert len(self.data) == len(self.label), "Data len is not equal as label len."

    def __len__(self):
        return len(self.data)  

    def __getitem__(self, index):
        data_path = self.data[index]
        label_path = self.label[index]

        img_arr = cv2.imread(data_path)
        label_arr = cv2.imread(label_path)

        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        label_arr = cv2.cvtColor(label_arr, cv2.COLOR_BGR2RGB)

        resize = Resize(size=self.RESOLUTION)
        img_arr = resize(torch.tensor(img_arr).permute(2,0,1)).permute(1,2,0).numpy()
        label_arr = resize(torch.tensor(label_arr).permute(2,0,1)).permute(1,2,0).numpy()

        img_arr = (img_arr.astype(np.float32) / 127.5) -  1.0
        label_arr = label_arr.astype(np.float32) / 255.0

        prompt = "City road scenes."

        label_trainId = cv2.imread(self.labelIds[index], cv2.IMREAD_GRAYSCALE)

        return dict(jpg=img_arr, txt=prompt, hint=label_arr, label=resize(torch.tensor(label_trainId)[None, ...])[0].numpy(), orig_img_path=data_path, orig_label_path=self.labelIds[index])

