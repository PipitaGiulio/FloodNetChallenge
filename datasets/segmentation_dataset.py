import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms



class SegmentationDataset(Dataset):

    def __init__(self, og_root, mask_root, transforms = None):
        super(SegmentationDataset, self).__init__()

        self.x_list = sorted([os.path.join(og_root, file_name) for file_name in os.listdir(og_root)])
        self.y_list = sorted([os.path.join(mask_root, mask_name) for mask_name in os.listdir(mask_root)])
        
        self.transforms = transforms

    def __len__(self):
        return len(self.x_list)
    
    def __getitem__(self, index):
        
        x_sample = np.array(Image.open(self.x_list[index]))
        y_sample = np.array(Image.open(self.y_list[index]))
        
        if self.transforms is not None:
            transformed = self.transforms(image = x_sample, mask = y_sample)
            x_sample = transformed['image']
            y_sample = transformed['mask'].long()
        return x_sample, y_sample
    



