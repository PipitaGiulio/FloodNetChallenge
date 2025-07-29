import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms

class ClassificationDataset(Dataset):
    def __init__(self, root_path, y_path, transforms = None):
        super(ClassificationDataset, self).__init__()
        self.x_list = [os.path.join(root_path, file_name) for file_name in os.listdir(root_path)]
        self.y_labels = torch.tensor(np.load(y_path), dtype = torch.long)
        self.transforms = transforms

    def __len__(self):
        return len(self.x_list)
    
    def __getitem__(self, index):

        x_sample = Image.open(self.x_list[index])
        if self.transforms is not None:
            x_sample = self.transforms(x_sample)
        y_sample = self.y_labels[index]
        return x_sample, y_sample