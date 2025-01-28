import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class FabricDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        # Convert image to RGB (OpenCV loads it in BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)  # Apply any transform like ToTensor() or normalization
        
        return image
