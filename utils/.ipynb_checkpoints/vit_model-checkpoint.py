import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super(VisionTransformer, self).__init__()
        # Define the layers of your ViT model (you can use a pre-built ViT or implement your own)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x
