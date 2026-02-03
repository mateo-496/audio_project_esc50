import torch
import torch.nn as nn
from torch.nn import functional as F



class PatchEmbed(nn.Module):
    def __init__(
        self, 
        img_size=224,
        patch_size=16,
        stride=10,
        in_channels=1,
        embed_dimension=768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dimension,
            kernel_size=patch_size,
            stride=stride
        )
    def forward(self, x):
        B, C, F, T = x.shape
        x = self.projection(x)
        