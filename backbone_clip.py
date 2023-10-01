import torch
from torch import optim, nn
import torchvision

import clip

class BackboneClip():
    def __init__(self, freeze=True, device="cuda") -> None:
        self.type = "clip"
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    
    def forward(self, sample):
        out = self.model.encode_image(sample.to(self.device))

        return out

