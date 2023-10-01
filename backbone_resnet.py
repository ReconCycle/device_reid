import torch
from torch import optim, nn
import torchvision


class BackboneResnet():
    def __init__(self, freeze=True, device="cuda") -> None:
        self.type = "resnet_imagenet"

        self.backbone_model = torchvision.models.resnet50(pretrained=True).to(device)
        self.backbone_model = torch.nn.Sequential(*(list(self.backbone_model.children())[:-1]))

        if freeze:
            for param in self.backbone_model.parameters():
                param.requires_grad = False
    
    def forward(self, sample):
        out = self.backbone_model(sample) # shape (batch, 1, 2048)
        out = torch.squeeze(out) # shape(batch, 2048)

        return out

