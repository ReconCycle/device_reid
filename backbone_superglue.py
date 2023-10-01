import torch
from torch import optim, nn
import torchvision

from superglue.models.matching import Matching


class BackboneSuperglue():
    def __init__(self, opt, freeze=True, device="cuda") -> None:
        self.type = "superglue"

        matching_config = {
            'superpoint': {
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints
            },
            'superglue': {
                'weights': opt.superglue,
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
            }
        }

        matching = Matching(matching_config).to(device)
        self.superpoint= matching.superpoint

        if freeze:
            for param in self.superpoint.parameters():
                param.requires_grad = False
    
    def forward(self, sample):
        # TODO: convert image to grayscale

        img_data = self.superpoint({'image': sample})
        out = img_data["x_waypoint"] # shape: (1, 65, 50, 50)

        return out

