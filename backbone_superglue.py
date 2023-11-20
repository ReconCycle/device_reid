import torch
from torch import optim, nn
import torchvision
from types import SimpleNamespace

from superglue.models.matching import Matching


class BackboneSuperglue():
    def __init__(self, model="indoor", freeze=True, device="cuda") -> None:
        self.type = "superglue"

        opt = SimpleNamespace()
        opt.superglue = model
        opt.nms_radius = 4
        opt.sinkhorn_iterations = 20
        opt.match_threshold = 0.5 # default 0.2
        opt.show_keypoints = True
        opt.keypoint_threshold = 0.005
        opt.max_keypoints = -1

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

