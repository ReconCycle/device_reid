from __future__ import annotations
import os
import numpy as np
import time
import cv2
from rich import print
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, Point
import matplotlib.cm as cm
from PIL import Image

from types import SimpleNamespace

from graph_relations import GraphRelations, exists_detection, compute_iou

from helpers import scale_img
from object_reid import ObjectReId
from pathlib import Path
import torch
from superglue.models.matching import Matching
from superglue.utils.common import make_matching_plot_fast, frame2tensor, VideoStreamer



class ObjectReIdSuperGlue(ObjectReId):
    def __init__(self, model="indoor") -> None:
        super().__init__()
    
        torch.set_grad_enabled(False)
        
        opt = SimpleNamespace()
        opt.superglue = model
        opt.nms_radius = 4
        opt.sinkhorn_iterations = 20
        opt.match_threshold = 0.2
        opt.show_keypoints = True
        opt.keypoint_threshold = 0.005
        opt.max_keypoints = -1
        
        weights_mapping = {
                'superpoint': Path(__file__).parent / 'superglue/models/weights/superpoint_v1.pth',
                'indoor': Path(__file__).parent / 'superglue/models/weights/superglue_indoor.pth',
                'outdoor': Path(__file__).parent / 'superglue/models/weights/superglue_outdoor.pth',
                'coco_homo': Path(__file__).parent / 'superglue/models/weights/superglue_cocohomo.pt'
            }

        try:
            curr_weights_path = str(weights_mapping[opt.superglue])
            if not os.path.isfile(curr_weights_path):
                print(f"[red]{curr_weights_path} is not a file!")
        except:
            if os.path.isfile(opt.superglue) and (os.path.splitext(opt.superglue)[-1] in ['.pt', '.pth']):
                curr_weights_path = str(opt.superglue)
            else:
                raise ValueError("Given --superglue path doesn't exist or invalid")

        print("curr_weights_path", curr_weights_path)

        self.opt = opt
        config = {
            'superpoint': {
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints
            },
            'superglue': {
                'weights_path': curr_weights_path,
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
            }
        }


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matching = Matching(config).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']
        
        # self.timer = AverageTimer()

    def compare_full_img(self, img0, graph0, img1, graph1, visualise=False):
        img0_cropped, obb_poly1 = self.find_and_crop_det(img0, graph0)
        img1_cropped, obb_poly2 = self.find_and_crop_det(img1, graph1)
        
        img0 = cv2.cvtColor(img0_cropped, cv2.COLOR_RGB2GRAY)
        img1 = cv2.cvtColor(img1_cropped, cv2.COLOR_RGB2GRAY)

        self.compare(img0, img1, visualise=visualise)


    def compare(self, img1, img2, visualise=False, debug=True):
        if visualise:
            print("[blue]starting compare...[/blue]")
        # self.timer.update('data')
        item = 0

        # cv2.imshow("img1 superglue", img1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print("saving image for visualisation!")
        # im = Image.fromarray(img1).convert("L")
        # im.save("TEMP_img1.jpg")
        
        img1_tensor = frame2tensor(img1, self.device)
        last_data = self.matching.superpoint({'image': img1_tensor})
        last_data = {k+'0': last_data[k] for k in self.keys}
        last_data['image0'] = img1_tensor
        
            
        # TODO: ignore matches outside OBB
        
        img2_tensor = frame2tensor(img2, self.device)
        
        pred = self.matching({**last_data, 'image1': img2_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        # self.timer.update('forward')

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        k_thresh = self.matching.superpoint.config['keypoint_threshold']
        m_thresh = self.matching.superglue.config['match_threshold']
        
        # print("matches[valid].shape", len(matches[valid]), matches[valid].shape)
        # print("kpts0", len(kpts0), kpts0.shape)
        # print("kpts0", len(kpts1), kpts1.shape)
        # 3 matches will always score perfectly because of affine transform
        # let's say we want at least 5 matches to work
        if len(matches[valid]) <= 5:
            if debug:
                print("not enough matches for SuperGlue", len(matches[valid]))
            # todo: return something else than 0.0, more like undefined.
            return 0.0
        else:
            mean_error, median_error, max_error = self.calculate_matching_error(mkpts0, mkpts1)
            
            # a median error of less than 0.5 is good
            strength = 1.0 # increase strength for harsher score function
            score = 1/(strength*median_error + 1) #! we should test this score function
            
            if debug:
                print("median_error", median_error)
                print("score", score)
            
            min_num_kpts = min(len(kpts0), len(kpts1))
            
            score_ratio = len(matches[valid])/min_num_kpts
            
        if visualise:
            color = cm.jet(confidence[valid])
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0))
            ]
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                # 'Image Pair: {:06}:{:06}'.format(stem0, stem1),
            ]
            out = make_matching_plot_fast(
                img1, img2, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=self.opt.show_keypoints, small_text=small_text)
            cv2.imshow('SuperGlue matches', out)
            cv2.waitKey() # visualise

        if debug:
            print("[green]score_ratio", score_ratio)
        
        return score_ratio


if __name__ == '__main__':
    # object_reid_superglue = ObjectReIdSuperGlue(model="indoor")
    object_reid_superglue = ObjectReIdSuperGlue(model="/home/sruiz/projects/reconcycle/superglue_training/output/train/2023-11-18_superglue_model/weights/best.pt")

    vs = VideoStreamer(
        "/home/sruiz/datasets2/reconcycle/2023-02-20_hca_backs_processed/hca_0/", 
        [640, 480], 
        1,
        ['*.png', '*.jpg', '*.jpeg'], 
        1000000)

    img1 = vs.load_image("/home/sruiz/datasets2/reconcycle/2023-02-20_hca_backs_processed/hca_0/0001.jpg")
    img2 = vs.load_image("/home/sruiz/datasets2/reconcycle/2023-02-20_hca_backs_processed/hca_0/0002.jpg")

    print("img1.shape", img1.shape)
    print("img2.shape", img2.shape)

    object_reid_superglue.compare(img1, img2)

    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    # cv2.waitKey() # visualise

    

