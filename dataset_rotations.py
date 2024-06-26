from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import PIL.Image as Image
import numpy as np
import cv2
import os
import sys
import yaml
import math
import regex as re
from tqdm import tqdm
# from rich import print
import jsonpickle
import pandas as pd
import jsonpickle.ext.numpy as jsonpickle_numpy
import seaborn_image as isns
from exp_utils import scale_img
# do as if we are in the parent directory
# from action_predictor.graph_relations import GraphRelations
# from vision_pipeline.object_reid import ObjectReId
import device_reid.exp_utils as exp_utils
import random
from natsort import os_sorted
from sklearn.model_selection import train_test_split
from superglue_training.utils.preprocess_utils import get_perspective_mat


class ImageDatasetRotations(datasets.ImageFolder):
    def __init__(self,
                 main_path,
                 class_dirs=[],
                 unseen_class_offset=0,
                 transform=None,
                 exemplar_transform=None,
                 limit_imgs_per_class=None,
                 template_imgs_only=False):
        
        self.main_path = main_path

        with open(os.path.expanduser("~/superglue_training/configs/get_perspective_hcas_firealarms_only.yaml"), 'r') as file:
            config = yaml.full_load(file)

        dataset_params = config["dataset_params"]
        self.aug_params = dataset_params['augmentation_params']

        self.transform = transform

        
        self.class_dirs = class_dirs
        self.unseen_class_offset = unseen_class_offset
        self.exemplar_transform = exemplar_transform

        if template_imgs_only:
            print("\n"+"="*20)
            print("limit to template image")
            print("="*20, "\n")
            def is_valid_file(file_path):
                file_name = os.path.basename(file_path)

                # num = int(re.findall(r'\d+', file_name)[-1]) # gets the last number in the filename
                # print("file_name", file_name, num)
                if "template" in file_name:
                    print("template", file_path)
                    return True
                
                else:
                    return False

        else:
            is_valid_file = None
        
        
        super(ImageDatasetRotations, self).__init__(main_path, transform, is_valid_file=is_valid_file)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, label) where target is class_index of the target class.
        """
        path, label = self.samples[index]


        image = cv2.imread(path)

            
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        label = label + self.unseen_class_offset
        
        # apply albumentations transform
        # sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB) #! do we need this??
        
        # if isinstance(self.transform, A.Compose):
        #     sample = self.transform(image=sample)["image"]

        #     # print("self.transform", self.transform)
        # else:
        #     print("[red]not using augmentations!")
        # # else:
        # #     sample_pil = Image.fromarray(sample)
        # #     sample = self.transform(sample_pil)

        # # print("sample.shape", sample.shape)

        # sample_rot = Image.fromarray(sample).copy()

        # # print("sample_rot.size", sample_rot.size)

        # rand_deg = int(random.random() * 360)
        # sample_rot = sample_rot.rotate(rand_deg, expand=True) # in degrees counter clockwise
        # sample_rot = np.array(sample_rot)

        # # print("sample_rot.size", sample_rot.size)

        #! =========================
        #! COPY PASTED FROM superglue_training/get_perspective_hcas_firealarms.py

        image = cv2.resize(image, (300, 300), interpolation = cv2.INTER_AREA) #! downscale!!
        height, width = image.shape[0:2]

        homo_matrix, rotation_angle_value = get_perspective_mat(self.aug_params['patch_ratio'], width//2, height//2, self.aug_params['perspective_x'], self.aug_params['perspective_y'], self.aug_params['shear_ratio'], self.aug_params['shear_angle'], self.aug_params['rotation_angle'], self.aug_params['scale'], self.aug_params['translation'])
        res_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))

        transform_normalise = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255, always_apply=True)

        aug_list = [A.OneOf([A.MotionBlur(p=0.5), A.GaussNoise(p=0.6)], p=0.5),
                    A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.5),
                    A.RandomBrightnessContrast(p=0.5)
                    ]

        if self.transform is None:
            aug_list.extend([transform_normalise, ToTensorV2()])

            aug_func = A.Compose(aug_list, p=1.0) # was: p=0.65
        elif self.transform == "don't normalise":
            #! special case!
            
            aug_list.extend([ToTensorV2()])

            aug_func = A.Compose(aug_list, p=1.0) # was: p=0.65

        else:
            aug_func = self.transform


        def apply_augmentations(image1, image2):
            image1_dict = {'image': image1}
            image2_dict = {'image': image2}
            result1, result2 = aug_func(**image1_dict), aug_func(**image2_dict)
            return result1['image'], result2['image']

        
        image, res_image = apply_augmentations(image, res_image)

        # def angle_from_homo(homo):
        #     # https://stackoverflow.com/questions/58538984/how-to-get-the-rotation-angle-from-findhomography
        #     u, _, vh = np.linalg.svd(homo[0:2, 0:2])
        #     R = u @ vh
        #     angle = math.atan2(R[1,0], R[0,0]) # angle between [-pi, pi)
        #     return angle
        
        # angle = angle_from_homo(homo_matrix)

        # ! WE CAN GET THE ANGLE DIRECTLY!!!!
        angle = rotation_angle_value

        #! =========================

        # if needed we could pass detections and original image too
        # return sample, label, path, poly
        return image, label, path, [], res_image, homo_matrix, angle
        
    # restrict classes to those in subfolder_dirs
    def find_classes(self, dir: str):
        print("dir", dir) #! DEBUGGING

        classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in self.class_dirs]
        classes.sort()
        # print("Directories in this dataset:", classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    