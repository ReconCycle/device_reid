#%%
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #! specify gpu here
os.chdir(os.path.dirname(__file__)) # set working dir to file dir

from datetime import datetime
import cv2
import numpy as np
# import json
from rich import print
from PIL import Image
from tqdm import tqdm
import logging
import torch
from torchvision.transforms import ToTensor
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

import clip

# import exp_utils as exp_utils
from exp_utils import str2bool, clip_transform

from data_loader import DataLoader, random_seen_unseen_class_split

#%%


img_path = os.path.expanduser("~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted_cropped")

seen_classes, unseen_classes = random_seen_unseen_class_split(img_path, seen_split=1.0)

# transform used by clip:
train_transform = clip_transform(224)
val_transform = clip_transform(224)

dl = DataLoader(img_path,
                batch_size=8,
                num_workers=8,
                combine_val_and_test=True, #! since we are not training, we merge val and test sets for .80/.20 split
                shuffle=True,
                seen_classes=seen_classes,
                unseen_classes=unseen_classes,
                train_transform=train_transform,
                val_transform=val_transform)

num_classes = len(dl.classes['seen'])

# print("dl.classes", dl.classes)
print(f"dataset lens: {dl.dataset_lens}")
print(f"seen classes: {dl.classes['seen']}")
print(f"num classes: {num_classes}")


parent_class_names = ["firealarm_back", "firealarm_front", "hca_back", "hca_front", "firealarm_inside"]
num_parent_classes = len(parent_class_names)

parent_classes_lookup = np.zeros(num_classes, dtype=int)
for index, a_class in enumerate(dl.classes['seen']):
    possible_parent_ids = [idx for idx, s in enumerate(parent_class_names) if s in a_class]
    if len(possible_parent_ids) > 0:
        parent_classes_lookup[index] = possible_parent_ids[0]
    else:
        # if parent class doesn't exist, call it -1
        parent_classes_lookup[index] = -1


print("parent_classes_lookup", parent_classes_lookup)

#%%

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)
model, preprocess = clip.load("ViT-B/32", device=device)


# %%

# run all train images through model.

def infer_all(data_subset):
    clip_encodings = None
    gt_labels = None

    with torch.no_grad():
        for batch in dl.dataloaders[data_subset]:
            imgs, labels, img_path, _ = batch
            # print("imgs", imgs.shape, ids.shape)
            clip_encoding = model.encode_image(imgs.to(device))
            # print("clip", clip_encoding.shape)
            if clip_encodings is None:
                clip_encodings = clip_encoding
                gt_labels = labels
            else:
                clip_encodings = torch.cat((clip_encodings, clip_encoding), dim=0)
                gt_labels = torch.cat((gt_labels, labels), dim=0)

            # print("clip_encodings", clip_encodings.shape)

    return clip_encodings, gt_labels
    
# image_features = model.encode_image(image)
# %%
train_encodings, train_gt_labels = infer_all("seen_train")

print("train_encodings", train_encodings.shape)
print("train_gt_labels", train_gt_labels.shape)

# %%

# run test images through model
# find top-k nearest neighbours from training images, and assign that label to it

def evaluate_topk(data_subset, k=5):

    gt_labels = None
    topk_values = None
    topk_indicies = None
    topk_pred_labels = None

    with torch.no_grad():
        for batch in dl.dataloaders[data_subset]:
            imgs, labels, img_path, _ = batch

            clip_encodings = model.encode_image(imgs.to(device))
            
            batch_topk_values = []
            batch_topk_indicies = []
            batch_topk_pred_labels = []
            for clip_encoding, label in zip(clip_encodings, labels):

                # test = clip_encoding[0]

                # print("test.shape", test.shape)

                diff = train_encodings - clip_encoding

                # print("diff.shape", diff.shape)

                dist = torch.norm(train_encodings - clip_encoding, dim=1, p=2)
                knn = dist.topk(k, largest=False)

                # print("dist", dist.shape)
                # print("knn", knn)
                # print("knn", knn.indices)

                knn_predicted_labels = train_gt_labels[knn.indices]
                
                # print("knn_predicted_labels", knn_predicted_labels)

                # print("ground truth label", label)

                batch_topk_values.append(knn.values)
                batch_topk_indicies.append(knn.indices)
                batch_topk_pred_labels.append(knn_predicted_labels)

            batch_topk_values = torch.stack(batch_topk_values, dim=0)
            batch_topk_indicies = torch.stack(batch_topk_indicies, dim=0)
            batch_topk_pred_labels = torch.stack(batch_topk_pred_labels, dim=0)
            # print("batch_topk_pred_labels", batch_topk_pred_labels.shape)

            if topk_pred_labels is None:
                topk_values = batch_topk_values
                topk_indicies = batch_topk_indicies
                topk_pred_labels = batch_topk_pred_labels
                gt_labels = labels
            else:
                topk_values = torch.cat((topk_values, batch_topk_values), dim=0)
                topk_indicies = torch.cat((topk_indicies, batch_topk_indicies), dim=0)
                topk_pred_labels = torch.cat((topk_pred_labels, batch_topk_pred_labels), dim=0)
                gt_labels = torch.cat((gt_labels, labels), dim=0)

            # break

        return gt_labels, topk_values, topk_indicies, topk_pred_labels

# test_gt_labels, test_topk_values, test_topk_indicies, test_topk_pred_labels = evaluate_topk("seen_val")
test_gt_labels, test_topk_values, test_topk_indicies, test_topk_pred_labels = evaluate_topk("seen_test")

print("test_gt_labels", test_gt_labels.shape)
print("test_topk_values", test_topk_values.shape)
print("test_topk_indicies", test_topk_indicies.shape)
print("test_topk_pred_labels", test_topk_pred_labels.shape)

# %%

# accuracy over all classes
def compute_acc(k=1):
    items_per_class = np.zeros(num_classes)
    acc_class = np.zeros(num_classes)
    items_per_parent_class = np.zeros(num_parent_classes)
    acc_parent_class = np.zeros(num_parent_classes)
    acc = 0
    for gt_label, topk_pred_label in zip(test_gt_labels, test_topk_pred_labels):
        items_per_class[gt_label] += 1 # count number of items in that class
        items_per_parent_class[parent_classes_lookup[gt_label]] += 1 # count number of items in parent class
        if gt_label in topk_pred_label[:k]:
            acc += 1
            acc_class[gt_label] += 1
            acc_parent_class[parent_classes_lookup[gt_label]] += 1

    acc = acc / test_gt_labels.size(dim=0)

    # divide by the number of items in that class
    for idx in np.arange(num_classes):
        acc_class[idx] = acc_class[idx] / items_per_class[idx]

    # divide by the number of items in that parent class
    for idx in np.arange(num_parent_classes):
        acc_parent_class[idx] = acc_parent_class[idx] / items_per_parent_class[idx]

    print("items_per_class", items_per_class)
    print("items_per_parent_class", items_per_parent_class)

    # we want the distribution of classes in the top-k, for each ground truth class
    topk_dist_foreach_class = []
    for class_id in np.arange(num_classes):
        topk_for_class = [topk_pred_label for gt_label, topk_pred_label in  zip(test_gt_labels, test_topk_pred_labels) if gt_label == class_id]
        if len(topk_for_class) > 0:
            topk_for_class = torch.stack(topk_for_class, dim=0)
            # print("topk_for_class", topk_for_class)
            # print(f"topk_for_class {class_id}", topk_for_class.shape)

            topk_dist_foreach_class.append(topk_for_class)
        else:
            topk_dist_foreach_class.append(None)


    return acc, acc_class, acc_parent_class, topk_dist_foreach_class

print("parent_class_names", parent_class_names)

acc1, acc1_class, acc1_parent_class, _ = compute_acc(k=1)
print("top-1 acc", acc1) # 0.935
print("acc1_class", acc1_class)
print("acc1_parent_class", acc1_parent_class)

acc3, acc3_class, acc3_parent_class, top3_dist_foreach_class = compute_acc(k=3)
print("top-3 acc", acc3)
print("acc3_class", acc3_class)
print("acc3_parent_class", acc3_parent_class)

acc5, acc5_class, acc5_parent_class, top5_dist_foreach_class = compute_acc(k=5)
print("top-5 acc", acc5)
print("acc5_class", acc5_class)
print("acc5_parent_class", acc5_parent_class)

# %%
from collections import Counter

dist = top5_dist_foreach_class[-1] #! specify index for each g.t. class
dist_flat = torch.flatten(dist).numpy()
count = Counter(dist_flat)
count_sorted = sorted(count.items(), key=lambda item: item[1], reverse=True)
count_sorted_freq = [(key, val/len(dist_flat)) for (key, val) in count_sorted]

cum_dict = {}

fig, ax = plt.subplots()

x, y = zip(*count_sorted_freq)
x = [str(x_item) for x_item in x]
ax.bar(x, y)
ax.set_title(f"distribution top5, for class ??")
ax.set_ylabel('frequency')
ax.set_xlabel('class')
plt.show()

# TODO: show image of class on each column
# %%