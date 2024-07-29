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

from collections import Counter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as PILImage
import tol_colors as tc

import clip

# import exp_utils as exp_utils
from exp_utils import str2bool, clip_transform

from data_loader import DataLoader, random_seen_unseen_class_split

#%%
######################################
# load the data
######################################

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

######################################
# make a lookup for parent classes
######################################
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
######################################
# load the model
######################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)
model, preprocess = clip.load("ViT-B/32", device=device)


# %%
######################################
# run all train images through model.
######################################

def infer_all(data_subset):
    clip_encodings = None
    gt_labels = None
    img_paths = []

    with torch.no_grad():
        for batch in dl.dataloaders[data_subset]:
            imgs_batch, labels_batch, img_path_batch, _ = batch
            # print("imgs", imgs.shape, ids.shape)
            clip_encoding = model.encode_image(imgs_batch.to(device))
            # print("clip", clip_encoding.shape)
            if clip_encodings is None:
                clip_encodings = clip_encoding
                gt_labels = labels_batch
            else:
                clip_encodings = torch.cat((clip_encodings, clip_encoding), dim=0)
                gt_labels = torch.cat((gt_labels, labels_batch), dim=0)
            
            img_paths.extend(img_path_batch)
            # print("clip_encodings", clip_encodings.shape)

    img_paths = np.array(img_paths)

    return clip_encodings, gt_labels, img_paths
    
# image_features = model.encode_image(image)
# %%
train_encodings, train_gt_labels, train_img_paths = infer_all("seen_train")

print("train_encodings", train_encodings.shape)
print("train_gt_labels", train_gt_labels.shape)
print("train_img_paths", train_img_paths.shape)

# %%
######################################
# run test images through model
# find top-k nearest neighbours from training images, and assign that label to it
######################################

def evaluate_topk(data_subset, k=5):

    gt_labels = None
    topk_values = None
    topk_indicies = None
    topk_pred_labels = None
    imgs = None
    img_paths = []

    with torch.no_grad():
        for batch in dl.dataloaders[data_subset]:
            imgs_batch, labels_batch, img_path_batch, _ = batch

            clip_encodings = model.encode_image(imgs_batch.to(device))
            
            batch_topk_values = []
            batch_topk_indicies = []
            batch_topk_pred_labels = []
            for clip_encoding in clip_encodings:

                # test = clip_encoding[0]

                # print("clip_encoding.shape", clip_encoding.shape)

                # diff = train_encodings - clip_encoding

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
                gt_labels = labels_batch
                imgs = imgs_batch
            else:
                topk_values = torch.cat((topk_values, batch_topk_values), dim=0)
                topk_indicies = torch.cat((topk_indicies, batch_topk_indicies), dim=0)
                topk_pred_labels = torch.cat((topk_pred_labels, batch_topk_pred_labels), dim=0)
                gt_labels = torch.cat((gt_labels, labels_batch), dim=0)
                imgs = torch.cat((imgs, imgs_batch), dim=0)

            img_paths.extend(img_path_batch)

        img_paths = np.array(img_paths)

        return gt_labels, topk_values, topk_indicies, topk_pred_labels, imgs, img_paths

# test_gt_labels, test_topk_values, test_topk_indicies, test_topk_pred_labels = evaluate_topk("seen_val")
test_gt_labels, test_topk_values, test_topk_indicies, test_topk_pred_labels, test_imgs, test_img_paths = evaluate_topk("seen_test")

print("test_gt_labels", test_gt_labels.shape)
print("test_topk_values", test_topk_values.shape)
print("test_topk_indicies", test_topk_indicies.shape)
print("test_topk_pred_labels", test_topk_pred_labels.shape)
print("test_img_paths", test_img_paths.shape)

# %%
######################################################
# show the images in the top-k for a random test image
######################################################

# choose a random sample
item = np.random.randint(len(test_gt_labels))

print("item", item)

print("test_gt_labels[item]", test_gt_labels[item])
print("test_topk_values[item]", test_topk_values[item])
print("test_topk_indicies[item]", test_topk_indicies[item])
print("test_topk_pred_labels[item]", test_topk_pred_labels[item])

topk_indices_for_item = test_topk_indicies[item].cpu().numpy()
topk_img_paths = np.array(train_img_paths)[topk_indices_for_item]

plt.imshow(test_imgs[item].permute(1,2,0)) # 
# PILImage.fromarray((test_imgs[item].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)).show()
PILImage.open(test_img_paths[item]).show()

def image_grid(img_paths, rows, cols):
    assert len(img_paths) == rows*cols
    imgs = []
    for img_path in img_paths:
        imgs.append(PILImage.open(img_path))

    w, h = imgs[0].size
    grid = PILImage.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

grid = image_grid(topk_img_paths, rows=1, cols=len(topk_img_paths))
grid.show()


# %%
######################################################
# compute accuracy for k=1,3,5
######################################################

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
##############################################################
# plot frequency dist. of classes for top-5 for a given class
##############################################################

def plot_dist_for_class(class_id):
    dist = top5_dist_foreach_class[class_id]
    dist_flat = torch.flatten(dist).numpy()
    count = Counter(dist_flat)
    count_sorted = sorted(count.items(), key=lambda item: item[1], reverse=True)
    count_sorted_freq = [(key, val/len(dist_flat)) for (key, val) in count_sorted]

    # https://stackoverflow.com/questions/61971090/how-can-i-add-images-to-bars-in-axes-matplotlib
    def offset_image(index, label, ax):
        img = PILImage.open(dl.template_imgs[int(label)])
        dpi_factor=3
        img.thumbnail((40*dpi_factor, 40*dpi_factor))

        im = OffsetImage(np.array(img), zoom=1/dpi_factor)

        im.image.axes = ax
        y_offset = -40
        ab = AnnotationBbox(im, (index, 0), xybox=(0, y_offset), frameon=False,
                            xycoords='data', boxcoords="offset points", pad=0)
        ax.add_artist(ab)

    cset = tc.tol_cset('bright')

    fig = plt.figure(figsize = (6, 4), dpi=200)
    ax = fig.add_subplot(111)

    # Customizing grid lines
    plt.grid(color='gray', linestyle=':', linewidth=0.5, zorder=0)
    plt.gca().set_axisbelow(True)  # Ensure grid lines are drawn below the bars

    xs, ys = zip(*count_sorted_freq)
    xs = [str(x_item) for x_item in xs]
    rects1 = ax.bar(xs, ys, color=cset[0])

    # add labels to bars
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.2f'%h,
                    ha='center', va='bottom', fontsize='small')

    autolabel(rects1)

    for i, x in enumerate(xs):
        offset_image(i, x, ax=plt.gca())

    plt.ylim(0, 1.1) 
    ax.set_title(f"distribution top-5, for class ID {class_id}")
    ax.set_ylabel('frequency')
    ax.set_xlabel('class ID', labelpad=50)

    plt.show()

# plot_dist_for_class(0) # boring
plot_dist_for_class(5)
# plot_dist_for_class(10) # boring
plot_dist_for_class(15)
# plot_dist_for_class(20) # boring
plot_dist_for_class(15)
plot_dist_for_class(25)
# plot_dist_for_class(30) # boring
plot_dist_for_class(35)
plot_dist_for_class(40)
plot_dist_for_class(50)
plot_dist_for_class(60)

# %%