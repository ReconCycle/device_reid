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
import regex as re
from tqdm import tqdm
from rich import print
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
from dataset_rotations import ImageDatasetRotations


def random_seen_unseen_class_split(img_path, seen_split=0.5):
    classes = [ f.name for f in os.scandir(img_path) if f.is_dir() and len(os.listdir(f)) > 0 ]
    os_sorted(classes)
    random.seed(42)
    random.shuffle(classes)
    split_len = int(np.rint(len(classes)*seen_split))

    seen_classes = os_sorted(classes[:split_len])
    unseen_classes = os_sorted(classes[split_len:])
    return seen_classes, unseen_classes

class ImageDataset(datasets.ImageFolder):
    def __init__(self,
                 main_path,
                 class_dirs=[],
                 unseen_class_offset=0,
                 transform=None,
                 exemplar_transform=None,
                 limit_imgs_per_class=None):
        
        self.main_path = main_path

        print("main_path", main_path)
        
        self.class_dirs = class_dirs
        self.unseen_class_offset = unseen_class_offset
        self.exemplar_transform = exemplar_transform
        
        if limit_imgs_per_class is not None:
            print("\n"+"="*20)
            print("Imgs per class limited to", limit_imgs_per_class)
            print("="*20, "\n")
            # to make sure each class is of approximately the same size, we set the number of images per class to 30
            def is_valid_file(file_path):
                file_name = os.path.basename(file_path)
                num = int(re.findall(r'\d+', file_name)[-1])
                
                if num < limit_imgs_per_class:
                    return True
                else:
                    return False
        else:
            is_valid_file = None
        
        
        super(ImageDataset, self).__init__(main_path, transform, is_valid_file=is_valid_file)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, label) where target is class_index of the target class.
        """
        path, label = self.samples[index]

        sample = cv2.imread(path)
        
        # convert to rgb if it is an rgbd image
        # sample = np.array(sample)
        # sample = sample[:, :, :3]
        # sample = Image.fromarray(sample)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        label = label + self.unseen_class_offset
        
        # apply albumentations transform
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB) #! do we need this??
        if isinstance(self.transform, A.Compose):
            sample = self.transform(image=sample)["image"]
        else:
            sample_pil = Image.fromarray(sample)
            sample = self.transform(sample_pil)

        # if needed we could pass detections and original image too
        # return sample, label, path, poly
        return sample, label, path, []
        
    # restrict classes to those in subfolder_dirs
    def find_classes(self, dir: str):
        if os.path.isdir(dir):
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in self.class_dirs]
            classes.sort()
        else:
            print("[red]not a dir!", dir)
            classes = []
        # print("Directories in this dataset:", classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    
class DataLoader():
    def __init__(self,
                 img_path,
                #  preprocessing_path=None,
                 batch_size=16,
                 num_workers=8,
                 validation_split=.1,
                 combine_val_and_test=False,
                 shuffle=True,
                 seen_classes=[],
                 unseen_classes=[],
                 train_transform=None,
                 val_transform=None,
                 limit_imgs_per_class=None,
                 cuda=True,
                 use_dataset_rotations=False):
        
        random_seed = 42
        np.random.seed(random_seed)
        
        self.img_path = img_path
        self.batch_size = batch_size
        
        # use seen_classes and unseen_classes to specify which directories to load
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes

        self.seen_dirs = seen_classes
        self.unseen_dirs = unseen_classes

        self.dataloader_imgs = self # hack for being able to call dataloader_imgs.visualise

        ChosenDatasetClass = ImageDataset
        if use_dataset_rotations:
            print("[red]*WARNING* Using use_dataset_rotations True")
            ChosenDatasetClass = ImageDatasetRotations
                
        #! note the only difference is the transform!
        train_tf_dataset = ChosenDatasetClass(self.img_path,
                                    class_dirs=self.seen_dirs,
                                    transform=train_transform,
                                    limit_imgs_per_class=limit_imgs_per_class)
    
        val_tf_dataset = ChosenDatasetClass(self.img_path,
                                    class_dirs=self.seen_dirs,
                                    transform=val_transform,
                                    limit_imgs_per_class=limit_imgs_per_class)
        
        all_labels = [x[1] for x in train_tf_dataset.samples]
        num_seen_classes = len(set(all_labels))
        print("num_seen_classes", num_seen_classes)

        #####################################
        # create list of template images
        #####################################

        template_imgs = [None] * num_seen_classes
        for _img_path, _label in train_tf_dataset.samples:
            if "template" in _img_path:
                template_imgs[_label] = _img_path

        # if a random image if some classes don't have a template image
        for _img_path, _label in train_tf_dataset.samples:
            if template_imgs[_label] is None:
                template_imgs[_label] = _img_path

        self.template_imgs = template_imgs

        #####################################
        # create seen train/val/test split
        # we use pandas sample to get even samples in the val and test set
        #####################################
        df = pd.DataFrame(data=all_labels, columns=["labels"])

        def round_to_multiple(number, multiple):
            return int(multiple * round(number / multiple))

        # approximate group size based on validation_split
        grouped_size = (df.size * 2 * validation_split) / len(self.seen_dirs)
        grouped_size = round_to_multiple(grouped_size, 2)

        print("grouped_size", grouped_size)

        df_test_sampled = df.groupby('labels').sample(n=int(grouped_size/2), random_state=random_seed)
        df2 = df.drop(df_test_sampled.index.values)
        seen_test_idxs = df_test_sampled.index.values
        df_val_sampled = df2.groupby('labels').sample(n=int(grouped_size/2), random_state=random_seed)
        df3 = df2.drop(df_val_sampled.index.values)

        seen_train_idxs = df3.index.values
        seen_val_idxs = df_val_sampled.index.values

        if combine_val_and_test:
            seen_test_idxs = np.concatenate((seen_test_idxs, seen_val_idxs), axis=0)
            seen_val_idxs = []
        #####################################
        # end split
        #####################################
        
        #! --- START OLD CODE
        # len_seen_train = int((1.0 - 2*validation_split) * len_dataset) # 0.8/0.1/0.1 split
        # len_seen_val = int(validation_split * len_dataset)
        # len_seen_test = len_dataset - len_seen_train - len_seen_val

        # generator = torch.Generator().manual_seed(random_seed)
        # seen_train_idxs, seen_val_idxs, seen_test_idxs = torch.utils.data.random_split(
        #     np.arange(len_dataset),
        #     (len_seen_train, len_seen_val, len_seen_test),
        #     generator=generator
        # )
        #! --- END OLD CODE

        self.datasets = {}
        self.datasets["seen_train"] = torch.utils.data.Subset(train_tf_dataset, seen_train_idxs)
        if not combine_val_and_test:
            self.datasets["seen_val"] = torch.utils.data.Subset(val_tf_dataset, seen_val_idxs)
        else:
            self.datasets["seen_val"] = None

        self.datasets["seen_test"] = torch.utils.data.Subset(val_tf_dataset, seen_test_idxs)

        # add unseen dataset
        if len(self.unseen_dirs) > 0:
            self.datasets["unseen_test"] = ChosenDatasetClass(self.img_path,
                                                    class_dirs=self.unseen_dirs,
                                                    unseen_class_offset=len(train_tf_dataset.classes),
                                                    transform=val_transform,
                                                    limit_imgs_per_class=limit_imgs_per_class)
            
        else:
            self.datasets["unseen_test"] = None
        
        # concat seen_test and unseen_test datasets
        if len(self.unseen_dirs) > 0:
            self.datasets["test"] = torch.utils.data.ConcatDataset([self.datasets["seen_test"], self.datasets["unseen_test"]])
        else:
            self.datasets["test"] = self.datasets["seen_test"]
        
        dataset_list = []
        dataset_name_list = []
        dataset_name_list_all = ["seen_train", "seen_val", "seen_test", "test", "unseen_test"]
        
        for dataset_name in dataset_name_list_all:
            if self.datasets[dataset_name] is not None:
                dataset_name_list.append(dataset_name)
                dataset_list.append(self.datasets[dataset_name])

        self.datasets["all"] = torch.utils.data.ConcatDataset([dataset_list])

        # create the dataloaders
        # todo: fix bug, either requiring: generator=torch.Generator(device='cuda'),
        # todo: or requiring shuffle=False
        generator = None
        if cuda:
            generator = torch.Generator(device='cuda')
            
        
        # def custom_collate(instances):
        #     # handle sample, label, and path like normal
        #     # but handle detections as list of lists.
            
        #     # elem = instances[0] # tuple: (sample, label, path, detections)
            
        #     batch = []
            
        #     for i in range(len(instances[0])):
        #         batch.append([instance[i] for instance in instances])

        #     # print("batch[0]", len(batch[0]), type(batch[0][0]))
            
        #     # apply default collate for: sample, label, path
        #     for i in range(len(batch)):
        #         if i != 3:
        #             batch[i] = torch.utils.data.default_collate(batch[i])
        #     # batch[1] = torch.utils.data.default_collate(batch[1])
        #     # batch[2] = torch.utils.data.default_collate(batch[2])

        #     return batch
        
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                                           num_workers=num_workers,
                                                           batch_size=batch_size,
                                                        #    generator=generator,
                                                           shuffle=shuffle,
                                                        #    collate_fn=custom_collate
                                                           )
                            for x in dataset_name_list}
        
        self.dataset_lens = {x: len(self.datasets[x]) for x in dataset_name_list}

        if len(self.unseen_dirs) > 0:
            self.classes = {
                "seen": train_tf_dataset.classes,
                # "seen_train": train_tf_dataset.classes,
                # "seen_val": train_tf_dataset.classes,
                # "seen_test": train_tf_dataset.classes,
                "unseen_test": self.datasets["unseen_test"].classes,
                "test": np.concatenate((train_tf_dataset.classes, self.datasets["unseen_test"].classes)),
                "all": np.concatenate((train_tf_dataset.classes, self.datasets["unseen_test"].classes))
            }
        else:
            self.classes = {
                "seen": train_tf_dataset.classes,
                # "seen_train": train_tf_dataset.classes,
                # "seen_val": train_tf_dataset.classes,
                # "seen_test": train_tf_dataset.classes,
                "test": train_tf_dataset.classes,
                "all": train_tf_dataset.classes
            }
        
    def visualise(self, type_name, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], save_path=None):

        #! This seems broken when dataloader is on cuda!
        dl_iter = iter(self.dataloaders[type_name])
        
        batch = next(dl_iter)
        if len(next(dl_iter)) == 4:
            sample, labels, path, detections = batch
        else:
            sample, labels, path, detections, res_sample, homo_matrix, angle  = batch
        # sample, labels, path, detections = 
            
        # for i, (sample, labels, path, detections) in enumerate(self.dataloaders[type_name]):
            
        if mean is None or std is None:
            samples_denorm = sample
        else:
            mean = np.array(mean)
            std = np.array(std)

            samples_denorm = []
            for sample_img in sample:
                # undo ToTensorV2
                
                # CHW -> HWC
                # Images must be in HWC format, not in CHW
                sample_np = sample_img.cpu().numpy()
                sample_np = sample_np.transpose(1, 2, 0) # W and H might be wrong way around

                # undo normalize
                denorm = A.Normalize(
                    mean=[-m / s for m, s in zip(mean, std)],
                    std=[1.0 / s for s in std],
                    always_apply=True,
                    max_pixel_value=1.0
                )
                sample_denorm = denorm(image=sample_np)["image"]

                # to view with opencv we convert to BGR
                # print("sample_denorm.shape", sample_denorm.shape)
                sample_denorm = cv2.cvtColor(sample_denorm, cv2.COLOR_RGB2BGR)

                # cv2.imshow("sample_denorm", sample_denorm)
                # k = cv2.waitKey(0)

                # we convert back to python format to view in grid
                # HWC -> CHW
                sample_denorm = sample_denorm.transpose(2, 0, 1)
                sample_denorm = torch.from_numpy(sample_denorm)

                samples_denorm.append(sample_denorm)

        # create grid
        img_grid = make_grid(samples_denorm)
        img_grid = transforms.ToPILImage()(img_grid)
        img_grid = np.array(img_grid)

        if save_path is not None:
            cv2.imwrite(os.path.join(save_path, f"train.jpg"), img_grid)

        # show grid
        # cv2.imshow("img_grid", scale_img(img_grid))
        # k = cv2.waitKey(0)
        
        scaled_img_grid = scale_img(img_grid)
        return scaled_img_grid




    def compute_mean_std(self):
        # compute mean and std for training dataset of images. From here the normalisation values can be set.
        # for example:
        # A.Normalize(mean=(154.5703, 148.4985, 152.1174), std=(31.3750, 29.3120, 29.2421), max_pixel_value=1),
        # after setting this, the (mean, std) should be (0, 1)
        
        batches = []
        for sample, label, *_ in tqdm(self.dataloaders["seen_train"]):
            batches.append(sample)
            # print("sample", type(sample), len(sample), type(sample[0]))
        
        all_imgs = torch.cat(batches)
        all_imgs = torch.Tensor.float(all_imgs)
        
        print("all_imgs.shape", all_imgs.shape)
        index_channels = all_imgs.shape.index(3)
        if index_channels != 3:
            # channels are not in the last dimension
            # move the channels to the last dimension:
            all_imgs = torch.transpose(all_imgs, index_channels, -1).contiguous()
        
        if all_imgs.shape[-1] != 3:
            print("\nERROR: the last dimension should be the channels!\n")
        
        print("all_imgs.shape", all_imgs.shape)
        
        all_imgs = all_imgs.view(-1, 3)
        
        print("all_imgs.shape", all_imgs.shape)
        
        min = all_imgs.min(dim=0).values
        max = all_imgs.max(dim=0).values
        
        print("min", min)
        print("max", max)
        
        mean = all_imgs.mean(dim=0)
        std = all_imgs.std(dim=0)
        print("mean", mean)
        print("std", std)
    
    def example_iterate(self, type_name):
        label_distribution = {}
        for a_class in self.classes[type_name]:
            label_distribution[a_class] = 0
        
        for i, (sample, labels, path, detections) in enumerate(self.dataloaders[type_name]):
            # print("i", i)
            # print("inputs.shape", inputs.shape)
            # print("labels", labels)
            # print("")
            # sample[0]

            print(f"len(labels) {len(labels)}")

            img_grid = make_grid(sample, nrow=4)
            img_grid = transforms.ToPILImage()(img_grid)
            # img_grid.show()
            img_grid = np.array(img_grid)
            img_grid = cv2.cvtColor(img_grid, cv2.COLOR_RGB2BGR)
            cv2.imshow("img_grid", scale_img(img_grid))
            k = cv2.waitKey(0)

            # labels_name = [self.classes[type_name][label] for label in labels]

            # for j in np.arange(len(labels)):
            #     label_distribution[labels_name[j]] += 1

        return label_distribution

    def example(self):
        
        print("classes:", self.classes)
        print("")

        label_dist_seen_train = dataloader.example_iterate("seen_train")
        label_dist_seen_val = dataloader.example_iterate("seen_val")

        print("label_dist_seen_train", label_dist_seen_train)
        print("label_dist_seen_val", label_dist_seen_val)


        
        
if __name__ == '__main__':
    print("Run this for testing the dataloader only.")
    img_path = "~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted_cropped"
    
    seen_classes = ["hca_front_00", "hca_front_01", "hca_front_02", "hca_front_03", "hca_front_04", "hca_front_05", "hca_front_06", "hca_front_07"]
    unseen_classes = ["hca_front_08", "hca_front_09", "hca_front_10", "hca_front_11", "hca_front_12", "hca_front_13", "hca_front_14"]

    transform_list = [
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.25, rotate_limit=45, p=0.5),
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.RandomResizedCrop(400, 400, p=0.3),
            A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
    ]

    # computed transform using compute_mean_std() to give:
    # transform_normalise = transforms.Normalize(mean=[0.5895, 0.5935, 0.6036],
                                # std=[0.1180, 0.1220, 0.1092])
    transform_normalise = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255, always_apply=True)
    

    val_transform = A.Compose([
        transform_normalise,
        ToTensorV2(),
    ])

    # add normalise
    transform_list.append(transform_normalise)
    transform_list.append(ToTensorV2())
    train_transform = A.Compose(transform_list)

    dataloader = DataLoader(img_path,
                            seen_classes=seen_classes,
                            unseen_classes=unseen_classes,
                            train_transform=train_transform,
                            val_transform=val_transform,
                            cuda=True)
    
    # dataloader.example_iterate(type_name="seen_train")
    # dataloader.example_iterate(type_name="test")
    dataloader.visualise(type_name="seen_train")
