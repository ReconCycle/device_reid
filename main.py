import sys
import os
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

import argparse

import exp_utils as exp_utils
from exp_utils import str2bool, clip_transform

from data_loader import DataLoader, random_seen_unseen_class_split
from data_loader_even_pairwise import DataLoaderEvenPairwise
from data_loader_triplet import DataLoaderTriplet

from backbone_resnet import BackboneResnet
from backbone_clip import BackboneClip
from backbone_superglue import BackboneSuperglue

from model_pw_concat_bce import PwConcatBCEModel
from model_pw_cos import CosModel
from model_pw_cos2 import Cos2Model

from model_triplet import TripletModel
from model_superglue import SuperGlueModel
from model_sift import SIFTModel
from model_clip import ClipModel
from model_classify import ClassifyModel

from model_rotation_est_cnn import RotationModel


# do as if we are in the parent directory
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
# from action_predictor.graph_relations import GraphRelations
# from vision_pipeline.object_reid import ObjectReId

from superglue_training.models.matching import Matching    


# TODO: get grayscale working for superglue
class Grayscale(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=0.5):
        super(Grayscale, self).__init__(always_apply, p)

    def apply(self, img, **params):
        img_np = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img_np
    
    def get_transform_init_args_names(self):
        return ()


class Main():
    def __init__(self, raw_args=None) -> None:
        exp_utils.init_seeds(1, cuda_deterministic=False)

        parser = argparse.ArgumentParser(
            description='pairwise classifier',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        base_path = os.path.dirname(os.path.abspath(__file__))
        print("base path:", base_path)

        img_path = os.path.expanduser("~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted_cropped")
        results_base_path = os.path.join(base_path, "results/")

        parser.add_argument('--mode', type=str, default="train") # train/eval
        parser.add_argument('--model', type=str, default='triplet')
        parser.add_argument('--superglue_model', type=str, default='indoor')
        # superglue/sift
        # pw_concat_bce/pw_cos/pw_cos2
        # triplet
        parser.add_argument('--backbone', type=str, default=None) # resnet_imagenet, clip, superglue
        parser.add_argument('--visualise', type=str2bool, default=False)
        parser.add_argument('--cutoff', type=float, default=0.01) # SIFT=0.01, superglue=0.5
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--batches_per_epoch', type=int, default=10) #! was 300, for debugging 10
        parser.add_argument('--train_epochs', type=int, default=50)
        parser.add_argument('--eval_epochs', type=int, default=1)
        parser.add_argument('--early_stopping', type=str2bool, default=True)

        parser.add_argument('--freeze_backbone', type=str2bool, default=True)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=0.0)

        parser.add_argument('--img_path', type=str, default=img_path)
        parser.add_argument('--results_base_path', type=str, default=results_base_path)
        parser.add_argument('--seen_classes', type=str, nargs='+', default=[])
        parser.add_argument('--unseen_classes', type=str, nargs='+', default=[])

        # for eval only:
        parser.add_argument('--results_path', type=str, default="")
        parser.add_argument('--checkpoint_path', type=str, default="")

        self.args = parser.parse_args(raw_args)
        
        eval_only_models = ["superglue", "cosine", "clip"]

        if self.args.mode == "train" or \
        (self.args.mode == "eval" and (self.args.model in eval_only_models)):
            # ignore checkpoint
            self.args.checkpoint_path = ""

            # create a new directory when we train
            results_path = os.path.join(self.args.results_base_path, datetime.now().strftime('%Y-%m-%d__%H-%M' + f"_{self.args.model}"))
            if self.args.backbone is not None:
                results_path += f"_{self.args.backbone}"

            if os.path.isdir(results_path):
                print(f"[red]results_path already exists! {results_path}")
                return
            else:
                os.makedirs(results_path)
            self.args.results_path = results_path
        elif self.args.mode == "eval":
            if self.args.results_path is None or self.args.results_path == "":
                split = self.args.checkpoint_path.split("/")
                idx_results = split.index("results")
                newsplit = split[:idx_results+2]
                self.args.results_path = '/'.join(split[:idx_results+2])
                print("[green]set self.args.results_path", self.args.results_path)
        
        torch.set_grad_enabled(True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        if self.args.seen_classes == [] and self.args.unseen_classes == []:
            if self.args.model in ["classify"]:
                self.args.seen_classes, self.args.unseen_classes = random_seen_unseen_class_split(img_path, seen_split=1.0)
            else:
                self.args.seen_classes, self.args.unseen_classes = random_seen_unseen_class_split(img_path)

        self.model_rgb = True
        if self.args.model in ["superglue", "sift"] or self.args.backbone == "superglue":
            self.model_rgb = False

        self.norm_mean = None
        self.norm_std = None

        if self.args.model == "clip" or self.args.backbone == "clip":
            print("\n[blue]Using special CLIP transform!!!!!!!!!!!!!\n")
            self.train_transform = clip_transform(224)
            self.val_transform = clip_transform(224)
        elif self.args.model == "superglue" or self.args.backbone == "superglue":
            # don't normalise. We only evaluate these models
            self.val_transform = A.Compose([A.Resize(300, 300), ToTensorV2()])
            self.train_transform = A.Compose([A.Resize(300, 300), ToTensorV2()])

        elif self.args.model in eval_only_models:
            #! what other eval_only_models do we have?
            # don't normalise. We only evaluate these models
            self.val_transform = A.Compose([A.Resize(300, 300), ToTensorV2()])
            self.train_transform = A.Compose([A.Resize(300, 300), ToTensorV2()])

        else:
            val_tf_list = []
            train_tf_list = [
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.25, rotate_limit=45, p=0.5),
                A.OpticalDistortion(p=0.5),
                A.GridDistortion(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RandomResizedCrop(400, 400, p=0.3),
                A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ]

            # computed transform using compute_mean_std() to give:
            # transform_normalise = A.Normalize(mean=(0.5895, 0.5935, 0.6036), std=(0.1180, 0.1220, 0.1092))
            # imagenet normalize
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229, 0.224, 0.225]
            if not self.model_rgb:
                self.norm_mean = [np.mean(self.norm_mean)]
                self.norm_std = [np.mean(self.norm_std)]
            
            transform_normalise = A.Normalize(mean=self.norm_mean, std=self.norm_std) # imagenet

            # validation transforms
            if not self.model_rgb:
                val_tf_list.append(Grayscale(always_apply=True))

            val_tf_list.append(A.Resize(300, 300)) 
            val_tf_list.append(transform_normalise)
            val_tf_list.append(ToTensorV2())
        
            # train transforms
            if not self.model_rgb:
                train_tf_list.append(Grayscale(always_apply=True))
            
            train_tf_list.append(A.Resize(300, 300)) 
            train_tf_list.append(transform_normalise)    
            train_tf_list.append(ToTensorV2())

            self.val_transform = A.Compose(val_tf_list)
            self.train_transform = A.Compose(train_tf_list)


        # setup backbone
        if self.args.backbone == "resnet_imagenet":
            self.backbone = BackboneResnet(freeze=self.args.freeze_backbone, device=self.device)
        elif self.args.backbone == "clip":
            self.backbone = BackboneClip(freeze=self.args.freeze_backbone, device=self.device)
        elif self.args.backbone == "superglue":
            self.backbone = BackboneSuperglue(model=self.args.superglue_model, freeze=self.args.freeze_backbone, device=self.device)

        # setup model
        if self.args.model == "superglue":
            print("using SuperGlueModel")
            if self.args.mode == "train":
                print("[red]This model is eval only![/red]")
                return
            self.model = SuperGlueModel(self.args.batch_size, model=self.args.superglue_model, visualise=self.args.visualise)
        elif self.args.model == "sift":
            print("using SIFT")
            self.model = SIFTModel(self.args.batch_size, cutoff=self.args.cutoff, visualise=self.args.visualise)

        elif self.args.model == "pw_concat_bce":
            print("using pw_concat_bce model")
            self.model = PwConcatBCEModel(self.args.batch_size, 
                                                 self.args.learning_rate, 
                                                 self.args.weight_decay,
                                                 backbone=self.backbone)
        elif self.args.model == "pw_cos":
            print("using pw_cos model")
            self.model = CosModel(self.args.batch_size, 
                                   cutoff=self.args.cutoff, 
                                   visualise=self.args.visualise)

        elif self.args.model == "pw_cos2":
            print("using pw_cos2 model")
            self.model = Cos2Model(self.args.batch_size,
                                                  self.args.learning_rate,
                                                  self.args.weight_decay,
                                                  cutoff=self.args.cutoff, 
                                                  freeze_backbone=self.args.freeze_backbone,
                                                  visualise=self.args.visualise)
            
        elif self.args.model == "clip":
            self.model = ClipModel(self.args.batch_size, 
                                   cutoff=self.args.cutoff, 
                                   visualise=self.args.visualise)
        elif self.args.model == "triplet":
            print("using triplet")
            self.model = TripletModel(self.args.batch_size,
                                      learning_rate=self.args.learning_rate,
                                      weight_decay=self.args.weight_decay,
                                      cutoff=self.args.cutoff,
                                      freeze_backbone=self.args.freeze_backbone,
                                      visualise=self.args.visualise)

        elif self.args.model in ["classify"]: #! debug
            print("classify model")
            self.model = ClassifyModel(num_classes=len(self.args.seen_classes),
                                      batch_size=self.args.batch_size,
                                      learning_rate=self.args.learning_rate,
                                      weight_decay=self.args.weight_decay,
                                      freeze_backbone=self.args.freeze_backbone,
                                      labels=self.args.seen_classes,
                                      visualise=self.args.visualise)
        elif self.args.model == "rotation":
            self.model = RotationModel(num_classes=len(self.args.seen_classes),
                                      batch_size=self.args.batch_size,
                                      learning_rate=self.args.learning_rate,
                                      weight_decay=self.args.weight_decay,
                                      freeze_backbone=self.args.freeze_backbone,
                                      labels=self.args.seen_classes,
                                      visualise=self.args.visualise)

        
        # setup dataloader
        if self.args.model in ["superglue", "sift", "pairwise_classifier", "pairwise_classifier2", "pairwise_classifier3", "cosine", "clip"]:
            self.dl = DataLoaderEvenPairwise(self.args.img_path,
                                        batch_size=self.args.batch_size,
                                        num_workers=8,
                                        shuffle=True,
                                        seen_classes=self.args.seen_classes,
                                        unseen_classes=self.args.unseen_classes,
                                        train_transform=self.train_transform,
                                        val_transform=self.val_transform)
        elif self.args.model in ["triplet"]:
            self.dl = DataLoaderTriplet(self.args.img_path,
                                        batch_size=self.args.batch_size,
                                        num_workers=8,
                                        shuffle=True,
                                        seen_classes=self.args.seen_classes,
                                        unseen_classes=self.args.unseen_classes,
                                        train_transform=self.train_transform,
                                        val_transform=self.val_transform)
        
        elif self.args.model in ["classify"]:
            self.dl = DataLoader(self.args.img_path,
                                        batch_size=self.args.batch_size,
                                        num_workers=8,
                                        shuffle=True,
                                        seen_classes=self.args.seen_classes,
                                        unseen_classes=self.args.unseen_classes,
                                        train_transform=self.train_transform,
                                        val_transform=self.val_transform)
        elif self.args.model in ["rotation"]:
            # when using use_dataset_rotations, augmentations are applied in the dataloader
            self.dl = DataLoader(self.args.img_path,
                                        batch_size=self.args.batch_size,
                                        num_workers=8,
                                        shuffle=True,
                                        seen_classes=self.args.seen_classes,
                                        unseen_classes=self.args.unseen_classes,
                                        use_dataset_rotations=True
                                        ) # important flag.


        
        logging.basicConfig(filename=os.path.join(self.args.results_path, f'{self.args.mode}.log'), level=logging.DEBUG)
        self.log_args()

        self.dl.dataloader_imgs.visualise("seen_train", mean=self.norm_mean, std=self.norm_std, save_path=self.args.results_path)

        if self.args.mode == "train":
            self.train()
        elif self.args.mode == "eval":
            self.eval()

    
    def log_args(self):
        for arg in vars(self.args):
            print(f"{arg}: {getattr(self.args, arg)}")
            logging.info(f"{arg}: {getattr(self.args, arg)}")

        print(f"device: {self.device}")
        logging.info(f"device: {self.device}")

        print(f"train transforms: {self.train_transform}")
        logging.info(f"train transforms: {self.train_transform}")

        print(f"val transforms: {self.val_transform}")
        logging.info(f"val transforms: {self.val_transform}")

        print(f"model: {self.model}")
        logging.info(f"model: {self.model}")

        # dataset split and size:
        logging.info(f"dataset sizes:\nseen_train: {len(self.dl.dataloaders['seen_train'])}\nseen_val: {len(self.dl.dataloaders['seen_val'])}\nseen_test: {len(self.dl.dataloaders['seen_test'])}\ntest: {len(self.dl.dataloaders['test'])}")

        logging.info(f"dataset lens: {self.dl.dataset_lens}")

        logging.info(f"classes: {self.dl.classes}")

    def train(self):
        
        logging.info(f"starting training...")

        callbacks = [OverrideEpochStepCallback()]
        checkpoint_callback = ModelCheckpoint(monitor="val/seen_val/loss_epoch", mode="min", save_top_k=2)
        callbacks.append(checkpoint_callback)
        if self.args.early_stopping:
            early_stop_callback = EarlyStopping(monitor="val/seen_val/loss_epoch", mode="min", patience=20, min_delta=0.01, verbose=False)
            # early_stop_callback = EarlyStopping(monitor="val/seen_val/acc_epoch", mode="max", patience=20, verbose=False)
            callbacks.append(early_stop_callback)
            

        trainer = pl.Trainer(
            callbacks=callbacks,
            enable_checkpointing=True,
            default_root_dir=self.args.results_path,
            limit_train_batches=self.args.batches_per_epoch,
            limit_val_batches=self.args.batches_per_epoch,
            limit_test_batches=self.args.batches_per_epoch,
            max_epochs=self.args.train_epochs,
            accelerator="gpu",
            devices=1)

        self.model.val_datasets = ["seen_val"]
        trainer.fit(model=self.model, 
                    train_dataloaders=self.dl.dataloaders["seen_train"],
                    val_dataloaders=self.dl.dataloaders["seen_val"])
        
        trainer.save_checkpoint(os.path.join(os.path.dirname(checkpoint_callback.best_model_path), "last.ckpt"))
        
        logging.info(f"best model path: {checkpoint_callback.best_model_path}")
        logging.info(f"best model score: {checkpoint_callback.best_model_score}")
        print(f"best model path: {checkpoint_callback.best_model_path}")
        print(f"best model score: {checkpoint_callback.best_model_score}")

        # immediately run eval
        self.eval(model_path=checkpoint_callback.best_model_path)
        

    def eval(self, model_path=None):
        logging.info(f"[blue]running eval...")

        if model_path is None:
            if self.args.checkpoint_path is None or self.args.checkpoint_path == "":
                print("[red]: provide checkpoint_path")
                return
            model_path = self.args.checkpoint_path
        
        print(f"model_path {model_path}")
        logging.info(f"model_path {model_path}")

        if self.args.model == "pw_concat_bce":
            self.model = PwConcatBCEModel.load_from_checkpoint(model_path, strict=False)
            print("loaded PwConcatBCEModel checkpoint!")
        elif self.args.model == "pw_cos":
            self.model = CosModel.load_from_checkpoint(model_path, strict=False)
            print("loaded CosModel checkpoint!")
        elif self.args.model == "pw_cos2":
            self.model = Cos2Model.load_from_checkpoint(model_path, strict=False)
            print("loaded Cos2Model checkpoint!")
        elif self.args.model == "triplet":
            self.model = TripletModel.load_from_checkpoint(model_path, strict=False)
            print("loaded TripletModel checkpoint!")
        elif self.args.model == "classify":
            self.model = ClassifyModel.load_from_checkpoint(model_path, strict=False)
            print("loaded ClassifyModel checkpoint!")
        elif self.args.model == "rotation":
            print("[red]not implemented.")
        
        trainer = pl.Trainer(callbacks=[OverrideEpochStepCallback()],
                            default_root_dir=self.args.results_path,
                            limit_train_batches=self.args.batches_per_epoch,
                            limit_val_batches=self.args.batches_per_epoch,
                            limit_test_batches=self.args.batches_per_epoch,
                            max_epochs=self.args.eval_epochs,
                            accelerator='gpu',
                            devices=1)

        # test the model
        print("[blue]eval_results:[/blue]")

        if self.args.model == "classify":
            test_datasets = ["seen_train", "seen_val", "seen_test"]
        else:
            test_datasets = ["seen_train", "seen_val", "test"]
        self.model.test_datasets = test_datasets
        output = trainer.test(self.model,
                                  dataloaders=[self.dl.dataloaders[name] for name in test_datasets])
        for i, name in enumerate(test_datasets):
            logging.info(f"eval {name}:" + str(output[i]))
    
        if self.args.model == "classify":
            self.plot_confusion()


    def plot_confusion(self):

        print("plotting confusion matrix...")

        y_pred = []
        y_true = []

        # iterate over test data
        for inputs, labels, *_ in tqdm(self.dl.dataloaders["seen_train"]):
                logits = self.model(inputs) # Feed Network

                preds = torch.argmax(logits, dim=1)
                y_pred.extend(preds) # Save Prediction
                
                labels = labels.data.cpu().numpy()
                y_true.extend(labels) # Save Truth

        # constant for classes
        classes = self.args.seen_classes

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)

        print("classes", len(classes))
        print("y_true", len(y_true))
        print("y_pred", len(y_pred))
        print("cf_matrix", cf_matrix.shape)

        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                            columns = [i for i in classes])
        plt.figure(figsize = (24,14))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join(self.args.results_path, "confusion.png"))

    # def eval_manual(self):
    #     self.results_path = "experiments/results/2023-03-10__15-28-03"
    #     self.checkpoint = "lightning_logs/version_0/checkpoints/epoch=19-step=2000.ckpt"
    #     self.model.load_from_checkpoint(os.path.join(self.results_path, self.checkpoint), strict=False)
    #     self.model.to(self.device)
    #     self.model.eval()
        
    #     for i, (sample1, label1, dets1, sample2, label2, dets2) in enumerate(self.dl.dataloaders["seen_train"]):
    #         sample1 = sample1.to(self.device)
    #         sample2 = sample2.to(self.device)
            
    #         out = self.model(sample1, sample2)
            
    #         print("out", out)

# https://github.com/Lightning-AI/lightning/issues/2110#issuecomment-1114097730
# logging: https://github.com/Lightning-AI/lightning/discussions/6182
class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        
        # print("on_train_epoch_end metrics", metrics)
        train_info = "train (epoch " + str(trainer.current_epoch) + "): "
        if 'train/loss_epoch' in metrics:
            train_info += str(metrics["train/loss_epoch"].cpu().numpy()) + ", "
        
        if 'train/acc_epoch' in metrics:
            train_info += str(metrics["train/acc_epoch"].cpu().numpy())

        logging.info(train_info)
        print(train_info)

        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        # print("on_test_epoch_end metrics", metrics)

        test_info = "test (epoch " + str(trainer.current_epoch) + "): "
        if 'test/test/loss_epoch' in metrics:
            test_info += str(metrics["test/test/loss_epoch"].cpu().numpy()) + ", "
        
        if "test/test/acc_epoch" in metrics:
            test_info += str(metrics["test/test/acc_epoch"].cpu().numpy())

        logging.info(test_info)
        print(test_info)

        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        metrics = trainer.callback_metrics
        # print("on_validation_epoch_end metrics", metrics)

        valid_info = "valid (epoch " + str(trainer.current_epoch) + "): "
        if 'val/seen_val/loss_epoch' in metrics:
            valid_info += str(metrics["val/seen_val/loss_epoch"].cpu().numpy()) + ", "
        
        if "val/seen_val/acc_epoch" in metrics:
            valid_info += str(metrics["val/seen_val/acc_epoch"].cpu().numpy())

        logging.info(valid_info)
        print(valid_info)
    
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", torch.tensor(trainer.current_epoch, dtype=torch.float32))
        

if __name__ == '__main__':
    main = Main()
    