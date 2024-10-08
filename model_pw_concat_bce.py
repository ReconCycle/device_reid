import sys
import os
import numpy as np
# import json
from rich import print
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy

import exp_utils as exp_utils

from data_loader_even_pairwise import DataLoaderEvenPairwise


# define the LightningModule
class PwConcatBCEModel(pl.LightningModule):
    def __init__(self, batch_size, learning_rate, weight_decay, backbone=None):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.backbone = backbone

        self.accuracy = BinaryAccuracy().to(self.device)

        self.test_datasets = None
        self.val_datasets = None

        # TODO: make model better
        # TODO: backbone changes what model we want...
        if backbone.type == "resnet_imagenet":
            self.model = nn.Sequential(
                nn.Linear(2 * 2048, 256),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(256, 1)
            )
        elif backbone.type == "clip":
            #! WIP: something like this. but what is the layer size from clip backbone?
            self.model = nn.Sequential(
                nn.Linear(2048, 256),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(256, 1)
            )

        elif backbone.type == "superglue":
            self.model = nn.Sequential(
                    # (1, 65+65, 50, 50)
                    nn.Conv2d(130, 64, kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                    nn.Flatten(),
                    nn.Linear(1875, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1)
                    # nn.LeakyReLU(),
                    # nn.Linear(1024, 256),
                    # nn.LeakyReLU(),
                    # nn.Linear(256, 64),
                    # nn.LeakyReLU(),
                    # nn.Linear(64, 1)
                    )
        
        # https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html
        # ? maybe I should end with F.log_softmax(out, dim=1)
        # ? then do: loss = F.nll_loss(logits, y)

    def forward(self, sample1, sample2):

        x1 = self.backbone.forward(sample1)
        x2 = self.backbone.forward(sample2)
        
        # print("x1.requires_grad", x1.requires_grad)
        
        x = torch.cat((x1, x2), 1) # shape: (1, 130, 50, 50)
        
        # print("x.requires_grad", x.requires_grad)
        # print("x.shape", x.shape)
        
        x_out = self.model(x) #! CHECK IF THIS HAS GRAD
        
        # print("x_out.requires_grad", x_out.requires_grad)
        
        return x_out
    
    def run(self, label1, label2, sample1, sample2):
        # ground_truth = (label1 == label2)
        ground_truth = (label1 == label2).float()
        ground_truth = torch.unsqueeze(ground_truth, 1)
        
        # run the forward step
        x_out = self(sample1, sample2)
        
        criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and BCELoss in one class
        loss = criterion(x_out, ground_truth)
        acc = self.accuracy(x_out, ground_truth)

        return loss, acc

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
    
        sample1, label1, dets1, sample2, label2, dets2 = batch
        
        loss, acc = self.run(label1, label2, sample1, sample2)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train/acc', acc, on_step=True, on_epoch=True, batch_size=self.batch_size)

        return loss

    def evaluate(self, batch, name, stage=None):
        sample1, label1, dets1, sample2, label2, dets2 = batch
        
        loss, acc = self.run(label1, label2, sample1, sample2)

        self.log(f"{stage}/{name}/loss_epoch", loss, on_step=False, on_epoch=True, batch_size=self.batch_size, add_dataloader_idx=False)
        self.log(f"{stage}/{name}/acc_epoch", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size, add_dataloader_idx=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        name = dataloader_idx + 1
        if self.val_datasets is not None:
            name = self.val_datasets[dataloader_idx]
        
        self.evaluate(batch, name, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        name = dataloader_idx + 1
        if self.test_datasets is not None:
            name = self.test_datasets[dataloader_idx]

        self.evaluate(batch, name, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=self.weight_decay)
        return optimizer
