
# TODO: classification model
# example: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb#scrollTo=bQ-YEaU6-f6h

import sys
import os
import numpy as np
# import json
from rich import print
import torch
from torch import optim, nn, utils, Tensor
import torchvision
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.nn import functional as F

import device_reid.exp_utils as exp_utils

# define the LightningModule
class ClassifyModel(pl.LightningModule):
    def __init__(self, num_classes, batch_size, learning_rate, weight_decay, freeze_backbone=True, labels=None, visualise=False):
        super().__init__()
        self.save_hyperparameters() # save paramaters (matching_config) to checkpoint

        self.labels = labels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_backbone = freeze_backbone
        self.visualise = visualise
        self.test_datasets = None
        self.val_datasets = None

        # self.backbone_model = torchvision.models.resnet18(pretrained=True).to(self.device)
        self.backbone_model = torchvision.models.resnet50(pretrained=True).to(self.device)

        self.backbone_model = torch.nn.Sequential(*(list(self.backbone_model.children())[:-1]))
        
        # todo: remove last two layers:
        #   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        #   (fc): Linear(in_features=2048, out_features=1000, bias=True)

        # we freeze the backbone like this:
        # https://stackoverflow.com/questions/63785319/pytorch-torch-no-grad-versus-requires-grad-false
        if self.freeze_backbone:
            print("[red]Freezing backbone[/red]")
            for param in self.backbone_model.parameters():
                param.requires_grad = False
        
        # n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.backbone_model(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def backbone(self, sample):
        # print("sample", sample.shape) # shape (batch, 3, 400, 400)
        out = self.backbone_model(sample) # shape (batch, 1000)

        # print("out.shape", out.shape)
        # out = torch.squeeze(out) # shape(batch, 2048)


        return out

    def forward(self, sample):

        # print("sample.shape", sample.shape)
        
        x = self.backbone(sample)

        # print("x.shape", x.shape)

        x = x.view(x.size(0), -1)

        # print("x.shape", x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        # print("x.shape", x.shape)

        return x


    def training_step(self, batch, batch_idx):
        sample, label, *_ = batch
        
        logits = self(sample)
        loss = F.nll_loss(logits, label)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, label)

        # loss = self.criterion(logits, label)

        # visualise = False
        # if batch_idx == 0 and self.visualise:
        #     visualise = True

        # acc = self.accuracy(out, label, visualise, train=True)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train/acc', acc, on_step=True, on_epoch=True, batch_size=self.batch_size)

        if batch_idx == 99:
            print("train loss:", loss)
            print("train acc:", acc)

        return loss

    def evaluate(self, batch, name, stage=None):
        sample, label, *_  = batch

        # print("evaluate sample.shape", sample.shape, sample.get_device())
        
        logits = self(sample)
        loss = F.nll_loss(logits, label)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, label)

        print(f"{stage}/{name} loss:", loss.detach().cpu().numpy())
        print(f"{stage}/{name} acc:", acc.detach().cpu().numpy())

        self.log(f"{stage}/{name}/loss_epoch", loss, on_step=False, on_epoch=True, batch_size=self.batch_size, add_dataloader_idx=False)
        self.log(f"{stage}/{name}/acc_epoch", acc, on_step=False, on_epoch=True, batch_size=self.batch_size, add_dataloader_idx=False)

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
