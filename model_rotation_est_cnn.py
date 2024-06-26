
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

# import device_reid.exp_utils as exp_utils


# TODO: build feed-forward CNN that learns the rotation of images.
# TODO: CODE COPIED FROM CLASSIFY MODEL
# Resources:
# https://stats.stackexchange.com/questions/416915/what-is-a-correct-loss-for-a-model-predicting-angles-from-images


# define the LightningModule
class RotationModel(pl.LightningModule):
    def __init__(self, num_classes, batch_size, learning_rate, weight_decay, loss_type="rot_loss", freeze_backbone=True, labels=None, visualise=False):
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
        self.loss_type = loss_type
        self.nn_version = "v1"

        if self.loss_type == "rot_loss":
            self.n = 1 # 1 for angle
        elif self.loss_type == "sin_cos_loss":
            self.n = 2 # 2 for sin, cos
        elif self.loss_type == "bin_loss":
            self.n = 36 # n bins
            self.accuracy = Accuracy(task="multiclass", num_classes=self.n)

        if self.nn_version == "v1":
            self.head = nn.Sequential(
                nn.Linear(2048 * 2, 512), # *2 because we did concat
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.n) # n bins
            )
        elif self.nn_version == "v2":
            self.head1 = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 128)
            )

            self.head2 = nn.Sequential(
                nn.Linear(128 * 2, 64), # *2 because we did concat
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.n) # n bins
            )

        self.mse_loss = nn.MSELoss()

    
    def rot_difference(self, angle_pred, angle_gt, input_range_minusone_one=True):
        if input_range_minusone_one:
            # ! we want angle_pred in [-1, 1], so we multiple by pi to get to range [-ip, pi]
            angle_pred = angle_pred * np.pi

        # difference of angles
        difference = (angle_pred - angle_gt) % (2*np.pi)
        difference = torch.where(difference > np.pi, difference - 2*np.pi, difference)

        return difference
        

    def rot_loss(self, angle_pred, angle_gt, input_range_minusone_one=True):
        difference = self.rot_difference(angle_pred, angle_gt, input_range_minusone_one)

        # compute MSE
        square_diff = torch.square(difference)
        loss = torch.mean(square_diff)

        return loss
    
    def sin_cos_loss(self, sin_cos_pred, angle_gt):
        # print("sin_cos_pred", sin_cos_pred.shape)
        # # sin_pred, cos_pred = sin_cos_pred

        sin_pred = sin_cos_pred[:, 0] # try with -1
        cos_pred = sin_cos_pred[:, 1] # try with -1
        loss = self.mse_loss(sin_pred, torch.sin(angle_gt)) + self.mse_loss(cos_pred, torch.cos(angle_gt))

        return loss

    def angle_bin(self, angle_gt):
        # angle_gt in [-pi, pi). We translate to [0, 2pi]
        angle_gt_from_zero =  angle_gt + torch.pi 
        bin_gt = torch.div(angle_gt_from_zero, (2* torch.pi)/self.n, rounding_mode='floor').long()
        # bin_gt in [0, 1, ..., self.n-1]
        return bin_gt
    
    def midpoint_bin_angle(self, bin_preds):
        # input: bins, output: the midpoint of the bin.
        midpoint_bin_angle = -torch.pi + (bin_preds + 0.5) * (2 * torch.pi / self.n)

        return midpoint_bin_angle

    def bin_loss(self, logits, bin_gt):
        loss = F.nll_loss(F.log_softmax(logits, dim=1), bin_gt) 

        return loss

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

    def forward(self, sample, res_sample):

        # print("sample.shape", sample.shape)
        
        x1 = self.backbone(sample)
        x2 = self.backbone(res_sample)

        # print("x1.shape after backbone", x1.shape)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        if self.nn_version == "v2":
            x1 = self.head1(x1)
            x2 = self.head1(x2)

        # print("x1.shape after view", x1.shape)

        x_cat = torch.cat((x1, x2), 1) # input: (64, 2048), axis=0 is batch, concatenate along axis=1

        # print("x_cat.shape after cat", x_cat.shape)

        if self.nn_version == "v1":
            x = self.head(x_cat) # linear layers
        elif self.nn_version == "v2":
            x = self.head2(x_cat) # linear layers

        # print("x.shape after head", x.shape)

        return x


    # def run_losses(self, batch, name, stage=None):
    #     sample, label, _, _, res_sample, homo_matrix, angle = batch

    #     logits = self(sample, res_sample)
        
    #     if self.loss_type == "rot_loss":
    #         loss = self.rot_loss(logits.float(), angle.unsqueeze(1).float())
    #     elif self.loss_type == "sin_cos_loss":
    #         loss = self.sin_cos_loss(logits.float(), angle.unsqueeze(1).float())
    #     elif self.loss_type == "bin_loss":
    #         bin_gt = self.angle_bin(angle.float())
    #         loss = self.bin_loss(logits, bin_gt)

    #         bin_preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
    #         acc = self.accuracy(bin_preds, bin_gt)

    #         converted_reg_preds = self.midpoint_bin_angle(bin_preds)
    #         converted_rot_loss = self.rot_loss(converted_reg_preds, angle.float(), input_range_minusone_one=False) # now pred range is -pi to pi

    #     if name == "train":
    #         if self.loss_type == "bin_loss":
    #             self.log('train/acc', acc, on_step=True, on_epoch=True, batch_size=self.batch_size)

    #         self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
    #     else:
    #         if self.loss_type == "bin_loss":
    #             self.log(f"{stage}/{name}/acc_epoch", acc, on_step=False, on_epoch=True, batch_size=self.batch_size, add_dataloader_idx=False)

    #         self.log(f"{stage}/{name}/loss_epoch", loss, on_step=False, on_epoch=True, batch_size=self.batch_size, add_dataloader_idx=False)

    #     return loss


    def training_step(self, batch, batch_idx):
        # self.run_losses(batch, name="train")

        sample, label, _, _, res_sample, homo_matrix, angle = batch

        logits = self(sample, res_sample)
        
        # loss = self.mse_loss(logits.float(), angle.unsqueeze(1).float())

        if self.loss_type == "rot_loss":
            loss = self.rot_loss(logits.float(), angle.unsqueeze(1).float())
        elif self.loss_type == "sin_cos_loss":
            loss = self.sin_cos_loss(logits.float(), angle.unsqueeze(1).float())
        elif self.loss_type == "bin_loss":
            bin_gt = self.angle_bin(angle.float())
            loss = self.bin_loss(logits, bin_gt)

            bin_preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            acc = self.accuracy(bin_preds, bin_gt)

            converted_reg_preds = self.midpoint_bin_angle(bin_preds)
            converted_rot_loss = self.rot_loss(converted_reg_preds, angle.float(), input_range_minusone_one=False) # now pred range is -pi to pi

            self.log('train/acc', acc, on_step=True, on_epoch=True, batch_size=self.batch_size)

            self.log('train/converted_loss', converted_rot_loss, on_step=True, on_epoch=True, batch_size=self.batch_size)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        if batch_idx == 99:
            print("train loss:", loss)

        return loss

    

    def evaluate(self, batch, name, stage=None):
        # self.run_losses(batch, name, stage=stage)
        sample, label, _, _, res_sample, homo_matrix, angle = batch

        logits = self(sample, res_sample)

        if self.loss_type == "rot_loss":
            loss = self.rot_loss(logits.float(), angle.unsqueeze(1).float())
        elif self.loss_type == "sin_cos_loss":
            loss = self.sin_cos_loss(logits.float(), angle.unsqueeze(1).float())
        elif self.loss_type == "bin_loss":
            bin_gt = self.angle_bin(angle.float())
            loss = self.bin_loss(logits, bin_gt)

            bin_preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            acc = self.accuracy(bin_preds, bin_gt)

            converted_reg_preds = self.midpoint_bin_angle(bin_preds)
            converted_rot_loss = self.rot_loss(converted_reg_preds, angle.float(), input_range_minusone_one=False) # now pred range is -pi to pi
        
        if stage == "train":
            if self.loss_type == "bin_loss":
                self.log('train/acc', acc, on_step=True, on_epoch=True, batch_size=self.batch_size)

            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        else:
            if self.loss_type == "bin_loss":
                self.log(f"{stage}/{name}/acc_epoch", acc, on_step=False, on_epoch=True, batch_size=self.batch_size, add_dataloader_idx=False)

                self.log(f"{stage}/{name}/converted_loss_epoch", converted_rot_loss, on_step=False, on_epoch=True, batch_size=self.batch_size, add_dataloader_idx=False)


            self.log(f"{stage}/{name}/loss_epoch", loss, on_step=False, on_epoch=True, batch_size=self.batch_size, add_dataloader_idx=False)


    def manual_eval(self, dataloader):
        differences = []

        # eval model
        self.eval()

        with torch.no_grad():
            for batch in dataloader:
                image0s, labels, paths, _, image1s, homo_matrixes, angle_gt = batch

                image0s = image0s.to(self.device)
                image1s = image1s.to(self.device)
                angle_gt = angle_gt.to(self.device)

                logits = self(image0s, image1s)
                if self.loss_type == "rot_loss":
                    difference = self.rot_difference(logits.float(), angle_gt.unsqueeze(1).float())
                elif self.loss_type == "bin_loss":
                    bin_preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
                    converted_reg_preds = self.midpoint_bin_angle(bin_preds)
                    difference = self.rot_difference(converted_reg_preds, angle_gt.float(), input_range_minusone_one=False) # now pred range is -pi to pi
                else:
                    raise ValueError(f"{self.loss_type} not compatible with manual_eval")

                difference_cpu = difference.squeeze().cpu().numpy()
                differences.extend(difference_cpu)

        # set model back to train
        self.train()

        differences = np.array(differences)

        diff_abs = np.abs(differences)
        diff_square = np.square(differences)

        diff_abs_median = np.median(diff_abs)
        diff_abs_mean = np.mean(diff_abs)
        diff_abs_std = np.std(diff_abs)

        diff_square_median = np.median(diff_square)
        diff_square_mean = np.mean(diff_square)
        diff_square_std = np.std(diff_square)

        return diff_abs_median, diff_abs_mean, diff_abs_std, diff_square_median, diff_square_mean, diff_square_std




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
