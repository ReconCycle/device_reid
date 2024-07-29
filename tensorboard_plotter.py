import os
import cv2
import pandas as pd
import numpy as np
from rich import print
import matplotlib.pyplot as plt
# from IPython.display import display

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tol_colors as tc

cset = tc.tol_cset('bright')

class TensorboardPlotter():
    def __init__(self):
        pass

    def tabulate_events(self, log_path):
        if os.path.isdir(log_path):
            # find matching file
            for file in os.listdir(log_path):
                if "events.out.tfevents" in file:
                    log_path = os.path.join(log_path, file)
                    break

        ea = EventAccumulator(log_path).Reload()
        tags = ea.Tags()['scalars']

        out = {}

        for tag in tags:
            tag_values=[]
            wall_time=[]
            steps=[]

            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                wall_time.append(event.wall_time)
                steps.append(event.step)
                
            out[tag] = pd.DataFrame(tag_values, columns=[tag], index=steps)
            print("tag:", tag)
            # print("out[tag]", out[tag])
            # out[tag].plot()
        return out

    def plot(self, tensorboard_out, keys=None, labels=None, xlabel="epoch", ylabel="loss", is_show=True, save_path=None):
        out_subset = dict((k, tensorboard_out[k]) for k in keys if k in tensorboard_out)
        df = pd.concat(out_subset.values())

        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        for idx, key in enumerate(keys):
            label = key
            if labels is not None and len(labels) == len(keys):
                label = labels[idx]

            if label == "val/seen_val/loss_epoch":
                print(f"renaming {label} to validation")
                label = "val"
                
            elif label == "train/loss_epoch":
                print(f"renaming {label} to train")
                label = "train"

            elif label == "val/seen_val/acc_epoch":
                print(f"renaming {label} to validation")
                label = "val"
            elif label == "train/acc_epoch":
                print(f"renaming {label} to train")
                label = "train"
            
            ax.plot(df[key], label=label, color=cset[idx])
        
        if ylabel == "acc":
            plt.legend(loc="upper left")
        else:
            plt.legend(loc="upper right")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if is_show:
            print("is_show", is_show)
            plt.show()

        if save_path is not None:
            plt.savefig(save_path)

    def auto_plot(self, tensorboard_out, is_show=True, save_path=None):
        # TODO: get all tags ending with loss_epoch, and plot them together
        if save_path is not None and save_path is os.path.isdir(save_path):
            raise ValueError(f"save_path should be a directory but isn't.")
            
        print("tensorboard_out.keys()", tensorboard_out.keys())

        loss_epoch_keys = []
        for key in tensorboard_out.keys():
            if key.endswith("loss_epoch"):
                loss_epoch_keys.append(key)

        print("loss_epoch_keys", loss_epoch_keys)
        if len(loss_epoch_keys) > 0:
            save_filepath = None
            if save_path is not None:
                save_filepath=os.path.join(save_path, "epoch_loss.png")
            self.plot(tensorboard_out, loss_epoch_keys, is_show=is_show, save_path=save_filepath)


        acc_epoch_keys = []
        for key in tensorboard_out.keys():
            if key.endswith("acc_epoch"):
                acc_epoch_keys.append(key)

        print("acc_epoch_keys", acc_epoch_keys)
        if len(acc_epoch_keys) > 0:
            save_filepath = None
            if save_path is not None:
                save_filepath=os.path.join(save_path, "epoch_acc.png")
            self.plot(tensorboard_out, acc_epoch_keys, ylabel="acc", is_show=is_show, save_path=save_filepath)

if __name__ == '__main__':
    tplot = TensorboardPlotter()

    log_dir = os.path.expanduser("~/device_reid/results/2024-07-03__08-33_classify/lightning_logs/version_0")

    tensorboard_out = tplot.tabulate_events(log_dir)
    tplot.auto_plot(tensorboard_out, is_show=False, save_path=log_dir)