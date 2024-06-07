import sys
import os
import cv2
import numpy as np
from rich import print
import commentjson
from PIL import Image
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from tqdm import tqdm
import click
import shutil
from natsort import os_sorted
from pathlib import Path
print(sys.path)

sys.path.insert(1, os.path.join(sys.path[0], '../vision_pipeline')) # for yolact_pkg

from vision_pipeline.object_reid import ObjectReId
from vision_pipeline.object_detection_model import ObjectDetectionModel
from vision_pipeline.object_detection import ObjectDetection
from vision_pipeline.config import load_config

from vision_pipeline.llm_data_generator.labelme_importer import LabelMeImporter

from context_action_framework.types import Detection, Label, Module, Camera


class Main():
    def __init__(self) -> None:

        self.ignore_labels = ["wires"]
        
        self.labelme_importer = LabelMeImporter(ignore_labels=self.ignore_labels)
                    
        self.run()

    def run(self):
        dataset_dir = os.path.expanduser("~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted")
        cropped_dir = os.path.expanduser("~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted_cropped_retry")

        if os.path.isdir(cropped_dir) and os.listdir(cropped_dir):
            print("[red]processed dir already exists!")
            if click.confirm(f"Do you want to delete {cropped_dir}?", default=True):
                shutil.rmtree(cropped_dir)
            else:
                sys.exit()
                return

        # make the processed_dir directory    
        os.makedirs(cropped_dir, exist_ok=True)
        
        subfolders = [ (f.path, f.name) for f in os.scandir(dataset_dir) if f.is_dir() ]
        subfolders = os_sorted(subfolders)


        for sub_path, sub_name in (pbar := tqdm(subfolders)):
            pbar.set_description(f"{sub_name}")
            print("sub_path", sub_path)
            print("sub_name", sub_name)
            # files = [f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f)) and os.path.join(sub_path, f).endswith(('.png', '.jpg', '.jpeg'))]

            os.mkdir(os.path.join(cropped_dir, sub_name))

            for img_path, detections, graph_relations, module, camera, batch_crop_imgs in self.labelme_importer.process_labelme_dir(sub_path, use_yield=True):
                # save create cropped image
                stem = Path(img_path).stem

                # if not cropped_img_path.is_file() and cropped_img is not None:
                if batch_crop_imgs is not None:
                    for idx, crop_img in enumerate(batch_crop_imgs):
                        if len(batch_crop_imgs) == 1:
                            cropped_img_name = str(stem) + "_crop.jpg"  
                        else:
                            cropped_img_name = str(stem) + f"_crop_{idx}.jpg"  
                
                        cropped_img_path = os.path.join(cropped_dir, sub_name, cropped_img_name)
                        print(f"[green]saving crop {cropped_img_name}")
                        cv2.imwrite(str(cropped_img_path), crop_img)
                else:
                    print(f"[red]ERROR: crop is None for {img_path}")


                # break #! debug

            # break #! debug




    
if __name__ == '__main__':
    main = Main()
