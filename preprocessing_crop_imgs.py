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
print(sys.path)

sys.path.insert(1, os.path.join(sys.path[0], '../vision_pipeline')) # for yolact_pkg

from vision_pipeline.object_reid import ObjectReId
from vision_pipeline.object_detection_model import ObjectDetectionModel
from vision_pipeline.object_detection import ObjectDetection
from vision_pipeline.config import load_config

from context_action_framework.types import Detection, Label, Module, Camera


class Main():
    def __init__(self) -> None:

        self.ignore_labels = ["wires"]

        # load config
        self.config = load_config(os.path.expanduser("~/vision_pipeline/config.yaml"))

        self.config.obj_detection.debug = False #! don't show debug messages
        self.config.obj_detection.yolov8_model_file = os.path.expanduser("~/vision_pipeline/data_limited/yolov8/output_2023-08-25_20000/best.pt") # fix relative path     

        dataset = None        
        self.cuda = True
        
        object_reid = None
        
        # pretend to use Basler camera
        self.camera_type = Camera.basler
        self.camera_name = self.camera_type.name
        
        self.camera_config = self.config.basler
        
        self.camera_config.enable_topic = "set_sleeping" # basler camera specific
        self.camera_config.enable_camera_invert = True # enable = True, but the topic is called set_sleeping, so the inverse
        self.camera_config.use_worksurface_detection = True
        
        self.worksurface_detection = None

        # model = ObjectDetectionModel(self.config.obj_detection)
        model = None
        
        self.object_detection = ObjectDetection(self.config, 
                                                self.camera_config,
                                                model=model,
                                                object_reid=None,
                                                camera=Camera.basler,
                                                frame_id="",
                                                use_ros=False)
                    

        self.run()


    def run(self):
        dataset_dir = os.path.expanduser("~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted")
        cropped_dir = os.path.expanduser("~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted_cropped")

        if os.path.isdir(cropped_dir) and os.listdir(cropped_dir):
            print("[red]processed dir already exists!")
            if click.confirm(f"Do you want to delete {cropped_dir}?", default=True):
                shutil.rmtree(cropped_dir)
            else:
                sys.exit()
                return

        # make the processed_dir directory    
        os.mkdir(cropped_dir)
        
        subfolders = [ (f.path, f.name) for f in os.scandir(dataset_dir) if f.is_dir() ]
        subfolders = os_sorted(subfolders)

        for sub_path, sub_name in (pbar := tqdm(subfolders)):
            print("sub_name", sub_name)
            files = [f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f)) and os.path.join(sub_path, f).endswith(('.png', '.jpg', '.jpeg'))]

            os.mkdir(os.path.join(cropped_dir, sub_name))

            for file in files:
                pbar.set_description(f"{file}")
                base_filename = os.path.splitext(file)[0]
                labelme_file = base_filename + ".json"
                if os.path.isfile(os.path.join(sub_path, labelme_file)):
                    sample_crop = self.crop_img(sub_path, file, labelme_file)

                    if sample_crop is not None:
                        cv2.imwrite(os.path.join(cropped_dir, sub_name, file), sample_crop)

                    else:
                        print(f"[red]ERROR: crop is None {os.path.join(sub_path, labelme_file)}")

                else:
                    print(f"[red]{sub_path}, {labelme_file}, MISSING!")
            

    def crop_img(self, sub_path, img_file, labelme_file):

        sample = cv2.imread(os.path.join(sub_path, img_file))
        
        if sample.shape[0] <= 1450:
            # for hca_front that were taken at a resolution of 1450x1450
            size = 300
        else:
            size = 600

        try:
            with open(os.path.join(sub_path, labelme_file), 'r') as json_file:
                labelme = jsonpickle.decode(json_file.read(), keys=True)
                
        except ValueError as e:
            print("couldn't read json file properly: ", e)
        
        # convert labelme to detections!
        detections, graph_relations = self.labelme_to_detections(labelme, sample)

        labels = [Label.hca_front, Label.hca_back, Label.firealarm_front, Label.firealarm_back]
        sample_crop, poly = ObjectReId.find_and_crop_det(sample, graph_relations, labels=labels, size=size)

        if size == 300:
            # upscale to 600
            print("upscaling!", sample.shape[0], sub_path)
            sample_crop = cv2.resize(sample_crop, (600, 600), interpolation=cv2.INTER_AREA)


        return sample_crop

    #! this function is also used in vision_pipeline/llm_data_generator
    def labelme_to_detections(self, json_data, sample):
        detections = []
        img_h, img_w = sample.shape[:2]    

        idx = 0
        for shape in json_data['shapes']:
            # only add items that are in the allowed
            if shape['label'] not in self.ignore_labels:

                if shape['shape_type'] == "polygon":

                    detection = Detection()
                    detection.id = idx
                    detection.tracking_id = idx

                    detection.label = Label[shape['label']]
                    # print("detection.label", detection.label)
                    detection.score = float(1.0)

                    detection.valid = True

                    detection.mask_contour = self.points_to_contour(shape['points'])
                    detection.box_px = self.contour_to_box(detection.mask_contour)

                    mask = np.zeros((img_h, img_w), np.uint8)
                    cv2.drawContours(mask, [detection.mask_contour], -1, (255), -1)
                    detection.mask = mask
                    
                    detections.append(detection)
                    idx += 1

        detections, markers, poses, graph_img, graph_relations, fps_obb = self.object_detection.get_detections(detections, depth_img=None, worksurface_detection=None, camera_info=None, use_classify=False)

        return detections, graph_relations

    #! this function is also used in vision_pipeline/llm_data_generator
    def points_to_contour(self, points):
        obj_point_list =  points # [(x1,y1),(x2,y2),...]
        obj_point_list = np.array(obj_point_list).astype(int) # convert to int
        # obj_point_list = [tuple(point) for point in obj_point_list] # convert back to list of tuples

        # contour
        return obj_point_list

    #! this function is also used in vision_pipeline/llm_data_generator
    def contour_to_box(self, contour):
        x,y,w,h = cv2.boundingRect(contour)
        # (x,y) is the top-left coordinate of the rectangle and (w,h) its width and height
        box = np.array([x, y, x + w, y + h]).reshape((-1,2)) # convert tlbr (top left bottom right)
        return box


    
if __name__ == '__main__':
    main = Main()
