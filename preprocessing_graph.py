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

from graph_relations import GraphRelations #! USE THE REAL ONE

sys.path.insert(1, os.path.join(sys.path[0], '../vision_pipeline')) # for yolact_pkg

from vision_pipeline.object_reid import ObjectReId
from vision_pipeline.object_detection_model import ObjectDetectionModel
from vision_pipeline.object_detection import ObjectDetection
from vision_pipeline.config import load_config

from context_action_framework.types import Detection, Label, Module, Camera #! USE THE REAL ONE

# from data_loader_even_pairwise import DataLoaderEvenPairwise
# from data_loader import DataLoader


class Main():
    def __init__(self) -> None:

        self.ignore_labels = []

        # load config
        self.config = load_config("../vision_pipeline/config.yaml")
        self.config.obj_detection.yolov8_model_file = "/home/sruiz/projects/reconcycle/vision_pipeline/data_limited/yolov8/output_2023-08-25_20000/best.pt" # fix relative path     

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

        model = ObjectDetectionModel(self.config.obj_detection)
        
        self.object_detection = ObjectDetection(self.config, 
                                                self.camera_config,
                                                model=model,
                                                object_reid=None,
                                                camera=Camera.basler,
                                                frame_id="",
                                                use_ros=False)
                    

        self.run()


    def run(self):
        dataset_dir = "/home/sruiz/datasets2/reconcycle/2023-12-04_fire_alarms_sorted"
        preprocessing_dir = "experiments/datasets/2023-02-20_hca_backs_preprocessing_opencv2_ASDFGLKLK"

        # dl = DataLoader(img_dir,
        #                 shuffle=False,
        #                 seen_classes=["hca_0", "hca_1", "hca_2", "hca_2a", "hca_3", "hca_4", "hca_5", "hca_6"],
        #                 unseen_classes=["hca_7", "hca_8", "hca_9", "hca_10", "hca_11", "hca_11a", "hca_12"],
        #                 cuda=self.cuda)
        
        
        #! DO THE SAME AS I DO IN data_loader.py
        subfolders = [ (f.path, f.name) for f in os.scandir(dataset_dir) if f.is_dir() ]
        for sub_path, sub_name in subfolders:
            print("sub_name", sub_name)
            files = [f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f)) and os.path.join(sub_path, f).endswith(('.png', '.jpg', '.jpeg'))]

            for file in files:
                print("file", file)
                base_filename = os.path.splitext(file)[0]
                print("base_filename", base_filename)
                labelme_file = base_filename + ".json"
                if os.path.isfile(os.path.join(sub_path, labelme_file)):

                    print("labelme exists!")
                    sample_crop = self.crop_img(sub_path, file, labelme_file)

                    if sample_crop is not None:
                        print("sample_crop.shape", sample_crop.shape)
                    else:
                        print(f"[red]ERROR: {os.path.join(sub_path, labelme_file)}")

                else:
                    print(f"[red]{sub_path}, {labelme_file}, MISSING!")


                break
            

    def crop_img(self, sub_path, img_file, labelme_file):

        sample = cv2.imread(os.path.join(sub_path, img_file))
        print("sample.shape", sample.shape)
        
        try:
            with open(os.path.join(sub_path, labelme_file), 'r') as json_file:
                labelme = jsonpickle.decode(json_file.read(), keys=True)
                
        except ValueError as e:
            print("couldn't read json file properly: ", e)
        
        # TODO: convert labelme to detections!
        detections, graph_relations = self.labelme_to_detections(labelme, sample)

        # graph = GraphRelations(detections)

        # form groups, adds group_id property to detections
        # graph.make_groups()
        
        labels = [Label.hca_front, Label.hca_back, Label.firealarm_front, Label.firealarm_back]
        sample_crop, poly = ObjectReId.find_and_crop_det(sample, graph_relations, labels=labels)

        return sample_crop

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
                    detection.score = float(1.0)

                    detection.mask_contour = self.points_to_contour(shape['points'])
                    detection.box_px = self.contour_to_box(detection.mask_contour)

                    mask = np.zeros((img_h, img_w), np.uint8)
                    cv2.drawContours(mask, [detection.mask_contour], -1, (255), -1)
                    detection.mask = mask
                    
                    detections.append(detection)
                    idx += 1

        detections, markers, poses, graph_img, graph_relations, fps_obb = self.object_detection.get_detections(detections, depth_img=None, worksurface_detection=None, camera_info=None)

        return detections, graph_relations


    def points_to_contour(self, points):
        obj_point_list =  points # [(x1,y1),(x2,y2),...]
        obj_point_list = np.array(obj_point_list).astype(int) # convert to int
        # obj_point_list = [tuple(point) for point in obj_point_list] # convert back to list of tuples

        # contour
        return obj_point_list

    def contour_to_box(self, contour):
        x,y,w,h = cv2.boundingRect(contour)
        # (x,y) is the top-left coordinate of the rectangle and (w,h) its width and height
        box = np.array([x, y, x + w, y + h]).reshape((-1,2)) # convert tlbr (top left bottom right)
        return box


    
if __name__ == '__main__':
    main = Main()
