import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from ruamel import yaml


def get_data_dicts(data_dir):
    anno_file_path = os.path.join(data_dir, 'processed/peg_in_hole.yaml')
    img_folder_path = os.path.join(data_dir, 'processed/images')
    with open(anno_file_path, 'r') as f:
        anno_list = yaml.load(f, Loader=yaml.RoundTripLoader)

    dataset_dicts = []
    for idx in range(len(anno_list)):
        filename = os.path.join(img_folder_path, anno_list[idx]["rgb_image_filename"])
        # height, width = cv2.imread(filename).shape[:2]
        record = {}
        record['file_name'] = filename
        record["image_id"] = idx
        record["height"] = 256
        record["width"] = 256
        bbox_top_left = anno_list[idx]['bbox_top_left_xy']
        bbox_bottom_right = anno_list[idx]['bbox_bottom_right_xy']
        objs = []
        obj = {
            "bbox": [bbox_top_left[0], bbox_top_left[1], bbox_bottom_right[0], bbox_bottom_right[1]],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [],
            "category_id": 0,
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


data_dir = '/Users/cmlab/data/pdc/logs_proto/box_insertion_fix_2_small'
d = data_dir
DatasetCatalog.register("box_insertion", lambda d=d: get_data_dicts(d))
MetadataCatalog.get("box_insertion").set(thing_classes=["box_insertion"])
box_insertion_metadata = MetadataCatalog.get("box_insertion")

