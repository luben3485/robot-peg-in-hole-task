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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from ruamel import yaml
from detectron2.engine import DefaultTrainer

data_dir = '/tmp2/r09944001/data/pdc/logs_proto/box_insertion_fix_3_2021-06-28'

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
            #"segmentation": [],
            "category_id": 0,
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts



d = data_dir
DatasetCatalog.register("box_insertion", lambda d=d: get_data_dicts(d))
MetadataCatalog.get("box_insertion").set(thing_classes=["box_insertion"])
box_insertion_metadata = MetadataCatalog.get("box_insertion")

cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("box_insertion",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.025  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.OUTPUT_DIR = 'ckpnt_box_insertion_fix_3'
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
