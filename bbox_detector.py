import torch, torchvision
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
from detectron2.utils.visualizer import ColorMode

class BboxDetection(object):

    def __init__(self, network_ckpnt_path):
        cfg = get_cfg()
        #cfg.OUTPUT_DIR = '/Users/cmlab/robot-peg-in-hole-task/mrcnn/ckpnt'
        cfg.OUTPUT_DIR = network_ckpnt_path
        cfg.MODEL.DEVICE = 'cpu'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.DATASETS.TRAIN = ("box_insertion",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        self.predictor = DefaultPredictor(cfg)

    def predict(self, cv_rgb=None, cv_rgb_path=''):
        if cv_rgb_path != '':
            cv_rgb = cv2.imread(cv_rgb_path, cv2.IMREAD_COLOR)
        outputs = self.predictor(cv_rgb)
        result = outputs["instances"].get_fields()
        bbox_list = [v.tolist() for v in result['pred_boxes']]
        bbox = [int(v) for v in bbox_list[0]]
        bbox = np.array(bbox)
        return bbox