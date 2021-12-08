"""
Author: Benny
Date: Nov 2019
"""

import os
'''HYPER PARAMETER'''
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('/home/luben/robot-peg-in-hole-task')
import torch
import open3d as o3d
from torch.utils.data import DataLoader
import numpy as np

import datetime
import logging
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm

import mankey.provider as provider
import mankey.config.parameter as parameter
from mankey.models.utils import compute_rotation_matrix_from_ortho6d

focal_length = 309.019
principal = 128
factor = 1
intrinsic_matrix = np.array([[focal_length, 0, principal],
                             [0, focal_length, principal],
                             [0, 0, 1]], dtype=np.float64)
class PointnetMover(object):
    def __init__(self, date, model_name='pointnet2_reg_msg', checkpoint_name='best_model_e_185.pth',use_cpu=False, out_channel=9):
        '''MODEL LOADING'''
        exp_dir = '/home/luben/robot-peg-in-hole-task/mankey/log/regression/' + date
        model = importlib.import_module('mankey.models.' + model_name)
        self.network = model.get_model(out_channel, normal_channel=False)
        self.network.apply(self.inplace_relu)
        self.use_cpu = use_cpu
        if not self.use_cpu:
            self.network = self.network.cuda()
        try:
            checkpoint = torch.load(exp_dir + '/checkpoints/'+checkpoint_name)
            # print(checkpoint['epoch'])
            self.network.load_state_dict(checkpoint['model_state_dict'])
            print('Loading model successfully !')
        except:
            print('No existing model...')
            assert False


    def depth_2_pcd(self, depth, factor, K):
        xmap = np.array([[j for i in range(depth.shape[0])] for j in range(depth.shape[1])])
        ymap = np.array([[i for i in range(depth.shape[0])] for j in range(depth.shape[1])])

        if len(depth.shape) > 2:
            depth = depth[:, :, 0]
        mask_depth = depth > 1e-6
        choose = mask_depth.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 1:
            return None

        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        pt2 = depth_masked / factor
        cam_cx, cam_cy = K[0][2], K[1][2]
        cam_fx, cam_fy = K[0][0], K[1][1]
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        pcd = np.concatenate((pt0, pt1, pt2), axis=1)

        return pcd, choose

    def inplace_relu(self, m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace=True

    def process_raw(self, depth):
        xyz, choose = self.depth_2_pcd(depth, factor, intrinsic_matrix)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        down_pcd = pcd.uniform_down_sample(every_k_points=8)
        points = np.asarray(down_pcd.points).astype(np.float32)
        points = points[:8000, :]
        points = np.expand_dims(points, axis=0)
        points = provider.normalize_data(points)
        points = torch.Tensor(points)
        if not self.use_cpu:
            points = points.cuda()
        points = points.transpose(2, 1)
        return points  # 1 x C x N

    def inference(self, depth):
        self.network = self.network.eval()
        with torch.no_grad():
            points = self.process_raw(depth)
            pred, _ = self.network(points)
            delta_rot_pred_6d = pred[:, 0:6]
            delta_rot_pred = compute_rotation_matrix_from_ortho6d(delta_rot_pred_6d, self.use_cpu) # batch*3*3
            delta_xyz_pred = pred[:, 6:9].view(-1,3) # batch*3

        return delta_rot_pred[0], delta_xyz_pred[0]

