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
class CoarseMover(object):
    def __init__(self, model_path='reg_seg_heatmap_v2/2022-01-18_03-48', model_name='pointnet2_reg_seg_heatmap_v2_msg', checkpoint_name='best_model_e_185.pth',use_cpu=False, out_channel=9):
        '''MODEL LOADING'''
        exp_dir = os.path.join('/home/luben/robot-peg-in-hole-task/mankey/log', model_path)
        model = importlib.import_module('mankey.models.' + model_name)
        self.network = model.get_model(out_channel)
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

    def process_raw_mutliple_camera(self, depth_mm_list, camera2world_list):
        xyz_in_world_list = []
        for idx, depth_mm in enumerate(depth_mm_list):
            depth = depth_mm / 1000  # unit: mm to m
            xyz, choose = self.depth_2_pcd(depth, factor, intrinsic_matrix)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            down_pcd = pcd.uniform_down_sample(every_k_points=8)
            down_xyz_in_camera = np.asarray(down_pcd.points).astype(np.float32)
            down_xyz_in_camera = down_xyz_in_camera[:8000, :]

            # camera coordinate to world coordinate
            down_xyz_in_world = []
            for xyz in down_xyz_in_camera:
                camera2world = np.array(camera2world_list[idx])
                xyz = np.append(xyz, [1], axis=0).reshape(4, 1)
                xyz_world = camera2world.dot(xyz)
                xyz_world = xyz_world[:3] * 1000
                down_xyz_in_world.append(xyz_world)
            xyz_in_world_list.append(down_xyz_in_world)
        concat_xyz_in_world = np.array(xyz_in_world_list)
        concat_xyz_in_world = concat_xyz_in_world.reshape(-1, 3)
        points = np.expand_dims(concat_xyz_in_world, axis=0)
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
            heatmap, pred = self.network(points)
            delta_rot_pred_6d = pred[:, 0:6]
            delta_rot_pred = compute_rotation_matrix_from_ortho6d(delta_rot_pred_6d, self.use_cpu) # batch*3*3
            delta_xyz_pred = pred[:, 6:9].view(-1,3) # batch*3

        return delta_rot_pred[0], delta_xyz_pred[0]

    def inference_multiple_camera(self, depth_mm_list, camera2world_list):
        self.network = self.network.eval()
        with torch.no_grad():
            points = self.process_raw_mutliple_camera(depth_mm_list, camera2world_list)
            heatmap, pred = self.network(points)
            delta_rot_pred_6d = pred[:, 0:6]
            delta_rot_pred = compute_rotation_matrix_from_ortho6d(delta_rot_pred_6d, self.use_cpu) # batch*3*3
            delta_xyz_pred = pred[:, 6:9].view(-1,3) # batch*3

            delta_rot_pred = delta_rot_pred[0].cpu().numpy()
            delta_xyz_pred = delta_xyz_pred[0].cpu().numpy()
        return delta_rot_pred, delta_xyz_pred, None

    def test_network(self, points):
        self.network = self.network.eval()
        with torch.no_grad():
            heatmap, pred = self.network(points)
            return heatmap, pred

class FineMover(object):
    def __init__(self, model_path='reg_seg_heatmap_v3/2022-01-18_03-48', model_name='pointnet2_reg_seg_heatmap_v3_msg', checkpoint_name='best_model_e_185.pth',use_cpu=False, out_channel=9):
        '''MODEL LOADING'''
        exp_dir = os.path.join('/home/luben/robot-peg-in-hole-task/mankey/log', model_path)
        model = importlib.import_module('mankey.models.' + model_name)
        self.network = model.get_model(out_channel)
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
        return points # points:(1, C, N)

    def process_raw_mutliple_camera(self, depth_mm_list, camera2world_list):
        xyz_in_world_list = []
        for idx, depth_mm in enumerate(depth_mm_list):
            depth = depth_mm / 1000  # unit: mm to m
            xyz, choose = self.depth_2_pcd(depth, factor, intrinsic_matrix)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            down_pcd = pcd.uniform_down_sample(every_k_points=8)
            down_xyz_in_camera = np.asarray(down_pcd.points).astype(np.float32)
            down_xyz_in_camera = down_xyz_in_camera[:8000, :]

            # camera coordinate to world coordinate
            down_xyz_in_world = []
            for xyz in down_xyz_in_camera:
                camera2world = np.array(camera2world_list[idx])
                xyz = np.append(xyz, [1], axis=0).reshape(4, 1)
                xyz_world = camera2world.dot(xyz)
                xyz_world = xyz_world[:3] * 1000
                down_xyz_in_world.append(xyz_world)
            xyz_in_world_list.append(down_xyz_in_world)
        concat_xyz_in_world = np.array(xyz_in_world_list)
        concat_xyz_in_world = concat_xyz_in_world.reshape(-1, 3)
        points = np.expand_dims(concat_xyz_in_world, axis=0)
        points = provider.normalize_data(points)
        points = torch.Tensor(points)
        if not self.use_cpu:
            points = points.cuda()
        points = points.transpose(2, 1)
        return points  # points:(1, C, N)

    def inference(self, depth):
        self.network = self.network.eval()
        with torch.no_grad():
            points, centroid, m = self.process_raw(depth)
            heatmap, pred = self.network(points)
            delta_rot_pred_6d = pred[:, 0:6]
            delta_rot_pred = compute_rotation_matrix_from_ortho6d(delta_rot_pred_6d, self.use_cpu) # batch*3*3
            delta_xyz_pred = pred[:, 6:9].view(-1,3) # batch*3

        return delta_rot_pred[0], delta_xyz_pred[0]

    def inference_multiple_camera(self, depth_mm_list, camera2world_list):
        self.network = self.network.eval()
        with torch.no_grad():
            points = self.process_raw_mutliple_camera(depth_mm_list, camera2world_list)
            heatmap_pred, action_pred, step_size_pred = self.network(points)

            delta_rot_pred_6d = action_pred[:, 0:6]
            delta_rot_pred = compute_rotation_matrix_from_ortho6d(delta_rot_pred_6d, self.use_cpu) # batch*3*3
            delta_rot_pred = delta_rot_pred[0]
            delta_xyz_pred = action_pred[:, 6:9].view(-1,3) # batch*3
            delta_xyz_pred = delta_xyz_pred[0]
            step_size_pred = step_size_pred[0][0] / 100
            delta_xyz_pred = delta_xyz_pred * step_size_pred

            delta_rot_pred = delta_rot_pred.cpu().numpy()
            delta_xyz_pred = delta_xyz_pred.cpu().numpy()
            step_size_pred = step_size_pred.cpu().numpy()

        return delta_rot_pred, delta_xyz_pred, step_size_pred

    def test_network(self, points):
        self.network = self.network.eval()
        with torch.no_grad():
            heatmap, pred = self.network(points)
            return heatmap, pred

if __name__ == '__main__':
    pass