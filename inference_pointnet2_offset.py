"""
Author: Benny
Date: Nov 2019
"""

import os
import math
import yaml
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
import copy

import mankey.provider as provider
import mankey.config.parameter as parameter
from mankey.models.utils import compute_rotation_matrix_from_ortho6d
from mankey.network.loss import RMSELoss
#from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

focal_length = 309.019
principal = 128
factor = 1
intrinsic_matrix = np.array([[focal_length, 0, principal],
                             [0, focal_length, principal],
                             [0, 0, 1]], dtype=np.float64)

def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N, C = data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += data
    return jittered_data

class FineMover(object):
    def __init__(self, model_path='kpts/2022-02-??_??-??', model_name='pointnet2_kpt_dir', checkpoint_name='best_model_e_?.pth', use_cpu=False, out_channel=9):
        '''MODEL LOADING'''
        exp_dir = os.path.join('/home/luben/robot-peg-in-hole-task/mankey/log', model_path)
        model = importlib.import_module('mankey.models.' + model_name)
        self.network = model.get_model()
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
        # normalize the pcd
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / m
        points = torch.Tensor(points)
        if not self.use_cpu:
            points = points.cuda()
        points = torch.unsqueeze(points, 0)
        points = points.transpose(2, 1)
        return points, centroid, m  # points:(1, C, N)

    def process_raw_mutliple_camera(self, depth_mm_list, camera2world_list, add_noise=False):
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
        if add_noise==True:
            concat_xyz_in_world = jitter_point_cloud(concat_xyz_in_world, sigma=1, clip=3)
        # visualize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(concat_xyz_in_world)
        o3d.io.write_point_cloud(os.path.join('fine-1.ply'), pcd)

        # normalize the pcd
        centroid = np.mean(concat_xyz_in_world, axis=0)
        points = concat_xyz_in_world - centroid
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / m
        points = torch.Tensor(points)
        points = torch.unsqueeze(points, 0)
        points = points.transpose(2, 1)
        return points, centroid, m  # points:(1, C, N)

    def crop_pcd(self, points, centroid, m, gripper_pos):
        # input param: points Tensor(1, C, N)
        points = points.transpose(2, 1)
        points = points[0].numpy()
        real_pcd = (points * m) + centroid
        crop_xyz = []
        bound = 0.05
        for xyz in real_pcd:
            x = xyz[0] / 1000  # unit:m
            y = xyz[1] / 1000  # unit:m
            z = xyz[2] / 1000  # unit:m
            if x >= gripper_pos[0] - bound and x <= gripper_pos[0] + bound and \
                    y >= gripper_pos[1] - bound and y <= gripper_pos[1] + bound and \
                    z >= gripper_pos[2] - bound and z <= gripper_pos[2] + bound:
                crop_xyz.append(xyz)
        if len(crop_xyz) == 0:
            crop_xyz = real_pcd
        crop_xyz = np.array(crop_xyz).reshape(-1, 3)
        if crop_xyz.shape[0] >= 2048:
            crop_xyz = crop_xyz[:2048, :]
        points = copy.deepcopy(crop_xyz)
        # visualize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join('fine-2.ply'), pcd)

        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / m
        points = torch.Tensor(points)
        points = torch.unsqueeze(points, 0)
        points = points.transpose(2, 1)

        return points, centroid, m  # points:(1, C, N)

    def inference_from_pcd_with_error(self, xyz, centroid, m, delta_xyz, delta_rot_euler):
        # input param: xyz Tensor(1, C, N)
        criterion_rmse = RMSELoss()
        criterion_cos = torch.nn.CosineSimilarity(dim=1)
        delta_xyz = torch.tensor(delta_xyz).view(1,3)
        delta_rot_euler = torch.tensor(delta_rot_euler).view(1,3)
        self.network = self.network.eval()
        with torch.no_grad():
            if not self.use_cpu:
                xyz = xyz.cuda()
                criterion_rmse = criterion_rmse.cuda()
                criterion_cos = criterion_cos.cuda()
                delta_xyz = delta_xyz.cuda()
                delta_rot_euler = delta_rot_euler.cuda()
            # use euler angle now
            '''
            delta_xyz_pred, delta_rot_6d_pred = self.network(xyz)
            delta_rot_pred = compute_rotation_matrix_from_ortho6d(delta_rot_6d_pred, use_cpu=self.use_cpu)
            '''
            delta_xyz_pred, delta_rot_euler_pred = self.network(xyz)
            loss_t = (1 - criterion_cos(delta_xyz_pred, delta_xyz)).mean() + criterion_rmse(delta_xyz_pred, delta_xyz)
            loss_r = criterion_rmse(delta_rot_euler_pred, delta_rot_euler)
            loss = loss_t + loss_r
            delta_xyz_pred = delta_xyz_pred[0].cpu().numpy()
            delta_xyz_pred = delta_xyz_pred / 200
            delta_rot_euler_pred = delta_rot_euler_pred[0].cpu().numpy()
            delta_rot_euler_pred = delta_rot_euler_pred * 5
            r = R.from_euler('zyx', delta_rot_euler_pred, degrees=True)
            delta_rot_pred = r.as_matrix()

            return delta_xyz_pred, delta_rot_pred, delta_rot_euler_pred, loss.item()

    def inference_from_pcd(self, xyz, centroid, m, method='fine'):
        # input param: xyz Tensor(1, C, N)
        self.network = self.network.eval()
        with torch.no_grad():
            if not self.use_cpu:
                xyz = xyz.cuda()
            # use euler angle now
            '''
            delta_xyz_pred, delta_rot_6d_pred = self.network(xyz)
            delta_rot_pred = compute_rotation_matrix_from_ortho6d(delta_rot_6d_pred, use_cpu=self.use_cpu)
            '''
            delta_xyz_pred, delta_rot_euler_pred = self.network(xyz)
            delta_xyz_pred = delta_xyz_pred[0].cpu().numpy()
            delta_rot_euler_pred = delta_rot_euler_pred[0].cpu().numpy()
            if method=='coarse':
                delta_xyz_pred = delta_xyz_pred / 20
                delta_rot_euler_pred = delta_rot_euler_pred * 50
            elif method=='fine':
                delta_xyz_pred = delta_xyz_pred / 200
                delta_rot_euler_pred = delta_rot_euler_pred * 10
            r = R.from_euler('zyx', delta_rot_euler_pred, degrees=True)
            delta_rot_pred = r.as_matrix()

            return delta_xyz_pred, delta_rot_pred, delta_rot_euler_pred

if __name__ == '__main__':
    crop_pcd = True
    model_path = 'offset/2022-05-18_00-29'
    mover = FineMover(model_path=model_path, model_name='pointnet2_offset',
                               checkpoint_name='best_model_e_60.pth', use_cpu=False, out_channel=9)
    #data_root = '/home/luben/data/pdc/logs_proto/2022-02-26-test/fine_insertion_square_2022-02-26-test/processed'
    data_root = '/home/luben/data/pdc/logs_proto/fine_insertion_square_7x12x12_2022-05-15-test/processed'
    pcd_seg_heatmap_kpt_folder_path = os.path.join(data_root, 'pcd_seg_heatmap_3kpt')

    with open(os.path.join(data_root, 'peg_in_hole.yaml'), 'r') as f_r:
        data = yaml.load(f_r)

    loss_ = []
    for key, value in tqdm(data.items()):
        pcd_filename = data[key]['pcd']
        pcd = np.load(os.path.join(pcd_seg_heatmap_kpt_folder_path, pcd_filename))
        pcd = pcd[:, :3]
        centroid = np.array(data[key]['pcd_centroid'])
        m = data[key]['pcd_mean']
        real_pcd = (pcd * m) + centroid
        if crop_pcd == True:
            gripper_pos = np.array(data[key]['gripper_pose'])[:3,3]
            crop_xyz = []
            bound = 0.05
            for xyz in real_pcd:
                x = xyz[0] / 1000  # unit:m
                y = xyz[1] / 1000  # unit:m
                z = xyz[2] / 1000  # unit:m
                if x >= gripper_pos[0] - bound and x <= gripper_pos[0] + bound and \
                        y >= gripper_pos[1] - bound and y <= gripper_pos[1] + bound and \
                        z >= gripper_pos[2] - bound and z <= gripper_pos[2] + bound:
                    crop_xyz.append(xyz)
            crop_xyz = np.array(crop_xyz).reshape(-1, 3)
            if crop_xyz.shape[0] >= 2048:
                crop_xyz = crop_xyz[:2048, :]
            pcd = copy.deepcopy(crop_xyz)
            centroid = np.mean(pcd, axis=0)
            pcd = pcd - centroid
            m = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)))
            pcd = pcd / m

            pcd = np.expand_dims(pcd, axis=0)
            pcd = torch.Tensor(pcd)
            pcd = pcd.transpose(2, 1)

        else:
            pcd = np.expand_dims(pcd, axis=0)
            pcd = torch.Tensor(pcd)
            pcd = pcd.transpose(2, 1)

        delta_xyz = np.array(data[key]['delta_translation'])
        delta_rot = np.array(data[key]['delta_rotation_matrix'])
        delta_rot_euler = np.array(data[key]['r_euler'])


        delta_xyz_pred, delta_rot_pred, delta_rot_euler_pred, loss = mover.inference_from_pcd_with_error(pcd, centroid, m, delta_xyz, delta_rot_euler)
        '''
        print('pred:', delta_xyz_pred)
        print('gt:', delta_xyz)
        print('pred:', delta_rot_euler_pred)
        print('gt:', delta_rot_euler)
        '''
        loss_.append(loss)
    print(sum(loss_)/len(loss_))


