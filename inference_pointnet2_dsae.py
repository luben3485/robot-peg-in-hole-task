"""
Author: Benny
Date: Nov 2019
"""

import os
import cv2
import copy
import yaml
'''HYPER PARAMETER'''
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('/home/luben/robot-peg-in-hole-task')
import torch
import torchvision.transforms as transforms
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
from scipy.spatial.transform import Rotation as R

focal_length = 309.019
principal = 128
factor = 1
intrinsic_matrix = np.array([[focal_length, 0, principal],
                             [0, focal_length, principal],
                             [0, 0, 1]], dtype=np.float64)

class DSAEMover(object):
    def __init__(self, model_path='kpts/2022-02-??_??-??', model_name='pointnet2_kpts', checkpoint_name='best_model_e_?.pth', use_cpu=False, out_channel=9):
        '''MODEL LOADING'''
        exp_dir = os.path.join('/home/luben/robot-peg-in-hole-task/mankey/log', model_path)
        model = importlib.import_module('mankey.models.' + model_name)
        self.network = model.DSAE()
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
        # normalize the pcd
        centroid = np.mean(concat_xyz_in_world, axis=0)
        points = concat_xyz_in_world - centroid
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / m
        points = torch.Tensor(points)
        if not self.use_cpu:
            points = points.cuda()
        points = torch.unsqueeze(points, 0)
        points = points.transpose(2, 1)
        return points, centroid, m  # points:(1, C, N)

    def inference(self, depth):
        self.network = self.network.eval()
        with torch.no_grad():
            points = self.process_raw(depth)
            heatmap, pred = self.network(points)
            delta_rot_pred_6d = pred[:, 0:6]
            delta_rot_pred = compute_rotation_matrix_from_ortho6d(delta_rot_pred_6d, self.use_cpu) # batch*3*3
            delta_xyz_pred = pred[:, 6:9].view(-1,3) # batch*3

        return delta_rot_pred[0], delta_xyz_pred[0]
    '''
    def inference_multiple_camera(self, depth_mm_list, camera2world_list, gripper_pose):
        self.network = self.network.eval()
        with torch.no_grad():
            points, pcd_centroid, pcd_mean = self.process_raw_mutliple_camera(depth_mm_list, camera2world_list)
            pcd_mean = torch.tensor(pcd_mean, dtype=torch.float32)
            pcd_centroid = torch.tensor(pcd_centroid, dtype=torch.float32)
            gripper_pose = torch.unsqueeze(torch.tensor(gripper_pose, dtype=torch.float32), 0)
            if not self.use_cpu:
                pcd_mean = pcd_mean.cuda()
                pcd_centroid = pcd_centroid.cuda()
                gripper_pose = gripper_pose.cuda()

            kpt_of_pred, trans_of_pred, rot_of_pred, mean_kpt_pred, rot_mat_pred, confidence = self.network(points)
            gripper_pos = gripper_pose[:, :3, 3]
            gripper_rot = gripper_pose[:, :3, :3]
            real_kpt_pred = (mean_kpt_pred * pcd_mean) + pcd_centroid
            real_kpt_pred = real_kpt_pred / 1000  # unit: mm to m
            real_trans_of_pred = (trans_of_pred * pcd_mean) / 1000  # unit: mm to m
            delta_trans_pred = real_kpt_pred - gripper_pos + real_trans_of_pred
            delta_rot_pred = torch.bmm(torch.transpose(gripper_rot, 1, 2), rot_mat_pred)
            delta_rot_pred = torch.bmm(delta_rot_pred, rot_of_pred)

            delta_trans_pred = delta_trans_pred[0].cpu().numpy()
            delta_rot_pred = delta_rot_pred[0].cpu().numpy()

        return delta_trans_pred, delta_rot_pred
    '''
    def inference_multiple_camera(self, stacked_rgb):
        self.network = self.network.eval()
        with torch.no_grad():
            #normalize
            rgb_mean = np.array([(0.485, 0.456, 0.406)])
            rgb = copy.deepcopy(stacked_rgb)
            rgb /= 255.0
            rgb -= rgb_mean
            rgb = torch.tensor(rgb, dtype=torch.float32)
            rgb = torch.unsqueeze(rgb, 0)
            rgb = rgb.permute(0, 3, 1, 2)
            if not self.use_cpu:
                rgb = rgb.cuda()
            delta_xyz_pred, delta_rot_euler_pred, depth_pred, kpts = self.network.forward_inference(rgb)

        kpts = kpts[0].cpu().numpy()
        depth_pred = depth_pred[0].permute(1, 2, 0).cpu().numpy()
        delta_xyz_pred = delta_xyz_pred[0].cpu().numpy()
        delta_rot_euler_pred = delta_rot_euler_pred[0].cpu().numpy()
        delta_xyz_pred[0] = delta_xyz_pred[0] * (0.00625 - (-0.00625)) + (-0.00625)
        delta_xyz_pred[1] = delta_xyz_pred[1] * (0.00625 - (-0.00625)) + (-0.00625)
        delta_xyz_pred[2] = max(delta_xyz_pred[2] * (0 - (-0.00299982)) + (-0.00299982), -0.00299982)

        #delta_rot_euler_pred = delta_rot_euler_pred * 5
        r = R.from_euler('zyx', delta_rot_euler_pred, degrees=True)
        delta_rot_pred = r.as_matrix()

        return delta_xyz_pred, delta_rot_pred, depth_pred, kpts

    def test_network(self, points):
        # input param: points Tensor(1, C, N)
        self.network = self.network.eval()
        with torch.no_grad():
            if not self.use_cpu:
                points = points.cuda()
            kpt_of_pred, trans_of_pred, rot_of_pred, mean_kpt_pred, mean_kpt_x_pred, mean_kpt_y_pred, rot_mat_pred, confidence = self.network.forward_test(points)
            mean_kpt_pred = mean_kpt_pred[0].cpu().numpy()
            mean_kpt_x_pred = mean_kpt_x_pred[0].cpu().numpy()
            mean_kpt_y_pred = mean_kpt_y_pred[0].cpu().numpy()
            rot_mat_pred = rot_mat_pred[0].cpu().numpy()
            confidence = confidence[0].cpu().numpy()

            return mean_kpt_pred, mean_kpt_x_pred, mean_kpt_y_pred, rot_mat_pred, confidence

if __name__ == '__main__':

    model_path = 'dsae/2022-05-03_22-20'
    mover = DSAEMover(model_path=model_path, model_name='dsae',
                               checkpoint_name='best_model_e_177.pth', use_cpu=False, out_channel=9)
    data_root = '/home/luben/data/pdc/logs_proto/fine_insertion_square_7x12x12_2022-05-02-notilt/processed'
    visualize_kpt_path = os.path.join(data_root, 'visualize_kpt')
    visualize_depth_path = os.path.join(data_root, 'visualize_depth')
    # create folder
    cwd = os.getcwd()
    os.chdir(data_root)
    if not os.path.exists('visualize_kpt'):
        os.makedirs('visualize_kpt')
    if not os.path.exists('visualize_depth'):
        os.makedirs('visualize_depth')
    os.chdir(cwd)

    with open(os.path.join(data_root, 'peg_in_hole_small.yaml'), 'r') as f_r:
        data = yaml.load(f_r)

    for key, value in tqdm(data.items()):
        rgb_name_list = data[key]['rgb_image_filename']
        '''
        if key==1:
            depth_name_list = data[key]['depth_image_filename']
            depth = cv2.imread(os.path.join(data_root, 'images', depth_name_list[0]), cv2.IMREAD_ANYDEPTH)
            depth = np.clip((depth/320), 0, 1)
            depth*=255
            cv2.imwrite(os.path.join(data_root, str(key) + '_depth.png'), depth)
            assert False
        '''
        rgb_name = rgb_name_list[0]
        rgb = cv2.imread(os.path.join(data_root, 'images', rgb_name), cv2.IMREAD_COLOR)
        rgb = rgb[:,:,::-1].astype(np.float32)
        delta_xyz_pred, delta_rot_pred, depth_pred, kpts = mover.inference_multiple_camera(rgb)
        assert rgb.shape == (256, 256, 3)
        depth_draw = copy.deepcopy(depth_pred)
        depth_draw = cv2.resize(depth_draw, (64, 64))*255
        depth_draw = depth_draw.astype(int)
        rgb_draw = copy.deepcopy(rgb)
        rgb_draw = cv2.resize(rgb_draw, (64, 64)).astype(int)
        rgb_draw = rgb_draw[:, :, ::-1]
        for kpt in kpts:
            x = int(kpt[0])
            y = int(kpt[1])
            rgb_draw[x, y] = [255, 0, 0]
        # plt.imshow(depth_draw)
        # plt.imshow(rgb_draw)
        # plt.show()
        cv2.imwrite(os.path.join(visualize_kpt_path, str(key)+'.png'), rgb_draw)
        cv2.imwrite(os.path.join(visualize_depth_path, str(key)+'.png'), depth_draw)

    '''
    total_mean = []
    total_max = []
    total_min = []
    t_xyz = []
    for key, value in tqdm(data.items()):
        depth_mean = []
        depth_max = []
        depth_min = []
        xyz = []
        for i in range(len(value)):
            rgb_name_list = data[key][i]['rgb_image_filename']
            depth_name_list = data[key][i]['depth_image_filename']
            delta_xyz = data[key][i]['delta_translation']
            delta_rot = data[key][i]['delta_rotation_matrix']

            depth = cv2.imread(os.path.join(data_root, 'images', depth_name_list[0]), cv2.IMREAD_ANYDEPTH)
            depth_mean.append(np.mean(depth))
            depth_max.append(np.max(depth))
            depth_min.append(np.min(depth))
            xyz.append(delta_xyz)
        xyz = np.array(xyz)
        xyz = np.min(xyz, 0)
        t_xyz.append(xyz)
        total_mean.append(sum(depth_mean)/len(depth_mean))
        total_max.append(max(depth_max))
        total_min.append(max(depth_min))
    t_xyz = np.array(t_xyz)
    print(np.min(t_xyz, 0))
    print(sum(total_mean)/len(total_mean))
    print(max(total_max))
    print(min(total_min))
    '''



