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

import mankey.provider as provider
import mankey.config.parameter as parameter
from mankey.models.utils import compute_rotation_matrix_from_ortho6d
from sklearn.decomposition import PCA

focal_length = 309.019
principal = 128
factor = 1
intrinsic_matrix = np.array([[focal_length, 0, principal],
                             [0, focal_length, principal],
                             [0, 0, 1]], dtype=np.float64)

class CoarseMover(object):
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

    def inference_from_pcd(self, xyz, centroid, m):
        # input param: xyz Tensor(1, C, N)
        self.network = self.network.eval()
        with torch.no_grad():
            if not self.use_cpu:
                xyz = xyz.cuda()
            kpt_of_pred, confidence = self.network.forward(xyz)
            xyz = xyz.permute(0, 2, 1)
            kpt_pred = xyz - kpt_of_pred[:, :, :3]  # (B, N, 3)
            kpt_x_pred = xyz - kpt_of_pred[:, :, 3:6]  # (B, N, 3)
            kpt_y_pred = xyz - kpt_of_pred[:, :, 6:]  # (B, N, 3)
            mean_kpt_pred = torch.sum(kpt_pred * confidence, dim=1) / torch.sum(confidence, dim=1)
            mean_kpt_pred = mean_kpt_pred[0].cpu().numpy()
            real_kpt_pred = (mean_kpt_pred * m) + centroid
            real_kpt_pred = real_kpt_pred.reshape(1,3)
            confidence = confidence[0].cpu().numpy()
            #compute dir_x and dir_y
            kpt_x_pred = kpt_x_pred[0].cpu().numpy()
            kpt_y_pred = kpt_y_pred[0].cpu().numpy()
            real_kpt_x_pred = (kpt_x_pred * m) + centroid
            real_kpt_y_pred = (kpt_y_pred * m) + centroid
            confidence = np.squeeze(confidence, axis=1) #(8000,)
            threshold = 0.8
            real_kpt_x_pred = real_kpt_x_pred[confidence > threshold, :]
            real_kpt_y_pred = real_kpt_y_pred[confidence > threshold, :]

            print(real_kpt_x_pred.shape)
            print(real_kpt_y_pred.shape)
            print(confidence.shape)

            pca = PCA(n_components=3)
            pca.fit(real_kpt_x_pred)
            dir_x_pred = -pca.components_[0]
            dir_x_pred = dir_x_pred / np.linalg.norm(dir_x_pred)
            #length = pca.explained_variance_[0]
            pca.fit(real_kpt_y_pred)
            dir_y_pred = pca.components_[0]
            if dir_y_pred[0] > 0:
                dir_y_pred = -dir_y_pred
            dir_y_pred = dir_y_pred / np.linalg.norm(dir_y_pred)
            #length = pca.explained_variance_[0]

            return real_kpt_pred, dir_x_pred, dir_y_pred, real_kpt_x_pred, real_kpt_y_pred, confidence

if __name__ == '__main__':
    model_path = 'kpt_dir/2022-04-08_00-14'
    mover = CoarseMover(model_path=model_path, model_name='pointnet2_kpt_dir',
                               checkpoint_name='best_model_e_71.pth', use_cpu=False, out_channel=9)
    #data_root = '/home/luben/data/pdc/logs_proto/2022-02-26-test/fine_insertion_square_2022-02-26-test/processed'
    data_root = '/home/luben/data/pdc/logs_proto/coarse_insertion_square_2022-03-28-test/processed'
    pcd_seg_heatmap_kpt_folder_path = os.path.join(data_root, 'pcd_seg_heatmap_kpt_dir')
    visualize_kpt_path = os.path.join(data_root, 'visualize_kpt')
    visualize_dir_path = os.path.join(data_root, 'visualize_dir')
    visualize_heatmap_path = os.path.join(data_root, 'visualize_heatmap')
    # create folder
    cwd = os.getcwd()
    os.chdir(data_root)
    if not os.path.exists('visualize_kpt'):
        os.makedirs('visualize_kpt')
    if not os.path.exists('visualize_dir'):
        os.makedirs('visualize_dir')
    if not os.path.exists('visualize_heatmap'):
        os.makedirs('visualize_heatmap')
    os.chdir(cwd)

    with open(os.path.join(data_root, 'peg_in_hole.yaml'), 'r') as f_r:
        data = yaml.load(f_r)
    dist_list = []
    angle_x_list = []
    angle_y_list = []
    for key, value in tqdm(data.items()):
        pcd_filename = data[key]['pcd']
        pcd = np.load(os.path.join(pcd_seg_heatmap_kpt_folder_path, pcd_filename))
        pcd = pcd[:, :3]
        centroid = np.array(data[key]['pcd_centroid'])
        m = data[key]['pcd_mean']
        real_pcd = (pcd * m) + centroid
        pcd = np.expand_dims(pcd, axis=0)
        pcd = torch.Tensor(pcd)
        pcd = pcd.transpose(2, 1)
        #real_kpt_pred, dir_pred, rot_mat_pred, confidence = mover.inference_from_pcd(pcd, centroid, m)
        real_kpt_pred, dir_x_pred, dir_y_pred, real_kpt_x_pred, real_kpt_y_pred, confidence = mover.inference_from_pcd(pcd, centroid, m)
        #print('hole keypoint prediction:', real_kpt_pred)

        # compute keypoint error
        hole_keypoint_top_pose_in_world = np.array(data[key]['hole_keypoint_obj_top_pose_in_world'])
        hole_keypoint_top_pos_in_world = hole_keypoint_top_pose_in_world[:3, 3]*1000  # unit: m to mm
        dist = np.linalg.norm(real_kpt_pred - hole_keypoint_top_pos_in_world)
        dist_list.append(dist)
        print(dist)

        # compute direction error(use two keypoints to calculate insertion vector)
        dir = hole_keypoint_top_pose_in_world[:3, 0].reshape(3, 1)
        dot_product = np.dot(dir_x_pred, dir)
        angle = math.degrees(math.acos(dot_product / (np.linalg.norm(dir_x_pred) * np.linalg.norm(dir))))
        print(angle)
        angle_x_list.append(angle)
        dir = hole_keypoint_top_pose_in_world[:3, 1].reshape(3, 1)
        dot_product = np.dot(dir_y_pred, dir)
        angle = math.degrees(math.acos(dot_product / (np.linalg.norm(dir_y_pred) * np.linalg.norm(dir))))
        print(angle)
        angle_y_list.append(angle)
        # visualize hole keypoint
        visualize_pcd = o3d.geometry.PointCloud()
        visualize_pcd.points = o3d.utility.Vector3dVector(np.concatenate((real_pcd, real_kpt_pred), axis=0))
        color = np.concatenate((np.repeat(np.array([[1, 1, 1]]),real_pcd.shape[0], axis=0), np.array([[1, 0, 0]])), axis=0)
        visualize_pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.io.write_point_cloud(os.path.join(visualize_kpt_path, 'kptof_' + str(key) + '.ply'), visualize_pcd)

        # visualize direction
        visualize_pcd = o3d.geometry.PointCloud()
        visualize_pcd.points = o3d.utility.Vector3dVector(np.concatenate((real_pcd, real_kpt_x_pred, real_kpt_y_pred), axis=0))
        pcd_color = np.repeat(np.array([[1, 1, 1]]), real_pcd.shape[0], axis=0)
        kpt_x_color = np.repeat(np.array([[1, 0, 0]]), real_kpt_x_pred.shape[0], axis=0)
        kpt_y_color = np.repeat(np.array([[0, 0, 1]]), real_kpt_y_pred.shape[0], axis=0)
        color = np.concatenate((pcd_color, kpt_x_color,kpt_y_color ),axis=0)
        visualize_pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.io.write_point_cloud(os.path.join(visualize_dir_path, 'dir_' + str(key) + '.ply'), visualize_pcd)

        # visualize heatmap
        visualize_pcd = o3d.geometry.PointCloud()
        visualize_pcd.points = o3d.utility.Vector3dVector(real_pcd)
        heatmap_color = np.repeat(np.expand_dims(confidence, 1), 3, axis=1).reshape(-1, 3)  # n x 3
        visualize_pcd.colors = o3d.utility.Vector3dVector(heatmap_color)
        o3d.io.write_point_cloud(os.path.join(visualize_heatmap_path, 'heatmap_' + str(key) + '.ply'), visualize_pcd)

    print(max(dist_list))
    print(max(angle_x_list))
    print(max(angle_y_list))
    dist = sum(dist_list) / len(dist_list)
    print('Keypoint error distance: {} (mm)'.format(dist))
    angle = sum(angle_x_list) / len(angle_x_list)
    print('Angle x error: {} (degrees)'.format(angle))
    angle = sum(angle_y_list) / len(angle_y_list)
    print('Angle y error: {} (degrees)'.format(angle))





