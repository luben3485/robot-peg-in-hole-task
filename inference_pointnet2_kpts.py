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

class CoarseMover(object):
    def __init__(self, model_path='kpts/2022-02-??_??-??', model_name='pointnet2_kpts', checkpoint_name='best_model_e_?.pth', use_cpu=False, out_channel=9):
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

    def inference_from_pcd(self, points, centroid, m, use_offset=False):
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
            trans_of_pred = trans_of_pred[0].cpu().numpy()
            real_kpt_pred = (mean_kpt_pred * m) + centroid  # unit:mm
            real_kpt_pred = real_kpt_pred.reshape(1, 3)
            real_kpt_x_pred = (mean_kpt_x_pred * m) + centroid  # unit:mm
            real_kpt_x_pred = real_kpt_x_pred.reshape(1, 3)
            dir_pred = real_kpt_pred - real_kpt_x_pred
            dir_pred = dir_pred / np.linalg.norm(dir_pred)
            # offset means translation offset now, and rotation offset is not used.
            if not use_offset:
                return real_kpt_pred, dir_pred, rot_mat_pred, confidence
            if use_offset:
                real_trans_of_pred =  trans_of_pred * m # unit: mm
                real_kpt_pred += real_trans_of_pred
                return real_kpt_pred, dir_pred, rot_mat_pred, confidence

if __name__ == '__main__':
    add_noise = True
    use_offset = False
    model_path = 'kpts/2022-04-25_07-26'
    mover = CoarseMover(model_path=model_path, model_name='pointnet2_kpts',
                               checkpoint_name='best_model_e_60.pth', use_cpu=False, out_channel=9)
    #data_root = '/home/luben/data/pdc/logs_proto/2022-02-26-test/fine_insertion_square_2022-02-26-test/processed'
    data_root = '/home/luben/data/pdc/logs_proto/coarse_insertion_square_2022-03-28-test/processed'
    pcd_seg_heatmap_kpt_folder_path = os.path.join(data_root, 'pcd_seg_heatmap_3kpt')
    visualize_kpt_path = os.path.join(data_root, 'visualize_kpt')
    visualize_heatmap_path = os.path.join(data_root, 'visualize_heatmap')
    # create folder
    cwd = os.getcwd()
    os.chdir(data_root)
    if not os.path.exists('visualize_kpt'):
        os.makedirs('visualize_kpt')
    if not os.path.exists('visualize_heatmap'):
        os.makedirs('visualize_heatmap')
    os.chdir(cwd)

    with open(os.path.join(data_root, 'peg_in_hole.yaml'), 'r') as f_r:
        data = yaml.load(f_r)
    dist_list = []
    angle_list = []
    for key, value in tqdm(data.items()):
        pcd_filename = data[key]['pcd']
        pcd = np.load(os.path.join(pcd_seg_heatmap_kpt_folder_path, pcd_filename))
        pcd = pcd[:, :3]
        centroid = np.array(data[key]['pcd_centroid'])
        m = data[key]['pcd_mean']
        if add_noise == True:
            # add noise on pcd
            real_pcd = pcd * m + centroid
            noise_real_pcd = jitter_point_cloud(real_pcd, sigma=1, clip=5)
            # normalize the pcd again
            centroid = np.mean(noise_real_pcd, axis=0)
            pcd = noise_real_pcd - centroid
            m = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)))
            pcd = pcd / m

        origin_real_pcd = (pcd * m) + centroid
        pcd = np.expand_dims(pcd, axis=0)
        pcd = torch.Tensor(pcd)
        pcd = pcd.transpose(2, 1)
        real_kpt_pred, dir_pred, rot_mat_pred, confidence = mover.inference_from_pcd(pcd, centroid, m, use_offset=use_offset)
        #print('hole keypoint prediction:', real_kpt_pred)

        # compute keypoint error
        hole_keypoint_top_pose_in_world = np.array(data[key]['hole_keypoint_obj_top_pose_in_world'])
        hole_keypoint_top_pos_in_world = hole_keypoint_top_pose_in_world[:3, 3]*1000  # unit: m to mm
        dist = np.linalg.norm(real_kpt_pred - hole_keypoint_top_pos_in_world)
        dist_list.append(dist)
        # compute direction error(use two keypoints to calculate insertion vector)
        dir = hole_keypoint_top_pose_in_world[:3, 0].reshape(3, 1)
        dot_product = np.dot(dir_pred, dir)
        angle = math.degrees(math.acos(dot_product / (np.linalg.norm(dir_pred) * np.linalg.norm(dir))))
        angle_list.append(angle)
        # visualize hole keypoint
        visualize_pcd = o3d.geometry.PointCloud()
        visualize_pcd.points = o3d.utility.Vector3dVector(np.concatenate((origin_real_pcd, real_kpt_pred), axis=0))
        color = np.concatenate((np.repeat(np.array([[1, 1, 1]]),origin_real_pcd.shape[0], axis=0), np.array([[1, 0, 0]])), axis=0)
        visualize_pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.io.write_point_cloud(os.path.join(visualize_kpt_path, 'kptof_' + str(key) + '.ply'), visualize_pcd)

        # visualize heatmap
        visualize_pcd = o3d.geometry.PointCloud()
        visualize_pcd.points = o3d.utility.Vector3dVector(origin_real_pcd)
        heatmap_color = np.repeat(confidence, 3, axis=1).reshape(-1, 3)  # n x 3
        visualize_pcd.colors = o3d.utility.Vector3dVector(heatmap_color)
        o3d.io.write_point_cloud(os.path.join(visualize_heatmap_path, 'heatmap_' + str(key) + '.ply'), visualize_pcd)

    dist = sum(dist_list) / len(dist_list)
    print('Keypoint error distance: {} mm, Max:{} mm'.format(dist, max(dist_list)))
    angle = sum(angle_list) / len(angle_list)
    print('Angle error: {} (degrees), Max:{} mm'.format(angle, max(angle_list)))





