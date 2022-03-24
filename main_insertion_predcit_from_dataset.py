from env.single_robotic_arm import SingleRoboticArm
import numpy as np
import cv2
import math
from transforms3d.quaternions import mat2quat, quat2axangle, quat2mat, qmult
import random
from scipy.spatial.transform import Rotation as R
from inference_pointnet2_reg_seg_heatmap import PointnetMover
import mankey.provider as provider
import os
import sys
import yaml
import torch
import open3d as o3d
sys.path.append('/home/luben/robot-peg-in-hole-task')

def main():
    data_root = '/home/luben/data/pdc/logs_proto/testset'
    pcd_seg_heatmap_folder_path = os.path.join(data_root, 'pcd_seg_heatmap')
    result_heatmap_folder_path = os.path.join(data_root, 'result_heatmap')

    model_path_coarse = 'reg_seg_heatmap_v2/2022-01-18_03-48'
    model_path_fine = 'reg_seg_heatmap_v2/2022-01-18_01-33'
    coarse_mover = PointnetMover(model_path=model_path_coarse, model_name='pointnet2_reg_seg_heatmap_msg_v2', checkpoint_name='best_model_e_83.pth', use_cpu=False, out_channel=9)
    fine_mover = PointnetMover(model_path=model_path_fine, model_name='pointnet2_reg_seg_heatmap_msg_v2', checkpoint_name='best_model_e_114.pth', use_cpu=False, out_channel=9)
    with open(os.path.join(data_root, 'peg_in_hole_small.yaml'), 'r') as f_r:
        data = yaml.load(f_r)
    for key, value in data.items():
        raw = np.load(os.path.join(pcd_seg_heatmap_folder_path, data[key]['pcd']))
        points = raw[:, :3]
        points = np.expand_dims(points, axis=0)
        points = provider.normalize_data(points)
        points =np.array(points).astype(np.float32)
        gt_seg = raw[:, 3].reshape(-1, 1)
        gt_heatmap = raw[:, 4].reshape(-1, 1)
        points = torch.tensor(points).transpose(2, 1).cuda()
        pred_heatmap, action = fine_mover.test_network(points)

        # visualize predicted heatmap
        pred_heatmap = pred_heatmap.cpu().squeeze(0).numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw[:,:3])
        heatmap_color = np.repeat(pred_heatmap, 3, axis=1).reshape(-1, 3)  # n x 3
        pcd.colors = o3d.utility.Vector3dVector(heatmap_color)
        o3d.io.write_point_cloud(os.path.join(result_heatmap_folder_path, str(key) + '_pred_heatmap.ply'), pcd)

        # visualize gt heatmap
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw[:, :3])
        heatmap_color = np.repeat(gt_heatmap, 3, axis=1).reshape(-1, 3)  # n x 3
        pcd.colors = o3d.utility.Vector3dVector(heatmap_color)
        o3d.io.write_point_cloud(os.path.join(result_heatmap_folder_path, str(key) + '_gt_heatmap.ply'), pcd)

if __name__ == '__main__':
    main()
