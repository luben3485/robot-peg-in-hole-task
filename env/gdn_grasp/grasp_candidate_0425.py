import sys
import numpy as np
import math

'''
For GDN
'''
import json
from gdn.representation.euler_scene_att_ce import *
from gdn.detector.pointnet2_att.backbone import Pointnet2MSG
from nms import decode_euler_feature
from nms import initEigen, sanity_check
from nms import crop_index, generate_gripper_edge
from scipy.spatial.transform import Rotation

import time
initEigen(0)


class Grasper(object):
    def __init__(self, gdn_config_path = "/home/luben/GDN_libs/egad.json", gdn_weight_path = "/home/luben/GDN_libs/egad.ckpt"):
        with open(gdn_config_path, "r") as fp:
            self.gdn_config = json.load(fp)
            self.gdn_config['thickness'] = 0.03 # Force overwrite
        self.gdn_gripper_length = self.gdn_config['hand_height']
        self.gdn_input_points = self.gdn_config['input_points']
        self.gdn_model = Pointnet2MSG(self.gdn_config, activation_layer=EulerActivation())
        self.gdn_model = self.gdn_model.cuda()
        self.gdn_model = self.gdn_model.eval()
        self.gdn_model.load_state_dict(torch.load(gdn_weight_path)['base_model'])
        self.gdn_representation = EulerRepresentation(self.gdn_config)

    def get_grasping_candidates(self, point_cloud):
        '''
        point_cloud: Point cloud in world coordinate. Shape: (im_height, im_width, 3), float32
        '''
        config = self.gdn_config
        grasping_candidates = []

        # point_cloud = build_point_cloud(depth_img, intrinsic, trans_mat).astype(np.float32)
        # get the segmentation image
        # seg_img = self._camera.get_image(sensor_type=['seg'])['seg'] #(H, W, 3)
        # segmentation = np.any(seg_img, axis=2) # (H, W)
        # get partial point cloud
        pc_npy = point_cloud.reshape(-1, 3)
        pc_npy_max = np.max(pc_npy, axis=0)
        pc_npy_min = np.min(pc_npy, axis=0) 
        trans_to_frame = (pc_npy_max + pc_npy_min) / 2.0
        trans_to_frame[2] = np.min(pc_npy[:,2])

        while pc_npy.shape[0] < self.gdn_input_points:
            new_pts = pc_npy[np.random.choice(len(pc_npy), self.gdn_input_points-len(pc_npy), replace=True)]
            new_pts = new_pts + np.random.randn(*new_pts.shape) * 1e-6
            pc_npy = np.append(pc_npy, new_pts, axis=0)
            pc_npy = np.unique(pc_npy, axis=0)
        if pc_npy.shape[0]>self.gdn_input_points:
            pc_npy = pc_npy[np.random.choice(len(pc_npy), self.gdn_input_points, replace=False),:]

        # generate grasping candidates
        pc_npy -= trans_to_frame
        with torch.no_grad():
            pc_cuda = torch.from_numpy(pc_npy).float().unsqueeze(0).cuda()
            pred, ind, att = self.gdn_model(pc_cuda)
            pc_subsampled = pointnet2_utils.gather_operation(pc_cuda.transpose(1, 2).contiguous(), ind)
            pc_npy = pc_subsampled.cpu().transpose(1, 2).numpy() + trans_to_frame.reshape(1,1,3) # (B, N, 3) + (B, 1, 3)
            pc_npy = pc_npy.astype(np.float32)
            pred = pred.cpu().numpy().astype(np.float32)
            grasping_candidates = np.asarray(sanity_check(pc_npy[0],
                            np.asarray(decode_euler_feature(
                            pc_npy[0],
                            pred[0].reshape(1,-1),
                            *pred[0].shape[:-1],
                            config['hand_height'],
                            config['gripper_width'],
                            config['thickness_side'],
                            config['rot_th'],
                            config['trans_th'],
                            1000, # max number of candidate
                            -np.inf, # threshold of candidate
                            300,  # max number of grasp in NMS
                            4,    # number of threads
                            True  # use NMS
                            ), dtype=np.float32)
                            , 2,
                            config['gripper_width'],
                            config['thickness'],
                            config['hand_height'],
                            config['thickness_side'],
                            4 # num threads
                        ), dtype=np.float32)
        return grasping_candidates

    def build_point_cloud(self, depth_img, intrinsic, trans_mat):
        inv_intrinsic = np.linalg.pinv(intrinsic) # (3, 3)
        y = np.arange(depth_img.shape[0]-1, -0.5, -1)
        x = np.arange(depth_img.shape[1]-1, -0.5, -1)
        xv, yv = np.meshgrid(x, y)
        xy = np.append(xv[np.newaxis], yv[np.newaxis], axis=0) # (2, H, W)
        xy_homogeneous = np.pad(xy, ((0,1),(0,0),(0,0)), mode='constant', constant_values=1) # (3, H, W)
        xy_homogeneous_shape = xy_homogeneous.shape
        xy_h_flat = xy_homogeneous.reshape(3, -1) # (3, H*W) 
        xy_h_flat_t = np.dot(inv_intrinsic, xy_h_flat) # (3,3) x (3, H*W) -> (3, H*W)
        xy_homogeneous_t = xy_h_flat_t.reshape(xy_homogeneous_shape) # (3, H, W)
        
        xyz_T = (xy_homogeneous_t * depth_img[np.newaxis]).reshape(3, -1) # (3, H*W)
        xyz_T_h = np.pad(xyz_T, ((0,1), (0,0)), mode='constant', constant_values=1) # (4, H*W)
        xyz = np.dot(trans_mat, xyz_T_h).reshape(xy_homogeneous_shape) # (3, H, W)
        xyz = np.transpose(xyz, (1, 2, 0)) # (H, W, 3)
        return xyz

    def grasping_filter(self, grasp_list, ang_face, ang_bottom):
        
        grasp_list_filter = []
        for grasp in grasp_list:
            if (np.arccos(np.dot( grasp[:,0].reshape(1,3), np.array([0, -1, 0]).reshape(3,1) )) > np.radians(ang_face) or
            np.arccos(np.dot( grasp[:,0].reshape(1,3), np.array([0, 0, -1]).reshape(3,1) )) > np.radians(ang_bottom)):
                continue
            grasp_list_filter.append(grasp)
        return np.array(grasp_list_filter)


if __name__ == '__main__':
    

    g = Grasper()
    
    ### test
    pcd = np.random.randn(100,100,3).astype(np.float32)

    #pcd = g.build_point_cloud(depth_img, intrinsic, extrinsic).astype(np.float32)
    #np.save('dump.npy', point_cloud.reshape(-1, 3))
    print(g.get_grasping_candidates(pcd))

