from env.single_robotic_arm import SingleRoboticArm
import numpy as np
import cv2
import math
from transforms3d.quaternions import mat2quat, quat2axangle, quat2mat, qmult
import random
from scipy.spatial.transform import Rotation as R
#from inference_pointnet2_kpt import CoarseMover, FineMover
from inference_pointnet2_kpts import CoarseMover
from inference_pointnet2_offset import FineMover
from inference_pointnet2_reg import PointnetMover
from inference_pointnet2_dsae import DSAEMover
import os
import sys
import time
import copy
import argparse
import open3d as o3d
sys.path.append('/home/luben/robot-peg-in-hole-task')

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    _EPS = np.finfo(float).eps * 4.0
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def random_tilt(rob_arm, obj_name_list, min_tilt_degree, max_tilt_degree):
    while True:
        u = random.uniform(0, 1)
        v = random.uniform(0, 1)
        theta = 2 * math.pi * u
        phi = math.acos(2 * v - 1)
        x = math.sin(theta) * math.sin(phi)
        y = math.cos(theta) * math.sin(phi)
        z = math.cos(phi)
        dst_hole_dir = np.array([x, y, z])  # world coordinate
        src_hole_dir = np.array([0, 0, 1])  # world coordinate

        cross_product = np.cross(src_hole_dir, dst_hole_dir)
        if cross_product.nonzero()[0].size == 0:  # to check if it is zero vector
            rot_dir = np.array([0, 0, 1])
        else:
            rot_dir = cross_product / np.linalg.norm(cross_product)
        dot_product = np.dot(src_hole_dir, dst_hole_dir)
        tilt_degree = math.degrees(
            math.acos(dot_product / (np.linalg.norm(src_hole_dir) * np.linalg.norm(dst_hole_dir))))
        if abs(tilt_degree) <= max_tilt_degree and abs(tilt_degree) >= min_tilt_degree:
            break
    print('rot_dir:', rot_dir)
    print('tilt degree:', tilt_degree)
    w = math.cos(math.radians(tilt_degree / 2))
    x = math.sin(math.radians(tilt_degree / 2)) * rot_dir[0]
    y = math.sin(math.radians(tilt_degree / 2)) * rot_dir[1]
    z = math.sin(math.radians(tilt_degree / 2)) * rot_dir[2]
    rot_quat = [w, x, y, z]
    for obj_name in obj_name_list:
        obj_quat = rob_arm.get_object_quat(obj_name)  # [x,y,z,w]
        obj_quat = [obj_quat[3], obj_quat[0], obj_quat[1], obj_quat[2]]  # change to [w,x,y,z]
        obj_quat = qmult(rot_quat, obj_quat)  # [w,x,y,z]
        obj_quat = [obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]  # change to [x,y,z,w]
        rob_arm.set_object_quat(obj_name, obj_quat)

    return rot_dir, tilt_degree

def random_tilt_2d(rob_arm, obj_name_list, tilt_degree):
    u = random.uniform(0, 1)
    v = math.sqrt(1 - u**2)
    rot_dir = np.array([u, v, 0])
    rot_dir = rot_dir / np.linalg.norm(rot_dir)
    if random.uniform(0, 1) > 0.5:
        pass
    else:
        rot_dir = -rot_dir

    print('rot_dir:', rot_dir)
    print('tilt degree:', tilt_degree)
    w = math.cos(math.radians(tilt_degree / 2))
    x = math.sin(math.radians(tilt_degree / 2)) * rot_dir[0]
    y = math.sin(math.radians(tilt_degree / 2)) * rot_dir[1]
    z = math.sin(math.radians(tilt_degree / 2)) * rot_dir[2]
    rot_quat = [w, x, y, z]
    for obj_name in obj_name_list:
        obj_quat = rob_arm.get_object_quat(obj_name)  # [x,y,z,w]
        obj_quat = [obj_quat[3], obj_quat[0], obj_quat[1], obj_quat[2]]  # change to [w,x,y,z]
        obj_quat = qmult(rot_quat, obj_quat)  # [w,x,y,z]
        obj_quat = [obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]  # change to [x,y,z,w]
        rob_arm.set_object_quat(obj_name, obj_quat)

    return rot_dir, tilt_degree

def specific_tilt(rob_arm, obj_name_list, rot_dir, tilt_degree):
    rot_dir = rot_dir / np.linalg.norm(rot_dir)
    w = math.cos(math.radians(tilt_degree / 2))
    x = math.sin(math.radians(tilt_degree / 2)) * rot_dir[0]
    y = math.sin(math.radians(tilt_degree / 2)) * rot_dir[1]
    z = math.sin(math.radians(tilt_degree / 2)) * rot_dir[2]
    rot_quat = [w, x, y, z]
    for obj_name in obj_name_list:
        obj_quat = rob_arm.get_object_quat(obj_name)  # [x,y,z,w]
        obj_quat = [obj_quat[3], obj_quat[0], obj_quat[1], obj_quat[2]]  # change to [w,x,y,z]
        obj_quat = qmult(rot_quat, obj_quat)  # [w,x,y,z]
        obj_quat = [obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]  # change to [x,y,z,w]
        rob_arm.set_object_quat(obj_name, obj_quat)

def get_init_pos(hole_pos, peg_pos):
    # position based on hole
    #x = random.uniform(0.1,0.375)
    #y = random.uniform(-0.65,-0.375)
    x = 0.2
    y = -0.5
    hole_pos[0] = x
    hole_pos[1] = y
    peg_pos[0] = x
    peg_pos[1] = y
    #peg_pos[2] = 7.1168e-02
    peg_pos[2] = 6.1368e-02
    return hole_pos, peg_pos

def predict_xyzrot_from_single_camera(cam_name, mover, rob_arm):
    gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
    rgb = rob_arm.get_rgb(cam_name=cam_name)
    depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=1.5)

    # rotate 180
    (h, w) = rgb.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rgb = cv2.warpAffine(rgb, M, (w, h))
    # rotate 180
    (h, w) = depth.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    depth = cv2.warpAffine(depth, M, (w, h))

    depth_mm = (depth * 1000).astype(np.uint16)  # type: np.uint16 ; uint16 is needed by keypoint detection network
    delta_rot_pred, delta_xyz_pred = mover.inference(depth_mm)
    delta_rot_pred = delta_rot_pred.cpu().numpy()
    delta_xyz_pred = delta_xyz_pred.cpu().numpy()

    return delta_rot_pred, delta_xyz_pred

# for regression approach
def predict_xyzrot_from_multiple_camera(cam_name_list, mover, rob_arm):
    gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
    depth_mm_list = []
    camera2world_list = []
    for cam_name in cam_name_list:
        cam_matrix = rob_arm.get_object_matrix(obj_name=cam_name)
        camera2world_list.append(cam_matrix)
        depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=1.5)
        # rotate 180
        (h, w) = depth.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        depth = cv2.warpAffine(depth, M, (w, h))
        depth_mm = (depth * 1000).astype(np.uint16)  # type: np.uint16 ; uint16 is needed by keypoint detection network
        depth_mm_list.append(depth_mm)

    delta_rot_pred, delta_xyz_pred = mover.inference_multiple_camera(depth_mm_list, camera2world_list)
    delta_rot_pred = delta_rot_pred.cpu().numpy()
    delta_xyz_pred = delta_xyz_pred.cpu().numpy()

    return delta_rot_pred, delta_xyz_pred

# for fine approach (pointnet2_kpt) (offset included)
def predict_kpt_xyz_from_multiple_camera(cam_name_list, gripper_pose, mover, rob_arm):
    depth_mm_list = []
    camera2world_list = []
    for cam_name in cam_name_list:
        cam_matrix = rob_arm.get_object_matrix(obj_name=cam_name)
        camera2world_list.append(cam_matrix)
        depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=1.5)
        # rotate 180
        (h, w) = depth.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        depth = cv2.warpAffine(depth, M, (w, h))
        depth_mm = (depth * 1000).astype(np.uint16)  # type: np.uint16 ; uint16 is needed by keypoint detection network
        depth_mm_list.append(depth_mm)
    delta_trans_pred = mover.inference_multiple_camera(depth_mm_list, camera2world_list, gripper_pose)

    return delta_trans_pred

# for coarse & fine approach (pointnet2_kpts) (offset included)
def predict_kpts_xyzrot_from_multiple_camera(cam_name_list, gripper_pose, mover, rob_arm):
    depth_mm_list = []
    camera2world_list = []
    for cam_name in cam_name_list:
        cam_matrix = rob_arm.get_object_matrix(obj_name=cam_name)
        camera2world_list.append(cam_matrix)
        depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=1.5)
        # rotate 180
        (h, w) = depth.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        depth = cv2.warpAffine(depth, M, (w, h))
        depth_mm = (depth * 1000).astype(np.uint16)  # type: np.uint16 ; uint16 is needed by keypoint detection network
        depth_mm_list.append(depth_mm)
    delta_trans_pred, delta_rot_pred = mover.inference_multiple_camera(depth_mm_list, camera2world_list, gripper_pose)

    return delta_trans_pred, delta_rot_pred

# for coarse & fine approach (pointnet2_kpts) (no offset)
def predict_kpts_no_oft_from_multiple_camera(cam_name_list, gripper_pose, mover, rob_arm):
    depth_mm_list = []
    camera2world_list = []
    for cam_name in cam_name_list:
        cam_matrix = rob_arm.get_object_matrix(obj_name=cam_name)
        camera2world_list.append(cam_matrix)
        depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=1.5)
        # rotate 180
        (h, w) = depth.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        depth = cv2.warpAffine(depth, M, (w, h))
        depth_mm = (depth * 1000).astype(np.uint16)  # type: np.uint16 ; uint16 is needed by keypoint detection network
        depth_mm_list.append(depth_mm)

    points, pcd_centroid, pcd_mean = mover.process_raw_mutliple_camera(depth_mm_list, camera2world_list, add_noise=False)
    real_kpt_pred, dir_pred, rot_mat_pred, confidence = mover.inference_from_pcd(points, pcd_centroid, pcd_mean, use_offset=False)
    real_kpt_pred = real_kpt_pred / 1000  # unit: mm to m
    gripper_pos = gripper_pose[:3, 3] #(3,)
    gripper_rot = gripper_pose[:3, :3] #(3, 3)
    delta_trans_pred = real_kpt_pred - gripper_pos #(3,)
    delta_rot_pred = np.dot(np.transpose(gripper_rot), rot_mat_pred)

    return delta_trans_pred, delta_rot_pred

# for fine approach (pointnet2_offset)
def predict_offset_from_multiple_camera(cam_name_list, gripper_pose, mover, rob_arm, crop_pcd=False, method='fine'):
    depth_mm_list = []
    camera2world_list = []
    for cam_name in cam_name_list:
        cam_matrix = rob_arm.get_object_matrix(obj_name=cam_name)
        camera2world_list.append(cam_matrix)
        depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=1.5)
        # rotate 180
        (h, w) = depth.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        depth = cv2.warpAffine(depth, M, (w, h))
        depth_mm = (depth * 1000).astype(np.uint16)  # type: np.uint16 ; uint16 is needed by keypoint detection network
        depth_mm_list.append(depth_mm)

    points, pcd_centroid, pcd_mean = mover.process_raw_mutliple_camera(depth_mm_list, camera2world_list, add_noise=False)
    if crop_pcd == True:
        points, pcd_centroid, pcd_mean = mover.crop_pcd(points, pcd_centroid, pcd_mean, gripper_pose[:3, 3])
    delta_xyz_pred, delta_rot_pred, delta_rot_euler_pred = mover.inference_from_pcd(points, pcd_centroid, pcd_mean, method)

    return delta_xyz_pred, delta_rot_pred, delta_rot_euler_pred

def predict_dsae_xyzrot_from_multiple_camera(cam_name_list, mover, rob_arm):
    #camera2world_list = []
    for idx, cam_name in enumerate(cam_name_list):
        #cam_matrix = rob_arm.get_object_matrix(obj_name=cam_name)
        #camera2world_list.append(cam_matrix)
        rgb = rob_arm.get_rgb(cam_name=cam_name)
        # rotate 180
        (h, w) = rgb.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rgb = cv2.warpAffine(rgb, M, (w, h))
        if idx == 0:
            stacked_rgb = copy.deepcopy(rgb)
        else:
            stacked_rgb = np.append(stacked_rgb, rgb, axis=2)

    delta_xyz_pred, delta_rot_pred = mover.inference_multiple_camera(stacked_rgb)

    return delta_xyz_pred, delta_rot_pred

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=250)

    return parser.parse_args()

def main(args):
    # create folder
    #benchmark_folder = 'pcd_benchmark/reg_seg_heatmap_v3'
    benchmark_folder = 'pcd_benchmark/offset'
    if not os.path.exists(benchmark_folder):
        os.makedirs(benchmark_folder)
    coarse_mover = FineMover(model_path='offset/2022-05-26_02-15', model_name='pointnet2_offset',
                           checkpoint_name='best_model_e_38.pth', use_cpu=False, out_channel=9)
    #noisecoarse_mover = CoarseMover(model_path='kpts/2022-04-25_07-26', model_name='pointnet2_kpts', checkpoint_name='best_model_e_101.pth', use_cpu=False, out_channel=9)
    #fine_mover = FineMover(model_path='offset/2022-04-26_22-24', model_name='pointnet2_offset', checkpoint_name='best_model_e_60.pth', use_cpu=False, out_channel=9)
    fine_mover = FineMover(model_path='offset/2022-05-19_22-32', model_name='pointnet2_offset',
                           checkpoint_name='best_model_e_95.pth', use_cpu=False, out_channel=9)
    #noisefine_mover = FineMover(model_path='offset/2022-04-25_07-09', model_name='pointnet2_offset', checkpoint_name='best_model_e_64.pth', use_cpu=False, out_channel=9)
    #fine_mover = DSAEMover(model_path='dsae/2022-04-19_15-48', model_name='cnn_dsae', checkpoint_name='best_model_e_72.pth', use_cpu=False, out_channel=9)

    iter_num = args.iter
    gripper_init_move = False
    tilt = True
    #cam_name_list = ['vision_eye_left', 'vision_eye_right']
    cam_name_list = ['vision_eye_front']
    peg_top = 'peg_dummy_top'
    peg_bottom = 'peg_dummy_bottom'
    peg_name = 'peg_in_arm'
    '''
    hole_setting = {'square': ['square', 'hole_keypoint_top0', 'hole_keypoint_bottom0'],
                    'small_square': ['small_square', 'hole_keypoint_top1', 'hole_keypoint_bottom1'],
                    'circle': ['circle', 'hole_keypoint_top2', 'hole_keypoint_bottom2'],
                    'rectangle': ['rectangle', 'hole_keypoint_top3', 'hole_keypoint_bottom3'],
                    'triangle': ['triangle', 'hole_keypoint_top4', 'hole_keypoint_bottom4'],}
    '''
    hole_setting = {'square_7x10x10': ['square_7x10x10', 'hole_keypoint_top0', 'hole_keypoint_bottom0', 'hole_keypoint_obj_bottom0'],
                    'square_7x11x11': ['square_7x11x11', 'hole_keypoint_top1', 'hole_keypoint_bottom1', 'hole_keypoint_obj_bottom1'],
                    'square_7x12x12': ['square_7x12x12', 'hole_keypoint_top2', 'hole_keypoint_bottom2', 'hole_keypoint_obj_bottom2'],
                    'square_7x13x13': ['square_7x13x13', 'hole_keypoint_top3', 'hole_keypoint_bottom3', 'hole_keypoint_obj_bottom3'],
                    'square_7x14x14': ['square_7x14x14', 'hole_keypoint_top4', 'hole_keypoint_bottom4', 'hole_keypoint_obj_bottom4'],
                    'rectangle_7x8x11': ['rectangle_7x8x11', 'hole_keypoint_top5', 'hole_keypoint_bottom5', 'hole_keypoint_obj_bottom5'],
                    'rectangle_7x9x12': ['rectangle_7x9x12', 'hole_keypoint_top6', 'hole_keypoint_bottom6', 'hole_keypoint_obj_bottom6'],
                    'rectangle_7x10x13': ['rectangle_7x10x13', 'hole_keypoint_top7', 'hole_keypoint_bottom7', 'hole_keypoint_obj_bottom7'],
                    'rectangle_7x11x14': ['rectangle_7x11x14', 'hole_keypoint_top8', 'hole_keypoint_bottom8', 'hole_keypoint_obj_bottom8'],
                    'rectangle_7x12x15': ['rectangle_7x12x15', 'hole_keypoint_top9', 'hole_keypoint_bottom9', 'hole_keypoint_obj_bottom9'],
                    'circle_7x10': ['circle_7x10', 'hole_keypoint_top10', 'hole_keypoint_bottom10', 'hole_keypoint_obj_bottom10'],
                    'circle_7x11': ['circle_7x11', 'hole_keypoint_top11', 'hole_keypoint_bottom11', 'hole_keypoint_obj_bottom11'],
                    'circle_7x12': ['circle_7x12', 'hole_keypoint_top12', 'hole_keypoint_bottom12', 'hole_keypoint_obj_bottom12'],
                    'circle_7x13': ['circle_7x13', 'hole_keypoint_top13', 'hole_keypoint_bottom13', 'hole_keypoint_obj_bottom13'],
                    'circle_7x14': ['circle_7x14', 'hole_keypoint_top14', 'hole_keypoint_bottom14', 'hole_keypoint_obj_bottom14'],
                    'octagon_7x5': ['octagon_7x5', 'hole_keypoint_top15', 'hole_keypoint_bottom15', 'hole_keypoint_obj_bottom15'],
                    'pentagon_7x7': ['pentagon_7x7', 'hole_keypoint_top16', 'hole_keypoint_bottom16', 'hole_keypoint_obj_bottom16'],
                    'hexagon_7x6': ['hexagon_7x6', 'hole_keypoint_top17', 'hole_keypoint_bottom17', 'hole_keypoint_obj_bottom17'], }

    #selected_hole_list = ['square_7x10x10', 'square_7x11x11', 'square_7x12x12', 'square_7x13x13', 'square_7x14x14']
    #selected_hole_list = ['rectangle_7x8x11', 'rectangle_7x9x12', 'rectangle_7x10x13', 'rectangle_7x11x14', 'rectangle_7x12x15']
    #selected_hole_list = ['circle_7x10', 'circle_7x11', 'circle_7x12', 'circle_7x13', 'circle_7x14']
    #selected_hole_list = ['square_7x12x12', 'square_7x14x14', 'square_7x10x10', 'rectangle_7x8x11', 'rectangle_7x10x13', 'rectangle_7x12x15', 'circle_7x10', 'circle_7x12', 'circle_7x14', 'octagon_7x5', 'pentagon_7x7', 'hexagon_7x6']
    selected_hole_list = ['square_7x12x12', 'square_7x10x10', 'rectangle_7x8x11', 'rectangle_7x10x13', 'circle_7x10', 'circle_7x12', 'circle_7x14', 'pentagon_7x7', 'hexagon_7x6', 'octagon_7x5']
    for selected_hole in selected_hole_list:
        f = open(os.path.join(benchmark_folder, "hole_score.txt"), "a")
        rob_arm = SingleRoboticArm()
        hole_name = hole_setting[selected_hole][0]
        hole_top = hole_setting[selected_hole][1]
        hole_bottom = hole_setting[selected_hole][2]
        gripper_init_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
        origin_hole_pose = rob_arm.get_object_matrix(hole_name)
        origin_hole_pos = origin_hole_pose[:3, 3]
        origin_hole_quat = rob_arm.get_object_quat(hole_name)
        rob_arm.finish()

        insertion_succ_list = []
        succ_kpt_error_list = []
        succ_kpt_yz_error_list = []
        succ_dir_error_list = []
        fail_kpt_error_list = []
        fail_kpt_yz_error_list = []
        fail_dir_error_list = []
        kpt_error_list = []
        kpt_yz_error_list = []
        dir_error_list = []
        skip_cnt = 0
        for iter in range(iter_num):
            rob_arm = SingleRoboticArm()
            print('=' * 8 + str(iter) + '=' * 8)
            rob_arm.movement(gripper_init_pose)
            # set init pos of peg nad hole
            if gripper_init_move == True:
                hole_pos = np.array([0.2, -0.5, 3.6200e-02])
                rob_arm.set_object_position(hole_name, hole_pos)
                rob_arm.set_object_quat(hole_name, origin_hole_quat)
                if tilt:
                    _, tilt_degree = random_tilt(rob_arm, [hole_name], 0, 50)
                # start pose
                delta_move = np.array([random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03), random.uniform(0.10, 0.12)])
                start_pose = rob_arm.get_object_matrix('UR5_ikTip')
                hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
                start_pos = hole_top_pose[:3, 3]
                start_pos += delta_move
                start_pose[:3, 3] = start_pos
                rob_arm.movement(start_pose)
            else:
                #hole_pos = np.array([random.uniform(0.0, 0.2), random.uniform(-0.45, -0.55), 0.035])
                hole_pos = np.array([random.uniform(0.02, 0.18), random.uniform(-0.52574, -0.44574), 0.035]) #np.array([random.uniform(0.0, 0.2), random.uniform(-0.45, -0.55), 0.035])
                rob_arm.set_object_position(hole_name, hole_pos)
                rob_arm.set_object_quat(hole_name, origin_hole_quat)
                if tilt:
                    _, tilt_degree = random_tilt(rob_arm, [hole_name], 0, 50)

            '''
            # (test)move to hole top
            for i in range(2):
                hole_insert_dir = - rob_arm.get_object_matrix(hole_top)[:3, 0]  # x-axis
                peg_insert_dir = - rob_arm.get_object_matrix(peg_top)[:3, 0]  # x-axis
                dot_product = np.dot(peg_insert_dir, hole_insert_dir)
                degree = math.degrees(
                    math.acos(dot_product / (np.linalg.norm(peg_insert_dir) * np.linalg.norm(hole_insert_dir))))
                print('degree between peg and hole : ', degree)
                if degree > 0 and degree < 70:
                    cross_product = np.cross(peg_insert_dir, hole_insert_dir)
                    cross_product = cross_product / np.linalg.norm(cross_product)
                    w = math.cos(math.radians(degree / 2))
                    x = math.sin(math.radians(degree / 2)) * cross_product[0]
                    y = math.sin(math.radians(degree / 2)) * cross_product[1]
                    z = math.sin(math.radians(degree / 2)) * cross_product[2]
                    quat = [w, x, y, z]
                    rot_pose = quaternion_matrix(quat)
                    robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
                    rot_matrix = np.dot(rot_pose[:3, :3], robot_pose[:3, :3])
                    robot_pose[:3, :3] = rot_matrix
                    hole_keypoint_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
                    robot_pose[:3, 3] = hole_keypoint_top_pose[:3, 3] #+ hole_keypoint_top_pose[:3, 0] * 0.01
                    rob_arm.movement(robot_pose)
            # tilt gripper randomly to test
            #degree = args.tilt_gripper
            #_, tilt_degree = random_tilt(rob_arm, ['UR5_ikTarget'], degree, degree + 0.1)
            #_, tilt_degree = random_tilt_2d(rob_arm, ['UR5_ikTarget'], degree)
            '''

            # coarse approach
            gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
            delta_xyz_pred, delta_rot_pred, delta_rot_euler_pred = predict_offset_from_multiple_camera(
                cam_name_list, gripper_pose, coarse_mover, rob_arm, crop_pcd=False, method='coarse')
            print(delta_xyz_pred)
            print(delta_rot_euler_pred)
            rot_matrix = np.dot(delta_rot_pred[:3, :3], gripper_pose[:3, :3])
            gripper_pose[:3, :3] = rot_matrix
            gripper_pose[:3, 3] += delta_xyz_pred  # (3,)
            gripper_pose[:3, 3] += np.array(rot_matrix[:3, 0] * 0.010)
            rob_arm.movement(gripper_pose)

            '''
            if delta_xyz_pred[2] > 0:
                print('crash!' )
                insertion_succ_list.append(0)
                rob_arm.finish()
                continue
            '''
            '''
            r = R.from_matrix(delta_rot_pred)
            r_euler = r.as_euler('zyx', degrees=True)
            if abs(r_euler[0]) < 90 and abs(r_euler[1]) < 90 and abs(r_euler[2]) < 90:
            '''
            '''
            gripper_pose_after = rob_arm.get_object_matrix(obj_name='UR5_ikTip')
            dist = np.linalg.norm(gripper_pose_after[:3, 3] - gripper_pose[:3, 3])
            if dist > 0.0005:
                rob_arm.finish()
                print('skip', dist)
                skip_cnt = skip_cnt + 1
                continue
            '''

            # compute error
            peg_hole_dir = (rob_arm.get_object_matrix(hole_top)[:3, 3] - rob_arm.get_object_matrix(peg_bottom)[:3, 3]).reshape(1, 3)
            hole_dir = rob_arm.get_object_matrix(hole_top)[:3, 0].reshape(3, 1)
            c_kpt_error = np.linalg.norm(peg_hole_dir) * 1000  # unit:mm
            dot_product = np.dot(peg_hole_dir, hole_dir)
            angle = math.acos(dot_product / (np.linalg.norm(peg_hole_dir) * np.linalg.norm(hole_dir)))  # rad
            peg_hole_dis = np.linalg.norm(peg_hole_dir)
            c_kpt_yz_error = peg_hole_dis * math.sin(angle) *1000
            print('coarse keypiont error', c_kpt_error)
            print('coarse keypiont yz error', c_kpt_yz_error)
            peg_dir = rob_arm.get_object_matrix(peg_bottom)[:3, 0].reshape(1, 3)
            hole_dir = rob_arm.get_object_matrix(hole_top)[:3, 0].reshape(3, 1)
            dot_product = np.dot(peg_dir, hole_dir)
            c_dir_error = math.degrees(math.acos(dot_product / (np.linalg.norm(peg_dir) * np.linalg.norm(hole_dir))))
            print('coarse direction error', c_dir_error)
            '''
            if c_dir_error > 30:
                print('crash! Angle is too large')
                print(c_dir_error)
                insertion_succ_list.append(0)
                rob_arm.finish()
                continue
            '''

            # fine approach
            # closed-loop
            cnt = 0
            while True:
                ### start
                gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
                #delta_xyz_pred = predict_kpt_xyz_from_multiple_camera(cam_name_list, gripper_pose, fine_mover, rob_arm)
                #delta_xyz_pred, _ = predict_kpts_xyzrot_from_multiple_camera(cam_name_list, gripper_pose, fine_mover, rob_arm)
                #delta_xyz_pred, delta_rot_pred = predict_kpts_no_oft_from_multiple_camera(cam_name_list, gripper_pose, fine_mover, rob_arm)
                #delta_xyz_pred, delta_rot_pred = predict_dsae_xyzrot_from_multiple_camera(cam_name_list, fine_mover, rob_arm)
                delta_xyz_pred, delta_rot_pred, delta_rot_euler_pred = predict_offset_from_multiple_camera(cam_name_list, gripper_pose, fine_mover, rob_arm, crop_pcd=True)
                step_size = np.linalg.norm(delta_xyz_pred)
                print(delta_xyz_pred)
                print(delta_rot_euler_pred)
                rot_matrix = np.dot(delta_rot_pred[:3, :3], gripper_pose[:3, :3])
                gripper_pose[:3, :3] = rot_matrix
                gripper_pose[:3, 3] += delta_xyz_pred #(3,)
                rob_arm.movement(gripper_pose)

                # compute error
                peg_hole_dir = (rob_arm.get_object_matrix(hole_top)[:3, 3] - rob_arm.get_object_matrix(peg_bottom)[:3,3]).reshape(1, 3)
                f_kpt_error = np.linalg.norm(peg_hole_dir) * 1000  # unit:mm
                hole_dir = rob_arm.get_object_matrix(hole_top)[:3, 0].reshape(3, 1)
                dot_product = np.dot(peg_hole_dir, hole_dir)
                angle = math.acos(dot_product / (np.linalg.norm(peg_hole_dir) * np.linalg.norm(hole_dir)))  # rad
                peg_hole_dis = np.linalg.norm(peg_hole_dir)
                f_kpt_yz_error = peg_hole_dis * math.sin(angle) * 1000
                print('fine keypiont error', f_kpt_error)
                print('fine keypiont_yz error', f_kpt_yz_error)
                peg_dir = rob_arm.get_object_matrix(peg_bottom)[:3, 0].reshape(1, 3)
                hole_dir = rob_arm.get_object_matrix(hole_top)[:3, 0].reshape(3, 1)
                dot_product = np.dot(peg_dir, hole_dir)
                f_dir_error = math.degrees(
                    math.acos(dot_product / (np.linalg.norm(peg_dir) * np.linalg.norm(hole_dir))))
                print('fine direction error', f_dir_error)
                if f_dir_error > 10.0:
                    print('crash! Angle is too large.')
                    break
                if (step_size < 0.005 and (abs(delta_rot_euler_pred)< 1.5).all()) or cnt >= 5 :
                    print('servoing done!')
                    break
                cnt = cnt + 1

            # insertion
            robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            robot_pose[:3, 3] -= robot_pose[:3, 0] * 0.08  # x-axis
            rob_arm.movement(robot_pose)
            # record insertion
            peg_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name=peg_bottom)
            hole_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name=hole_bottom)
            dist = np.linalg.norm(peg_keypoint_bottom_pose[:3, 3] - hole_keypoint_bottom_pose[:3, 3])
            #print('dist', dist)
            #f.write(str(tilt_degree) + ' ' + str(dist) + '\n')
            kpt_error_list.append(f_kpt_error)
            kpt_yz_error_list.append(f_kpt_yz_error)
            dir_error_list.append(f_dir_error)
            if dist < 0.010:
                print('success')
                succ_kpt_error_list.append(f_kpt_error)
                succ_kpt_yz_error_list.append(f_kpt_yz_error)
                succ_dir_error_list.append(f_dir_error)
                insertion_succ_list.append(1)
            else:
                print('fail')
                fail_kpt_error_list.append(f_kpt_error)
                fail_kpt_yz_error_list.append(f_kpt_yz_error)
                fail_dir_error_list.append(f_dir_error)
                insertion_succ_list.append(0)

            rob_arm.finish()
            '''
            # pull up
            robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            robot_pose[:3, 3] += robot_pose[:3, 0] * 0.1  # x-axis
            rob_arm.movement(robot_pose)
            '''
        insertion_succ = sum(insertion_succ_list) / len(insertion_succ_list)
        msg = '    * hole success rate : ' + str(insertion_succ * 100) + '% (' + str(sum(insertion_succ_list)) + '/' + str(len(insertion_succ_list)) + ')'
        print(selected_hole + '\n' + msg )
        f.write('* ' + selected_hole + '\n')
        f.write(msg + '\n')
        #f.write('skip coount:' + str(skip_cnt) + '\n')
        kpt_error = sum(kpt_error_list) / len(kpt_error_list)
        kpt_yz_error = sum(kpt_yz_error_list) / len(kpt_yz_error_list)
        dir_error = sum(dir_error_list) / len(dir_error_list)
        f.write("    * total average: kpt{:10.4f}".format(kpt_error) + ' ' + "kpt_yz{:10.4f}".format(kpt_yz_error) + ' ' + "dir{:10.4f}".format(dir_error) + '\n')
        '''
        fine_kpt_error = sum(fine_kpt_error_list) / len(fine_kpt_error_list)
        fine_dir_error = sum(fine_dir_error_list)/ len(fine_dir_error_list)
        f.write(str(coarse_kpt_error) + ' ' + str(coarse_dir_error) + ' ' + str(fine_kpt_error) + ' ' + str(fine_dir_error) + '\n')
        '''
        f_c = open(os.path.join(benchmark_folder, selected_hole + "_kpt_dir_error.txt"), "w")
        f_c.write('succ:\n')
        for i in range(len(succ_kpt_error_list)):
            kpt_error = succ_kpt_error_list[i]
            kpt_yz_error = succ_kpt_yz_error_list[i]
            dir_error = succ_dir_error_list[i]
            f_c.write("{:10.4f}".format(kpt_error) + "{:10.4f}".format(kpt_yz_error) + "{:10.4f}".format(dir_error) + '\n')
        if len(succ_kpt_error_list) != 0 and len(succ_dir_error_list) != 0:
            succ_kpt_error = sum(succ_kpt_error_list)/len(succ_kpt_error_list)
            succ_kpt_yz_error = sum(succ_kpt_yz_error_list) / len(succ_kpt_yz_error_list)
            succ_dir_error = sum(succ_dir_error_list) / len(succ_dir_error_list)
            f_c.write("Average:" + "{:10.4f}".format(succ_kpt_error) + "{:10.4f}".format(succ_kpt_yz_error) + "{:10.4f}".format(succ_dir_error) + '\n')
            f.write("    * succ average:" + "kpt{:10.4f}".format(succ_kpt_error) + "kpt_yz{:10.4f}".format(succ_kpt_yz_error) + "dir{:10.4f}".format(succ_dir_error) + '\n')

        f_c.write('fail:\n')
        for i in range(len(fail_kpt_error_list)):
            kpt_error = fail_kpt_error_list[i]
            kpt_yz_error = fail_kpt_yz_error_list[i]
            dir_error = fail_dir_error_list[i]
            f_c.write("{:10.4f}".format(kpt_error) + "{:10.4f}".format(kpt_yz_error) + "{:10.4f}".format(dir_error) + '\n')
        if len(fail_kpt_error_list) != 0 and len(fail_dir_error_list) != 0:
            fail_kpt_error = sum(fail_kpt_error_list) / len(fail_kpt_error_list)
            fail_kpt_yz_error = sum(fail_kpt_yz_error_list) / len(fail_kpt_yz_error_list)
            fail_dir_error = sum(fail_dir_error_list) / len(fail_dir_error_list)
            f_c.write("Average:" + "{:10.4f}".format(fail_kpt_error) +  "{:10.4f}".format(fail_kpt_yz_error) + "{:10.4f}".format(fail_dir_error) + '\n')
            f.write("    * fail average:" + "kpt{:10.4f}".format(fail_kpt_error) + "kpt_yz{:10.4f}".format(fail_kpt_yz_error) + "dir{:10.4f}".format(fail_dir_error) + '\n')
        f.close()
        f_c.close()
        #rob_arm.finish()


if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print('Time elasped:{:.02f}'.format((end_time - start_time) / 3600))
