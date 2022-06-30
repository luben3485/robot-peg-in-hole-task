import os
'''HYPER PARAMETER'''
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from env.single_robotic_arm import SingleRoboticArm
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import math
from transforms3d.quaternions import mat2quat, quat2axangle, quat2mat, qmult
import random
from scipy.spatial.transform import Rotation as R
from inference_pointnet2_kpts import CoarseMover
from inference_dsae import DSAEMover
from inference_dsae_bykovis import DSAEMoverByKovis

import sys
import time
import copy
from config.hole_setting import hole_setting
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

def random_yaw(rob_arm, obj_name_list, degree=45):
    for obj_name in obj_name_list:
        yaw_degree = random.uniform(-math.radians(degree), math.radians(degree))
        rot_dir = rob_arm.get_object_matrix(obj_name)[:3, 0]
        if obj_name in ['pentagon_7x7_squarehole', 'pentagon_7x9_squarehole', 'rectangle_7x9x12_squarehole', 'rectangle_7x10x13_squarehole']:
            rot_dir = rob_arm.get_object_matrix(obj_name)[:3, 1]
        w = math.cos(yaw_degree / 2)
        x = math.sin(yaw_degree / 2) * rot_dir[0]
        y = math.sin(yaw_degree / 2) * rot_dir[1]
        z = math.sin(yaw_degree / 2) * rot_dir[2]
        rot_quat = [w, x, y, z]

        obj_quat = rob_arm.get_object_quat(obj_name)  # [x,y,z,w]
        obj_quat = [obj_quat[3], obj_quat[0], obj_quat[1 ], obj_quat[2]]  # change to [w,x,y,z]
        obj_quat = qmult(rot_quat, obj_quat)  # [w,x,y,z]
        obj_quat = [obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]  # change to [x,y,z,w]
        rob_arm.set_object_quat(obj_name, obj_quat)

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

def depth_2_pcd(depth, factor, K):
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

def process_raw_mutliple_camera(depth_mm_list, camera2world_list, noise):
    focal_length = 309.019
    principal = 128
    factor = 1
    intrinsic_matrix = np.array([[focal_length, 0, principal],
                                 [0, focal_length, principal],
                                 [0, 0, 1]], dtype=np.float64)
    xyz_in_world_list = []
    for idx, depth_mm in enumerate(depth_mm_list):
        depth = depth_mm / 1000  # unit: mm to m
        xyz, choose = depth_2_pcd(depth, factor, intrinsic_matrix)
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
    concat_xyz_in_world = np.array(xyz_in_world_list) #(2, 8000, 3, 1)
    concat_xyz_in_world = concat_xyz_in_world.reshape(-1, 3) #(16000, 3)
    if noise:
        concat_xyz_in_world = jitter_point_cloud(concat_xyz_in_world, sigma=1, clip=3)
    return concat_xyz_in_world

def get_hole_pcd_from_scene_pcd(concat_xyz_in_world, hole_keypoint_top_pose_in_world, hole_keypoint_bottom_pose_in_world):
    hole_keypoint_top_pos_in_world = hole_keypoint_top_pose_in_world[:3, 3]
    hole_keypoint_bottom_pos_in_world = hole_keypoint_bottom_pose_in_world[:3, 3]

    # for segmentation
    x_normal_vector = hole_keypoint_top_pose_in_world[:3, 0]
    x_1 = np.dot(x_normal_vector, hole_keypoint_top_pos_in_world + [0., 0., 0.003])
    x_2 = np.dot(x_normal_vector, hole_keypoint_bottom_pos_in_world - [0., 0., 0.002])
    x_value = np.sort([x_1,x_2])
    y_normal_vector = hole_keypoint_top_pose_in_world[:3, 1]
    y_1 = np.dot(y_normal_vector, hole_keypoint_top_pos_in_world + hole_keypoint_top_pose_in_world[:3,1] * 0.067)
    y_2 = np.dot(y_normal_vector, hole_keypoint_top_pos_in_world - hole_keypoint_top_pose_in_world[:3,1] * 0.067)
    y_value = np.sort([y_1, y_2])
    z_normal_vector = hole_keypoint_top_pose_in_world[:3, 2]
    z_1 = np.dot(z_normal_vector, hole_keypoint_top_pos_in_world + hole_keypoint_top_pose_in_world[:3,2] * 0.067)
    z_2 = np.dot(z_normal_vector, hole_keypoint_top_pos_in_world - hole_keypoint_top_pose_in_world[:3,2] * 0.067)
    z_value = np.sort([z_1, z_2])

    hole_seg_xyz = []
    for xyz in concat_xyz_in_world:
        x = np.dot(x_normal_vector, xyz / 1000)
        y = np.dot(y_normal_vector, xyz / 1000)
        z = np.dot(z_normal_vector, xyz / 1000)

        if x >= x_value[0] and x <= x_value[1] and y >= y_value[0] and y <= y_value[1] and z >= z_value[0] and z <= z_value[1]:
            hole_seg_xyz.append(xyz)

    return  hole_seg_xyz

def get_pcd_from_multi_camera(rob_arm, cam_name_list, hole_top_pose, hole_obj_bottom_pose, noise=False):
    depth_mm_list = []
    camera2world_list = []
    for idx, cam_name in enumerate(cam_name_list):
        # Get depth from camera
        depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=1.5)
        # rotate depth 180
        (h, w) = depth.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        depth = cv2.warpAffine(depth, M, (w, h))
        depth_mm = (depth * 1000).astype(np.uint16)  # type: np.uint16 ; uint16 is needed by keypoint detection network
        depth_mm_list.append(depth_mm)
        cam_matrix = rob_arm.get_object_matrix(obj_name=cam_name)
        camera2world_list.append(cam_matrix)

    scene_xyz = process_raw_mutliple_camera(depth_mm_list, camera2world_list, noise=noise)
    hole_xyz = get_hole_pcd_from_scene_pcd(scene_xyz, hole_top_pose, hole_obj_bottom_pose)

    return scene_xyz, hole_xyz

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    #source_2_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([1, 0.706, 0])
    #source_2_temp.paint_uniform_color([1, 0, 0.706])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([target_temp, source_temp])

def coarse_controller_with_icp(rob_arm, cam_name_list, trans_init, hole_top, hole_obj_bottom, add_noise=False):
    hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
    hole_obj_bottom_pose = rob_arm.get_object_matrix(obj_name=hole_obj_bottom)
    scene_xyz, _ = get_pcd_from_multi_camera(rob_arm, cam_name_list, hole_top_pose, hole_obj_bottom_pose, noise=add_noise)

    # square hole
    #source = o3d.io.read_point_cloud('square_7x12x12_squarehole_for_icp.pcd')
    #source.points = o3d.utility.Vector3dVector(np.asarray(source.points) / 1000)
    # round hole
    source = o3d.io.read_point_cloud('full_hole.pcd')
    print(source)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(scene_xyz/1000)
    threshold = 0.02 # 0.02
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())

    #draw_registration_result(source, target, reg_p2p.transformation)
    transformation = copy.deepcopy(reg_p2p.transformation)
    #print(transformation)
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_xyz)
    o3d.io.write_point_cloud(os.path.join('scene_tilt_pcd.ply'), pcd)
    '''
    return transformation

def predict_kpts_no_oft_from_multiple_camera(cam_name_list, mover, rob_arm, detect_kpt=False, add_noise=False):
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

    points, pcd_centroid, pcd_mean = mover.process_raw_mutliple_camera(depth_mm_list, camera2world_list, add_noise=add_noise)
    real_kpt_pred, dir_pred, rot_mat_pred, confidence = mover.inference_from_pcd(points, pcd_centroid, pcd_mean, use_offset=False)
    real_kpt_pred = real_kpt_pred / 1000  # unit: mm to m
    trans_init = np.zeros((4, 4))
    trans_init[3, 3] = 1.0

    ''' rot_mat_pred is related to gripper
    trans_init[:3, :3] = np.dot(rot_mat_pred, np.transpose(np.array([[0, -1, 0],[0, 0, -1],[1, 0, 0]])))
    trans_init[:3, 3] = real_kpt_pred - np.dot(trans_init[:3, :3], np.array([0., 0, 0.07]).reshape(3, 1))[:, 0]
    '''
    if detect_kpt == True:
        # trans_init[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        trans_init[:3, :3] = np.dot(rot_mat_pred, np.transpose(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])))
        trans_init[:3, 3] = real_kpt_pred - np.dot(trans_init[:3, :3], np.array([0., 0, 0.07]).reshape(3, 1))[:, 0]
    else:
        trans_init[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        trans_init[:3, 3] = np.array([0.1, -0.5, 0.035]) - np.dot(trans_init[:3, :3], np.array([0., 0, 0.07]).reshape(3, 1))[:, 0]

    return trans_init

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

def predict_xyzrot(cam_name_list, mover, rob_arm, tilt, yaw):
    assert len(cam_name_list) == 1

    im = rob_arm.get_rgb(cam_name=cam_name_list[0])
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    delta_xyz, delta_rot_euler, speed = mover.inference(im, visualize=True, tilt=tilt, yaw=yaw)

    return delta_xyz, delta_rot_euler, speed

def main():
    # create folder
    benchmark_folder = 'pcd_benchmark/dsae'
    if not os.path.exists(benchmark_folder):
        os.makedirs(benchmark_folder)
    tilt = False
    yaw = False
    use_kovis = True
    coarse_mover = CoarseMover(model_path='kpts/2022-05-17_21-15', model_name='pointnet2_kpts', checkpoint_name='best_model_e_101.pth', use_cpu=False, out_channel=9)
    if not use_kovis:
        dsae_mover = DSAEMover(ckpt_folder='insert-0626-notilt-noyaw', num=21) #tilt 0512-2 notilt 0506 insert-0616-notilt-yaw
    else:
        dsae_mover = DSAEMoverByKovis(ckpt_folder='insert-0626-notilt-noyaw-bykovis', num=23)
    iter_num = 50
    cam_name_list = ['vision_eye_front']
    peg_top = 'peg_dummy_top'
    peg_bottom = 'peg_dummy_bottom'
    peg_name = 'peg_in_arm'

    #selected_hole_list = ['square_7x10x10', 'square_7x11x11', 'square_7x12x12', 'square_7x13x13', 'square_7x14x14']
    #selected_hole_list = ['rectangle_7x8x11', 'rectangle_7x9x12', 'rectangle_7x10x13', 'rectangle_7x11x14', 'rectangle_7x12x15']
    #selected_hole_list = ['circle_7x10', 'circle_7x11', 'circle_7x12', 'circle_7x13', 'circle_7x14']
    #selected_hole_list = ['octagon_7x5', 'pentagon_7x7', 'hexagon_7x6']
    #selected_hole_list = ['square_7x12x12', 'square_7x10x10', 'square_7x14x14', 'rectangle_7x8x11', 'rectangle_7x10x13', 'rectangle_7x12x15', 'circle_7x10', 'circle_7x12', 'circle_7x14']
    #selected_hole_list = ['rectangle_7x12x13', 'rectangle_7x10x12', 'square_7x11_5x11_5', 'circle_7x14', 'circle_7x10', 'pentagon_7x7', 'octagon_7x5']
    #selected_hole_list = ['square_7x11_5x11_5_squarehole', 'rectangle_7x10x12_squarehole', 'circle_7x14_squarehole', 'pentagon_7x9_squarehole']
    selected_hole_list = ['square_7x11_5x11_5', 'circle_7x12', 'rectangle_7x10x12',  'pentagon_7x7']
    for selected_hole in selected_hole_list:
        f = open(os.path.join(benchmark_folder, "hole_score.txt"), "a")
        rob_arm = SingleRoboticArm()
        hole_name = hole_setting[selected_hole][0]
        hole_top = hole_setting[selected_hole][1]
        hole_bottom = hole_setting[selected_hole][2]
        hole_obj_bottom = hole_setting[selected_hole][3]
        gripper_init_pose = rob_arm.get_object_matrix('UR5_ikTip')
        rob_arm.finish()

        insertion_succ_list = []
        time_list = []
        for iter in range(iter_num):
            rob_arm = SingleRoboticArm()
            print('=' * 8 + str(iter) + '=' * 8)
            hole_pos = [random.uniform(0, 0.2), random.uniform(-0.45, -0.55), +3.5001e-02]
            rob_arm.set_object_position(hole_name, hole_pos)
            if yaw:
                random_yaw(rob_arm, [hole_name], degree=20)
            if tilt:
                _, tilt_degree = random_tilt(rob_arm, [hole_name], 0, 50)

            start_time = time.time()
            # coarse approach with ICP
            trans_init = predict_kpts_no_oft_from_multiple_camera(cam_name_list, coarse_mover, rob_arm, detect_kpt=True, add_noise=True)
            transformation = coarse_controller_with_icp(rob_arm, cam_name_list, trans_init, hole_top, hole_obj_bottom, add_noise=False)
            gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
            gripper_pose[:3, 3] = np.array([0.0, 0.0, 0.07])
            rot_matrix = np.dot(transformation[:3, :3], gripper_pose[:3, :3])
            gripper_pose[:3, :3] = rot_matrix
            gripper_pos = np.append(gripper_pose[:3, 3], 1).reshape(4, 1)
            gripper_pos = np.dot(transformation, gripper_pos)[:3, 0]
            gripper_pose[:3, 3] = gripper_pos
            gripper_pose[:3, 3] += np.array(rot_matrix[:3, 0] * 0.03)
            gripper_pose[:3, 3] -= np.array(rot_matrix[:3, 2] * 0.015)
            if tilt == False and yaw == False:
                gripper_pose[:3, :3] = gripper_init_pose[:3, :3]
            rob_arm.movement(gripper_pose)

            # closed-loop
            cnt = 0
            cnt_end = 0
            while True:
                ### start
                gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
                delta_xyz_pred, delta_rot_euler_red, speed = predict_xyzrot(cam_name_list, dsae_mover, rob_arm, tilt, yaw)

                print('xyz:', delta_xyz_pred)
                print('speed:', speed)
                # bykovis
                if speed > 0.8:  # 3DoF 0.8
                    delta_xyz_pred = xyz * speed * 0.01  # 3DoF 0.01
                else:
                    delta_xyz_pred = xyz * speed * 0.001  # 3DoF 0.001
                ''' #dsae
                if speed > 2.8:
                    delta_xyz_pred = delta_xyz_pred * speed * 0.001
                else:
                    delta_xyz_pred = delta_xyz_pred * speed * 0.001
                '''
                gripper_pose[:3, 3] = gripper_pose[:3, 3] + gripper_pose[:3, 0] * delta_xyz_pred[0] + gripper_pose[:3, 1] * delta_xyz_pred[1] + gripper_pose[:3, 2] * delta_xyz_pred[2]

                if yaw:
                    delta_rot_euler_pred[0] = 0
                    delta_rot_euler_pred[1] = 0

                if tilt and yaw:
                    print(delta_rot_euler_pred)
                    r = R.from_euler('zyx', delta_rot_euler_pred, degrees=True)
                    delta_rot_pred = r.as_matrix()
                    # rot_matrix = np.dot(gripper_pose[:3, :3], delta_rot_pred[:3, :3])
                    rot_matrix = np.dot(delta_rot_pred[:3, :3], gripper_pose[:3, :3])
                    gripper_pose[:3, :3] = rot_matrix

                rob_arm.movement(gripper_pose)

                #avoid crash
                hole_top_pose = rob_arm.get_object_matrix(hole_top)
                if np.linalg.norm(gripper_pose[:3, 3] -  hole_top_pose[:3, 3]) > 0.3:
                    print('crash! Distance between peg and hole is too far.')
                    break
                peg_dir = rob_arm.get_object_matrix(peg_bottom)[:3, 0].reshape(1, 3)
                hole_dir = rob_arm.get_object_matrix(hole_top)[:3, 0].reshape(3, 1)
                dot_product = np.dot(peg_dir, hole_dir)
                angle = math.degrees(math.acos(dot_product / (np.linalg.norm(peg_dir) * np.linalg.norm(hole_dir))))
                if angle > 3.0 and not tilt :
                    print('break! Angle is too large')
                    break
                if angle > 80 and tilt:
                    print('break! Angle is too large')
                    break
                if tilt or yaw:
                    if (speed < 0.8 and (abs(delta_rot_euler_pred) < 1.5).all()) or cnt >= 30:
                        print('servoing done!')
                        break
                else:
                    # bykovis
                    if speed > 0.8:  # 3DoF 0.8 #4 6DoF 0.75
                        cnt_end = 0
                    elif speed <= 0.8:  # 3DoF 0.8 #4 6DoF 0.75
                        cnt_end += 1
                        if cnt_end > 3:  # 3DoF 3 15cm:0 30cm:0
                            break
                    # dsae
                    '''
                    if speed < 2.8 or cnt >= 15:
                        cnt_end += 1
                        if cnt_end > 2:
                            print('servoing done!')
                            break
                    '''
                cnt = cnt + 1

            # insertion
            robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            robot_pose[:3, 3] -= robot_pose[:3, 0] * 0.25  # x-axis
            rob_arm.movement(robot_pose)

            # record insertion
            peg_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name=peg_bottom)
            hole_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name=hole_bottom)
            dist = np.linalg.norm(peg_keypoint_bottom_pose[:3, 3] - hole_keypoint_bottom_pose[:3, 3])
            print('dist', dist)
            #f.write(str(tilt_degree) + ' ' + str(dist) + '\n')
            if dist < 0.025:
                print('success!')
                insertion_succ_list.append(1)
                end_time = time.time()
                time_list.append((end_time - start_time))
                print('Time elasped:{:.02f}'.format((end_time - start_time)))
            else:
                print('fail!')
                insertion_succ_list.append(0)
            rob_arm.finish()
            '''
            # pull up
            robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            robot_pose[:3, 3] += robot_pose[:3, 0] * 0.1  # x-axis
            rob_arm.movement(robot_pose)
            '''
        insertion_succ = sum(insertion_succ_list) / len(insertion_succ_list)
        msg = '* ' + hole_name + ' hole success rate : ' + str(insertion_succ * 100) + '% (' + str(sum(insertion_succ_list)) + '/' + str(len(insertion_succ_list)) + ')'
        print(msg)
        f.write(msg + '\n')
        f.close()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Time elasped:{:.02f}'.format((end_time - start_time) / 3600))
