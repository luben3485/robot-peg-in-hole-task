import argparse
from env.single_robotic_arm import SingleRoboticArm
import open3d as o3d
import numpy as np
import cv2
import os
import time
import random
import math
import copy
from inference_pointnet2_kpts import CoarseMover
from transforms3d.quaternions import mat2quat, quat2axangle, quat2mat, qmult
from config.hole_setting import hole_setting
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

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

def save_template_hole_pcd():
    hole_pos = [0.2, -0.5, 3.5001e-02]
    rob_arm = SingleRoboticArm()
    rob_arm.set_object_position(hole_name, hole_pos)
    gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
    gripper_pose[:2, 3] = [0.2, -0.5]
    hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
    rob_arm.movement(gripper_pose)

    print('relative translation:', hole_top_pose[:3, 3] - gripper_pose[:3, 3])

    hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
    hole_obj_bottom_pose = rob_arm.get_object_matrix(obj_name=hole_obj_bottom)
    scene_xyz, hole_xyz = get_pcd_from_multi_camera(rob_arm, cam_name_list, hole_top_pose, hole_obj_bottom_pose)

    # save pcd for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_xyz)
    o3d.io.write_point_cloud(os.path.join('scene_pcd.ply'), pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hole_xyz)
    o3d.io.write_point_cloud(os.path.join('hole_pcd.ply'), pcd)

    # save npy
    print( np.array(hole_xyz)/1000)
    np.save('hole_pcd.npy', np.array(hole_xyz)/1000)

    rob_arm.finish()

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    #source_2_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([1, 0.706, 0])
    #source_2_temp.paint_uniform_color([1, 0, 0.706])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([target_temp, source_temp])
    '''
    # visualize on meshlab
    source_temp.points = o3d.utility.Vector3dVector(np.asarray(source_temp.points) * 1000)
    target_temp.points = o3d.utility.Vector3dVector(np.asarray(target_temp.points) * 1000)
    o3d.io.write_point_cloud(os.path.join('source_temp.ply'), source_temp)
    o3d.io.write_point_cloud(os.path.join('target_temp.ply'), target_temp)
    '''
def coarse_controller_with_icp_gt_pos(rob_arm, cam_name_list, hole_top, hole_obj_bottom):
    hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
    hole_obj_bottom_pose = rob_arm.get_object_matrix(obj_name=hole_obj_bottom)
    scene_xyz, _ = get_pcd_from_multi_camera(rob_arm, cam_name_list, hole_top_pose, hole_obj_bottom_pose)

    #source = o3d.geometry.PointCloud()
    #source.points = o3d.utility.Vector3dVector(np.load('hole.npy'))
    source = o3d.io.read_point_cloud('full_hole.pcd')
    print(source)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(scene_xyz/1000)
    threshold = 0.02

    trans_init = np.zeros((4,4))
    trans_init[3, 3] = 1.0
    hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
    random_x = random.uniform(0.005, 0.010)
    if random.uniform(0, 1) > 0.5:
        random_y = random.uniform(0.005, 0.010)
    else:
        random_y = random.uniform(-0.010, -0.005)
    if random.uniform(0, 1) > 0.5:
        random_z = random.uniform(0.005, 0.010)
    else:
        random_z = random.uniform(-0.010, -0.005)

    hole_top_rough_pos = hole_top_pose[:3, 3] + hole_top_pose[:3, 0] * random_x + hole_top_pose[:3, 1] * random_y + hole_top_pose[:3, 2] * random_z
    # we assume that the vertical direction is [0, 0, 1], and there is no any rotation around the vertical direction.
    trans_init[:3, :3] = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    # the hole top position of the cad model is [0.2, -0.5, 0.07]
    trans_init[:3, 3] = hole_top_rough_pos - np.dot(trans_init[:3, :3], np.array([0., 0, 0.07]).reshape(3, 1))[:, 0]

    '''
    # we assume that the keypoints have 5-mm error. hole_top_pose[:3, 0] * random.uniform(-0.001, 0.001) + hole_top_pose[:3, 1] * random.uniform(-0.001, 0.001) + hole_top_pose[:3, 2] * random.uniform(-0.001, 0.001)
    hole_top_rough_pos = hole_top_pose[:3, 3] + hole_top_pose[:3, 0] * random.uniform(-0.005, 0.005) + hole_top_pose[:3, 1] * random.uniform(-0.005, 0.005) + hole_top_pose[:3, 2] * random.uniform(-0.005, 0.005)
    hole_bottom_pos = hole_top_pose[:3, 3] - hole_top_pose[:3, 0] * 0.005
    hole_bottom_rough_pos = hole_bottom_pos + hole_top_pose[:3, 0] * random.uniform(-0.005, 0.005) + hole_top_pose[:3, 1] * random.uniform(-0.005, 0.005) + hole_top_pose[:3, 2] * random.uniform(-0.005, 0.005)
    hole_y_plus_pos = hole_top_pose[:3, 3] + hole_top_pose[:3, 1] * 0.005
    hole_y_plus_rough_pos = hole_y_plus_pos + hole_top_pose[:3, 0] * random.uniform(-0.005, 0.005) + hole_top_pose[:3, 1] * random.uniform(-0.005, 0.005) + hole_top_pose[:3, 2] * random.uniform(-0.005, 0.005)

    x_dir = (hole_top_rough_pos - hole_bottom_pos) / np.linalg.norm(hole_top_rough_pos - hole_bottom_pos)
    y_dir = hole_y_plus_rough_pos - hole_top_rough_pos
    z_dir = np.cross(x_dir, y_dir)
    z_dir = z_dir / np.linalg.norm(z_dir)
    y_dir = np.cross(z_dir, x_dir)
    rot = np.concatenate((x_dir.reshape(3, 1), y_dir.reshape(3, 1), z_dir.reshape(3, 1)), axis=1)
    trans_init[:3, :3] = np.dot(rot, np.transpose(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])))
    trans_init[:3, 3] = hole_top_rough_pos - np.dot(trans_init[:3, :3], np.array([0.2, -0.5, 0.07]).reshape(3, 1))[:,0]
    '''

    '''
    #ground truth
    trans_init[:3, :3] = np.dot(hole_top_pose[:3, :3], np.transpose(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])))
    trans_init[:3, 3] = hole_top_pose[:3,3] - np.dot(trans_init[:3, :3], np.array([0.2, -0.5, 0.07]).reshape(3, 1))[:, 0]
    '''
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())

    draw_registration_result(source, target, reg_p2p.transformation)
    transformation = copy.deepcopy(reg_p2p.transformation)
    print(transformation)
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_xyz)
    o3d.io.write_point_cloud(os.path.join('scene_tilt_pcd.ply'), pcd)
    '''
    return transformation

def coarse_controller_with_icp(rob_arm, cam_name_list, trans_init, hole_top, hole_obj_bottom):
    hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
    hole_obj_bottom_pose = rob_arm.get_object_matrix(obj_name=hole_obj_bottom)
    scene_xyz, _ = get_pcd_from_multi_camera(rob_arm, cam_name_list, hole_top_pose, hole_obj_bottom_pose, noise=True)

    # square hole
    source = o3d.io.read_point_cloud('square_7x12x12_squarehole_for_icp.pcd')
    source.points = o3d.utility.Vector3dVector(np.asarray(source.points) / 1000)
    # round hole
    # source = o3d.io.read_point_cloud('full_hole.pcd')
    print(source)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(scene_xyz/1000)
    threshold = 0.02 # 0.02
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    draw_registration_result(source, target, reg_p2p.transformation)
    transformation = copy.deepcopy(reg_p2p.transformation)
    #print(transformation)
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_xyz)
    o3d.io.write_point_cloud(os.path.join('scene_tilt_pcd.ply'), pcd)
    '''
    return transformation

def predict_kpts_no_oft_from_multiple_camera(cam_name_list, mover, rob_arm):
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
    #trans_init = np.zeros((4, 4))
    #trans_init[3, 3] = 1.0

    ''' rot_mat_pred is related to gripper
    trans_init[:3, :3] = np.dot(rot_mat_pred, np.transpose(np.array([[0, -1, 0],[0, 0, -1],[1, 0, 0]])))
    trans_init[:3, 3] = real_kpt_pred - np.dot(trans_init[:3, :3], np.array([0., 0, 0.07]).reshape(3, 1))[:, 0]
    '''
    '''
    if detect_kpt == True:
        # trans_init[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        trans_init[:3, :3] = np.dot(rot_mat_pred, np.transpose(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])))
        trans_init[:3, 3] = real_kpt_pred - np.dot(trans_init[:3, :3], np.array([0., 0, 0.07]).reshape(3, 1))[:, 0]
    else:
        trans_init[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        trans_init[:3, 3] = np.array([0.1, -0.5, 0.035]) - np.dot(trans_init[:3, :3], np.array([0., 0, 0.07]).reshape(3, 1))[:, 0]
    '''
    return real_kpt_pred

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

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--tilt', action='store_true', default=False, help='tilt')
    parser.add_argument('--yaw', action='store_true', default=False, help='yaw')
    return parser.parse_args()

def main(args):
    '''
    pcd = o3d.io.read_point_cloud('full_hole.pcd')
    xyz = np.asarray(pcd.points)
    print(xyz.shape)
    pcd.points = o3d.utility.Vector3dVector(xyz/1000)
    o3d.io.write_point_cloud('full_hole.pcd', pcd)
    assert False
    '''
    #save_template_hole_pcd()

    peg_top = 'peg_dummy_top'
    peg_bottom = 'peg_dummy_bottom'
    peg_name = 'peg_in_arm'
    #cam_name_list = ['vision_eye_left', 'vision_eye_right']
    cam_name_list = ['vision_eye_front'] #'vision_eye_delaunay'
    tilt = args.tilt
    yaw = args.yaw
    benchmark_folder = os.path.join('pcd_benchmark', 'delaunay')
    if not os.path.exists(benchmark_folder):
        os.makedirs(benchmark_folder)
    # 6-DoF
    coarse_mover = CoarseMover(model_path='kpts/2022-06-17_12-24', model_name='pointnet2_kpts', checkpoint_name='best_model_e_103.pth', use_cpu=False, out_channel=9)
    # round hole for tilt
    #coarse_mover = CoarseMover(model_path='kpts/2022-06-10_22-43', model_name='pointnet2_kpts', checkpoint_name='best_model_e_101.pth', use_cpu=False, out_channel=9)
    # 3-DoF sr:101 err:76
    #coarse_mover = CoarseMover(model_path='kpts/2022-05-17_21-15', model_name='pointnet2_kpts', checkpoint_name='best_model_e_76.pth', use_cpu=False, out_channel=9)

    #fine_mover = Mover(model_path='kpts/2022-04-01_15-11', model_name='pointnet2_kpts', checkpoint_name='best_model_e_43.pth', use_cpu=False, out_channel=9)
    #selected_hole_list = ['square', 'small_square', 'circle', 'rectangle', 'triangle']
    #selected_hole_list = ['square_7x11_5x11_5', 'square_7x12_5x12_5', 'rectangle_7x11_5x12', 'rectangle_7x12x11_5', 'rectangle_7x9x13', 'rectangle_7x10x12', 'rectangle_7x12x13']
    #selected_hole_list = ['square_7x11_5x11_5', 'rectangle_7x12x13', 'rectangle_7x10x12', 'circle_7x14', 'circle_7x12', 'circle_7x10', 'pentagon_7x7', 'octagon_7x5']
    selected_hole_list = ['square_7x11_5x11_5_squarehole',]# 'rectangle_7x10x12_squarehole', 'circle_7x14_squarehole', 'pentagon_7x9_squarehole']
    #selected_hole_list = ['square_7x11_5x11_5', 'rectangle_7x10x12', 'circle_7x12', 'pentagon_7x7']
    #selected_hole_list = ['square_7x11_5x11_5']
    for selected_hole in selected_hole_list:
        f = open(os.path.join(benchmark_folder, "hole_score.txt"), "a")
        hole_name = hole_setting[selected_hole][0]
        hole_top = hole_setting[selected_hole][1]
        hole_bottom = hole_setting[selected_hole][2]
        hole_obj_bottom = hole_setting[selected_hole][3]
        insertion_succ_list = []
        error_x, error_y, error_xy = [], [], []
        time_list = []
        r_error = []
        t_error = []
        for i in range(args.iter):
            print('='*10 + 'iter:' + str(i) + '='*10)
            rob_arm = SingleRoboticArm()
            hole_pos = [random.uniform(0, 0.2), random.uniform(-0.45, -0.55), +3.5001e-02]

            # tmp compute time
            hole_pos = np.array([0.1, -0.525, 0.035])
            rob_arm.set_object_position(hole_name, hole_pos)
            gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
            # 15cm
            #delta_move = np.array([0.04, 0.04, 0.14])
            # 30cm
            delta_move = np.array([0.08, 0.08, 0.28])
            gripper_pose[:3, 3] = hole_top_pose[:3, 3] + delta_move
            rob_arm.movement(gripper_pose)
            # tmp hide
            #rob_arm.set_object_position(hole_name, hole_pos)
            if yaw:
                random_yaw(rob_arm, [hole_name])
            if tilt:
                _, tilt_degree = random_tilt(rob_arm, [hole_name], 0, 50)
            '''
            # start pose
            delta_move = np.array([random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03), random.uniform(0.10, 0.12)])
            start_pose = rob_arm.get_object_matrix('UR5_ikTip')
            hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
            start_pos = hole_top_pose[:3, 3]
            start_pos += delta_move
            start_pose[:3, 3] = start_pos
            rob_arm.movement(start_pose)
            '''
            start_time = time.time()
            gripper_init_rot = rob_arm.get_object_matrix('UR5_ikTip')[:3, :3]
            hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
            hole_obj_bottom_pose = rob_arm.get_object_matrix(obj_name=hole_obj_bottom)
            real_kpt_pred = predict_kpts_no_oft_from_multiple_camera(cam_name_list, coarse_mover, rob_arm)
            print('position', real_kpt_pred)
            scene_xyz, _ = get_pcd_from_multi_camera(rob_arm, cam_name_list, hole_top_pose, hole_obj_bottom_pose, noise=True)
            scene_xyz /= 1000
            #scene_xyz = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0],[0.5,0.5,1]])
            tri = Delaunay(scene_xyz[:, :2])
            faces = scene_xyz[tri.simplices]
            normals = []
            for face in faces:
                if np.linalg.norm(face[0] - real_kpt_pred) < 0.035:
                    normal = np.cross(face[0]-face[1], face[0]-face[2])
                    normal = normal / np.linalg.norm(normal)
                    normals.append(normal)

            if len(normals) != 0:
                normal = np.array(normals).mean(axis=0)
            else:
                normal = np.array([0,0,1])

            #print(tri.simplices)
            #print(scene_xyz[tri.simplices])
            #fig = plt.figure()
            #ax = fig.add_subplot(1, 1, 1, projection='3d')

            # visualize
            #ax.plot_trisurf(scene_xyz[:,0], scene_xyz[:,1], scene_xyz[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
            #plt.show()
            gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
            gripper_pose[:3, 3] = real_kpt_pred
            gripper_init_rot[:3, 0] = normal
            gripper_pose[:3, :3] = gripper_init_rot

            # To ensure the rotation matrix is orthogonal
            gripper_pose[:3, 0] = gripper_pose[:3, 0] / np.linalg.norm(gripper_pose[:3, 0])
            gripper_pose[:3, 2] = np.cross(gripper_pose[:3, 0], gripper_pose[:3, 1])
            gripper_pose[:3, 2] = gripper_pose[:3, 2] / np.linalg.norm(gripper_pose[:3, 2])
            gripper_pose[:3, 1] = np.cross(gripper_pose[:3, 2], gripper_pose[:3, 0])
            print('normal', gripper_pose[:3, 0])
            rob_arm.movement(gripper_pose)


            '''
            # coarse approach with ICP
            trans_init = predict_kpts_no_oft_from_multiple_camera(cam_name_list, coarse_mover, rob_arm, args.detect_kpt)
            transformation = coarse_controller_with_icp(rob_arm, cam_name_list, trans_init, hole_top, hole_obj_bottom)
            gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
            gripper_pose[:3, 3] = np.array([0.0, 0.0, 0.07])
            rot_matrix = np.dot(transformation[:3,:3], gripper_pose[:3, :3])
            gripper_pose[:3, :3] = rot_matrix
            gripper_pos = np.append(gripper_pose[:3, 3], 1).reshape(4, 1)
            gripper_pos = np.dot(transformation, gripper_pos)[:3, 0]
            gripper_pose[:3, 3] = gripper_pos
            #gripper_pose[:3, 3] += gripper_pose[:3, 0] * 0.01  # x-axis
            rob_arm.movement(gripper_pose)
            '''
            # compute distance error
            hole_keypoint_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
            robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            r_error.append(np.sqrt(np.mean((hole_keypoint_top_pose[:3, :3] - robot_pose[:3, :3]) ** 2)))
            t_error.append(np.sqrt(np.mean((hole_keypoint_top_pose[:3, 3] - robot_pose[:3, 3]) ** 2)))

            # compute distance error
            hole_top_pose = rob_arm.get_object_matrix(hole_top)
            hole_top_pos = hole_top_pose[:3, 3]
            gripper_pos = rob_arm.get_object_matrix('UR5_ikTip')[:3, 3]
            error = gripper_pos - hole_top_pos
            error_xy.append(np.linalg.norm(error[:2]))
            print('error x:', error[0])
            print('error y:', error[1])
            print('error xy:', np.linalg.norm(error[:2]))
            error_x.append(error[0])
            error_y.append(error[1])

            # insertion
            gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
            gripper_pose[:3, 3] -= gripper_pose[:3, 0] * 0.08  # x-axis
            rob_arm.movement(gripper_pose)
            # record insertion
            peg_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name=peg_bottom)
            hole_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name=hole_bottom)
            dist = np.linalg.norm(peg_keypoint_bottom_pose[:3, 3] - hole_keypoint_bottom_pose[:3, 3])
            print('dist', dist)
            if dist < 0.010:
                insertion_succ_list.append(1)
                print('success!')

            else:
                insertion_succ_list.append(0)
                error_x.pop()
                error_y.pop()
                error_xy.pop()
                print('fail!')
            end_time = time.time()
            time_list.append((end_time - start_time))
            rob_arm.finish()
        insertion_succ = sum(insertion_succ_list) / len(insertion_succ_list)
        msg = '* ' + hole_name + ' hole success rate : ' + str(insertion_succ * 100) + '% (' + str(sum(insertion_succ_list)) + '/' + str(len(insertion_succ_list)) + ')'
        print(msg)
        f.write(msg + '\n')
        if len(r_error) != 0 and len(t_error)!=0:
            r_error = sum(r_error) / len(r_error)
            t_error = sum(t_error) / len(t_error)
            f.write('    * r t error' + str(r_error) + ' ' + str(t_error) + '\n')
        if len(time_list) != 0:
            time_ = sum(time_list) / len(time_list)
            print('time:' + str(time_))
            f.write('    * time:' + str(time_) + '\n')
        if len(error_x)!=0 and len(error_y)!=0:
            print('average x error', sum(error_x) / len(error_x))
            print('average y error', sum(error_y) / len(error_y))
            print('average xy error', sum(error_xy) / len(error_xy))
            f.write('    * average x error' + str(sum(error_x) / len(error_x)) + '\n')
            f.write('    * average y error' + str(sum(error_y) / len(error_y)) + '\n')
            f.write('    * average xy error' + str(sum(error_xy) / len(error_xy)) + '\n')
        f.close()

if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print('Time elasped:{:.02f}'.format((end_time - start_time) / 3600))
