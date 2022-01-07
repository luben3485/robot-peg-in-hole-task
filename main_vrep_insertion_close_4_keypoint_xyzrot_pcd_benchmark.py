from env.single_robotic_arm import SingleRoboticArm
import numpy as np
import cv2
import math
from transforms3d.quaternions import mat2quat, quat2axangle, quat2mat, qmult
import random
from scipy.spatial.transform import Rotation as R
from inference_pointnet2_reg import PointnetMover
import os
import sys
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


def main():
    # create folder
    benchmark_folder = 'pcd_benchmark'
    if not os.path.exists(benchmark_folder):
        os.makedirs(benchmark_folder)
    f = open(os.path.join(benchmark_folder, "benchmark_1226_reg_close_e_67_vision_eye_concat.txt"), "w")

    rob_arm = SingleRoboticArm()
    init_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
    net_date = '2021-12-26_20-35'
    mover = PointnetMover(date=net_date, model_name='pointnet2_reg_msg', checkpoint_name='best_model_e_67.pth', use_cpu=False, out_channel=9)

    iter_num = 300
    cam_name = 'vision_eye'
    #cam_name = 'vision_fix_front'
    peg_top = 'peg_keypoint_top2'
    peg_bottom = 'peg_keypoint_bottom2'
    hole_top = 'hole_keypoint_top'
    hole_bottom = 'hole_keypoint_bottom'
    peg_name = 'peg2'
    hole_name = 'hole'
    gripper_init_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')

    origin_hole_pose = rob_arm.get_object_matrix(hole_name)
    origin_hole_pos = origin_hole_pose[:3, 3]
    origin_hole_quat = rob_arm.get_object_quat(hole_name)

    origin_peg_pose = rob_arm.get_object_matrix(peg_name)
    origin_peg_pos = origin_peg_pose[:3, 3]
    origin_peg_quat = rob_arm.get_object_quat(peg_name)

    insertion_succ_list = []
    for iter in range(iter_num):
        print('=' * 8 + str(iter) + '=' * 8)
        # set init pos of peg nad hole
        rob_arm.set_object_position(hole_name, np.array([0.2, -0.5, 3.6200e-02]))
        rob_arm.set_object_quat(hole_name, origin_hole_quat)
        rob_arm.set_object_position(peg_name, origin_peg_pos)
        rob_arm.set_object_quat(peg_name, origin_peg_quat)
        _, tilt_degree = random_tilt(rob_arm, [hole_name], 0, 60)
        rob_arm.movement(gripper_init_pose)
        rob_arm.open_gripper(mode=1)

        # gt grasp
        peg_keypoint_top_pose = rob_arm.get_object_matrix(obj_name=peg_top)
        grasp_pose = peg_keypoint_top_pose.copy()
        # gripper offset
        grasp_pose[:3, 3] += peg_keypoint_top_pose[:3, 1] * 0.0015  # y-axis
        rob_arm.gt_run_grasp(grasp_pose)
        grasp_pose[2, 3] += 0.2
        rob_arm.movement(grasp_pose)
        # move to hole top
        hole_keypoint_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
        grasp_pose[:2, 3] = hole_keypoint_top_pose[:2, 3]
        rob_arm.movement(grasp_pose)

        ### start pose ###
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
            robot_pose[:3, 3] = hole_keypoint_top_pose[:3, 3] + hole_keypoint_top_pose[:3, 0] * 0.07
            rob_arm.movement(robot_pose)

            # second
            hole_insert_dir = - rob_arm.get_object_matrix(hole_top)[:3, 0]  # x-axis
            peg_insert_dir = - rob_arm.get_object_matrix(peg_top)[:3, 0]  # x-axis
            dot_product = np.dot(peg_insert_dir, hole_insert_dir)
            degree = math.degrees(
                math.acos(dot_product / (np.linalg.norm(peg_insert_dir) * np.linalg.norm(hole_insert_dir))))
            print('degree between peg and hole : ', degree)
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
            robot_pose[:3, 3] = hole_keypoint_top_pose[:3, 3] + hole_keypoint_top_pose[:3, 0] * 0.07
            rob_arm.movement(robot_pose)

            # gripper offset
            robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            peg_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name=peg_bottom)

            robot_pose[:3, 3] += peg_keypoint_bottom_pose[:3, 1] * 0.004  # y-axis
            robot_pose[:3, 3] -= peg_keypoint_bottom_pose[:3, 0] * 0.006  # x-axis
            robot_pose[:3, 3] += 0.001
            rob_arm.movement(robot_pose)

        # move peg to random x,y,z
        start_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
        random_x = random.uniform(0, 0.02)
        random_y = random.uniform(-0.03, 0.03)
        random_z = random.uniform(-0.03, 0.03)
        start_pose[:3, 3] = start_pose[:3, 3] + start_pose[:3, 0] * random_x + start_pose[:3, 1] * random_y + start_pose[:3, 2] * random_z
        rob_arm.movement(start_pose)
        ### ###

        for i in range(1):
            ### start
            delta_rot_pred, delta_xyz_pred = predict_xyzrot_from_camera(cam_name, mover, rob_arm)
            # rot_matrix = np.dot(delta_rot_pred[:3, :3], gripper_pose[:3, :3])
            gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
            rot_matrix = np.dot(gripper_pose[:3, :3], delta_rot_pred[:3, :3])
            gripper_pose[:3, :3] = rot_matrix
            gripper_pose[:3, 3] += delta_xyz_pred
            rob_arm.movement(gripper_pose)

        # insertion
        robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
        peg_keypoint_top_pose = rob_arm.get_object_matrix(obj_name=peg_top)
        hole_keypoint_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
        robot_pose[:3, 3] -= robot_pose[:3, 0] * 0.05  # x-axis
        # robot_pose[:3, 3] -= peg_keypoint_top_pose[:3, 0] * 0.05  # x-axis
        # robot_pose[:3, 3] -= hole_keypoint_top_pose[:3, 0] * 0.03  # x-axis
        rob_arm.movement(robot_pose)

        # record insertion
        peg_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name=peg_bottom)
        hole_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name=hole_bottom)
        dist = np.linalg.norm(peg_keypoint_bottom_pose[:3, 3] - hole_keypoint_bottom_pose[:3, 3])
        print('dist', dist)
        f.write(str(tilt_degree) + ' ' + str(dist))
        if dist < 0.012:
            insertion_succ_list.append(1)
        else:
            insertion_succ_list.append(0)
    insertion_succ = sum(insertion_succ_list) / len(insertion_succ_list)
    print('insertion success rate : '+ str(insertion_succ))
    f.close()
    rob_arm.finish()


if __name__ == '__main__':
    main()
