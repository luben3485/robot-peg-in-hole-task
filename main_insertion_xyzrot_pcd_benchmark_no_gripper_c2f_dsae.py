from env.single_robotic_arm import SingleRoboticArm
import numpy as np
import cv2
import math
from transforms3d.quaternions import mat2quat, quat2axangle, quat2mat, qmult
import random
from scipy.spatial.transform import Rotation as R
from inference_pointnet2_dsae import Mover
import os
import sys
import time
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
    rgb_list = []
    camera2world_list = []
    for cam_name in cam_name_list:
        cam_matrix = rob_arm.get_object_matrix(obj_name=cam_name)
        camera2world_list.append(cam_matrix)
        rgb = rob_arm.get_rgb(cam_name=cam_name)
        # rotate 180
        (h, w) = rgb.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rgb = cv2.warpAffine(rgb, M, (w, h))
        rgb_list.append(rgb)

    delta_xyz_pred, delta_rot_pred = mover.inference_multiple_camera(rgb_list)

    return delta_xyz_pred, delta_rot_pred


def main():
    # create folder
    benchmark_folder = 'pcd_benchmark/dsae'
    if not os.path.exists(benchmark_folder):
        os.makedirs(benchmark_folder)
    f = open(os.path.join(benchmark_folder, "hole_score.txt"), "w")
    coarse_mover = Mover(model_path='dsae/2022-03-18_19-14', model_name='cnn_dsae', checkpoint_name='best_model_e_33.pth', use_cpu=False, out_channel=9)
    fine_mover = Mover(model_path='dsae/2022-03-18_19-15', model_name='cnn_dsae', checkpoint_name='best_model_e_44.pth', use_cpu=False, out_channel=9)
    iter_num = 500
    cam_name_list = ['vision_eye_left', 'vision_eye_right']
    peg_top = 'peg_dummy_top'
    peg_bottom = 'peg_dummy_bottom'
    peg_name = 'peg_in_arm'
    hole_setting = {'square': ['square', 'hole_keypoint_top0', 'hole_keypoint_bottom0'],
                    'small_square': ['small_square', 'hole_keypoint_top1', 'hole_keypoint_bottom1'],
                    'circle': ['circle', 'hole_keypoint_top2', 'hole_keypoint_bottom2'],
                    'rectangle': ['rectangle', 'hole_keypoint_top3', 'hole_keypoint_bottom3'],
                    'triangle': ['triangle', 'hole_keypoint_top4', 'hole_keypoint_bottom4']}
    #selected_hole_list = ['square', 'small_square', 'circle', 'rectangle', 'triangle']
    selected_hole_list = ['square']
    for selected_hole in selected_hole_list:
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
        for iter in range(iter_num):
            rob_arm = SingleRoboticArm()
            print('=' * 8 + str(iter) + '=' * 8)
            rob_arm.movement(gripper_init_pose)
            # set init pos of peg nad hole
            rob_arm.set_object_position(hole_name, np.array([0.2, -0.5, 3.6200e-02]))
            rob_arm.set_object_quat(hole_name, origin_hole_quat)
            _, tilt_degree = random_tilt(rob_arm, [hole_name], 0, 60)

            # start pose
            delta_move = np.array([random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03), random.uniform(0.10, 0.12)])
            start_pose = rob_arm.get_object_matrix('UR5_ikTip')
            hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
            start_pos = hole_top_pose[:3, 3]
            start_pos += delta_move
            start_pose[:3, 3] = start_pos
            rob_arm.movement(start_pose)
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
                    robot_pose[:3, 3] = hole_keypoint_top_pose[:3, 3] + hole_keypoint_top_pose[:3, 0] * 0.025
                    rob_arm.movement(robot_pose)
            '''
            # coarse approach
            gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
            delta_xyz_pred, delta_rot_pred = predict_xyzrot_from_multiple_camera(cam_name_list, coarse_mover, rob_arm)
            print(delta_xyz_pred)
            print(delta_rot_pred)
            r = R.from_matrix(delta_rot_pred)
            r_euler = r.as_euler('zyx', degrees=True)
            if abs(r_euler[0]) < 90 and abs(r_euler[1]) < 90 and abs(r_euler[2]) < 90:
                gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
                rot_matrix = np.dot(gripper_pose[:3, :3], delta_rot_pred[:3, :3])
                gripper_pose[:3, :3] = rot_matrix
                gripper_pose[:3, 3] += delta_xyz_pred
                rob_arm.movement(gripper_pose)
            else:
                continue

            # fine approach
            # open-loop
            '''
            for i in range(1):
                ### start
                gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
                delta_xyz_pred = predict_kpt_xyz_from_multiple_camera(cam_name_list, gripper_pose, fine_mover, rob_arm)
                gripper_pose[:3, 3] += delta_xyz_pred
                rob_arm.movement(gripper_pose)
            '''

            # closed-loop
            cnt = 0
            while True:
                ### start
                gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
                delta_xyz_pred, _ = predict_xyzrot_from_multiple_camera(cam_name_list, fine_mover, rob_arm)
                step_size = np.linalg.norm(delta_xyz_pred)
                '''
                # compute distance error
                hole_top_pose = rob_arm.get_object_matrix(hole_top)
                hole_top_pos = hole_top_pose[:3, 3]
                hole_top_pos_pred = delta_xyz_pred + gripper_pose[:3, 3]
                distance_error = np.linalg.norm(hole_top_pos - hole_top_pos_pred)*1000 # unit:mm
                print(distance_error)
                #print(step_size)
                '''
                print(step_size)
                gripper_pose[:3, 3] += delta_xyz_pred
                rob_arm.movement(gripper_pose)
                if step_size < 0.005 or cnt >= 5:
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
            print('dist', dist)
            #f.write(str(tilt_degree) + ' ' + str(dist) + '\n')
            if dist < 0.010:
                insertion_succ_list.append(1)
            else:
                insertion_succ_list.append(0)
            rob_arm.finish()
            '''
            # pull up
            robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            robot_pose[:3, 3] += robot_pose[:3, 0] * 0.1  # x-axis
            rob_arm.movement(robot_pose)
            '''
        insertion_succ = sum(insertion_succ_list) / len(insertion_succ_list)
        msg = hole_name + ' hole success rate : ' + str(insertion_succ * 100) + '% (' + str(sum(insertion_succ_list)) + '/' + str(len(insertion_succ_list)) + ')'
        print(msg)
        f.write(msg + '\n')
        #rob_arm.finish()
    f.close()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Time elasped:{:.02f}'.format((end_time - start_time) / 3600))
