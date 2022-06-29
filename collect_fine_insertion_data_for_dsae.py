from env.single_robotic_arm import SingleRoboticArm
import yaml
import cv2
import os
import numpy as np
from ruamel import yaml
import math
import random
import time
from transforms3d.quaternions import mat2quat, quat2axangle, quat2mat, qmult
from scipy.spatial.transform import Rotation as R
import argparse
import copy

def world2camera_from_map(camera2world_map):
    camera2world = camera2world_from_map(camera2world_map)
    return inverse_matrix(camera2world)

def inverse_matrix(matrix):
    inv = np.linalg.inv(matrix)
    return inv

def camera2world_from_map(camera2world_map):
    """
    Get the transformation matrix from the storage map
    See ${processed_log_path}/processed/images/pose_data.yaml for an example
    :param camera2world_map:
    :return: The (4, 4) transform matrix from camera 2 world
    """
    # The rotation part
    camera2world_quat = [1, 0, 0, 0]
    camera2world_quat[0] = camera2world_map['quaternion']['w']
    camera2world_quat[1] = camera2world_map['quaternion']['x']
    camera2world_quat[2] = camera2world_map['quaternion']['y']
    camera2world_quat[3] = camera2world_map['quaternion']['z']
    camera2world_quat = np.asarray(camera2world_quat)
    camera2world_matrix = quaternion_matrix(camera2world_quat)

    # The linear part
    camera2world_matrix[0, 3] = camera2world_map['translation']['x']
    camera2world_matrix[1, 3] = camera2world_map['translation']['y']
    camera2world_matrix[2, 3] = camera2world_map['translation']['z']
    return camera2world_matrix

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

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def point2pixel(keypoint_in_camera, focal_x=309.019, focal_y=309.019, principal_x=128, principal_y=128):
    """
    Given keypoint in camera frame, project them into image
    space and compute the depth value expressed in [mm]
    :param keypoint_in_camera: (4, n_keypoint) keypoint expressed in camera frame in meter
    :return: (3, n_keypoint) where (xy, :) is pixel location and (z, :) is depth in mm
    """
    assert len(keypoint_in_camera.shape) is 2
    n_keypoint = keypoint_in_camera.shape[0]
    xy_depth = np.zeros((n_keypoint, 3), dtype=np.int)
    xy_depth[:, 0] = (np.divide(keypoint_in_camera[:, 0], keypoint_in_camera[:, 2]) * focal_x + principal_x).astype(np.int)
    xy_depth[:, 1] = (np.divide(keypoint_in_camera[:, 1], keypoint_in_camera[:, 2]) * focal_y + principal_y).astype(np.int)
    xy_depth[:, 2] = (1000.0 * keypoint_in_camera[:, 2]).astype(np.int)
    return xy_depth


def generate_one_im_anno(cnt, cam_name_list, cam_name_seg_peg, cam_name_seg_hole, peg_top, peg_bottom, hole_top, hole_bottom, hole_obj_bottom, rob_arm, im_data_path,
                         delta_rotation, delta_translation, gripper_pose, r_euler):
    # Get keypoint location
    peg_keypoint_top_pose = rob_arm.get_object_matrix(peg_top)
    peg_keypoint_bottom_pose = rob_arm.get_object_matrix(peg_bottom)
    hole_keypoint_top_pose = rob_arm.get_object_matrix(hole_top)
    hole_keypoint_bottom_pose = rob_arm.get_object_matrix(hole_bottom)
    hole_keypoint_obj_bottom_pose = rob_arm.get_object_matrix(hole_obj_bottom)

    peg_keypoint_top_in_world = peg_keypoint_top_pose[:3, 3]
    peg_keypoint_bottom_in_world = peg_keypoint_bottom_pose[:3, 3]
    hole_keypoint_top_in_world = hole_keypoint_top_pose[:3, 3]
    hole_keypoint_bottom_in_world = hole_keypoint_bottom_pose[:3, 3]

    rgb_file_name = []
    depth_file_name = []
    cam_matrix_list = []
    for idx, cam_name in enumerate(cam_name_list):
        # Get RGB images from camera
        rgb = rob_arm.get_rgb(cam_name=cam_name)

        # rotate rgb 180
        (h, w) = rgb.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rgb = cv2.warpAffine(rgb, M, (w, h))

        # Get depth from camera
        depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=1.5)

        # rotate depth 180
        (h, w) = depth.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        depth = cv2.warpAffine(depth, M, (w, h))

        depth_mm = (depth * 1000).astype(np.uint8)  # type: np.uint16 ; uint16 is needed by keypoint detection network
        # rob_arm.visualize_image(rgb=rgb, depth=depth)

        # Save image
        rgb_file_name.append(str(cnt + idx).zfill(6) + '_rgb.png')
        rgb_file_path = os.path.join(im_data_path, str(cnt + idx).zfill(6) + '_rgb.png')
        depth_file_name.append(str(cnt + idx).zfill(6) + '_depth.png')
        depth_file_path = os.path.join(im_data_path, str(cnt + idx).zfill(6) + '_depth.png')
        cv2.imwrite(rgb_file_path, rgb)
        cv2.imwrite(depth_file_path, depth_mm)

        cam_matrix = rob_arm.get_object_matrix(obj_name=cam_name)
        cam_matrix_list.append(cam_matrix.tolist())
        if idx == 0:  # idx = 0  main camera
            # Get camera info
            cam_pos, cam_quat, cam_matrix = rob_arm.get_camera_pos(cam_name=cam_name)
            camera_to_world = {'quaternion': {'x': cam_quat[0], 'y': cam_quat[1], 'z': cam_quat[2], 'w': cam_quat[3]},
                               'translation': {'x': cam_pos[0], 'y': cam_pos[1], 'z': cam_pos[2]}}
            camera2world_map = camera_to_world
            world2camera = world2camera_from_map(camera2world_map)

            peg_keypoint_top_in_world = np.append(peg_keypoint_top_in_world, [1], axis=0).reshape(4, 1)
            peg_keypoint_bottom_in_world = np.append(peg_keypoint_bottom_in_world, [1], axis=0).reshape(4, 1)
            hole_keypoint_top_in_world = np.append(hole_keypoint_top_in_world, [1], axis=0).reshape(4, 1)
            hole_keypoint_bottom_in_world = np.append(hole_keypoint_bottom_in_world, [1], axis=0).reshape(4, 1)

            peg_keypoint_top_in_camera = world2camera.dot(peg_keypoint_top_in_world)
            peg_keypoint_bottom_in_camera = world2camera.dot(peg_keypoint_bottom_in_world)
            hole_keypoint_top_in_camera = world2camera.dot(hole_keypoint_top_in_world)
            hole_keypoint_bottom_in_camera = world2camera.dot(hole_keypoint_bottom_in_world)

            peg_keypoint_top_in_camera = peg_keypoint_top_in_camera[:3].reshape(3, )
            peg_keypoint_bottom_in_camera = peg_keypoint_bottom_in_camera[:3].reshape(3, )
            hole_keypoint_top_in_camera = hole_keypoint_top_in_camera[:3].reshape(3, )
            hole_keypoint_bottom_in_camera = hole_keypoint_bottom_in_camera[:3].reshape(3, )

            keypoint_in_camera = np.array(
                [peg_keypoint_top_in_camera, peg_keypoint_bottom_in_camera, hole_keypoint_top_in_camera,
                 hole_keypoint_bottom_in_camera])
            # keypoint_in_camera = np.array([peg_keypoint_bottom_in_camera, hole_keypoint_top_in_camera])
            xy_depth = point2pixel(keypoint_in_camera)
            xy_depth_list = []
            (h, w) = rgb.shape[:2]
            for i in range(len(xy_depth)):
                x = int(xy_depth[i][0])
                y = int(xy_depth[i][1])
                z = int(xy_depth[i][2])
                xy_depth_list.append([x, y, z])
    seg_peg_file_name = []
    for idx, cam_name in enumerate(cam_name_seg_peg):
        # Get RGB images from camera
        rgb = rob_arm.get_rgb(cam_name=cam_name)
        # rotate rgb 180
        (h, w) = rgb.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rgb = cv2.warpAffine(rgb, M, (w, h))

        seg_peg_file_name.append(str(cnt + idx).zfill(6) + '_seg_peg.png')
        seg_peg_file_path = os.path.join(im_data_path, str(cnt + idx).zfill(6) + '_seg_peg.png')
        cv2.imwrite(seg_peg_file_path, rgb)

    seg_hole_file_name = []
    for idx, cam_name in enumerate(cam_name_seg_hole):
        # Get RGB images from camera
        rgb = rob_arm.get_rgb(cam_name=cam_name)
        # rotate rgb 180
        (h, w) = rgb.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rgb = cv2.warpAffine(rgb, M, (w, h))

        seg_hole_file_name.append(str(cnt + idx).zfill(6) + '_seg_hole.png')
        seg_hole_file_path = os.path.join(im_data_path, str(cnt + idx).zfill(6) + '_seg_hole.png')
        cv2.imwrite(seg_hole_file_path, rgb)

    # rot_vec, rot_theta = quat2axangle(delta_rotation_quat)
    # delta_rotation_matrix = quat2mat(delta_rotation_quat)

    # compute bbox
    '''
    xyd = np.array(xy_depth_list)
    top_left_x = max(int(np.min(xyd[:, 0])) - 30, 0)
    top_left_y = max(int(np.min(xyd[:, 1])) - 20, 0)
    bottom_right_x = min(int(np.max(xyd[:, 0])) + 30, 255)
    bottom_right_y = min(int(np.max(xyd[:, 1])) + 20, 255)
    bbox_top_left_xy = [top_left_x, top_left_y]
    bbox_bottom_right_xy = [bottom_right_x, bottom_right_y]
    '''
    bbox_top_left_xy = [0, 0]
    bbox_bottom_right_xy = [rgb.shape[1], rgb.shape[0]]
    # visualize
    '''
    img = rgb.copy()
    cv2.circle(img, (top_left_x, top_left_y), 3, (0, 0, 255), 1)
    cv2.circle(img, (bottom_right_x, bottom_right_y), 3, (255, 0, 0), 1)
    cv2.imshow('visualize keypoint', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    info = {'3d_keypoint_camera_frame': [[float(v) for v in peg_keypoint_top_in_camera],
                                         [float(v) for v in peg_keypoint_bottom_in_camera],
                                         [float(v) for v in hole_keypoint_top_in_camera],
                                         [float(v) for v in hole_keypoint_bottom_in_camera]],
            # '3d_keypoint_camera_frame':[ [float(v) for v in peg_keypoint_bottom_in_camera],
            #                         [float(v) for v in hole_keypoint_top_in_camera]],
            'hole_keypoint_obj_top_pose_in_world': hole_keypoint_top_pose.tolist(),
            'hole_keypoint_obj_bottom_pose_in_world': hole_keypoint_obj_bottom_pose.tolist(),
            'bbox_bottom_right_xy': bbox_bottom_right_xy,
            'bbox_top_left_xy': bbox_top_left_xy,
            'camera_to_world': camera_to_world,
            'camera_matrix': cam_matrix_list,
            'depth_image_filename': depth_file_name,  # list
            'seg_peg_image_filename': seg_peg_file_name,  # list
            'seg_hole_image_filename': seg_hole_file_name,  # list
            'rgb_image_filename': rgb_file_name,  # list
            'keypoint_pixel_xy_depth': xy_depth_list,
            # 'delta_rotation_quat': delta_rotation_quat.tolist(),
            'delta_rotation_matrix': delta_rotation.tolist(),
            'delta_translation': delta_translation.tolist(),
            # 'delta_rot_vec': rot_vec.tolist(),
            # 'delta_rot_theta': rot_theta,
            'gripper_pose': gripper_pose.tolist(),
            'r_euler': r_euler.tolist()
            }
    return info

def random_peg_hole_pos(peg_pos, hole_pos):
    while True:
        ran_peg_pos = peg_pos.copy()
        ran_hole_pos = hole_pos.copy()

        peg_delta_x = random.uniform(-0.15,0.15)
        peg_delta_y = random.uniform(-0.15,0.15)
        peg_delta_z = random.uniform(0,0.15)
        ran_peg_pos[0] += peg_delta_x
        ran_peg_pos[1] += peg_delta_y
        ran_peg_pos[2] += peg_delta_z

        hole_delta_x = random.uniform(-0.15,0.15)
        hole_delta_y = random.uniform(-0.15,0.15)
        ran_hole_pos[0] += hole_delta_x
        ran_hole_pos[1] += hole_delta_y

        diff_x = abs(ran_peg_pos[0] - ran_hole_pos[0])
        diff_y = abs(ran_peg_pos[1] - ran_hole_pos[1])
        if diff_x > 0.06 and diff_y > 0.06:
            return ran_peg_pos, ran_hole_pos

def get_init_pos(hole_pos):
    hole_pos[0] = 0.2
    hole_pos[1] = -0.5

    return hole_pos

def random_gripper_pose(ran_gripper_pose, hole_pos):
    while True:
        x = random.uniform(0.1, 0.375)
        y = random.uniform(-0.65, -0.375)
        if abs(x - hole_pos[0]) > 0.13 and abs(y - hole_pos[1])  > 0.13:
            break
    ran_gripper_pose[0,3] = x
    ran_gripper_pose[1,3] = y

    return ran_gripper_pose

def random_gripper_xy(hole_pos):

    while True:
        #delta_x = random.uniform(-0.15, 0.15)
        delta_x = random.uniform(-0.1, 0.1)
        if abs(delta_x) > 0.09:
            break
    y_reverse_prob = 0.4
    if y_reverse_prob > random.uniform(0, 1):
        delta_y = -0.10

    else:
        #delta_y = random.uniform(0.13, 0.25)
        delta_y = random.uniform(0.13, 0.2)


    #x = hole_pos[0] + delta_x
    #y = hole_pos[1] + delta_y
    return delta_x, delta_y

def random_tilt(rob_arm, obj_name_list, min_tilt_degree, max_tilt_degree):
    ### method 1
    #rot_dir = np.random.normal(size=(3,))
    #rot_dir = rot_dir / np.linalg.norm(rot_dir)

    ### method 2
    while True:
        u = random.uniform(0,1)
        v = random.uniform(0,1)
        theta = 2 * math.pi * u
        phi = math.acos(2 * v - 1)
        x = math.sin(theta) * math.sin(phi)
        y = math.cos(theta) * math.sin(phi)
        z = math.cos(phi)
        dst_hole_dir = np.array([x,y,z]) # world coordinate
        src_hole_dir = np.array([0,0,1]) # world coordinate

        cross_product = np.cross(src_hole_dir, dst_hole_dir)
        if cross_product.nonzero()[0].size == 0: # to check if it is zero vector
            rot_dir = np.array([0,0,1])
        else:
            rot_dir = cross_product / np.linalg.norm(cross_product)
        dot_product = np.dot(src_hole_dir, dst_hole_dir)
        tilt_degree = math.degrees(math.acos(dot_product / (np.linalg.norm(src_hole_dir) * np.linalg.norm(dst_hole_dir))))
        if abs(tilt_degree) <= max_tilt_degree and abs(tilt_degree) >= min_tilt_degree:
            break
    #print('rot_dir:', rot_dir)
    #print('tilt degree:', tilt_degree)
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

def getFootPoint(point, line_p1, line_p2):
    """
    @point, line_p1, line_p2 : [x, y, z]
    """
    x0 = point[0]
    y0 = point[1]
    z0 = point[2]

    x1 = line_p1[0]
    y1 = line_p1[1]
    z1 = line_p1[2]

    x2 = line_p2[0]
    y2 = line_p2[1]
    z2 = line_p2[2]

    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) / \
        ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) * 1.0

    xn = k * (x2 - x1) + x1
    yn = k * (y2 - y1) + y1
    zn = k * (z2 - z1) + z1
    return [xn, yn, zn]

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--hole', type=str, default='square_7x12x12', help='specify object hole')
    parser.add_argument('--iter', type=int, default=3, help='nb of input data')
    parser.add_argument('--date', type=str, default='2022-05-02-notilt-test', help='date')
    parser.add_argument('--data_root', type=str, default='/home/luben/data/pdc/logs_proto', help='data root path')

    return parser.parse_args()

def main(args):
    data_root = args.data_root
    date = args.date
    anno_data = 'fine_insertion_' + args.hole + '_' + date + '/processed'
    im_data = 'fine_insertion_' + args.hole + '_' + date + '/processed/images'
    anno_data_path = os.path.join(data_root, anno_data)
    im_data_path = os.path.join(data_root, im_data)

    # create folder
    cwd = os.getcwd()
    os.chdir(data_root)
    if not os.path.exists(im_data):
        os.makedirs(im_data)
    os.chdir(cwd)

    dataset = {}
    cnt = 0
    image_idx = 0
    iter = args.iter
    #cam_name = 'vision_eye_1'
    cam_name_list = ['vision_eye_front']
    cam_name_seg_peg = ['vision_eye_front_peg']
    cam_name_seg_hole = ['vision_eye_front_hole']
    peg_top = 'peg_dummy_top'
    peg_bottom = 'peg_dummy_bottom'
    peg_name = 'peg_in_arm'

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
                    'circle_7x14': ['circle_7x14', 'hole_keypoint_top14', 'hole_keypoint_bottom14', 'hole_keypoint_obj_bottom14'], }
    selected_hole = args.hole
    hole_name = hole_setting[selected_hole][0]
    hole_top = hole_setting[selected_hole][1]
    hole_bottom = hole_setting[selected_hole][2]
    hole_obj_bottom = hole_setting[selected_hole][3]

    #move_record = True
    #ompl_path_planning = False
    rob_arm = SingleRoboticArm()
    origin_hole_pose = rob_arm.get_object_matrix(hole_name)
    origin_hole_pos = origin_hole_pose[:3, 3]
    origin_hole_quat = rob_arm.get_object_quat(hole_name)
    rob_arm.finish()

    for i in range(iter):
        print('=' * 8 + 'Iteration '+ str(i) + '=' * 8)
        print('cnt:{}, image_idx:{}'.format(cnt, image_idx))
        rob_arm = SingleRoboticArm()
        # set init pos of hole
        hole_pos = np.array([random.uniform(0.0, 0.2), random.uniform(-0.45, -0.55), 0.035])
        rob_arm.set_object_position(hole_name, hole_pos)
        rob_arm.set_object_quat(hole_name, origin_hole_quat)
        #random_tilt(rob_arm, [hole_name], 0, 50)

        # set target pose
        random_z = random.uniform(0.011, 0.021)
        target_pose = rob_arm.get_object_matrix(obj_name=hole_top)
        target_pose[2, 3] += random_z

        # set start pose ( 2.5cm x 2.5cm x 1.5cm )
        random_x = random.uniform(-0.0125, 0.0125)  # random.uniform(0, 0.02)
        random_y = random.uniform(-0.0125, 0.0125)  # random.uniform(-0.03, 0.03)
        random_z = random.uniform(0, 0.015)  # random.uniform(-0.03, 0.03)
        start_pose = copy.deepcopy(target_pose)
        start_pose[:3, 3] += [random_x, random_y, random_z]

        rob_arm.movement(start_pose)
        loop_num = 0
        while True:
            robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            error = target_pose[:3,3] - robot_pose[:3,3]
            if np.linalg.norm(error) < 0.001 or loop_num > 20:
                print('loop:', loop_num)
                break
            else:
                robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
                gain = np.array([2.5, 2.5, 1])
                p = 0.2
                delta_translation = p * error * gain
                delta_rotation = np.array([[1,0,0],[0,1,0],[0,0,1]])
                r_euler = np.array([0,0,0])
                info = generate_one_im_anno(image_idx, cam_name_list, cam_name_seg_peg, cam_name_seg_hole, peg_top, peg_bottom, hole_top, hole_bottom, hole_obj_bottom,
                                            rob_arm, im_data_path, delta_rotation, delta_translation, robot_pose, r_euler)

                robot_pose[:3, 3] += delta_translation
                rob_arm.movement(robot_pose)
                dataset[cnt] = info
                cnt += 1
                image_idx += 1 * len(cam_name_list)
                loop_num += 1

        #insert test
        robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
        robot_pose[:3, 3] -= robot_pose[:3, 0] * 0.08  # x-axis
        rob_arm.movement(robot_pose)

        # score test
        peg_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name=peg_bottom)
        hole_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name=hole_bottom)
        dist = np.linalg.norm(peg_keypoint_bottom_pose[:3, 3] - hole_keypoint_bottom_pose[:3, 3])
        if dist < 0.001:
            print('Save!')
        else:
            print('Fail!')
            cnt -= loop_num
            image_idx -= loop_num * len(cam_name_list)
        rob_arm.finish()
        '''
        if ompl_path_planning == True:
            # move to target pose in order to get joint position
            gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            rob_arm.movement(target_pose)
            target_joint_config = rob_arm.get_joint_position()
            rob_arm.movement(gripper_pose)

            # determine the path
            interpolation_states = 5
            path = rob_arm.compute_path_from_joint_space(target_joint_config, interpolation_states)
            print('len of the path : ', int(len(path)/6))
            if len(path) > 0:
                lineHandle = rob_arm.visualize_path(path)
                if move_record == True:
                    # end point
                    print('iter: ' + str(i) + ' cnt: ' + str(cnt) + ' path: ' + str(0))
                    delta_rotation = np.array([[1.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 0.0, 1.0]])
                    delta_translation = np.array([0.0, 0.0, 0.0])
                    r_euler = np.array([0.0, 0.0, 0.0])
                    step_size = 0

                    gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTip')
                    pre_xyz = gripper_pose[:3, 3]
                    pre_rot = gripper_pose[:3, :3]
                    try:
                        info = generate_one_im_anno(cnt, cam_name_list, peg_top, peg_bottom, hole_top, hole_bottom,
                                                hole_obj_bottom, rob_arm,
                                                im_data_path, delta_rotation, delta_translation, gripper_pose,
                                                step_size, r_euler)
                    except:
                        continue
                    info_dic[cnt] = info
                    cnt += 1*len(cam_name_list)

                    path_len = int(len(path) / 6)
                    path_idx = 0
                    #for j in range(int(len(path) / 6)):
                    track_idx = 0
                    while path_idx < path_len:
                        print('iter: ' + str(i) + ' cnt: ' + str(cnt) + ' path: ' + str(track_idx +1))
                        track_idx += 1
                        p = path_idx / path_len
                        # step range: 0~100
                        step = 1+int(p*50)
                        #step = random.randint(1+int(p*10), 1+int(p*100))
                        if path_idx + step >= path_len:
                            subPath = path[path_idx * 6:]
                        else:
                            subPath = path[path_idx * 6:(path_idx + step) * 6]
                        path_idx += step
                        #print(subPath)
                        rob_arm.run_through_path(subPath)
                        step_size = step / (1+int(1*50))
                        if step_size >= 1.0:
                            step_size = 1.0
                        print('step_size', step_size)
                        gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTip')
                        cnt_xyz = gripper_pose[:3, 3]
                        cnt_rot = gripper_pose[:3, :3]
                        delta_translation = pre_xyz - cnt_xyz
                        # pre <==> target
                        # cnt <==> source
                        cnt_rot_t = np.transpose(cnt_rot)
                        delta_rotation = np.dot(cnt_rot_t, pre_rot)
                        r = R.from_matrix(delta_rotation)
                        r_euler = r.as_euler('zyx', degrees=True)
                        #print('delta_translation', delta_translation)
                        print('delta_rotation', delta_rotation)
                        print('r_euler', r_euler)
                        try:
                            info = generate_one_im_anno(cnt, cam_name_list, peg_top, peg_bottom, hole_top, hole_bottom,
                                                    hole_obj_bottom, rob_arm,
                                                    im_data_path, delta_rotation, delta_translation, gripper_pose,
                                                    step_size, r_euler)
                        except:
                            continue
                        info_dic[cnt] = info
                        cnt += 1*len(cam_name_list)
                        pre_rot = cnt_rot.copy()
                        pre_xyz = cnt_xyz.copy()
                else:
                    rob_arm.run_through_all_path(path)
                rob_arm.clear_path_visualization(lineHandle)
        else:
            if move_record == True:
                # only record start point
                gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTip')
                pre_xyz = gripper_pose[:3, 3]
                pre_rot = gripper_pose[:3, :3]
                rob_arm.movement(target_pose)
                gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTip')
                dist = np.linalg.norm(gripper_pose[:3, 3] - target_pose[:3, 3])
                if dist > 0.0005:
                    rob_arm.finish()
                    print('skip', dist)
                    continue
                #_, tilt_degree = random_tilt(rob_arm, ['UR5_ikTarget'], 0, 5)
                #print('tilt degree', tilt_degree)
                gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTip')
                cnt_xyz = gripper_pose[:3, 3]
                cnt_rot = gripper_pose[:3, :3]
                delta_translation = pre_xyz - cnt_xyz
                # pre <==> target
                # cnt <==> source
                cnt_rot_t = np.transpose(cnt_rot)
                delta_rotation = np.dot(cnt_rot_t, pre_rot)
                r = R.from_matrix(delta_rotation)
                r_euler = r.as_euler('zyx', degrees=True)
                print('iter: ' + str(i) + ' cnt: ' + str(cnt))
                print('delta_translation', delta_translation)
                print('delta_rotation', delta_rotation)
                print('r_euler', r_euler)
                # step_size is not used here
                step_size = 0
                info = generate_one_im_anno(cnt, cam_name_list, peg_top, peg_bottom, hole_top, hole_bottom,
                                            hole_obj_bottom, rob_arm,
                                            im_data_path, delta_rotation, delta_translation, rel_delta_translation, gripper_pose, step_size,
                                            r_euler)
                #except:
                #    continue
                info_dic[cnt] = info
                cnt += 1*len(cam_name_list)
            else:
                rob_arm.movement(target_pose)
        '''

    f = open(os.path.join(anno_data_path, 'peg_in_hole.yaml'), 'w')
    yaml.dump(dataset, f , Dumper=yaml.RoundTripDumper)
    f.close()



if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print('Time elasped:{:.02f}'.format((end_time - start_time)/3600))
