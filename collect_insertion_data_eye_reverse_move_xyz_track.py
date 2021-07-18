from env.single_robotic_arm import SingleRoboticArm
import yaml
import cv2
import os
import numpy as np
from ruamel import yaml
import math
import random
import time
from transforms3d.quaternions import mat2quat, quat2axangle, quat2mat

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


def generate_one_im_anno(cnt, track_idx, state_idx, cam_name, peg_top, peg_bottom, hole_top, hole_bottom, rob_arm, im_data_path, delta_rotation_quat, delta_translation, gripper_pose, step_size):

    # Get keypoint location
    peg_keypoint_top_pose = rob_arm.get_object_matrix(peg_top)
    peg_keypoint_bottom_pose = rob_arm.get_object_matrix(peg_bottom)
    hole_keypoint_top_pose = rob_arm.get_object_matrix(hole_top)
    hole_keypoint_bottom_pose = rob_arm.get_object_matrix(hole_bottom)

    peg_keypoint_top_in_world = peg_keypoint_top_pose[:3,3]
    peg_keypoint_bottom_in_world = peg_keypoint_bottom_pose[:3, 3]
    hole_keypoint_top_in_world = hole_keypoint_top_pose[:3,3]
    hole_keypoint_bottom_in_world = hole_keypoint_bottom_pose[:3, 3]


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


    depth_mm = (depth*1000).astype(np.uint16) # type: np.uint16 ; uint16 is needed by keypoint detection network

    #rob_arm.visualize_image(rgb=rgb, depth=depth)

    # Save image
    rgb_file_name = str(cnt).zfill(6)+'_rgb.png'
    rgb_file_path = os.path.join(im_data_path, rgb_file_name)
    depth_file_name = str(cnt).zfill(6)+'_depth.png'
    depth_file_path = os.path.join(im_data_path, depth_file_name)
    cv2.imwrite(rgb_file_path, rgb)
    cv2.imwrite(depth_file_path, depth_mm)

    # Get camera info
    cam_pos, cam_quat, cam_matrix= rob_arm.get_camera_pos(cam_name=cam_name)
    
 
    camera_to_world = {'quaternion':{'x':cam_quat[0], 'y':cam_quat[1], 'z':cam_quat[2], 'w':cam_quat[3]},
                                'translation':{'x':cam_pos[0], 'y':cam_pos[1], 'z':cam_pos[2]}}

    camera2world_map = camera_to_world
    world2camera = world2camera_from_map(camera2world_map)
    
    peg_keypoint_top_in_world = np.append(peg_keypoint_top_in_world,[1],axis = 0).reshape(4,1)
    peg_keypoint_bottom_in_world = np.append(peg_keypoint_bottom_in_world, [1], axis=0).reshape(4, 1)
    hole_keypoint_top_in_world = np.append(hole_keypoint_top_in_world,[1],axis = 0).reshape(4,1)
    hole_keypoint_bottom_in_world = np.append(hole_keypoint_bottom_in_world,[1],axis = 0).reshape(4,1)

    peg_keypoint_top_in_camera = world2camera.dot(peg_keypoint_top_in_world)
    peg_keypoint_bottom_in_camera = world2camera.dot(peg_keypoint_bottom_in_world)
    hole_keypoint_top_in_camera = world2camera.dot(hole_keypoint_top_in_world)
    hole_keypoint_bottom_in_camera = world2camera.dot(hole_keypoint_bottom_in_world)


    peg_keypoint_top_in_camera =  peg_keypoint_top_in_camera[:3].reshape(3,)
    peg_keypoint_bottom_in_camera = peg_keypoint_bottom_in_camera[:3].reshape(3, )
    hole_keypoint_top_in_camera =  hole_keypoint_top_in_camera[:3].reshape(3,)
    hole_keypoint_bottom_in_camera = hole_keypoint_bottom_in_camera[:3].reshape(3, )

    keypoint_in_camera = np.array([peg_keypoint_top_in_camera, peg_keypoint_bottom_in_camera, hole_keypoint_top_in_camera, hole_keypoint_bottom_in_camera])
    #keypoint_in_camera = np.array([peg_keypoint_bottom_in_camera, hole_keypoint_top_in_camera])
    xy_depth = point2pixel(keypoint_in_camera)
    xy_depth_list = []
    (h, w) = rgb.shape[:2]
    for i in range(len(xy_depth)):

        x = int(xy_depth[i][0])
        y = int(xy_depth[i][1])
        z = int(xy_depth[i][2])
        xy_depth_list.append([x,y,z])



    rot_vec, rot_theta = quat2axangle(delta_rotation_quat)
    delta_rotation_matrix = quat2mat(delta_rotation_quat)



    info = {'track_idx': track_idx,
            'state_idx': state_idx,
            '3d_keypoint_camera_frame': [[float(v) for v in peg_keypoint_top_in_camera], [float(v) for v in peg_keypoint_bottom_in_camera], [float(v) for v in hole_keypoint_top_in_camera], [float(v) for v in hole_keypoint_bottom_in_camera]],
            #'3d_keypoint_camera_frame':[ [float(v) for v in peg_keypoint_bottom_in_camera],
            #                         [float(v) for v in hole_keypoint_top_in_camera]],
            'bbox_bottom_right_xy': [rgb.shape[1], rgb.shape[0]],
            'bbox_top_left_xy': [0,0],
            'camera_to_world': camera_to_world,
            'depth_image_filename': depth_file_name,
            'rgb_image_filename': rgb_file_name,
            'keypoint_pixel_xy_depth': xy_depth_list,
            'delta_rotation_quat': delta_rotation_quat.tolist(),
            'delta_rotation_matrix': delta_rotation_matrix.tolist(),
            'delta_translation': delta_translation.tolist(),
            'delta_rot_vec': rot_vec.tolist(),
            'delta_rot_theta': rot_theta,
            'gripper_pose': gripper_pose.tolist(),
            'step_size': step_size
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

def obj_init_pos(ran_hole_pos, ran_peg_pos):
    # position based on hole
    #x = random.uniform(0.1,0.375)
    #y = random.uniform(-0.65,-0.375)
    x = 0.2
    y = -0.5
    ran_hole_pos[0] = x
    ran_hole_pos[1] = y
    ran_peg_pos[0] = x
    ran_peg_pos[1] = y
    ran_peg_pos[2] = 7.1168e-02

    return ran_hole_pos, ran_peg_pos

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

def main():
    rob_arm = SingleRoboticArm()
    data_root = '/Users/cmlab/data/pdc/logs_proto'
    date = '2021-06-20'
    anno_data = 'xyz_track_insertion_' + date + '/processed'
    im_data = 'xyz_track_insertion_' + date + '/processed/images'
    anno_data_path = os.path.join(data_root, anno_data)
    im_data_path = os.path.join(data_root, im_data)

    # create folder
    cwd = os.getcwd()
    os.chdir(data_root)
    if not os.path.exists(im_data):
        os.makedirs(im_data)
    os.chdir(cwd)

    info_dic = {}
    cnt = 0
    iter = 4
    cam_name = 'vision_eye'
    peg_top = 'peg_keypoint_top'
    peg_bottom = 'peg_keypoint_bottom'
    hole_top = 'hole_keypoint_top'
    hole_bottom = 'hole_keypoint_bottom'
    move_record = True

    origin_hole_pose = rob_arm.get_object_matrix('hole')
    origin_hole_pos = origin_hole_pose[:3, 3]
    origin_hole_quat = rob_arm.get_object_quat('hole')

    origin_peg_pose = rob_arm.get_object_matrix('peg')
    origin_peg_pos = origin_peg_pose[:3, 3]
    origin_peg_quat = rob_arm.get_object_quat('peg')

    #origin_ur5_pose = rob_arm.get_object_matrix('UR5_ikTarget')
    #origin_ur5_pos =  origin_ur5_pose[:3, 3]
    #origin_ur5_quat = rob_arm.get_object_quat('UR5_ikTarget')
    gripper_init_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')

    dir = np.array([0.0,0.0,-1.0])

    for track_idx in range(iter):
        track = {}
        # init pos of peg nad hole; now ths pos is  not random
        ran_hole_pos, ran_peg_pos = obj_init_pos(origin_hole_pos.copy(),origin_peg_pos.copy())
        rob_arm.set_object_position('hole', ran_hole_pos)
        rob_arm.set_object_position('peg', ran_peg_pos)
        rob_arm.set_object_quat('hole', origin_hole_quat)
        rob_arm.set_object_quat('peg', origin_peg_quat)
        rob_arm.movement(gripper_init_pose)
        rob_arm.open_gripper(mode=1)

        # gt grasp
        peg_keypoint_pose = rob_arm.get_object_matrix(obj_name='peg_keypoint_bottom')
        grasp_pose = peg_keypoint_pose.copy()
        grasp_pose[:3, 3] -= dir * 0.085
        grasp_pose[0, 3] -= 0.0015
        rob_arm.gt_run_grasp(grasp_pose)

        # pull up the peg to avoid collision  (hole's height = 0.07)
        #grasp_pose[:3, 3] -= dir * 0.075
        grasp_pose[:3, 3] -= dir * 0.049
        rob_arm.movement(grasp_pose)

        # peg hole alignment
        peg_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name='peg_keypoint_bottom')
        hole_keypoint_top_pose = rob_arm.get_object_matrix(obj_name='hole_keypoint_top')
        err_xy = hole_keypoint_top_pose[:2, 3] - peg_keypoint_bottom_pose[:2, 3]
        gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
        gripper_pose[:2, 3] += err_xy
        rob_arm.movement(gripper_pose)

        # move peg to random x,y,z
        gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
        delta_x = random.uniform(-0.08, 0.08)
        delta_y = random.uniform(-0.08, 0.08)
        delta_z = random.uniform(0.12, 0.17)
        #delta_x, delta_y = random_gripper_xy(ran_hole_pos)

        move_vec = np.array([delta_x, delta_y, delta_z])

        if move_record:
            '''
            step_size = 0.04
            for j in range(int(1 / step_size)):
                gripper_pose[:3, 3] += move_vec * step_size
                rob_arm.movement(gripper_pose)
                print('track: '+str(track_idx)+' cnt: '+str(cnt)+' pick: '+str(j))
                delta_rotation_quat = np.array([1.0, 0.0, 0.0, 0.0])
                delta_translation = - move_vec * step_size
                gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
                info = generate_one_im_anno(cnt, track_idx, state_idx, cam_name, peg_top, peg_bottom, hole_top, hole_bottom, rob_arm, im_data_path, delta_rotation_quat, delta_translation, gripper_pose)
                track[state_idx] = info
                cnt += 1
                state_idx += 1
            info_dic[track_idx] = track
            '''
            tmp_info_list = []
            distance = np.linalg.norm(move_vec) # max 0.173
            move_dir = move_vec / distance
            sum_move = 0.
            step_size_t = 0.
            const_factor = 0.02
            j = 0
            while sum_move <= distance:
                step_size = math.pow(step_size_t, 3)
                print(step_size)
                if step_size > 1:
                    break
                gripper_pose[:3, 3] += move_dir * step_size * const_factor
                sum_move = sum_move + step_size * const_factor
                step_size_t += 0.03
                rob_arm.movement(gripper_pose)
                print('track: '+str(track_idx)+' cnt: '+str(cnt)+' pick: '+str(j))
                j+=1
                delta_rotation_quat = np.array([1.0, 0.0, 0.0, 0.0])
                delta_translation = - move_vec * step_size
                gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
                state_idx = -1
                info = generate_one_im_anno(cnt, track_idx, state_idx, cam_name, peg_top, peg_bottom, hole_top, hole_bottom, rob_arm, im_data_path, delta_rotation_quat, delta_translation, gripper_pose, step_size)
                tmp_info_list.append(info)
                cnt += 1
            tmp_info_list_len = len(tmp_info_list)
            for i, info in enumerate(tmp_info_list):
                reversed_idx = tmp_info_list_len - i - 1
                track[reversed_idx] = info
                track[reversed_idx]['state_idx'] = reversed_idx
            info_dic[track_idx] = track

        else:
            gripper_pose[:3, 3] += move_vec
            rob_arm.movement(gripper_pose)

    rob_arm.finish()
    f = open(os.path.join(anno_data_path, 'peg_in_hole.yaml'),'w')
    yaml.dump(info_dic,f , Dumper=yaml.RoundTripDumper)
    f.close()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Time elasped:{:.02f}'.format((end_time - start_time)/3600))

