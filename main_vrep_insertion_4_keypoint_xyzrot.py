from env.single_robotic_arm import SingleRoboticArm
from keypoint_detection_xyzrot import KeypointDetection
import numpy as np
import cv2
import math
from transforms3d.quaternions import mat2quat, quat2axangle, quat2mat, qmult
import random
from scipy.spatial.transform import Rotation as R

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
    rot_dir = np.random.normal(size=(3,))
    rot_dir = rot_dir / np.linalg.norm(rot_dir)
    tilt_degree = random.uniform(min_tilt_degree, max_tilt_degree)
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

def predict_xyzrot_from_camera(cam_name, kp_detection, rob_arm, enable_gripper_pose, visualize, output_mode):
    gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
    bbox = np.array([0, 0, 255, 255])
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
    if output_mode == 'cls':
        camera_keypoint, delta_rot_x_pred, delta_rot_y_pred, delta_rot_z_pred, delta_xyz_pred, step_size_pred = kp_detection.inference(cv_rgb=rgb, cv_depth=depth, bbox=bbox, gripper_pose=gripper_pose, enable_gripper_pose=enable_gripper_pose)
    elif output_mode == 'reg':
        camera_keypoint, delta_rot_pred, delta_xyz_pred, step_size_pred = kp_detection.inference(cv_rgb=rgb, cv_depth=depth, bbox=bbox, gripper_pose=gripper_pose, enable_gripper_pose=enable_gripper_pose)
    # convert keypoint
    cam_pos, cam_quat, cam_matrix = rob_arm.get_camera_pos(cam_name=cam_name)
    cam_matrix = np.array(cam_matrix).reshape(3, 4)
    camera2world_matrix = cam_matrix
    world_keypoints = np.zeros((4, 3), dtype=np.float64)
    for idx, keypoint in enumerate(camera_keypoint):
        keypoint = np.append(keypoint, 1)
        world_keypoints[idx, :] = np.dot(camera2world_matrix, keypoint)[0:3]
    #print('world keypoints', world_keypoints)
    if visualize:
        kp_detection.visualize(cv_rgb=rgb, cv_depth=depth_mm, keypoints=camera_keypoint)

    if output_mode == 'cls':
        ### from euler angle to rotation matrix
        print('cls(xyz):', delta_rot_x_pred, delta_rot_y_pred, delta_rot_z_pred)
        x = (delta_rot_x_pred - 2 ) * 10
        y = (delta_rot_y_pred - 6 ) * 10
        z = (delta_rot_z_pred - 6 ) * 10
        print('degree(xyz):', x, y, z)
        r = R.from_euler('zyx', [z, y, x], degrees=True)
        delta_rot_pred = r.as_matrix()

    return world_keypoints, delta_rot_pred, delta_xyz_pred, step_size_pred[0], gripper_pose

def main():
    rob_arm = SingleRoboticArm()
    init_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
    #netpath = '/Users/cmlab/robot-peg-in-hole-task/mankey/experiment/ckpnt_xyzrot_coord_small_range_0801/checkpoint-200.pth'
    netpath = '/Users/cmlab/robot-peg-in-hole-task/mankey/experiment/ckpnt_xyzrot_coord_small_range_1015_no_kpt_reg_filter_400/checkpoint-180.pth'
    enable_gripper_pose = False
    visualize = False
    enableKeypointPos = False
    output_mode = 'reg'
    kp_detection = KeypointDetection(netpath, output_mode, enableKeypointPos)
    cam_name = 'vision_eye'
    peg_top = 'peg_keypoint_top2'
    peg_bottom = 'peg_keypoint_bottom2'
    hole_top = 'hole_keypoint_top'
    hole_bottom = 'hole_keypoint_bottom'
    peg_name = 'peg2'
    hole_name = 'hole'

    rob_arm.set_object_position(hole_name, np.array([0.2,-0.5,3.6200e-02]))
    #random_tilt(rob_arm, [hole_name], 0, 60)
    ### specific rule
    tilt_x = random.uniform(-1,1)
    tilt_y = math.sqrt(1 - math.pow(tilt_x, 2))
    if random.uniform(0,1) < 0.5:
        tilt_y = -tilt_y
    rot_dir = np.array([tilt_x, tilt_y, 0])
    tilt_degree = random.uniform(0,60)
    specific_tilt(rob_arm, [hole_name], rot_dir, tilt_degree)

    start_pose = rob_arm.get_object_matrix('UR5_ikTip')
    hole_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
    delta_move = np.array([random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03), random.uniform(0.09, 0.12)])
    start_pos = hole_top_pose[:3, 3]
    start_pos += delta_move
    # test
    #start_pos += [0, 0, 0.09]
    # start pose
    start_pose[:3, 3] = start_pos

    # gt grasp
    peg_keypoint_top_pose = rob_arm.get_object_matrix(obj_name=peg_top)
    grasp_pose = peg_keypoint_top_pose.copy()
    # gripper offset
    grasp_pose[:3, 3] += peg_keypoint_top_pose[:3, 1] * 0.0015  # y-axis
    rob_arm.gt_run_grasp(grasp_pose)
    grasp_pose[2, 3] += 0.2
    rob_arm.movement(grasp_pose)
    rob_arm.movement(start_pose)
    print('start...')
    ### start
    step_size_pred = 1.0
    #while step_size_pred > 0.006:
    while True:
        world_keypoints, delta_rot_pred, delta_xyz_pred, step_size_pred, gripper_pose = predict_xyzrot_from_camera(cam_name, kp_detection, rob_arm, enable_gripper_pose, visualize, output_mode)
        print('step_size', step_size_pred)
        print('delta_xyz_pred', delta_xyz_pred)
        print('delta_rot_pred', delta_rot_pred)
        #rot_matrix = np.dot(delta_rot_pred[:3, :3], gripper_pose[:3, :3])
        rot_matrix = np.dot(gripper_pose[:3, :3], delta_rot_pred[:3, :3])
        gripper_pose[:3, :3] = rot_matrix
        gripper_pose[:3, 3] += delta_xyz_pred
        rob_arm.movement(gripper_pose)
    # peg-hole insertion
    '''
    # move to top of hole
    gripper_pose = rob_arm.get_object_matrix('UR5_ikTip')
    gripper_pose[:2, 3] = world_keypoints[2, :2]
    rob_arm.movement(gripper_pose)

    for i in range(1):
        world_keypoints = predict_world_keypoints_from_camera(cam_name, kp_detection, rob_arm)
        hole_insert_dir = world_keypoints[3] - world_keypoints[2]  # x-axis
        hole_insert_dir = hole_insert_dir / np.linalg.norm(hole_insert_dir)
        peg_insert_dir = world_keypoints[1] - world_keypoints[0]  # x-axis
        peg_insert_dir = peg_insert_dir / np.linalg.norm(peg_insert_dir)
        dot_product = np.dot(peg_insert_dir, hole_insert_dir)
        degree = math.degrees(math.acos(dot_product / (np.linalg.norm(peg_insert_dir) * np.linalg.norm(hole_insert_dir))))
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

        #hole_keypoint_top_pose = rob_arm.get_object_matrix(obj_name=hole_top)
        #robot_pose[:3, 3] = hole_keypoint_top_pose[:3, 3] + hole_keypoint_top_pose[:3, 0] * 0.07
        robot_pose[:3, 3] = world_keypoints[2] + (-hole_insert_dir) * 0.07
        robot_pose[:3, 3] += robot_pose[:3, 1] * 0.003  # y-axis
        rob_arm.movement(robot_pose)
    '''
    # insertion
    world_keypoints, delta_rot_pred, delta_xyz_pred, step_size_pred, gripper_pose = predict_xyzrot_from_camera(cam_name,
                                                                                                               kp_detection,
                                                                                                               rob_arm,
                                                                                                               enable_gripper_pose,
                                                                                                               visualize,
                                                                                                               output_mode)
    #world_keypoints, delta_rot_pred, delta_xyz_pred, step_size_pred = predict_xyzrot_from_camera(cam_name, kp_detection, rob_arm)
    peg_insert_dir = world_keypoints[1] - world_keypoints[0]  # x-axis
    peg_insert_dir = peg_insert_dir / np.linalg.norm(peg_insert_dir)
    robot_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
    robot_pose[:3, 3] += peg_insert_dir * 0.05  # x-axis
    print('insertion')
    rob_arm.insertion(robot_pose)

    # after insertion
    robot_pose[:3, 3] -= peg_insert_dir * 0.05
    rob_arm.movement(robot_pose)

    # go back to initial pose
    rob_arm.movement(init_pose)

    ### arm move test
    '''   
    robot_pose = rob_arm.get_object_matrix('UR5_ikTip')
    robot_pose[2,3] += 10 
    rob_arm.movement(robot_pose)
    '''
    ### only grasp
    '''
    option = 'naive'
    if option == 'gdn':
        for mask in masks:
            rob_arm.visualize_image(mask, depth, rgb)
            grasp_list = rob_arm.get_grasping_candidate(depth, mask, 75, 75)
            #print(grasp_list)
            rob_arm.run_grasp(grasp_list, 1)
 
    else:
        grasp_list = rob_arm.naive_grasp_detection(rgb, depth)
        #print(grasp_list)
        rob_arm.run_grasp(grasp_list, 1, use_gdn = False)
    '''
    rob_arm.finish()


if __name__ == '__main__':
    main()
