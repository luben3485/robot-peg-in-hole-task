from env.single_robotic_arm import SingleRoboticArm
from keypoint_detection_grayscale import KeypointDetection
import numpy as np
import cv2
import math
import random
from bbox_detector import BboxDetection

rob_arm = SingleRoboticArm()
netpath = '/Users/cmlab/robot-peg-in-hole-task/mankey/experiment/ckpnt_box_fix_2_grayscale/checkpoint-100.pth'
#netpath = '/Users/cmlab/robot-peg-in-hole-task/mankey/experiment/box_ckpnt_grayscale_fix/checkpoint-100.pth'
kp_detection = KeypointDetection(netpath)
bbox_detection = BboxDetection('/Users/cmlab/robot-peg-in-hole-task/mrcnn/ckpnt')
#bbox = np.array([0, 0, 255, 255])


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


def main():
    init_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
    choose_hole = 0
    # detect empty hole
    #cam_name = 'vision_eye'
    cam_name = 'vision_fix3'
    down = False
    # random remove peg
    peg_list = list(range(6))
    num_remove = random.randint(2, 3)
    print('number of remove peg:{:d}'.format(num_remove))
    #remove_list = random.sample(peg_list, num_remove)
    # test
    remove_list = [0,2,5]
    for i, v in enumerate(remove_list):
        print('remove peg_idx:{:d}'.format(v))
        rob_arm.set_object_position('peg_large' + str(v), [2, 0.2 * i, 0.1])
        peg_list.pop(peg_list.index(v))
    if cam_name == 'vision_eye':
        hole_check_pose = rob_arm.get_object_matrix(obj_name='hole_check')
        rob_arm.movement(hole_check_pose)


    rgb = rob_arm.get_rgb(cam_name=cam_name)
    depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=3)

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
    #predict bbox
    bbox = bbox_detection.predict(rgb)
    camera_keypoint, keypointxy_depth_realunit = kp_detection.inference(cv_rgb=rgb, cv_depth=depth_mm, bbox=bbox)
    depth_hole_list = []
    for i in range(6):
        x = int(keypointxy_depth_realunit[2 * (i + 1)][0])
        y = int(keypointxy_depth_realunit[2 * (i + 1)][1])
        #print(x,y,depth_mm[y][x])
        depth_hole_list.append(depth_mm[y][x])

    choose_hole = depth_hole_list.index(max(depth_hole_list))
    # test
    choose_hole = 5
    print('Get ready to insert hole {:d} !'.format(choose_hole))

    # === test ====
    # gt grasp
    #peg_idx = 5
    #peg_top = 'peg_keypoint_top_large' + str(peg_idx)
    peg_top = 'peg_keypoint_top_large'
    peg_keypoint_pose = rob_arm.get_object_matrix(obj_name=peg_top)
    grasp_pose = peg_keypoint_pose.copy()
    grasp_pose[0, 3] += 0.0095
    grasp_pose[2, 3] -= 0.015
    rob_arm.gt_run_grasp(grasp_pose)

    # pull up the peg to avoid collision  (hole's height = 0.2)
    #grasp_pose[2, 3] += 0.18
    #grasp_pose[2, 3] += 0.23
    grasp_pose[2, 3] += 0.24
    rob_arm.movement(grasp_pose)
    #grasp_pose[0, 3] = 0.15
    grasp_pose[0, 3] -= 0.2
    #grasp_pose[1, 3] = -0.475
    rob_arm.movement(grasp_pose)

    # Get RGB images from camera
    rgb = rob_arm.get_rgb(cam_name=cam_name)
    # rotate rgb 180
    (h, w) = rgb.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rgb = cv2.warpAffine(rgb, M, (w, h))
    # Get depth from camera
    depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=3)
    # rotate depth 180
    (h, w) = depth.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    depth = cv2.warpAffine(depth, M, (w, h))

    depth_mm = (depth * 1000).astype(np.uint16)  # type: np.uint16 ; uint16 is needed by keypoint detection network

    #cv2.imwrite('rgb.png', rgb)
    #cv2.imwrite('depth.png', depth_mm)
    #rgb = cv2.imread('rgb.png', cv2.IMREAD_COLOR)
    #depth_mm = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)
    bbox = bbox_detection.predict(cv_rgb=rgb)
    print('bbox', bbox)
    camera_keypoint, keypointxy_depth_realunit = kp_detection.inference(cv_rgb=rgb, cv_depth=depth_mm, bbox=bbox)
    kp_detection.visualize(cv_rgb=rgb, cv_depth=depth_mm, keypoints=camera_keypoint)
    kp_detection.visualize_2d(bbox=bbox, cv_rgb=rgb, keypoints=keypointxy_depth_realunit)

    # === test ===

    ## gt grasp

    peg_keypoint_top_pose = rob_arm.get_object_matrix(obj_name='peg_keypoint_top_large')
    grasp_pose = peg_keypoint_top_pose.copy()
    grasp_pose[0, 3] += 0.0095
    grasp_pose[2, 3] -= 0.015
    rob_arm.gt_run_grasp(grasp_pose)
    grasp_pose[2, 3] += 0.24
    rob_arm.movement(grasp_pose)
    grasp_pose[0, 3] -= 0.2
    rob_arm.movement(grasp_pose)
    #fix_init_pose = rob_arm.get_object_matrix(obj_name='fix_init')
    #rob_arm.movement(fix_init_pose)
    #fix_init_pose[2,3] -=0.05
    #rob_arm.movement(fix_init_pose)
    print('servoing...')
    err_tolerance = 0.08
    alpha_err = 0.7
    alpha_target = 0.7
    filter_robot_pose = rob_arm.get_object_matrix('UR5_ikTip')
    err_rolling, err_size_rolling = None, err_tolerance * 10
    cnt = 0
    while err_size_rolling > err_tolerance:
        print('========cnt:' + str(cnt) + '=========\n')
        cnt += 1
        robot_pose = rob_arm.get_object_matrix('UR5_ikTip')
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

        # test
        cv2.imwrite('rgb.png', rgb)
        cv2.imwrite('depth.png', depth_mm)
        rgb = cv2.imread('rgb.png', cv2.IMREAD_COLOR)
        depth_mm = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)

        bbox = bbox_detection.predict(cv_rgb=rgb)
        print('bbox',bbox)
        camera_keypoint, keypointxy_depth_realunit = kp_detection.inference(cv_rgb=rgb, cv_depth=depth_mm, bbox=bbox)
        #kp_detection.visualize_2d(cv_rgb=rgb, keypoints=keypointxy_depth_realunit)
        #print(camera_keypoint)
        #print('xyd',keypointxy_depth_realunit)
        cam_pos, cam_quat, cam_matrix= rob_arm.get_camera_pos(cam_name=cam_name)
        cam_matrix = np.array(cam_matrix).reshape(3,4)
        camera2world_matrix = cam_matrix
        world_keypoints = np.zeros((14, 3), dtype=np.float64)
        for idx, keypoint in enumerate(camera_keypoint):
            keypoint = np.append(keypoint, 1)
            world_keypoints[idx, :] = np.dot(camera2world_matrix, keypoint)[0:3]
        print('world keypoints', world_keypoints)
        peg_keypoint_top = world_keypoints[0]
        peg_keypoint_bottom = world_keypoints[1]
        hole_keypoint_top = world_keypoints[2*(choose_hole+1)]
        hole_keypoint_bottom = world_keypoints[2*(choose_hole+1)+1]
        err = hole_keypoint_top - peg_keypoint_bottom

        # err*= 0.1
        print('err vector', err)
        dis = math.sqrt(math.pow(err[0], 2) + math.pow(err[1], 2))
        print('Distance:', dis)
        if cnt>= 0 :
            kp_detection.visualize(cv_rgb=rgb, cv_depth=depth_mm, keypoints=camera_keypoint)
            kp_detection.visualize_2d(bbox=bbox, cv_rgb=rgb, keypoints=keypointxy_depth_realunit)
        tilt = False
        if dis < 0.01 and tilt == True:
            hole_dir = hole_keypoint_bottom - hole_keypoint_top
            peg_dir = peg_keypoint_bottom - peg_keypoint_top
            dot_product = np.dot(peg_dir, hole_dir)
            degree =  math.degrees(math.acos(dot_product/(np.linalg.norm(peg_dir)* np.linalg.norm(hole_dir))))
            print('Degree:', degree)
            if degree > 2:
                cross_product = np.cross(peg_dir, hole_dir)
                cross_product = cross_product / np.linalg.norm(cross_product)
                #degree -= 0.5
                w = math.cos(math.radians(degree/ 2))
                x = math.sin(math.radians(degree/ 2)) * cross_product[0]
                y = math.sin(math.radians(degree/ 2)) * cross_product[1]
                z = math.sin(math.radians(degree/ 2)) * cross_product[2]
                quat = [w, x, y, z]
                rot_pose = quaternion_matrix(quat)
                rot_matrix = np.dot(rot_pose[:3, :3], robot_pose[:3, :3])
                robot_pose[:3, :3] = rot_matrix
                filter_robot_pose[:3, :3] = rot_matrix
                rob_arm.movement(robot_pose)
        # robot_pose[1,3] += (-err[0])
        # robot_pose[2,3] += (-err[1])
        # robot_pose[2,3] += err[1]
        robot_pose[:2, 3] += err[:2]

        #print('unfiltered robot pose', robot_pose[:3, 3])
        filter_robot_pose[:3,3] = alpha_target * filter_robot_pose[:3,3] + (1 - alpha_target) * robot_pose[:3,3]
        #print('filtered robot pose', filter_robot_pose[:3, 3])
        # robot moves to filter_robot_pose
        rob_arm.movement(filter_robot_pose)
        # self.movement(robot_pose)
        if err_rolling is None:
            err_rolling = err
        err_rolling = alpha_err * err_rolling + (1 - alpha_err) * err
        err_size_rolling = alpha_err * err_size_rolling + (1 - alpha_err) * np.linalg.norm(err_rolling)

    if down:
        downward(cam_name=cam_name, choose_hole=choose_hole)
        # after insertion
        move_pose = rob_arm.get_object_matrix('UR5_ikTip')
        dir = np.array([0,0,-1])
        move_pose[:3, 3] -= dir * 0.2
        rob_arm.movement(move_pose)
    else:
        # insertion
        print('insertion')
        dir = np.array([0,0,-1])
        move_pose = rob_arm.get_object_matrix('UR5_ikTip')
        move_pose[:3, 3] += dir * 0.2
        rob_arm.insertion(move_pose)


        # after insertion
        move_pose[:3, 3] -= dir * 0.2
        rob_arm.movement(move_pose)

    # go back to initial pose
    rob_arm.movement(init_pose)

    rob_arm.finish()

def downward(cam_name, choose_hole):
    print('downward servoing...')
    err_tolerance = 0.08
    alpha_err = 0.7
    alpha_target = 0.7
    filter_robot_pose = rob_arm.get_object_matrix('UR5_ikTip')
    err_rolling, err_size_rolling = None, err_tolerance * 10
    cnt = 0
    while err_size_rolling > err_tolerance:
        print('========cnt:' + str(cnt) + '=========\n')
        cnt += 1
        robot_pose = rob_arm.get_object_matrix('UR5_ikTip')
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
        camera_keypoint, keypointxy_depth_realunit = kp_detection.inference(cv_rgb=rgb, cv_depth=depth_mm, bbox=bbox)
        cam_pos, cam_quat, cam_matrix = rob_arm.get_camera_pos(cam_name=cam_name)
        cam_matrix = np.array(cam_matrix).reshape(3, 4)
        camera2world_matrix = cam_matrix
        world_keypoints = np.zeros((14, 3), dtype=np.float64)
        for idx, keypoint in enumerate(camera_keypoint):
            keypoint = np.append(keypoint, 1)
            world_keypoints[idx, :] = np.dot(camera2world_matrix, keypoint)[0:3]
        print('world keypoints', world_keypoints)
        peg_keypoint_top = world_keypoints[0]
        peg_keypoint_bottom = world_keypoints[1]
        hole_keypoint_top = world_keypoints[2 * (choose_hole + 1)]
        hole_keypoint_bottom = world_keypoints[2 * (choose_hole + 1) + 1]
        err = hole_keypoint_bottom - peg_keypoint_bottom
        print(err)
        if abs(err[2]) < 0.02:
            break
        # err*= 0.1
        print('err vector', err)
        dis = math.sqrt(math.pow(err[0], 2) + math.pow(err[1], 2) + math.pow(err[2], 2))
        print('Distance:', dis)
        if cnt >= 1:
            kp_detection.visualize(cv_rgb=rgb, cv_depth=depth_mm, keypoints=camera_keypoint)

        tilt = False
        if dis < 0.01 and tilt == True:
            hole_dir = hole_keypoint_bottom - hole_keypoint_top
            peg_dir = peg_keypoint_bottom - peg_keypoint_top
            dot_product = np.dot(peg_dir, hole_dir)
            degree = math.degrees(math.acos(dot_product / (np.linalg.norm(peg_dir) * np.linalg.norm(hole_dir))))
            print('Degree:', degree)
            if degree > 2:
                cross_product = np.cross(peg_dir, hole_dir)
                cross_product = cross_product / np.linalg.norm(cross_product)
                # degree -= 0.5
                w = math.cos(math.radians(degree / 2))
                x = math.sin(math.radians(degree / 2)) * cross_product[0]
                y = math.sin(math.radians(degree / 2)) * cross_product[1]
                z = math.sin(math.radians(degree / 2)) * cross_product[2]
                quat = [w, x, y, z]
                rot_pose = quaternion_matrix(quat)
                rot_matrix = np.dot(rot_pose[:3, :3], robot_pose[:3, :3])
                robot_pose[:3, :3] = rot_matrix
                filter_robot_pose[:3, :3] = rot_matrix
                rob_arm.movement(robot_pose)
        # robot_pose[1,3] += (-err[0])
        # robot_pose[2,3] += (-err[1])
        # robot_pose[2,3] += err[1]
        robot_pose[:3, 3] += err[:3]

        # print('unfiltered robot pose', robot_pose[:3, 3])
        filter_robot_pose[:3, 3] = alpha_target * filter_robot_pose[:3, 3] + (1 - alpha_target) * robot_pose[:3, 3]
        # print('filtered robot pose', filter_robot_pose[:3, 3])
        # robot moves to filter_robot_pose
        rob_arm.movement(filter_robot_pose)
        # self.movement(robot_pose)
        if err_rolling is None:
            err_rolling = err
        err_rolling = alpha_err * err_rolling + (1 - alpha_err) * err
        err_size_rolling = alpha_err * err_size_rolling + (1 - alpha_err) * np.linalg.norm(err_rolling)
    rob_arm.open_gripper(mode=1)
if __name__ == '__main__':
    main()
