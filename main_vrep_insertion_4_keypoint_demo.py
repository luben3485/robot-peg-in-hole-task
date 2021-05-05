from env.single_robotic_arm import SingleRoboticArm
#from single_robotic_arm import SingleRoboticArm
from keypoint_detection import KeypointDetection
import numpy as np
import cv2
import math


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
    rob_arm = SingleRoboticArm()
    init_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
    netpath = '/home/luben/robot-peg-in-hole-task/mankey/experiment/ckpnt_0420/checkpoint-100.pth'
    kp_detection = KeypointDetection(netpath)

    #extrinsic = rob_arm.get_camera_matrix(cam_name='Vision_sensor0')
    #print(extrinsic)
    #intrinsic = rob_arm.get_intrinsic_matrix()
    #print(intrinsic)
    #rgb = rob_arm.get_rgb(cam_name='Vision_sensor0')
    #print(rgb.shape)
    #depth = rob_arm.get_depth(cam_name='Vision_sensor0', near_plane=0.02, far_plane=1)
    #depth_mm = (depth * 1000).astype(np.uint16)  # type: np.uint16 ; uint16 is needed by keypoint detection network
    #print(depth.shape)
    #masks = rob_arm.get_mask()


    ### gt open-loop
    '''  
    hole_keypoint_pos = rob_arm.get_target_matrix('hole_keypoint')
    action_matrix = rob_arm.get_correct_action_matrix('peg_keypoint')
    grasp_list = rob_arm.naive_grasp_detection(rgb, depth)
    #rob_arm.run_grasp(grasp_list, 1, use_gdn=False)
    rob_arm.pick_and_place(grasp_list, 1, action_matrix,use_gdn=False)
    '''
     
    ### gt close-loop
    '''
    grasp_list = rob_arm.naive_grasp_detection(rgb, depth)
    rob_arm.run_grasp(grasp_list, 1, use_gdn = False)
    print('servoing...')
    rob_arm.gt_peg_in_hole_servo(target_name='hole_keypoint', object_name='peg_keypoint')
    '''

    ### close-loop

    #naive grasp
    '''
    rgb = rob_arm.get_rgb()
    depth = rob_arm.get_depth()
    grasp_list = rob_arm.naive_grasp_detection(rgb, depth)
    rob_arm.run_grasp(grasp_list, 1, use_gdn = False)
    '''

    #gt grasp
    '''
    peg_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name='peg_keypoint_bottom')
    grasp_pose = peg_keypoint_bottom_pose.copy()
    grasp_pose[2,3] += 0.08
    rob_arm.gt_run_grasp(grasp_pose)
    grasp_pose[2, 3] += 0.2
    rob_arm.movement(grasp_pose)
    '''

    # stage 1
    rob_arm.gdn_demo_grasp()

    # stage 2
    '''
    gripper_pose = np.array([[4.54187393e-05, -1.00000000e+00, 8.34465027e-06, -1.5510e-01],
                             [-1.97768211e-04, -8.34465027e-06, -1.00000000e+00, -3.0250e-01],
                             [1.00000000e+00, 4.54187393e-05, -1.97768211e-04, 0.329978],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
                             
    '''


    cam_name = 'vision_sensor'
    rgb = rob_arm.get_rgb(cam_name=cam_name)
    depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=0.5)
    rob_arm.naive_angle_grasp(rgb_img = rgb, depth_img = depth, cam_name=cam_name)


    # stage 3
    print('servoing...')
    bbox = np.array([0, 0, 255, 255])
    err_tolerance = 0.19
    alpha_err = 0.7
    alpha_target = 0.7
    filter_robot_pose = rob_arm.get_object_matrix('UR5_ikTip')
    err_rolling, err_size_rolling = None, err_tolerance * 10
    cnt = 0
    while err_size_rolling > err_tolerance:
        print('========cnt:' + str(cnt) + '=========\n')
        cnt += 1
        robot_pose = rob_arm.get_object_matrix('UR5_ikTip')

        cam_name = 'vision_eye'
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
        camera_keypoint = kp_detection.inference(cv_rgb=rgb, cv_depth=depth_mm, bbox=bbox)

        cam_pos, cam_quat, cam_matrix= rob_arm.get_camera_pos(cam_name=cam_name)
        cam_matrix = np.array(cam_matrix).reshape(3,4)
        camera2world_matrix = cam_matrix
        world_keypoints = np.zeros((4, 3), dtype=np.float64)
        for idx, keypoint in enumerate(camera_keypoint):
            keypoint = np.append(keypoint, 1)
            world_keypoints[idx, :] = np.dot(camera2world_matrix, keypoint)[0:3]
        print('world keypoints', world_keypoints)

        peg_keypoint_top = world_keypoints[0]
        peg_keypoint_bottom = world_keypoints[1]
        #object_keypoint[0] = object_keypoint[0] * -1
        #object_keypoint[1] = object_keypoint[1] * -1
        hole_keypoint_top = world_keypoints[2]
        hole_keypoint_bottom = world_keypoints[3]

        #target_keypoint[0] = target_keypoint[0]* -1
        #target_keypoint[1] = target_keypoint[1] * -1

        err = hole_keypoint_top - peg_keypoint_bottom
        # err*= 0.1
        print('err vector', err)
        if cnt>=100:
            kp_detection.visualize(cv_rgb=rgb, cv_depth=depth_mm, keypoints=camera_keypoint)


        dis = math.sqrt(math.pow(err[0], 2) + math.pow(err[1], 2))
        print('Distance:', dis)
        if dis < 0.01:
            hole_dir = hole_keypoint_bottom - hole_keypoint_top
            peg_dir = peg_keypoint_bottom - peg_keypoint_top
            dot_product = np.dot(peg_dir, hole_dir)
            degree =  math.degrees(math.acos(dot_product/(np.linalg.norm(peg_dir)* np.linalg.norm(hole_dir))))
            print('Degree:', degree)
            if degree > 2:
                cross_product = np.cross(peg_dir, hole_dir)
                cross_product = cross_product / np.linalg.norm(cross_product)
                degree -= 0.5
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

    # insertion
    print('insertion')
    dir = np.array([0,0,-1])
    move_pose = rob_arm.get_object_matrix('UR5_ikTip')
    move_pose[:3, 3] += dir * 0.185
    rob_arm.insertion(move_pose)

    # after insertion
    move_pose[:3, 3] -= dir * 0.185
    rob_arm.movement(move_pose)

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
