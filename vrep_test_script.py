from env.single_robotic_arm import SingleRoboticArm
import numpy as np

def main():
    rob_arm = SingleRoboticArm()
    cam_name = 'vision_fix'
    #extrinsic = rob_arm.get_camera_matrix(cam_name=cam_name)
    #print(extrinsic)
    #intrinsic = rob_arm.get_intrinsic_matrix()
    #print(intrinsic)
    #rgb = rob_arm.get_rgb(cam_name=cam_name)
    #print(rgb.shape)
    #depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=0.5)
    #print(depth.shape)
    #masks = rob_arm.get_mask(cam_name=cam_name)

    '''
    cam_name = 'vision_sensor'
    rgb = rob_arm.get_rgb(cam_name=cam_name)
    depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=0.5)
    rob_arm.naive_angle_grasp(rgb, depth, cam_name=cam_name)

    gripper_pose = np.array([[ 4.54187393e-05, -1.00000000e+00,  8.34465027e-06,  -1.5510e-01],
 [-1.97768211e-04, -8.34465027e-06, -1.00000000e+00, -3.0250e-01],
 [ 1.00000000e+00,  4.54187393e-05, -1.97768211e-04,  0.329978],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    rob_arm.movement(gripper_pose)
    '''

    '''
    init_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')

    cam_name = 'vision_sensor_gdn'
    rgb = rob_arm.get_rgb(cam_name=cam_name)
    depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=1.5)
    #masks = rob_arm.get_mask(cam_name=cam_name)
    extrinsic = rob_arm.get_camera_matrix(cam_name=cam_name)
    #for mask in masks:
    rob_arm.visualize_image(None, depth, rgb)
    grasp_list = rob_arm.get_grasping_candidate(depth, extrinsic, 90, 90)
    grasp_matrix = grasp_list[0]
    #print(grasp_list)
    #rob_arm.run_grasp(grasp_list, 1, use_gdn = True)
    rob_arm.run_single_grasp(grasp_matrix, use_gdn=True)

    step_2_pose = np.array([[ 7.14063644e-05, -1.00000000e+00,  1.22785568e-05,  1.69920236e-01],
    [-2.97635794e-04, -1.23977661e-05, -1.00000000e+00, -4.69989717e-01],
    [ 1.00000000e+00,  7.14957714e-05, -2.97665596e-04,  2.22262397e-01],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    step_2_rot = step_2_pose[:3, :3]
    grasp_matrix[:3, :3] = step_2_rot
    grasp_matrix[2, 3] += 0.2
    rob_arm.movement(grasp_matrix)

    rob_arm.movement(step_2_pose)
    rob_arm.open_gripper(True)
    '''
    rob_arm.finish()




if __name__ == '__main__':
    main()
