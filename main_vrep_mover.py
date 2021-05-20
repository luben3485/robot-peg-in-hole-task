import os
import numpy as np
from env.single_robotic_arm import SingleRoboticArm
from mover.inference import MoveInference

def main():
    rob_arm = SingleRoboticArm()
    checkpoints_file_path = 'mover/ckpnt/checkpoint-best.pth'
    mover = MoveInference(checkpoints_file_path)

    # gt grasp
    peg_keypoint_bottom_pose = rob_arm.get_object_matrix(obj_name='peg_keypoint_bottom')
    grasp_pose = peg_keypoint_bottom_pose.copy()
    grasp_pose[2, 3] += 0.08
    rob_arm.gt_run_grasp(grasp_pose)
    grasp_pose[2, 3] += 0.2
    rob_arm.movement(grasp_pose)

    cnt = 0
    for i in range(10):
        print('========cnt:' + str(cnt) + '=========\n')
        cnt += 1
        cam_name = 'vision_eye'
        rgb = rob_arm.get_rgb(cam_name=cam_name)
        depth = rob_arm.get_depth(cam_name=cam_name, near_plane=0.01, far_plane=1.5)
        depth_mm = (depth * 1000).astype(np.uint16)

        r, t = mover.inference_single_frame(rgb=rgb, depth_mm=depth_mm)
        print(t)
        robot_pose = rob_arm.get_object_matrix('UR5_ikTip')
        rot_matrix = np.dot(r, robot_pose[:3, :3])
        robot_pose[:3, :3] = rot_matrix
        robot_pose[:3, 3] += t
        print(robot_pose)
        rob_arm.movement(robot_pose)
    rob_arm.finish()

if __name__ == '__main__' :
    main()