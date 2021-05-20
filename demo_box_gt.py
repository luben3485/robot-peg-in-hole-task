from env.single_robotic_arm import SingleRoboticArm
from keypoint_detection import KeypointDetection
import numpy as np
import cv2
import math


def main():
    rob_arm = SingleRoboticArm()
    init_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
    # gt grasp
    peg_keypoint_top_pose = rob_arm.get_object_matrix(obj_name='peg_keypoint_top_large0')
    grasp_pose = peg_keypoint_top_pose.copy()
    grasp_pose[0, 3] += 0.0095
    grasp_pose[2, 3] -= 0.015
    rob_arm.gt_run_grasp(grasp_pose)

    grasp_pose[2, 3] += 0.24
    print(grasp_pose)
    rob_arm.movement(grasp_pose)

    hole_keypoint_top_pose = rob_arm.get_object_matrix(obj_name='hole_keypoint_top_large0')
    grasp_pose[:2, 3] = hole_keypoint_top_pose[:2, 3]
    grasp_pose[0, 3] += 0.0095
    rob_arm.movement(grasp_pose)

    # insertion
    grasp_pose[2, 3] -= 0.2
    rob_arm.movement(grasp_pose)
    rob_arm.open_gripper(True)

    rob_arm.movement(init_pose)
    rob_arm.finish()



if __name__ == '__main__':
    main()