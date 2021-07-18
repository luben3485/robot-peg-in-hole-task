from env.single_robotic_arm import SingleRoboticArm

def main():
    rob_arm = SingleRoboticArm()
    # movement need IK
    gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
    gripper_pose[0,3] += 0.2
    rob_arm.movement(gripper_pose)
    gripper_pose[0,3] -= 0.2
    rob_arm.movement(gripper_pose)

    # Using OMPL plugin to compute path, you need to disable IK.
    joint_config1 = [-1.0110, -1.1291, 0.7916, 0.3375, 2.1395, 0.7224]
    joint_config2 = [1.5710, -1.5702, 1.5702, -1.5693, -1.5706, 0.0002]
    rob_arm.compute_path_from_joint_space(joint_config1)
    rob_arm.compute_path_from_joint_space(joint_config2)

    # movement need IK
    gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
    gripper_pose[0,3] += 0.2
    rob_arm.movement(gripper_pose)
    gripper_pose[0,3] -= 0.2
    rob_arm.movement(gripper_pose)

if __name__ == '__main__':
    main()