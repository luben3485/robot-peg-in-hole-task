from single_robotic_arm import SingleRoboticArm


def main():
    rob_arm = SingleRoboticArm()    
    
    extrinsic = rob_arm.get_camera_matrix()
    print('camera matrix',extrinsic)
    intrinsic = rob_arm.get_intrinsic_matrix()
    #print(intrinsic)
    rgb = rob_arm.get_rgb()
    #print(rgb.shape)
    depth = rob_arm.get_depth()
    #print(depth.shape)
    masks = rob_arm.get_mask()
    
    for mask in masks:
        rob_arm.visualize_image(mask, depth, rgb)
        grasp_list = rob_arm.get_grasping_candidate(depth, mask, 90, 90)
        print(grasp_list)
        rob_arm.run_grasp(grasp_list, 1)
    rob_arm.finish()



if __name__ == '__main__':
    main()
