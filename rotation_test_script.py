from env.single_robotic_arm import SingleRoboticArm
from scipy.spatial.transform import Rotation as R
import numpy as np
from ruamel import yaml
import os
data_root = '/Users/cmlab/data/pdc/logs_proto'
date = '2021-08-22'
anno_data = 'insertion_xyzrot_eye_' + date + '/processed'
im_data = 'insertion_xyzrot_eye_' + date + '/processed/images'
anno_data_path = os.path.join(data_root, anno_data)

def main():
    '''
    with open(os.path.join(anno_data_path, 'peg_in_hole.yaml'), 'r') as f_r:
        data = yaml.load(f_r)
    print('num of path: ' + str(len(data)))
    for i in range(len(data)):
        for j in range(len(data[i])):
            delta_rotation_matrix = data[i][j]['delta_rotation_matrix']
            r = R.from_matrix(delta_rotation_matrix)
            euler_angle_zyx = r.as_euler('zyx', degrees=True).tolist()
            data[i][j]['euler_angle_zyx'] = euler_angle_zyx
    with open(os.path.join(anno_data_path, 'peg_in_hole_test.yaml'), 'w') as f_w:
        yaml.dump(data, f_w, Dumper=yaml.RoundTripDumper)
    '''

    '''
    source_rot = gripper_pose[:3, :3]
    target_rot = np.array([[ 1.29997730e-04, -9.99995530e-01,  3.07047926e-03],
                   [-1.00000012e+00, -1.28626823e-04,  5.98488143e-04],
                   [-5.98091050e-04, -3.07055679e-03, -9.99995470e-01]])
    try:
        source_rot_inv = np.linalg.inv(source_rot)
    except np.linalg.LinAlgError:
        print('Matrix is singular!')
        assert False
    delta_rot = np.dot(target_rot, source_rot_inv)
    r = R.from_matrix(delta_rot)
    r_eular = r.as_euler('zyx', degrees=True)
    print(r_eular)
    r = R.from_euler('zyx', r_eular, degrees=True)
    delta_rot = r.as_matrix()
    '''
    rob_arm = SingleRoboticArm()
    gripper_pose_1 = rob_arm.get_object_matrix(obj_name='UR5_ikTip')

    source_rot = gripper_pose_1[:3, :3].copy()
    r = R.from_euler('zyx', [45, 0, 0], degrees=True)
    delta_rot = r.as_matrix()
    rot_matrix = np.dot(gripper_pose_1[:3, :3], delta_rot)
    gripper_pose_1[:3, :3] = rot_matrix
    rob_arm.movement(gripper_pose_1)

    gripper_pose_2 = rob_arm.get_object_matrix(obj_name='UR5_ikTip')
    target_rot = gripper_pose_2[:3, :3].copy()
    source_rot_t = np.transpose(source_rot)
    delta_rot = np.dot(source_rot_t, target_rot)
    #delta_rot = np.dot(target_rot_t, source_rot)
    r = R.from_matrix(delta_rot)
    r_euler = r.as_euler('zyx', degrees=True)

    print('delta_rot',delta_rot)
    print('source_rot', source_rot)
    print('target_rot', target_rot)
    print(r_euler)
    rob_arm.finish()

    #test_r = R.from_euler('zyx', [150,50,150], degrees=True)
    #print(test_r.as_matrix())
    #print(test_r.as_euler('zyx', degrees=True))

if __name__ == '__main__':
    main()