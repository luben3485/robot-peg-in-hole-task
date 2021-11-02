from scipy.spatial.transform import Rotation as R
import numpy as np
from ruamel import yaml
import os
data_root = '/tmp2/r09944001/data/pdc/logs_proto'
date = '2021-08-22'
data_folder_prefix = 'insertion_xyzrot_eye_toy_'
anno_data = data_folder_prefix + date + '/processed'
im_data = data_folder_prefix  + date + '/processed/images'
anno_data_path = os.path.join(data_root, anno_data)

def main():
    with open(os.path.join(anno_data_path, 'peg_in_hole.yaml'), 'r') as f_r:
        data = yaml.load(f_r)
    print('num of path: ' + str(len(data)))
    for track_key in data.keys():
        for image_key in data[track_key].keys():
            print(track_key,image_key)
            delta_rotation_matrix = data[track_key][image_key]['delta_rotation_matrix']
            r = R.from_matrix(delta_rotation_matrix)
            euler_angle_zyx = r.as_euler('zyx', degrees=True).tolist()
            data[track_key][image_key]['euler_angle_zyx'] = euler_angle_zyx
    with open(os.path.join(anno_data_path, 'peg_in_hole_test.yaml'), 'w') as f_w:
        yaml.dump(data, f_w, Dumper=yaml.RoundTripDumper)
    '''
    rob_arm = SingleRoboticArm()
    gripper_pose = rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
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

    rot_matrix = np.dot(delta_rot, gripper_pose[:3, :3])
    gripper_pose[:3, :3] = rot_matrix
    rob_arm.movement(gripper_pose)
    rob_arm.finish()

    test_r = R.from_euler('zyx', [150,50,150], degrees=True)
    print(test_r.as_matrix())
    print(test_r.as_euler('zyx', degrees=True))
    '''
if __name__ == '__main__':
    main()
