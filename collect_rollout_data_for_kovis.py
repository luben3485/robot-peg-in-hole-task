import os
import cv2
import time
import yaml
import math
import random
import argparse
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from transforms3d.quaternions import mat2quat, quat2axangle, quat2mat, qmult
from env.single_robotic_arm import SingleRoboticArm
from config.hole_setting import hole_setting

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=3, help='nb of input data')
    parser.add_argument('--date', type=str, default='2022-06-16-notilt-test', help='date')
    parser.add_argument('--data_root', type=str, default='/home/luben/data/kovis', help='data root path')
    parser.add_argument('--data_type', type=str, default='train', help='data type')
    parser.add_argument('--tilt', action='store_true', default=False, help=' ')
    parser.add_argument('--yaw', action='store_true', default=False, help=' ')

    return parser.parse_args()
args = parse_args()

class CollectInsert(object):
    def __init__(self, ):
        self.sample = 0
        self.data_root = args.data_root
        self.data_type = args.data_type
        self.data_folder = 'insertion_' + args.date
        self.anno_path = os.path.join(self.data_root, self.data_folder, self.data_type, 'gt.yaml')
        # obj
        self.peg_top = 'peg_dummy_top'
        self.peg_bottom = 'peg_dummy_bottom'
        self.peg_name = 'peg_in_arm'
        ### for round hole
        #self.selected_hole_list = ['square_7x12x12', 'square_7x13x13', 'rectangle_7x9x12', 'rectangle_7x10x13']
        ### for square hole
        self.selected_hole_list = ['square_7x12x12_squarehole', 'square_7x13x13_squarehole', 'rectangle_7x9x12_squarehole', 'rectangle_7x10x13_squarehole']
        # cam
        self.rgbd_cam = ['vision_eye_left', 'vision_eye_right']
        self.seg_peg_cam = ['vision_eye_left_peg', 'vision_eye_right_peg']
        self.seg_hole_cam = ['vision_eye_left_hole', 'vision_eye_right_hole']

        self.iter = args.iter
        self.rollout_step = 5
        self.min_speed = 0.001
        self.max_speed = 0.022
        self.start_offset = [0, 0, 0.02]
        self.tilt = args.tilt
        self.yaw = args.yaw

        if not os.path.exists(os.path.join(self.data_root, self.data_folder, self.data_type)):
            os.makedirs(os.path.join(self.data_root, self.data_folder, self.data_type))
        with open(self.anno_path, 'w') as f:
            yaml.safe_dump([], f)

    def reset(self, selected_hole):
        self.rob_arm = SingleRoboticArm()
        # set init pos of hole
        hole_name = hole_setting[selected_hole][0]
        hole_top = hole_setting[selected_hole][1]
        hole_bottom = hole_setting[selected_hole][2]

        hole_pos = np.array([random.uniform(0.0, 0.2), random.uniform(-0.45, -0.55), 0.035])
        self.rob_arm.set_object_position(hole_name, hole_pos)
        self.rob_arm.set_object_quat(hole_name, self.origin_hole_quat)
        if self.yaw:
            self.random_yaw([hole_name], degree=45)
        if self.tilt:
            self.random_tilt([hole_name], 0, 50)
        # set start pose
        start_pose = self.rob_arm.get_object_matrix(obj_name=hole_top)
        start_pose[:3, 3] = start_pose[:3, 3] + start_pose[:3, 0] * self.start_offset[2]
        self.rob_arm.movement(start_pose)

    def rollout(self, selected_hole):
        hole_name = hole_setting[selected_hole][0]
        hole_top = hole_setting[selected_hole][1]
        hole_bottom = hole_setting[selected_hole][2]
        num_roll = 0
        # set motion vector & speed
        speed = np.random.uniform(self.min_speed, self.max_speed)
        vec = np.random.rand(3) - 0.5
        vec = vec / np.linalg.norm(vec)
        if self.tilt or self.yaw:
            # for move
            source_rot = self.rob_arm.get_object_matrix(obj_name='UR5_ikTarget')[:3, :3]
            source_rot_t = np.transpose(source_rot)
            target_rot = self.init_gripper_pose[:3, :3]
            delta_rotation = np.dot(source_rot_t, target_rot)
            r = R.from_matrix(delta_rotation)
            r_rotvec = r.as_rotvec()
            r = R.from_rotvec(r_rotvec / self.rollout_step)
            r_mat = r.as_matrix()
            # for record
            record_r_mat = np.transpose(r_mat)
            record_r = R.from_matrix(record_r_mat)
            r_euler = record_r.as_euler('zyx', degrees=True)

        for step in range(self.rollout_step):
            peg_keypoint_bottom_pose = self.rob_arm.get_object_matrix(obj_name=self.peg_bottom)
            hole_keypoint_top_pose = self.rob_arm.get_object_matrix(obj_name=hole_top)
            if (peg_keypoint_bottom_pose[2, 3] - hole_keypoint_top_pose[2, 3]) < (self.start_offset[2] / 2):
                print('too close...')
                break
            self.save_images(self.sample, step)
            robot_pose = self.rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            #robot_pose[:3, 3] += vec * speed
            print(robot_pose[:3, :3])
            robot_pose[:3, 3] = robot_pose[:3, 3] + (robot_pose[:3, 0] * vec[0] + robot_pose[:3, 1] * vec[1] + robot_pose[:3, 2] * vec[2]) * speed

            if self.tilt or self.yaw:
                rot = np.dot(robot_pose[:3, :3], r_mat)
                robot_pose[:3, :3] = rot
            self.rob_arm.movement(robot_pose)
            num_roll = step
        if num_roll >= 2:
            print('save!')
            if self.tilt or self.yaw:
                self.save_label((vec * -1.).tolist() + r_euler.tolist() + [speed])
            else:
                self.save_label((vec * -1.).tolist() + [0, 0, 0] + [speed])
            self.sample += 1
            self.cnt += 1
        self.rob_arm.finish()

    def run(self, ):
        for selected_hole in self.selected_hole_list:
            self.cnt = 0
            self.rob_arm = SingleRoboticArm()
            self.init_gripper_pose = self.rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            self.origin_hole_pose = self.rob_arm.get_object_matrix(hole_setting[selected_hole][0])
            self.origin_hole_pos = self.origin_hole_pose[:3, 3]
            self.origin_hole_quat = self.rob_arm.get_object_quat(hole_setting[selected_hole][0])
            self.rob_arm.finish()
            while self.cnt < self.iter:
                print('=' * 8 + 'Iteration ' + str(self.sample) + '=' * 8)
                self.reset(selected_hole)
                self.rollout(selected_hole)


    def random_tilt(self, obj_name_list, min_tilt_degree, max_tilt_degree):
        while True:
            u = random.uniform(0, 1)
            v = random.uniform(0, 1)
            theta = 2 * math.pi * u
            phi = math.acos(2 * v - 1)
            x = math.sin(theta) * math.sin(phi)
            y = math.cos(theta) * math.sin(phi)
            z = math.cos(phi)
            dst_hole_dir = np.array([x, y, z])  # world coordinate
            src_hole_dir = np.array([0, 0, 1])  # world coordinate

            cross_product = np.cross(src_hole_dir, dst_hole_dir)
            if cross_product.nonzero()[0].size == 0:  # to check if it is zero vector
                rot_dir = np.array([0, 0, 1])
            else:
                rot_dir = cross_product / np.linalg.norm(cross_product)
            dot_product = np.dot(src_hole_dir, dst_hole_dir)
            tilt_degree = math.degrees(
                math.acos(dot_product / (np.linalg.norm(src_hole_dir) * np.linalg.norm(dst_hole_dir))))
            if abs(tilt_degree) <= max_tilt_degree and abs(tilt_degree) >= min_tilt_degree:
                break
        # print('rot_dir:', rot_dir)
        # print('tilt degree:', tilt_degree)
        w = math.cos(math.radians(tilt_degree / 2))
        x = math.sin(math.radians(tilt_degree / 2)) * rot_dir[0]
        y = math.sin(math.radians(tilt_degree / 2)) * rot_dir[1]
        z = math.sin(math.radians(tilt_degree / 2)) * rot_dir[2]
        rot_quat = [w, x, y, z]
        for obj_name in obj_name_list:
            obj_quat = self.rob_arm.get_object_quat(obj_name)  # [x,y,z,w]
            obj_quat = [obj_quat[3], obj_quat[0], obj_quat[1], obj_quat[2]]  # change to [w,x,y,z]
            obj_quat = qmult(rot_quat, obj_quat)  # [w,x,y,z]
            obj_quat = [obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]  # change to [x,y,z,w]
            self.rob_arm.set_object_quat(obj_name, obj_quat)

        return rot_dir, tilt_degree

    def random_yaw(self, obj_name_list, degree=45):
        for obj_name in obj_name_list:
            yaw_degree = random.uniform(-math.radians(degree), math.radians(degree))
            rot_dir = self.rob_arm.get_object_matrix(obj_name)[:3, 0]
            if obj_name in ['pentagon_7x7_squarehole', 'pentagon_7x9_squarehole', 'rectangle_7x9x12_squarehole', 'rectangle_7x10x13_squarehole']:
                rot_dir = self.rob_arm.get_object_matrix(obj_name)[:3, 1]
            w = math.cos(yaw_degree / 2)
            x = math.sin(yaw_degree / 2) * rot_dir[0]
            y = math.sin(yaw_degree / 2) * rot_dir[1]
            z = math.sin(yaw_degree / 2) * rot_dir[2]
            rot_quat = [w, x, y, z]

            obj_quat = self.rob_arm.get_object_quat(obj_name)  # [x,y,z,w]
            obj_quat = [obj_quat[3], obj_quat[0], obj_quat[1], obj_quat[2]]  # change to [w,x,y,z]
            obj_quat = qmult(rot_quat, obj_quat)  # [w,x,y,z]
            obj_quat = [obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]  # change to [x,y,z,w]
            self.rob_arm.set_object_quat(obj_name, obj_quat)

    def save_images(self, sample, step):
        seg_folder_path = [os.path.join(self.data_root, self.data_folder, self.data_type, 'left/segme'),
                           os.path.join(self.data_root, self.data_folder, self.data_type, 'right/segme')]
        color_folder_path = [os.path.join(self.data_root, self.data_folder, self.data_type, 'left/color'),
                             os.path.join(self.data_root, self.data_folder, self.data_type, 'right/color')]
        depth_folder_path = [os.path.join(self.data_root, self.data_folder, self.data_type, 'left/depth'),
                             os.path.join(self.data_root, self.data_folder, self.data_type, 'right/depth')]
        # create folder
        for i in range(2):
            if not os.path.exists(seg_folder_path[i]):
                os.makedirs(seg_folder_path[i])
            if not os.path.exists(color_folder_path[i]):
                os.makedirs(color_folder_path[i])
            if not os.path.exists(depth_folder_path[i]):
                os.makedirs(depth_folder_path[i])
        length = 2
        assert len(self.rgbd_cam) == length and len(self.seg_peg_cam) == length and len(self.seg_hole_cam) == length
        for idx in range(length):
            # Get rgb image
            rgb = self.rob_arm.get_rgb(cam_name=self.rgbd_cam[idx])

            # Get depth image
            depth = self.rob_arm.get_depth(cam_name=self.rgbd_cam[idx], near_plane=0.01, far_plane=1.5)
            depth_mm = (depth * 1000).astype(np.uint8)

            # Get seg_peg rgb image
            seg_peg_rgb = self.rob_arm.get_rgb(cam_name=self.seg_peg_cam[idx])
            seg_peg_rgb = cv2.cvtColor(seg_peg_rgb, cv2.COLOR_RGB2GRAY)

            # Get seg_hole rgb image
            seg_hole_rgb = self.rob_arm.get_rgb(cam_name=self.seg_hole_cam[idx])
            seg_hole_rgb = cv2.cvtColor(seg_hole_rgb, cv2.COLOR_RGB2GRAY)

            # concat seg_peg & seg_hole
            seg = np.zeros((128, 128))
            y_idx, x_idx = np.nonzero(seg_hole_rgb)
            seg[y_idx, x_idx] = 2
            y_idx, x_idx = np.nonzero(seg_peg_rgb)
            seg[y_idx, x_idx] = 1

            # set bg to 255 for depth image
            y_idx, x_idx = np.where(seg == 0)
            depth_mm[y_idx, x_idx] = 255

            # save images
            file_name = str(sample).zfill(5) + '_' + str(step).zfill(2) + '.png'
            cv2.imwrite(os.path.join(color_folder_path[idx], file_name), rgb)
            cv2.imwrite(os.path.join(depth_folder_path[idx], file_name), depth_mm)
            cv2.imwrite(os.path.join(seg_folder_path[idx], file_name), seg)

    def save_label(self, label):

        if os.path.exists(self.anno_path):
            with open(self.anno_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            data = []

        data.append(label)
        with open(self.anno_path, 'w') as f:
            yaml.safe_dump(data, f)

if __name__ == '__main__':
    start_time = time.time()
    CollectInsert().run()
    end_time = time.time()
    print('Time elasped:{:.02f}'.format((end_time - start_time)/3600))

