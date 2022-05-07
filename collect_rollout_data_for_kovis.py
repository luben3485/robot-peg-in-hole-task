import os
import cv2
import time
import yaml
import math
import random
import argparse
import copy
import numpy as np

from env.single_robotic_arm import SingleRoboticArm

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--hole', type=str, default='square_7x12x12', help='specify object hole')
    parser.add_argument('--iter', type=int, default=3, help='nb of input data')
    parser.add_argument('--date', type=str, default='2022-05-02-notilt-test', help='date')
    parser.add_argument('--data_root', type=str, default='/home/luben/data/kovis', help='data root path')
    parser.add_argument('--data_type', type=str, default='train', help='data type')

    return parser.parse_args()
args = parse_args()
hole_setting = {
    'square_7x10x10': ['square_7x10x10', 'hole_keypoint_top0', 'hole_keypoint_bottom0', 'hole_keypoint_obj_bottom0'],
    'square_7x11x11': ['square_7x11x11', 'hole_keypoint_top1', 'hole_keypoint_bottom1', 'hole_keypoint_obj_bottom1'],
    'square_7x12x12': ['square_7x12x12', 'hole_keypoint_top2', 'hole_keypoint_bottom2', 'hole_keypoint_obj_bottom2'],
    'square_7x13x13': ['square_7x13x13', 'hole_keypoint_top3', 'hole_keypoint_bottom3', 'hole_keypoint_obj_bottom3'],
    'square_7x14x14': ['square_7x14x14', 'hole_keypoint_top4', 'hole_keypoint_bottom4', 'hole_keypoint_obj_bottom4'],
    'rectangle_7x8x11': ['rectangle_7x8x11', 'hole_keypoint_top5', 'hole_keypoint_bottom5',
                         'hole_keypoint_obj_bottom5'],
    'rectangle_7x9x12': ['rectangle_7x9x12', 'hole_keypoint_top6', 'hole_keypoint_bottom6',
                         'hole_keypoint_obj_bottom6'],
    'rectangle_7x10x13': ['rectangle_7x10x13', 'hole_keypoint_top7', 'hole_keypoint_bottom7',
                          'hole_keypoint_obj_bottom7'],
    'rectangle_7x11x14': ['rectangle_7x11x14', 'hole_keypoint_top8', 'hole_keypoint_bottom8',
                          'hole_keypoint_obj_bottom8'],
    'rectangle_7x12x15': ['rectangle_7x12x15', 'hole_keypoint_top9', 'hole_keypoint_bottom9',
                          'hole_keypoint_obj_bottom9'],
    'circle_7x10': ['circle_7x10', 'hole_keypoint_top10', 'hole_keypoint_bottom10', 'hole_keypoint_obj_bottom10'],
    'circle_7x11': ['circle_7x11', 'hole_keypoint_top11', 'hole_keypoint_bottom11', 'hole_keypoint_obj_bottom11'],
    'circle_7x12': ['circle_7x12', 'hole_keypoint_top12', 'hole_keypoint_bottom12', 'hole_keypoint_obj_bottom12'],
    'circle_7x13': ['circle_7x13', 'hole_keypoint_top13', 'hole_keypoint_bottom13', 'hole_keypoint_obj_bottom13'],
    'circle_7x14': ['circle_7x14', 'hole_keypoint_top14', 'hole_keypoint_bottom14', 'hole_keypoint_obj_bottom14'], }

class CollectInsert(object):
    def __init__(self, ):
        self.sample = 0
        self.data_root = args.data_root
        self.data_type = args.data_type
        self.data_folder = 'insertion_' + args.hole + '_' + args.date
        self.anno_path = os.path.join(self.data_root, self.data_folder, self.data_type, 'gt.yaml')
        # obj
        self.peg_top = 'peg_dummy_top'
        self.peg_bottom = 'peg_dummy_bottom'
        self.peg_name = 'peg_in_arm'
        self.selected_hole = args.hole
        self.hole_name = hole_setting[self.selected_hole][0]
        self.hole_top = hole_setting[self.selected_hole][1]
        self.hole_bottom = hole_setting[self.selected_hole][2]
        self.hole_obj_bottom = hole_setting[self.selected_hole][3]
        # cam
        self.rgbd_cam = ['vision_eye_left', 'vision_eye_right']
        self.seg_peg_cam = ['vision_eye_left_peg', 'vision_eye_right_peg']
        self.seg_hole_cam = ['vision_eye_left_hole', 'vision_eye_right_hole']

        self.rob_arm = SingleRoboticArm()
        self.origin_hole_pose = self.rob_arm.get_object_matrix(self.hole_name)
        self.origin_hole_pos = self.origin_hole_pose[:3, 3]
        self.origin_hole_quat = self.rob_arm.get_object_quat(self.hole_name)
        self.rob_arm.finish()

        self.iter = args.iter
        self.rollout_step = 5
        self.min_speed = 0.001
        self.max_speed = 0.022
        self.start_offset = [0, 0, 0.005]

        if not os.path.exists(os.path.join(self.data_root, self.data_folder, self.data_type)):
            os.makedirs(os.path.join(self.data_root, self.data_folder, self.data_type))
        with open(self.anno_path, 'w') as f:
            yaml.safe_dump([], f)

    def reset(self, ):
        self.rob_arm = SingleRoboticArm()
        # set init pos of hole
        hole_pos = np.array([random.uniform(0.0, 0.2), random.uniform(-0.45, -0.55), 0.035])
        self.rob_arm.set_object_position(self.hole_name, hole_pos)
        self.rob_arm.set_object_quat(self.hole_name, self.origin_hole_quat)
        #self.random_tilt([self.hole_name], 0, 50)
        # set start pose
        start_pose = self.rob_arm.get_object_matrix(obj_name=self.hole_top)
        start_pose[:3, 3] += self.start_offset
        self.rob_arm.movement(start_pose)

    def rollout(self, ):
        num_roll = 0
        # set motion vector & speed
        speed = np.random.uniform(self.min_speed, self.max_speed)
        vec = np.random.rand(3) - 0.5
        vec[2]*=4
        vec = vec / np.linalg.norm(vec)
        for step in range(self.rollout_step):
            peg_keypoint_bottom_pose = self.rob_arm.get_object_matrix(obj_name=self.peg_bottom)
            hole_keypoint_top_pose = self.rob_arm.get_object_matrix(obj_name=self.hole_top)
            #if np.linalg.norm(peg_keypoint_bottom_pose[:3, 3] - hole_keypoint_top_pose[:3, 3]) < (np.linalg.norm(self.start_offset) / 3):
            if (peg_keypoint_bottom_pose[2, 3] - hole_keypoint_top_pose[2, 3]) < (self.start_offset[2] / 2):
                print('too close...')
                break
            self.save_images(self.sample, step)
            robot_pose = self.rob_arm.get_object_matrix(obj_name='UR5_ikTarget')
            robot_pose[:3, 3] += vec * speed
            self.rob_arm.movement(robot_pose)
            num_roll = step
        if num_roll >= 2:
            print('save!')
            self.save_label((vec * -1.).tolist() + [0, 0, 0] + [speed])
            self.sample += 1
        self.rob_arm.finish()

    def run(self, ):
        while self.sample < self.iter:
            print('=' * 8 + 'Iteration ' + str(self.sample) + '=' * 8)
            self.reset()
            self.rollout()

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

