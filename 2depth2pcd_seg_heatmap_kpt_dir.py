import numpy as np
import cv2
import os
import open3d as o3d
import glob
import time
import yaml
import math
import tqdm
import copy
import argparse
from PointWOLF import PointWOLF

def depth_2_pcd(depth, factor, K):
    xmap = np.array([[j for i in range(depth.shape[0])] for j in range(depth.shape[1])])
    ymap = np.array([[i for i in range(depth.shape[0])] for j in range(depth.shape[1])])

    if len(depth.shape) > 2:
        depth = depth[:, :, 0]
    mask_depth = depth > 1e-6
    choose = mask_depth.flatten().nonzero()[0].astype(np.uint32)
    if len(choose) < 1:
        return None

    depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = depth_masked / factor
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    pcd = np.concatenate((pt0, pt1, pt2), axis=1)

    return pcd, choose

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='2022-02-26-test/coarse_insertion_square_2022-02-26-test', help='folder path')
    parser.add_argument('--offset', type=str, default='kpts')
    parser.add_argument('--crop_pcd', action='store_true', default=False, help='crop point cloud')
    parser.add_argument('--visualize', action='store_true', default=False, help='output .ply')
    return parser.parse_args()

def main(args):
    data_root = os.path.join('/home/luben/data/pdc/logs_proto', args.folder_path, 'processed')
    image_folder_path = os.path.join(data_root, 'images')
    pcd_folder_path = os.path.join(data_root, 'pcd')
    if args.offset == 'kpts':
        pcd_seg_heatmap_kpt_folder_path = os.path.join(data_root, 'pcd_seg_heatmap_3kpt')
    elif args.offset == 'kpt_dir':
        pcd_seg_heatmap_kpt_folder_path = os.path.join(data_root, 'pcd_seg_heatmap_kpt_dir')
    if args.visualize == True:
        visualize_folder_path = os.path.join(data_root, 'visualize')
    # create folder
    cwd = os.getcwd()
    os.chdir(data_root)
    if not os.path.exists('pcd'):
        os.makedirs('pcd')
    if not os.path.exists('pcd_seg_heatmap_3kpt'):
        os.makedirs('pcd_seg_heatmap_3kpt')
    if not os.path.exists('pcd_seg_heatmap_kpt_dir'):
        os.makedirs('pcd_seg_heatmap_kpt_dir')
    if (os.path.exists('visualize') == False) and (args.visualize == True):
        os.makedirs('visualize')
    os.chdir(cwd)

    focal_length = 309.019
    principal = 128
    factor = 1
    intrinsic_matrix = np.array([[focal_length, 0, principal],
                                 [0, focal_length, principal],
                                 [0, 0, 1]], dtype=np.float64)


    with open(os.path.join(data_root, 'peg_in_hole.yaml'), 'r') as f_r:
        data = yaml.load(f_r)
    #for key, value in data.items():
    for key, value in tqdm.tqdm(list(data.items())):
        depth_image_filename_list = data[key]['depth_image_filename'][:1]
        camera2world_list = data[key]['camera_matrix'][:1]
        gripper_pos = np.array(data[key]['gripper_pose'])[:3, 3]
        hole_top_pose = np.array(data[key]['hole_keypoint_obj_top_pose_in_world'])
        xyz_in_world_list = []
        for idx, depth_image_filename in enumerate(depth_image_filename_list):
            depth_img = cv2.imread(os.path.join(image_folder_path, depth_image_filename), cv2.IMREAD_ANYDEPTH)
            depth_img = depth_img / 1000 # unit: mm to m
            xyz, choose = depth_2_pcd(depth_img, factor, intrinsic_matrix)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            down_pcd = pcd.uniform_down_sample(every_k_points=8)
            down_xyz = np.asarray(down_pcd.points)
            down_xyz_in_camera = down_xyz[:8000, :]

            down_xyz_in_world = []
            for xyz in down_xyz_in_camera:
                camera2world = np.array(camera2world_list[idx])
                xyz = np.append(xyz, [1], axis=0).reshape(4, 1)
                xyz_world = camera2world.dot(xyz)
                xyz_world = xyz_world[:3] * 1000
                down_xyz_in_world.append(xyz_world)
            xyz_in_world_list.append(down_xyz_in_world)
        concat_xyz_in_world = np.array(xyz_in_world_list)
        concat_xyz_in_world = concat_xyz_in_world.reshape(-1,3)

        if args.crop_pcd == True:
            crop_concat_xyz_in_world = []
            bound = 0.05
            for xyz in concat_xyz_in_world:
                x = xyz[0] / 1000  # unit:m
                y = xyz[1] / 1000  # unit:m
                z = xyz[2] / 1000  # unit:m
                if x >= gripper_pos[0] - bound and x <= gripper_pos[0] + bound and \
                    y >= gripper_pos[1] - bound and y <= gripper_pos[1] + bound and \
                    z >= gripper_pos[2] - bound and z <= gripper_pos[2] + bound:
                    crop_concat_xyz_in_world.append(xyz)
            if len(crop_concat_xyz_in_world) == 0:
                print('Error',key)
                assert False
            concat_xyz_in_world = np.array(crop_concat_xyz_in_world)
            if concat_xyz_in_world.shape[0] >= 2400:
                concat_xyz_in_world = concat_xyz_in_world[:2400, :]
                concat_xyz_in_world = concat_xyz_in_world.reshape(-1, 3)
            else:
                data.pop(key, None)
                print('pop key: ', key)
                continue

        hole_keypoint_top_pose_in_world = np.array(data[key]['hole_keypoint_obj_top_pose_in_world'])
        hole_keypoint_top_pos_in_world = hole_keypoint_top_pose_in_world[:3, 3]
        hole_keypoint_bottom_pose_in_world = np.array(data[key]['hole_keypoint_obj_bottom_pose_in_world'])
        hole_keypoint_bottom_pos_in_world = hole_keypoint_bottom_pose_in_world[:3, 3]

        # for heatmap & segmentation
        x_normal_vector = hole_keypoint_top_pose_in_world[:3, 0]
        x_1 = np.dot(x_normal_vector, hole_keypoint_top_pos_in_world + [0., 0., 0.003])
        x_2 = np.dot(x_normal_vector, hole_keypoint_bottom_pos_in_world - [0., 0., 0.002])
        x_value = np.sort([x_1,x_2])
        y_normal_vector = hole_keypoint_top_pose_in_world[:3, 1]
        y_1 = np.dot(y_normal_vector, hole_keypoint_top_pos_in_world + hole_keypoint_top_pose_in_world[:3,1] * 0.067)
        y_2 = np.dot(y_normal_vector, hole_keypoint_top_pos_in_world - hole_keypoint_top_pose_in_world[:3,1] * 0.067)
        y_value = np.sort([y_1, y_2])
        z_normal_vector = hole_keypoint_top_pose_in_world[:3, 2]
        z_1 = np.dot(z_normal_vector, hole_keypoint_top_pos_in_world + hole_keypoint_top_pose_in_world[:3,2] * 0.067)
        z_2 = np.dot(z_normal_vector, hole_keypoint_top_pos_in_world - hole_keypoint_top_pose_in_world[:3,2] * 0.067)
        z_value = np.sort([z_1, z_2])

        hole_seg_xyz = []
        seg_label = []
        seg_color = []
        heatmap_label = []
        for xyz in concat_xyz_in_world:
            x = np.dot(x_normal_vector, xyz / 1000)
            y = np.dot(y_normal_vector, xyz / 1000)
            z = np.dot(z_normal_vector, xyz / 1000)

            if x >= x_value[0] and x <= x_value[1] and y >= y_value[0] and y <= y_value[1] and z >= z_value[0] and z <= z_value[1]:
                hole_seg_xyz.append(xyz)
                # segmentation label
                seg_label.append(1)
                seg_color.append([1, 0, 0])
            else:
                seg_label.append(0)
                seg_color.append([0, 0, 1])

            # heatmap label
            x_vec = xyz[0] / 1000 - hole_keypoint_top_pos_in_world[0]
            y_vec = xyz[1] / 1000 - hole_keypoint_top_pos_in_world[1]
            z_vec = xyz[2] / 1000 - hole_keypoint_top_pos_in_world[2]
            vec = np.array([x_vec, y_vec, z_vec])
            x_scalar = np.dot(vec, x_normal_vector) / np.linalg.norm(x_normal_vector)
            y_scalar = np.dot(vec, y_normal_vector) / np.linalg.norm(y_normal_vector)
            z_scalar = np.dot(vec, z_normal_vector) / np.linalg.norm(z_normal_vector)
            sigma = 0.05 # smaller 0.025
            heatmap_value = math.exp(-(y_scalar ** 2 + z_scalar ** 2 + x_scalar ** 2) / (2.0 * sigma ** 2))  # / (2 * math.pi * sigma**3 * math.sqrt(2*math.pi))
            heatmap_label.append(heatmap_value)
        for idx, seg in enumerate(seg_label):
            if seg_label[idx] == 0:
                heatmap_label[idx] = 0

        # normalize pcd data(unit:mm)
        concat_xyz_in_world = np.array(concat_xyz_in_world)
        normal_xyz_in_world = copy.deepcopy(concat_xyz_in_world)
        centroid = np.mean(normal_xyz_in_world, axis=0)
        normal_xyz_in_world = normal_xyz_in_world - centroid
        m = np.max(np.sqrt(np.sum(normal_xyz_in_world ** 2, axis=1)))
        normal_xyz_in_world = normal_xyz_in_world / m
        if args.offset == 'kpts':
            # determine x-axis & y-axis keypoint
            hole_x_vec = hole_keypoint_top_pose_in_world[:3, 0]
            hole_y_vec = hole_keypoint_top_pose_in_world[:3, 1]
            hole_keypoint_top_x_pos_in_world = hole_keypoint_top_pos_in_world - hole_x_vec * 0.025
            hole_keypoint_top_y_pos_in_world = hole_keypoint_top_pos_in_world - hole_y_vec * 0.025
            # normalize hole's top keypoint
            normal_hole_keypoint_top_pos_in_world = (hole_keypoint_top_pos_in_world * 1000 - centroid) / m
            normal_hole_keypoint_top_x_pos_in_world = (hole_keypoint_top_x_pos_in_world * 1000 - centroid) / m
            normal_hole_keypoint_top_y_pos_in_world = (hole_keypoint_top_y_pos_in_world * 1000 - centroid) / m
            # keypoint offset
            kpt_of_gt = normal_xyz_in_world - normal_hole_keypoint_top_pos_in_world
            kpt_of_x_gt = normal_xyz_in_world - normal_hole_keypoint_top_x_pos_in_world
            kpt_of_y_gt = normal_xyz_in_world - normal_hole_keypoint_top_y_pos_in_world
        elif args.offset == 'kpt_dir':
            # determine x-axis & y-axis vector
            x_vec = hole_keypoint_top_pose_in_world[:3, 0] #unit vec
            y_vec = hole_keypoint_top_pose_in_world[:3, 1] #unit vec
            x_0 = hole_keypoint_top_pos_in_world[0] * 1000
            x_1 = hole_keypoint_top_pos_in_world[1] * 1000
            x_2 = hole_keypoint_top_pos_in_world[2] * 1000
            dir_x = []
            dir_y = []
            for xyz in concat_xyz_in_world:
                t_x =(x_vec[0] * (xyz[0]-x_0) + x_vec[1] * (xyz[1]-x_1) + x_vec[2] * (xyz[2]-x_2)) / (x_vec[0]**2 + x_vec[1]**2 + x_vec[2] **2)
                shortest_kpt_x = np.array([x_0+x_vec[0]*t_x, x_1+x_vec[1]*t_x, x_2+x_vec[2]*t_x])
                dir_x.append(shortest_kpt_x)
                t_y = (y_vec[0] * (xyz[0] - x_0) + y_vec[1] * (xyz[1] - x_1) + y_vec[2] * (xyz[2] - x_2)) / ( y_vec[0] ** 2 + y_vec[1] ** 2 + y_vec[2] ** 2)
                shortest_kpt_y = np.array([x_0 + y_vec[0] * t_y, x_1 + y_vec[1] * t_y, x_2 + y_vec[2] * t_y])
                dir_y.append(shortest_kpt_y)
            dir_x = np.array(dir_x)
            dir_y = np.array(dir_y)

            # keypoint offset
            normal_hole_keypoint_top_pos_in_world = (hole_keypoint_top_pos_in_world * 1000 - centroid) / m
            normal_dir_x = (dir_x - centroid) / m
            normal_dir_y = (dir_y - centroid) / m
            kpt_of_gt = normal_xyz_in_world - normal_hole_keypoint_top_pos_in_world
            kpt_of_x_gt = normal_xyz_in_world - normal_dir_x
            kpt_of_y_gt = normal_xyz_in_world - normal_dir_y
            # test dir_x & dir_y
            if key == 0 and args.visualize == True:
                normal_dir_x = normal_xyz_in_world - kpt_of_x_gt
                normal_dir_y = normal_xyz_in_world - kpt_of_y_gt
                dir_x = normal_dir_x * m + centroid
                dir_y = normal_dir_y * m + centroid
                xyz_in_world = normal_xyz_in_world * m + centroid
                xyz = np.concatenate((xyz_in_world, dir_x, dir_y), axis=0).astype(np.float32)
                point_color = np.repeat([[0.9, 0.9, 0.9]], xyz_in_world.shape[0], axis=0)
                dir_x_color = np.repeat([[1.0, 0.0, 0.0]], dir_x.shape[0], axis=0)
                dir_y_color = np.repeat([[0.0, 0.0, 1.0]], dir_y.shape[0], axis=0)
                color = np.concatenate((point_color, dir_x_color, dir_y_color), axis=0).astype(np.float32)
                hole_dir_pcd = o3d.geometry.PointCloud()
                hole_dir_pcd.points = o3d.utility.Vector3dVector(xyz)
                hole_dir_pcd.colors = o3d.utility.Vector3dVector(color)
                o3d.io.write_point_cloud(os.path.join(data_root, 'hole_dir_pcd_' + str(key) + '.ply'), hole_dir_pcd)
        '''
        # hole pcd visualize(make sure that the scale of pcd is in mm)
        hole_seg_pcd = o3d.geometry.PointCloud()
        hole_seg_pcd.points = o3d.utility.Vector3dVector(hole_seg_xyz)
        o3d.io.write_point_cloud(os.path.join(data_root, 'hole_seg_pcd_' + str(key) + '.ply'), hole_seg_pcd)
        '''
        pcd_filename = str(key).zfill(6) + '_xyz.npy'
        seg_label = np.array(seg_label).reshape(-1,1) # n x 1
        heatmap_label = np.array(heatmap_label).reshape(-1, 1)  # n x 1
        xyz_seg = np.concatenate((normal_xyz_in_world, seg_label), axis=1).astype(np.float32)  # n x 4
        xyz_seg_heatmap = np.concatenate((xyz_seg, heatmap_label), axis=1).astype(np.float32)  # n x 5
        xyz_seg_heatmap_kptof = np.concatenate((xyz_seg_heatmap, kpt_of_gt), axis=1).astype(np.float32)  # n x 8
        xyz_seg_heatmap_kptof = np.concatenate((xyz_seg_heatmap_kptof, kpt_of_x_gt), axis=1).astype(np.float32)  # n x 11
        xyz_seg_heatmap_kptof = np.concatenate((xyz_seg_heatmap_kptof, kpt_of_y_gt), axis=1).astype(np.float32)  # n x 14

        np.save(os.path.join(pcd_seg_heatmap_kpt_folder_path, pcd_filename), xyz_seg_heatmap_kptof)
        data[key]['pcd'] = pcd_filename
        # save mean and centroid of pcd
        data[key]['pcd_mean'] = m.item()
        data[key]['pcd_centroid'] = centroid.tolist()

        if args.visualize == True:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(concat_xyz_in_world)
            o3d.io.write_point_cloud(os.path.join(visualize_folder_path, 'pcd_' + str(key) + '.ply'), pcd)

            if key == 0 and args.offset == 'kpts':
                kpts = np.array([hole_keypoint_top_pos_in_world*1000,hole_keypoint_top_x_pos_in_world*1000, hole_keypoint_top_y_pos_in_world*1000])
                concat_pcd = o3d.geometry.PointCloud()
                concat_pcd.points = o3d.utility.Vector3dVector(np.concatenate((concat_xyz_in_world, kpts), axis=0))
                kpt_color = np.array([[1,0,0],[1,1,0],[0,1,0]])
                point_color = np.repeat([[0.9,0.9,0.9]], concat_xyz_in_world.shape[0], axis=0)
                color = np.concatenate((point_color, kpt_color), axis=0)
                concat_pcd.colors = o3d.utility.Vector3dVector(color)
                o3d.io.write_point_cloud(os.path.join(data_root, 'concat_xyzkpt_' + str(key) + '.ply'), concat_pcd)

                concat_pcd = o3d.geometry.PointCloud()
                concat_pcd.points = o3d.utility.Vector3dVector(concat_xyz_in_world)
                seg_color = np.array(seg_color).reshape(-1, 3)  # n x 3
                concat_pcd.colors = o3d.utility.Vector3dVector(seg_color)
                o3d.io.write_point_cloud(os.path.join(data_root, 'concat_xyzseg_' + str(key) + '.ply'), concat_pcd)

                concat_pcd = o3d.geometry.PointCloud()
                concat_pcd.points = o3d.utility.Vector3dVector(concat_xyz_in_world)
                heatmap_color = np.repeat(heatmap_label, 3, axis=1).reshape(-1, 3)  # n x 3
                concat_pcd.colors = o3d.utility.Vector3dVector(heatmap_color)
                o3d.io.write_point_cloud(os.path.join(data_root, 'concat_xyzheatmap_' + str(key) + '.ply'), concat_pcd)
        

    with open(os.path.join(data_root, 'peg_in_hole.yaml'), 'w') as f_w:
        yaml.dump(data, f_w)

if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print('Time elasped:{:.02f} seconds.'.format((end_time - start_time)))
