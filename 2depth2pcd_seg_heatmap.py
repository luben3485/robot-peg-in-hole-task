import numpy as np
import cv2
import os
import open3d as o3d
import glob
import yaml
import math
import tqdm

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

def main():

    data_root = '/home/luben/data/pdc/logs_proto/insertion_xyzrot_eye_close_2022-01-12_tmp/processed'
    #data_root = '/home/luben/data/pdc/logs_proto/insertion_xyzrot_eye_close_2021-12-30/processed'
    image_folder_path = os.path.join(data_root, 'images')
    pcd_folder_path = os.path.join(data_root, 'pcd')
    pcd_seg_heatmap_folder_path = os.path.join(data_root, 'pcd_seg_heatmap')
    # create folder
    cwd = os.getcwd()
    os.chdir(data_root)
    if not os.path.exists('pcd'):
        os.makedirs('pcd')
    if not os.path.exists('pcd_seg_heatmap'):
        os.makedirs('pcd_seg_heatmap')
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
    for key, value in tqdm.tqdm(data.items()):
        depth_image_filename_list = data[key]['depth_image_filename']
        camera2world_list = data[key]['camera_matrix']
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

        # convert first hole's keypoint from camera to world coordinate
        hole_keypoint_top_pose_in_world = np.array(data[key]['hole_keypoint_obj_top_pose_in_world'])
        hole_keypoint_top_pos_in_world = hole_keypoint_top_pose_in_world[:3, 3]
        hole_keypoint_bottom_pose_in_world = np.array(data[key]['hole_keypoint_obj_bottom_pose_in_world'])
        hole_keypoint_bottom_pos_in_world = hole_keypoint_bottom_pose_in_world[:3, 3]

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
            sigma = 0.05
            heatmap_value = math.exp(-(y_scalar ** 2 + z_scalar ** 2 + x_scalar ** 2) / (2.0 * sigma ** 2))  # / (2 * math.pi * sigma**3 * math.sqrt(2*math.pi))
            heatmap_label.append(heatmap_value)

        '''
        # hole pcd
        hole_seg_pcd = o3d.geometry.PointCloud()
        hole_seg_pcd.points = o3d.utility.Vector3dVector(hole_seg_xyz*1000)
        o3d.io.write_point_cloud(os.path.join(data_root, 'hole_seg_pcd_' + str(key) + '.ply'), hole_seg_pcd)
        '''

        pcd_filename = str(key).zfill(6) + '_concat_xyz_seg_heatmap.npy'
        seg_label = np.array(seg_label).reshape(-1,1) # n x 1
        heatmap_label = np.array(heatmap_label).reshape(-1, 1)  # n x 1
        xyz_seg = np.concatenate((concat_xyz_in_world, seg_label), axis=1).astype(np.float32)  # n x 4
        xyz_seg_heatmap = np.concatenate((xyz_seg, heatmap_label), axis=1).astype(np.float32)  # n x 5
        np.save(os.path.join(pcd_seg_heatmap_folder_path, pcd_filename), xyz_seg_heatmap)
        data[key]['pcd'] = pcd_filename


        if key == 0:
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
    main()