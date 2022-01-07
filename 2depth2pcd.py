import numpy as np
import cv2
import os
import open3d as o3d
import glob
import yaml

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

    data_root = '/tmp2/r09944001/data/pdc/logs_proto/insertion_xyzrot_eye_2022-01-01/processed'
    #data_root = '/home/luben/data/pdc/logs_proto/insertion_xyzrot_eye_toy_2021-09-27/processed'
    image_folder_path = os.path.join(data_root, 'images')
    pcd_folder_path = os.path.join(data_root, 'pcd')
    # create folder
    cwd = os.getcwd()
    os.chdir(data_root)
    if not os.path.exists('pcd'):
        os.makedirs('pcd')
    os.chdir(cwd)

    focal_length = 309.019
    principal = 128
    factor = 1
    intrinsic_matrix = np.array([[focal_length, 0, principal],
                                 [0, focal_length, principal],
                                 [0, 0, 1]], dtype=np.float64)


    with open(os.path.join(data_root, 'peg_in_hole.yaml'), 'r') as f_r:
        data = yaml.load(f_r)
    for key, value in data.items():
        depth_image_filename_list = data[key]['depth_image_filename']
        camera2world_list = data[key]['camera_matrix']
        xyz_in_world_list = []
        for idx, depth_image_filename in enumerate(depth_image_filename_list):
            depth_img = cv2.imread(os.path.join(image_folder_path, depth_image_filename), cv2.IMREAD_ANYDEPTH)
            depth_img = depth_img / 1000 # unit: mm to m
            xyz, choose = depth_2_pcd(depth_img, factor, intrinsic_matrix)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            #pcd = create_point_cloud_from_depth_image(depth_img, o3d.camera.PinholeCameraIntrinsic(256,256,focal_length,focal_length,principal,principal))
            # o3d.io.write_point_cloud(os.path.join(pcd_folder_path, depth_image_filename.split('.')[0] + '.ply'), pcd)
            down_pcd = pcd.uniform_down_sample(every_k_points=8)
            if key == 0:
                o3d.io.write_point_cloud(os.path.join(data_root, depth_image_filename.split('.')[0] + '_down.ply'), down_pcd)
            down_xyz = np.asarray(down_pcd.points)
            #print(key, new_xyz.shape)
            down_xyz_in_camera = down_xyz[:8000, :]
            #pcd_filename = depth_image_filename.split('.')[0] + '.npy'
            #np.save(os.path.join(pcd_folder_path, pcd_filename), new_xyz_in_camera)

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

        '''
        concat_xyz_in_camera = []
        for xyz in concat_xyz_in_world:
            camera2world = np.array(camera2world_list[0])
            world2camera = np.linalg.inv(camera2world)
            xyz = np.append(xyz, [1], axis=0).reshape(4, 1)
            xyz_camera = world2camera.dot(xyz)
            xyz_camera = xyz_camera[:3]
            concat_xyz_in_camera.append(xyz_camera)

        concat_xyz_in_camera = np.array(concat_xyz_in_camera)
        '''
        pcd_filename = str(key).zfill(6) + '_concat_pcd.npy'
        np.save(os.path.join(pcd_folder_path, pcd_filename), concat_xyz_in_world)
        data[key]['pcd'] = pcd_filename

        # test
        if key == 0:
            concat_pcd = o3d.geometry.PointCloud()
            concat_pcd.points = o3d.utility.Vector3dVector(concat_xyz_in_world)
            o3d.io.write_point_cloud(os.path.join(data_root, 'concat_' + str(key) + '.ply'), concat_pcd)

    with open(os.path.join(data_root, 'peg_in_hole.yaml'), 'w') as f_w:
        yaml.dump(data, f_w)

    '''
    num = '020000'
    xyz = np.load('/home/luben/data/pdc/logs_proto/insertion_xyzrot_eye_2021-11-19/processed/pcd/' + num + '_depth.npy')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(num+ '_depth.ply', pcd)
    '''
    '''
    folder_path = '/home/luben/data/pdc/logs_proto/insertion_xyzrot_eye_2021-11-19/processed/images'
    depth_img_name = '000001_depth.png'
    depth_img = cv2.imread(os.path.join(folder_path, depth_img_name), cv2.IMREAD_ANYDEPTH)
    focal_length = 309.019
    principal = 128
    factor = 1
    intrinsic_matrix = np.array([[focal_length,0, principal],
                                 [0, focal_length, principal],
                                 [0,0,1]],dtype=np.float64)
    xyz, choose = depth_2_pcd(depth_img, factor, intrinsic_matrix)
    
    '''

if __name__ == '__main__':
    main()
