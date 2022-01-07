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

    data_root = '/tmp2/r09944001/data/pdc/logs_proto/insertion_xyzrot_eye_2021-11-13/processed'
    #data_root = '/tmp2/r09944001/data/pdc/logs_proto/insertion_xyzrot_eye_toy_2021-09-27/processed'
    
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
        depth_image_filename = data[key]['depth_image_filename']
        #for depth_image_filename in depth_image_filename_list:
        #depth_image_filename = depth_image_filename_list
        if 1==1:
            depth_img = cv2.imread(os.path.join(image_folder_path, depth_image_filename), cv2.IMREAD_ANYDEPTH)
            xyz, choose = depth_2_pcd(depth_img, factor, intrinsic_matrix)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            #o3d.io.write_point_cloud(os.path.join(pcd_folder_path, depth_image_filename.split('.')[0] + '.ply'), pcd)
            down_pcd = pcd.uniform_down_sample(every_k_points=8)
            #o3d.io.write_point_cloud(os.path.join(pcd_folder_path, depth_image_filename.split('.')[0] + '_down.ply'), down_pcd)
            new_xyz = np.asarray(down_pcd.points)
            print(key, new_xyz.shape)
            new_xyz = new_xyz[:8000,:]
            pcd_filename = depth_image_filename.split('.')[0] + '.npy'
            np.save(os.path.join(pcd_folder_path, pcd_filename), new_xyz)

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