import os
import numpy as np
from PIL import Image
from ruamel import yaml
from .utils import rgb_image_normalize, depth_image_normalize, normalize
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RobotDataset(Dataset):
    def __init__(self, data_folder):
        self.anno_file_path = os.path.join(data_folder, 'processed/peg_in_hole.yaml')
        self.img_folder_path = os.path.join(data_folder, 'processed/images')
        self.anno_list = self.load_anno()
        self.transforms = transforms

    def __getitem__(self, idx):
        
        track = self.anno_list[idx]
        track_len = len(track)
        
        delta_rotation_matrix_track = []
        delta_translation_matrix_track = []
        rgb_img_track = []
        depth_img_track = []
        step_size_track = []
        pos_track = []
        for i in range(track_len):
            delta_rotation_matrix_track.append(track[i]['delta_rotation_matrix'])
            
            delta_translation = track[i]['delta_translation'] # normalization
            #delta_translation[0] *= 66
            #delta_translation[1] *= 40
            #delta_translation[2] *= 33
            delta_translation = normalize(delta_translation) # normalization
            delta_translation_matrix_track.append(delta_translation)
            pos_x = track[i]['gripper_pose'][0][3]
            pos_y = track[i]['gripper_pose'][1][3]
            pos_z = track[i]['gripper_pose'][2][3]
            pos_track.append([pos_x, pos_y, pos_z])

            rgb_image_filename = track[i]['rgb_image_filename']
            rgb_img_path = os.path.join(self.img_folder_path, rgb_image_filename)
            rgb_img = Image.open(rgb_img_path).convert("RGB")
            # The rgb normalization parameter
            rgb_mean = [0.485, 0.456, 0.406]
            rgb_std = [0.229, 0.224, 0.225]
            #rgb_std = [1, 1, 1]
            rgb_scale = [1.0, 1.0, 1.0]
            rgb_img = np.array(rgb_img) #PIL image to ndarray
            rgb_img = rgb_image_normalize(rgb_img, rgb_mean, rgb_scale, rgb_std)
            rgb_img_track.append(rgb_img)

            # The original depth image's unit is mm and it's data type is np.uint16.
            depth_image_filename = track[i]['depth_image_filename']
            depth_img_path = os.path.join(self.img_folder_path, depth_image_filename)
            depth_mm_img = Image.open(depth_img_path)
            # The depth normalization parameter
            depth_image_clip = 2000  # Clip the depth image further than 1500 mm
            depth_image_mean = 650  # origin 580
            depth_image_scale = 256  # scaled_depth = (raw_depth - depth_image_mean) / depth_image_scale
            depth_mm_img = np.array(depth_mm_img) #PIL image to ndarray
            depth_img = depth_image_normalize(depth_mm_img, depth_image_clip, depth_image_mean, depth_image_scale)
            depth_img_track.append(depth_img)

            step_size = track[i]['step_size']
            step_size_track.append(step_size)
        
        delta_rotation_matrix_track = np.array(delta_rotation_matrix_track, dtype = np.float).reshape(track_len, 3, 3)
        gt_r_track = torch.from_numpy(delta_rotation_matrix_track).float()
        
        delta_translation_matrix_track = np.array(delta_translation_matrix_track, dtype = np.float).reshape(track_len, 3, 1)
        gt_t_track = torch.from_numpy(delta_translation_matrix_track).float()
        
        gt_next_t_track = np.zeros((track_len, 3, 1))
        for i in range(len(delta_translation_matrix_track)-1):
               gt_next_t_track[i] = delta_translation_matrix_track[i+1]
        gt_next_t_track = torch.from_numpy(gt_next_t_track).float()

        step_size_track = np.array(step_size_track, dtype=np.float).reshape(-1,1)
        step_size_track = torch.from_numpy(step_size_track).float()
        
        pos_track = np.array(pos_track).reshape(track_len,3,1)
        gt_pos_track = torch.from_numpy(pos_track).float()

        rgb_img_track = np.array(rgb_img_track)
        rgb_img_track = torch.from_numpy(rgb_img_track) # ndarray to tensor

        depth_img_track = np.array(depth_img_track)
        depth_img_track = torch.unsqueeze(torch.from_numpy(depth_img_track), 3) # ndarray to tensor
        rgbd_track = torch.cat((rgb_img_track, depth_img_track), 3).permute(0,3,1,2).float()
        

        return rgbd_track, gt_r_track, gt_t_track, step_size_track, gt_pos_track, gt_next_t_track

    def __len__(self):
        return len(self.anno_list)

    def load_anno(self):
        with open(self.anno_file_path,'r') as f:
            anno_list = yaml.load(f,  Loader=yaml.RoundTripLoader)
        return anno_list

if __name__ == '__main__' :
    data_folder = '/tmp2/r09944001/data/pdc/logs_proto/xyz_track_curve_insertion_2021-06-20'
    dataset = RobotDataset(data_folder)
    dataiter = DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, (rgbd_track, gt_r_track, gt_t_track, step_size_track, gt_pos_track, gt_next_t_track) in enumerate(dataiter):
        print(idx, rgbd_track.shape, gt_r_track.shape, gt_t_track.shape, step_size_track.shape, gt_pos_track.shape, gt_next_t_track.shape)
        print(gt_t_track)
        if torch.isnan(gt_t_track).any():
            print(idx, 'nan')
            assert False
        
        print('='*30)

    #x max  0.014   0.015   *66
    #y max 0.0226   0.025   *40
    #z max 0.0295   0.03    *33
