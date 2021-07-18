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
        delta_rotation_matrix = np.array(self.anno_list[idx]['delta_rotation_matrix'], dtype= np.float)
        delta_translation = np.array(self.anno_list[idx]['delta_translation'], dtype= np.float)
        # ‘F’ means to flatten in column-major (Fortran- style) order
        #delta_rotation_matrix = delta_rotation_matrix.flatten('F')[:9].reshape(9,1)
        delta_rotation_matrix = delta_rotation_matrix.reshape(3,3)
        delta_translation = normalize(delta_translation) # normalization
        delta_translation = delta_translation.reshape(3,1)
        #delta_translation[0, 0] *= 66 # scale x
        #delta_translation[1, 0] *= 40  # scale y
        #delta_translation[2, 0] *= 33  # scale z

        #label = np.concatenate((delta_rotation_matrix_flatten, delta_translation_flatten), axis=0)
        #label = torch.squeeze(torch.from_numpy(label)).float()
        gt_rmat = torch.from_numpy(delta_rotation_matrix).float()
        gt_t = torch.from_numpy(delta_translation).float()


        rgb_img_path = os.path.join(self.img_folder_path, str(idx).zfill(6)+'_rgb.png')
        rgb_img = Image.open(rgb_img_path).convert("RGB")
        # The rgb normalization parameter
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]
        #rgb_std = [1, 1, 1]
        rgb_scale = [1.0, 1.0, 1.0]
        rgb_img = np.array(rgb_img) #PIL image to ndarray
        rgb_img = rgb_image_normalize(rgb_img, rgb_mean, rgb_scale, rgb_std)
        rgb_img = torch.from_numpy(rgb_img) # ndarray to tensor

        # The original depth image's unit is mm and it's data type is np.uint16.
        depth_img_path = os.path.join(self.img_folder_path, str(idx).zfill(6) + '_depth.png')
        depth_mm_img = Image.open(depth_img_path)
        # The depth normalization parameter
        depth_image_clip = 2000  # Clip the depth image further than 1500 mm
        depth_image_mean = 650  # origin 580
        depth_image_scale = 256  # scaled_depth = (raw_depth - depth_image_mean) / depth_image_scale
        depth_mm_img = np.array(depth_mm_img) #PIL image to ndarray
        depth_img = depth_image_normalize(depth_mm_img, depth_image_clip, depth_image_mean, depth_image_scale)
        depth_img = torch.unsqueeze(torch.from_numpy(depth_img), 2) # ndarray to tensor

        rgbd = torch.cat((rgb_img, depth_img), 2).permute(2,0,1).float()
        #if self.transforms is not None:
        #    rgbd = self.transforms(rgbd)
        return rgbd, gt_rmat, gt_t

    def __len__(self):
        return len(self.anno_list)

    def load_anno(self):
        with open(self.anno_file_path,'r') as f:
            anno_list = yaml.load(f,  Loader=yaml.RoundTripLoader)
        return anno_list

if __name__ == '__main__' :
    data_folder = '/tmp2/r09944001/data/pdc/logs_proto/xyz_track_curve_insertion_2021-06-20'
    '''
    transforms = transforms.Compose([
        transforms.CenterCrop(10),
        transforms.ToTensor(),
    ])
    '''
    dataset = RobotDataset(data_folder)
    dataiter = DataLoader(dataset, batch_size=16, shuffle=True)

    for idx, (rgbd, gt_rmat, gt_t) in enumerate(dataiter):
        print(idx,rgbd.shape)
        print(gt_t)
        print('='*30)

    #x max  0.014   0.015   *66
    #y max 0.0226   0.025   *40
    #z max 0.0295   0.03    *33
