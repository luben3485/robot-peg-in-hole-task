import os
from PIL import Image
import torch
import numpy as np
from .models.model import Model
from .dataset.utils import rgb_image_normalize, depth_image_normalize

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MoveInference():
    def __init__(self, checkpoints_file_path):
        assert os.path.exists(checkpoints_file_path)
        resnet_num_layers = 18
        image_channels = 4
        self.model = Model(resnet_layers=resnet_num_layers, in_channel=image_channels)
        self.model.load_state_dict(torch.load(checkpoints_file_path))
        self.model.to(device)


    def process_single_raw(self, rgb_img, depth_mm_img):
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]
        rgb_scale = [1.0, 1.0, 1.0]
        rgb_img = np.array(rgb_img)
        rgb_img = rgb_image_normalize(rgb_img, rgb_mean, rgb_scale, rgb_std)
        rgb_img = torch.from_numpy(rgb_img)  # ndarray to tensor

        depth_image_clip = 2000  # Clip the depth image further than 1500 mm
        depth_image_mean = 650  # origin 580
        depth_image_scale = 256  # scaled_depth = (raw_depth - depth_image_mean) / depth_image_scale
        depth_mm_img = np.array(depth_mm_img)  # PIL image to ndarray
        depth_img = depth_image_normalize(depth_mm_img, depth_image_clip, depth_image_mean, depth_image_scale)
        depth_img = torch.unsqueeze(torch.from_numpy(depth_img), 2)  # ndarray to tensor

        rgbd = torch.cat((rgb_img, depth_img), 2).permute(2, 0, 1).float()
        rgbd = torch.unsqueeze(rgbd, 0)

        return rgbd

    def inference_single_frame(self, rgb=None, depth_mm=None , rgb_path='', depth_mm_path=''):
        if rgb_path != '' and depth_mm_path != '':
            rgb = Image.open(rgb_path)
            depth_mm = Image.open(depth_mm_path)
        rgbd = self.process_single_raw(rgb, depth_mm)

        self.model.eval()
        with torch.no_grad():
            out_r, out_t = self.model(rgbd.to(device))

        out_r = torch.squeeze(out_r).detach().cpu().numpy()  # (3,3)
        out_t = torch.squeeze(out_t).detach().cpu().numpy()  # (3, )
        out_t[0] /= 66 # scale x
        out_t[1] /= 40  # scale y
        out_t[2] /= 33  # scale z

        #rotation_6d = logits[:, 0:6]
        #rotation = self.compute_rotation_matrix_from_ortho6d(rotation_6d)
        #translation = logits[:, 6:9].view(-1,3,1)
        #pose_matrix = torch.cat((out_r, out_t), 2) # 1*3*4
        #pose_matrix = torch.squeeze(pose_matrix).detach().cpu() # 3*4
        #pose_matrix = torch.cat((pose_matrix, torch.tensor([[0., 0., 0., 1.]])),0) # 4*4
        #pose_matrix = pose_matrix.numpy()
        return out_r, out_t

if __name__ == '__main__':
    checkpoints_file_path = 'ckpnt/checkpoints_best.pth'
    data_folder = '/home/luben/data/pdc/logs_proto/insertion_2021-04-30/processed/images'
    idx = 0
    rgb_path = os.path.join(data_folder, str(idx).zfill(6)+'_rgb.png')
    depth_mm_path = os.path.join(data_folder, str(idx).zfill(6)+'_depth.png')
    mover = MoveInference(checkpoints_file_path)
    r, t = mover.inference_single_frame(rgb_path=rgb_path, depth_mm_path=depth_mm_path)
    print(r)
    print(t)





