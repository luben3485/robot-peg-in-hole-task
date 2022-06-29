import os
import cv2
import yaml
import numpy as np
from PIL import Image
from collections import namedtuple
from skimage import transform
import torch
import torchvision.transforms.functional as F
import dsae.dsae as model
from dsae.train_util import sample_range, img_patch, obj_looks, bg_image, fractal_image, depth_flip

# settings
#arg = yaml.load(open(sys.argv[1], 'r'), yaml.Loader)

class DSAEMover(object):
    def __init__(self, ckpt_folder, num):
        arg = yaml.load(open('dsae/result/' + ckpt_folder + '/servo.yaml', 'r'), yaml.Loader)
        arg = namedtuple('Arg', arg.keys())(**arg)

        # images
        self.im_size = arg.im_size
        self.mean = arg.mean
        self.std = arg.std
        # model
        self.dsae = model.DSAE().cuda()

        # load model
        #self.load_checkpoint(arg.dir_base)
        self.load_checkpoint(os.path.join('result', ckpt_folder), num)
        self.dsae.eval()

        # visualize
        self.color = yaml.load(open('kovis/cfg/color.yaml', 'r'), Loader=yaml.Loader)
        self.num_obj = len(set(arg.obj_class))
    def load_checkpoint(self, base_dir, num):
        ckpt = 'ckpt_' + str(num) + '.pth'
        cp_net = torch.load(os.path.join('dsae', base_dir, ckpt))
        self.dsae.load_state_dict(cp_net['dsae_state_dict'])
        print('checkpoint loaded.')

    def img_proc(self, img, sizes):
        img = cv2.resize(img, tuple(sizes))
        '''
        grey = True
        noise = arg.noise
        inC = Image.fromarray(img)

        inC = obj_looks(inC,
                        a_hue=sample_range(arg.hue), a_saturate=sample_range(arg.saturation), a_value=sample_range(arg.brightness),
                        a_contrast=sample_range(arg.contrast), a_sharp=sample_range(arg.sharp), a_gamma=sample_range(arg.gamma))

        inC = np.array(inC, np.uint8)
        if sum(noise) > 0:
            fractal = (np.array(fractal_image(sizes)) - 127.) / 128.
            inC = (inC + (fractal if grey else fractal[:, :, None]) * np.random.uniform(*noise)). \
                clip(0, 255).round().astype(np.uint8)

        inC = F.to_tensor(inC)
        img = F.normalize(inC, [arg.mean], [arg.std])
        '''
        mean = torch.tensor(self.mean)
        std = torch.tensor(self.std)
        img = torch.from_numpy(img).float().permute(2, 0, 1).div(255).sub_(mean[:, None, None]).div_(std[:, None, None])

        #img = torch.from_numpy(img[None, ...]).float().div(255).sub_(self.mean).div_(self.std)
        return img.unsqueeze(0)

    def inference(self, img, visualize, tilt=False, yaw=False):
        img = self.img_proc(img, self.im_size[0])
        img = img.cuda()
        # forward-pass
        delta_xyz, delta_rot_euler, depth, speed, kpts = self.dsae.forward_inference(img)
        #speed = torch.sigmoid(speed).detach().cpu().item()
        speed = speed.detach().cpu().item()
        speed = max(0.00, speed)
        delta_xyz = (delta_xyz / torch.norm(delta_xyz)).detach().cpu().numpy()[0]
        delta_rot_euler = delta_rot_euler.detach().cpu().numpy()[0]
        ''''
        delta_xyz /= 80
        delta_rot_euler *= 10
        '''
        if tilt or yaw:
            pass
        else:
            pass
        if visualize:
            img = ((img.detach().squeeze().cpu().numpy() * 0.25) * 255 + 128).round().clip(0, 255).astype(np.uint8)
            depth = ((depth.detach().squeeze().cpu().numpy() * 0.25) * 255 + 128).round().clip(0, 255).astype(np.uint8)
            img = img.transpose(1, 2, 0)
            Image.fromarray(np.hstack((img, np.tile(depth[:, :, None], [1, 1, 3])))). \
                save('dsae/result.png')


        return delta_xyz, delta_rot_euler, speed

def main():
    kovis_mover = KOVISMover()
    inL = cv2.imread('kovis/left.png', cv2.IMREAD_GRAYSCALE)
    inR = cv2.imread('kovis/right.png', cv2.IMREAD_GRAYSCALE)
    vec, speed = kovis_mover.inference(inL, inR, visualize=True)
    print(vec, speed)

if __name__ == '__main__':
   main()
