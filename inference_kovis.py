import os
import cv2
import yaml
import numpy as np
from PIL import Image
from collections import namedtuple
from skimage import transform
import torch
import torchvision.transforms.functional as F
import kovis.train_model as model
from kovis.train_util import sample_range, img_patch, obj_looks, bg_image, fractal_image, depth_flip

# settings
#arg = yaml.load(open(sys.argv[1], 'r'), yaml.Loader)

class KOVISMover(object):
    def __init__(self, ckpt_folder):
        arg = yaml.load(open('kovis/result/' + ckpt_folder + '/servo.yaml', 'r'), yaml.Loader)
        arg = namedtuple('Arg', arg.keys())(**arg)

        # images
        self.im_size = arg.im_size
        self.mean = arg.mean
        self.std = arg.std
        # model
        self.kper = model.KeyPointGaussian(arg.sigma_kp[0], (arg.num_keypoint, *arg.im_size[1]))
        self.enc = model.Encoder(arg.num_input, arg.num_keypoint, arg.growth_rate[0], arg.blk_cfg_enc, arg.drop_rate, self.kper).cuda()
        self.dec = model.Decoder(arg.num_keypoint, arg.growth_rate[1], arg.blk_cfg_dec, arg.num_output).cuda()
        self.cvt = model.ConverterServo(arg.num_keypoint * 2 * 3, arg.growth_rate[2], arg.blk_cfg_cvt, [sum(arg.motion_vec), 1]).cuda()
        # load model
        #self.load_checkpoint(arg.dir_base)
        self.load_checkpoint(os.path.join('result', ckpt_folder))
        self.enc.eval()
        self.dec.eval()
        self.cvt.eval()
        self.kper.sigma = arg.sigma_kp[1]
        # visualize
        self.color = yaml.load(open('kovis/cfg/color.yaml', 'r'), Loader=yaml.Loader)
        self.num_obj = len(set(arg.obj_class))
    def load_checkpoint(self, base_dir):
        cp_net = torch.load(os.path.join('kovis', base_dir, 'ckpt.pth'))
        self.enc.load_state_dict(cp_net['enc_state_dict'])
        self.dec.load_state_dict(cp_net['dec_state_dict'])
        self.cvt.load_state_dict(cp_net['cvt_state_dict'])
        print('checkpoint loaded.')

    def img_proc(self, img, sizes):
        img = cv2.resize(img, sizes)
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
        img = torch.from_numpy(img[None, ...]).float().div(255).sub_(self.mean).div_(self.std)
        return img.unsqueeze(0)

    def inference(self,imgs, visualize, tilt=False, yaw=False):
        inL = self.img_proc(imgs[0], self.im_size[0])
        inR = self.img_proc(imgs[1], self.im_size[0])
        inL = inL.cuda()
        inR = inR.cuda()
        # forward-pass
        keypL = self.enc(inL)
        keypR = self.enc(inR)
        depth, seg = self.dec(keypL[1])
        vec, speed = self.cvt(torch.cat((keypL[0], keypR[0]), dim=1))
        speed = torch.sigmoid(speed).detach().cpu().item()
        speed = max(0.00, speed - 0.1)
        xyz = (vec[:3] / torch.norm(vec[:3])).detach().cpu().numpy()
        if tilt or yaw:
            rot = vec[3:].detach().cpu().numpy()
            rot *= 5
        else:
            rot = None
        if visualize:
            # left
            keyp = keypL[1].detach().squeeze().cpu().numpy()
            keyps = np.zeros((inL.size(2), inL.size(3), 3), dtype=float)
            for j in range(keyp.shape[0]):
                keyps = keyps + np.tile(transform.resize(keyp[j], keyps.shape[:2])[:, :, np.newaxis], [1, 1, 3]) * \
                        np.array(self.color[j]).reshape(1, 1, 3)
            keyps = (keyps * 255).round().astype(np.uint8)
            img = ((inL.detach().squeeze().cpu().numpy() * 0.25) * 255 + 128).round().clip(0, 255).astype(np.uint8)
            depth = ((depth.detach().squeeze().cpu().numpy() * 0.25) * 255 + 128).round().clip(0, 255).astype(np.uint8)
            seg = (seg.squeeze().argmax(dim=0).detach().cpu().numpy() * 255. / (self.num_obj - 1)).astype(np.uint8)
            Image.fromarray(np.hstack((np.tile(img[:, :, None], [1, 1, 3]), keyps,
                                       np.tile(depth[:, :, None], [1, 1, 3]), np.tile(seg[:, :, None], [1, 1, 3])))). \
                save('kovis/result_left.png')
            # right
            depth, seg = self.dec(keypR[1])
            keyp = keypR[1].detach().squeeze().cpu().numpy()
            keyps = np.zeros((inR.size(2), inR.size(3), 3), dtype=float)
            for j in range(keyp.shape[0]):
                keyps = keyps + np.tile(transform.resize(keyp[j], keyps.shape[:2])[:, :, np.newaxis], [1, 1, 3]) * \
                        np.array(self.color[j]).reshape(1, 1, 3)
            keyps = (keyps * 255).round().astype(np.uint8)
            img = ((inR.detach().squeeze().cpu().numpy() * 0.25) * 255 + 128).round().clip(0, 255).astype(np.uint8)
            depth = ((depth.detach().squeeze().cpu().numpy() * 0.25) * 255 + 128).round().clip(0, 255).astype(np.uint8)
            seg = (seg.squeeze().argmax(dim=0).detach().cpu().numpy() * 255. / (self.num_obj - 1)).astype(np.uint8)
            Image.fromarray(np.hstack((np.tile(img[:, :, None], [1, 1, 3]), keyps,
                                       np.tile(depth[:, :, None], [1, 1, 3]), np.tile(seg[:, :, None], [1, 1, 3])))). \
                save('kovis/result_right.png')

        return xyz, rot, speed

def main():
    kovis_mover = KOVISMover()
    inL = cv2.imread('kovis/left.png', cv2.IMREAD_GRAYSCALE)
    inR = cv2.imread('kovis/right.png', cv2.IMREAD_GRAYSCALE)
    vec, speed = kovis_mover.inference(inL, inR, visualize=True)
    print(vec, speed)

if __name__ == '__main__':
   main()
