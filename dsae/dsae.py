"""
This module implements, using PyTorch, the deep spatial autoencoder architecture presented in [1].
References:
    [1]: "Deep Spatial Autoencoders for Visuomotor Learning"
    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel
    Available at: https://arxiv.org/pdf/1509.06113.pdf
    [2]: https://github.com/tensorflow/tensorflow/issues/6271
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import torch
from torch import nn
import torch.nn.functional as F
import cv2

class CoordinateUtils(object):
    @staticmethod
    def get_image_coordinates(h, w, normalise):
        x_range = torch.arange(w, dtype=torch.float32)
        y_range = torch.arange(h, dtype=torch.float32)
        if normalise:
            x_range = (x_range / (w - 1)) * 2 - 1
            y_range = (y_range / (h - 1)) * 2 - 1
        image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
        image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
        return image_x, image_y


class SpatialSoftArgmax(nn.Module):
    def __init__(self, temperature=None, normalise=False):
        """
        Applies a spatial soft argmax over the input images.
        :param temperature: The temperature parameter (float). If None, it is learnt.
        :param normalise: Should spatial features be normalised to range [-1, 1]?
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)) if temperature is None else torch.tensor([temperature])
        self.normalise = normalise

    def forward(self, x):
        """
        Applies Spatial SoftArgmax operation on the input batch of images x.
        :param x: batch of images, of size (N, C, H, W)
        :return: Spatial features (one point per channel), of size (N, C, 2)
        """
        n, c, h, w = x.size()
        spatial_softmax_per_map = nn.functional.softmax(x.view(n * c, h * w) / self.temperature, dim=1)
        spatial_softmax = spatial_softmax_per_map.view(n, c, h, w)

        # calculate image coordinate maps
        image_x, image_y = CoordinateUtils.get_image_coordinates(h, w, normalise=self.normalise)
        # size (H, W, 2)
        image_coordinates = torch.cat((image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1)
        # send to device
        image_coordinates = image_coordinates.to(device=x.device)

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax.unsqueeze(-1)
        image_coordinates = image_coordinates.unsqueeze(0)
        out = torch.sum(expanded_spatial_softmax * image_coordinates, dim=[2, 3])
        # (N, C, 2)
        return out
### coarse-to-fine for sim2real
'''
class DSAE_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, temperature=None, normalise=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], 5, 2)
        self.conv1_bn = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], 3, 1)
        self.conv2_bn = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], 3, 2)
        self.conv3_bn = nn.BatchNorm2d(out_channels[2])
        self.conv4 = nn.Conv2d(out_channels[2], out_channels[3], 3, 1)
        self.conv4_bn = nn.BatchNorm2d(out_channels[3])
        self.conv5 = nn.Conv2d(out_channels[3], out_channels[4], 3, 1)
        self.conv5_bn = nn.BatchNorm2d(out_channels[4])
        self.spatial_soft_argmax = SpatialSoftArgmax(temperature=temperature, normalise=normalise)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        print(x.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        print(x.shape)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        print(x.shape)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        print(x.shape)
        x = F.relu(self.conv5_bn(self.conv5(x)))
        print(x.shape)
        out = self.spatial_soft_argmax(x)
        print(out.shape)
        return out
'''
### original DSAE
class DSAE_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, temperature=None, normalise=False):
        """
        Creates a Deep Spatial Autoencoder encoder
        :param in_channels: Input channels in the input image
        :param out_channels: Output channels for each of the layers. The last output channel corresponds to half the
        size of the low-dimensional latent representation.
        :param temperature: Temperature for spatial soft argmax operation. See SpatialSoftArgmax.
        :param normalise: Normalisation of spatial features. See SpatialSoftArgmax.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=7, stride=2, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=5)
        self.batch_norm2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=5)
        self.batch_norm3 = nn.BatchNorm2d(out_channels[2])
        self.activ = nn.ReLU()
        self.spatial_soft_argmax = SpatialSoftArgmax(temperature=temperature, normalise=normalise)

    def forward(self, x):
        out_conv1 = self.activ(self.batch_norm1(self.conv1(x)))
        #print(out_conv1.shape)
        out_conv2 = self.activ(self.batch_norm2(self.conv2(out_conv1)))
        #print(out_conv2.shape)
        out_conv3 = self.activ(self.batch_norm3(self.conv3(out_conv2)))
        #print(out_conv3.shape)
        out = self.spatial_soft_argmax(out_conv3)
        #print(out.shape)
        return out
### coarse-to-fine for sim2real
'''
class DSAE_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, normalise=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], 5, 1, 2)
        self.conv1_bn = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(out_channels[2])
        self.conv4 = nn.Conv2d(out_channels[2], out_channels[3], 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(out_channels[3])
        self.conv5 = nn.Conv2d(out_channels[3], 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
        #for 64x64
        #self.conv1 = nn.ConvTranspose2d(in_channels, out_channels[0], 5, 1, 2)
        #self.conv1_bn = nn.BatchNorm2d(out_channels[0])
        #self.conv2 = nn.ConvTranspose2d(out_channels[0], out_channels[1], 3, 1, 1)
        #self.conv2_bn = nn.BatchNorm2d(out_channels[1])
        #self.conv3 = nn.ConvTranspose2d(out_channels[1], out_channels[2], 3, 1, 1)
        #self.conv3_bn = nn.BatchNorm2d(out_channels[2])
        #self.conv4 = nn.ConvTranspose2d(out_channels[2], out_channels[3], 3, 1, 1)
        #self.conv4_bn = nn.BatchNorm2d(out_channels[3])
        #self.conv5 = nn.ConvTranspose2d(out_channels[3], 1, 3, 1, 1)
        #self.sigmoid = nn.Sigmoid()
        
        #for 256x256  self.decoder = DSAE_Decoder(in_channels=16, out_channels=(128, 64, 64, 32))
        #self.conv1 = nn.ConvTranspose2d(in_channels, out_channels[0], 5, 2, 2, 1)
        #self.conv1_bn = nn.BatchNorm2d(out_channels[0])
        #self.conv2 = nn.ConvTranspose2d(out_channels[0], out_channels[1], 3, 1, 1)
        #self.conv2_bn = nn.BatchNorm2d(out_channels[1])
        #self.conv3 = nn.ConvTranspose2d(out_channels[1], out_channels[2], 3, 2, 1, 1)
        #self.conv3_bn = nn.BatchNorm2d(out_channels[2])
        #self.conv4 = nn.ConvTranspose2d(out_channels[2], out_channels[3], 3, 1, 1)
        #self.conv4_bn = nn.BatchNorm2d(out_channels[3])
        #self.conv5 = nn.ConvTranspose2d(out_channels[3], 1, 3, 1, 1)
        #self.sigmoid = nn.Sigmoid()
    
    def generate_laplace_heatmap(self, kpts, scale=0.05, w=64, h=64):
        # kpts: (B x 16 X 2)
        device = kpts.device
        b, c, xy = kpts.shape
        kpts = kpts.view(b*c, -1) #(96, 2)
        heatmap = torch.zeros(b*c, h, w)
        xc, yc = torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
            )
        ) # xc(64,64) yc(64,64)
        xc = xc.repeat(b*c, 1, 1).view(b*c, h, w) #(96, 64, 64)
        yc = yc.repeat(b*c, 1, 1).view(b*c, h, w) #(96, 64, 64)
        x_pred = kpts[:, 0].view(-1, 1 ,1) #(96, 1, 1)
        y_pred = kpts[:, 1].view(-1, 1, 1) #(96, 1, 1)
        xc = torch.pow(xc - x_pred, 2)
        yc = torch.pow(yc - y_pred, 2)

        heatmap = 1/(2 * scale) * torch.exp(-1 * torch.sqrt(xc + yc) / scale) 

        return heatmap.view(b, c, h, w)
    
    def forward(self, kpts):
        #test for heatmap (kpt range [0-64])
        #kpts[0, 0,:] =  torch.tensor([32, 32]).cuda()
        #heatmap = self.generate_laplace_heatmap(kpts)
        #tmp = heatmap.permute(0, 2, 3, 1)[0, :,:,:1].cpu().detach().numpy()
        #print(tmp.shape)
        #cv2.imwrite('heatmap.jpg', tmp*255)
        
        heatmap = self.generate_laplace_heatmap(kpts)
        #print(heatmap.shape)
        x = F.relu(self.conv1_bn(self.conv1(heatmap)))
        #print('ConvTranpose1:', x.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        #print('ConvTranpose2:', x.shape)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        #print('ConvTranpose3:', x.shape)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        #print('ConvTranpose4:', x.shape)
        depth_pred = self.sigmoid(self.conv5(x))
        #print('ConvTranpose5:', depth_pred.shape)
        
        return depth_pred
'''
### original DSAE
class DSAE_Decoder(nn.Module):
    def __init__(self, image_output_size, latent_dimension, normalise=True):
        """
        Creates a Deep Spatial Autoencoder decoder
        :param image_output_size: (height, width) of the output, grayscale image
        :param latent_dimension: dimension of the low-dimensional encoded features.
        :param normalise: True if output in range [-1, 1], False for range [0, 1]
        """
        super().__init__()
        self.height, self.width = image_output_size
        self.latent_dimension = latent_dimension
        self.decoder = nn.Linear(in_features=latent_dimension, out_features=self.height * self.width)
        self.activ = nn.Tanh() if normalise else nn.Sigmoid()

    def forward(self, x):
        out = self.activ(self.decoder(x))
        out = out.view(-1, 1, self.height, self.width)
        return out

class DSAE(nn.Module):
    def __init__(self):
        super().__init__()
        ### original DSAE
        self.encoder = DSAE_Encoder(in_channels=3, out_channels=(64, 32, 16), temperature=None, normalise=False)
        self.decoder = DSAE_Decoder(image_output_size=(64, 64), latent_dimension=32, normalise=False)
    
        ### coarse-to-fine for sim2real
        #self.encoder = DSAE_Encoder(in_channels=3, out_channels=(64, 128, 256, 128, 16), normalise=False)
        #self.decoder = DSAE_Decoder(in_channels=16, out_channels=(128, 64, 64, 32))
        self.actionnet = ActionNet()
        
    def forward(self, x):
        kpts = self.encoder(x)
        b, c, _2 = kpts.size()
        kpts = kpts.view(b, c * 2)
        # kpts (b, n, 2)
        recon = self.decoder(kpts)
        delta_xyz_pred, delta_rot_euler_pred = self.actionnet(kpts)
        return delta_xyz_pred, delta_rot_euler_pred, recon
    
    def forward_inference(self, x):
        kpts = self.encoder(x)
        b, c, _2 = kpts.size()
        kpts = kpts.view(b, c * 2)
        # kpts (b, n, 2)
        recon = self.decoder(kpts)
        delta_xyz_pred, delta_rot_euler_pred = self.actionnet(kpts)
        return delta_xyz_pred, delta_rot_euler_pred, recon, kpts.view(b, c, _2)


class DSAE_Loss(object):
    def __init__(self, add_g_slow=True):
        """
        Loss for deep spatial autoencoder.
        :param add_g_slow: Should g_slow contribution be added? See [1].
        """
        self.add_g_slow = add_g_slow
        self.mse_loss = nn.MSELoss(reduction="sum")

    def __call__(self, reconstructed, target, ft_minus1=None, ft=None, ft_plus1=None):
        """
        Performs the loss computation, and returns both loss components.
        For the start of a trajectory, where ft_minus1 = ft, simply pass in ft_minus1=ft, ft=ft
        For the end of a trajectory, where ft_plus1 = ft, simply pass in ft=ft, ft_plus1=ft
        :param reconstructed: Reconstructed, grayscale image
        :param target: Target, grayscale image
        :param ft_minus1: Features produced by the encoder for the previous image in the trajectory to the target one
        :param ft: Features produced by the encoder for the target image
        :param ft_plus1: Features produced by the encoder for the next image in the trajectory to the target one
        :return: A tuple (mse, g_slow) where mse = the MSE reconstruction loss and g_slow = g_slow contribution term ([1])
        """
        loss = self.mse_loss(reconstructed, target)
        g_slow_contrib = torch.zeros(1, device=loss.device)
        if self.add_g_slow:
            g_slow_contrib = self.mse_loss(ft_plus1 - ft, ft - ft_minus1)
        return loss, g_slow_contrib
    
    
class ActionNet(nn.Module):
    def __init__(self):
        super(ActionNet, self).__init__()
        self.fc1 = nn.Linear(32, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 6)

    def forward(self, kpts):
        b = kpts.shape[0]
        kpts = kpts.view(b, -1)
        x = F.relu(self.bn1(self.fc1(kpts)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
    
        delta_xyz_pred = x[:, :3]
        delta_rot_euler_pred = x[:, 3:] 
        
        return delta_xyz_pred, delta_rot_euler_pred
    
class RMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb = torch.randn(4, 3, 64, 64).to(device)

    dsae_model = DSAE().to(device)
    
    xyz, rot, recon = dsae_model(rgb)
    print('depth:', recon.shape)
    print('xyz:', xyz.shape)
    print('rot:', rot.shape)
