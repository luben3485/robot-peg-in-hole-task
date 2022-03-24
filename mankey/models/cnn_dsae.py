import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import compute_rotation_matrix_from_ortho6d
Tensor = torch.Tensor

class get_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = dsae_encoder()
        self.decoder = dsae_decoder()
        self.action_net = action_net()
    def forward(self, x):
        feat, kpts = self.encoder(x)
        depth_pred = self.decoder(kpts)
        xyz_pred, rot_pred = self.action_net(kpts)
        '''
        print('feat', feat.shape)
        print('kpts', kpts.shape)
        print('depth_pred', depth_pred.shape)
        print('xyz_pred', xyz_pred.shape)
        print('rot_pred', rot_pred.shape)
        '''

        return xyz_pred, rot_pred, depth_pred

class dsae_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, 5, 2, 2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 16, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(16)
        self.spatial_soft_argmax = SpatialSoftArgmax()
        
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        kpts = self.spatial_soft_argmax(x)
        return x, kpts
    
    
class dsae_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(16, 128, 5, 2, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 3, 2, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 2, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
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
        heatmap = self.generate_laplace_heatmap(kpts)
        x = F.relu(self.conv1_bn(self.conv1(heatmap)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        depth_pred = self.sigmoid(self.conv5(x))
        
        return depth_pred
    
    
class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in `1`_.

    Concretely, the spatial softmax of each feature map is used to compute a
    weighted mean of the pixel locations, effectively performing a soft arg-max
    over the feature dimension.

    .. _1: https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize: bool = False):
        """Constructor.

        Args:
            normalize: Whether to use normalized image coordinates, i.e.
                coordinates in the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize


    def _coord_grid(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> Tensor:
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # Compute a spatial softmax over the input:
        # Given an input of shape (B, C, H, W), reshape it to (B*C, H*W) then
        # apply the softmax operator over the last dimension.
        b, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # Create a meshgrid of normalized pixel coordinates.
        xc, yc = self._coord_grid(h, w, x.device)

        # Element-wise multiply the x and y coordinates with the softmax, then
        # sum over the h*w dimension. This effectively computes the weighted
        # mean x and y locations.
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # Concatenate and reshape the result to (B, C, 2) where for every feature
        # we have the expected x and y pixel locations.
        
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c, 2)
      
class action_net(nn.Module):
    def __init__(self):
        super(action_net, self).__init__()
        self.use_cpu = False
        self.fc1 = nn.Linear(32, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, kpts):
        b = kpts.shape[0]
        kpts = kpts.view(b, -1)
        x = F.relu(self.bn1(self.fc1(kpts)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        xyz_pred = x[:, :3]
        rot_pred = x[:, 3:9]
        rot_pred = compute_rotation_matrix_from_ortho6d(rot_pred, use_cpu=self.use_cpu)
        
        return xyz_pred, rot_pred
'''
        
class get_model(nn.Module):
    def __init__(self,):
        super(get_model, self).__init__()
        self.use_cpu = False
        self.backbone = pointnet2_backbone()
        self.kpt_of_net = kpt_of_net()
        self.mask_net = mask_net()
        self.actionnet = action_net()
        #self.masknet = mask_net()
        #self.heatmapnet = heatmap_net()
        #self.actionnet = action_net(self.action_out_channel)
        
        
    def forward(self, xyz):
        global_features, point_features = self.backbone(xyz)
        kpt_of_pred = self.kpt_of_net(point_features)
        confidence = self.mask_net(point_features)
        # compute kpt
        xyz = xyz.permute(0, 2, 1)
        kpt_pred = xyz - kpt_of_pred[:, :, :3]  # (B, N, 3)
        kpt_x_pred = xyz - kpt_of_pred[:, :, 3:6]  # (B, N, 3)
        kpt_y_pred = xyz - kpt_of_pred[:, :, 6:]  # (B, N, 3)
        mean_kpt_pred = torch.sum(kpt_pred * confidence, dim=1) / torch.sum(confidence, dim=1)
        mean_kpt_x_pred = torch.sum(kpt_x_pred * confidence, dim=1) / torch.sum(confidence, dim=1)
        mean_kpt_y_pred = torch.sum(kpt_y_pred * confidence, dim=1) / torch.sum(confidence, dim=1)
        #mean_kpt_pred = torch.mean(kpt_pred, dim=1)
        vec_x_pred = mean_kpt_pred - mean_kpt_x_pred
        vec_y_pred = mean_kpt_pred - mean_kpt_y_pred
        ortho6d = torch.cat((vec_x_pred, vec_y_pred), axis=1)
        rot_mat_pred = compute_rotation_matrix_from_ortho6d(ortho6d, use_cpu=self.use_cpu)
        trans_of_pred, rot_of_6d_pred = self.actionnet(global_features, mean_kpt_pred, rot_mat_pred)
        rot_of_pred = compute_rotation_matrix_from_ortho6d(rot_of_6d_pred, use_cpu=self.use_cpu)
        
        return kpt_of_pred, trans_of_pred, rot_of_pred, mean_kpt_pred, rot_mat_pred, confidence
'''
    
if __name__ == '__main__':
    import os
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    model = get_model().cuda()
    rgb = torch.rand(5, 6, 256, 256) # (B, C, H, W)
    rgb = rgb.cuda()
    xyz_pred, rot_pred, depth_pred = model(rgb)
    print('xyz_pred', xyz_pred.shape)
    print('rot_pred', rot_pred.shape)
   
    