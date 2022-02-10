import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation, PointNetSetAbstraction


class pointnet2_backbone(nn.Module):
    def __init__(self):
        super(pointnet2_backbone, self).__init__()

        #self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 0, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        #self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.sa4 = PointNetSetAbstraction(None, None, None, 256+256+3, [256, 512, 512], True)
        self.fp4 = PointNetFeaturePropagation(512+512, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        '''
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        '''
    def forward(self, xyz):
        B, _, _ = xyz.shape
        # normal channel is not used here
        norm = None
        l0_xyz = xyz[:,:3,:]
        
        l1_xyz, l1_points = self.sa1(l0_xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        global_features = l4_points.view(B, 512)  # (B, 512)
        point_features = l0_points.transpose(1,2) # (B, N, 128)
        
        return global_features, point_features
        '''
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points
        '''
        
        
class kpt_of_net(nn.Module):
    def __init__(self):
        super(kpt_of_net, self).__init__()
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 32, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 3, 1)
    
    def forward(self, point_features):
        point_features = point_features.permute(0,2,1)
        point_features = F.relu(self.bn1(self.conv1(point_features)))
        point_features = F.relu(self.bn2(self.conv2(point_features)))
        kpt_of_pred = self.conv3(point_features)
        kpt_of_pred = kpt_of_pred.permute(0, 2, 1) # (B, N, C)

        return kpt_of_pred
    
    
class mask_net(nn.Module):
    def __init__(self):
        super(mask_net, self).__init__()
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 2, 1)
        
    def forward(self, global_features, point_features):
        point_features = point_features.permute(0,2,1)
        point_features = self.drop1(F.relu(self.bn1(self.conv1(point_features))))
        point_features = self.conv2(point_features)
        mask = F.log_softmax(point_features, dim=1)
        mask = point_features.permute(0, 2, 1)

        return mask 
    
    
class heatmap_net(nn.Module):
    def __init__(self):
        super(heatmap_net, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.bn1= nn.BatchNorm1d(64)
        self.bn2= nn.BatchNorm1d(64)
        self.sigmoid = nn.Sigmoid()
        
        ###this version concatenate global features and point features
        '''
        self.fc1 = nn.Linear(640, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc3_2 = nn.Linear(256, 1)
        self.bn1= nn.BatchNorm1d(512)
        self.bn2= nn.BatchNorm1d(256)
        self.sigmoid = nn.Sigmoid()
        '''
        
    def forward(self, global_features, point_features):
        b1 = F.leaky_relu(self.fc1(point_features) , negative_slope = 0.2)
        b2 = F.leaky_relu(self.fc2(b1) , negative_slope = 0.2)
        heatmap = self.sigmoid(self.fc3(b2))

        ###this version concatenate global features and point features
        ''' 
        p_num = point_features.shape[1]
        global_features = global_features.unsqueeze(1)
        global_features = global_features.repeat(1,p_num,1)
        all_feature = torch.cat((global_features ,point_features) , dim = -1)
        b1 = F.leaky_relu(self.fc1(all_feature) , negative_slope = 0.2)
        b2 = F.leaky_relu(self.fc2(b1) , negative_slope = 0.2)
        #sub_branch 1
        b3 = F.leaky_relu(self.fc3(b2) , negative_slope = 0.2)
        heatmap = self.sigmoid(self.fc3_2(b3))
        '''
        return heatmap
    
    
class action_net(nn.Module):
    def __init__(self, out_channel):
        super(action_net, self).__init__()
        self.fc1 = nn.Linear(16000, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(4096, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(512+512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.drop4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(0.5)
        self.fc6_1 = nn.Linear(256, out_channel)
        
        self.fc6_2 = nn.Linear(256, 64)
        self.bn6_2 = nn.BatchNorm1d(64)
        self.fc7_2 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, global_features, heatmap):
        heatmap = heatmap.view(-1, 16000)
        heatmap = self.drop1(F.relu(self.bn1(self.fc1(heatmap))))
        heatmap = self.drop2(F.relu(self.bn2(self.fc2(heatmap))))
        heatmap = self.drop3(F.relu(self.bn3(self.fc3(heatmap))))
        all_feature = torch.cat((global_features, heatmap) , dim = -1)
        #print('all feature:', all_feature.size())
        all_feature = self.drop4(F.relu(self.bn4(self.fc4(all_feature))))
        all_feature = self.drop5(F.relu(self.bn5(self.fc5(all_feature))))
        action = self.fc6_1(all_feature)
        scalar = self.fc6_2(all_feature)
        scalar = self.sigmoid(self.fc7_2(scalar))
        return action, scalar
        
        
class get_model(nn.Module):
    def __init__(self, action_out_channel):
        super(get_model, self).__init__()
        self.action_out_channel = action_out_channel
        self.backbone = pointnet2_backbone()
        #self.masknet = mask_net()
        #self.heatmapnet = heatmap_net()
        #self.actionnet = action_net(self.action_out_channel)
        self.kpt_of_net = kpt_of_net()
        
    def forward(self, xyz):
        global_features, point_features = self.backbone(xyz)
        '''
        #mask = self.masknet(global_features, point_features)
        heatmap = self.heatmapnet(global_features, point_features)
        #filtered_heatmap = torch.mul(mask, heatmap)
        action, scalar = self.actionnet(global_features, heatmap)
        '''
        kpt_of_pred = self.kpt_of_net(point_features)
        #print('global features:', global_features.size())
        #print('point features', point_features.size())
        #print('mask:', mask.size())
        #print('heatmap:', heatmap.size())
        #print('filtered_heatmap:', filtered_heatmap.size())
        #print('action:', action.size())

        return kpt_of_pred

    
if __name__ == '__main__':
    model = get_model(9)
    xyz = torch.rand(6, 3, 1024) # (B, N, 3)
    kpt_of_pred = model(xyz)
    print(kpt_of_pred.size()) # (B, N, 3)
    