import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from .utils import compute_rotation_matrix_from_ortho6d
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# The specification of resnet
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101')}

# The backbone for resnet
class ResNetBackbone(nn.Module):
    def __init__(self, block, layers, in_channel=3, out_channel=9):
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return x

class Model(nn.Module):
    def __init__(self, resnet_layers=34, in_channel=4, out_channel=9):
        super(Model, self).__init__()
        block_type, layers, channels, name = resnet_spec[resnet_layers]
        self.backbone_net = ResNetBackbone(block_type, layers, in_channel=in_channel, out_channel=out_channel)
        self.mlp = nn.Sequential(
            nn.Linear(512, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 60),
            nn.LeakyReLU(),
            nn.Linear(60, out_channel+1)   # one channel for step_size
        )
        self.EEf = nn.Sequential(
            nn.Linear(3, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 60),
            nn.LeakyReLU(),
            nn.Linear(60, 3) 
        )

    def forward(self, rgbd, pos):
        x = rgbd
        x = self.backbone_net(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        out_r_6d = x[:, 0:6]
        out_r = compute_rotation_matrix_from_ortho6d(out_r_6d) # batch*3*3
        out_t = x[:, 6:9].view(-1,3,1) # batch*3*1
        #_t = out_t.clone()
        #_t[:,0,0] /= 66
        #_t[:,1,0] /= 40
        #_t[:,2,0] /= 33
        out_step = x[:, 9]
        out_step = nn.Sigmoid()(out_step.view(-1,1))
        return out_r, out_t, out_step
        #next_pos = pos + _t
        #next_pos = next_pos.view(-1,3)
        #out_next_t = self.EEf(next_pos)
        #out_next_t = out_next_t.view(-1,3,1)            
        #return out_r, out_t, out_step, out_next_t

def initialize_backbone_from_modelzoo(
        backbone,  # type: ResNetBackbone,
        resnet_num_layers,  # type: int
        image_channels,  # type: int
    ):
    assert image_channels == 2 or image_channels == 3 or image_channels == 4
    _, _, _, name = resnet_spec[resnet_num_layers]
    org_resnet = model_zoo.load_url(model_urls[name])
    # Drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
    org_resnet.pop('fc.weight', None)
    org_resnet.pop('fc.bias', None)
    # Load the backbone
    if image_channels is 3:
        backbone.load_state_dict(org_resnet)
    elif image_channels is 2:
        # Modify the first conv
        conv1_weight_old = org_resnet['conv1.weight']
        avg_weight = conv1_weight_old.mean(dim=1, keepdim=False)
        conv1_weight = torch.zeros((64, 2, 7, 7))
        conv1_weight[:, 0, :, :] = avg_weight
        conv1_weight[:, 1, :, :] = avg_weight
        org_resnet['conv1.weight'] = conv1_weight
        # Load it
        backbone.load_state_dict(org_resnet)
    elif image_channels is 4:
        # Modify the first conv
        conv1_weight_old = org_resnet['conv1.weight']
        conv1_weight = torch.zeros((64, 4, 7, 7))
        conv1_weight[:, 0:3, :, :] = conv1_weight_old
        avg_weight = conv1_weight_old.mean(dim=1, keepdim=False)
        conv1_weight[:, 3, :, :] = avg_weight
        org_resnet['conv1.weight'] = conv1_weight
        # Load it
        backbone.load_state_dict(org_resnet)


def init_from_modelzoo(
        model,  # type: Model,
        resnet_num_layers,  # type: int,
        image_channels, # type: int,
    ):
    initialize_backbone_from_modelzoo(
        model.backbone_net,
        resnet_num_layers,
        image_channels)


def test():
    model = Model()
    model.to(device)
    batch_size = 2
    out_r, out_t, out_step, out_next_t = model(torch.randn(batch_size, 4, 256, 256).float().to(device), torch.randn(batch_size,3,1).float().to(device))
    print(model)
    print(out_r.size())
    print(out_t.size())
    print(out_step.size())
    print(out_next_t.size())

if __name__ == '__main__':
    test()
