import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from .utils import compute_rotation_matrix_from_ortho6d

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
        self.fc = nn.Linear(512 * block.expansion, out_channel)

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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class Model(nn.Module):
    def __init__(self, resnet_layers=34, in_channel=4):
        super(Model, self).__init__()
        block_type, layers, channels, name = resnet_spec[resnet_layers]
        self.backbone_net = ResNetBackbone(block_type, layers, in_channel=in_channel, out_channel=9)
        '''
        self.mlp = nn.Sequential(
            nn.Linear(3 * 3, self.inner_size),
            nn.LeakyReLU(),
            nn.Linear(self.inner_size, self.inner_size),
            nn.LeakyReLU(),
            nn.Linear(self.inner_size, self.out_channel)
        )
        '''
    def forward(self, x):
        x = self.backbone_net(x)
        out_r_6d = x[:, 0:6]
        out_r = compute_rotation_matrix_from_ortho6d(out_r_6d) # batch*3*3
        out_t = x[:, 6:9].view(-1,3,1) # batch*3*1
        return out_r, out_t

def test():
    model = Model()
    model.to(device)
    out_r, out_t = model(torch.randn(1, 4, 256, 256).float().to(device))
    print(model)
    print(out_r.size())
    print(out_t.size())

test()