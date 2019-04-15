# ------------------------------用50行代码搭建ResNet-------------------------------------------
from torch import nn
import torch as t
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    # 实现子module: Residual    Block
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, int(outchannel*0.25), 1,1, bias=False),
            nn.BatchNorm2d(int(outchannel*0.25)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel*0.25), int(outchannel*0.25), 3, stride, 1, bias=False),
            nn.BatchNorm2d(int(outchannel * 0.25)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel*0.25),outchannel,1,1,bias=False)
            #nn.BatchNorm2d(outchannel)
        )
        self.bn=nn.BatchNorm2d(outchannel)  
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(self.bn(out))

def _make_layer(inchannel, outchannel, block_num, stride=1):
    # 构建layer,包含多个residual block
    shortcut = nn.Sequential(
        nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
        nn.BatchNorm2d(outchannel))

    layers = []
    layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResidualBlock(outchannel, outchannel))
    return nn.Sequential(*layers)


def addresnet():
    layers=[]
    layer1 = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 3, 2, 1, bias=False),
        nn.ReLU(inplace=True)
    )
    layers.append(layer1)

    layer2 = _make_layer(32, 32, 3)
    layer21=_make_layer(32,64,3,stride=2)
    layer3 = _make_layer(64, 128, 4, stride=2)
    layers.append(layer2)
    layers.append(layer21)
    layers.append(layer3)

    layer4=_make_layer(128,256,6,stride=2)
    layers.append(layer4)

    layer5=_make_layer(256,512,5,stride=2)
    layers.append(layer5)

    #layer6=_make_layer(256,512,3,stride=2)
    #layers.append(layer6)

    return layers


