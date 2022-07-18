'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, neuron

tau_global = 1./(1. - 0.25)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.lif1 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= tau_global,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)

        # self.lif1 = neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
        #                          surrogate_function=surrogate.ATan(),
        #                          detach_reset=True)

        self.lif2 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        # self.lif2 = neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
        #                          surrogate_function=surrogate.ATan(),
        #                          detach_reset=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.lif1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.lif2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)


        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, total_timestep =6):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.total_timestep = total_timestep

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.lif_input = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                        surrogate_function=surrogate.ATan(),
                                        detach_reset=True)
        # self.lif_input = neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
        #                          surrogate_function=surrogate.ATan(),
        #                          detach_reset=True)


        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512*block.expansion, 256)
        self.lif_fc = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= tau_global,
                                        surrogate_function=surrogate.ATan(),
                                        detach_reset=True)
        # self.lif_fc = neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
        #                                           surrogate_function=surrogate.ATan(),
        #                                           detach_reset=True)
        self.fc2 = nn.Linear(256, num_classes)

        # for m in self.modules():
        #     if isinstance(m, Bottleneck):
        #         nn.init.constant_(m.bn3.weight, 0)
        #     elif isinstance(m, BasicBlock):
        #         nn.init.constant_(m.bn2.weight, 0)
        #     elif isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        # acc_voltage = 0
        output_list = []

        static_x = self.bn1(self.conv1(x))

        for t in range(self.total_timestep):
            out = self.lif_input(static_x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.lif_fc(self.fc1(out))
            out = self.fc2(out)

            # acc_voltage = acc_voltage + out
            output_list.append(out)

        # acc_voltage = acc_voltage / self.total_timestep

        return output_list


def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet19(num_classes, total_timestep):
    return ResNet(BasicBlock, [3,3,2], num_classes, total_timestep)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()