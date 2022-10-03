
import torch
import torch.nn as nn

from typing import Union, Type


class BasicBlock(nn.Module):
    def __init__(self, input_channels: int, internal_channels: int, output_channels: int, stride: int) -> None:
        super(BasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=internal_channels)

        self.conv2 = nn.Conv2d(in_channels=internal_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=output_channels)

        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    def __init__(self, input_channels: int, internal_channels: int, output_channels: int, stride: int) -> None:
        super(BottleneckBlock, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_channels, kernel_size=1, padding=0, stride=1) # torch实现的版本做了优化，stride放在conv2里面进行
        self.bn1 = nn.BatchNorm2d(num_features=internal_channels)

        self.conv2 = nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=internal_channels)

        self.conv3 = nn.Conv2d(in_channels=internal_channels, out_channels=output_channels, kernel_size=1, padding=0, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=output_channels)

        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, padding=0, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity

        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self, conf: dict, num_classes: int = 1000) -> None:
        super(Resnet, self).__init__()

        block_type = conf["block"]
        assert block_type in ["BasicBlock", "BottleneckBlock"]
        block = BasicBlock if block_type == "BasicBlock" else BottleneckBlock
        expansion = 1 if block_type == "BasicBlock" else 4

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = self._make_layers(block=block, input_channels=64, interval_channels=64, output_channels=64 * expansion, layer_counts=conf["layer_counts"])

        self.conv3 = self._make_layers(block=block, input_channels=64 * expansion, interval_channels=128, output_channels=128 * expansion, layer_counts=conf["layer_counts"])

        self.conv4 = self._make_layers(block=block, input_channels=128 * expansion, interval_channels=256, output_channels=256 * expansion, layer_counts=conf["layer_counts"])

        self.conv5 = self._make_layers(block=block, input_channels=256 * expansion, interval_channels=512, output_channels=512 * expansion, layer_counts=conf["layer_counts"])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(in_features=512 * expansion, out_features=num_classes)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

    def _make_layers(self, block: Type[Union[BasicBlock, BottleneckBlock]], input_channels: int, interval_channels: int, output_channels: int, layer_counts: list) -> nn.Module:

        layers = nn.ModuleList()

        for i, layer_count in enumerate(layer_counts):
            in_channels = input_channels if i == 0 else output_channels
            stride = 2 if i == 0 else 1
            layer = block(input_channels=in_channels, internal_channels=interval_channels, output_channels=output_channels, stride=stride)
            layers.append(layer)

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 把残差分支的权重设为0，这样残差块的初始输出就等于输入（设置最后一个bn层即可）
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
            elif isinstance(m, BottleneckBlock):
                nn.init.constant_(m.bn3.weight, 0)


conf: dict = {
    "18": {
        "block": "BasicBlock",
        "layer_counts": [2, 2, 2, 2]
    },
    "50": {
        "block": "BottleneckBlock",
        "layer_counts": [3, 4, 6, 3]
    }
}


def get_resnet18():
    return Resnet(conf=conf["18"])


def get_resnet50():
    return Resnet(conf=conf["50"])


if __name__ == "__main__":
    input = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    net = get_resnet18()
    output = net(input)

    del net, output






