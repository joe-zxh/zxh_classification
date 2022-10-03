
import torch
import torch.nn as nn

from typing import Callable, List


def _make_divisible(val: float, divider: int = 8):
    """向上取整"""
    result = int(val + divider * 0.5) // divider * divider
    result = max(divider, result)
    if result < val * 0.9:
        result += divider
    return result


class BottleNeckConfig:
    def __init__(self, input_channels: int, expan_channels: int, output_channels: int, kernel_size: int, use_se: bool, activation: str, stride: int):
        super(BottleNeckConfig, self).__init__()
        self.input_channels = input_channels
        self.expan_channels = expan_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.use_se = use_se
        self.activation = activation
        self.stride = stride


class ConvBNAct(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, stride: int, padding: int, activation: Callable[..., torch.nn.Module], groups: int = 1) -> None:
        
        super(ConvBNAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

        self.bn = nn.BatchNorm2d(num_features=output_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(SqueezeExcitation, self).__init__()

        intermediate_channels = _make_divisible(int(input_channels / 4))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=intermediate_channels, kernel_size=1, padding=0)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels=intermediate_channels, out_channels=output_channels, kernel_size=1, padding=0)
        self.act2 = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.act1(scale)
        scale = self.fc2(scale)
        scale = self.act2(scale)

        result = scale * x
        return result


class BottleNeck(nn.Module):

    def __init__(self, conf: BottleNeckConfig):
        super(BottleNeck, self).__init__()

        activation = nn.ReLU if conf.activation == "ReLU" else nn.Hardswish

        self.short_cut = conf.stride == 1 and conf.input_channels == conf.output_channels

        if conf.input_channels == conf.expan_channels:
            self.conv1 = nn.Identity()
        else:
            self.conv1 = ConvBNAct(input_channels=conf.input_channels, output_channels=conf.expan_channels, kernel_size=1, stride=1, padding=0, activation=activation)

        if conf.kernel_size == 3:
            padding = 1
        elif conf.kernel_size == 5:
            padding = 2
        self.conv2 = ConvBNAct(input_channels=conf.expan_channels, output_channels=conf.expan_channels, kernel_size=conf.kernel_size, stride=conf.stride, padding=padding, activation=activation)

        self.use_se = conf.use_se

        if self.use_se:
            self.se = SqueezeExcitation(input_channels=conf.expan_channels, output_channels=conf.expan_channels)

        self.conv3 = ConvBNAct(input_channels=conf.expan_channels, output_channels=conf.output_channels, kernel_size=1, stride=1, padding=0, activation=nn.Identity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)
        out = self.conv3(out)

        if self.short_cut:
            out += x
        return out


class MobileNetV3(nn.Module):

    def __init__(self, bneck_conf: List[BottleNeckConfig], scale: float = 1, last_channels: int = 1280, num_classes: int = 1000):
        super(MobileNetV3, self).__init__()

        self.conv1 = ConvBNAct(input_channels=3, output_channels=_make_divisible(16 * scale), kernel_size=3, activation=nn.Hardswish, stride=2, padding=1)

        self.bneck_layers = self._make_layers(bneck_conf, scale)

        conv2_input_channels = _make_divisible(bneck_conf[-1].output_channels)
        conv2_output_channels = _make_divisible(bneck_conf[-1].output_channels * 6)

        self.conv2 = ConvBNAct(input_channels=conv2_input_channels, output_channels=conv2_output_channels, kernel_size=1, activation=nn.Hardswish, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=conv2_output_channels, out_features=last_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=last_channels, out_features=num_classes)
        )

    def _make_layers(self, bneck_conf: List[BottleNeckConfig], scale: float) -> nn.Module:
        layers = nn.ModuleList()
        for conf in bneck_conf:
            conf.input_channels = _make_divisible(conf.input_channels * scale)
            conf.expan_channels = _make_divisible(conf.expan_channels * scale)
            conf.output_channels = _make_divisible(conf.output_channels * scale)
            layers.append(BottleNeck(conf))
        return nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bneck_layers(out)
        out = self.conv2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.classifier(out)
        return out


def get_mobilev3_large(scale: float = 1, num_classes: int = 1000):
    bneck_conf :List[BottleNeckConfig] = [
        BottleNeckConfig(16, 16, 16, 3, False, "ReLU", 1),
        BottleNeckConfig(16, 64, 24, 3, False, "ReLU", 2),
        BottleNeckConfig(24, 72, 24, 3, False, "ReLU", 1),
        BottleNeckConfig(24, 72, 40, 5, True, "ReLU", 2),
        BottleNeckConfig(40, 120, 40, 5, True, "ReLU", 1),
        BottleNeckConfig(40, 120, 40, 5, True, "ReLU", 1),
        BottleNeckConfig(40, 240, 80, 3, False, "HardSwish", 2),
        BottleNeckConfig(80, 200, 80, 3, False, "HardSwish", 1),
        BottleNeckConfig(80, 184, 80, 3, False, "HardSwish", 1),
        BottleNeckConfig(80, 184, 80, 3, False, "HardSwish", 1),
        BottleNeckConfig(80, 480, 112, 3, True, "HardSwish", 1),
        BottleNeckConfig(112, 672, 112, 3, True, "HardSwish", 1),
        BottleNeckConfig(112, 672, 160, 5, True, "HardSwish", 2),
        BottleNeckConfig(160, 960, 160, 5, True, "HardSwish", 1),
        BottleNeckConfig(160, 960, 160, 5, True, "HardSwish", 1),
    ]
    return MobileNetV3(bneck_conf=bneck_conf, scale=scale, last_channels=1280, num_classes=num_classes)


def get_mobilev3_small(scale: float = 1, num_classes: int = 1000):
    bneck_conf : List[BottleNeckConfig] = [
        BottleNeckConfig(16, 16, 16, 3, True, "ReLU", 2),
        BottleNeckConfig(16, 72, 24, 3, False, "ReLU", 2),
        BottleNeckConfig(24, 88, 24, 3, False, "ReLU", 1),
        BottleNeckConfig(24, 96, 40, 5, True, "HardSwish", 2),
        BottleNeckConfig(40, 240, 40, 5, True, "HardSwish", 1),
        BottleNeckConfig(40, 240, 40, 5, True, "HardSwish", 1),
        BottleNeckConfig(40, 120, 48, 5, True, "HardSwish", 1),
        BottleNeckConfig(48, 144, 48, 5, True, "HardSwish", 1),
        BottleNeckConfig(48, 288, 96, 5, True, "HardSwish", 2),
        BottleNeckConfig(96, 576, 96, 5, True, "HardSwish", 1),
        BottleNeckConfig(96, 576, 96, 5, True, "HardSwish", 1),
    ]
    return MobileNetV3(bneck_conf=bneck_conf, scale=scale, last_channels=1024, num_classes=num_classes)


if __name__ == "__main__":
    input = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    net = get_mobilev3_small(1.25)
    output = net(input)

    del net, output


