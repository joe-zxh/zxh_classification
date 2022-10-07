
import torch
import torch.nn as nn


from typing import Callable, List


def _make_divisible(val: float, divisor: int = 8) -> int:
    result = int(val * 0.5 * divisor) // divisor * divisor # 上取整
    result = max(result, divisor)
    if result < 0.9 * val:
        result += divisor
    return result


class BNeckConfig:
    def __init__(self, input_channels: int, expan_channels: int, output_channels: int, kernel_size: int, use_se: bool, act: str, stride: int, scale: float = 1.0) -> None:
        self.input_channels = _make_divisible(input_channels * scale)
        self.expan_channels = _make_divisible(expan_channels * scale)
        self.output_channels = _make_divisible(output_channels * scale)
        self.kernel_size = kernel_size
        self.use_se = use_se
        self.act = act
        self.stride = stride


class ConvBNAct(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, stride: int, padding: int, groups: int = 1, act: Callable[..., nn.Module] = nn.ReLU) -> None:
        super(ConvBNAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(num_features=output_channels)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, intermediate_channels: int, output_channels: int) -> None:
        super(SqueezeExcitation, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels=intermediate_channels, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        self.act2 = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg(x)
        scale = self.fc1(scale)
        scale = self.act1(scale)
        scale = self.fc2(scale)
        scale = self.act2(scale)

        return x * scale


class BNeckBlock(nn.Module):
    def __init__(self, bneck_conf: BNeckConfig):

        super(BNeckBlock, self).__init__()

        self.use_se = bneck_conf.use_se
        act = nn.ReLU if bneck_conf.act == "ReLU" else nn.Hardswish

        self.conv1 = ConvBNAct(input_channels=bneck_conf.input_channels, output_channels=bneck_conf.expan_channels, kernel_size=1, stride=1, padding=0, act=act)

        padding = 2 if bneck_conf.kernel_size == 5 else 1
        self.conv2 = ConvBNAct(input_channels=bneck_conf.expan_channels, output_channels=bneck_conf.expan_channels, kernel_size=bneck_conf.kernel_size, stride=bneck_conf.stride, padding=padding, act=act, groups=bneck_conf.expan_channels)

        if self.use_se:
            self.se = SqueezeExcitation(input_channels=bneck_conf.expan_channels, intermediate_channels=_make_divisible(bneck_conf.expan_channels / 4.), output_channels=bneck_conf.expan_channels)

        self.conv3 = ConvBNAct(input_channels=bneck_conf.expan_channels, output_channels=bneck_conf.output_channels, kernel_size=1, stride=1, padding=0, act=nn.Identity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.conv1(x)
        result = self.conv2(result)
        if self.use_se:
            result = self.se(result)
        result = self.conv3(result)
        return result


class MobilenetV3(nn.Module):

    def __init__(self, bneck_confs: List[BNeckConfig], last_channels: int, num_classes: int = 1000) -> None:
        super(MobilenetV3, self).__init__()

        self.conv1 = ConvBNAct(input_channels=3, output_channels=bneck_confs[0].input_channels, kernel_size=3, stride=2, padding=1, act=nn.Hardswish)

        self.bneck_layes = self._make_layers(bneck_confs)

        self.conv2 = ConvBNAct(input_channels=bneck_confs[-1].output_channels, output_channels=bneck_confs[-1].output_channels*6, kernel_size=1, stride=1, padding=0, act=nn.Hardswish)

        self.pool = nn.AvgPool2d((7, 7))

        self.fc1 = nn.Conv2d(in_channels=bneck_confs[-1].output_channels*6, out_channels=last_channels, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels=last_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.conv1(x)
        result = self.bneck_layes(result)
        result = self.conv2(result)
        result = self.pool(result)
        result = self.fc1(result)
        result = self.fc2(result)
        return result

    def _make_layers(self, bneck_confs: List[BNeckConfig]) -> nn.Module:
        layers = nn.ModuleList()
        for bneck_conf in bneck_confs:
            layers.append(BNeckBlock(bneck_conf=bneck_conf))
        return nn.Sequential(*layers)


def get_mobilev3_large(scale: float = 1, num_classes: int = 1000):
    bneck_conf :List[BNeckConfig] = [
        BNeckConfig(16, 16, 16, 3, False, "ReLU", 1, scale),
        BNeckConfig(16, 64, 24, 3, False, "ReLU", 2, scale),
        BNeckConfig(24, 72, 24, 3, False, "ReLU", 1, scale),
        BNeckConfig(24, 72, 40, 5, True, "ReLU", 2, scale),
        BNeckConfig(40, 120, 40, 5, True, "ReLU", 1, scale),
        BNeckConfig(40, 120, 40, 5, True, "ReLU", 1, scale),
        BNeckConfig(40, 240, 80, 3, False, "HardSwish", 2, scale),
        BNeckConfig(80, 200, 80, 3, False, "HardSwish", 1, scale),
        BNeckConfig(80, 184, 80, 3, False, "HardSwish", 1, scale),
        BNeckConfig(80, 184, 80, 3, False, "HardSwish", 1, scale),
        BNeckConfig(80, 480, 112, 3, True, "HardSwish", 1, scale),
        BNeckConfig(112, 672, 112, 3, True, "HardSwish", 1, scale),
        BNeckConfig(112, 672, 160, 5, True, "HardSwish", 2, scale),
        BNeckConfig(160, 960, 160, 5, True, "HardSwish", 1, scale),
        BNeckConfig(160, 960, 160, 5, True, "HardSwish", 1, scale),
    ]
    return MobilenetV3(bneck_confs=bneck_conf, last_channels=1280, num_classes=num_classes)


def get_mobilev3_small(scale: float = 1, num_classes: int = 1000):
    bneck_conf : List[BNeckConfig] = [
        BNeckConfig(16, 16, 16, 3, True, "ReLU", 2, scale),
        BNeckConfig(16, 72, 24, 3, False, "ReLU", 2, scale),
        BNeckConfig(24, 88, 24, 3, False, "ReLU", 1, scale),
        BNeckConfig(24, 96, 40, 5, True, "HardSwish", 2, scale),
        BNeckConfig(40, 240, 40, 5, True, "HardSwish", 1, scale),
        BNeckConfig(40, 240, 40, 5, True, "HardSwish", 1, scale),
        BNeckConfig(40, 120, 48, 5, True, "HardSwish", 1, scale),
        BNeckConfig(48, 144, 48, 5, True, "HardSwish", 1, scale),
        BNeckConfig(48, 288, 96, 5, True, "HardSwish", 2, scale),
        BNeckConfig(96, 576, 96, 5, True, "HardSwish", 1, scale),
        BNeckConfig(96, 576, 96, 5, True, "HardSwish", 1, scale),
    ]
    return MobilenetV3(bneck_confs=bneck_conf, last_channels=1024, num_classes=num_classes)


if __name__ == "__main__":
    input = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    net = get_mobilev3_small(1.5)
    output = net(input)

    print(output.shape)
