
import torch
import torch.nn as nn


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





class BottleNeck(nn.Module):

    def __init__(self, conf: BottleNeckConfig):
        super(BottleNeck, self).__init__()

        if conf.input_channels == conf.expan_channels:
            self.conv1 = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels=conf.input_channels, out_channels=conf.output_channels, kernel_size=conf.kernel_size)









