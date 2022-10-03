# 感觉这样写是拆的比较好的一种写法，VGG负责内部模块的构建。外部只需要传递结构信息，而不需要传递某个结构（torchvision的官方实现里面，外部要把features_layer传进去，我感觉不太好）


import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, conf: list, need_bn: bool = False, num_classes: int = 1000, dropout: float = 0.5, init_weight: bool = False) -> None:
        super().__init__()

        self.feature = self._make_features_layer(conf=conf, need_bn=need_bn)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self._make_classifier_layer(num_classes=num_classes, dropout=dropout)

        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _make_classifier_layer(self, num_classes: int, dropout: float) -> nn.Module:
        classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        return classifier

    def _make_features_layer(self, conf: list, need_bn: bool = False) -> nn.Module:

        in_channels = 3
        layers = nn.ModuleList()

        for v in conf:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1))
                if need_bn:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v

        features = nn.Sequential(*layers)
        return features

    def _init_weight(self, init_weight: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


confs: dict = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
}


def get_vgg11():
    return VGG(confs["A"])


if __name__ == "__main__":
    net = get_vgg11()
    input = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    output = net(input)

    a = 1




