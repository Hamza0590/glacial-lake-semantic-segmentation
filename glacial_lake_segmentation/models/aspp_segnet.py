import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch, out_ch, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, **kwargs),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ASPPModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = _conv_bn_relu(256, 256, kernel_size=1)
        self.b2 = _conv_bn_relu(256, 256, kernel_size=3, dilation=6,  padding=6)
        self.b3 = _conv_bn_relu(256, 256, kernel_size=3, dilation=12, padding=12)
        self.b4 = _conv_bn_relu(256, 256, kernel_size=3, dilation=18, padding=18)
        self.b5_pool = nn.AdaptiveAvgPool2d(1)
        self.b5_conv = _conv_bn_relu(256, 256, kernel_size=1)
        self.fusion = _conv_bn_relu(1280, 256, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        b5 = self.b5_conv(self.b5_pool(x))
        b5 = F.interpolate(b5, size=(h, w), mode="bilinear", align_corners=True)
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x), b5], dim=1)
        return self.fusion(out)


class ASPPSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            _conv_bn_relu(3,   64,  kernel_size=3, padding=1),
            _conv_bn_relu(64,  64,  kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
        )
        self.enc2 = nn.Sequential(
            _conv_bn_relu(64,  128, kernel_size=3, padding=1),
            _conv_bn_relu(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
        )
        self.enc3 = nn.Sequential(
            _conv_bn_relu(128, 256, kernel_size=3, padding=1),
            _conv_bn_relu(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.aspp = ASPPModule()

        # Decoder
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            _conv_bn_relu(256, 128, kernel_size=3, padding=1),
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            _conv_bn_relu(128, 64, kernel_size=3, padding=1),
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            _conv_bn_relu(64, 64, kernel_size=3, padding=1),
        )

        self.head = nn.Conv2d(64, 1, kernel_size=1)  # raw logits; sigmoid applied by loss/inference

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.aspp(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return self.head(x)
