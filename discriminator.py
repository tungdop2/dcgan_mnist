import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, num_classes=1, num_channels=64):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels

        self.dis = nn.Sequential(
            nn.Conv2d(num_classes, num_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_channels, num_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_channels * 2, num_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_channels * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.dis(x)
        # print(out.shape)
        return out.view(-1, 1).squeeze(1)