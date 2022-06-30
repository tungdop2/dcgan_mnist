import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, num_classes=1, num_d_filters=64):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(num_classes, num_d_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_d_filters, num_d_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_d_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_d_filters * 2, num_d_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_d_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_d_filters * 2, num_d_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_d_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_d_filters * 4, num_d_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_d_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_d_filters * 8, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x).view(-1, 1).squeeze(1)