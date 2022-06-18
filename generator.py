import torch
import torch.nn as nn
import torch.nn.functional as F
class Generator(nn.Module):
    def __init__(self, num_classes=1, noise_size=128, num_g_filters=64):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(noise_size, num_g_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_g_filters * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_g_filters * 8, num_g_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_g_filters * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_g_filters * 4, num_g_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_g_filters * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_g_filters * 2, num_g_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_g_filters),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_g_filters, num_classes, 1, 1, 2, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


