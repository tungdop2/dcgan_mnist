import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, b):
        super(GeneratorBlock, self).__init__()

        self.gen_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, k, s, p, bias=b),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.gen_block(x)

class Generator(nn.Module):
    def __init__(self, num_classes=1, noise_size=128, num_channels=64):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels

        self.gen = nn.Sequential(
            GeneratorBlock(noise_size, num_channels * 8, 4, 1, 0, False),
            GeneratorBlock(num_channels * 8, num_channels * 4, 4, 2, 1, False),
            GeneratorBlock(num_channels * 4, num_channels * 2, 4, 2, 1, False),
            GeneratorBlock(num_channels * 2, num_channels, 4, 2, 1, False),

            nn.ConvTranspose2d(num_channels, num_classes, 1, 1, 2, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


