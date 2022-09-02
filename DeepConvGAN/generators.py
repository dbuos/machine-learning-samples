import torch
from torch.nn import Module, Conv2d, LeakyReLU, Sequential, BatchNorm2d, ConvTranspose2d
from torch import nn

class Discriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.sequential = Sequential(
            self.create_conv_module(in_channels=1, out_channels=32, kernel_size=4, batch_norm=False),
            self.create_conv_module(in_channels=32, out_channels=64, kernel_size=4), 
            self.create_conv_module(in_channels=64, out_channels=128, kernel_size=3),
            Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=1),
        )

    @staticmethod
    def create_conv_module(in_channels, out_channels, kernel_size=3, stride=2, batch_norm=True):
        module = Sequential(
            Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False),
        )
        if batch_norm:
            module.add_module('batch_norm', BatchNorm2d(num_features=out_channels))
        module.add_module('leaky_relu', LeakyReLU(0.2))

        return module 

    def forward(self, x):
        return self.sequential(x).view(-1, 1).squeeze(0)

class Generator(Module):

    def __init__(self) -> None:
        super().__init__()
        self.sequential = Sequential(
            Generator.create_trans_conv_module(in_channels=100, out_channels=128, kernel_size=4),
            Generator.create_trans_conv_module(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            Generator.create_trans_conv_module(in_channels=64, out_channels=32, kernel_size=4,stride=2, padding=1),
            ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, padding=1, bias=False, stride=2),
            nn.Tanh(),
        )

    @staticmethod
    def create_trans_conv_module(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        module = Sequential(
            ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_channels),
            LeakyReLU(0.2)
        )
        return module        

    def forward(self, x):
        return self.sequential(x)        