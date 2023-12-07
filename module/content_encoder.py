import torch
import torch.nn as nn
import torch.nn.functional as F

from module.common import ConvNeXt1d, ChannelNorm


class ContentEncoder(nn.Module):
    def __init__(self,
                 n_fft=1280,
                 internal_channels=512,
                 hidden_channels=1536,
                 output_channels=768,
                 num_layers=4):
        super().__init__()
        self.input_layer = nn.Conv1d(n_fft//2+1, internal_channels, 1)
        self.mid_layers = nn.Sequential(*[ConvNeXt1d(internal_channels, hidden_channels, scale=1/num_layers)
                                          for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(internal_channels, output_channels, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        return x
