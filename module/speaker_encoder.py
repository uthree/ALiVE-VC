import torch
import torch.nn as nn
import torch.nn.fucntional as F
from module.common import ConvNeXt1d


class SpeakerEncoder(nn.Module):
    def __init__(
            self,
            n_fft=3840,
            internal_channels=192,
            hidden_channels=384,
            num_layers=4,
            ):
        super().__init__()
        self.input_layer = nn.Conv1d(n_fft//2+1, internal_channels, 1)
        self.mid_layers = nn.Sequential(
                *[ConvNeXt1d(internal_channels, hidden_channels) for _ in range(num_layers)])

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = x.mean(dim=2, keepdim=True)
        return x
