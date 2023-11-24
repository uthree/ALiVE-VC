import torch
import torch.nn as nn
import torch.nn.functional as F

from module.common import ConvNeXt1d, ChannelNorm


class F0Estimator(nn.Module):
    def __init__(self,
                 n_fft=3840,
                 internal_channels=256,
                 hidden_channels=512,
                 output_channels=4096,
                 num_layers=4):
        super().__init__()
        self.input_layer = nn.Conv1d(n_fft//2+1, internal_channels, 1)
        self.mid_layers = nn.Sequential(*[ConvNeXt1d(internal_channels, hidden_channels, scale=1/num_layers)
                                          for _ in range(num_layers)])
        self.last_norm = ChannelNorm(internal_channels)
        self.output_layer = nn.Conv1d(internal_channels, output_channels, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.last_norm(x)
        x = self.output_layer(x)
        return x

    def estimate(self, x, downsample_factor=1):
        dtype = x.dtype
        with torch.no_grad():
            x = self.forward(x)
            f0 = torch.argmax(x, dim=1, keepdim=False).to(x.dtype).unsqueeze(1)
            return f0


class PitchEstimatorOnnxWraper(nn.Module):
    def __init__(self, pe):
        super().__init__()
        self.pe = pe

    def forward(self, x):
        return self.pe.estimate(x)
