import torch
import torch.nn as nn
import torch.nn.functional as F

from module.common import AdaptiveConvNeXt1d, AdaptiveChannelNorm


class F0Encoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.c1 = nn.Conv1d(1, output_dim, 1, 1, 0)
        self.c2 = nn.Conv1d(output_dim, output_dim, 1, 1, 0)
        self.c1.weight.data.normal_(0, 0.3)

    def forward(self, x):
        x = self.c1(x)
        x = torch.sin(x)
        x = self.c2(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 input_channels=768,
                 internal_channels=512,
                 hidden_channels=1536,
                 f0_channels=512,
                 n_fft=1024,
                 num_layers=8):
        super().__init__()
        self.pad = nn.ReflectionPad1d([1, 0])
        self.f0_encoder = F0Encoder(f0_channels)
        self.input_layer = nn.Conv1d(input_channels, internal_channels, 1)
        self.mid_layers = nn.ModuleList([
            AdaptiveConvNeXt1d(internal_channels, hidden_channels, f0_channels, scale=1/num_layers)
            for _ in range(num_layers)])
        self.last_norm = AdaptiveChannelNorm(internal_channels, f0_channels)
        self.output_layer = nn.Conv1d(internal_channels, n_fft+2, 1)
        self.n_fft = n_fft

    def forward(self, x, p):
        x = self.pad(x)
        x = self.input_layer(x)
        p = self.pad(p)
        p = self.f0_encoder(p)
        for l in self.mid_layers:
            x = l(x, p)
        x = self.last_norm(x, p)
        mag, phase = self.output_layer(x).chunk(2, dim=1)
        mag = torch.clamp_max(mag, 6.0)
        mag = torch.exp(mag)
        mag = mag.to(torch.float)
        phase = phase.to(torch.float)
        s = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        return torch.istft(s, n_fft=self.n_fft, center=True, hop_length=256, onesided=True)


