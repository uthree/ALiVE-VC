import torch
import torch.nn as nn
import torch.nn.functional as F

from module.common import AdaptiveConvNeXt1d


class F0Encoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.c1 = nn.Conv1d(1, output_dim, 1, 1, 0)
        self.c2 = nn.Conv1d(output_dim, output_dim, 1, 1, 0)
        self.c1.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.c1(x)
        x = torch.sin(x)
        x = self.c2(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            content_dim=768,
            internal_channels=512,
            hidden_channels=1536,
            segment_size=256,
            n_fft=1024,
            num_layers=8,
            ):
        super().__init__()
        self.input_layer = nn.Conv1d(content_dim, internal_channels, 1, 1, 0)
        self.f0_enc = F0Encoder(internal_channels)
        self.mid_layers = nn.ModuleList([
            AdaptiveConvNeXt1d(internal_channels, hidden_channels, internal_channels, scale=1/num_layers) for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(internal_channels, n_fft+2, 1, 1, 0)
        self.segment_size = segment_size
        self.pad = nn.ReflectionPad1d([1, 0])
        self.n_fft = n_fft

    def mag_phase(self, con, f0):
        x = self.input_layer(con)
        f0 = self.f0_enc(f0)
        for l in self.mid_layers:
            x = l(x, f0)
        x = self.output_layer(x)
        x = self.pad(x)
        mag, phase = x.chunk(2, dim=1)
        return mag, phase

    def forward(self, con, f0):
        dtype = con.dtype
        mag, phase = self.mag_phase(con, f0)
        mag = mag.to(torch.float)
        phase = phase.to(torch.float)
        mag = torch.clamp_max(mag, 6.0)
        mag = torch.exp(mag)
        phase = torch.cos(phase) + 1j * torch.sin(phase)
        s = mag * phase
        return torch.istft(s, self.n_fft, self.segment_size)

        
