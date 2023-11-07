import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
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


class AmplitudeEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.c1 = nn.Conv1d(1, output_dim, 1, 1, 0)

    def forward(self, amp):
        return self.c1(amp)


class Decoder(nn.Module):
    def __init__(self,
                 input_channels=768,
                 channels=512,
                 hidden_channels=1536,
                 num_layers=10,
                 n_fft=1024,
                 hop_length=256):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, channels, 1)
        self.f0_enc = F0Encoder(channels)
        self.amp_enc = AmplitudeEncoder(channels)
        self.mid_layers = nn.ModuleList([])
        self.pad = nn.ReflectionPad1d([0, 1])
        for _ in range(num_layers):
            self.mid_layers.append(
                    AdaptiveConvNeXt1d(channels, hidden_channels, channels, scale=1/num_layers))
        self.last_norm = AdaptiveChannelNorm(channels, channels)
        self.output_layer = nn.Conv1d(channels, n_fft+2, 1)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def mag_phase(self, x, f0, amp):
        condition = self.f0_enc(f0) + self.amp_enc(amp)
        condition = self.pad(condition)
        x = self.pad(x)
        x = self.input_layer(x)
        for layer in self.mid_layers:
            x = layer(x, condition)
        x = self.last_norm(x, condition)
        x = self.output_layer(x)
        return x.chunk(2, dim=1)

    def forward(self, x, f0, amp):
        dtype = x.dtype
        mag, phase = self.mag_phase(x, f0, amp)
        mag = mag.to(torch.float)
        phase = phase.to(torch.float)
        mag = torch.clamp_max(mag, 6.0)
        mag = torch.exp(mag)
        phase = torch.cos(phase) + 1j * torch.sin(phase)
        s = mag * phase
        return torch.istft(s, self.n_fft, hop_length=self.hop_length)


class DecoderOnnxWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x, f0, amp):
        mag, phase = self.decoder.mag_phase(x, f0, amp)
        return mag, phase
