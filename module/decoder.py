import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from module.common import AdaptiveConvNeXt1d, AdaptiveChannelNorm, ConvNeXt1d
import math


class SineGenerator(nn.Module):
    def __init__(
            self,
            input_channels=768,
            channels=256,
            hidden_channels=512,
            num_layers=4,
            num_bands=8,
            segment_size=256
            ):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, channels, 1)
        self.mid_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.mid_layers.append(
                    ConvNeXt1d(channels, hidden_channels, scale=1/num_layers))
        self.to_amps = nn.Conv1d(channels, num_bands, 1, 1, 0)
        self.num_bands = num_bands
        self.segment_size = segment_size

    def forward(self, x, t, f0):
        x = self.input_layer(x)
        for layer in self.mid_layers:
            x = layer(x)
        amps = self.to_amps(x)
        bands = torch.arange(self.num_bands, device=x.device) + 1
        N = x.shape[0]
        L = x.shape[2] * self.segment_size
        bands = bands.unsqueeze(0).unsqueeze(2).expand(N, self.num_bands, L)
        f0 = F.interpolate(f0, L, mode='linear')
        amps = F.interpolate(amps, L, mode='linear')
        sinewaves = torch.sin(t * math.pi * 2 * f0 * bands) * amps
        return sinewaves.mean(dim=1)


class NoiseGenerator(nn.Module):
    def __init__(self,
                 input_channels=768,
                 channels=256,
                 hidden_channels=512,
                 num_layers=4,
                 n_fft=1024,
                 hop_length=256):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, channels, 1)
        self.amp_enc = nn.Conv1d(1, channels, 1, 1, 0)
        self.f0_enc = nn.Conv1d(1, channels, 1, 1, 0)
        self.mid_layers = nn.ModuleList([])
        self.pad = nn.ReflectionPad1d([0, 1])
        for _ in range(num_layers):
            self.mid_layers.append(
                    AdaptiveConvNeXt1d(channels, hidden_channels, channels, scale=1/num_layers))
        self.output_layer = nn.Conv1d(channels, n_fft+2, 1)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def mag_phase(self, x, amp, f0):
        condition = self.amp_enc(amp) + self.f0_enc(f0)
        condition = self.pad(condition)
        x = self.pad(x)
        x = self.input_layer(x)
        for layer in self.mid_layers:
            x = layer(x, condition)
        x = self.output_layer(x)
        return x.chunk(2, dim=1)

    def forward(self, x, amp, f0):
        dtype = x.dtype
        mag, phase = self.mag_phase(x, amp, f0)
        mag = mag.to(torch.float)
        phase = phase.to(torch.float)
        mag = torch.clamp_max(mag, 6.0)
        mag = torch.exp(mag)
        phase = torch.cos(phase) + 1j * torch.sin(phase)
        s = mag * phase
        return torch.istft(s, self.n_fft, hop_length=self.hop_length)


class Decoder(nn.Module):
    def __init__(self, segment_size=256, sample_rate=16000):
        super().__init__()
        self.noise_gen = NoiseGenerator()
        self.sin_gen = SineGenerator()
        self.segment_size = segment_size
        self.sample_rate = sample_rate

    def forward(self, x, t, f0, amp):
        noise = self.noise_gen(x, amp, f0)
        sinewave = self.sin_gen(x, t, f0)
        output = sinewave + noise
        return output

    def forward_without_t(self, x, f0, amp):
        l = x.shape[2] * self.segment_size
        b = x.shape[0]
        t = torch.linspace(0, l-1, l) / self.sample_rate
        t = t.unsqueeze(0).unsqueeze(0).expand(b, 1, l)
        t = t.to(x.device).to(x.dtype)
        return self.forward(x, t, f0, amp)
