import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from module.common import ConvNeXt1d, AdaptiveConvNeXt1d
import math


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


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            input_channels=768,
            channels=512,
            hidden_channels=1536,
            num_layers=8,
            kernel_size=7,
            ):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, channels, 1)
        self.f0_encoder = F0Encoder(channels)
        scale = 1 / num_layers
        self.mid_layers = nn.ModuleList(
                [ AdaptiveConvNeXt1d(channels, hidden_channels, channels, kernel_size, scale) for _ in range(num_layers)])
    
    def forward(self, x, f0):
        x = self.input_layer(x)
        condition = self.f0_encoder(f0)
        for l in self.mid_layers:
            x = l(x, condition)
        return x


class HarmonicOscillator(nn.Module):
    def __init__(self,
                 channels=512,
                 num_harmonics=64,
                 segment_size=960,
                 sample_rate=48000,
                 ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.segment_size = segment_size
        self.sample_rate = sample_rate

        self.to_amps = nn.Conv1d(channels, num_harmonics, 1)
    
    # x: [N, input_channels, Lf]
    def forward(self, x, f0, t0=0):
        N = x.shape[0] # batch size
        Nh = self.num_harmonics # number of harmonics
        Lf = x.shape[2] # frame length
        Lw = Lf * self.segment_size # wave length

        # to amplitudes
        amps = self.to_amps(x)

        # magnitude to amplitude
        amps = torch.exp(amps)
    
        # frequency multiplyer
        mul = (torch.arange(Nh, device=f0.device) + 1).unsqueeze(0).unsqueeze(2).expand(N, Nh, Lf)

        # Calculate formants
        formants = f0 * mul

        # Interpolate folmants
        formants = F.interpolate(formants, Lw, mode='linear')

        # Interpolate amp
        amps = F.interpolate(amps, Lw, mode='linear')

        # Generate harmonics
        dt = torch.cumsum(formants / self.sample_rate, dim=2)
        harmonics = torch.sin(2 * math.pi * dt) * amps

        # Sum all harmonics
        wave = harmonics.mean(dim=1, keepdim=True)

        return wave


class NoiseGenerator(nn.Module):
    def __init__(self,
                 input_channels=512,
                 n_fft=3840,
                 hop_length=960):
        super().__init__()
        self.to_mag_phase = nn.Conv1d(input_channels, n_fft+2, 1)
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.pad = nn.ReflectionPad1d([1, 0])

    def mag_phase(self, x):
        x = self.pad(x)
        x = self.to_mag_phase(x)
        return x.chunk(2, dim=1)

    def forward(self, x):
        dtype = x.dtype
        mag, phase = self.mag_phase(x)
        mag = mag.to(torch.float)
        phase = phase.to(torch.float)
        mag = torch.clamp_max(mag, 6.0)
        mag = torch.exp(mag)
        phase = torch.cos(phase) + 1j * torch.sin(phase)
        s = mag * phase
        wf = torch.istft(s, self.n_fft, hop_length=self.hop_length, center=True, window=torch.hann_window(self.n_fft))
        wf = wf.unsqueeze(1)
        return wf


class CausalConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=5, dilation=1):
        super().__init__()
        self.pad = nn.ReflectionPad1d([kernel_size*dilation-dilation, 0])
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(self.pad(x))


class DilatedCausalConvStack(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=5, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([])
        self.convs.append(CausalConv1d(input_channels, output_channels, kernel_size, 1))
        for d in range(num_layers-1):
            self.convs.append(CausalConv1d(output_channels, output_channels, kernel_size, 2**(d+1)))

    def forward(self, x):
        for c in self.convs:
            F.gelu(x)
            x = c(x)
        return x


class PostFilter(nn.Module):
    def __init__(
            self,
            channels=8,
            kernel_size=7,
            num_layers=6,
            ):
        super().__init__()
        self.input_layer = nn.Conv1d(1, channels, 1, 1)
        self.mid_layer = DilatedCausalConvStack(channels, channels, kernel_size, num_layers)
        self.output_layer = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        res = x
        x = self.input_layer(x)
        x = self.mid_layer(x)
        x = self.output_layer(x)
        return x + res


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.harmonic_oscillator = HarmonicOscillator()
        self.noise_generator = NoiseGenerator()
        self.post_filter = PostFilter()

    def forward(self, x, f0, t0=0):
        x = self.feature_extractor(x, f0)
        harmonics = self.harmonic_oscillator(x, f0, t0)
        noise = self.noise_generator(x)
        wave = harmonics + noise
        wave = self.post_filter(wave)
        wave = wave.squeeze(1)
        return wave
