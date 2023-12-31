import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from module.common import ConvNeXt1d, AdaptiveConvNeXt1d, CausalConv1d
import math


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


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
            num_layers=4,
            kernel_size=7,
            ):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, channels, 1)
        self.f0_enc = F0Encoder(channels)
        scale = 1 / num_layers
        self.mid_layers = nn.ModuleList(
                [ AdaptiveConvNeXt1d(channels, hidden_channels, channels, kernel_size, scale) for _ in range(num_layers)])
    
    def forward(self, x, f0):
        x = self.input_layer(x)
        c = self.f0_enc(f0)
        for l in self.mid_layers:
            x = l(x, c)
        return x


class HarmonicOscillator(nn.Module):
    def __init__(self,
                 channels=512,
                 num_harmonics=64,
                 segment_size=320,
                 sample_rate=16000,
                 ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.segment_size = segment_size
        self.sample_rate = sample_rate

        self.to_amps = nn.Conv1d(channels, num_harmonics, 1)
    
    # x: [N, input_channels, Lf]
    def forward(self, x, f0, phi=0, crop=(0, -1)):
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
        dt = torch.cumsum(formants / self.sample_rate, dim=2) # numerical integrate
        dt = dt - dt[:, :, crop[0]].unsqueeze(2)
        theta = 2 * math.pi * dt + phi
        harmonics = torch.sin(theta)
        phi = torch.asin(harmonics)
    
        harmonics = harmonics * amps

        # Sum all harmonics
        wave = harmonics.mean(dim=1, keepdim=True)

        return wave, phi


class ModulatedCausalConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, condition_channels, kernel_size=5, dilation=1):
        super().__init__()
        self.conv = CausalConv1d(input_channels, output_channels, kernel_size, dilation)
        self.to_scale = nn.Conv1d(condition_channels, input_channels, 1)
        self.to_shift = nn.Conv1d(condition_channels, input_channels, 1)

    def forward(self, x, c):
        scale = self.to_scale(c) + 1
        shift = self.to_shift(c)
        scale = F.interpolate(scale, x.shape[2], mode='linear')
        shift = F.interpolate(shift, x.shape[2], mode='linear')
        x = x * scale + shift
        x = self.conv(x)
        return x


class FilterResBlock(nn.Module):
    def __init__(self, channels, condition_channels, kernel_size=5, dilation=1):
        super().__init__()
        self.c1 = ModulatedCausalConv1d(channels, channels, condition_channels, kernel_size, dilation)
        self.c2 = ModulatedCausalConv1d(channels, channels, condition_channels, kernel_size, dilation)

    def forward(self, x, c):
        res = x
        x = F.gelu(x)
        x = self.c1(x, c)
        x = F.gelu(x)
        x = self.c2(x, c)
        return x + res


class FilterBlock(nn.Module):
    def __init__(self, input_channels, output_channels, condition_channels, kernel_size=5, dilations=3):
        super().__init__()
        self.input_conv = nn.Conv1d(input_channels, output_channels, 1)
        self.blocks = nn.ModuleList([])
        for d in range(dilations):
            self.blocks.append(
                    FilterResBlock(output_channels, condition_channels, kernel_size, 2**d))

    def forward(self, x, c):
        x = self.input_conv(x)
        for b in self.blocks:
            x = b(x, c)
        return x


class Filter(nn.Module):
    def __init__(
            self,
            feat_channels=512,
            rates=[2, 2, 8, 10],
            channels=[8, 16, 64, 256],
            kernel_size=5,
            dilations=3
            ):
        super().__init__()
        self.source_in = nn.Conv1d(1, channels[0], 7, 1, 3)
        self.downs = nn.ModuleList([])
        self.mid_conv = CausalConv1d(channels[-1], channels[-1], kernel_size)
        self.ups = nn.ModuleList([])
        self.blocks = nn.ModuleList([])

        channels_nexts = channels[1:] + [channels[-1]]
        for c, c_next, r in zip(channels, channels_nexts, rates):
            self.downs.append(nn.Conv1d(c, c_next, r, r, 0))

        channels = list(reversed(channels))
        rates = list(reversed(rates))
        channels_prevs = [channels[0]] + channels[:-1]
        
        for c, c_prev, r in zip(channels, channels_prevs, rates):
            self.ups.append(nn.ConvTranspose1d(c_prev, c, r, r, 0))
            self.blocks.append(FilterBlock(c, c, feat_channels, kernel_size, dilations))

        self.source_out = nn.Conv1d(c, 1, 7, 1, 3)
    
    # x: [N, 1, Lw], c: [N, channels, Lf]
    def forward(self, x, c):
        skips = []
        x = self.source_in(x)
        for d in self.downs:
            x = d(x)
            skips.append(x)
        x = self.mid_conv(x)
        for u, b, s in zip(self.ups, self.blocks, reversed(skips)):
            x = u(x + s)
            x = b(x, c)
        x = self.source_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.harmonic_oscillator = HarmonicOscillator()
        self.filter = Filter()

    def forward(self, x, f0, phi=0, harmonics_scale=1, crop=(0, -1)):
        x = self.feature_extractor(x, f0)
        source, phi = self.harmonic_oscillator(x, f0, phi, crop) * harmonics_scale
        out = self.filter(source, x)
        out = out.squeeze(1)
        return out, phi


# Onnx Wrappers

class FeatureExtractorOnnxWrapper(nn.Module):
    def __init__(self, model: FeatureExtractor):
        super().__init__()
        self.model = model

    def forward(self, x, f0):
        return self.model(x, f0)


class HarmonicOscillatorOnnxWrapper(nn.Module):
    def __init__(self, model: HarmonicOscillator, segment_size=320):
        super().__init__()
        self.model = model
        self.segment_size = segment_size

    def forward(self, x):
        Lf = x.shape[2] # frame length
        Lw = Lf * self.segment_size # wave length
        amps = self.model.to_amps(x)
        amps = F.interpolate(amps, Lw, mode='linear')
        return amps


class FilterOnnxWrapper(nn.Module):
    def __init__(self, model: Filter):
        super().__init__()
        self.model = model
    
    def forward(self, x, c):
        return self.model(x, c).squeeze(1)
