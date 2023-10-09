import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from module.common import UNetLayer, AdaptiveChannelNorm
from module.ddpm import DDPM


class ConditionEncoder(nn.Module):
    def __init__(self, channels=512, condition_channels=768):
        super().__init__()
        self.pad = nn.ReflectionPad1d([1, 0])
        self.c1 = nn.Conv1d(condition_channels, channels*2, 1)
        self.c2 = nn.Conv1d(1, channels*2, 1)
        self.c3 = nn.Conv1d(channels*2, channels, 1)
        self.c2.weight.data.normal_(0, 0.3)

    def forward(self, c, f0):
        c = self.pad(c)
        f0 = self.pad(f0)
        c = self.c1(c) + torch.sin(self.c2(f0))
        c = F.gelu(c)
        c = self.c3(c)
        return c


class TimeEncoding1d(nn.Module):
    def __init__(self, channels=512, max_timesteps=10000, return_encoding_only=False):
        super().__init__()
        self.channels = channels
        self.max_timesteps = max_timesteps
        self.return_encoding_only = return_encoding_only

    # t: [batch_size]
    def forward(self, x, t):
        emb = t.unsqueeze(1).expand(t.shape[0], self.channels).unsqueeze(-1)
        e1, e2 = torch.chunk(emb, 2, dim=1)
        factors = 1 / (self.max_timesteps ** (torch.arange(self.channels//2, device=x.device) / (self.channels//2)))
        factors = factors.unsqueeze(0).unsqueeze(2)
        e1 = torch.sin(e1 * math.pi * factors)
        e2 = torch.cos(e2 * math.pi * factors)
        emb = torch.cat([e1, e2], dim=1).expand(*x.shape)

        ret = emb if self.return_encoding_only else x + emb
        return ret


# U-Net
class UNet(nn.Module):
    def __init__(self,
                 internal_channels=512,
                 hubert_channels=768,
                 hidden_channels=1536,
                 n_fft=1024,
                 hop_length=256,
                 num_layers=16,
                 ):
        super().__init__()
        self.input_layer = nn.Conv1d(n_fft + 2, internal_channels, 1)
        self.last_norm = AdaptiveChannelNorm(internal_channels, 512)
        self.output_layer = nn.Conv1d(internal_channels, n_fft + 2, 1)
        self.time_scales = nn.ModuleList([
            nn.Conv1d(internal_channels, internal_channels, 1)
            for _ in range(num_layers)
            ])
        self.time_shifts = nn.ModuleList([
            nn.Conv1d(internal_channels, internal_channels, 1)
            for _ in range(num_layers)
            ])
        self.mid_layers = nn.ModuleList([
            UNetLayer(internal_channels, hidden_channels, 512, scale=1/num_layers)
            for _ in range(num_layers)
            ])
        self.time_enc = TimeEncoding1d(return_encoding_only=True)
        self.n_fft = n_fft
        self.hop_length = hop_length


    def forward(self, x, condition, time):
        res = x
        x = self.wav2spec(x)
        x = self.input_layer(x)
        time_emb = self.time_enc(x, time)

        for l, tsc, tsh in zip(self.mid_layers, self.time_scales, self.time_shifts):
            x = x + tsh(time_emb) * tsc(time_emb)
            x = l(x, condition)

        x = self.last_norm(x, condition)
        x = self.output_layer(x)
        x = self.spec2wave(x)
        x = x + res
        return x

    def wav2spec(self, x):
        dtype = x.dtype
        x = x.to(torch.float)
        x = torch.stft(
                x,
                self.n_fft,
                self.hop_length,
                center=True,
                return_complex=True)
        x = torch.cat([x.real, x.imag], dim=1)
        x = x.to(dtype)
        x = x / self.n_fft
        return x

    def spec2wave(self, x):
        x = x * self.n_fft
        dtype = x.dtype
        x = x.to(torch.float)
        real, imag = torch.chunk(x, 2, dim=1)
        x = torch.complex(real, imag)
        x = torch.istft(
                x,
                self.n_fft,
                self.hop_length,
                center=True,
                onesided=True)
        x = x.to(dtype)
        return x

class DiffusionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition_encoder = ConditionEncoder()
        self.ddpm = DDPM(UNet())
