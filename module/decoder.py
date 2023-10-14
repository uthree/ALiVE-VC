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


class AmplitudeEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.c1 = nn.Conv1d(1, output_dim, 1, 1, 0)

    def forward(self, amp):
        return self.c1(amp)


class GaussianEncoder(nn.Module):
    def __init__(self, hubert_dim=768, output_dim=512):
        super().__init__()
        self.c1 = nn.Conv1d(hubert_dim, hubert_dim, 7, 1, 3, groups=768)
        self.c2 = nn.Conv1d(hubert_dim, hubert_dim, 1, 1, 0)
        self.c3 = nn.Conv1d(hubert_dim, output_dim*2, 1, 1, 0)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = F.gelu(x)
        x = self.c3(x)
        return x.chunk(2, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 input_channels=768,
                 internal_channels=512,
                 hidden_channels=1536,
                 condition_channels=512,
                 n_fft=1024,
                 num_layers=8):
        super().__init__()
        self.pad = nn.ReflectionPad1d([1, 0])
        self.f0_encoder = F0Encoder(condition_channels)
        self.amp_encoder = AmplitudeEncoder(condition_channels)
        self.gaussian_encoder = GaussianEncoder(input_channels, internal_channels)

        self.input_layer = nn.Conv1d(internal_channels, internal_channels, 1)
        self.mid_layers = nn.ModuleList([
            AdaptiveConvNeXt1d(internal_channels, hidden_channels, condition_channels, scale=1/num_layers)
            for _ in range(num_layers)])
        self.last_norm = AdaptiveChannelNorm(internal_channels, condition_channels)
        self.output_layer = nn.Conv1d(internal_channels, n_fft+2, 1)

        self.n_fft = n_fft
        self.internal_channels = internal_channels

    def forward(self, x, f0, amplitude, noise=None):
        mu, sigma = self.gaussian_encoder(x)
        amp = self.amp_encoder(amplitude)
        f0 = self.f0_encoder(f0)
        
        if noise == None:
            x = mu + torch.exp(sigma) * torch.randn_like(sigma)
        else:
            x = mu + torch.exp(sigma) * noise

        condition = f0 + amp

        condition = self.pad(condition)
        x = self.pad(x)

        x = self.input_layer(x)
        for layer in self.mid_layers:
            x = layer(x, condition)
        x = self.output_layer(x)

        dtype = x.dtype
        x = x.to(torch.float)

        mag, phase = x.chunk(2, dim=1)
        
        mag = torch.exp(mag.clamp_max(6.0))
        phase = (torch.cos(phase) + 1j * torch.sin(phase))
        s = mag * phase
        x = torch.istft(s, 1024, 256)

        x = x.to(dtype)

        return x, mu, sigma


    def decode(self, x, f0, amp, noise_gain=1):
        noise = torch.randn(x.shape[0], self.internal_channels,  x.shape[2], device=x.device) * noise_gain
        x, _, _ = self.forward(x, f0, amp)
        return x


class DecoderOnnxWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x, f0, amplitude, noise):
        mu, sigma = self.decoder.gaussian_encoder(x)
        amp = self.decoder.amp_encoder(amplitude)
        f0 = self.decoder.f0_encoder(f0)
        
        x = mu + torch.exp(sigma) * noise

        condition = f0 + amp

        condition = self.decoder.pad(condition)
        x = self.decoder.pad(x)

        x = self.decoder.input_layer(x)
        for layer in self.decoder.mid_layers:
            x = layer(x, condition)
        x = self.decoder.output_layer(x)

        dtype = x.dtype
        x = x.to(torch.float)

        mag, phase = x.chunk(2, dim=1)

        return mag, phase

